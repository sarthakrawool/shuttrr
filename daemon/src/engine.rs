//! The engine: job store, scheduler, admission (busy render, memory guard, flock, dir
//! locks), the worker lifecycle and the watchdog. Port of `shutter/core/engine.py` with the
//! model code moved into the worker child process.

use crate::config::Config;
use crate::events::EventBus;
use crate::locks::{DirLocks, GpuLock};
use crate::modes;
use crate::memory::{self, MemoryGuard, MemorySample, OmlxService, ProcessRssService, Protected};
use crate::protocol;
use crate::stats::{client_label, ClientTracker, StepRate, THROUGHPUT_WINDOW_S};
use crate::store::{Job, Store};
use crate::util::{abspath, jstr, now, py_repr, round_to, secs};
use crate::worker::{EstimateError, JobMsg, StartError, Worker};
use regex_lite::Regex;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::sync::{watch, Notify};

pub const PROTECT_PREFIX: &str = "Stopped to protect your Mac: ";
pub const RENDER_BUSY_MESSAGE: &str = "another render is running";
pub const MAX_MIDRUN_REFUSALS: u32 = 3;
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The plain inference request fields (Python `REQUEST_FIELDS`), in serialisation order.
pub const REQUEST_FIELDS: [&str; 15] = [
    "model", "prompt", "negative_prompt", "width", "height", "frames", "duration_s", "fps", "steps", "guidance",
    "seed", "inputs", "output", "priority", "params",
];
/// Older top-level spellings of `inputs` entries, still accepted.
pub const LEGACY_INPUT_KEYS: [&str; 4] = ["image", "audio", "video", "mask"];
const NUM_FIELDS: [&str; 3] = ["duration_s", "fps", "guidance"];
const POS_INT_FIELDS: [&str; 4] = ["width", "height", "frames", "steps"];
const STR_FIELDS: [&str; 8] = ["prompt", "negative_prompt", "output", "client", "image", "audio", "video", "mask"];

/// An HTTP-level error: status and message.
#[derive(Debug)]
pub struct ApiError(pub u16, pub String);

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(400, msg.into())
}

/// Engine-wide mode defaults (docs/MODES.md), copied into a job's params at submit.
#[derive(Debug, Clone)]
struct Settings {
    mode: String,
    efficiency: f64,
}

struct State {
    paused: bool,
    running: Option<Job>,
    cancel_requested: bool,
    wait_reasons: HashMap<String, String>,
    estimates: HashMap<String, u64>,
    midrun_refusals: HashMap<String, u32>,
    last_activity: Instant,
    steprate: StepRate,
}

pub struct Engine {
    pub cfg: Config,
    store: Mutex<Store>,
    pub bus: EventBus,
    guard: Arc<MemoryGuard>,
    st: Mutex<State>,
    wake: Notify,
    cancel_notify: Notify,
    worker: Mutex<Option<Arc<Worker>>>,
    gpu: Mutex<GpuLock>,
    dirs: Mutex<DirLocks>,
    busy_re: Option<Regex>,
    stop_tx: watch::Sender<bool>,
    pub worker_cwd: Option<PathBuf>,
    settings: Mutex<Settings>,
    /// Clients seen recently (GET /v1/clients).
    pub clients: ClientTracker,
}

async fn until_true(mut rx: watch::Receiver<bool>) {
    loop {
        if *rx.borrow() {
            return;
        }
        if rx.changed().await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

fn lk<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

enum Outcome {
    Done(Value),
    Failed { error: String, cancelled: bool, memory_refused: bool },
    Crashed(String),
    Protect(String),
    ForcedCancel,
    Shutdown,
}

/// The repository root (the directory holding the `shutter/` package) found from the
/// executable's location, e.g. `<repo>/daemon/target/release/shutterd`.
pub fn find_repo_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let exe = exe.canonicalize().unwrap_or(exe);
    for anc in exe.ancestors().skip(1) {
        if anc.join("shutter").join("server").is_dir() && anc.join("shutter").join("__init__.py").is_file() {
            return Some(anc.to_path_buf());
        }
    }
    None
}

/// Where `/` and `/static/*` are served from: SHUTTER_STATIC, then config `static_dir`,
/// then `<repo>/shutter/server`, then `<exe dir>/../Resources/server` (app bundle).
pub fn resolve_static_dir(cfg: &Config) -> Option<PathBuf> {
    if let Ok(s) = std::env::var("SHUTTER_STATIC") {
        if !s.is_empty() {
            return Some(abspath(&s));
        }
    }
    if let Some(d) = &cfg.static_dir {
        return Some(d.clone());
    }
    if let Some(r) = find_repo_root() {
        return Some(r.join("shutter").join("server"));
    }
    let exe = std::env::current_exe().ok()?;
    let res = exe.parent()?.parent()?.join("Resources").join("server");
    res.is_dir().then_some(res)
}

impl Engine {
    pub fn new(cfg: Config) -> Result<Arc<Engine>, String> {
        cfg.ensure_dirs().map_err(|e| format!("cannot create {}: {e}", cfg.home.display()))?;
        let store = Store::open(&cfg.db_path.to_string_lossy())?;
        let mut protected: Vec<Box<dyn Protected>> = Vec::new();
        if let Some(u) = &cfg.omlx_url {
            protected.push(Box::new(OmlxService::new(u, cfg.omlx_api_key.clone(), secs(cfg.omlx_timeout_s), None)));
        }
        if let Some(p) = &cfg.protected_process_pattern {
            protected.push(Box::new(ProcessRssService::new(p, cfg.protected_process_min_bytes(), None)));
        }
        let guard = MemoryGuard {
            probe: memory::default_probe(cfg.test_probe_file.as_deref()),
            protected,
            headroom_bytes: cfg.headroom_bytes().unwrap_or(0),
            headroom_auto: cfg.headroom_bytes().is_none(),
            swap_growth_bytes: cfg.swap_growth_bytes(),
            max_fraction: cfg.max_fraction,
        };
        let busy_re = match &cfg.busy_process_pattern {
            Some(p) => Some(Regex::new(p).map_err(|e| format!("busy_process_pattern: {e}"))?),
            None => None,
        };
        let worker_cwd = cfg.worker_cwd.clone().or_else(find_repo_root);
        let (stop_tx, _) = watch::channel(false);
        Ok(Arc::new(Engine {
            gpu: Mutex::new(GpuLock::new(cfg.gpu_lock_path.clone(), format!("shutterd (pid {})", std::process::id()))),
            dirs: Mutex::new(DirLocks::new(cfg.gpu_lock_dirs.clone(), "shutter")),
            store: Mutex::new(store),
            bus: EventBus::new(1000),
            guard: Arc::new(guard),
            st: Mutex::new(State {
                paused: false,
                running: None,
                cancel_requested: false,
                wait_reasons: HashMap::new(),
                estimates: HashMap::new(),
                midrun_refusals: HashMap::new(),
                last_activity: Instant::now(),
                steprate: StepRate::default(),
            }),
            wake: Notify::new(),
            cancel_notify: Notify::new(),
            worker: Mutex::new(None),
            busy_re,
            stop_tx,
            worker_cwd,
            settings: Mutex::new(Settings { mode: cfg.default_mode.clone(), efficiency: cfg.default_efficiency }),
            clients: ClientTracker::default(),
            cfg,
        }))
    }

    // ------------------------------------------------------------------ helpers

    fn publish(&self, kind: &str, job: Option<&Job>, mut data: Map<String, Value>) {
        if let Some(j) = job {
            data.entry("state").or_insert_with(|| json!(j.state));
            if !j.message.is_empty() {
                data.entry("message").or_insert_with(|| json!(j.message));
            }
        }
        self.bus.publish(kind, job.map(|j| j.id.as_str()), Value::Object(data));
    }

    fn current_worker(&self) -> Option<Arc<Worker>> {
        lk(&self.worker).clone().filter(|w| w.alive())
    }

    async fn sample(&self) -> Result<MemorySample, String> {
        let g = self.guard.clone();
        tokio::task::spawn_blocking(move || g.sample()).await.unwrap_or_else(|e| Err(e.to_string()))
    }

    /// Interruptible sleep: returns early on submit/cancel/resume/stop.
    async fn wait(&self, s: f64) {
        let stop = self.stop_tx.subscribe();
        tokio::select! {
            _ = tokio::time::sleep(secs(s.max(0.001))) => {}
            _ = self.wake.notified() => {}
            _ = until_true(stop) => {}
        }
    }

    fn stopping(&self) -> bool {
        *self.stop_tx.borrow()
    }

    fn update_job(&self, job: &Job) {
        if let Err(e) = lk(&self.store).update(job) {
            crate::log!("shutterd: store update failed for {}: {e}", job.id);
        }
    }

    // ------------------------------------------------------------------ public API

    pub fn known_models(&self) -> Vec<String> {
        let mut ids = Vec::new();
        if self.cfg.dummy_model {
            ids.push("dummy".to_string());
        }
        for k in self.cfg.models.keys() {
            if !ids.contains(k) {
                ids.push(k.clone());
            }
        }
        ids
    }

    /// Parse a JSON request body (empty = null).
    pub fn parse_body(body: &[u8]) -> Result<Value, ApiError> {
        if body.iter().all(u8::is_ascii_whitespace) {
            Ok(Value::Null)
        } else {
            serde_json::from_slice(body).map_err(|_| bad("body must be JSON"))
        }
    }

    /// Validate a POST /v1/jobs body and build the normalized request (the Python
    /// `JobRequest.to_dict()` shape). Same checks and messages as `validate_submit` +
    /// `Engine.submit` in the Python reference. Returns (request, the body's `client`).
    pub fn normalize_request(&self, data: Value) -> Result<(Value, Option<String>), ApiError> {
        let Value::Object(d) = data else {
            return Err(bad("body must be a JSON object with at least \"model\""));
        };
        let model = match d.get("model") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return Err(bad("\"model\" is required (a model id from GET /v1/models)")),
        };
        match d.get("framing") {
            None | Some(Value::Null) | Some(Value::String(_)) => {}
            Some(_) => return Err(bad("framing must be a string (it is passed to the model as params.framing)")),
        }
        if let Some(p) = d.get("params") {
            if !p.is_object() {
                return Err(bad("params must be an object"));
            }
        }
        match d.get("inputs") {
            None | Some(Value::Null) => {}
            Some(Value::Object(o)) => {
                for (k, v) in o {
                    if !v.is_null() && !v.is_string() {
                        return Err(bad(format!("inputs.{k} must be a file path or an upload id")));
                    }
                }
            }
            Some(_) => return Err(bad("inputs must be an object: {\"image\": path, \"audio\": path, ...}")),
        }
        for k in NUM_FIELDS {
            if let Some(v) = d.get(k) {
                if !v.is_null() && !v.is_number() {
                    return Err(bad(format!("{k} must be a number")));
                }
            }
        }
        for k in ["seed", "priority"] {
            if let Some(v) = d.get(k) {
                if v.as_i64().is_none() {
                    return Err(bad(format!("{k} must be an integer")));
                }
            }
        }
        for k in POS_INT_FIELDS {
            if let Some(v) = d.get(k) {
                if !v.is_null() && v.as_i64().is_none() {
                    return Err(bad(format!("{k} must be an integer")));
                }
            }
        }
        for k in STR_FIELDS {
            if let Some(v) = d.get(k) {
                if !v.is_null() && !v.is_string() {
                    return Err(bad(format!("{k} must be a string")));
                }
            }
        }
        // JobRequest.from_dict: legacy input keys -> inputs, framing -> params, unknown keys
        // -> params (explicit inputs / params win)
        let mut inputs = Map::new();
        if let Some(Value::Object(o)) = d.get("inputs") {
            for (k, v) in o {
                if !v.is_null() {
                    inputs.insert(k.clone(), v.clone());
                }
            }
        }
        for k in LEGACY_INPUT_KEYS {
            if let Some(v) = d.get(k).filter(|v| !v.is_null()) {
                inputs.entry(k.to_string()).or_insert_with(|| v.clone());
            }
        }
        let mut params = Map::new();
        for (k, v) in d.iter() {
            if !REQUEST_FIELDS.contains(&k.as_str())
                && !LEGACY_INPUT_KEYS.contains(&k.as_str())
                && k != "framing"
                && k != "client"
            {
                params.insert(k.clone(), v.clone());
            }
        }
        if let Some(Value::Object(p)) = d.get("params") {
            for (k, v) in p {
                params.insert(k.clone(), v.clone());
            }
        }
        if let Some(f) = d.get("framing").filter(|v| !v.is_null()) {
            params.entry("framing".to_string()).or_insert_with(|| f.clone());
        }
        let known = self.known_models();
        if !known.contains(&model) {
            let mut msg = format!("unknown model {}", py_repr(&model));
            if known.is_empty() {
                msg.push_str("; no models are configured");
            } else {
                msg.push_str(&format!("; known models: {}", known.join(", ")));
            }
            return Err(bad(msg));
        }
        let uploads = self.cfg.uploads_dir();
        let mut resolved = Map::new();
        for (name, v) in inputs {
            let Some(p) = v.as_str().filter(|s| !s.is_empty()) else { continue };
            let ok_name = !name.is_empty()
                && name.len() <= 40
                && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
            if !ok_name {
                return Err(bad(format!("input names are short words such as image, audio, video (got {})", py_repr(&name))));
            }
            let ap = crate::files::resolve_input(&uploads, p);
            if !ap.is_file() {
                return Err(bad(format!("{name} file not found: {}", ap.display())));
            }
            resolved.insert(name, json!(ap.to_string_lossy()));
        }
        for k in POS_INT_FIELDS {
            if let Some(n) = d.get(k).and_then(Value::as_i64) {
                if n <= 0 {
                    return Err(bad(format!("{k} must be a positive integer")));
                }
            }
        }
        for k in ["fps", "guidance"] {
            if let Some(x) = d.get(k).and_then(Value::as_f64) {
                if x < 0.0 {
                    return Err(bad(format!("{k} must be a number >= 0")));
                }
            }
        }
        let output = d
            .get("output")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|s| json!(abspath(s).to_string_lossy()))
            .unwrap_or(Value::Null);
        let duration = d.get("duration_s").cloned().unwrap_or(Value::Null);
        if duration.as_f64().map(|x| x < 0.0).unwrap_or(false) {
            return Err(bad("duration_s must be >= 0"));
        }
        // Mode defaults (docs/MODES.md): copied in when the job did not choose.
        match params.get("mode") {
            None | Some(Value::Null) => {
                params.insert("mode".into(), json!(self.settings_now().mode));
            }
            Some(Value::String(m)) if modes::normalize_mode(m).is_some() => {
                params.insert("mode".into(), json!(modes::normalize_mode(m)));
            }
            Some(_) => return Err(bad("mode must be one of: fast, low_memory")),
        }
        match params.get("efficiency") {
            None | Some(Value::Null) => {
                params.insert("efficiency".into(), json!(self.settings_now().efficiency));
            }
            Some(v) if v.is_number() => {
                let e = modes::clamp_efficiency(v.as_f64().unwrap_or(modes::DEFAULT_EFFICIENCY));
                params.insert("efficiency".into(), json!(e));
            }
            Some(_) => return Err(bad("efficiency must be a number from 0 to 1")),
        }
        let opt = |k: &str| d.get(k).cloned().unwrap_or(Value::Null);
        let request = json!({
            "model": model,
            "prompt": d.get("prompt").and_then(Value::as_str).unwrap_or(""),
            "negative_prompt": d.get("negative_prompt").and_then(Value::as_str).unwrap_or(""),
            "width": opt("width"), "height": opt("height"), "frames": opt("frames"),
            "duration_s": duration, "fps": opt("fps"), "steps": opt("steps"), "guidance": opt("guidance"),
            "seed": d.get("seed").and_then(Value::as_i64).unwrap_or(42),
            "inputs": resolved,
            "output": output,
            "priority": d.get("priority").and_then(Value::as_i64).unwrap_or(0),
            "params": params,
        });
        let client = d.get("client").and_then(Value::as_str).map(String::from);
        Ok((request, client))
    }

    /// Validate and queue a job (POST /v1/jobs). `client` is the submitter's label (the
    /// body's `client`, else the X-Shutter-Client header, else the User-Agent).
    pub fn submit_value(&self, data: Value, header_client: Option<&str>, user_agent: Option<&str>) -> Result<Job, ApiError> {
        let (request, body_client) = self.normalize_request(data)?;
        let client = client_label(body_client.as_deref(), header_client, user_agent);
        let mut job = Job::new(request);
        job.client = client;
        lk(&self.store).add(&job).map_err(|e| ApiError(500, e))?;
        self.clients.submitted(job.client.as_deref());
        let mut data = Map::new();
        data.insert("priority".into(), json!(job.priority()));
        data.insert("model".into(), json!(job.model()));
        if let Some(c) = &job.client {
            data.insert("client".into(), json!(c));
        }
        self.publish("job.queued", Some(&job), data);
        self.wake.notify_one();
        Ok(job)
    }

    /// POST /v1/jobs from raw bytes (no headers).
    pub fn submit(&self, body: &[u8]) -> Result<Job, ApiError> {
        self.submit_value(Self::parse_body(body)?, None, None)
    }

    pub fn get(&self, id: &str) -> Result<Option<Job>, String> {
        let st = lk(&self.st);
        if let Some(r) = st.running.as_ref().filter(|r| r.id == id) {
            return Ok(Some(r.clone()));
        }
        drop(st);
        lk(&self.store).get(id)
    }

    pub fn list(&self, states: Option<&[String]>, limit: i64) -> Result<Vec<Job>, String> {
        let jobs = lk(&self.store).list(states, limit)?;
        let st = lk(&self.st);
        Ok(match &st.running {
            Some(r) => jobs.into_iter().map(|j| if j.id == r.id { r.clone() } else { j }).collect(),
            None => jobs,
        })
    }

    /// Queued/waiting: cancelled at once. Running: `cancel` to the worker, killed after
    /// `cancel_grace_s`. None when unknown.
    pub fn cancel(&self, id: &str) -> Result<Option<Job>, String> {
        let mut st = lk(&self.st);
        if let Some(r) = st.running.as_mut().filter(|r| r.id == id) {
            r.message = "cancelling".into();
            let copy = r.clone();
            st.cancel_requested = true;
            drop(st);
            self.cancel_notify.notify_one();
            return Ok(Some(copy));
        }
        let Some(mut job) = lk(&self.store).get(id)? else { return Ok(None) };
        if job.state == "queued" || job.state == "waiting_memory" {
            job.state = "cancelled".into();
            job.finished_at = Some(now());
            job.message = "cancelled before it started".into();
            lk(&self.store).update(&job)?;
            st.wait_reasons.remove(id);
            st.estimates.remove(id);
            drop(st);
            self.publish("job.cancelled", Some(&job), Map::new());
        } else {
            drop(st);
        }
        self.wake.notify_one();
        Ok(Some(job))
    }

    pub fn pause(&self) {
        lk(&self.st).paused = true;
        self.bus.publish("queue.paused", None, json!({}));
    }

    pub fn resume(&self) {
        lk(&self.st).paused = false;
        self.bus.publish("queue.resumed", None, json!({}));
        self.wake.notify_one();
    }

    pub fn paused(&self) -> bool {
        lk(&self.st).paused
    }

    /// Unload one warm component (or all with None). Returns the keys unloaded now.
    pub async fn release(&self, key: Option<&str>) -> Vec<String> {
        let Some(w) = self.current_worker() else { return vec![] };
        let r = w.release(key, Duration::from_secs(5)).await;
        let keys = r.map(|(k, _)| k).unwrap_or_default();
        if !keys.is_empty() {
            self.bus.publish("model.unloaded", None, json!({"keys": keys}));
        }
        keys
    }

    fn settings_now(&self) -> Settings {
        lk(&self.settings).clone()
    }

    /// `{mode, efficiency, rung}`: the short form for /v1/system and events.
    pub fn settings_brief(&self) -> Value {
        let s = self.settings_now();
        json!({"mode": s.mode, "efficiency": s.efficiency, "rung": modes::rung_id(&s.mode, s.efficiency)})
    }

    /// GET /v1/settings: the defaults, the rung they pick and the ladder (with measured
    /// numbers from measured.json where present).
    pub fn settings(&self) -> Value {
        let mut v = self.settings_brief();
        let (ladder, measured) = modes::ladder_json(&self.cfg.home);
        v["ladder"] = ladder;
        v["measured_for"] = measured.unwrap_or(Value::Null);
        v
    }

    /// POST /v1/settings `{"mode"?, "efficiency"?}`: validate, persist to config.json,
    /// publish `settings.changed`, return the GET shape.
    pub fn update_settings(&self, body: &[u8]) -> Result<Value, ApiError> {
        let data: Value = if body.iter().all(u8::is_ascii_whitespace) {
            Value::Object(Map::new())
        } else {
            serde_json::from_slice(body).map_err(|_| bad("body must be JSON"))?
        };
        let Value::Object(d) = data else {
            return Err(bad("body must be a JSON object: {\"mode\"?, \"efficiency\"?}"));
        };
        let mode = match d.get("mode") {
            None | Some(Value::Null) => None,
            Some(Value::String(m)) => {
                Some(modes::normalize_mode(m).ok_or_else(|| bad("mode must be one of: fast, low_memory"))?)
            }
            Some(_) => return Err(bad("mode must be one of: fast, low_memory")),
        };
        let efficiency = match d.get("efficiency") {
            None | Some(Value::Null) => None,
            Some(v) if v.is_number() => Some(modes::clamp_efficiency(v.as_f64().unwrap_or(modes::DEFAULT_EFFICIENCY))),
            Some(_) => return Err(bad("efficiency must be a number from 0 to 1")),
        };
        {
            // Held across the write so two clients cannot interleave their saves.
            let mut s = lk(&self.settings);
            let next = Settings {
                mode: mode.map(String::from).unwrap_or_else(|| s.mode.clone()),
                efficiency: efficiency.unwrap_or(s.efficiency),
            };
            modes::persist(&self.cfg.path, &next.mode, next.efficiency)
                .map_err(|e| ApiError(500, format!("could not save the settings: {e}")))?;
            *s = next;
        }
        self.bus.publish("settings.changed", None, self.settings_brief());
        Ok(self.settings())
    }

    /// The kind and whether a model dir is needed, from the configured adapter family.
    fn family_info(family: &str) -> (&'static str, bool) {
        match family {
            "dummy" => ("video", false),
            "dummy-image" => ("image", false),
            "z-image" => ("image", true),
            "longcat-avatar" => ("video", true),
            _ => ("", true),
        }
    }

    fn family_of(&self, id: &str) -> String {
        if id == "dummy" && !self.cfg.models.contains_key("dummy") {
            return "dummy".into();
        }
        self.cfg.models.get(id).and_then(|s| jstr(s, "adapter")).unwrap_or(id).to_string()
    }

    /// "image" | "video" | "" from the configured family (the worker's own answer wins in /v1/models).
    pub fn family_kind(&self, id: &str) -> &'static str {
        Self::family_info(&self.family_of(id)).0
    }

    pub async fn models(&self) -> Value {
        let worker_rows = match self.current_worker() {
            Some(w) => match w.models(Duration::from_secs(2)).await {
                Some(v) => Some(v),
                None => w.last_models(),
            },
            None => None,
        };
        let by_id: HashMap<String, Value> = worker_rows
            .as_ref()
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|r| Some((jstr(r, "id")?.to_string(), r.clone())))
                    .collect()
            })
            .unwrap_or_default();
        let mut out = Vec::new();
        for id in self.known_models() {
            let spec = if id == "dummy" && !self.cfg.models.contains_key("dummy") {
                json!({"adapter": "dummy"})
            } else {
                self.cfg.models.get(&id).cloned().unwrap_or_else(|| json!({}))
            };
            let family = jstr(&spec, "adapter").unwrap_or(&id).to_string();
            let dir = self.cfg.model_dir(&id);
            let (kind, needs_dir) = Self::family_info(&family);
            let mut row = json!({
                "id": id, "family": family, "kind": kind, "dir": dir.to_string_lossy(),
                "installed": dir.is_dir() || !needs_dir, "error": null,
                "options": spec.get("options").cloned().filter(Value::is_object).unwrap_or_else(|| json!({})),
                // this daemon has no load step: the app shows Load disabled with this hint instead of an error
                "can_load": false, "load_hint": "Loads when a request needs it.",
            });
            if let Some(wr) = by_id.get(&id) {
                for k in ["kind", "installed", "error", "family", "dir"] {
                    if let Some(v) = wr.get(k) {
                        row[k] = v.clone();
                    }
                }
            }
            out.push(row);
        }
        json!({"models": out})
    }

    pub async fn system(&self) -> Value {
        let s = self.sample().await;
        let headroom = match &s {
            Ok(s) => self.guard.headroom_for(s.total),
            Err(_) if self.guard.headroom_auto => 0,
            Err(_) => self.guard.headroom_bytes,
        };
        let memory = match &s {
            Ok(s) => json!({"total": s.total, "available": s.available, "swap_used": s.swap_used, "pressure": s.pressure}),
            Err(e) => json!({"total": 0, "available": 0, "swap_used": 0, "pressure": "unknown", "error": e}),
        };
        let g = self.guard.clone();
        let protected = tokio::task::spawn_blocking(move || g.protected_status()).await.unwrap_or_default();
        let w = self.current_worker();
        let pool_raw = match &w {
            Some(w) => match w.stats(Duration::from_millis(500)).await {
                Some(p) => Some(p),
                None => w.last_pool().map(|(p, _)| p),
            },
            None => None,
        };
        let entries: Vec<Value> = pool_raw
            .as_ref()
            .and_then(|p| p.get("entries"))
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|e| {
                        json!({"key": e.get("key").cloned().unwrap_or(Value::Null),
                               "size_bytes": e.get("size_bytes").cloned().unwrap_or(json!(0)),
                               "idle_s": round_to(e.get("idle_s").and_then(Value::as_f64).unwrap_or(0.0), 1),
                               "pinned": e.get("pinned").and_then(Value::as_bool).unwrap_or(false),
                               "in_use": e.get("in_use").and_then(Value::as_bool).unwrap_or(false),
                               "residency": if e.get("residency").and_then(Value::as_str) == Some("streamed") {
                                   "streamed"
                               } else {
                                   "resident"
                               }})
                    })
                    .collect()
            })
            .unwrap_or_default();
        let total_bytes = pool_raw.as_ref().and_then(|p| p.get("total_bytes")).cloned().unwrap_or(json!(0));
        let counts = lk(&self.store).counts().unwrap_or_default();
        let (paused, current) = {
            let st = lk(&self.st);
            (st.paused, st.running.as_ref().map(|r| r.id.clone()))
        };
        let c = |k: &str| counts.get(k).cloned().unwrap_or(json!(0));
        let (ours, holder, path) = {
            let g = lk(&self.gpu);
            (g.held(), g.holder(), g.path.to_string_lossy().into_owned())
        };
        let (dirs, dir_holders) = {
            let d = lk(&self.dirs);
            let dirs: Vec<Value> =
                d.paths.iter().map(|p| json!({"path": p.to_string_lossy(), "ours": d.ours(p)})).collect();
            (dirs, d.holders())
        };
        json!({
            "version": VERSION,
            "engine": "shutterd",
            "settings": self.settings_brief(),
            "memory": memory,
            "guard": {
                "headroom_bytes": headroom,
                "headroom_auto": self.guard.headroom_auto,
                "swap_growth_bytes": self.guard.swap_growth_bytes,
                "protected": protected.iter().map(|p| json!({"name": p["name"], "busy": p["busy"]})).collect::<Vec<_>>(),
            },
            "pool": {"total_bytes": total_bytes, "entries": entries},
            "queue": {
                "queued": c("queued"), "running": c("running"), "waiting_memory": c("waiting_memory"),
                "done": c("done"), "failed": c("failed"), "cancelled": c("cancelled"),
                "paused": paused, "current": current,
            },
            "gpu_lock": {
                "held": ours || holder.is_some(), "ours": ours, "holder": holder, "path": path,
                "dirs": dirs, "dir_holders": dir_holders,
            },
            "throughput": self.throughput(),
            "worker": match &w {
                Some(w) => json!({"running": true, "pid": w.pid, "version": w.version,
                                  "started_at": w.started_at, "uptime_s": round_to(w.started.elapsed().as_secs_f64(), 1)}),
                None => json!({"running": false, "pid": null, "version": null, "started_at": null, "uptime_s": null}),
            },
        })
    }

    // ------------------------------------------------------------------ native API additions

    /// Done requests in the last hour, their mean wall seconds, the running job's step rate.
    pub fn throughput(&self) -> Value {
        let t = now();
        let jobs = lk(&self.store).finished_since(t - THROUGHPUT_WINDOW_S).unwrap_or_default();
        let mut n = 0usize;
        let mut durs = Vec::new();
        for j in &jobs {
            let Some(fin) = j.finished_at else { continue };
            if fin < t - THROUGHPUT_WINDOW_S {
                continue;
            }
            n += 1;
            if let Some(st) = j.started_at.filter(|st| fin >= *st) {
                durs.push(fin - st);
            }
        }
        let mean = (!durs.is_empty()).then(|| round_to(durs.iter().sum::<f64>() / durs.len() as f64, 3));
        let running = {
            let st = lk(&self.st);
            st.running.as_ref().map(|r| {
                let mut v = st.steprate.snapshot();
                v["id"] = json!(r.id);
                v["model"] = json!(r.model());
                v
            })
        };
        json!({"window_s": THROUGHPUT_WINDOW_S as i64, "finished": n, "mean_seconds": mean, "running": running})
    }

    fn model_of_key(&self, key: &str) -> Option<String> {
        let mut ids = self.known_models();
        ids.sort_by_key(|m| std::cmp::Reverse(m.len()));
        ids.into_iter().find(|m| key.split('/').any(|p| p == m) || key.starts_with(&format!("{m}/")))
    }

    /// GET /v1/memory: what the Mac has, what Shutter keeps back, what Shutter holds (each
    /// component resident or streamed) and, with `model`, whether that model's request fits.
    pub async fn memory(&self, model: Option<&str>, extra: Map<String, Value>) -> Value {
        let s = self.sample().await;
        let (total, available, swap, pressure, error) = match &s {
            Ok(s) => (s.total, s.available, s.swap_used, s.pressure.clone(), None),
            Err(e) => (0, 0, 0, "unknown".to_string(), Some(e.clone())),
        };
        let reserve = match &s {
            Ok(s) => self.guard.headroom_for(s.total),
            Err(_) if self.guard.headroom_auto => 0,
            Err(_) => self.guard.headroom_bytes,
        };
        let mut free = available.saturating_sub(reserve);
        let cap = self.guard.max_fraction.filter(|_| total > 0).map(|f| (f * total as f64) as u64);
        if let Some(c) = cap {
            free = free.min(c);
        }
        let pool = match self.current_worker() {
            Some(w) => match w.stats(Duration::from_millis(500)).await {
                Some(p) => Some(p),
                None => w.last_pool().map(|(p, _)| p),
            },
            None => None,
        };
        let comps: Vec<Value> = pool
            .as_ref()
            .and_then(|p| p.get("entries"))
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|e| {
                        let key = e.get("key").and_then(Value::as_str).unwrap_or("").to_string();
                        let residency = match e.get("residency").and_then(Value::as_str) {
                            Some("streamed") => "streamed",
                            _ => "resident",
                        };
                        json!({"key": key, "model": self.model_of_key(&key),
                               "size_bytes": e.get("size_bytes").and_then(Value::as_u64).unwrap_or(0),
                               "residency": residency,
                               "in_use": e.get("in_use").and_then(Value::as_bool).unwrap_or(false),
                               "pinned": e.get("pinned").and_then(Value::as_bool).unwrap_or(false),
                               "idle_s": round_to(e.get("idle_s").and_then(Value::as_f64).unwrap_or(0.0), 1)})
                    })
                    .collect()
            })
            .unwrap_or_default();
        let sum = |r: Option<&str>| -> u64 {
            comps.iter()
                .filter(|c| r.map(|r| c["residency"] == r).unwrap_or(true))
                .map(|c| c["size_bytes"].as_u64().unwrap_or(0))
                .sum()
        };
        let g = self.guard.clone();
        let protected = tokio::task::spawn_blocking(move || g.protected_status()).await.unwrap_or_default();
        let mut out = json!({
            "total": total, "available": available, "reserve": reserve, "reserve_auto": self.guard.headroom_auto,
            "free_for_models": free, "max_job_bytes": cap, "swap": swap, "pressure": pressure,
            "held": sum(None), "resident_bytes": sum(Some("resident")), "streamed_bytes": sum(Some("streamed")),
            "components": comps,
            "protected": protected.iter().map(|p| json!({"name": p["name"], "busy": p["busy"]})).collect::<Vec<_>>(),
        });
        if let Some(e) = error {
            out["error"] = json!(e);
        }
        if let Some(m) = model {
            let fit = self.model_fit(m, extra, free, &comps).await;
            out["model"] = fit;
        }
        out
    }

    async fn model_fit(&self, model: &str, extra: Map<String, Value>, free: u64, comps: &[Value]) -> Value {
        let mut row = json!({"id": model, "need_bytes": null, "warm_bytes": 0, "fits": null,
                             "short_by_bytes": null, "error": null});
        let mut body = extra;
        body.insert("model".into(), json!(model));
        let request = match self.normalize_request(Value::Object(body)) {
            Ok((r, _)) => r,
            Err(e) => {
                row["error"] = json!(e.1);
                return row;
            }
        };
        let w = match self.ensure_worker().await {
            Ok(w) => w,
            Err(e) => {
                row["error"] = json!(e.message);
                return row;
            }
        };
        lk(&self.st).last_activity = Instant::now();
        let id = format!("memory-{}", crate::util::new_id());
        let need = match w.estimate(&id, &request, secs(self.cfg.estimate_timeout_s)).await {
            Ok(msg) => match protocol::estimate_peak(&msg) {
                Ok(n) => n,
                Err(e) => {
                    row["error"] = json!(format!("estimate failed: {e}"));
                    return row;
                }
            },
            Err(_) => {
                row["error"] = json!("the worker did not give a memory estimate");
                return row;
            }
        };
        let warm: u64 = comps
            .iter()
            .filter(|c| c["model"].as_str() == Some(model))
            .map(|c| c["size_bytes"].as_u64().unwrap_or(0))
            .sum();
        let short = need.saturating_sub(warm).saturating_sub(free);
        row["need_bytes"] = json!(need);
        row["warm_bytes"] = json!(warm);
        row["fits"] = json!(short == 0);
        row["short_by_bytes"] = json!(short);
        row
    }

    /// POST /v1/files: store one upload.
    pub fn save_upload(&self, name: Option<&str>, data: &[u8], content_type: Option<&str>) -> Result<Value, ApiError> {
        crate::files::save_upload(&self.cfg.uploads_dir(), name, data, content_type, self.cfg.max_upload_bytes())
            .map_err(|(c, m)| ApiError(c, m))
    }

    /// GET /v1/jobs/{id}/output: the finished file, or (status, message).
    pub fn output_file(&self, id: &str) -> Result<PathBuf, ApiError> {
        let job = self.get(id).map_err(|e| ApiError(500, e))?.ok_or_else(|| ApiError(404, format!("no job {id}")))?;
        if job.state != "done" {
            return Err(ApiError(409, format!("job {id} is {}; it has no output yet", job.state)));
        }
        let out = job
            .result
            .get("output")
            .and_then(Value::as_str)
            .or_else(|| jstr(&job.request, "output"))
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ApiError(404, format!("job {id} has no output file")))?;
        let p = PathBuf::from(out);
        if !p.is_file() {
            return Err(ApiError(404, format!("the output file of job {id} is gone: {out}")));
        }
        Ok(p)
    }

    /// Model rows' `kind` by id (GET /v1/models).
    pub async fn model_kinds(&self) -> HashMap<String, String> {
        let m = self.models().await;
        m["models"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|r| Some((jstr(r, "id")?.to_string(), jstr(r, "kind").unwrap_or("").to_string())))
                    .collect()
            })
            .unwrap_or_default()
    }

    // ------------------------------------------------------------------ worker lifecycle

    async fn ensure_worker(&self) -> Result<Arc<Worker>, StartError> {
        if let Some(w) = self.current_worker() {
            return Ok(w);
        }
        let w = Worker::spawn(&self.cfg, self.worker_cwd.as_deref()).await?;
        *lk(&self.worker) = Some(w.clone());
        self.bus.publish("worker.started", None, json!({"pid": w.pid, "version": w.version}));
        Ok(w)
    }

    async fn stop_worker(&self, reason: &str) {
        let w = lk(&self.worker).take();
        if let Some(w) = w {
            let was_alive = w.alive();
            w.stop(secs(self.cfg.shutdown_grace_s)).await;
            if was_alive {
                self.bus.publish("worker.stopped", None, json!({"pid": w.pid, "reason": reason}));
            }
        }
    }

    /// Forget a dead worker (after a crash or a kill).
    fn drop_dead_worker(&self, w: &Arc<Worker>) {
        let mut g = lk(&self.worker);
        if g.as_ref().map(|c| Arc::ptr_eq(c, w)).unwrap_or(false) {
            *g = None;
        }
    }

    async fn idle_maintenance(&self) {
        let (idle, running) = {
            let st = lk(&self.st);
            (st.last_activity.elapsed().as_secs_f64(), st.running.is_some())
        };
        if running {
            return;
        }
        let present = lk(&self.worker).clone();
        if let Some(w) = present {
            if !w.alive() {
                self.drop_dead_worker(&w);
                self.bus.publish("worker.stopped", None, json!({"pid": w.pid, "reason": w.exit_status()}));
            } else if idle >= self.cfg.idle_unload_s.max(0.0) {
                self.stop_worker("idle").await;
            }
        }
    }

    fn idle_tick(&self) -> f64 {
        let idle = lk(&self.st).last_activity.elapsed().as_secs_f64();
        if self.current_worker().is_some() {
            (self.cfg.idle_unload_s - idle).clamp(0.01, 1.0)
        } else {
            1.0
        }
    }

    // ------------------------------------------------------------------ scheduler

    /// Run until `shutdown()`. Never returns early because of a worker or store error.
    pub async fn run(self: Arc<Self>) {
        loop {
            if self.stopping() {
                break;
            }
            let paused = lk(&self.st).paused;
            if paused {
                self.idle_maintenance().await;
                let t = self.idle_tick();
                self.wait(t).await;
                continue;
            }
            let next = lk(&self.store).next_queued();
            match next {
                Ok(Some(job)) => self.step(job).await,
                Ok(None) => {
                    self.idle_maintenance().await;
                    let t = self.idle_tick();
                    self.wait(t).await;
                }
                Err(e) => {
                    crate::log!("shutterd: store error: {e}");
                    self.wait(0.5).await;
                }
            }
        }
        self.stop_worker("daemon shutting down").await;
        lk(&self.gpu).release();
        lk(&self.dirs).release();
    }

    pub fn shutdown(&self) {
        let _ = self.stop_tx.send(true);
        self.wake.notify_one();
    }

    fn set_waiting(&self, id: &str, reason: &str) {
        let mut st = lk(&self.st);
        let cur = match lk(&self.store).get(id) {
            Ok(Some(j)) if j.state == "queued" || j.state == "waiting_memory" => j,
            _ => return,
        };
        let changed = st.wait_reasons.get(id).map(String::as_str) != Some(reason);
        let mut cur = cur;
        if cur.state != "waiting_memory" || cur.message != reason {
            cur.state = "waiting_memory".into();
            cur.message = reason.into();
            self.update_job(&cur);
        }
        st.wait_reasons.insert(id.into(), reason.into());
        drop(st);
        if changed {
            let mut d = Map::new();
            d.insert("reason".into(), json!(reason));
            self.publish("job.waiting_memory", Some(&cur), d);
        }
    }

    fn finish(&self, job: &mut Job, state: &str, message: &str) {
        job.state = state.into();
        job.finished_at = Some(now());
        job.message = message.into();
        if state == "done" {
            job.progress = 1.0;
        }
        self.update_job(job);
        {
            let mut st = lk(&self.st);
            st.wait_reasons.remove(&job.id);
            st.estimates.remove(&job.id);
        }
        let mut d = Map::new();
        if state == "done" {
            d.insert("result".into(), Value::Object(job.result.clone()));
        }
        self.publish(&format!("job.{state}"), Some(job), d);
    }

    fn render_busy(&self) -> Option<&'static str> {
        let re = self.busy_re.as_ref()?;
        let text = memory::run_ps().ok()?;
        let me = std::process::id() as i32;
        memory::parse_ps(&text)
            .iter()
            .any(|(pid, _, cmd)| *pid != me && re.is_match(cmd))
            .then_some(RENDER_BUSY_MESSAGE)
    }

    fn release_locks(&self) {
        lk(&self.dirs).release();
        lk(&self.gpu).release();
    }

    async fn waiting(&self, id: &str, reason: &str) {
        self.set_waiting(id, reason);
        self.idle_maintenance().await;
        self.wait(self.cfg.memory_poll_s).await;
    }

    async fn step(self: &Arc<Self>, job: Job) {
        let id = job.id.clone();

        // 1. another render running on this Mac?
        if self.busy_re.is_some() {
            let me = self.clone();
            let busy = tokio::task::spawn_blocking(move || me.render_busy()).await.ok().flatten();
            if let Some(b) = busy {
                return self.waiting(&id, b).await;
            }
        }

        // 2. the worker's memory estimate (asked once per job, then cached)
        let cached = lk(&self.st).estimates.get(&id).copied();
        let need = match cached {
            Some(n) => n,
            None => match self.estimate(&job).await {
                Ok(n) => {
                    lk(&self.st).estimates.insert(id.clone(), n);
                    n
                }
                Err((msg, tail)) => {
                    let mut j = job;
                    if let Some(t) = tail {
                        j.result.insert("log_tail".into(), json!(t));
                    }
                    self.finish(&mut j, "failed", &msg);
                    return;
                }
            },
        };

        // 3. memory guard
        let label = format!("job {} ({})", id, job.model());
        let mut d = self.check(need, &label).await;
        if !d.0 {
            if let Some(w) = self.current_worker() {
                let warm = w
                    .last_pool()
                    .and_then(|(p, _)| p.get("total_bytes").and_then(Value::as_u64))
                    .unwrap_or(0);
                if warm > 0 {
                    let freed = w.release(None, Duration::from_secs(5)).await.map(|r| r.0).unwrap_or_default();
                    if !freed.is_empty() {
                        self.bus.publish("model.unloaded", None, json!({"keys": freed}));
                        d = self.check(need, &label).await;
                    }
                }
            }
        }
        if !d.0 {
            return self.waiting(&id, &d.1).await;
        }

        // 4. GPU locks: flock, then the shared mkdir dirs (all or nothing)
        let got = lk(&self.gpu).try_acquire();
        match got {
            Ok(true) => {}
            Ok(false) => {
                let h = lk(&self.gpu).holder().unwrap_or(json!({}));
                let who = h
                    .get("label")
                    .and_then(Value::as_str)
                    .map(String::from)
                    .or_else(|| h.get("raw").and_then(Value::as_str).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()))
                    .unwrap_or_else(|| "another process".into());
                return self.waiting(&id, &format!("GPU busy: {who}")).await;
            }
            Err(e) => return self.waiting(&id, &format!("GPU busy: {e}")).await,
        }
        let r = lk(&self.dirs).try_acquire();
        if let Some(reason) = r {
            lk(&self.gpu).release();
            return self.waiting(&id, &format!("GPU busy: {reason}")).await;
        }

        // 5. still wanted? then it runs
        let job = {
            let mut st = lk(&self.st);
            let cur = match lk(&self.store).get(&id) {
                Ok(Some(j)) if j.state == "queued" || j.state == "waiting_memory" => j,
                _ => {
                    drop(st);
                    self.release_locks();
                    return;
                }
            };
            let mut job = cur;
            job.state = "running".into();
            job.started_at = Some(now());
            job.finished_at = None;
            job.progress = 0.0;
            job.stage = "starting".into();
            job.message = String::new();
            job.result = Map::new();
            st.cancel_requested = false;
            st.running = Some(job.clone());
            st.last_activity = Instant::now();
            st.steprate = StepRate::default();
            self.update_job(&job);
            job
        };
        self.run_job(job, need).await;
        {
            let mut st = lk(&self.st);
            st.running = None;
            st.cancel_requested = false;
            st.last_activity = Instant::now();
        }
        self.release_locks();
        if self.cfg.idle_unload_s <= 0.0 && !self.stopping() {
            self.stop_worker("idle").await;
        }
    }

    async fn check(&self, need: u64, label: &str) -> (bool, String) {
        let g = self.guard.clone();
        let label = label.to_string();
        tokio::task::spawn_blocking(move || {
            let d = g.check(need, &label);
            (d.ok, d.reason)
        })
        .await
        .unwrap_or((false, "memory check failed".into()))
    }

    /// The worker's memory estimate for the job. Err = (message, log tail) and the job fails.
    async fn estimate(&self, job: &Job) -> Result<u64, (String, Option<String>)> {
        let w = self.ensure_worker().await.map_err(|e| (e.message, e.tail))?;
        lk(&self.st).last_activity = Instant::now();
        match w.estimate(&job.id, &job.request, secs(self.cfg.estimate_timeout_s)).await {
            Ok(msg) => protocol::estimate_peak(&msg).map_err(|e| (format!("estimate failed: {e}"), None)),
            Err(EstimateError::Timeout) => {
                w.kill();
                let _ = w.wait_exit(Duration::from_secs(3)).await;
                self.drop_dead_worker(&w);
                Err((
                    format!(
                        "the worker did not answer the memory estimate within {} s and was stopped",
                        self.cfg.estimate_timeout_s
                    ),
                    Some(w.tail().join("\n")),
                ))
            }
            Err(EstimateError::Died(desc)) => {
                let _ = w.wait_exit(Duration::from_secs(3)).await;
                self.drop_dead_worker(&w);
                let last = w.last_line().map(|l| format!(": {l}")).unwrap_or_default();
                Err((format!("the worker stopped while estimating memory ({desc}){last}"), Some(w.tail().join("\n"))))
            }
        }
    }

    fn set_running<F: FnOnce(&mut Job)>(&self, f: F) -> Option<Job> {
        let mut st = lk(&self.st);
        let r = st.running.as_mut()?;
        f(r);
        Some(r.clone())
    }

    async fn run_job(self: &Arc<Self>, mut job: Job, need: u64) {
        let id = job.id.clone();
        let w = match self.ensure_worker().await {
            Ok(w) => w,
            Err(e) => {
                self.release_locks();
                if let Some(t) = e.tail {
                    job.result.insert("log_tail".into(), json!(t));
                }
                self.finish(&mut job, "failed", &e.message);
                return;
            }
        };
        // work dir and an absolute output path (like the Python engine)
        let work_dir = self.cfg.work_dir.join(&id);
        let _ = std::fs::create_dir_all(&work_dir);
        let output = match jstr(&job.request, "output").filter(|s| !s.is_empty()) {
            Some(o) => PathBuf::from(o),
            None => {
                let image = w
                    .last_models()
                    .and_then(|m| {
                        m.as_array()?.iter().find(|r| jstr(r, "id") == Some(job.model())).and_then(|r| jstr(r, "kind").map(String::from))
                    })
                    .map(|k| k == "image")
                    .unwrap_or_else(|| self.family_kind(job.model()) == "image");
                self.cfg.output_dir.join(format!("{id}.{}", if image { "png" } else { "mp4" }))
            }
        };
        if let Some(p) = output.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        job.request["output"] = json!(output.to_string_lossy());
        if let Some(r) = self.set_running(|r| r.request = job.request.clone()) {
            self.update_job(&r);
        }

        let base = self.sample().await.ok();
        let swap_base = base.as_ref().map(|s| s.swap_used);
        let mut peak_used = base.as_ref().map(|s| s.used()).unwrap_or(0);
        let base_used = peak_used;
        let mut peak_swap = swap_base.unwrap_or(0);
        let mut avail_min = base.as_ref().map(|s| s.available).unwrap_or(0);

        let mut rx = w.attach_job(&id);
        let mut d = Map::new();
        d.insert("estimate_peak".into(), json!(need));
        d.insert("model".into(), json!(job.model()));
        d.insert("pid".into(), json!(w.pid));
        self.publish("job.started", Some(&job), d);

        let t0 = Instant::now();
        let mut outcome: Option<Outcome> = None;
        if let Err(e) = w.send(&protocol::op_run(&id, &job.to_json(), &work_dir.to_string_lossy())).await {
            w.kill();
            outcome = Some(Outcome::Crashed(e));
        }
        let mut wd = base.as_ref().map(|b| memory::Watchdog::new(self.guard.swap_growth_bytes, 10.0, b));
        let mut wd_events: Vec<Value> = Vec::new();
        let mut watch_tick = tokio::time::interval(secs(self.cfg.watch_interval_s.max(0.01)));
        watch_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut last_sample_pub = Instant::now();
        let mut last_persist = Instant::now();
        let mut last_pub = Instant::now() - Duration::from_secs(1);
        let mut last_stage = String::new();
        let mut cancel_deadline: Option<tokio::time::Instant> = None;
        let stop = self.stop_tx.subscribe();
        let grace = self.cfg.cancel_grace_s;

        while outcome.is_none() {
            if cancel_deadline.is_none() && lk(&self.st).cancel_requested {
                let _ = w.send(&protocol::op_cancel(&id)).await;
                cancel_deadline = Some(tokio::time::Instant::now() + secs(grace));
            }
            tokio::select! {
                msg = rx.recv() => match msg {
                    Some(JobMsg::Progress { progress, stage }) => {
                        if let Some((a, b)) = crate::stats::parse_step(&stage) {
                            lk(&self.st).steprate.update(a, b);
                        }
                        let job_now = self.set_running(|r| {
                            r.progress = progress.max(r.progress).clamp(0.0, 1.0);
                            if !stage.is_empty() { r.stage = stage.clone(); }
                        });
                        if let Some(j) = job_now {
                            if last_persist.elapsed() >= Duration::from_secs(1) || j.progress >= 1.0 {
                                last_persist = Instant::now();
                                self.update_job(&j);
                            }
                            if stage != last_stage || last_pub.elapsed() >= Duration::from_millis(100) || j.progress >= 1.0 {
                                last_pub = Instant::now();
                                last_stage = stage.clone();
                                let el = t0.elapsed().as_secs_f64();
                                let eta = (j.progress > 0.01).then(|| round_to(el * (1.0 - j.progress) / j.progress, 1));
                                let mut d = Map::new();
                                d.insert("progress".into(), json!(j.progress));
                                d.insert("stage".into(), json!(j.stage));
                                d.insert("eta_s".into(), json!(eta));
                                self.publish("job.progress", Some(&j), d);
                            }
                        }
                    }
                    Some(JobMsg::Done(r)) => outcome = Some(Outcome::Done(r)),
                    Some(JobMsg::Failed { error, cancelled, memory_refused }) =>
                        outcome = Some(Outcome::Failed { error, cancelled, memory_refused }),
                    Some(JobMsg::Log { level, message }) => {
                        let mut d = Map::new();
                        d.insert("level".into(), json!(level));
                        d.insert("message".into(), json!(message));
                        self.bus.publish("job.log", Some(&id), Value::Object(d));
                    }
                    Some(JobMsg::Exited(desc)) => outcome = Some(Outcome::Crashed(desc)),
                    None => outcome = Some(Outcome::Crashed(w.exit_status().unwrap_or_else(|| "worker pipe closed".into()))),
                },
                _ = watch_tick.tick() => {
                    if let Ok(s) = self.sample().await {
                        peak_used = peak_used.max(s.used());
                        peak_swap = peak_swap.max(s.swap_used);
                        avail_min = avail_min.min(s.available);
                        if last_sample_pub.elapsed().as_secs_f64() >= self.cfg.sample_interval_s {
                            last_sample_pub = Instant::now();
                            self.bus.publish("memory.sample", Some(&id), s.to_json());
                        }
                        if wd.is_none() {
                            wd = Some(memory::Watchdog::new(self.guard.swap_growth_bytes, 10.0, &s));
                        }
                        let el = round_to(t0.elapsed().as_secs_f64(), 1);
                        match wd.as_mut().map(|x| x.step(&s, t0.elapsed().as_secs_f64())) {
                            Some(memory::WatchAction::Relieve(why)) => {
                                // step down first: the worker frees MLX's cache and every idle component
                                let _ = w.send(&protocol::op_release(None)).await;
                                wd_events.push(json!({"t": el, "action": "relieve", "why": why}));
                                let mut d = Map::new();
                                d.insert("action".into(), json!("relieve"));
                                d.insert("reason".into(), json!(why));
                                self.bus.publish("job.memory", Some(&id), Value::Object(d));
                            }
                            Some(memory::WatchAction::Stop(why)) => {
                                wd_events.push(json!({"t": el, "action": "stop", "why": why}));
                                w.kill();
                                outcome = Some(Outcome::Protect(format!("{} ({why})", memory::STOP_BODY)));
                            }
                            _ => {}
                        }
                    }
                }
                _ = self.cancel_notify.notified(), if cancel_deadline.is_none() => {}
                _ = async { match cancel_deadline { Some(d) => tokio::time::sleep_until(d).await, None => std::future::pending().await } } => {
                    w.kill();
                    outcome = Some(Outcome::ForcedCancel);
                }
                _ = until_true(stop.clone()) => outcome = Some(Outcome::Shutdown),
            }
        }
        let outcome = outcome.unwrap_or(Outcome::Crashed("unknown".into()));
        w.detach_job();
        let killed = matches!(outcome, Outcome::Protect(_) | Outcome::ForcedCancel | Outcome::Crashed(_));
        if killed {
            if !w.wait_exit(Duration::from_secs(3)).await {
                w.kill();
                let _ = w.wait_exit(Duration::from_secs(2)).await;
            }
            self.drop_dead_worker(&w);
            self.bus.publish("worker.stopped", None, json!({"pid": w.pid, "reason": w.exit_status()}));
        }
        if let Ok(s) = self.sample().await {
            peak_used = peak_used.max(s.used());
            peak_swap = peak_swap.max(s.swap_used);
            avail_min = avail_min.min(s.available);
        }
        let cancel_requested = lk(&self.st).cancel_requested;
        let tail = w.tail().join("\n");
        let seconds = round_to(t0.elapsed().as_secs_f64(), 3);
        let mut result = Map::new();
        let (state, message): (&str, String) = match outcome {
            Outcome::Done(r) => {
                if let Value::Object(m) = r {
                    result = m;
                }
                if cancel_requested {
                    ("cancelled", "cancelled while running".into())
                } else {
                    ("done", String::new())
                }
            }
            Outcome::Failed { cancelled, .. } if cancelled || cancel_requested => {
                ("cancelled", "cancelled while running".into())
            }
            Outcome::Failed { error, memory_refused: true, .. } => {
                let n = {
                    let mut st = lk(&self.st);
                    let n = st.midrun_refusals.entry(id.clone()).or_insert(0);
                    *n += 1;
                    *n
                };
                if n >= MAX_MIDRUN_REFUSALS {
                    ("failed", format!("memory refused {n} times mid-run: {error}"))
                } else {
                    ("waiting_memory", error)
                }
            }
            Outcome::Failed { error, .. } => {
                result.insert("log_tail".into(), json!(tail));
                ("failed", error)
            }
            Outcome::Crashed(_) if cancel_requested => ("cancelled", "cancelled while running".into()),
            Outcome::Crashed(desc) => {
                result.insert("log_tail".into(), json!(tail));
                let last = w
                    .tail()
                    .iter()
                    .rev()
                    .find(|l| !l.trim().is_empty() && !l.starts_with("stdout (ignored"))
                    .map(|l| format!(": {l}"))
                    .unwrap_or_default();
                ("failed", format!("the worker stopped unexpectedly ({desc}){last}"))
            }
            Outcome::Protect(reason) => {
                result.insert("log_tail".into(), json!(tail));
                ("failed", format!("{PROTECT_PREFIX}{reason}"))
            }
            Outcome::ForcedCancel => (
                "cancelled",
                format!("cancelled while running; the worker did not stop within {grace} s and was stopped"),
            ),
            Outcome::Shutdown => ("queued", "interrupted by shutdown; it will run again".into()),
        };
        result.entry("seconds").or_insert(json!(seconds));
        let pb = result.get("peak_bytes").or_else(|| result.get("peak_memory_bytes")).and_then(Value::as_f64);
        result.insert("peak_bytes".into(), json!(pb.map(|x| x as u64).unwrap_or(peak_used)));
        result.insert(
            "peak_memory".into(),
            json!({"used_bytes": peak_used, "delta_bytes": peak_used as i64 - base_used as i64,
                   "available_min": avail_min, "swap_used_max": peak_swap,
                   "swap_used_start": swap_base.unwrap_or(0)}),
        );
        result.insert("estimate_peak_bytes".into(), json!(need));
        result.insert(
            "watchdog".into(),
            json!({"events": wd_events,
                   "max_pressure": wd.as_ref().map(|x| x.max_pressure.clone()),
                   "swap_growth_max_bytes": wd.as_ref().map(|x| x.max_swap_growth).unwrap_or(0)}),
        );
        if state == "done" {
            result.entry("output").or_insert(json!(output.to_string_lossy()));
        }
        // Give the GPU back before the final state is visible.
        self.release_locks();
        let mut job = self.set_running(|_| {}).unwrap_or(job);
        job.result = result;
        if state == "queued" || state == "waiting_memory" {
            job.state = state.into();
            job.message = message.clone();
            job.started_at = None;
            job.progress = 0.0;
            job.stage = String::new();
            {
                let mut st = lk(&self.st);
                st.running = None;
                if state == "waiting_memory" {
                    st.wait_reasons.insert(id.clone(), message.clone());
                }
            }
            self.update_job(&job);
            if state == "waiting_memory" {
                let mut d = Map::new();
                d.insert("reason".into(), json!(message));
                self.publish("job.waiting_memory", Some(&job), d);
            } else {
                self.publish("job.requeued", Some(&job), Map::new());
            }
        } else {
            lk(&self.st).midrun_refusals.remove(&id);
            lk(&self.st).running = None;
            self.finish(&mut job, state, &message);
        }
        if state == "done" && !self.cfg.keep_work_dirs {
            let inside = |p: &Path| p.starts_with(&work_dir);
            let mut keep = inside(&output);
            for v in job.result.values() {
                if let Some(s) = v.as_str() {
                    if s.contains('/') && inside(Path::new(s)) {
                        keep = true;
                    }
                }
            }
            if !keep {
                let _ = std::fs::remove_dir_all(&work_dir);
            }
        }
        if matches!(state, "failed") && job.message.starts_with(PROTECT_PREFIX) {
            crate::log!("shutterd: job {id} {}", job.message);
        }
    }
}

#[cfg(test)]
mod family_tests {
    use super::Engine;

    #[test]
    fn image_families_write_png() {
        assert_eq!(Engine::family_info("z-image"), ("image", true));
        assert_eq!(Engine::family_info("dummy-image"), ("image", false));
        assert_eq!(Engine::family_info("wan").0, "");
    }
}
