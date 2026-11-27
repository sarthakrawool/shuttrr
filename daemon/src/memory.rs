//! Memory safety: samples, macOS/Linux probes, protected services and the guard.
//! A port of `shutter/core/memory.py`; decisions and their human reasons are identical.
//!
//! The macOS probe shells out to `sysctl` and `vm_stat` and parses their text, exactly like
//! the Python reference. The parsers are pure functions compiled on every platform so they
//! are unit-tested here with captured macOS output.

use crate::util::{fmt_gb, now, py_repr};
use regex_lite::Regex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

pub const PRESSURE_NORMAL: &str = "normal";
pub const PRESSURE_WARN: &str = "warn";
pub const PRESSURE_CRITICAL: &str = "critical";

#[derive(Debug, Clone, PartialEq)]
pub struct MemorySample {
    pub total: u64,
    pub available: u64,
    pub swap_used: u64,
    pub pressure: String,
    pub ts: f64,
}

impl MemorySample {
    pub fn used(&self) -> u64 {
        self.total.saturating_sub(self.available)
    }
    pub fn to_json(&self) -> Value {
        json!({"total": self.total, "available": self.available, "swap_used": self.swap_used,
               "pressure": self.pressure, "used": self.used()})
    }
}

pub trait Probe: Send + Sync {
    fn sample(&self) -> Result<MemorySample, String>;
}

// ---- pure parsers -------------------------------------------------------------------

/// Parse `vm_stat` output into bytes. Keys: free, active, inactive, speculative,
/// throttled, wired, purgeable, compressed, ..., `page_size` and `available`
/// (= free + inactive + purgeable + speculative + active file-backed pages). The page size in vm_stat's header wins.
pub fn parse_vm_stat(text: &str, page_size: Option<u64>) -> HashMap<String, u64> {
    let mut ps = page_size.unwrap_or(0);
    if let Some(i) = text.find("page size of ") {
        let digits: String = text[i + 13..].chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = digits.parse::<u64>() {
            ps = n;
        }
    }
    if ps == 0 {
        ps = 4096;
    }
    let fields: &[(&str, &str)] = &[
        ("Pages free", "free"),
        ("Pages active", "active"),
        ("Pages inactive", "inactive"),
        ("Pages speculative", "speculative"),
        ("Pages throttled", "throttled"),
        ("Pages wired down", "wired"),
        ("Pages purgeable", "purgeable"),
        ("Pages occupied by compressor", "compressed"),
        ("Pages stored in compressor", "stored_in_compressor"),
        ("File-backed pages", "file_backed"),
        ("Anonymous pages", "anonymous"),
    ];
    let mut out = HashMap::new();
    out.insert("page_size".to_string(), ps);
    for line in text.lines() {
        let Some((label, value)) = line.split_once(':') else { continue };
        let label = label.trim().trim_matches('"').trim();
        let value = value.trim();
        let value = value.strip_suffix('.').unwrap_or(value).trim();
        let Ok(n) = value.parse::<u64>() else { continue };
        if let Some((_, key)) = fields.iter().find(|(l, _)| *l == label) {
            out.insert(key.to_string(), n.saturating_mul(ps));
        }
    }
    // Active file-backed pages are clean copies of files that macOS drops without swapping. vm_stat does not split
    // the inactive list, so file_backed - inactive (never below 0) is a lower bound: nothing is counted twice.
    let file = out.get("file_backed").copied().unwrap_or(0).saturating_sub(out.get("inactive").copied().unwrap_or(0));
    out.insert("reclaimable_file".into(), file);
    let avail = ["free", "inactive", "purgeable", "speculative", "reclaimable_file"]
        .iter()
        .map(|k| out.get(*k).copied().unwrap_or(0))
        .sum();
    out.insert("available".into(), avail);
    out
}

/// Parse `sysctl vm.swapusage` ("total = 2048.00M  used = 1024.50M  free = ...").
/// Returns (total, used, free) in bytes; missing values are 0.
pub fn parse_swapusage(text: &str) -> (u64, u64, u64) {
    let re = Regex::new(r"(?i)(total|used|free)\s*=\s*([\d.]+)\s*([KMGT]?)").expect("static regex");
    let (mut t, mut u, mut f) = (0u64, 0u64, 0u64);
    for c in re.captures_iter(text) {
        let num: f64 = c[2].parse().unwrap_or(0.0);
        let mult = match c[3].to_ascii_uppercase().as_str() {
            "K" => 1024f64,
            "M" => 1024f64.powi(2),
            "G" => 1024f64.powi(3),
            "T" => 1024f64.powi(4),
            _ => 1.0,
        };
        let v = (num * mult) as u64;
        match c[1].to_ascii_lowercase().as_str() {
            "total" => t = v,
            "used" => u = v,
            _ => f = v,
        }
    }
    (t, u, f)
}

/// kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical. Unparseable maps to
/// normal so a broken sysctl does not block the engine forever.
pub fn parse_pressure_level(text: &str) -> &'static str {
    match text.split_whitespace().last().and_then(|s| s.parse::<i64>().ok()) {
        Some(l) if l >= 4 => PRESSURE_CRITICAL,
        Some(l) if l >= 2 => PRESSURE_WARN,
        _ => PRESSURE_NORMAL,
    }
}

/// /proc/meminfo into bytes (keys as in the file).
pub fn parse_meminfo(text: &str) -> HashMap<String, u64> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.replace(':', " ");
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let Ok(mut v) = parts[1].parse::<u64>() else { continue };
        if parts.len() >= 3 && parts[2].eq_ignore_ascii_case("kb") {
            v *= 1024;
        }
        out.insert(parts[0].to_string(), v);
    }
    out
}

/// `ps -axo pid=,rss=,command=` into (pid, rss_bytes, command).
pub fn parse_ps(text: &str) -> Vec<(i32, u64, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim_start();
        let Some((a, r)) = t.split_once(char::is_whitespace) else { continue };
        let r = r.trim_start();
        let Some((b, c)) = r.split_once(char::is_whitespace) else { continue };
        if c.trim().is_empty() {
            continue;
        }
        let (Ok(pid), Ok(rss)) = (a.parse::<i32>(), b.parse::<u64>()) else { continue };
        out.push((pid, rss * 1024, c.trim().to_string()));
    }
    out
}

/// Run a command with a timeout; stdout as text. Used for sysctl, vm_stat and ps.
pub fn run_cmd(cmd: &[&str], timeout: Duration) -> Result<String, String> {
    use std::process::{Command, Stdio};
    let mut child = Command::new(cmd[0])
        .args(&cmd[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("{}: {e}", cmd[0]))?;
    let mut stdout = child.stdout.take().ok_or("no stdout")?;
    let reader = std::thread::spawn(move || {
        let mut s = Vec::new();
        let _ = std::io::Read::read_to_end(&mut stdout, &mut s);
        s
    });
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(st)) => {
                let out = reader.join().unwrap_or_default();
                if !st.success() {
                    return Err(format!("{} exited with {st}", cmd[0]));
                }
                return Ok(String::from_utf8_lossy(&out).into_owned());
            }
            Ok(None) if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{} timed out", cmd[0]));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(e) => return Err(e.to_string()),
        }
    }
}

// ---- probes -------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub struct MacProbe {
    stat: Mutex<Option<(u64, u64)>>,
}

#[cfg(target_os = "macos")]
impl MacProbe {
    pub fn new() -> Self {
        MacProbe { stat: Mutex::new(None) }
    }
    fn statics(&self) -> Result<(u64, u64), String> {
        let mut g = self.stat.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = *g {
            return Ok(v);
        }
        let out = run_cmd(&["sysctl", "-n", "hw.memsize", "hw.pagesize"], Duration::from_secs(10))?;
        let mut it = out.split_whitespace();
        let total: u64 = it.next().and_then(|s| s.parse().ok()).ok_or("cannot parse hw.memsize")?;
        let page: u64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(16384);
        *g = Some((total, page));
        Ok((total, page))
    }
}

#[cfg(target_os = "macos")]
impl Probe for MacProbe {
    fn sample(&self) -> Result<MemorySample, String> {
        let (total, page) = self.statics()?;
        let t = Duration::from_secs(10);
        let vm = parse_vm_stat(&run_cmd(&["vm_stat"], t)?, Some(page));
        let swap = run_cmd(&["sysctl", "-n", "vm.swapusage"], t)
            .map(|s| parse_swapusage(&s))
            .unwrap_or((0, 0, 0));
        let pressure = run_cmd(&["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], t)
            .map(|s| parse_pressure_level(&s))
            .unwrap_or(PRESSURE_NORMAL);
        Ok(MemorySample {
            total,
            available: vm.get("available").copied().unwrap_or(0),
            swap_used: swap.1,
            pressure: pressure.into(),
            ts: now(),
        })
    }
}

/// /proc/meminfo; pressure is always "normal" (as in the Python LinuxProbe).
#[cfg(not(target_os = "macos"))]
pub struct LinuxProbe {
    pub path: String,
}

#[cfg(not(target_os = "macos"))]
impl Probe for LinuxProbe {
    fn sample(&self) -> Result<MemorySample, String> {
        let text = std::fs::read_to_string(&self.path).map_err(|e| format!("{}: {e}", self.path))?;
        let mi = parse_meminfo(&text);
        let g = |k: &str| mi.get(k).copied().unwrap_or(0);
        let available = mi.get("MemAvailable").copied().unwrap_or(g("MemFree") + g("Cached"));
        Ok(MemorySample {
            total: g("MemTotal"),
            available,
            swap_used: g("SwapTotal").saturating_sub(g("SwapFree")),
            pressure: PRESSURE_NORMAL.into(),
            ts: now(),
        })
    }
}

/// Test probe: reads `{"total","available","swap_used","pressure"}` from a JSON file on every
/// sample (config `test_probe_file`). Missing fields use 64 GB total / 48 GB available.
pub struct FileProbe {
    pub path: std::path::PathBuf,
}

impl Probe for FileProbe {
    fn sample(&self) -> Result<MemorySample, String> {
        let text = std::fs::read_to_string(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", self.path.display()))?;
        let gb = 1024u64.pow(3);
        let n = |k: &str, d: u64| v.get(k).and_then(Value::as_f64).map(|x| x.max(0.0) as u64).unwrap_or(d);
        Ok(MemorySample {
            total: n("total", 64 * gb),
            available: n("available", 48 * gb),
            swap_used: n("swap_used", 0),
            pressure: v.get("pressure").and_then(Value::as_str).unwrap_or(PRESSURE_NORMAL).to_string(),
            ts: now(),
        })
    }
}

/// Settable in-memory probe for unit tests.
pub struct FakeProbe {
    pub s: Mutex<Result<MemorySample, String>>,
}

impl FakeProbe {
    pub fn new(total_gb: f64, avail_gb: f64) -> Self {
        let gb = crate::util::GB;
        FakeProbe {
            s: Mutex::new(Ok(MemorySample {
                total: (total_gb * gb) as u64,
                available: (avail_gb * gb) as u64,
                swap_used: 0,
                pressure: PRESSURE_NORMAL.into(),
                ts: 0.0,
            })),
        }
    }
    pub fn set(&self, f: impl FnOnce(&mut MemorySample)) {
        let mut g = self.s.lock().unwrap_or_else(|e| e.into_inner());
        if let Ok(s) = g.as_mut() {
            f(s)
        }
    }
}

impl Probe for FakeProbe {
    fn sample(&self) -> Result<MemorySample, String> {
        self.s.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

pub fn default_probe(test_file: Option<&std::path::Path>) -> Box<dyn Probe> {
    if let Some(p) = test_file {
        return Box::new(FileProbe { path: p.to_path_buf() });
    }
    #[cfg(target_os = "macos")]
    {
        Box::new(MacProbe::new())
    }
    #[cfg(not(target_os = "macos"))]
    {
        Box::new(LinuxProbe { path: "/proc/meminfo".into() })
    }
}

// ---- protected services ---------------------------------------------------------------

pub trait Protected: Send + Sync {
    fn name(&self) -> String;
    /// A human reason when the service holds memory we must not compete with.
    fn busy(&self) -> Option<String>;
    fn last_error(&self) -> Option<String>;
}

/// fetch(url, headers, timeout) -> (status, body). There is deliberately no method
/// parameter: the oMLX check can only ever GET.
pub type Fetcher = Box<dyn Fn(&str, &[(&str, &str)], Duration) -> Result<(u16, Vec<u8>), String> + Send + Sync>;

pub fn http_get(url: &str, headers: &[(&str, &str)], timeout: Duration) -> Result<(u16, Vec<u8>), String> {
    let r = crate::httpc::request("GET", url, headers, None, timeout)?;
    Ok((r.status, r.body))
}

const LOADED_STATUS: &[&str] = &["loaded", "ready", "active", "running", "serving"];

fn entry_name(e: &Value) -> String {
    for k in ["id", "model", "name", "model_id"] {
        if let Some(s) = e.get(k).and_then(Value::as_str) {
            if !s.is_empty() {
                return s.to_string();
            }
        }
    }
    "unnamed model".into()
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|x| x != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn entry_state(e: &Value) -> Option<bool> {
    let o = e.as_object()?;
    let mut seen = false;
    for k in ["loaded", "is_loaded"] {
        if let Some(v) = o.get(k) {
            seen = true;
            if truthy(v) {
                return Some(true);
            }
        }
    }
    let is_loaded = |s: &str| LOADED_STATUS.contains(&s.trim().to_lowercase().as_str());
    for k in ["status", "state"] {
        match o.get(k) {
            Some(Value::String(s)) => {
                seen = true;
                if is_loaded(s) {
                    return Some(true);
                }
            }
            Some(Value::Object(inner)) => {
                let iv = ["value", "status", "state"]
                    .iter()
                    .filter_map(|k| inner.get(*k))
                    .find(|v| truthy(v));
                if let Some(Value::String(s)) = iv {
                    seen = true;
                    if is_loaded(s) {
                        return Some(true);
                    }
                }
            }
            _ => {}
        }
    }
    if seen {
        Some(false)
    } else {
        None
    }
}

fn entries(data: &Value) -> Option<&Vec<Value>> {
    match data {
        Value::Array(a) => Some(a),
        Value::Object(o) => ["models", "data", "items", "loaded_models"]
            .iter()
            .find_map(|k| o.get(*k).and_then(Value::as_array)),
        _ => None,
    }
}

/// Names of loaded models, [] when state is known and none are loaded, None when the
/// answer carries no load state at all.
pub fn omlx_loaded(data: &Value) -> Option<Vec<String>> {
    if let Some(list) = data.get("loaded_models").and_then(Value::as_array) {
        return Some(
            list.iter()
                .filter_map(|m| match m {
                    Value::String(s) => Some(s.clone()),
                    Value::Object(_) => Some(entry_name(m)),
                    _ => None,
                })
                .collect(),
        );
    }
    let es = entries(data)?;
    let mut known = false;
    let mut loaded = Vec::new();
    for e in es {
        match entry_state(e) {
            None => continue,
            Some(st) => {
                known = true;
                if st {
                    loaded.push(entry_name(e));
                }
            }
        }
    }
    known.then_some(loaded)
}

/// The user's chat model server (oMLX). READ-ONLY: GET /v1/models/status, then
/// GET /v1/models. Unreachable or no load state = not busy.
pub struct OmlxService {
    pub base_url: String,
    pub timeout: Duration,
    pub api_key: Option<String>,
    pub name: String,
    fetch: Fetcher,
    last_error: Mutex<Option<String>>,
}

impl OmlxService {
    pub fn new(base_url: &str, api_key: Option<String>, timeout: Duration, fetch: Option<Fetcher>) -> Self {
        OmlxService {
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout,
            api_key,
            name: "oMLX".into(),
            fetch: fetch.unwrap_or_else(|| Box::new(http_get)),
            last_error: Mutex::new(None),
        }
    }
    fn set_err(&self, e: Option<String>) {
        *self.last_error.lock().unwrap_or_else(|p| p.into_inner()) = e;
    }
}

impl Protected for OmlxService {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn last_error(&self) -> Option<String> {
        self.last_error.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
    fn busy(&self) -> Option<String> {
        self.set_err(None);
        let auth = self.api_key.as_ref().map(|k| format!("Bearer {k}"));
        let mut headers: Vec<(&str, &str)> = vec![("Accept", "application/json")];
        if let Some(a) = auth.as_deref() {
            headers.push(("Authorization", a));
        }
        for path in ["/v1/models/status", "/v1/models"] {
            let url = format!("{}{}", self.base_url, path);
            let (status, body) = match (self.fetch)(&url, &headers, self.timeout) {
                Ok(r) => r,
                Err(e) => {
                    self.set_err(Some(format!("{path}: unreachable ({e})")));
                    return None;
                }
            };
            if status != 200 {
                self.set_err(Some(format!("{path}: HTTP {status}")));
                continue;
            }
            let data: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let Some(loaded) = omlx_loaded(&data) else {
                self.set_err(Some(format!("{path}: no load-state fields in the answer")));
                continue;
            };
            self.set_err(None);
            if loaded.is_empty() {
                return None;
            }
            let mut names = loaded.iter().take(3).cloned().collect::<Vec<_>>().join(", ");
            if loaded.len() > 3 {
                names.push_str(" and more");
            }
            return Some(format!(
                "{} has a model loaded ({names}); Shuttrr waits so it does not compete with it for memory.",
                self.name
            ));
        }
        None
    }
}

pub type PsRunner = Box<dyn Fn() -> Result<String, String> + Send + Sync>;

pub fn run_ps() -> Result<String, String> {
    run_cmd(&["ps", "-axo", "pid=,rss=,command="], Duration::from_secs(10))
}

/// Busy when any process whose command line matches `pattern` (case-insensitive) has RSS
/// above `min_bytes`. This process is ignored.
pub struct ProcessRssService {
    pattern: Option<Regex>,
    pub pattern_src: String,
    pub min_bytes: u64,
    ps: PsRunner,
    last_error: Mutex<Option<String>>,
}

impl ProcessRssService {
    pub fn new(pattern: &str, min_bytes: u64, ps: Option<PsRunner>) -> Self {
        let re = Regex::new(&format!("(?i){pattern}")).ok();
        let err = re.is_none().then(|| format!("bad pattern {pattern:?}"));
        ProcessRssService {
            pattern: re,
            pattern_src: pattern.into(),
            min_bytes,
            ps: ps.unwrap_or_else(|| Box::new(run_ps)),
            last_error: Mutex::new(err),
        }
    }
}

impl Protected for ProcessRssService {
    fn name(&self) -> String {
        format!("processes matching {}", py_repr(&self.pattern_src))
    }
    fn last_error(&self) -> Option<String> {
        self.last_error.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
    fn busy(&self) -> Option<String> {
        let re = self.pattern.as_ref()?;
        let text = match (self.ps)() {
            Ok(t) => {
                *self.last_error.lock().unwrap_or_else(|p| p.into_inner()) = None;
                t
            }
            Err(e) => {
                *self.last_error.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
                return None;
            }
        };
        let me = std::process::id() as i32;
        let hit = parse_ps(&text)
            .into_iter()
            .filter(|(pid, rss, cmd)| *pid != me && *rss > self.min_bytes && re.is_match(cmd))
            .max_by(|a, b| (a.1, &a.2).cmp(&(b.1, &b.2)))?;
        let (_, rss, cmd) = hit;
        let first = cmd.split_whitespace().next().unwrap_or(&cmd);
        let prog = first.rsplit('/').next().unwrap_or(first);
        Some(format!(
            "{prog} is using {} of memory (more than {}); Shuttrr waits so it does not compete with it.",
            fmt_gb(rss as f64),
            fmt_gb(self.min_bytes as f64)
        ))
    }
}

// ---- guard --------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Decision {
    pub ok: bool,
    pub reason: String,
    pub sample: Option<MemorySample>,
}

/// Since 0.2.3 there is no automatic reserve: a job may use all the memory macOS can give it
/// right now, and the `Watchdog` protects the Mac while it runs. Kept for callers of the old name.
pub fn auto_headroom(_total: u64) -> u64 {
    0
}

/// What the user reads when the watchdog stops a job: PROTECT_PREFIX (engine.rs) + STOP_BODY + (detail).
pub const STOP_BODY: &str = "it was running out of memory. Close other apps or pick a smaller size.";

/// What the watchdog wants done after a sample.
#[derive(Debug, Clone, PartialEq)]
pub enum WatchAction {
    /// Nothing to do.
    Calm,
    /// First sign of trouble: free caches and idle models (a step down).
    Relieve(String),
    /// Trouble did not ease (or swap jumped past the hard limit): stop the job.
    Stop(String),
}

/// Live protection while a job runs (port of `shutter/core/memory.py` Watchdog). Trouble =
/// critical memory pressure, or swap grew by more than `swap_soft` ("warn" pressure never acts).
/// First trouble: Relieve. `grace_s` later: Stop only if pressure is still critical or swap grew by
/// another `swap_soft / 2` since the relief (swap that grew once and holds steady is fine).
/// Swap grown past `swap_hard` (3x `swap_soft`: 768 MB with the 256 MB default) since the start: Stop at once.
#[derive(Debug, Clone)]
pub struct Watchdog {
    pub swap_soft: u64,
    pub swap_hard: u64,
    pub grace_s: f64,
    pub swap_baseline: u64,
    pub relieved_at: Option<f64>,
    pub relieved_swap: u64,
    pub calm_swap: Option<u64>,
    pub stopped: Option<String>,
    pub max_pressure: String,
    pub max_swap_growth: u64,
}

impl Watchdog {
    pub fn new(swap_soft: u64, grace_s: f64, start: &MemorySample) -> Watchdog {
        Watchdog {
            swap_soft,
            swap_hard: swap_soft.saturating_mul(3),
            grace_s,
            swap_baseline: start.swap_used,
            relieved_at: None,
            relieved_swap: 0,
            calm_swap: None,
            stopped: None,
            max_pressure: start.pressure.clone(),
            max_swap_growth: 0,
        }
    }

    pub fn trouble(&self, s: &MemorySample) -> Option<String> {
        let base = self.calm_swap.map_or(self.swap_baseline, |c| c.max(self.swap_baseline));
        let grown = s.swap_used.saturating_sub(base);
        if s.pressure == PRESSURE_CRITICAL {
            return Some("macOS reports critical memory pressure".into());
        }
        if grown > self.swap_soft {
            return Some(format!("swap grew by {}", fmt_gb(grown as f64)));
        }
        None
    }

    pub fn step(&mut self, s: &MemorySample, now_s: f64) -> WatchAction {
        let grown = s.swap_used.saturating_sub(self.swap_baseline);
        self.max_swap_growth = self.max_swap_growth.max(grown);
        let rank = |p: &str| match p {
            PRESSURE_CRITICAL => 2,
            PRESSURE_WARN => 1,
            _ => 0,
        };
        if rank(&s.pressure) > rank(&self.max_pressure) {
            self.max_pressure = s.pressure.clone();
        }
        if let Some(r) = &self.stopped {
            return WatchAction::Stop(r.clone());
        }
        if grown > self.swap_hard {
            let r = format!(
                "swap grew by {} since the request started (limit {})",
                fmt_gb(grown as f64),
                fmt_gb(self.swap_hard as f64)
            );
            self.stopped = Some(r.clone());
            return WatchAction::Stop(r);
        }
        if let Some(t) = self.relieved_at {
            if now_s - t < self.grace_s {
                return WatchAction::Calm;
            }
            let mut worse = Vec::new();
            if s.pressure == PRESSURE_CRITICAL {
                worse.push("memory pressure is still critical".to_string());
            }
            let more = s.swap_used.saturating_sub(self.relieved_swap);
            if more > self.swap_soft / 2 {
                worse.push(format!("swap kept growing (another {})", fmt_gb(more as f64)));
            }
            if !worse.is_empty() {
                let r = format!("{} {:.0} s after freeing caches", worse.join(" and "), self.grace_s);
                self.stopped = Some(r.clone());
                return WatchAction::Stop(r);
            }
            self.relieved_at = None;
            self.calm_swap = Some(s.swap_used);
            return WatchAction::Calm;
        }
        match self.trouble(s) {
            None => WatchAction::Calm,
            Some(why) => {
                self.relieved_at = Some(now_s);
                self.relieved_swap = s.swap_used;
                WatchAction::Relieve(format!("{why}; freeing caches"))
            }
        }
    }
}

pub struct MemoryGuard {
    pub probe: Box<dyn Probe>,
    pub protected: Vec<Box<dyn Protected>>,
    /// The explicit headroom; ignored when `headroom_auto`.
    pub headroom_bytes: u64,
    /// Headroom follows the Mac's total memory (`auto_headroom`).
    pub headroom_auto: bool,
    pub swap_growth_bytes: u64,
    pub max_fraction: Option<f64>,
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

impl MemoryGuard {
    pub fn sample(&self) -> Result<MemorySample, String> {
        self.probe.sample()
    }

    /// The headroom in force for a Mac with `total` bytes of memory.
    pub fn headroom_for(&self, total: u64) -> u64 {
        if self.headroom_auto {
            auto_headroom(total)
        } else {
            self.headroom_bytes
        }
    }

    pub fn protected_reason(&self) -> Option<String> {
        self.protected.iter().find_map(|p| p.busy())
    }

    /// Admission. Refuses when (in this order): a protected service is busy; pressure is
    /// critical ("warn" is fine); `max_fraction` is exceeded; need > available - headroom, where
    /// available = free + inactive + purgeable + speculative + active file cache, and headroom is 0 unless set.
    pub fn check(&self, need: u64, what: &str) -> Decision {
        let label = if what.is_empty() { "this job" } else { what };
        let s = match self.probe.sample() {
            Ok(s) => s,
            Err(e) => {
                return Decision {
                    ok: false,
                    reason: format!("Cannot read this Mac's memory state ({e}); waiting before starting {label}."),
                    sample: None,
                }
            }
        };
        let done = |ok: bool, reason: String| Decision { ok, reason, sample: Some(s.clone()) };
        if let Some(r) = self.protected_reason() {
            return done(false, r);
        }
        if s.pressure == PRESSURE_CRITICAL {
            return done(
                false,
                format!("macOS reports critical memory pressure; waiting until it eases before starting {label}."),
            );
        }
        let need_f = need as f64;
        let head = self.headroom_for(s.total);
        if let Some(mf) = self.max_fraction {
            let cap = mf * s.total as f64;
            if need_f > cap {
                return done(
                    false,
                    format!(
                        "{} needs {}, more than the {:.0}% limit of this Mac's {} ({}).",
                        capitalize(label),
                        fmt_gb(need_f),
                        mf * 100.0,
                        fmt_gb(s.total as f64),
                        fmt_gb(cap)
                    ),
                );
            }
        }
        let budget = s.available.saturating_sub(head);
        let extra = if head > 0 { format!(" (after the {} headroom you set)", fmt_gb(head as f64)) } else { String::new() };
        if need > budget {
            return done(
                false,
                format!(
                    "Not enough free memory for {label}: it needs {}, but macOS can give only {} right now{extra}.",
                    fmt_gb(need_f),
                    fmt_gb(budget as f64)
                ),
            );
        }
        done(true, format!("{} free right now{extra} covers {} for {label}.", fmt_gb(budget as f64), fmt_gb(need_f)))
    }

    /// Protected services with their current status (for /v1/system).
    pub fn protected_status(&self) -> Vec<Value> {
        self.protected
            .iter()
            .map(|p| {
                let busy = p.busy();
                json!({"name": p.name(), "busy": busy, "last_error": p.last_error()})
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GB: u64 = 1024 * 1024 * 1024;

    const VM_STAT: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               12345.
Pages active:                            400000.
Pages inactive:                          300000.
Pages speculative:                         5000.
Pages throttled:                              0.
Pages wired down:                        150000.
Pages purgeable:                           2000.
\"Translation faults\":                 123456789.
Pages copy-on-write:                    9876543.
Pages zero filled:                    111111111.
Pages reactivated:                       222222.
Pages purged:                             33333.
File-backed pages:                       250000.
Anonymous pages:                         455000.
Pages stored in compressor:              100000.
Pages occupied by compressor:             40000.
Decompressions:                           55555.
Compressions:                             66666.
Pageins:                                 777777.
Pageouts:                                  8888.
Swapins:                                      0.
Swapouts:                                     0.
";

    #[test]
    fn vm_stat() {
        let v = parse_vm_stat(VM_STAT, Some(4096));
        assert_eq!(v["page_size"], 16384);
        assert_eq!(v["free"], 12345 * 16384);
        assert_eq!(v["compressed"], 40000 * 16384);
        assert_eq!(v["wired"], 150000 * 16384);
        assert_eq!(v["available"], (12345 + 300000 + 2000 + 5000) * 16384);
        // header missing: the given page size is used
        let v = parse_vm_stat("Pages inactive: 3.\nFile-backed pages: 5.\n", Some(16384));
        assert_eq!(v["reclaimable_file"], 2 * 16384);
        assert_eq!(v["available"], 5 * 16384, "inactive + active file pages, counted once");
        let v = parse_vm_stat("Pages free: 10.\n", Some(16384));
        assert_eq!(v["available"], 10 * 16384);
        let v = parse_vm_stat("garbage", None);
        assert_eq!(v["available"], 0);
    }

    #[test]
    fn swapusage_and_pressure() {
        let (t, u, f) = parse_swapusage("total = 2048.00M  used = 1024.50M  free = 1023.50M  (encrypted)");
        assert_eq!(t, 2048 * 1024 * 1024);
        assert_eq!(u, (1024.5 * 1024.0 * 1024.0) as u64);
        assert_eq!(f, (1023.5 * 1024.0 * 1024.0) as u64);
        assert_eq!(parse_swapusage("total = 0.00M  used = 0.00M  free = 0.00M"), (0, 0, 0));
        assert_eq!(parse_swapusage("vm.swapusage: total = 1.50G  used = 0.25G  free = 1.25G").1, GB / 4);
        assert_eq!(parse_pressure_level("1\n"), "normal");
        assert_eq!(parse_pressure_level("kern.memorystatus_vm_pressure_level: 2"), "warn");
        assert_eq!(parse_pressure_level("4"), "critical");
        assert_eq!(parse_pressure_level(""), "normal");
        assert_eq!(parse_pressure_level("x"), "normal");
    }

    #[test]
    fn meminfo_and_ps() {
        let m = parse_meminfo("MemTotal:       16384 kB\nMemAvailable:    8192 kB\nHugePages_Total: 0\n");
        assert_eq!(m["MemTotal"], 16384 * 1024);
        assert_eq!(m["HugePages_Total"], 0);
        let p = parse_ps("  101  2048 /usr/bin/omlx serve --port 8000\nbad line\n 7 x y\n 9 10 a\n");
        assert_eq!(p, vec![(101, 2048 * 1024, "/usr/bin/omlx serve --port 8000".into()), (9, 10240, "a".into())]);
    }

    fn guard(p: FakeProbe, protected: Vec<Box<dyn Protected>>) -> MemoryGuard {
        MemoryGuard { probe: Box::new(p), protected, headroom_bytes: 8 * GB, headroom_auto: false, swap_growth_bytes: 256 << 20, max_fraction: None }
    }

    #[test]
    fn no_reserve_by_default() {
        assert_eq!(auto_headroom(16 * GB), 0);
        assert_eq!(auto_headroom(512 * GB), 0);
        // the user's example: 24 GB Mac, 17 GB reclaimable, a model that peaks at 12 GB starts
        let mut g = guard(FakeProbe::new(24.0, 17.0), vec![]);
        g.headroom_auto = true;
        let d = g.check(12 * GB, "a 12 GB model");
        assert!(d.ok, "{}", d.reason);
        assert_eq!(d.reason, "17.0 GB free right now covers 12.0 GB for a 12 GB model.");
        assert!(!g.check(18 * GB, "x").ok);
        // 16 GB Mac with 6.5 GB free: a 6.4 GB peak starts
        let mut g = guard(FakeProbe::new(16.0, 6.5), vec![]);
        g.headroom_auto = true;
        assert!(g.check((6.4 * GB as f64) as u64, "Wan 1.3B").ok);
        // a headroom the user set still counts
        g.headroom_auto = false;
        assert!(!g.check(GB, "a 1 GB job").ok, "the 8 GB headroom set in the test guard refuses it");
    }

    #[test]
    fn watchdog_relieves_then_stops() {
        let start = FakeProbe::new(16.0, 6.0).sample().unwrap();
        let mut w = Watchdog::new(GB, 10.0, &start);
        let mut s = start.clone();
        s.pressure = "warn".into();
        assert_eq!(w.step(&s, 0.0), WatchAction::Calm, "warn never acts");
        s.swap_used = start.swap_used + GB + GB / 5;
        assert!(matches!(w.step(&s, 1.0), WatchAction::Relieve(r) if r.contains("swap grew by 1.2 GB")));
        assert_eq!(w.step(&s, 5.0), WatchAction::Calm, "inside the grace window");
        let held = s.clone();
        s.swap_used += GB * 6 / 10;
        assert!(matches!(w.step(&s, 11.5), WatchAction::Stop(r) if r.contains("swap kept growing")));
        // swap that grew once and then holds steady does not stop the job
        let mut w = Watchdog::new(GB, 10.0, &start);
        assert!(matches!(w.step(&held, 1.0), WatchAction::Relieve(_)));
        assert_eq!(w.step(&held, 12.0), WatchAction::Calm);
        assert_eq!(w.step(&held, 60.0), WatchAction::Calm);
        // a relief that works resets
        let mut w = Watchdog::new(GB, 10.0, &start);
        let mut s = start.clone();
        s.pressure = "critical".into();
        assert!(matches!(w.step(&s, 0.0), WatchAction::Relieve(_)));
        s.pressure = "normal".into();
        assert_eq!(w.step(&s, 20.0), WatchAction::Calm);
        assert_eq!(w.max_pressure, "critical");
        // a jump past the hard limit stops at once
        let mut w = Watchdog::new(GB, 10.0, &start);
        let mut s = start.clone();
        s.swap_used = start.swap_used + 4 * GB;      // hard limit = 3x the soft one (0.3.0)
        assert!(matches!(w.step(&s, 0.0), WatchAction::Stop(r) if r.contains("limit 3.0 GB")));
        // 0.3.0 default: 256 MB soft, so a run that grows swap by 1.1 GB and then holds is stopped
        let mut w = Watchdog::new(256 * 1024 * 1024, 10.0, &start);
        let mut s = start.clone();
        s.swap_used = start.swap_used + 1100 * 1024 * 1024;
        assert!(matches!(w.step(&s, 0.0), WatchAction::Stop(_)));
    }

    #[test]
    fn guard_decisions() {
        let g = guard(FakeProbe::new(64.0, 48.0), vec![]);
        let d = g.check(10 * GB, "job a (dummy)");
        assert!(d.ok, "{}", d.reason);
        assert_eq!(d.reason, "40.0 GB free right now (after the 8.0 GB headroom you set) covers 10.0 GB for job a (dummy).");
        let d = g.check(41 * GB, "");
        assert!(!d.ok);
        assert_eq!(
            d.reason,
            "Not enough free memory for this job: it needs 41.0 GB, but macOS can give only 40.0 GB right now (after the 8.0 GB headroom you set)."
        );
        let p = FakeProbe::new(64.0, 48.0);
        p.set(|s| s.pressure = "warn".into());
        assert!(guard(p, vec![]).check(1, "job x").ok, "warn pressure admits");
        let p = FakeProbe::new(64.0, 48.0);
        p.set(|s| s.pressure = "critical".into());
        let g = guard(p, vec![]);
        assert_eq!(
            g.check(1, "job x").reason,
            "macOS reports critical memory pressure; waiting until it eases before starting job x."
        );
        let mut g = guard(FakeProbe::new(64.0, 48.0), vec![]);
        g.max_fraction = Some(0.5);
        assert_eq!(
            g.check(33 * GB, "job x").reason,
            "Job x needs 33.0 GB, more than the 50% limit of this Mac's 64.0 GB (32.0 GB)."
        );
    }

    #[test]
    fn guard_protected_first_and_probe_error() {
        let ps: PsRunner = Box::new(|| Ok(format!("  42 {} /opt/omlx/bin/omlx serve\n", 30 * 1024 * 1024)));
        let svc = ProcessRssService::new("omlx|splash", 20 * GB, Some(ps));
        assert_eq!(svc.name(), "processes matching 'omlx|splash'");
        let g = guard(FakeProbe::new(64.0, 48.0), vec![Box::new(svc)]);
        let d = g.check(1, "");
        assert!(!d.ok);
        assert_eq!(d.reason, "omlx is using 30.0 GB of memory (more than 20.0 GB); Shuttrr waits so it does not compete with it.");
        let p = FakeProbe::new(1.0, 1.0);
        *p.s.lock().unwrap() = Err("vm_stat failed".into());
        let g = guard(p, vec![]);
        let d = g.check(1, "");
        assert!(!d.ok && d.reason.contains("vm_stat failed"));
    }

    fn fetcher(map: Vec<(&'static str, u16, &'static str)>) -> Fetcher {
        Box::new(move |url: &str, _h: &[(&str, &str)], _t: Duration| {
            for (suffix, st, body) in &map {
                if url.ends_with(suffix) {
                    return Ok((*st, body.as_bytes().to_vec()));
                }
            }
            Err("connection refused".into())
        })
    }

    #[test]
    fn omlx() {
        let t = Duration::from_millis(10);
        let s = OmlxService::new("http://x:8000/", None, t, Some(fetcher(vec![])));
        assert!(s.busy().is_none());
        assert!(s.last_error().unwrap().contains("unreachable"));
        let s = OmlxService::new(
            "http://x",
            None,
            t,
            Some(fetcher(vec![("/v1/models/status", 200, r#"{"models":[{"id":"qwen","loaded":true},{"id":"b","loaded":false}]}"#)])),
        );
        assert_eq!(
            s.busy().unwrap(),
            "oMLX has a model loaded (qwen); Shuttrr waits so it does not compete with it for memory."
        );
        let s = OmlxService::new(
            "http://x",
            None,
            t,
            Some(fetcher(vec![
                ("/v1/models/status", 404, "nf"),
                ("/v1/models", 200, r#"{"data":[{"id":"a","status":{"value":"loaded"}},{"id":"b","state":"ready"},{"id":"c","status":"ready"},{"id":"d","loaded":1}]}"#),
            ])),
        );
        assert!(s.busy().unwrap().contains("(a, b, c and more)"));
        let s = OmlxService::new(
            "http://x",
            None,
            t,
            Some(fetcher(vec![("/v1/models/status", 200, r#"{"x":1}"#), ("/v1/models", 200, r#"{"data":[{"id":"a"}]}"#)])),
        );
        assert!(s.busy().is_none());
        assert!(s.last_error().unwrap().contains("no load-state"));
        assert_eq!(omlx_loaded(&json!({"loaded_models": ["m1", {"name": "m2"}, 3]})), Some(vec!["m1".into(), "m2".into()]));
        assert_eq!(omlx_loaded(&json!([{"id": "a", "loaded": false}])), Some(vec![]));
    }
}
