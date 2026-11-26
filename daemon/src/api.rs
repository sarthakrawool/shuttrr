//! HTTP API (axum). Paths, status codes and JSON shapes of the Python reference server
//! (`shutter/server/api.py`). Loopback-only unless `api_key` is set; then non-loopback
//! clients must send `Authorization: Bearer <key>` or `?key=<key>`.

use crate::engine::{resolve_static_dir, ApiError, Engine, VERSION};
use crate::store::{Job, STATES};
use crate::util::{now, round_to};
use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, DefaultBodyLimit, Path as AxPath, Query, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::{IpAddr, SocketAddr};
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};

pub const RECENT_EVENTS: i64 = 50;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub api_key: Option<String>,
    pub static_dir: Option<PathBuf>,
}

pub fn is_loopback(host: &str) -> bool {
    if host.is_empty() {
        return false;
    }
    if host == "localhost" {
        return true;
    }
    host.trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<IpAddr>()
        .map(|ip| match ip {
            IpAddr::V4(v) => v.is_loopback(),
            IpAddr::V6(v) => v.is_loopback() || v.to_ipv4_mapped().map(|m| m.is_loopback()).unwrap_or(false),
        })
        .unwrap_or(false)
}

/// Refuse a non-loopback bind without an api key.
pub fn check_bind(host: &str, api_key: Option<&str>) -> Result<(), String> {
    if !is_loopback(host) && api_key.map(str::is_empty).unwrap_or(true) {
        return Err(format!(
            "refusing to listen on {host}: Shuttrr has no auth on loopback. Set \"api_key\" in config.json to listen on other interfaces (clients then send Authorization: Bearer)."
        ));
    }
    Ok(())
}

fn json_resp(status: u16, v: &Value) -> Response {
    let mut r = Response::new(Body::from(v.to_string()));
    *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    r
}

fn err(status: u16, msg: &str) -> Response {
    json_resp(status, &json!({"error": msg, "detail": msg}))
}

fn api_err(e: ApiError) -> Response {
    err(e.0, &e.1)
}

fn job_json(job: &Job) -> Value {
    let mut d = job.to_json();
    if job.state == "running" && job.progress > 0.01 {
        if let Some(s) = job.started_at {
            let el = (now() - s).max(0.0);
            d["eta_s"] = json!(round_to(el * (1.0 - job.progress) / job.progress, 1));
        }
    }
    d
}

fn constant_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn auth(State(s): State<AppState>, req: Request, next: Next) -> Response {
    if let Some(key) = s.api_key.as_deref().filter(|k| !k.is_empty()) {
        let client = req
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|c| c.0.ip().to_string())
            .unwrap_or_default();
        if !is_loopback(&client) {
            let h = req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).unwrap_or("");
            let mut token = if h.len() >= 7 && h[..7].eq_ignore_ascii_case("bearer ") { h[7..].trim().to_string() } else { String::new() };
            if token.is_empty() {
                if let Some(q) = req.uri().query() {
                    for pair in q.split('&') {
                        if let Some(v) = pair.strip_prefix("key=") {
                            token = percent_decode(v);
                        }
                    }
                }
            }
            if !constant_eq(token.as_bytes(), key.as_bytes()) {
                return err(401, "missing or wrong API key (Authorization: Bearer ...)");
            }
        }
    }
    if let Some(c) = req.headers().get("x-shutter-client").and_then(|v| v.to_str().ok()) {
        s.engine.clients.seen(crate::stats::client_label(Some(c), None, None).as_deref());
    }
    next.run(req).await
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => {
                match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16) {
                    Ok(v) => {
                        out.push(v);
                        i += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn router(state: AppState) -> Router {
    let upload_limit = DefaultBodyLimit::max(state.engine.cfg.max_upload_bytes().saturating_add(1 << 20));
    Router::new()
        .route("/v1/jobs/{id}/output", get(job_output))
        .route("/v1/files", post(upload).layer(upload_limit))
        .route("/v1/memory", get(memory))
        .route("/v1/clients", get(clients))
        .route("/v1/images/generations", post(images_generations).layer(DefaultBodyLimit::max(16 << 20)))
        .route("/v1/images/edits", post(images_edits).layer(upload_limit))
        .route("/", get(dashboard))
        .route("/static/{*path}", get(static_file))
        .route("/v1/system", get(system))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}/load", post(load_model))
        .route("/v1/models/{id}/unload", post(unload_model))
        .route("/v1/catalog", get(catalog))
        .route("/v1/catalog/{id}/download", post(start_download).delete(stop_download))
        .route("/v1/settings", get(get_settings).post(post_settings))
        .route("/v1/jobs", get(list_jobs).post(submit))
        .route("/v1/jobs/{id}", get(get_job).delete(cancel_job))
        .route("/v1/queue/pause", post(pause))
        .route("/v1/queue/resume", post(resume))
        .route("/v1/pool/release", post(release))
        .route("/v1/events", get(events))
        .fallback(|| async { err(404, "Not Found") })
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .with_state(state)
}

async fn system(State(s): State<AppState>) -> Response {
    json_resp(200, &s.engine.system().await)
}

async fn get_settings(State(s): State<AppState>) -> Response {
    json_resp(200, &s.engine.settings())
}

async fn post_settings(State(s): State<AppState>, body: Bytes) -> Response {
    match s.engine.update_settings(&body) {
        Ok(v) => json_resp(200, &v),
        Err(e) => api_err(e),
    }
}

async fn models(State(s): State<AppState>) -> Response {
    json_resp(200, &s.engine.models().await)
}

// ---- load / unload. The Python engine (`shutter serve`) loads a model ahead of a request through its queue and
// memory guard; this daemon's worker protocol has no load step yet, so it says so instead of pretending.

async fn load_model(AxPath(id): AxPath<String>) -> Response {
    err(501, &format!("This engine cannot load {id} ahead of a request yet; it loads when a request needs it. \
                       The Shuttrr app's own engine (shutter serve) has Load and Unload."))
}

async fn unload_model(AxPath(id): AxPath<String>) -> Response {
    err(501, &format!("This engine cannot unload {id} by name yet. Use Unload All (POST /v1/pool/release) instead."))
}

// ---- model downloads. The download is a `shutter download` process that writes its progress to a file; the
// daemon only starts it, stops it and asks the Python side for the catalog (docs/INSTALL.md).

fn shutter_cli(s: &AppState, args: &[&str]) -> Option<tokio::process::Command> {
    let cfg = &s.engine.cfg;
    let prog = cfg.worker_cmd.first()?;
    let mut cmd = tokio::process::Command::new(prog);
    cmd.args(["-m", "shutter.cli"]).args(args).env("SHUTTER_HOME", &cfg.home).stdin(std::process::Stdio::null());
    if let Some(d) = &cfg.worker_cwd {
        cmd.current_dir(d);
    }
    Some(cmd)
}

async fn catalog_value(s: &AppState) -> Result<Value, String> {
    let mut cmd = shutter_cli(s, &["--json", "catalog"]).ok_or("worker_cmd is empty")?;
    let out = tokio::time::timeout(Duration::from_secs(20), cmd.output())
        .await
        .map_err(|_| "the model list took too long to read".to_string())?
        .map_err(|e| format!("could not run Python: {e}"))?;
    if !out.status.success() {
        let text = String::from_utf8_lossy(&out.stderr);
        return Err(text.lines().last().unwrap_or("the model list could not be read").to_string());
    }
    let mut v: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("bad model list: {e}"))?;
    // A model that finished downloading after this engine started needs a restart to be used.
    let restart = v.get("models").and_then(Value::as_array).is_some_and(|rows| {
        rows.iter().any(|r| {
            r.get("installed").and_then(Value::as_bool).unwrap_or(false)
                && r.get("id").and_then(Value::as_str).is_some_and(|id| !s.engine.cfg.models.contains_key(id))
        })
    });
    if let Some(o) = v.as_object_mut() {
        o.insert("restart_needed".into(), json!(restart));
    }
    Ok(v)
}

async fn catalog(State(s): State<AppState>) -> Response {
    match catalog_value(&s).await {
        Ok(v) => json_resp(200, &v),
        Err(e) => err(500, &e),
    }
}

fn valid_model_id(id: &str) -> bool {
    !id.is_empty() && id.len() < 100 && id.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
}

async fn start_download(
    State(s): State<AppState>,
    AxPath(id): AxPath<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !valid_model_id(&id) {
        return err(404, "unknown model");
    }
    let before = match catalog_value(&s).await {
        Ok(v) => v,
        Err(e) => return err(500, &e),
    };
    let row = before["models"].as_array().and_then(|a| a.iter().find(|r| r["id"] == id.as_str())).cloned();
    let Some(row) = row else {
        return err(404, &format!("unknown model '{id}'"));
    };
    // the catalog's fit classifier (Python, catalog.classify_fit) is the one source of truth
    let forced = matches!(q.get("force").map(String::as_str), Some("1") | Some("true"));
    if row["fit"]["level"] == "no_fit" && !forced {
        let name = row["name"].as_str().unwrap_or(&id);
        let why = row["fit"]["reason"].as_str().unwrap_or("");
        return err(409, &format!("{name} does not fit this Mac. {why}"));
    }
    if row["download"]["state"] != "downloading" {
        let Some(mut cmd) = shutter_cli(&s, &["download", &id, "--quiet"]) else {
            return err(500, "worker_cmd is empty");
        };
        let log = std::fs::OpenOptions::new().create(true).append(true).open(s.engine.cfg.home.join("download.log"));
        match log.and_then(|f| Ok((f.try_clone()?, f))) {
            Ok((a, b)) => {
                cmd.stdout(a).stderr(b);
            }
            Err(_) => {
                cmd.stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
            }
        }
        // its own session: the download outlives an engine restart
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
        match cmd.spawn() {
            Ok(mut child) => {
                tokio::spawn(async move {
                    let _ = child.wait().await;
                });
            }
            Err(e) => return err(500, &format!("could not start the download: {e}")),
        }
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }
    match catalog_value(&s).await {
        Ok(v) => json_resp(202, &v),
        Err(e) => err(500, &e),
    }
}

async fn stop_download(State(s): State<AppState>, AxPath(id): AxPath<String>) -> Response {
    if !valid_model_id(&id) {
        return err(404, "unknown model");
    }
    if let Some(mut cmd) = shutter_cli(&s, &["download", &id, "--cancel"]) {
        let _ = tokio::time::timeout(Duration::from_secs(20), cmd.output()).await;
    }
    tokio::time::sleep(Duration::from_millis(400)).await;
    match catalog_value(&s).await {
        Ok(v) => json_resp(200, &v),
        Err(e) => err(500, &e),
    }
}

fn header<'a>(h: &'a HeaderMap, k: &str) -> Option<&'a str> {
    h.get(k).and_then(|v| v.to_str().ok())
}

async fn submit(State(s): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let data = match Engine::parse_body(&body) {
        Ok(d) => d,
        Err(e) => return api_err(e),
    };
    match s.engine.submit_value(data, header(&headers, "x-shutter-client"), header(&headers, "user-agent")) {
        Ok(job) => json_resp(201, &job_json(&job)),
        Err(e) => api_err(e),
    }
}

async fn job_output(State(s): State<AppState>, AxPath(id): AxPath<String>) -> Response {
    match s.engine.output_file(&id) {
        Ok(p) => match std::fs::read(&p) {
            Ok(bytes) => {
                let mut r = Response::new(Body::from(bytes));
                let h = r.headers_mut();
                h.insert(header::CONTENT_TYPE, HeaderValue::from_static(crate::files::content_type_for(&p)));
                let name = crate::files::safe_name(p.file_name().and_then(|n| n.to_str()), None);
                if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{name}\"")) {
                    h.insert(header::CONTENT_DISPOSITION, v);
                }
                r
            }
            Err(e) => err(404, &format!("the output file of job {id} cannot be read: {e}")),
        },
        Err(e) => api_err(e),
    }
}

async fn upload(State(s): State<AppState>, headers: HeaderMap, Query(q): Query<HashMap<String, String>>, body: Bytes) -> Response {
    let ctype = header(&headers, "content-type").unwrap_or("");
    let (name, data, ptype): (Option<String>, Vec<u8>, Option<String>) = if let Some(b) = crate::files::boundary(ctype) {
        let parts = match crate::files::parse_multipart(&body, &b) {
            Ok(p) => p,
            Err(e) => return err(400, &format!("bad upload: {e}")),
        };
        let pick = parts
            .iter()
            .position(|p| p.name == "file" && p.filename.is_some())
            .or_else(|| parts.iter().position(|p| p.filename.is_some()));
        let Some(i) = pick else {
            return err(400, "multipart upload needs a file part (field name \"file\")");
        };
        let p = parts.into_iter().nth(i).unwrap_or_else(|| unreachable!());
        (p.filename, p.data, p.content_type)
    } else {
        let name = q.get("name").cloned().or_else(|| header(&headers, "x-filename").map(String::from));
        (name, body.to_vec(), (!ctype.is_empty()).then(|| ctype.to_string()))
    };
    match s.engine.save_upload(name.as_deref(), &data, ptype.as_deref()) {
        Ok(v) => json_resp(201, &v),
        Err(e) => api_err(e),
    }
}

async fn memory(State(s): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let mut extra = serde_json::Map::new();
    for k in ["width", "height", "frames", "steps"] {
        if let Some(v) = q.get(k).filter(|v| !v.is_empty()) {
            match v.trim().parse::<i64>() {
                Ok(n) => {
                    extra.insert(k.into(), json!(n));
                }
                Err(_) => return err(400, &format!("{k} must be an integer")),
            }
        }
    }
    let model = q.get("model").map(String::as_str).filter(|m| !m.is_empty());
    json_resp(200, &s.engine.memory(model, extra).await)
}

async fn clients(State(s): State<AppState>) -> Response {
    json_resp(200, &s.engine.clients.list())
}

fn base_url(headers: &HeaderMap) -> String {
    let host = header(headers, "host").unwrap_or("127.0.0.1:8484");
    format!("http://{host}")
}

fn shim_resp(r: Result<Value, crate::images::ShimError>) -> Response {
    match r {
        Ok(v) => json_resp(200, &v),
        Err(e) => json_resp(e.status, &e.body()),
    }
}

async fn images_generations(State(s): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let client = crate::images::shim_client(header(&headers, "x-shutter-client"), header(&headers, "user-agent"));
    shim_resp(crate::images::generations(s.engine.clone(), &body, client, &base_url(&headers)).await)
}

async fn images_edits(State(s): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let client = crate::images::shim_client(header(&headers, "x-shutter-client"), header(&headers, "user-agent"));
    let ctype = header(&headers, "content-type").unwrap_or("").to_string();
    shim_resp(crate::images::edits(s.engine.clone(), &ctype, &body, client, &base_url(&headers)).await)
}

async fn list_jobs(State(s): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let limit = match q.get("limit").map(|l| l.trim().parse::<i64>()) {
        None => 200,
        Some(Ok(n)) => n,
        Some(Err(_)) => return err(400, "limit must be an integer"),
    };
    let mut states = None;
    if let Some(st) = q.get("state").filter(|s| !s.is_empty()) {
        let mut v = Vec::new();
        for x in st.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            if !STATES.contains(&x) {
                return err(400, &format!("state must be one of: {}", STATES.join(", ")));
            }
            v.push(x.to_string());
        }
        states = Some(v);
    }
    match s.engine.list(states.as_deref(), limit.clamp(1, 1000)) {
        Ok(jobs) => json_resp(200, &json!({"jobs": jobs.iter().map(job_json).collect::<Vec<_>>()})),
        Err(e) => err(500, &e),
    }
}

async fn get_job(State(s): State<AppState>, AxPath(id): AxPath<String>) -> Response {
    match s.engine.get(&id) {
        Ok(Some(j)) => json_resp(200, &job_json(&j)),
        Ok(None) => err(404, &format!("no job {id}")),
        Err(e) => err(500, &e),
    }
}

async fn cancel_job(State(s): State<AppState>, AxPath(id): AxPath<String>) -> Response {
    match s.engine.cancel(&id) {
        Ok(Some(j)) => json_resp(200, &job_json(&j)),
        Ok(None) => err(404, &format!("no job {id}")),
        Err(e) => err(500, &e),
    }
}

async fn pause(State(s): State<AppState>) -> Response {
    s.engine.pause();
    json_resp(200, &json!({"paused": true}))
}

async fn resume(State(s): State<AppState>) -> Response {
    s.engine.resume();
    json_resp(200, &json!({"paused": false}))
}

async fn release(State(s): State<AppState>, body: Bytes) -> Response {
    let mut key: Option<String> = None;
    if !body.iter().all(u8::is_ascii_whitespace) {
        let Ok(data) = serde_json::from_slice::<Value>(&body) else {
            return err(400, "body must be JSON");
        };
        if let Some(k) = data.get("key") {
            match k {
                Value::Null => {}
                Value::String(s) => key = Some(s.clone()),
                _ => return err(400, "key must be a string or null"),
            }
        }
    }
    let released = s.engine.release(key.as_deref()).await;
    json_resp(200, &json!({"released": released}))
}

// ---- SSE ------------------------------------------------------------------------------

struct ChanStream(mpsc::Receiver<Bytes>);

impl futures_core::Stream for ChanStream {
    type Item = Result<Bytes, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.0.poll_recv(cx).map(|o| o.map(Ok))
    }
}

async fn events(State(s): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let parse = |k: &str, d: i64| q.get(k).map(|v| v.trim().parse::<i64>()).unwrap_or(Ok(d));
    let (Ok(max_events), Ok(recent_n)) = (parse("max_events", 0), parse("recent", RECENT_EVENTS)) else {
        return err(400, "max_events and recent must be integers");
    };
    let (mut rx, recent) = s.engine.bus.subscribe_with_recent(recent_n.max(0) as usize);
    let (tx, brx) = mpsc::channel::<Bytes>(64);
    let heartbeat = crate::util::secs(s.engine.cfg.sse_heartbeat_s.max(0.05));
    tokio::spawn(async move {
        let frame = |v: &Value| Bytes::from(format!("data: {v}\n\n"));
        if tx.send(Bytes::from_static(b": shutter events\n\n")).await.is_err() {
            return;
        }
        let mut sent = 0i64;
        for ev in recent {
            if tx.send(frame(&ev.to_json())).await.is_err() {
                return;
            }
            sent += 1;
            if max_events > 0 && sent >= max_events {
                return;
            }
        }
        let mut hb = tokio::time::interval_at(tokio::time::Instant::now() + heartbeat, heartbeat);
        loop {
            tokio::select! {
                r = rx.recv() => match r {
                    Ok(ev) => {
                        if tx.send(frame(&ev.to_json())).await.is_err() {
                            return;
                        }
                        sent += 1;
                        if max_events > 0 && sent >= max_events {
                            return;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return,
                },
                _ = hb.tick() => {
                    if tx.send(Bytes::from_static(b": heartbeat\n\n")).await.is_err() {
                        return;
                    }
                }
                _ = tx.closed() => return,
            }
        }
    });
    let mut r = Response::new(Body::from_stream(ChanStream(brx)));
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream; charset=utf-8"));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    h.insert("x-accel-buffering", HeaderValue::from_static("no"));
    r
}

// ---- dashboard and static files ---------------------------------------------------------

pub fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("html") | Some("htm") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") | Some("mjs") => "text/javascript; charset=utf-8",
        Some("json") | Some("map") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("ttf") => "font/ttf",
        Some("otf") => "font/otf",
        Some("wasm") => "application/wasm",
        Some("txt") | Some("md") => "text/plain; charset=utf-8",
        Some("mp4") => "video/mp4",
        Some("webm") => "video/webm",
        _ => "application/octet-stream",
    }
}

/// Resolve a request path under `root`, refusing anything that could leave it.
pub fn safe_join(root: &Path, rel: &str) -> Option<PathBuf> {
    if rel.is_empty() || rel.contains('\0') || rel.contains('\\') || rel.starts_with('/') {
        return None;
    }
    let p = Path::new(rel);
    if p.components().any(|c| !matches!(c, Component::Normal(_))) {
        return None;
    }
    let full = root.join(p);
    let canon_root = root.canonicalize().ok()?;
    let canon = full.canonicalize().ok()?;
    (canon.starts_with(&canon_root) && canon.is_file()).then_some(canon)
}

fn file_response(path: &Path) -> Response {
    match std::fs::read(path) {
        Ok(bytes) => {
            let mut r = Response::new(Body::from(bytes));
            r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type(path)));
            r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            r
        }
        Err(_) => err(404, "Not Found"),
    }
}

async fn dashboard(State(s): State<AppState>) -> Response {
    if let Some(d) = &s.static_dir {
        let p = d.join("dashboard.html");
        if p.is_file() {
            return file_response(&p);
        }
    }
    let mut r = Response::new(Body::from(format!(
        "<!doctype html><title>Shutter</title><h1>Shutter</h1><p>dashboard.html is missing (shutterd {VERSION}).</p>"
    )));
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
    r
}

async fn static_file(State(s): State<AppState>, AxPath(path): AxPath<String>) -> Response {
    let Some(root) = s.static_dir.as_ref().map(|d| d.join("static")) else {
        return err(404, "Not Found");
    };
    match safe_join(&root, &path) {
        Some(p) => file_response(&p),
        None => err(404, "Not Found"),
    }
}

// ---- serve ----------------------------------------------------------------------------

/// Run the daemon until SIGTERM/SIGINT. Prints one line with the bound address on stdout.
pub async fn serve(engine: Arc<Engine>, host: &str, port: u16) -> Result<(), String> {
    check_bind(host, engine.cfg.api_key.as_deref())?;
    let static_dir = resolve_static_dir(&engine.cfg);
    let state = AppState { engine: engine.clone(), api_key: engine.cfg.api_key.clone(), static_dir };
    let bind_host = host.trim_start_matches('[').trim_end_matches(']');
    let bind_host = if bind_host == "localhost" { "127.0.0.1" } else { bind_host };
    let listener = tokio::net::TcpListener::bind((bind_host, port))
        .await
        .map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    for w in &engine.cfg.warnings {
        crate::log!("shutterd: config: {w}");
    }
    {
        use std::io::Write;
        let shown = if addr.is_ipv6() { format!("[{}]", addr.ip()) } else { addr.ip().to_string() };
        let _ = writeln!(std::io::stdout(), "shutterd {VERSION} on http://{shown}:{}  (home {})", addr.port(), engine.cfg.home.display());
        let _ = std::io::stdout().flush();
    }
    let sched = tokio::spawn(engine.clone().run());
    let app = router(state).into_make_service_with_connect_info::<SocketAddr>();
    let server = axum::serve(listener, app);
    let signals = async {
        use tokio::signal::unix::{signal, SignalKind};
        match (signal(SignalKind::terminate()), signal(SignalKind::interrupt())) {
            (Ok(mut t), Ok(mut i)) => {
                tokio::select! { _ = t.recv() => {}, _ = i.recv() => {} }
            }
            _ => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        r = server => { if let Err(e) = r { crate::log!("shutterd: server error: {e}"); } }
        _ = signals => {}
    }
    engine.shutdown();
    let wait = Duration::from_secs_f64(engine.cfg.shutdown_grace_s.max(0.0) + 5.0);
    let _ = tokio::time::timeout(wait, sched).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_rules() {
        assert!(is_loopback("127.0.0.1") && is_loopback("::1") && is_loopback("localhost") && is_loopback("[::1]"));
        assert!(!is_loopback("0.0.0.0") && !is_loopback("testclient") && !is_loopback(""));
        assert!(check_bind("127.0.0.1", None).is_ok());
        assert!(check_bind("0.0.0.0", None).is_err());
        assert!(check_bind("0.0.0.0", Some("")).is_err());
        assert!(check_bind("0.0.0.0", Some("key")).is_ok());
    }

    #[test]
    fn static_paths() {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("static");
        std::fs::create_dir_all(root.join("css")).unwrap();
        std::fs::write(root.join("css/a.css"), "x").unwrap();
        std::fs::write(d.path().join("secret.txt"), "s").unwrap();
        std::os::unix::fs::symlink(d.path().join("secret.txt"), root.join("link.txt")).unwrap();
        assert!(safe_join(&root, "css/a.css").is_some());
        assert!(safe_join(&root, "../secret.txt").is_none());
        assert!(safe_join(&root, "css/../../secret.txt").is_none());
        assert!(safe_join(&root, "/etc/passwd").is_none());
        assert!(safe_join(&root, "link.txt").is_none(), "symlinks out of the root are refused");
        assert!(safe_join(&root, "css").is_none());
        assert!(safe_join(&root, "a\\b").is_none());
        assert_eq!(content_type(Path::new("x.JS")), "text/javascript; charset=utf-8");
        assert_eq!(content_type(Path::new("x.woff2")), "font/woff2");
        assert_eq!(content_type(Path::new("x")), "application/octet-stream");
    }

    #[test]
    fn decode() {
        assert_eq!(percent_decode("a%20b+c"), "a b c");
        assert_eq!(percent_decode("s3cret"), "s3cret");
        assert_eq!(percent_decode("x%"), "x%");
        assert_eq!(percent_decode("x%zz"), "x%zz");
    }
}
