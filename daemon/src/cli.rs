//! Command line: `shutterd serve` runs the daemon; every other subcommand talks to it
//! over HTTP. Human-readable output by default, `--json` for the raw answer.

use crate::config::Config;
use crate::engine::{Engine, VERSION};
use crate::httpc;
use crate::util::GB;
use serde_json::{json, Map, Value};
use std::time::Duration;

const USAGE: &str = "shutterd: the Shutter engine daemon

usage: shutterd [--json] [--url URL] <command> [options]

commands:
  serve [--host H] [--port P]       run the daemon (default 127.0.0.1:8484 from config)
  submit --model ID [--prompt T] [--prompt-file F] [--negative-prompt T]
         [--image F] [--audio F] [--video F] [--input ROLE=F ...] [--output F]
         [--width N] [--height N] [--frames N] [--fps X] [--steps N] [--guidance X]
         [--seed N] [--duration S] [--priority N]
         [--param K=V ...] [--mode fast|low_memory] [--efficiency X] [--wait]
                                    queue a job (K=V values are parsed as JSON when possible;
                                    without --mode/--efficiency the engine defaults apply)
  status [ID] [--limit N] [--state S]  list jobs, or show one
  cancel ID                         cancel a job
  settings [--mode M] [--efficiency X]
                                    show, or change, the default mode (fast | low_memory) and
                                    the low-memory efficiency (0..1) with the ladder
  models                            list models
  system                            memory, queue, GPU lock, warm components
  pause | resume                    stop / restart taking new jobs
  unload [KEY]                      drop warm components (all when no KEY)
  version                           print the version

client options: --url URL (or SHUTTER_URL), --host/--port, SHUTTER_API_KEY for a remote daemon.
config: $SHUTTER_HOME/config.json (default ~/.shutter/config.json).";

struct Args {
    json: bool,
    url: Option<String>,
    host: Option<String>,
    port: Option<u16>,
    rest: Vec<String>,
}

fn parse_global(argv: Vec<String>) -> Result<Args, String> {
    let mut a = Args { json: false, url: None, host: None, port: None, rest: Vec::new() };
    let mut it = argv.into_iter();
    while let Some(x) = it.next() {
        match x.as_str() {
            "--json" => a.json = true,
            "--url" => a.url = Some(it.next().ok_or("--url needs a value")?),
            "--host" => a.host = Some(it.next().ok_or("--host needs a value")?),
            "--port" => {
                let v = it.next().ok_or("--port needs a value")?;
                a.port = Some(v.parse().map_err(|_| format!("--port: not a port: {v}"))?);
            }
            _ => {
                if let Some(v) = x.strip_prefix("--url=") {
                    a.url = Some(v.into());
                } else if let Some(v) = x.strip_prefix("--host=") {
                    a.host = Some(v.into());
                } else if let Some(v) = x.strip_prefix("--port=") {
                    a.port = Some(v.parse().map_err(|_| format!("--port: not a port: {v}"))?);
                } else {
                    a.rest.push(x)
                }
            }
        }
    }
    Ok(a)
}

/// Options of one subcommand: flags with values, boolean flags and positionals.
struct Opts {
    vals: Vec<(String, String)>,
    flags: Vec<String>,
    pos: Vec<String>,
}

fn parse_opts(args: &[String], bool_flags: &[&str], value_flags: &[&str]) -> Result<Opts, String> {
    let mut o = Opts { vals: vec![], flags: vec![], pos: vec![] };
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(name) = a.strip_prefix("--") {
            let (name, inline) = match name.split_once('=') {
                Some((n, v)) => (n.to_string(), Some(v.to_string())),
                None => (name.to_string(), None),
            };
            if bool_flags.contains(&name.as_str()) {
                o.flags.push(name);
            } else if value_flags.contains(&name.as_str()) {
                let v = match inline {
                    Some(v) => v,
                    None => {
                        i += 1;
                        args.get(i).cloned().ok_or_else(|| format!("--{name} needs a value"))?
                    }
                };
                o.vals.push((name, v));
            } else {
                return Err(format!("unknown option --{name}"));
            }
        } else {
            o.pos.push(a.clone());
        }
        i += 1;
    }
    Ok(o)
}

impl Opts {
    fn get(&self, k: &str) -> Option<&str> {
        self.vals.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
    }
    fn all(&self, k: &str) -> Vec<&str> {
        self.vals.iter().filter(|(n, _)| n == k).map(|(_, v)| v.as_str()).collect()
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
}

struct Client {
    base: String,
    api_key: Option<String>,
}

impl Client {
    fn from_args(a: &Args) -> Client {
        let cfg = Config::load().ok();
        let url = a.url.clone().or_else(|| std::env::var("SHUTTER_URL").ok().filter(|s| !s.is_empty()));
        let base = url.unwrap_or_else(|| {
            let mut host = a.host.clone().or_else(|| cfg.as_ref().map(|c| c.host.clone())).unwrap_or_else(|| "127.0.0.1".into());
            if host == "0.0.0.0" || host == "::" || host.is_empty() {
                host = "127.0.0.1".into();
            }
            if host.contains(':') && !host.starts_with('[') {
                host = format!("[{host}]");
            }
            let port = a.port.or_else(|| cfg.as_ref().map(|c| c.port)).unwrap_or(8484);
            format!("http://{host}:{port}")
        });
        let api_key = std::env::var("SHUTTER_API_KEY").ok().filter(|s| !s.is_empty()).or_else(|| cfg.and_then(|c| c.api_key));
        Client { base: base.trim_end_matches('/').to_string(), api_key }
    }

    fn request(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Value, String> {
        let auth = self.api_key.as_ref().map(|k| format!("Bearer {k}"));
        let mut headers = vec![("Accept", "application/json")];
        if body.is_some() {
            headers.push(("Content-Type", "application/json"));
        }
        if let Some(a) = auth.as_deref() {
            headers.push(("Authorization", a));
        }
        let data = body.map(|b| b.to_string().into_bytes());
        let r = httpc::request(method, &format!("{}{}", self.base, path), &headers, data.as_deref(), Duration::from_secs(30))
            .map_err(|e| format!("cannot reach the Shuttrr engine at {} ({e}). Start it with `shutterd serve`.", self.base))?;
        let v: Value = if r.body.is_empty() { Value::Null } else { serde_json::from_slice(&r.body).unwrap_or(Value::Null) };
        if r.status >= 400 {
            let msg = v.get("error").and_then(Value::as_str).map(String::from).unwrap_or_else(|| r.text());
            return Err(format!("HTTP {}: {msg}", r.status));
        }
        Ok(v)
    }
}

fn gb(v: &Value) -> String {
    match v.as_f64() {
        Some(n) => format!("{:.1} GB", n / GB),
        None => "-".into(),
    }
}

fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::as_str).unwrap_or("")
}

pub fn job_line(j: &Value) -> String {
    let p = j.get("progress").and_then(Value::as_f64).unwrap_or(0.0);
    let state = s(j, "state");
    let detail = if state == "running" { s(j, "stage") } else { s(j, "message") };
    let model = j.get("request").map(|r| s(r, "model")).unwrap_or("");
    format!("{}  {:<15} {:5.1}%  {:<26} {}", s(j, "id"), state, 100.0 * p, model, detail).trim_end().to_string()
}

fn out(json_mode: bool, data: &Value, text: String) {
    if json_mode {
        println!("{}", serde_json::to_string_pretty(data).unwrap_or_default());
    } else {
        println!("{text}");
    }
}

fn parse_param(p: &str) -> Result<(String, Value), String> {
    let (k, v) = p.split_once('=').ok_or_else(|| format!("--param needs key=value, got {p:?}"))?;
    let k = k.trim();
    if k.is_empty() {
        return Err(format!("--param needs a key, got {p:?}"));
    }
    Ok((k.to_string(), serde_json::from_str(v).unwrap_or_else(|_| Value::String(v.to_string()))))
}

fn build_request(o: &Opts) -> Result<Value, String> {
    let model = o.get("model").ok_or("submit needs --model ID")?;
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("client".into(), json!(format!("shutterd-cli/{VERSION}")));
    let int = |k: &str, d: i64| -> Result<i64, String> {
        o.get(k).map(|v| v.parse::<i64>().map_err(|_| format!("--{k} must be an integer"))).unwrap_or(Ok(d))
    };
    body.insert("seed".into(), json!(int("seed", 42)?));
    body.insert("priority".into(), json!(int("priority", 0)?));
    if let Some(p) = o.get("prompt") {
        body.insert("prompt".into(), json!(p));
    }
    if let Some(f) = o.get("prompt-file") {
        let t = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
        body.insert("prompt".into(), json!(t.trim()));
    }
    let mut inputs = Map::new();
    for k in ["image", "audio", "video"] {
        if let Some(v) = o.get(k).filter(|v| !v.is_empty()) {
            inputs.insert(k.into(), json!(crate::util::abspath(v).to_string_lossy()));
        }
    }
    for p in o.all("input") {
        let (k, v) = p.split_once('=').ok_or_else(|| format!("--input needs role=path, got {p:?}"))?;
        inputs.insert(k.trim().to_string(), json!(crate::util::abspath(v).to_string_lossy()));
    }
    if !inputs.is_empty() {
        body.insert("inputs".into(), Value::Object(inputs));
    }
    if let Some(v) = o.get("output").filter(|v| !v.is_empty()) {
        body.insert("output".into(), json!(crate::util::abspath(v).to_string_lossy()));
    }
    if let Some(v) = o.get("negative-prompt") {
        body.insert("negative_prompt".into(), json!(v));
    }
    for k in ["width", "height", "frames", "steps"] {
        if let Some(v) = o.get(k) {
            body.insert(k.into(), json!(v.parse::<i64>().map_err(|_| format!("--{k} must be an integer"))?));
        }
    }
    for k in ["fps", "guidance"] {
        if let Some(v) = o.get(k) {
            body.insert(k.into(), json!(v.parse::<f64>().map_err(|_| format!("--{k} must be a number"))?));
        }
    }
    if let Some(d) = o.get("duration") {
        body.insert("duration_s".into(), json!(d.parse::<f64>().map_err(|_| "--duration must be a number")?));
    }
    let params = o.all("param");
    let mut m = Map::new();
    for p in params {
        let (k, v) = parse_param(p)?;
        m.insert(k, v);
    }
    let (mode, eff) = mode_opts(o)?;
    if let Some(x) = mode {
        m.insert("mode".into(), json!(x));
    }
    if let Some(e) = eff {
        m.insert("efficiency".into(), json!(e));
    }
    if !m.is_empty() {
        body.insert("params".into(), Value::Object(m));
    }
    Ok(Value::Object(body))
}

/// `--mode` and `--efficiency`, checked here so a typo fails before anything is sent.
fn mode_opts(o: &Opts) -> Result<(Option<&'static str>, Option<f64>), String> {
    let mode = match o.get("mode") {
        Some(m) => Some(crate::modes::normalize_mode(m).ok_or_else(|| format!("--mode must be fast or low_memory, got {m:?}"))?),
        None => None,
    };
    let eff = match o.get("efficiency") {
        Some(e) => {
            let x: f64 = e.trim().parse().map_err(|_| format!("--efficiency must be a number from 0 to 1, got {e:?}"))?;
            if !(0.0..=1.0).contains(&x) {
                return Err(format!("--efficiency must be from 0 to 1, got {e}"));
            }
            Some(x)
        }
        None => None,
    };
    Ok((mode, eff))
}

pub fn settings_text(v: &Value) -> String {
    let mode = s(v, "mode");
    let eff = v["efficiency"].as_f64().unwrap_or(0.0);
    let rung = s(v, "rung");
    let mut lines = vec![
        format!("mode        {}", if mode == "low_memory" { "low memory" } else { mode }),
        format!("efficiency  {eff:.2}{}", if mode == "fast" { " (used in low memory mode)" } else { "" }),
        format!("rung        {rung}"),
    ];
    if let Some(m) = v["measured_for"].as_object() {
        lines.push(format!(
            "measured    {} at {}",
            m.get("model").and_then(Value::as_str).unwrap_or("?"),
            m.get("size").and_then(Value::as_str).unwrap_or("?")
        ));
    }
    lines.push("ladder".into());
    for r in v["ladder"].as_array().cloned().unwrap_or_default() {
        let here = if s(&r, "id") == rung { "*" } else { " " };
        let peak = r.get("peak_bytes").map(gb).unwrap_or_else(|| "–".into());
        let secs = r.get("seconds").and_then(Value::as_f64).map(|x| format!("{x:.0} s")).unwrap_or_else(|| "–".into());
        lines.push(format!(
            "  {here} {:.2}  {:<13} {:>9}  {:>6}  {}",
            r["at"].as_f64().unwrap_or(0.0),
            s(&r, "id"),
            peak,
            secs,
            s(&r, "detail")
        ));
    }
    lines.join("\n")
}

const TERMINAL: [&str; 3] = ["done", "failed", "cancelled"];

fn wait_for(c: &Client, id: &str, poll: f64, to_stderr: bool) -> Result<Value, String> {
    let mut last = String::new();
    loop {
        let j = c.request("GET", &format!("/v1/jobs/{id}"), None)?;
        let line = job_line(&j);
        if line != last {
            if to_stderr {
                eprintln!("{line}");
            } else {
                println!("{line}");
            }
            last = line;
        }
        if TERMINAL.contains(&s(&j, "state")) {
            return Ok(j);
        }
        std::thread::sleep(crate::util::secs(poll.max(0.05)));
    }
}

fn system_text(sy: &Value) -> String {
    let m = &sy["memory"];
    let q = &sy["queue"];
    let g = &sy["gpu_lock"];
    let used = m["total"].as_f64().unwrap_or(0.0) - m["available"].as_f64().unwrap_or(0.0);
    let mut lines = vec![
        format!("Shuttrr {} (shutterd)", s(sy, "version")),
        format!(
            "memory   {} used of {}, {} available, swap {}, pressure {}",
            gb(&json!(used)),
            gb(&m["total"]),
            gb(&m["available"]),
            gb(&m["swap_used"]),
            s(m, "pressure")
        ),
        format!(
            "headroom {}{}",
            gb(&sy["guard"]["headroom_bytes"]),
            if sy["guard"]["headroom_auto"].as_bool().unwrap_or(false) { " (automatic: an eighth of memory, 2 to 8 GB)" } else { "" }
        ),
    ];
    for p in sy["guard"]["protected"].as_array().cloned().unwrap_or_default() {
        lines.push(format!("protect  {}: {}", s(&p, "name"), p["busy"].as_str().unwrap_or("idle")));
    }
    let n = |k: &str| q[k].as_i64().unwrap_or(0);
    lines.push(format!(
        "queue    {} queued, {} waiting, {} running, {} done, {} failed{}",
        n("queued"),
        n("waiting_memory"),
        n("running"),
        n("done"),
        n("failed"),
        if q["paused"].as_bool().unwrap_or(false) { " (paused)" } else { "" }
    ));
    let held = g["held"].as_bool().unwrap_or(false);
    let holder = g["holder"].get("label").and_then(Value::as_str).unwrap_or("another process");
    lines.push(format!("gpu lock {}", if held { format!("held by {holder}") } else { "free".into() }));
    let w = &sy["worker"];
    if w["running"].as_bool().unwrap_or(false) {
        lines.push(format!("worker   pid {} (version {}), up {} s", w["pid"], s(w, "version"), w["uptime_s"]));
    } else {
        lines.push("worker   not running (started on the next job)".into());
    }
    let entries = sy["pool"]["entries"].as_array().cloned().unwrap_or_default();
    lines.push(format!("pool     {} in {} components", gb(&sy["pool"]["total_bytes"]), entries.len()));
    for e in entries {
        lines.push(format!(
            "         {}  {}  idle {:.0} s",
            s(&e, "key"),
            gb(&e["size_bytes"]),
            e["idle_s"].as_f64().unwrap_or(0.0)
        ));
    }
    lines.join("\n")
}

fn serve(a: &Args) -> i32 {
    let cfg = match Config::load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("shutterd: {e}");
            return 2;
        }
    };
    let host = a.host.clone().unwrap_or_else(|| cfg.host.clone());
    let port = a.port.unwrap_or(cfg.port);
    if let Err(e) = crate::api::check_bind(&host, cfg.api_key.as_deref()) {
        eprintln!("shutterd: {e}");
        return 2;
    }
    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("shutterd: cannot start the runtime: {e}");
            return 1;
        }
    };
    rt.block_on(async move {
        let engine = match Engine::new(cfg) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("shutterd: {e}");
                return 1;
            }
        };
        match crate::api::serve(engine, &host, port).await {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("shutterd: {e}");
                2
            }
        }
    })
}

fn run(a: Args) -> Result<i32, String> {
    let Some(cmd) = a.rest.first().cloned() else {
        println!("{USAGE}");
        return Ok(2);
    };
    let rest = &a.rest[1..];
    let j = a.json;
    match cmd.as_str() {
        "serve" => {
            parse_opts(rest, &[], &[])?;
            Ok(serve(&a))
        }
        "help" | "-h" | "--help" => {
            println!("{USAGE}");
            Ok(0)
        }
        "version" | "--version" | "-V" => {
            println!("shutterd {VERSION}");
            Ok(0)
        }
        "submit" => {
            let o = parse_opts(
                rest,
                &["wait"],
                &["model", "prompt", "prompt-file", "negative-prompt", "image", "audio", "video", "input", "output", "width",
                  "height", "frames", "fps", "steps", "guidance", "seed", "duration", "priority", "param", "poll", "mode",
                  "efficiency"],
            )?;
            let body = build_request(&o)?;
            let c = Client::from_args(&a);
            let job = c.request("POST", "/v1/jobs", Some(&body))?;
            if !o.flag("wait") {
                out(j, &job, s(&job, "id").to_string());
                return Ok(0);
            }
            if !j {
                println!("submitted {}", s(&job, "id"));
            }
            let poll = o.get("poll").and_then(|p| p.parse().ok()).unwrap_or(1.0);
            let fin = wait_for(&c, s(&job, "id"), poll, j)?;
            if j {
                out(true, &fin, String::new());
            } else if s(&fin, "state") == "done" {
                let r = &fin["result"];
                let secs = r.get("seconds").map(|v| v.to_string()).unwrap_or_else(|| "?".into());
                println!("done in {secs} s, peak {}: {}", gb(&r["peak_bytes"]), s(r, "output"));
            } else {
                println!("{}: {}", s(&fin, "state"), s(&fin, "message"));
                if let Some(t) = fin["result"]["log_tail"].as_str().filter(|t| !t.is_empty()) {
                    let lines: Vec<&str> = t.lines().collect();
                    for l in &lines[lines.len().saturating_sub(10)..] {
                        println!("  | {l}");
                    }
                }
            }
            Ok(if s(&fin, "state") == "done" { 0 } else { 1 })
        }
        "status" => {
            let o = parse_opts(rest, &[], &["limit", "state"])?;
            let c = Client::from_args(&a);
            if let Some(id) = o.pos.first() {
                let jb = c.request("GET", &format!("/v1/jobs/{id}"), None)?;
                let mut lines = vec![job_line(&jb)];
                for k in ["message", "stage"] {
                    if !s(&jb, k).is_empty() {
                        lines.push(format!("  {k}: {}", s(&jb, k)));
                    }
                }
                if let Some(r) = jb["result"].as_object() {
                    for (k, v) in r {
                        if k == "traceback" || k == "log_tail" {
                            continue;
                        }
                        let shown = v.as_str().map(String::from).unwrap_or_else(|| v.to_string());
                        lines.push(format!("  {k}: {shown}"));
                    }
                    if let Some(t) = r.get("log_tail").and_then(Value::as_str).filter(|t| !t.is_empty()) {
                        lines.push("  log tail:".into());
                        let ls: Vec<&str> = t.lines().collect();
                        for l in &ls[ls.len().saturating_sub(20)..] {
                            lines.push(format!("    | {l}"));
                        }
                    }
                }
                out(j, &jb, lines.join("\n"));
                return Ok(0);
            }
            let limit = o.get("limit").unwrap_or("20");
            let mut path = format!("/v1/jobs?limit={limit}");
            if let Some(st) = o.get("state") {
                path.push_str(&format!("&state={st}"));
            }
            let d = c.request("GET", &path, None)?;
            let lines: Vec<String> = d["jobs"].as_array().map(|a| a.iter().map(job_line).collect()).unwrap_or_default();
            out(j, &d, if lines.is_empty() { "no jobs".into() } else { lines.join("\n") });
            Ok(0)
        }
        "cancel" => {
            let o = parse_opts(rest, &[], &[])?;
            let id = o.pos.first().ok_or("cancel needs a job id")?;
            let jb = Client::from_args(&a).request("DELETE", &format!("/v1/jobs/{id}"), None)?;
            out(j, &jb, job_line(&jb));
            Ok(0)
        }
        "settings" => {
            let o = parse_opts(rest, &[], &["mode", "efficiency"])?;
            let (mode, eff) = mode_opts(&o)?;
            let c = Client::from_args(&a);
            let v = if mode.is_none() && eff.is_none() {
                c.request("GET", "/v1/settings", None)?
            } else {
                let mut b = Map::new();
                if let Some(m) = mode {
                    b.insert("mode".into(), json!(m));
                }
                if let Some(e) = eff {
                    b.insert("efficiency".into(), json!(e));
                }
                c.request("POST", "/v1/settings", Some(&Value::Object(b)))?
            };
            out(j, &v, settings_text(&v));
            Ok(0)
        }
        "models" => {
            parse_opts(rest, &[], &[])?;
            let d = Client::from_args(&a).request("GET", "/v1/models", None)?;
            let rows: Vec<String> = d["models"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .map(|m| {
                    let status = match m["error"].as_str() {
                        Some(e) => format!("error: {e}"),
                        None if m["installed"].as_bool().unwrap_or(false) => "installed".into(),
                        None => "not installed".into(),
                    };
                    format!("{:<28} {:<16} {}  {}", s(m, "id"), s(m, "family"), status, s(m, "dir"))
                })
                .collect();
            out(j, &d, if rows.is_empty() { "no models configured".into() } else { rows.join("\n") });
            Ok(0)
        }
        "system" => {
            parse_opts(rest, &[], &[])?;
            let sy = Client::from_args(&a).request("GET", "/v1/system", None)?;
            out(j, &sy, system_text(&sy));
            Ok(0)
        }
        "pause" | "resume" => {
            parse_opts(rest, &[], &[])?;
            let d = Client::from_args(&a).request("POST", &format!("/v1/queue/{cmd}"), None)?;
            out(j, &d, if cmd == "pause" { "queue paused (the running job continues)".into() } else { "queue resumed".into() });
            Ok(0)
        }
        "unload" => {
            let o = parse_opts(rest, &[], &[])?;
            let body = json!({"key": o.pos.first()});
            let d = Client::from_args(&a).request("POST", "/v1/pool/release", Some(&body))?;
            let keys: Vec<String> =
                d["released"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect()).unwrap_or_default();
            out(j, &d, if keys.is_empty() { "nothing unloaded".into() } else { format!("unloaded {}", keys.join(", ")) });
            Ok(0)
        }
        other => Err(format!("unknown command {other:?} (try `shutterd help`)")),
    }
}

pub fn main(argv: Vec<String>) -> i32 {
    let a = match parse_global(argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("shutterd: {e}");
            return 2;
        }
    };
    match run(a) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("shutterd: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_building() {
        let args: Vec<String> = ["--model", "dummy", "--param", "steps=3", "--param", "name=x", "--priority=2", "--wait"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let o = parse_opts(&args, &["wait"], &["model", "param", "priority"]).unwrap();
        let b = build_request(&o).unwrap();
        assert_eq!(b["params"], json!({"steps": 3, "name": "x"}));
        assert_eq!(b["priority"], 2);
        assert_eq!(b["seed"], 42);
        assert!(o.flag("wait"));
        assert!(parse_opts(&["--nope".into()], &[], &[]).is_err());
        assert!(parse_param("=1").is_err() && parse_param("x").is_err());
        let args: Vec<String> = ["--model", "dummy", "--mode", "low-memory", "--efficiency", "0.75"].iter().map(|s| s.to_string()).collect();
        let o = parse_opts(&args, &[], &["model", "mode", "efficiency"]).unwrap();
        assert_eq!(build_request(&o).unwrap()["params"], json!({"mode": "low_memory", "efficiency": 0.75}));
        for bad in [["--mode", "turbo"], ["--efficiency", "2"], ["--efficiency", "x"]] {
            let mut a: Vec<String> = vec!["--model".into(), "dummy".into()];
            a.extend(bad.iter().map(|s| s.to_string()));
            let o = parse_opts(&a, &[], &["model", "mode", "efficiency"]).unwrap();
            assert!(build_request(&o).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn lines() {
        let j = json!({"id": "abc", "state": "running", "progress": 0.5, "stage": "denoise 1/2",
                       "message": "", "request": {"model": "dummy"}});
        assert_eq!(job_line(&j), format!("abc  {:<15}  50.0%  {:<26} denoise 1/2", "running", "dummy"));
        let g = parse_global(vec!["--json".into(), "status".into(), "--port".into(), "9".into()]).unwrap();
        assert!(g.json && g.port == Some(9) && g.rest == vec!["status"]);
    }
}
