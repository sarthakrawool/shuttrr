//! Test harness: start the real `shutterd serve` on a free port with the fake worker.
#![allow(dead_code)]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub const GB: u64 = 1024 * 1024 * 1024;

pub fn fake_worker() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fake_worker.py").to_string_lossy().into_owned()
}

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

pub struct Daemon {
    pub child: Option<Child>,
    pub port: u16,
    pub home: PathBuf,
    _tmp: Option<tempfile::TempDir>,
    pub stdout_rest: Option<std::thread::JoinHandle<String>>,
}

pub fn base_config(home: &Path) -> Value {
    json!({
        "worker_cmd": ["python3", fake_worker()],
        "memory_poll_s": 0.05,
        "watch_interval_s": 0.05,
        "sample_interval_s": 0.05,
        "idle_unload_s": 60,
        "cancel_grace_s": 5,
        "shutdown_grace_s": 1,
        "sse_heartbeat_s": 0.2,
        "worker_start_timeout_s": 20,
        "estimate_timeout_s": 10,
        "busy_process_pattern": null,
        "omlx_url": null,
        "protected_process_pattern": null,
        "test_probe_file": home.join("probe.json").to_string_lossy(),
        "static_dir": home.join("static").to_string_lossy(),
        "some_future_key": {"ignored": true},
    })
}

fn merge(base: &mut Value, extra: &Value) {
    if let (Some(b), Some(e)) = (base.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            b.insert(k.clone(), v.clone());
        }
    }
}

impl Daemon {
    pub fn start(extra: Value) -> Daemon {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        let mut d = Self::start_in(&home, extra);
        d._tmp = Some(tmp);
        d
    }

    /// Start (or restart) with this home. Writes config.json and, if missing, probe.json.
    pub fn start_in(home: &Path, extra: Value) -> Daemon {
        let mut cfg = base_config(home);
        merge(&mut cfg, &extra);
        std::fs::write(home.join("config.json"), serde_json::to_string_pretty(&cfg).unwrap()).unwrap();
        if !home.join("probe.json").exists() {
            set_probe_in(home, json!({"total": 64 * GB, "available": 48 * GB, "swap_used": 0, "pressure": "normal"}));
        }
        Self::spawn(home, &["serve", "--port", "0"])
    }

    pub fn spawn(home: &Path, args: &[&str]) -> Daemon {
        let mut child = Command::new(env!("CARGO_BIN_EXE_shutterd"))
            .args(args)
            .env("SHUTTER_HOME", home)
            .env_remove("SHUTTER_STATIC")
            .env_remove("SHUTTER_URL")
            .env("FAKE_WORKER_OPS_LOG", home.join("ops.log"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start shutterd");
        let stdout = child.stdout.take().unwrap();
        let mut r = BufReader::new(stdout);
        let mut line = String::new();
        r.read_line(&mut line).unwrap();
        let port: u16 = line
            .split("http://")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.rsplit(':').next())
            .and_then(|p| p.parse().ok())
            .unwrap_or_else(|| panic!("no port in {line:?}"));
        let rest = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = r.read_to_string(&mut s);
            s
        });
        Daemon { child: Some(child), port, home: home.to_path_buf(), _tmp: None, stdout_rest: Some(rest) }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{}", self.port, path)
    }

    pub fn req(&self, method: &str, path: &str, body: Option<&[u8]>) -> (u16, Value) {
        let headers = [("Content-Type", "application/json")];
        let r = shutterd::httpc::request(method, &self.url(path), &headers, body, Duration::from_secs(20))
            .unwrap_or_else(|e| panic!("{method} {path}: {e}"));
        let v = serde_json::from_slice(&r.body).unwrap_or(Value::String(r.text()));
        (r.status, v)
    }

    pub fn raw(&self, method: &str, path: &str) -> shutterd::httpc::Response {
        shutterd::httpc::request(method, &self.url(path), &[], None, Duration::from_secs(20)).unwrap()
    }

    pub fn get(&self, path: &str) -> Value {
        let (st, v) = self.req("GET", path, None);
        assert_eq!(st, 200, "GET {path}: {v}");
        v
    }

    pub fn post(&self, path: &str, body: Value) -> (u16, Value) {
        let b = body.to_string();
        self.req("POST", path, Some(b.as_bytes()))
    }

    pub fn submit(&self, body: Value) -> Value {
        let (st, v) = self.post("/v1/jobs", body);
        assert_eq!(st, 201, "submit: {v}");
        v
    }

    pub fn job(&self, id: &str) -> Value {
        self.get(&format!("/v1/jobs/{id}"))
    }

    pub fn wait_state(&self, id: &str, states: &[&str], timeout_s: f64) -> Value {
        let t0 = Instant::now();
        loop {
            let j = self.job(id);
            if states.contains(&j["state"].as_str().unwrap_or("")) {
                return j;
            }
            if t0.elapsed().as_secs_f64() > timeout_s {
                panic!("job {id} never reached {states:?}: {j}");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    pub fn wait_for<F: Fn(&Daemon) -> bool>(&self, what: &str, timeout_s: f64, f: F) {
        let t0 = Instant::now();
        while !f(self) {
            if t0.elapsed().as_secs_f64() > timeout_s {
                panic!("timed out waiting for {what}");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    pub fn set_probe(&self, v: Value) {
        set_probe_in(&self.home, v);
    }

    pub fn ops(&self) -> Vec<String> {
        std::fs::read_to_string(self.home.join("ops.log"))
            .unwrap_or_default()
            .lines()
            .map(String::from)
            .collect()
    }

    pub fn worker_running(&self) -> bool {
        self.get("/v1/system")["worker"]["running"].as_bool().unwrap()
    }

    pub fn signal(&self, sig: i32) {
        if let Some(c) = &self.child {
            unsafe {
                libc::kill(c.id() as i32, sig);
            }
        }
    }

    /// SIGTERM and wait; returns the exit code.
    pub fn stop(&mut self) -> Option<i32> {
        self.signal(libc::SIGTERM);
        let mut c = self.child.take()?;
        let t0 = Instant::now();
        loop {
            if let Ok(Some(st)) = c.try_wait() {
                return st.code();
            }
            if t0.elapsed() > Duration::from_secs(15) {
                let _ = c.kill();
                let _ = c.wait();
                panic!("daemon did not stop on SIGTERM");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    pub fn kill9(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }

    /// Read SSE frames until `n` data events arrived or `timeout_s` passed.
    pub fn sse(&self, query: &str, n: usize, timeout_s: f64) -> (Vec<Value>, Vec<String>) {
        sse_collect(self.port, query, timeout_s, |evs| evs.len() >= n)
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        self.kill9();
    }
}

pub fn set_probe_in(home: &Path, v: Value) {
    let tmp = home.join("probe.json.tmp");
    std::fs::write(&tmp, v.to_string()).unwrap();
    std::fs::rename(&tmp, home.join("probe.json")).unwrap();
}

/// Open /v1/events and collect data events until `done(events)` or timeout. Returns the
/// events and the comment lines seen.
pub fn sse_collect<F: Fn(&[Value]) -> bool>(port: u16, query: &str, timeout_s: f64, done: F) -> (Vec<Value>, Vec<String>) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(s, "GET /v1/events{query} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").unwrap();
    s.set_read_timeout(Some(Duration::from_millis(100))).unwrap();
    let mut buf = Vec::new();
    let mut evs = Vec::new();
    let mut comments = Vec::new();
    let t0 = Instant::now();
    let mut chunk = [0u8; 4096];
    let mut scanned = 0usize;
    while t0.elapsed().as_secs_f64() < timeout_s {
        match s.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => {}
        }
        let text = String::from_utf8_lossy(&buf).into_owned();
        let lines: Vec<&str> = text.split('\n').collect();
        // only complete lines
        let complete = lines.len().saturating_sub(1);
        for l in &lines[scanned.min(complete)..complete] {
            let l = l.trim_end_matches('\r');
            if let Some(d) = l.strip_prefix("data: ") {
                if let Ok(v) = serde_json::from_str::<Value>(d) {
                    evs.push(v);
                }
            } else if l.starts_with(':') {
                comments.push(l.to_string());
            }
        }
        scanned = complete;
        if done(&evs) {
            break;
        }
    }
    (evs, comments)
}

/// A one-shot raw HTTP request returning the head and body as text.
pub fn raw_http(port: u16, request: &str) -> String {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.write_all(request.as_bytes()).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut out = Vec::new();
    let _ = s.read_to_end(&mut out);
    String::from_utf8_lossy(&out).into_owned()
}
