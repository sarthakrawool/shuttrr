//! Integration tests: the real daemon on a free port with the fake worker.

mod common;
use common::*;
use serde_json::{json, Value};
use std::io::{BufRead, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn kinds_for(evs: &[Value], id: &str) -> Vec<String> {
    evs.iter()
        .filter(|e| e["job_id"] == id)
        .map(|e| e["kind"].as_str().unwrap_or("").to_string())
        .collect()
}

#[test]
fn job_runs_to_done_with_progress_over_sse() {
    let d = Daemon::start(json!({}));
    let port = d.port;
    let h = std::thread::spawn(move || {
        sse_collect(port, "?recent=0", 20.0, |evs| evs.iter().any(|e| e["kind"] == "job.done"))
    });
    std::thread::sleep(Duration::from_millis(300));
    let job = d.submit(json!({"model": "dummy", "prompt": "hi", "params": {"steps": 4, "step_s": 0.05}}));
    let id = job["id"].as_str().unwrap().to_string();
    assert_eq!(job["state"], "queued");
    let (evs, _) = h.join().unwrap();
    let kinds = kinds_for(&evs, &id);
    assert_eq!(kinds.first().map(String::as_str), Some("job.queued"));
    assert!(kinds.contains(&"job.started".to_string()), "{kinds:?}");
    let progress: Vec<f64> = evs
        .iter()
        .filter(|e| e["kind"] == "job.progress" && e["job_id"] == id.as_str())
        .map(|e| e["data"]["progress"].as_f64().unwrap())
        .collect();
    assert!(progress.len() >= 2, "{progress:?}");
    assert!(progress.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(*progress.last().unwrap(), 1.0);
    assert_eq!(kinds.last().map(String::as_str), Some("job.done"));
    assert!(evs.iter().any(|e| e["kind"] == "worker.started"));
    let j = d.wait_state(&id, &["done"], 10.0);
    let r = &j["result"];
    assert!(r["output"].as_str().unwrap().ends_with(&format!("outputs/{id}.mp4")), "{r}");
    assert!(std::path::Path::new(r["output"].as_str().unwrap()).is_file());
    assert_eq!(r["peak_bytes"], 123456789);
    assert!(r["seconds"].is_number() && r["peak_memory"].is_object());
    assert_eq!(r["estimate_peak_bytes"], GB);
    assert_eq!(j["progress"], 1.0);
    assert_eq!(j["request"]["output"], r["output"]);
    // work dir removed after success
    assert!(!d.home.join("work").join(&id).exists());
    // the GPU lock is free again and the worker stays warm
    let s = d.get("/v1/system");
    assert_eq!(s["gpu_lock"]["ours"], false);
    assert_eq!(s["worker"]["running"], true);
}

#[test]
fn priority_order_and_pause_resume() {
    let d = Daemon::start(json!({}));
    assert_eq!(d.post("/v1/queue/pause", json!({})).1, json!({"paused": true}));
    let fast = json!({"steps": 1, "step_s": 0.01});
    let low = d.submit(json!({"model": "dummy", "priority": 0, "params": fast}));
    let low2 = d.submit(json!({"model": "dummy", "priority": 0, "params": fast}));
    let high = d.submit(json!({"model": "dummy", "priority": 5, "params": fast}));
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(d.job(low["id"].as_str().unwrap())["state"], "queued", "paused: nothing starts");
    assert_eq!(d.get("/v1/system")["queue"]["paused"], true);
    assert_eq!(d.post("/v1/queue/resume", json!({})).1, json!({"paused": false}));
    let ids = [high["id"].as_str().unwrap(), low["id"].as_str().unwrap(), low2["id"].as_str().unwrap()];
    let started: Vec<f64> = ids
        .iter()
        .map(|id| d.wait_state(id, &["done"], 15.0)["started_at"].as_f64().unwrap())
        .collect();
    assert!(started[0] < started[1] && started[1] < started[2], "priority, then oldest: {started:?}");
}

#[test]
fn cancel_queued_and_running() {
    let d = Daemon::start(json!({}));
    let run = d.submit(json!({"model": "dummy", "params": {"steps": 200, "step_s": 0.05}}));
    let rid = run["id"].as_str().unwrap();
    let queued = d.submit(json!({"model": "dummy"}));
    let qid = queued["id"].as_str().unwrap();
    d.wait_state(rid, &["running"], 10.0);
    let (st, c) = d.req("DELETE", &format!("/v1/jobs/{qid}"), None);
    assert_eq!(st, 200);
    assert_eq!(c["state"], "cancelled");
    assert_eq!(c["message"], "cancelled before it started");
    let (_, c) = d.req("DELETE", &format!("/v1/jobs/{rid}"), None);
    assert_eq!(c["state"], "running");
    assert_eq!(c["message"], "cancelling");
    let j = d.wait_state(rid, &["cancelled", "failed", "done"], 10.0);
    assert_eq!(j["state"], "cancelled");
    assert_eq!(j["message"], "cancelled while running");
    assert!(d.ops().contains(&"cancel".to_string()));
    assert!(d.worker_running(), "a cooperative cancel keeps the worker");
    assert_eq!(d.get("/v1/system")["gpu_lock"]["ours"], false);
}

#[test]
fn cancel_grace_then_kill() {
    let d = Daemon::start(json!({"cancel_grace_s": 0.5}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 5, "step_s": 0.02, "hang": true}}));
    let id = j["id"].as_str().unwrap();
    d.wait_for("first progress", 10.0, |d| d.job(id)["progress"].as_f64().unwrap() > 0.0);
    let pid = d.get("/v1/system")["worker"]["pid"].as_i64().unwrap() as i32;
    let t0 = Instant::now();
    d.req("DELETE", &format!("/v1/jobs/{id}"), None);
    let j = d.wait_state(id, &["cancelled", "failed"], 10.0);
    assert_eq!(j["state"], "cancelled");
    assert!(j["message"].as_str().unwrap().contains("did not stop within 0.5 s"), "{j}");
    assert!(t0.elapsed() >= Duration::from_millis(450));
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(shutterd::util::pid_alive(pid), Some(false), "hung worker was killed");
    assert!(!d.worker_running());
    // the next job gets a fresh worker
    let n = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0}}));
    d.wait_state(n["id"].as_str().unwrap(), &["done"], 15.0);
}

#[test]
fn slow_cancel_within_grace_is_cooperative() {
    let d = Daemon::start(json!({"cancel_grace_s": 3}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 100, "step_s": 0.05, "slow_cancel_s": 0.4}}));
    let id = j["id"].as_str().unwrap();
    d.wait_state(id, &["running"], 10.0);
    d.req("DELETE", &format!("/v1/jobs/{id}"), None);
    let j = d.wait_state(id, &["cancelled"], 10.0);
    assert_eq!(j["message"], "cancelled while running");
    assert!(d.worker_running());
}

#[test]
fn memory_refusal_waits_then_runs_when_memory_frees() {
    let d = Daemon::start(json!({}));
    d.set_probe(json!({"total": 64 * GB, "available": GB, "swap_used": 0}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0, "peak_gb": 2}}));
    let id = j["id"].as_str().unwrap();
    let w = d.wait_state(id, &["waiting_memory"], 10.0);
    let id12 = &id;
    assert_eq!(
        w["message"],
        format!("Not enough free memory for job {id12} (dummy): it needs 2.0 GB, but macOS can give only 1.0 GB right now.")
    );
    let s = d.get("/v1/system");
    assert_eq!(s["queue"]["waiting_memory"], 1);
    assert_eq!(s["memory"]["available"], GB);
    // critical pressure has its own reason ("warn" would admit)
    d.set_probe(json!({"total": 64 * GB, "available": 48 * GB, "swap_used": 0, "pressure": "critical"}));
    d.wait_for("pressure reason", 10.0, |d| d.job(id)["message"].as_str().unwrap().contains("critical memory pressure"));
    d.set_probe(json!({"total": 64 * GB, "available": 48 * GB, "swap_used": 0}));
    d.wait_state(id, &["done"], 10.0);
    let (evs, _) = d.sse("?recent=100", 1000, 0.5);
    let kinds = kinds_for(&evs, id);
    assert!(kinds.iter().filter(|k| *k == "job.waiting_memory").count() >= 2, "{kinds:?}");
}

/// A tiny fake oMLX server; records every request line.
fn fake_omlx(body: Arc<Mutex<String>>, seen: Arc<Mutex<Vec<String>>>) -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let Ok(mut s) = s else { continue };
            let mut r = std::io::BufReader::new(s.try_clone().unwrap());
            let mut first = String::new();
            let _ = r.read_line(&mut first);
            loop {
                let mut h = String::new();
                if r.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
                    break;
                }
            }
            seen.lock().unwrap().push(first.trim().to_string());
            let b = body.lock().unwrap().clone();
            let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", b.len(), b);
        }
    });
    port
}

#[test]
fn protected_service_busy_waits_and_is_only_read() {
    let body = Arc::new(Mutex::new(r#"{"models":[{"id":"qwen-72b","loaded":true}]}"#.to_string()));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let port = fake_omlx(body.clone(), seen.clone());
    let d = Daemon::start(json!({"omlx_url": format!("http://127.0.0.1:{port}")}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0}}));
    let id = j["id"].as_str().unwrap();
    let w = d.wait_state(id, &["waiting_memory"], 10.0);
    assert_eq!(w["message"], "oMLX has a model loaded (qwen-72b); Shuttrr waits so it does not compete with it for memory.");
    let s = d.get("/v1/system");
    assert_eq!(s["guard"]["protected"][0]["name"], "oMLX");
    assert!(s["guard"]["protected"][0]["busy"].as_str().unwrap().contains("qwen-72b"));
    *body.lock().unwrap() = r#"{"models":[{"id":"qwen-72b","loaded":false}]}"#.to_string();
    d.wait_state(id, &["done"], 10.0);
    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty());
    assert!(seen.iter().all(|l| l.starts_with("GET /v1/models")), "read-only: {seen:?}");
}

#[test]
fn swap_growth_mid_run_kills_worker_and_releases_locks() {
    let tmp = tempfile::tempdir().unwrap();
    let lockdir = tmp.path().join("studio-gpu.lock.d");
    let d = Daemon::start(json!({"gpu_lock_dirs": [lockdir.to_string_lossy()]}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 400, "step_s": 0.05}}));
    let id = j["id"].as_str().unwrap();
    d.wait_state(id, &["running"], 10.0);
    let s = d.get("/v1/system");
    assert_eq!(s["gpu_lock"]["ours"], true);
    assert_eq!(s["gpu_lock"]["dirs"][0]["ours"], true);
    assert!(lockdir.join("owner").is_file());
    let pid = s["worker"]["pid"].as_i64().unwrap() as i32;
    d.set_probe(json!({"total": 64 * GB, "available": 40 * GB, "swap_used": 3 * GB}));
    let j = d.wait_state(id, &["failed"], 10.0);
    assert_eq!(
        j["message"],
        "Stopped to protect your Mac: it was running out of memory. Close other apps or pick a smaller size. \
         (swap grew by 3.0 GB since the request started (limit 2.0 GB))"
    );
    assert_eq!(j["result"]["watchdog"]["events"][0]["action"], "stop");
    assert!(j["result"]["log_tail"].is_string());
    assert!(!lockdir.exists(), "our dir lock is released");
    let s = d.get("/v1/system");
    assert_eq!(s["gpu_lock"]["ours"], false);
    assert_eq!(s["gpu_lock"]["held"], false);
    assert_eq!(s["worker"]["running"], false);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(shutterd::util::pid_alive(pid), Some(false));
    // critical pressure first frees caches; if it stays critical past the 10 s grace the run stops
    d.set_probe(json!({"total": 64 * GB, "available": 40 * GB, "swap_used": 3 * GB}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 400, "step_s": 0.05}}));
    let id = j["id"].as_str().unwrap();
    d.wait_state(id, &["running"], 10.0);
    d.set_probe(json!({"total": 64 * GB, "available": 40 * GB, "swap_used": 3 * GB, "pressure": "critical"}));
    let j = d.wait_state(id, &["failed"], 20.0);
    let m = j["message"].as_str().unwrap();
    assert!(m.starts_with("Stopped to protect your Mac:") && m.contains("memory pressure is still critical"), "{m}");
    assert_eq!(j["result"]["watchdog"]["events"][0]["action"], "relieve");
}

#[test]
fn worker_crash_mid_job_fails_with_log_tail() {
    let d = Daemon::start(json!({}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 5, "step_s": 0.02, "crash_at": 2, "stderr_lines": 3}}));
    let id = j["id"].as_str().unwrap();
    let j = d.wait_state(id, &["failed", "done"], 10.0);
    assert_eq!(j["state"], "failed");
    let msg = j["message"].as_str().unwrap();
    assert!(msg.starts_with("the worker stopped unexpectedly (exit code 3)"), "{msg}");
    assert!(msg.contains("RuntimeError: fake crash"), "{msg}");
    let tail = j["result"]["log_tail"].as_str().unwrap();
    assert!(tail.contains("stderr line 0") && tail.contains("Traceback"), "{tail}");
    assert_eq!(d.get("/v1/system")["gpu_lock"]["ours"], false);
    let n = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0}}));
    d.wait_state(n["id"].as_str().unwrap(), &["done"], 15.0);
}

#[test]
fn worker_reported_failure_and_estimate_error() {
    let d = Daemon::start(json!({}));
    let j = d.submit(json!({"model": "dummy", "params": {"fail": "CUDA? no: boom", "stderr_lines": 2}}));
    let j = d.wait_state(j["id"].as_str().unwrap(), &["failed"], 10.0);
    assert_eq!(j["message"], "CUDA? no: boom");
    assert!(j["result"]["log_tail"].as_str().unwrap().contains("stderr line 1"));
    let j = d.submit(json!({"model": "dummy", "params": {"estimate_error": "no weights"}}));
    let j = d.wait_state(j["id"].as_str().unwrap(), &["failed"], 10.0);
    assert_eq!(j["message"], "estimate failed: no weights");
}

#[test]
fn memory_refused_mid_run_goes_back_to_waiting_then_fails() {
    let d = Daemon::start(json!({}));
    let j = d.submit(json!({"model": "dummy", "params": {"memory_refused": "DiT needs 30 GB"}}));
    let id = j["id"].as_str().unwrap();
    let j = d.wait_state(id, &["failed"], 15.0);
    assert_eq!(j["message"], "memory refused 3 times mid-run: DiT needs 30 GB");
    let (evs, _) = d.sse("?recent=200", 1000, 0.5);
    let n = evs.iter().filter(|e| e["job_id"] == id && e["kind"] == "job.waiting_memory").count();
    assert!(n >= 1, "went back to waiting_memory first");
}

#[test]
fn worker_start_failure_is_the_jobs_error() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let w = fake_worker();
    let d = Daemon::start_in(&home, json!({"worker_cmd": ["env", "FAKE_WORKER_START=fail", "python3", w]}));
    let j = d.submit(json!({"model": "dummy"}));
    let j = d.wait_state(j["id"].as_str().unwrap(), &["failed"], 15.0);
    let m = j["message"].as_str().unwrap();
    assert_eq!(
        m,
        "The model worker could not start: ImportError: no mlx. Set worker_cmd in config.json to a Python that has numpy and mlx."
    );
    let tail = j["result"]["log_tail"].as_str().unwrap();
    assert!(tail.contains("ImportError: no module named mlx") && tail.contains("exit code 1"), "{tail}");
    drop(d);
    let d = Daemon::start_in(&home, json!({"worker_cmd": ["/no/such/python"]}));
    let j = d.submit(json!({"model": "dummy"}));
    let j = d.wait_state(j["id"].as_str().unwrap(), &["failed"], 15.0);
    assert!(j["message"].as_str().unwrap().starts_with("could not start the worker: cannot run"), "{j}");
}

#[test]
fn worker_failing_at_import_says_so_plainly() {
    // A Python without numpy: the traceback's last line is the reason, the rest is the tail.
    let d = Daemon::start(json!({"worker_cmd": ["python3", "-c", "import shutter_no_such_module_numpy"]}));
    let j = d.submit(json!({"model": "dummy"}));
    let j = d.wait_state(j["id"].as_str().unwrap(), &["failed"], 15.0);
    let m = j["message"].as_str().unwrap();
    assert!(
        m.starts_with("The model worker could not start: ModuleNotFoundError: No module named 'shutter_no_such_module_numpy'. "),
        "{m}"
    );
    assert!(m.ends_with("Set worker_cmd in config.json to a Python that has numpy and mlx."), "{m}");
    let tail = j["result"]["log_tail"].as_str().unwrap();
    assert!(tail.contains("Traceback (most recent call last)") && tail.contains("exit code 1"), "{tail}");
    assert!(!d.worker_running());
}

#[test]
fn no_reserve_unless_set() {
    // 16 GB Mac, 4 GB free: no reserve, so a job that peaks at 4 GB runs.
    let d = Daemon::start(json!({"headroom_gb": null}));
    d.set_probe(json!({"total": 16 * GB, "available": 4 * GB, "swap_used": 0, "pressure": "normal"}));
    let s = d.get("/v1/system");
    assert_eq!(s["guard"]["headroom_bytes"], 0);
    assert_eq!(s["guard"]["headroom_auto"], true);
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0, "peak_gb": 4}}));
    d.wait_state(j["id"].as_str().unwrap(), &["done"], 15.0);
    drop(d);
    let d = Daemon::start(json!({"headroom_gb": 3}));
    d.set_probe(json!({"total": 16 * GB, "available": 4 * GB, "swap_used": 0, "pressure": "normal"}));
    let s = d.get("/v1/system");
    assert_eq!(s["guard"]["headroom_bytes"], 3 * GB);
    assert_eq!(s["guard"]["headroom_auto"], false);
}

#[test]
fn estimate_timeout_kills_worker() {
    let d = Daemon::start(json!({"estimate_timeout_s": 0.3}));
    let j = d.submit(json!({"model": "dummy", "params": {"estimate_hang": true}}));
    let j = d.wait_state(j["id"].as_str().unwrap(), &["failed"], 10.0);
    assert!(j["message"].as_str().unwrap().contains("did not answer the memory estimate within 0.3 s"));
    assert!(!d.worker_running());
}

#[test]
fn garbage_on_stdout_is_ignored() {
    let d = Daemon::start(json!({}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 3, "step_s": 0.01, "garbage": true}}));
    let j = d.wait_state(j["id"].as_str().unwrap(), &["done", "failed"], 10.0);
    assert_eq!(j["state"], "done", "{j}");
    assert!(d.worker_running());
}

#[test]
fn idle_unload_stops_the_worker() {
    let d = Daemon::start(json!({"idle_unload_s": 0.4}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0}}));
    d.wait_state(j["id"].as_str().unwrap(), &["done"], 10.0);
    d.wait_for("worker stopped", 10.0, |d| !d.worker_running());
    assert!(d.ops().contains(&"shutdown".to_string()), "{:?}", d.ops());
    let (evs, _) = d.sse("?recent=200", 1000, 0.3);
    assert!(evs.iter().any(|e| e["kind"] == "worker.stopped" && e["data"]["reason"] == "idle"));
    assert_eq!(d.get("/v1/system")["pool"]["entries"], json!([]));
}

#[test]
fn idle_unload_kills_a_worker_that_ignores_shutdown() {
    let w = fake_worker();
    let d = Daemon::start(json!({"idle_unload_s": 0.2, "shutdown_grace_s": 0.3,
                                 "worker_cmd": ["env", "FAKE_IGNORE_SHUTDOWN=1", "python3", w]}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0}}));
    d.wait_state(j["id"].as_str().unwrap(), &["done"], 10.0);
    let pid = d.get("/v1/system")["worker"]["pid"].as_i64();
    d.wait_for("worker stopped", 10.0, |d| !d.worker_running());
    if let Some(pid) = pid {
        // shutdown is ignored, so it takes shutdown_grace_s before the SIGKILL
        let t0 = Instant::now();
        while shutterd::util::pid_alive(pid as i32) == Some(true) && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(shutterd::util::pid_alive(pid as i32), Some(false));
        assert!(t0.elapsed() < Duration::from_secs(3));
    }
}

#[test]
fn restart_requeues_running_jobs() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let mut d = Daemon::start_in(&home, json!({}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 30, "step_s": 0.05}}));
    let id = j["id"].as_str().unwrap().to_string();
    d.wait_state(&id, &["running"], 10.0);
    let pid = d.get("/v1/system")["worker"]["pid"].as_i64().unwrap() as i32;
    d.kill9();
    // the orphaned worker sees EOF on stdin and exits
    let t0 = Instant::now();
    while shutterd::util::pid_alive(pid) == Some(true) && t0.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(shutterd::util::pid_alive(pid), Some(false), "worker exits when the daemon dies");
    let restart_at = shutterd::util::now();
    let d2 = Daemon::start_in(&home, json!({}));
    let j = d2.job(&id);
    assert!(j["state"] == "queued" || j["state"] == "running", "{j}");
    let j = d2.wait_state(&id, &["done"], 15.0);
    assert!(j["started_at"].as_f64().unwrap() >= restart_at);
}

#[test]
fn sigterm_requeues_the_running_job_and_stops_the_worker() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let mut d = Daemon::start_in(&home, json!({}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 200, "step_s": 0.05}}));
    let id = j["id"].as_str().unwrap().to_string();
    d.wait_state(&id, &["running"], 10.0);
    let pid = d.get("/v1/system")["worker"]["pid"].as_i64().unwrap() as i32;
    assert_eq!(d.stop(), Some(0));
    assert_eq!(shutterd::util::pid_alive(pid), Some(false));
    let s = shutterd::store::Store::open(&home.join("jobs.db").to_string_lossy()).unwrap();
    let j = s.get(&id).unwrap().unwrap();
    assert_eq!(j.state, "queued");
    assert_eq!(j.message, "interrupted by shutdown; it will run again");
    assert!(!home.join("gpu.lock").exists() || std::fs::read_to_string(home.join("gpu.lock")).unwrap().is_empty());
}

#[test]
fn dir_lock_held_by_someone_else_waits() {
    let tmp = tempfile::tempdir().unwrap();
    let lockdir = tmp.path().join("ImaginationStudio/queue/.render.lock");
    std::fs::create_dir_all(&lockdir).unwrap();
    std::fs::write(lockdir.join("owner_speed"), "studio fast\n").unwrap();
    std::fs::write(lockdir.join("pid"), format!("{}\n", std::process::id())).unwrap();
    let d = Daemon::start(json!({"gpu_lock_dirs": [lockdir.to_string_lossy()]}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0}}));
    let id = j["id"].as_str().unwrap();
    let w = d.wait_state(id, &["waiting_memory"], 10.0);
    assert_eq!(
        w["message"],
        format!("GPU busy: GPU lock {} is held by studio fast, pid {}", lockdir.display(), std::process::id())
    );
    let s = d.get("/v1/system");
    assert_eq!(s["gpu_lock"]["dir_holders"][0]["owner_speed"], "studio fast");
    assert_eq!(s["gpu_lock"]["ours"], false, "flock given back when the dirs are busy");
    std::thread::sleep(Duration::from_millis(200));
    assert!(lockdir.join("owner_speed").is_file(), "never removed");
    std::fs::remove_dir_all(&lockdir).unwrap();
    d.wait_state(id, &["done"], 10.0);
    assert!(!lockdir.exists(), "our own lock is removed after the job");
}

#[test]
fn flock_held_by_another_process_waits() {
    let d = Daemon::start(json!({}));
    let lock = d.home.join("gpu.lock");
    let script = format!(
        "import fcntl,json,time,sys\nf=open({:?},'w')\nfcntl.flock(f,fcntl.LOCK_EX)\nf.write(json.dumps({{'pid':1,'label':'other tool','time':0}}));f.flush()\nprint('ok',flush=True)\nsys.stdin.read()",
        lock.to_string_lossy()
    );
    let mut holder = std::process::Command::new("python3")
        .args(["-c", &script])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut ok = [0u8; 2];
    holder.stdout.as_mut().unwrap().read_exact(&mut ok).unwrap();
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0}}));
    let id = j["id"].as_str().unwrap();
    let w = d.wait_state(id, &["waiting_memory"], 10.0);
    assert_eq!(w["message"], "GPU busy: other tool");
    let s = d.get("/v1/system");
    assert_eq!(s["gpu_lock"]["held"], true);
    assert_eq!(s["gpu_lock"]["ours"], false);
    assert_eq!(s["gpu_lock"]["holder"]["label"], "other tool");
    drop(holder.stdin.take());
    holder.wait().unwrap();
    d.wait_state(id, &["done"], 10.0);
}

#[test]
fn busy_render_process_waits() {
    let mut sleeper = std::process::Command::new("sleep").arg("31415").spawn().unwrap();
    let d = Daemon::start(json!({"busy_process_pattern": "sleep 31415"}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0}}));
    let id = j["id"].as_str().unwrap();
    let w = d.wait_state(id, &["waiting_memory"], 10.0);
    assert_eq!(w["message"], "another render is running");
    assert!(!d.ops().contains(&"estimate".to_string()), "busy check comes before the worker is asked");
    sleeper.kill().unwrap();
    sleeper.wait().unwrap();
    d.wait_state(id, &["done"], 10.0);
}

// ---- API shapes mirrored from tests/test_api.py ----------------------------------------

#[test]
fn api_system_models_submit_list_pool() {
    let d = Daemon::start(json!({}));
    let s = d.get("/v1/system");
    for k in ["version", "memory", "guard", "pool", "queue", "gpu_lock"] {
        assert!(s.get(k).is_some(), "{k}");
    }
    for k in ["total", "available", "swap_used", "pressure"] {
        assert!(s["memory"].get(k).is_some(), "{k}");
    }
    assert_eq!(s["guard"]["headroom_bytes"], 0);
    assert_eq!(s["guard"]["swap_growth_bytes"], GB);
    assert_eq!(s["guard"]["protected"], json!([]));
    assert!(s["pool"].get("total_bytes").is_some() && s["pool"]["entries"].is_array());
    for k in ["queued", "running", "waiting_memory", "done", "failed", "cancelled", "paused"] {
        assert!(s["queue"].get(k).is_some(), "{k}");
    }
    assert_eq!(s["queue"]["paused"], false);
    assert!(s["gpu_lock"].get("held").is_some() && s["gpu_lock"].get("holder").is_some());

    let models = d.get("/v1/models")["models"].as_array().unwrap().clone();
    let dummy = models.iter().find(|m| m["id"] == "dummy").unwrap();
    assert_eq!(dummy["installed"], true);
    assert_eq!(dummy["family"], "dummy");

    let (st, job) = d.post("/v1/jobs", json!({"model": "dummy", "prompt": "x", "steps": 2, "params": {"step_s": 0}}));
    assert_eq!(st, 201, "{job}");
    // steps is a core request field now (it was folded into params before)
    assert_eq!(job["request"]["steps"], 2);
    assert!(job["request"]["params"].get("steps").is_none());
    assert_eq!(job["request"]["params"]["step_s"], 0);
    assert_eq!(job["request"]["seed"], 42);
    assert!(job["request"].get("framing").is_none(), "framing is never emitted");
    let keys: Vec<&str> = job["request"].as_object().unwrap().keys().map(String::as_str).collect();
    let mut want = vec!["model", "prompt", "negative_prompt", "width", "height", "frames", "duration_s", "fps",
                        "steps", "guidance", "seed", "inputs", "output", "priority", "params"];
    let mut got = keys.clone();
    got.sort();
    want.sort();
    assert_eq!(got, want);
    let id = job["id"].as_str().unwrap();
    assert_eq!(id.len(), 12);
    let j = d.wait_state(id, &["done"], 10.0);
    for k in ["output", "peak_bytes", "seconds"] {
        assert!(j["result"].get(k).is_some(), "{k}");
    }
    let lst = d.get("/v1/jobs?limit=5&state=done")["jobs"].as_array().unwrap().clone();
    assert_eq!(lst.iter().map(|x| x["id"].clone()).collect::<Vec<_>>(), vec![json!(id)]);
    assert_eq!(d.get("/v1/jobs?state=queued")["jobs"], json!([]));
    assert_eq!(d.get("/v1/jobs?state=done,failed")["jobs"].as_array().unwrap().len(), 1);
    let pool = d.get("/v1/system")["pool"].clone();
    assert_eq!(pool["entries"][0]["key"], "dummy/dit");
    assert_eq!(pool["entries"][0]["size_bytes"], GB / 2);
    assert!(pool["entries"][0].get("in_use").is_some());
    assert_eq!(pool["entries"][0]["residency"], "resident");
    assert_eq!(d.post("/v1/pool/release", json!({"key": null})).1, json!({"released": ["dummy/dit"]}));
    let (st, v) = d.req("POST", "/v1/pool/release", None);
    assert_eq!((st, v), (200, json!({"released": []})));
    assert_eq!(d.post("/v1/pool/release", json!({"key": 3})).0, 400);
    assert_eq!(d.req("POST", "/v1/pool/release", Some(b"nope")).0, 400);
    // the models list is enriched by the worker once it runs
    let models = d.get("/v1/models")["models"].as_array().unwrap().clone();
    assert_eq!(models[0]["kind"], "video");
}

#[test]
fn api_submit_400s() {
    let d = Daemon::start(json!({}));
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (b"not json".to_vec(), "JSON"),
        (b"[1]".to_vec(), "object"),
        (b"{}".to_vec(), "model"),
        (json!({"model": "nope"}).to_string().into_bytes(), "unknown model 'nope'; known models: dummy"),
        (json!({"model": "dummy", "framing": 3}).to_string().into_bytes(), "framing must be a string"),
        (json!({"model": "dummy", "width": "64"}).to_string().into_bytes(), "width must be an integer"),
        (json!({"model": "dummy", "steps": 0}).to_string().into_bytes(), "steps must be a positive integer"),
        (json!({"model": "dummy", "inputs": ["x"]}).to_string().into_bytes(), "inputs must be an object"),
        (json!({"model": "dummy", "inputs": {"image": 3}}).to_string().into_bytes(), "inputs.image must be"),
        (json!({"model": "dummy", "inputs": {"image": "/no/such.png"}}).to_string().into_bytes(), "image file not found"),
        (json!({"model": "dummy", "inputs": {"a b": "/x"}}).to_string().into_bytes(), "input names"),
        (json!({"model": "dummy", "seed": "x"}).to_string().into_bytes(), "seed"),
        (json!({"model": "dummy", "priority": true}).to_string().into_bytes(), "priority"),
        (json!({"model": "dummy", "params": []}).to_string().into_bytes(), "params"),
        (json!({"model": "dummy", "audio": "/no/such.wav"}).to_string().into_bytes(), "audio file not found: /no/such.wav"),
        (json!({"model": "dummy", "duration_s": -1}).to_string().into_bytes(), "duration_s must be >= 0"),
        (json!({"model": "dummy", "prompt": 3}).to_string().into_bytes(), "prompt must be a string"),
    ];
    for (body, needle) in cases {
        let (st, v) = d.req("POST", "/v1/jobs", Some(&body));
        assert_eq!(st, 400, "{}", String::from_utf8_lossy(&body));
        let e = v["error"].as_str().unwrap();
        assert!(e.contains(needle), "{e:?} lacks {needle:?}");
        assert_eq!(v["detail"], v["error"]);
    }
    // the old keys still work: framing goes into params (null: dropped), a top-level image into
    // inputs (made absolute); neither is emitted at the top level
    let img = d.home.join("start.png");
    std::fs::write(&img, b"png").unwrap();
    let (st, v) = d.post("/v1/jobs", json!({"model": "dummy", "framing": null, "image": img.to_string_lossy(), "params": {"steps": 1, "step_s": 0}}));
    assert_eq!(st, 201, "{v}");
    assert!(v["request"].get("framing").is_none() && v["request"]["params"].get("framing").is_none());
    assert!(v["request"].get("image").is_none());
    assert_eq!(v["request"]["inputs"]["image"], img.to_string_lossy().as_ref());
    let (st, v) = d.post("/v1/jobs", json!({"model": "dummy", "framing": "small", "params": {"steps": 1, "step_s": 0}}));
    assert_eq!(st, 201, "{v}");
    assert_eq!(v["request"]["params"]["framing"], "small");
    assert!(v["request"].get("framing").is_none());
}

#[test]
fn api_bad_query_and_404() {
    let d = Daemon::start(json!({}));
    assert_eq!(d.req("GET", "/v1/jobs?state=bogus", None).0, 400);
    assert_eq!(d.req("GET", "/v1/jobs?limit=x", None).0, 400);
    let (st, v) = d.req("GET", "/v1/jobs/nope", None);
    assert_eq!((st, v["error"].as_str().unwrap()), (404, "no job nope"));
    assert_eq!(d.req("DELETE", "/v1/jobs/nope", None).0, 404);
    assert_eq!(d.req("GET", "/v1/events?recent=x", None).0, 400);
    assert_eq!(d.req("GET", "/v1/nothing", None).0, 404);
}

#[test]
fn api_cancel_while_paused() {
    let d = Daemon::start(json!({}));
    assert_eq!(d.post("/v1/queue/pause", json!({})).1, json!({"paused": true}));
    let job = d.submit(json!({"model": "dummy"}));
    assert_eq!(d.get("/v1/system")["queue"]["paused"], true);
    let (_, r) = d.req("DELETE", &format!("/v1/jobs/{}", job["id"].as_str().unwrap()), None);
    assert_eq!(r["state"], "cancelled");
    assert_eq!(d.post("/v1/queue/resume", json!({})).1, json!({"paused": false}));
}

#[test]
fn api_sse_recent_first_live_and_heartbeat() {
    let d = Daemon::start(json!({}));
    let job = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0}}));
    let id = job["id"].as_str().unwrap();
    d.wait_state(id, &["done"], 10.0);
    let raw = raw_http(d.port, "GET /v1/events?max_events=3 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    assert!(raw.to_lowercase().contains("content-type: text/event-stream"), "{raw}");
    assert!(raw.contains(": shutter events"));
    let evs: Vec<Value> = raw
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    assert_eq!(evs.len(), 3, "the stream ends after max_events");
    assert_eq!(evs[0]["kind"], "job.queued");
    assert_eq!(evs[0]["job_id"], id);
    for k in ["kind", "job_id", "data", "ts"] {
        assert!(evs[0].get(k).is_some());
    }
    // live event + heartbeat
    let port = d.port;
    let h = std::thread::spawn(move || sse_collect(port, "?recent=0&max_events=1", 5.0, |e| !e.is_empty()));
    std::thread::sleep(Duration::from_millis(500));
    d.post("/v1/queue/pause", json!({}));
    let (evs, comments) = h.join().unwrap();
    assert_eq!(evs[0]["kind"], "queue.paused");
    assert!(comments.iter().any(|c| c == ": heartbeat"), "{comments:?}");
}

#[test]
fn api_dashboard_and_static_files() {
    let d = Daemon::start(json!({}));
    let r = d.raw("GET", "/");
    assert_eq!(r.status, 200);
    assert!(r.text().contains("<title>Shutter</title>"), "fallback page when missing");
    std::fs::create_dir_all(d.home.join("static/static/css")).unwrap();
    std::fs::write(d.home.join("static/dashboard.html"), "<title>Shutter</title><script>new EventSource('/v1/events')</script>").unwrap();
    std::fs::write(d.home.join("static/static/css/app.css"), "body{}").unwrap();
    std::fs::write(d.home.join("static/static/app.js"), "1").unwrap();
    std::fs::write(d.home.join("secret.txt"), "secret").unwrap();
    let r = d.raw("GET", "/");
    assert!(r.text().contains("EventSource"));
    assert!(r.header("content-type").unwrap().starts_with("text/html"));
    let r = d.raw("GET", "/static/css/app.css");
    assert_eq!((r.status, r.text().as_str()), (200, "body{}"));
    assert!(r.header("content-type").unwrap().starts_with("text/css"));
    assert!(d.raw("GET", "/static/app.js").header("content-type").unwrap().starts_with("text/javascript"));
    assert_eq!(d.raw("GET", "/static/nope.css").status, 404);
    for evil in ["/static/../../secret.txt", "/static/%2e%2e/%2e%2e/secret.txt", "/static/..%2f..%2fsecret.txt", "/static/css/../../../secret.txt"] {
        let raw = raw_http(d.port, &format!("GET {evil} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"));
        assert!(!raw.contains("secret\r\n") && !raw.ends_with("secret"), "{evil}: {raw}");
        assert!(raw.starts_with("HTTP/1.1 404") || raw.starts_with("HTTP/1.1 400"), "{evil}: {raw}");
    }
}

#[test]
fn api_key_required_for_non_loopback_clients() {
    // find a non-loopback address of this machine
    let ip = std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|s| s.connect("10.255.255.1:9").map(|_| s))
        .and_then(|s| s.local_addr())
        .map(|a| a.ip());
    let Ok(ip) = ip else { return };
    if ip.is_loopback() || ip.is_unspecified() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    // refused without a key
    std::fs::write(home.join("config.json"), "{}").unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_shutterd"))
        .args(["serve", "--host", "0.0.0.0", "--port", "0"])
        .env("SHUTTER_HOME", &home)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("refusing to listen on 0.0.0.0"));
    let mut cfg = base_config(&home);
    cfg["api_key"] = json!("s3cret");
    std::fs::write(home.join("config.json"), cfg.to_string()).unwrap();
    set_probe_in(&home, json!({}));
    let d = Daemon::spawn(&home, &["serve", "--host", "0.0.0.0", "--port", "0"]);
    let url = format!("http://{ip}:{}/v1/system", d.port);
    let t = Duration::from_secs(10);
    assert_eq!(shutterd::httpc::request("GET", &url, &[], None, t).unwrap().status, 401);
    assert_eq!(shutterd::httpc::request("GET", &url, &[("Authorization", "Bearer s3cret")], None, t).unwrap().status, 200);
    assert_eq!(shutterd::httpc::request("GET", &format!("{url}?key=s3cret"), &[], None, t).unwrap().status, 200);
    assert_eq!(shutterd::httpc::request("GET", &format!("{url}?key=wrong"), &[], None, t).unwrap().status, 401);
    // loopback needs no key
    assert_eq!(d.raw("GET", "/v1/system").status, 200);
}

#[test]
fn cli_talks_to_the_daemon() {
    let d = Daemon::start(json!({}));
    let cli = |args: &[&str]| {
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_shutterd"))
            .args(["--url", &format!("http://127.0.0.1:{}", d.port)])
            .args(args)
            .env("SHUTTER_HOME", &d.home)
            .output()
            .unwrap();
        (o.status.code(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
    };
    let (code, out, err) = cli(&["submit", "--model", "dummy", "--prompt", "hi", "--param", "steps=2", "--param", "step_s=0.01", "--wait", "--poll", "0.05"]);
    assert_eq!(code, Some(0), "{out}{err}");
    assert!(out.starts_with("submitted "), "{out}");
    assert!(out.contains("done in "), "{out}");
    let id = out.split_whitespace().nth(1).unwrap().to_string();
    let (code, out, _) = cli(&["status"]);
    assert_eq!(code, Some(0));
    assert!(out.contains(&id) && out.contains("done"), "{out}");
    let (_, out, _) = cli(&["--json", "status", &id]);
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["state"], "done");
    let (_, out, _) = cli(&["system"]);
    assert!(out.contains("memory") && out.contains("dummy/dit"), "{out}");
    let (_, out, _) = cli(&["models"]);
    assert!(out.contains("dummy") && out.contains("installed"), "{out}");
    let (_, out, _) = cli(&["unload"]);
    assert_eq!(out.trim(), "unloaded dummy/dit");
    let (_, out, _) = cli(&["pause"]);
    assert!(out.contains("paused"));
    let (_, out, _) = cli(&["--json", "resume"]);
    assert_eq!(serde_json::from_str::<Value>(&out).unwrap(), json!({"paused": false}));
    let (code, _, err) = cli(&["cancel", "nope"]);
    assert_eq!(code, Some(1));
    assert!(err.contains("HTTP 404: no job nope"), "{err}");
    let (code, out, _) = cli(&["submit", "--model", "dummy", "--param", "fail=\"bad\"", "--wait", "--poll", "0.05"]);
    assert_eq!(code, Some(1));
    assert!(out.contains("failed: bad"), "{out}");
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_shutterd"))
        .args(["--url", "http://127.0.0.1:1", "system"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("cannot reach the Shuttrr engine"));
}

#[test]
fn settings_get_post_persist_and_event() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let mut d = Daemon::start_in(&home, json!({}));
    let s = d.get("/v1/settings");
    assert_eq!(s["mode"], "fast");
    assert_eq!(s["efficiency"], 0.5);
    assert_eq!(s["rung"], "resident");
    let ladder = s["ladder"].as_array().unwrap();
    let ids: Vec<&str> = ladder.iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["resident", "split", "staged", "stream_wide", "stream_tight"]);
    assert_eq!(ladder.iter().map(|r| r["at"].as_f64().unwrap()).collect::<Vec<_>>(), [0.0, 0.25, 0.5, 0.75, 1.0]);
    assert!(ladder.iter().all(|r| r["label"].is_string() && r["detail"].is_string() && r.get("peak_bytes").is_none()));
    assert_eq!(s["measured_for"], Value::Null);
    assert_eq!(d.get("/v1/system")["settings"], json!({"mode": "fast", "efficiency": 0.5, "rung": "resident"}));

    // bad requests change nothing
    for (body, msg) in [
        (json!({"mode": "turbo"}), "mode must be one of: fast, low_memory"),
        (json!({"mode": 1}), "mode must be one of: fast, low_memory"),
        (json!({"efficiency": "0.5"}), "efficiency must be a number from 0 to 1"),
        (json!({"efficiency": true}), "efficiency must be a number from 0 to 1"),
        (json!([1]), "body must be a JSON object"),
    ] {
        let (st, v) = d.post("/v1/settings", body.clone());
        assert_eq!(st, 400, "{body}");
        assert!(v["error"].as_str().unwrap().starts_with(msg), "{v}");
    }
    let (st, _) = d.req("POST", "/v1/settings", Some(b"{nope"));
    assert_eq!(st, 400);

    let port = d.port;
    let h = std::thread::spawn(move || sse_collect(port, "?recent=0", 10.0, |evs| evs.iter().any(|e| e["kind"] == "settings.changed")));
    std::thread::sleep(Duration::from_millis(300));
    let (st, v) = d.post("/v1/settings", json!({"mode": "low_memory", "efficiency": 0.8}));
    assert_eq!(st, 200, "{v}");
    assert_eq!((v["mode"].as_str(), v["efficiency"].as_f64(), v["rung"].as_str()), (Some("low_memory"), Some(0.8), Some("stream_wide")));
    assert_eq!(v["ladder"].as_array().unwrap().len(), 5);
    let (evs, _) = h.join().unwrap();
    let ev = evs.iter().find(|e| e["kind"] == "settings.changed").expect("settings.changed");
    assert_eq!(ev["data"], json!({"mode": "low_memory", "efficiency": 0.8, "rung": "stream_wide"}));

    // clamped, and a partial update keeps the other field
    let (_, v) = d.post("/v1/settings", json!({"efficiency": 7}));
    assert_eq!((v["mode"].as_str(), v["efficiency"].as_f64()), (Some("low_memory"), Some(1.0)));
    let (_, v) = d.post("/v1/settings", json!({"efficiency": -2}));
    assert_eq!(v["efficiency"], 0.0);
    let (_, v) = d.post("/v1/settings", json!({"efficiency": 0.30000000000000004}));
    assert_eq!(v["efficiency"], 0.3);
    assert_eq!(v["rung"], "split");

    // persisted without dropping other keys, and read back after a restart
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(home.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["default_mode"], "low_memory");
    assert_eq!(cfg["default_efficiency"], 0.3);
    assert_eq!(cfg["some_future_key"], json!({"ignored": true}));
    assert_eq!(cfg["memory_poll_s"], 0.05);
    d.stop();
    let d = Daemon::spawn(&home, &["serve", "--port", "0"]);
    let s = d.get("/v1/settings");
    assert_eq!((s["mode"].as_str(), s["efficiency"].as_f64()), (Some("low_memory"), Some(0.3)));
    assert_eq!(d.get("/v1/system")["settings"]["rung"], "split");
}

#[test]
fn submit_copies_mode_defaults_into_params() {
    let d = Daemon::start(json!({"default_mode": "low_memory", "default_efficiency": 0.25}));
    d.post("/v1/queue/pause", json!({}));
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 1}}));
    assert_eq!(j["request"]["params"], json!({"steps": 1, "mode": "low_memory", "efficiency": 0.25}));
    let j = d.submit(json!({"model": "dummy", "params": {"mode": "fast"}}));
    assert_eq!(j["request"]["params"], json!({"mode": "fast", "efficiency": 0.25}), "the job's own mode wins");
    let j = d.submit(json!({"model": "dummy", "mode": "low-memory", "efficiency": 2}));
    assert_eq!(j["request"]["params"], json!({"mode": "low_memory", "efficiency": 1.0}));
    // a later change of the defaults does not touch queued jobs
    d.post("/v1/settings", json!({"mode": "fast", "efficiency": 0.9}));
    assert_eq!(d.job(j["id"].as_str().unwrap())["request"]["params"]["mode"], "low_memory");
    let j = d.submit(json!({"model": "dummy"}));
    assert_eq!(j["request"]["params"], json!({"mode": "fast", "efficiency": 0.9}));
    for body in [json!({"model": "dummy", "params": {"mode": "turbo"}}), json!({"model": "dummy", "params": {"efficiency": "hi"}})] {
        let (st, _) = d.post("/v1/jobs", body);
        assert_eq!(st, 400);
    }
}

#[test]
fn settings_ladder_uses_measured_json() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(home.join("measured.json"), "{broken").unwrap();
    let d = Daemon::start_in(&home, json!({}));
    assert_eq!(d.get("/v1/settings")["measured_for"], Value::Null);
    std::fs::write(
        home.join("measured.json"),
        json!({"longcat": {"704x1312": {"resident": {"peak_bytes": 40 * GB, "seconds": 100.0, "identical": true},
                                         "stream_tight": {"peak_bytes": 9 * GB, "seconds": 260.0, "identical": true}}}})
            .to_string(),
    )
    .unwrap();
    let s = d.get("/v1/settings");
    assert_eq!(s["measured_for"], json!({"model": "longcat", "size": "704x1312"}));
    assert_eq!(s["ladder"][0]["peak_bytes"], 40 * GB);
    assert_eq!(s["ladder"][4]["seconds"], 260.0);
    assert!(s["ladder"][2].get("peak_bytes").is_none());
}

#[test]
fn cli_settings_and_submit_mode() {
    let d = Daemon::start(json!({}));
    let cli = |args: &[&str]| {
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_shutterd"))
            .args(["--url", &format!("http://127.0.0.1:{}", d.port)])
            .args(args)
            .env("SHUTTER_HOME", &d.home)
            .output()
            .unwrap();
        (o.status.code(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
    };
    let (code, out, _) = cli(&["settings"]);
    assert_eq!(code, Some(0));
    assert!(out.contains("mode        fast") && out.contains("stream_tight") && out.contains("* 0.00  resident"), "{out}");
    let (code, out, err) = cli(&["settings", "--mode", "low_memory", "--efficiency", "0.5"]);
    assert_eq!(code, Some(0), "{err}");
    assert!(out.contains("mode        low memory") && out.contains("* 0.50  staged"), "{out}");
    let (_, out, _) = cli(&["--json", "settings", "--efficiency", "1"]);
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!((v["mode"].as_str(), v["rung"].as_str()), (Some("low_memory"), Some("stream_tight")));
    let (code, _, err) = cli(&["settings", "--mode", "turbo"]);
    assert_eq!(code, Some(1));
    assert!(err.contains("--mode must be fast or low_memory"), "{err}");
    d.post("/v1/queue/pause", json!({}));
    let (_, out, _) = cli(&["--json", "submit", "--model", "dummy", "--mode", "fast", "--efficiency", "0.25"]);
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["request"]["params"], json!({"mode": "fast", "efficiency": 0.25}));
}

// ---- native API additions: memory, files, output, clients, throughput ---------------------

fn multipart(boundary: &str, fields: &[(&str, &str)], files: &[(&str, &str, &str, &[u8])]) -> Vec<u8> {
    let mut b = Vec::new();
    for (k, v) in fields {
        b.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
    }
    for (field, name, ctype, data) in files {
        b.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n")
                .as_bytes(),
        );
        b.extend_from_slice(data);
        b.extend_from_slice(b"\r\n");
    }
    b.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    b
}

fn send(d: &Daemon, method: &str, path: &str, headers: &[(&str, &str)], body: Option<&[u8]>) -> shutterd::httpc::Response {
    shutterd::httpc::request(method, &d.url(path), headers, body, Duration::from_secs(30)).unwrap()
}

fn json_of(r: &shutterd::httpc::Response) -> Value {
    serde_json::from_slice(&r.body).unwrap_or(Value::String(r.text()))
}

#[test]
fn api_memory_reports_reserve_components_and_fit() {
    let d = Daemon::start(json!({"headroom_gb": 8}));
    let m = d.get("/v1/memory");
    for k in ["total", "available", "reserve", "free_for_models", "swap", "pressure", "held", "resident_bytes",
              "streamed_bytes", "components", "protected", "reserve_auto", "max_job_bytes"] {
        assert!(m.get(k).is_some(), "{k} missing: {m}");
    }
    assert_eq!(m["total"], 64 * GB);
    assert_eq!(m["available"], 48 * GB);
    assert_eq!(m["reserve"], 8 * GB);
    assert_eq!(m["free_for_models"], 40 * GB);
    assert_eq!(m["swap"], 0);
    assert_eq!(m["pressure"], "normal");
    assert_eq!(m["components"], json!([]));
    assert!(m.get("model").is_none());

    // a streamed component and a resident one (fake worker: params.residency)
    let a = d.submit(json!({"model": "dummy", "params": {"steps": 1, "step_s": 0, "residency": "streamed", "component_gb": 2}}));
    d.wait_state(a["id"].as_str().unwrap(), &["done"], 10.0);
    let m = d.get("/v1/memory");
    let c = &m["components"][0];
    assert_eq!(c["key"], "dummy/dit");
    assert_eq!(c["model"], "dummy");
    assert_eq!(c["residency"], "streamed");
    assert_eq!(d.get("/v1/system")["pool"]["entries"][0]["residency"], "streamed");
    assert_eq!(c["size_bytes"], 2 * GB);
    assert!(c.get("in_use").is_some() && c.get("idle_s").is_some());
    assert_eq!(m["streamed_bytes"], 2 * GB);
    assert_eq!(m["resident_bytes"], 0);
    assert_eq!(m["held"], 2 * GB);

    // fit for a model: need from the worker's estimate, warm bytes subtracted
    let m = d.get("/v1/memory?model=dummy&steps=3");
    assert_eq!(m["model"]["id"], "dummy");
    assert_eq!(m["model"]["need_bytes"], GB); // fake estimate: peak_gb 1.0
    assert_eq!(m["model"]["warm_bytes"], 2 * GB);
    assert_eq!(m["model"]["fits"], true);
    assert_eq!(m["model"]["short_by_bytes"], 0);
    d.post("/v1/pool/release", json!({"key": null}));
    d.set_probe(json!({"total": 64 * GB, "available": 8 * GB + GB / 2, "swap_used": 0, "pressure": "normal"}));
    let m = d.get("/v1/memory?model=dummy");
    assert_eq!(m["free_for_models"], GB / 2);
    assert_eq!(m["model"]["fits"], false);
    assert_eq!(m["model"]["short_by_bytes"], GB / 2);
    let m = d.get("/v1/memory?model=nope");
    assert!(m["model"]["error"].as_str().unwrap().contains("unknown model"), "{m}");
    assert_eq!(d.req("GET", "/v1/memory?width=x", None).0, 400);
}

#[test]
fn api_files_inputs_and_output_download() {
    let d = Daemon::start(json!({"max_upload_mb": 0.01}));
    // raw body upload
    let r = send(&d, "POST", "/v1/files?name=start.png", &[("Content-Type", "image/png")], Some(b"\x89PNGraw"));
    assert_eq!(r.status, 201, "{}", r.text());
    let up = json_of(&r);
    assert_eq!(up["name"], "start.png");
    assert_eq!(up["size"], 7);
    assert_eq!(up["content_type"], "image/png");
    assert_eq!(up["sha256"].as_str().unwrap().len(), 64);
    let path = up["path"].as_str().unwrap().to_string();
    assert!(path.contains("/work/uploads/"), "{path}");
    assert_eq!(std::fs::read(&path).unwrap(), b"\x89PNGraw");
    // multipart upload
    let body = multipart("bNd", &[("purpose", "input")], &[("file", "voice.wav", "audio/wav", b"RIFFxxxx")]);
    let r = send(&d, "POST", "/v1/files", &[("Content-Type", "multipart/form-data; boundary=bNd")], Some(&body));
    assert_eq!(r.status, 201, "{}", r.text());
    let up2 = json_of(&r);
    assert_eq!(up2["name"], "voice.wav");
    assert_eq!(up2["content_type"], "audio/wav");
    // errors
    assert_eq!(send(&d, "POST", "/v1/files", &[], Some(b"")).status, 400);
    let big = vec![7u8; 20 * 1024];
    assert_eq!(send(&d, "POST", "/v1/files?name=big.bin", &[], Some(&big)).status, 413);
    let body = multipart("bNd", &[("x", "y")], &[]);
    let r = send(&d, "POST", "/v1/files", &[("Content-Type", "multipart/form-data; boundary=bNd")], Some(&body));
    assert_eq!(r.status, 400);

    // inputs by upload id and by path
    let j = d.submit(json!({"model": "dummy", "inputs": {"image": up["id"], "audio": up2["path"]},
                            "params": {"steps": 1, "step_s": 0}}));
    assert_eq!(j["request"]["inputs"]["image"], path.as_str());
    assert_eq!(j["request"]["inputs"]["audio"], up2["path"]);
    let id = j["id"].as_str().unwrap();
    let done = d.wait_state(id, &["done"], 10.0);
    let r = send(&d, "GET", &format!("/v1/jobs/{id}/output"), &[], None);
    assert_eq!(r.status, 200);
    assert_eq!(r.header("content-type"), Some("video/mp4"));
    assert!(r.header("content-disposition").unwrap().contains(&format!("{id}.mp4")));
    assert_eq!(r.body, std::fs::read(done["result"]["output"].as_str().unwrap()).unwrap());

    // not finished yet: 409; unknown: 404; file gone: 404
    d.post("/v1/queue/pause", json!({}));
    let q = d.submit(json!({"model": "dummy"}));
    let r = send(&d, "GET", &format!("/v1/jobs/{}/output", q["id"].as_str().unwrap()), &[], None);
    assert_eq!(r.status, 409);
    assert!(json_of(&r)["error"].as_str().unwrap().contains("is queued"));
    assert_eq!(send(&d, "GET", "/v1/jobs/nope/output", &[], None).status, 404);
    std::fs::remove_file(done["result"]["output"].as_str().unwrap()).unwrap();
    let r = send(&d, "GET", &format!("/v1/jobs/{id}/output"), &[], None);
    assert_eq!(r.status, 404);
    assert!(json_of(&r)["error"].as_str().unwrap().contains("gone"));
}

#[test]
fn api_clients_and_throughput() {
    let d = Daemon::start(json!({}));
    let s = d.get("/v1/system");
    assert_eq!(s["throughput"], json!({"window_s": 3600, "finished": 0, "mean_seconds": null, "running": null}));
    assert_eq!(d.get("/v1/clients"), json!({"window_s": 600, "clients": []}));

    let a = d.submit(json!({"model": "dummy", "client": "comfyui/0.46", "params": {"steps": 2, "step_s": 0.05}}));
    assert_eq!(a["client"], "comfyui/0.46");
    d.wait_state(a["id"].as_str().unwrap(), &["done"], 10.0);
    // header form; a GET with the header only refreshes last_seen
    let r = send(&d, "POST", "/v1/jobs", &[("Content-Type", "application/json"), ("X-Shutter-Client", "my-script")],
                 Some(json!({"model": "dummy", "params": {"steps": 8, "step_s": 0.15}}).to_string().as_bytes()));
    assert_eq!(r.status, 201);
    let b = json_of(&r);
    assert_eq!(b["client"], "my-script");
    send(&d, "GET", "/v1/system", &[("X-Shutter-Client", "watcher/1")], None);

    // the running job's step rate
    let bid = b["id"].as_str().unwrap().to_string();
    d.wait_for("a step rate", 10.0, |d| {
        let t = &d.get("/v1/system")["throughput"];
        t["running"]["steps_per_s"].as_f64().map(|x| x > 0.0).unwrap_or(false)
    });
    let t = d.get("/v1/system")["throughput"].clone();
    assert_eq!(t["running"]["id"], bid.as_str());
    assert_eq!(t["running"]["steps"], 8);
    assert!(t["running"]["step"].as_u64().unwrap() >= 1);
    let rate = t["running"]["steps_per_s"].as_f64().unwrap();
    assert!(rate > 1.0 && rate < 20.0, "{rate}");
    d.wait_state(&bid, &["done"], 20.0);

    let t = d.get("/v1/system")["throughput"].clone();
    assert_eq!(t["finished"], 2);
    assert!(t["mean_seconds"].as_f64().unwrap() > 0.0);
    assert_eq!(t["running"], Value::Null);

    let c = d.get("/v1/clients");
    let rows = c["clients"].as_array().unwrap();
    let get = |n: &str| rows.iter().find(|r| r["client"] == n).cloned().unwrap_or_else(|| panic!("{n} in {c}"));
    assert_eq!(get("comfyui/0.46")["requests"], 1);
    assert_eq!(get("my-script")["requests"], 1);
    assert_eq!(get("watcher/1")["requests"], 0);
    for k in ["requests_total", "first_seen", "last_seen", "idle_s"] {
        assert!(get("my-script").get(k).is_some(), "{k}");
    }
    // stored with the job, so it survives in GET /v1/jobs
    assert_eq!(d.job(a["id"].as_str().unwrap())["client"], "comfyui/0.46");
}

// ---- OpenAI Images shim ------------------------------------------------------------------

#[test]
fn openai_images_generations_and_edits() {
    let d = Daemon::start(json!({"models": {"dummy-image": {"adapter": "dummy-image"}}}));
    let models = d.get("/v1/models")["models"].as_array().unwrap().clone();
    assert_eq!(models.iter().find(|m| m["id"] == "dummy-image").unwrap()["kind"], "image");

    let r = send(&d, "POST", "/v1/images/generations", &[("Content-Type", "application/json"), ("X-Shutter-Client", "open-webui")],
                 Some(json!({"model": "dummy-image", "prompt": "a red fox", "n": 2, "size": "64x48",
                             "params": {"steps": 1, "step_s": 0}}).to_string().as_bytes()));
    assert_eq!(r.status, 200, "{}", r.text());
    let v = json_of(&r);
    assert!(v["created"].as_i64().unwrap() > 0);
    assert_eq!(v["size"], "64x48");
    assert_eq!(v["output_format"], "png");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    let ids = v["shutter"]["jobs"].as_array().unwrap().clone();
    for (row, id) in data.iter().zip(&ids) {
        let job = d.job(id.as_str().unwrap());
        assert_eq!(job["request"]["width"], 64);
        assert_eq!(job["request"]["height"], 48);
        assert_eq!(job["client"], "open-webui");
        let out = job["result"]["output"].as_str().unwrap();
        assert!(out.ends_with(".png"), "{out}");
        let b = shutterd::files::b64_decode(row["b64_json"].as_str().unwrap()).unwrap();
        assert_eq!(b, std::fs::read(out).unwrap());
        assert_eq!(row["revised_prompt"], "a red fox");
    }
    let s0 = d.job(ids[0].as_str().unwrap())["request"]["seed"].as_i64().unwrap();
    assert_eq!(d.job(ids[1].as_str().unwrap())["request"]["seed"].as_i64().unwrap(), s0 + 1);

    // model defaults to the first image model; url response format
    let (st, v) = d.post("/v1/images/generations", json!({"prompt": "x", "response_format": "url", "seed": 7, "params": {"steps": 1, "step_s": 0}}));
    assert_eq!(st, 200, "{v}");
    let url = v["data"][0]["url"].as_str().unwrap();
    assert!(url.starts_with("http://127.0.0.1:") && url.ends_with("/output"), "{url}");
    let path = &url[url.find("/v1/").unwrap()..];
    assert_eq!(send(&d, "GET", path, &[], None).status, 200);

    // errors in OpenAI's shape
    for (body, status, needle) in [
        (json!({"model": "dummy", "prompt": "x"}), 400, "video model"),
        (json!({"model": "dummy-image"}), 400, "prompt is required"),
        (json!({"model": "nope", "prompt": "x"}), 404, "unknown model"),
        (json!({"model": "dummy-image", "prompt": "x", "n": 11}), 400, "n must be"),
        (json!({"model": "dummy-image", "prompt": "x", "size": "huge"}), 400, "size"),
        (json!({"model": "dummy-image", "prompt": "x", "output_format": "jpeg"}), 400, "png"),
        (json!({"model": "dummy-image", "prompt": "x", "stream": true}), 400, "stream"),
    ] {
        let (st, v) = d.post("/v1/images/generations", body.clone());
        assert_eq!(st, status, "{body} -> {v}");
        assert!(v["error"]["message"].as_str().unwrap().contains(needle), "{v}");
        assert!(v["error"].get("type").is_some() && v["error"].get("param").is_some() && v["error"].get("code").is_some());
    }
    // a failing job comes back as a server_error
    let (st, v) = d.post("/v1/images/generations", json!({"model": "dummy-image", "prompt": "x", "params": {"fail": "boom", "steps": 1}}));
    assert_eq!(st, 500, "{v}");
    assert_eq!(v["error"]["type"], "server_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("boom"));

    // edits: multipart with an image and a mask
    let body = multipart("e1", &[("prompt", "make it blue"), ("model", "dummy-image"), ("params", "{\"steps\": 1, \"step_s\": 0}")],
                         &[("image", "in.png", "image/png", b"\x89PNGin"), ("mask", "m.png", "image/png", b"\x89PNGmask")]);
    let r = send(&d, "POST", "/v1/images/edits", &[("Content-Type", "multipart/form-data; boundary=e1")], Some(&body));
    assert_eq!(r.status, 200, "{}", r.text());
    let v = json_of(&r);
    let job = d.job(v["shutter"]["jobs"][0].as_str().unwrap());
    let img = job["request"]["inputs"]["image"].as_str().unwrap();
    assert!(img.contains("/work/uploads/") && img.ends_with("in.png"), "{img}");
    assert_eq!(std::fs::read(img).unwrap(), b"\x89PNGin");
    assert!(job["request"]["inputs"]["mask"].as_str().unwrap().ends_with("m.png"));
    // edits: JSON with a data URL, and with an uploaded file id
    let (st, v) = d.post("/v1/images/edits", json!({"prompt": "p", "images": [{"image_url": "data:image/png;base64,iVBORw=="}],
                                                    "params": {"steps": 1, "step_s": 0}}));
    assert_eq!(st, 200, "{v}");
    let job = d.job(v["shutter"]["jobs"][0].as_str().unwrap());
    assert_eq!(std::fs::read(job["request"]["inputs"]["image"].as_str().unwrap()).unwrap(), vec![0x89, 0x50, 0x4e, 0x47]);
    let up = json_of(&send(&d, "POST", "/v1/files?name=a.png", &[], Some(b"img")));
    let (st, v) = d.post("/v1/images/edits", json!({"prompt": "p", "images": [{"file_id": up["id"]}], "params": {"steps": 1, "step_s": 0}}));
    assert_eq!(st, 200, "{v}");
    let job = d.job(v["shutter"]["jobs"][0].as_str().unwrap());
    assert_eq!(job["request"]["inputs"]["image"], up["path"]);
    let (st, v) = d.post("/v1/images/edits", json!({"prompt": "p"}));
    assert_eq!(st, 400);
    assert!(v["error"]["message"].as_str().unwrap().contains("image is required"));
    let (st, v) = d.post("/v1/images/edits", json!({"prompt": "p", "images": [{"image_url": "https://x/y.png"}]}));
    assert_eq!(st, 400);
    assert!(v["error"]["message"].as_str().unwrap().contains("data: URL"));
}
