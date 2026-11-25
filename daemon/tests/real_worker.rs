//! End to end with the REAL Python worker (`python3 -m shutter.worker`, built-in `dummy`
//! model). Skipped when shutter/worker.py is missing or its imports (numpy) are not
//! available.

mod common;
use common::*;
use serde_json::json;

fn real_worker_available() -> bool {
    let root = repo_root();
    if !root.join("shutter/worker.py").is_file() {
        eprintln!("skipped: shutter/worker.py not found");
        return false;
    }
    let ok = std::process::Command::new("python3")
        .args(["-c", "import shutter.worker, numpy"])
        .current_dir(&root)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipped: python3 cannot import shutter.worker");
    }
    ok
}

#[test]
fn real_worker_runs_dummy_model() {
    if !real_worker_available() {
        return;
    }
    let root = repo_root();
    let d = Daemon::start(json!({
        "worker_cmd": ["python3", "-m", "shutter.worker"],
        "worker_cwd": root.to_string_lossy(),
        "headroom_gb": 0.5,
        "worker_start_timeout_s": 60,
        "estimate_timeout_s": 60,
    }));
    let j = d.submit(json!({"model": "dummy", "prompt": "a cat", "params": {"steps": 3, "step_s": 0.05}}));
    let id = j["id"].as_str().unwrap().to_string();
    let j = d.wait_state(&id, &["done", "failed", "cancelled"], 60.0);
    assert_eq!(j["state"], "done", "{j}");
    let out = j["result"]["output"].as_str().unwrap();
    assert!(out.ends_with(&format!("outputs/{id}.mp4")), "{out}");
    let written: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out).unwrap()).unwrap();
    assert_eq!(written["prompt"], "a cat");
    assert_eq!(written["steps"], 3);
    assert!(j["result"]["peak_bytes"].as_u64().unwrap() > 0);
    assert_eq!(j["progress"], 1.0);

    // warm pool reported by the worker, then released through the API
    let s = d.get("/v1/system");
    assert_eq!(s["worker"]["running"], true);
    assert_eq!(s["pool"]["entries"][0]["key"], "dummy/dit", "{}", s["pool"]);
    assert_eq!(d.post("/v1/pool/release", json!({"key": null})).1, json!({"released": ["dummy/dit"]}));
    let models = d.get("/v1/models");
    assert_eq!(models["models"][0]["id"], "dummy");
    assert_eq!(models["models"][0]["installed"], true);

    // cancel mid-run through the real worker
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 200, "step_s": 0.05}}));
    let id = j["id"].as_str().unwrap().to_string();
    d.wait_for("progress", 30.0, |d| d.job(&id)["progress"].as_f64().unwrap_or(0.0) > 0.0);
    d.req("DELETE", &format!("/v1/jobs/{id}"), None);
    let j = d.wait_state(&id, &["cancelled", "failed", "done"], 30.0);
    assert_eq!(j["state"], "cancelled", "{j}");

    // an adapter error surfaces as the job's message
    let j = d.submit(json!({"model": "dummy", "params": {"steps": 2, "step_s": 0, "fail": "boom"}}));
    let j = d.wait_state(j["id"].as_str().unwrap(), &["failed", "done"], 30.0);
    assert_eq!(j["state"], "failed");
    assert!(j["message"].as_str().unwrap().contains("boom"), "{j}");
}

#[test]
fn real_worker_image_model_memory_and_core_fields() {
    if !real_worker_available() {
        return;
    }
    let root = repo_root();
    let d = Daemon::start(json!({
        "worker_cmd": ["python3", "-m", "shutter.worker"],
        "worker_cwd": root.to_string_lossy(),
        "headroom_gb": 0.5,
        "worker_start_timeout_s": 60,
        "estimate_timeout_s": 60,
        "models": {"dummy-image": {"adapter": "dummy-image"}},
    }));
    // top-level steps reach the adapter (mirrored into params by Adapter.prepare)
    let j = d.submit(json!({"model": "dummy", "steps": 2, "params": {"step_s": 0.01, "mode": "low_memory", "efficiency": 1.0}}));
    let j = d.wait_state(j["id"].as_str().unwrap(), &["done", "failed"], 60.0);
    assert_eq!(j["state"], "done", "{j}");
    assert_eq!(j["result"]["steps"], 2);
    // the stream rung's component is reported as streamed
    let m = d.get("/v1/memory");
    let c = m["components"].as_array().unwrap().iter().find(|c| c["key"] == "dummy/dit").cloned().unwrap();
    assert_eq!(c["residency"], "streamed", "{m}");
    assert_eq!(c["model"], "dummy");
    let m = d.get("/v1/memory?model=dummy");
    assert!(m["model"]["need_bytes"].as_u64().unwrap() > 0, "{m}");
    assert_eq!(m["model"]["fits"], true);

    // the image model through the OpenAI shim: a real PNG of the asked size
    let models = d.get("/v1/models")["models"].as_array().unwrap().clone();
    assert_eq!(models.iter().find(|m| m["id"] == "dummy-image").unwrap()["kind"], "image");
    let (st, v) = d.post("/v1/images/generations", json!({"prompt": "a fox", "size": "40x24", "params": {"steps": 1, "step_s": 0}}));
    assert_eq!(st, 200, "{v}");
    let png = shutterd::files::b64_decode(v["data"][0]["b64_json"].as_str().unwrap()).unwrap();
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(u32::from_be_bytes([png[16], png[17], png[18], png[19]]), 40);
    assert_eq!(u32::from_be_bytes([png[20], png[21], png[22], png[23]]), 24);
    let (st, v) = d.post("/v1/images/generations", json!({"model": "dummy", "prompt": "x"}));
    assert_eq!(st, 400, "{v}");
}
