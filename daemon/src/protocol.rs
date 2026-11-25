//! The daemon <-> worker JSON-lines protocol (docs/WORKER_PROTOCOL.md), version 1.

use crate::util::{jf64, jstr};
use serde_json::{json, Value};

pub const PROTOCOL: i64 = 1;

// ---- daemon -> worker ---------------------------------------------------------------

pub fn op_hello() -> Value {
    json!({"op": "hello"})
}
pub fn op_estimate(id: &str, request: &Value) -> Value {
    json!({"op": "estimate", "id": id, "request": request})
}
pub fn op_run(id: &str, job: &Value, work_dir: &str) -> Value {
    json!({"op": "run", "id": id, "job": job, "work_dir": work_dir})
}
pub fn op_cancel(id: &str) -> Value {
    json!({"op": "cancel", "id": id})
}
pub fn op_release(key: Option<&str>) -> Value {
    json!({"op": "release", "key": key})
}
pub fn op_stats() -> Value {
    json!({"op": "stats"})
}
pub fn op_models() -> Value {
    json!({"op": "models"})
}
pub fn op_shutdown() -> Value {
    json!({"op": "shutdown"})
}

/// One line, newline-terminated. serde_json never emits raw newlines inside a value.
pub fn encode(v: &Value) -> String {
    let mut s = v.to_string();
    s.push('\n');
    s
}

// ---- worker -> daemon ---------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Ready { version: String, pid: i64, protocol: i64 },
    /// The whole message; `error` set means the estimate failed.
    Estimate { id: String, msg: Value },
    Progress { id: String, progress: f64, stage: String },
    Done { id: String, result: Value },
    Failed { id: String, error: String, cancelled: bool, memory_refused: bool },
    Pool(Value),
    Released { keys: Vec<String>, deferred: Vec<String> },
    Models(Value),
    Log { level: String, message: String, id: Option<String> },
    /// A valid message with an unknown `ev` (ignored, forward compatibility).
    Unknown(String),
}

fn strings(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect())
        .unwrap_or_default()
}

fn id_of(v: &Value) -> String {
    match v.get("id") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

/// Decode one stdout line. Err for anything that is not a JSON object with an `ev`
/// string (the daemon logs it and carries on).
pub fn decode(line: &str) -> Result<Event, String> {
    let line = line.trim();
    let v: Value = serde_json::from_str(line).map_err(|e| format!("not JSON ({e})"))?;
    if !v.is_object() {
        return Err("not a JSON object".into());
    }
    let ev = jstr(&v, "ev").ok_or("no \"ev\" field")?;
    Ok(match ev {
        "ready" => Event::Ready {
            version: jstr(&v, "version").unwrap_or("").into(),
            pid: v.get("pid").and_then(Value::as_i64).unwrap_or(0),
            protocol: v.get("protocol").and_then(Value::as_i64).unwrap_or(PROTOCOL),
        },
        "estimate" => Event::Estimate { id: id_of(&v), msg: v.clone() },
        "progress" => Event::Progress {
            id: id_of(&v),
            progress: jf64(&v, "progress").filter(|p| p.is_finite()).unwrap_or(0.0).clamp(0.0, 1.0),
            stage: jstr(&v, "stage").unwrap_or("").into(),
        },
        "done" => Event::Done {
            id: id_of(&v),
            result: v.get("result").cloned().filter(Value::is_object).unwrap_or_else(|| json!({})),
        },
        "failed" => Event::Failed {
            id: id_of(&v),
            error: match v.get("error") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => "the worker reported a failure without a reason".into(),
                Some(other) => other.to_string(),
            },
            cancelled: v.get("cancelled").and_then(Value::as_bool).unwrap_or(false),
            memory_refused: v.get("memory_refused").and_then(Value::as_bool).unwrap_or(false),
        },
        "pool" => Event::Pool(v.clone()),
        "released" => Event::Released { keys: strings(v.get("keys")), deferred: strings(v.get("deferred")) },
        "models" => Event::Models(v.get("models").cloned().filter(Value::is_array).unwrap_or_else(|| json!([]))),
        "log" => Event::Log {
            level: jstr(&v, "level").unwrap_or("info").into(),
            message: match v.get("message") {
                Some(Value::String(s)) => s.clone(),
                Some(o) => o.to_string(),
                None => String::new(),
            },
            id: v.get("id").and_then(Value::as_str).map(String::from),
        },
        other => Event::Unknown(other.into()),
    })
}

/// A memory estimate in bytes from an `estimate` event (Err = the worker's error text).
pub fn estimate_peak(msg: &Value) -> Result<u64, String> {
    if let Some(e) = msg.get("error").filter(|e| !e.is_null()) {
        return Err(e.as_str().map(String::from).unwrap_or_else(|| e.to_string()));
    }
    match msg.get("peak").and_then(Value::as_f64) {
        Some(p) if p.is_finite() && p >= 0.0 => Ok(p as u64),
        _ => Err("the worker's estimate has no valid \"peak\"".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_ops() {
        assert_eq!(encode(&op_hello()), "{\"op\":\"hello\"}\n");
        let l = encode(&op_run("j1", &json!({"id": "j1", "request": {"prompt": "a\nb"}}), "/w/j1"));
        assert!(l.ends_with('\n') && l.matches('\n').count() == 1);
        let back: Value = serde_json::from_str(&l).unwrap();
        assert_eq!(back["op"], "run");
        assert_eq!(back["job"]["request"]["prompt"], "a\nb");
        assert_eq!(serde_json::from_str::<Value>(&encode(&op_release(None))).unwrap()["key"], Value::Null);
        assert_eq!(serde_json::from_str::<Value>(&encode(&op_release(Some("k")))).unwrap()["key"], "k");
        for v in [op_estimate("a", &json!({})), op_cancel("a"), op_stats(), op_models(), op_shutdown()] {
            assert!(v["op"].is_string());
        }
    }

    #[test]
    fn decode_events() {
        assert_eq!(
            decode(r#"{"ev":"ready","version":"0.1.0","pid":7,"protocol":1}"#).unwrap(),
            Event::Ready { version: "0.1.0".into(), pid: 7, protocol: 1 }
        );
        assert_eq!(
            decode(r#"{"ev":"progress","id":"a","progress":1.7,"stage":"x"}"#).unwrap(),
            Event::Progress { id: "a".into(), progress: 1.0, stage: "x".into() }
        );
        assert_eq!(
            decode(r#"{"ev":"failed","id":"a","error":"boom","cancelled":false,"memory_refused":true}"#).unwrap(),
            Event::Failed { id: "a".into(), error: "boom".into(), cancelled: false, memory_refused: true }
        );
        assert_eq!(
            decode(r#"{"ev":"released","keys":["a"],"deferred":["b"]}"#).unwrap(),
            Event::Released { keys: vec!["a".into()], deferred: vec!["b".into()] }
        );
        match decode(r#"{"ev":"done","id":"a","result":{"output":"/o"}}"#).unwrap() {
            Event::Done { result, .. } => assert_eq!(result["output"], "/o"),
            e => panic!("{e:?}"),
        }
        match decode(r#"{"ev":"estimate","id":"a","peak":1024}"#).unwrap() {
            Event::Estimate { msg, .. } => assert_eq!(estimate_peak(&msg), Ok(1024)),
            e => panic!("{e:?}"),
        }
        assert_eq!(estimate_peak(&json!({"error": "no adapter"})), Err("no adapter".into()));
        assert!(estimate_peak(&json!({"peak": -1})).is_err());
        assert_eq!(decode(r#"{"ev":"future_thing","x":1}"#).unwrap(), Event::Unknown("future_thing".into()));
        assert!(decode("hello from print()").is_err());
        assert!(decode("[1,2]").is_err());
        assert!(decode(r#"{"op":"x"}"#).is_err());
        assert!(matches!(decode(r#"{"ev":"log","level":"warning","message":"m","id":"j"}"#).unwrap(),
            Event::Log { id: Some(_), .. }));
    }
}
