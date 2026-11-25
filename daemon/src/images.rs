//! OpenAI Images compatible shim: POST /v1/images/generations and /v1/images/edits. Same
//! shapes and rules as `shutter/server/openai_images.py`: each image is one native job (seed,
//! seed + 1, ...), image-kind models only, the request waits for its jobs, errors use OpenAI's
//! `{"error": {"message", "type", "param", "code"}}`. If the request future is dropped (the
//! client went away) its unfinished jobs are cancelled.

use crate::engine::Engine;
use crate::files::{b64_encode, decode_data_url, parse_multipart, Part};
use crate::stats::client_label;
use crate::store::is_terminal;
use crate::util::now;
use serde_json::{json, Map, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const MAX_N: i64 = 10;
const DEFAULT_CLIENT: &str = "openai-images";

#[derive(Debug)]
pub struct ShimError {
    pub status: u16,
    pub message: String,
    pub param: Option<&'static str>,
    pub code: Option<String>,
    pub kind: &'static str,
}

impl ShimError {
    fn bad(message: impl Into<String>, param: Option<&'static str>) -> ShimError {
        ShimError { status: 400, message: message.into(), param, code: None, kind: "invalid_request_error" }
    }
    fn server(status: u16, message: impl Into<String>, code: &str) -> ShimError {
        ShimError { status, message: message.into(), param: None, code: Some(code.into()), kind: "server_error" }
    }
    pub fn body(&self) -> Value {
        json!({"error": {"message": self.message, "type": self.kind, "param": self.param, "code": self.code}})
    }
}

/// "WxH" -> Some((W, H)); null / "" / "auto" -> None.
pub fn parse_size(v: Option<&Value>) -> Result<Option<(i64, i64)>, ShimError> {
    let s = match v {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(s)) if s.is_empty() || s == "auto" => return Ok(None),
        Some(Value::String(s)) => s.trim().to_string(),
        Some(other) => other.to_string(),
    };
    let err = || ShimError::bad(format!("size must be \"WIDTHxHEIGHT\" (e.g. 1024x1024) or \"auto\", got {s:?}"), Some("size"));
    let (a, b) = s.split_once(['x', 'X']).ok_or_else(err)?;
    let (w, h) = (a.trim().parse::<i64>().map_err(|_| err())?, b.trim().parse::<i64>().map_err(|_| err())?);
    if w <= 0 || h <= 0 || w > 99999 || h > 99999 {
        return Err(err());
    }
    Ok(Some((w, h)))
}

fn int_of(v: &Value, name: &'static str, lo: i64, hi: Option<i64>) -> Result<i64, ShimError> {
    let n = match v {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    };
    match n {
        Some(n) if n >= lo && hi.map(|h| n <= h).unwrap_or(true) => Ok(n),
        _ => Err(ShimError::bad(
            match hi {
                Some(h) => format!("{name} must be an integer {lo}..{h}"),
                None => format!("{name} must be an integer >= {lo}"),
            },
            Some(name),
        )),
    }
}

fn num_of(v: &Value, name: &'static str) -> Result<f64, ShimError> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
    .ok_or_else(|| ShimError::bad(format!("{name} must be a number"), Some(name)))
}

fn present(v: Option<&Value>) -> Option<&Value> {
    v.filter(|x| !x.is_null() && x.as_str() != Some(""))
}

struct Opts {
    tmpl: Map<String, Value>,
    n: i64,
    seed: i64,
    url: bool,
    size: Option<String>,
    prompt: String,
    model: String,
}

async fn common(engine: &Engine, body: &Map<String, Value>) -> Result<Opts, ShimError> {
    let prompt = match body.get("prompt") {
        Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
        _ => return Err(ShimError::bad("prompt is required", Some("prompt"))),
    };
    if matches!(body.get("stream"), Some(Value::Bool(true))) || body.get("stream").and_then(Value::as_str) == Some("true") {
        return Err(ShimError::bad("stream is not supported by Shuttrr yet; send stream=false", Some("stream")));
    }
    let fmt = present(body.get("output_format")).and_then(Value::as_str).unwrap_or("png");
    if fmt != "png" {
        return Err(ShimError::bad(format!("output_format '{fmt}' is not supported; Shuttrr returns png"), Some("output_format")));
    }
    let rf = present(body.get("response_format")).and_then(Value::as_str).unwrap_or("b64_json");
    if rf != "b64_json" && rf != "url" {
        return Err(ShimError::bad("response_format must be b64_json or url", Some("response_format")));
    }
    let n = match body.get("n") {
        None | Some(Value::Null) => 1,
        Some(v) => int_of(v, "n", 1, Some(MAX_N))?,
    };
    let size = parse_size(body.get("size"))?;
    let kinds = engine.model_kinds().await;
    let image_models: Vec<String> = engine.known_models().into_iter().filter(|m| kinds.get(m).map(String::as_str) == Some("image")).collect();
    let model = match present(body.get("model")) {
        None => image_models
            .first()
            .cloned()
            .ok_or_else(|| ShimError::bad("no image model is configured (GET /v1/models lists kind \"image\")", Some("model")))?,
        Some(Value::String(m)) => m.clone(),
        Some(other) => other.to_string(),
    };
    let Some(kind) = kinds.get(&model) else {
        let list = if image_models.is_empty() { "none".to_string() } else { image_models.join(", ") };
        return Err(ShimError {
            status: 404,
            message: format!("unknown model '{model}'; image models: {list}"),
            param: Some("model"),
            code: Some("model_not_found".into()),
            kind: "invalid_request_error",
        });
    };
    if kind != "image" {
        let k = if kind.is_empty() { "unknown" } else { kind.as_str() };
        return Err(ShimError::bad(format!("model '{model}' is a {k} model; the Images API serves image models only"), Some("model")));
    }
    let params = match body.get("params") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(o)) => o.clone(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(o)) => o,
            _ => return Err(ShimError::bad("params must be a JSON object", Some("params"))),
        },
        Some(_) => return Err(ShimError::bad("params must be a JSON object", Some("params"))),
    };
    let mut tmpl = Map::new();
    tmpl.insert("model".into(), json!(model));
    tmpl.insert("prompt".into(), json!(prompt));
    tmpl.insert("params".into(), Value::Object(params));
    if let Some((w, h)) = size {
        tmpl.insert("width".into(), json!(w));
        tmpl.insert("height".into(), json!(h));
    }
    if let Some(v) = present(body.get("negative_prompt")) {
        tmpl.insert("negative_prompt".into(), json!(v.as_str().map(String::from).unwrap_or_else(|| v.to_string())));
    }
    if let Some(v) = present(body.get("steps")) {
        tmpl.insert("steps".into(), json!(int_of(v, "steps", 1, None)?));
    }
    if let Some(v) = present(body.get("guidance")) {
        tmpl.insert("guidance".into(), json!(num_of(v, "guidance")?));
    }
    let seed = match present(body.get("seed")) {
        Some(v) => int_of(v, "seed", 0, None)?,
        None => (now() * 1e6) as i64 % (1 << 31),
    };
    Ok(Opts { tmpl, n, seed, url: rf == "url", size: size.map(|(w, h)| format!("{w}x{h}")), prompt, model })
}

/// Cancels the jobs it still holds when dropped (the client went away mid-wait).
struct CancelOnDrop {
    engine: Arc<Engine>,
    ids: Vec<String>,
    armed: bool,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            for id in &self.ids {
                let _ = self.engine.cancel(id);
            }
        }
    }
}

async fn run(engine: Arc<Engine>, opts: Opts, inputs: Map<String, Value>, client: String, base: &str) -> Result<Value, ShimError> {
    let mut guard = CancelOnDrop { engine: engine.clone(), ids: Vec::new(), armed: true };
    for i in 0..opts.n {
        let mut d = opts.tmpl.clone();
        d.insert("seed".into(), json!(opts.seed + i));
        d.insert("inputs".into(), Value::Object(inputs.clone()));
        match engine.submit_value(Value::Object(d), Some(&client), None) {
            Ok(j) => guard.ids.push(j.id),
            Err(e) => return Err(ShimError::bad(e.1, None)),
        }
    }
    let deadline = Instant::now() + Duration::from_secs_f64(engine.cfg.images_timeout_s.max(0.0));
    let mut done = vec![false; guard.ids.len()];
    loop {
        for (k, id) in guard.ids.iter().enumerate() {
            if done[k] {
                continue;
            }
            if let Ok(Some(j)) = engine.get(id) {
                if is_terminal(&j.state) {
                    done[k] = true;
                    if j.state != "done" {
                        return Err(ShimError::server(500, format!("job {id} {}: {}", j.state, j.message), &format!("job_{}", j.state)));
                    }
                }
            }
        }
        if done.iter().all(|x| *x) {
            break;
        }
        if Instant::now() > deadline {
            return Err(ShimError::server(
                504,
                format!(
                    "the images were not ready within {} s (images_timeout_s); jobs {} were cancelled",
                    engine.cfg.images_timeout_s,
                    guard.ids.join(", ")
                ),
                "timeout",
            ));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    guard.armed = false;
    let mut data = Vec::new();
    for id in &guard.ids {
        if opts.url {
            data.push(json!({"url": format!("{base}/v1/jobs/{id}/output"), "revised_prompt": opts.prompt}));
            continue;
        }
        let p = engine.output_file(id).map_err(|e| ShimError::server(500, e.1, "output_missing"))?;
        let bytes = std::fs::read(&p).map_err(|e| ShimError::server(500, format!("cannot read {}: {e}", p.display()), "output_missing"))?;
        data.push(json!({"b64_json": b64_encode(&bytes), "revised_prompt": opts.prompt}));
    }
    let mut out = json!({"created": now() as i64, "data": data, "output_format": "png",
                         "shutter": {"model": opts.model, "jobs": guard.ids}});
    if let Some(s) = &opts.size {
        out["size"] = json!(s);
    }
    Ok(out)
}

/// Shared by both routes: the client label for the jobs.
pub fn shim_client(header: Option<&str>, ua: Option<&str>) -> String {
    client_label(None, header, ua).unwrap_or_else(|| DEFAULT_CLIENT.into())
}

pub async fn generations(engine: Arc<Engine>, body: &[u8], client: String, base: &str) -> Result<Value, ShimError> {
    let v = Engine::parse_body(body).map_err(|e| ShimError::bad(e.1, None))?;
    let Value::Object(m) = v else { return Err(ShimError::bad("body must be a JSON object", None)) };
    let opts = common(&engine, &m).await?;
    run(engine, opts, Map::new(), client, base).await
}

fn image_ref(v: &Value, param: &'static str) -> Result<Result<(Vec<u8>, Option<String>), String>, ShimError> {
    // Ok(Ok(bytes)) for a data URL, Ok(Err(file_id)) for an upload id
    let o = match v {
        Value::String(s) => {
            let mut m = Map::new();
            m.insert("image_url".into(), json!(s));
            m
        }
        Value::Object(o) => o.clone(),
        _ => return Err(ShimError::bad("images must be a list of {\"image_url\": data URL} or {\"file_id\"}", Some(param))),
    };
    if let Some(f) = o.get("file_id").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        return Ok(Err(f.to_string()));
    }
    let url = o.get("image_url").and_then(Value::as_str).unwrap_or("");
    decode_data_url(url)
        .map(Ok)
        .ok_or_else(|| ShimError::bad(format!("{param}: only data: URLs (base64) are accepted; Shuttrr does not fetch URLs"), Some(param)))
}

pub async fn edits(engine: Arc<Engine>, content_type: &str, body: &[u8], client: String, base: &str) -> Result<Value, ShimError> {
    let mut images: Vec<(Vec<u8>, Option<String>, Option<String>)> = Vec::new();
    let mut mask: Option<(Vec<u8>, Option<String>, Option<String>)> = None;
    let mut file_ids: Vec<String> = Vec::new();
    let mut mask_id: Option<String> = None;
    let fields: Map<String, Value>;
    if let Some(b) = crate::files::boundary(content_type) {
        let parts: Vec<Part> = parse_multipart(body, &b).map_err(|e| ShimError::bad(format!("bad multipart body: {e}"), None))?;
        let mut m = Map::new();
        for p in parts {
            match (p.filename.is_some(), p.name.as_str()) {
                (true, "image" | "image[]" | "images" | "images[]") => images.push((p.data, p.filename, p.content_type)),
                (true, "mask") => mask = Some((p.data, p.filename, p.content_type)),
                (false, _) => {
                    m.insert(p.name.clone(), json!(p.text()));
                }
                _ => {}
            }
        }
        fields = m;
    } else {
        let v = Engine::parse_body(body).map_err(|_| ShimError::bad("body must be JSON or multipart/form-data", None))?;
        let Value::Object(m) = v else { return Err(ShimError::bad("body must be a JSON object", None)) };
        let refs: Vec<Value> = match (m.get("images"), m.get("image")) {
            (Some(Value::Array(a)), _) => a.clone(),
            (Some(Value::Null) | None, Some(v)) if !v.is_null() => vec![v.clone()],
            (Some(Value::Null) | None, _) => vec![],
            _ => return Err(ShimError::bad("images must be a list", Some("images"))),
        };
        for r in &refs {
            match image_ref(r, "images")? {
                Ok((d, ct)) => images.push((d, None, ct)),
                Err(fid) => file_ids.push(fid),
            }
        }
        if let Some(mv) = m.get("mask").filter(|v| !v.is_null()) {
            match image_ref(mv, "mask")? {
                Ok((d, ct)) => mask = Some((d, None, ct)),
                Err(fid) => mask_id = Some(fid),
            }
        }
        fields = m;
    }
    let opts = common(&engine, &fields).await?;
    if images.is_empty() && file_ids.is_empty() {
        return Err(ShimError::bad("image is required (multipart file \"image\", or JSON images[])", Some("image")));
    }
    let mut inputs = Map::new();
    let mut n = 0usize;
    let role = |n: &mut usize| {
        *n += 1;
        if *n == 1 { "image".to_string() } else { format!("image_{n}") }
    };
    for (data, name, ct) in &images {
        let info = engine
            .save_upload(Some(name.as_deref().unwrap_or("image.png")), data, ct.as_deref())
            .map_err(|e| ShimError { status: e.0, message: e.1, param: Some("image"), code: None, kind: "invalid_request_error" })?;
        inputs.insert(role(&mut n), info["path"].clone());
    }
    if let Some((data, name, ct)) = &mask {
        let info = engine
            .save_upload(Some(name.as_deref().unwrap_or("mask.png")), data, ct.as_deref())
            .map_err(|e| ShimError { status: e.0, message: e.1, param: Some("mask"), code: None, kind: "invalid_request_error" })?;
        inputs.insert("mask".into(), info["path"].clone());
    }
    for fid in file_ids {
        inputs.insert(role(&mut n), json!(fid));
    }
    if let Some(fid) = mask_id {
        inputs.insert("mask".into(), json!(fid));
    }
    run(engine, opts, inputs, client, base).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(parse_size(Some(&json!("1024x768"))).unwrap(), Some((1024, 768)));
        assert_eq!(parse_size(Some(&json!("auto"))).unwrap(), None);
        assert_eq!(parse_size(None).unwrap(), None);
        assert!(parse_size(Some(&json!("big"))).is_err());
        assert!(parse_size(Some(&json!("0x5"))).is_err());
    }
}
