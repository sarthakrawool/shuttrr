//! Fast / Low memory modes (docs/MODES.md): the low-memory ladder, the engine-wide
//! defaults and the optional measured numbers per rung from `<home>/measured.json`.
//!
//! The ladder is a built-in copy of the table in docs/MODES.md and must stay in step with
//! `LADDER` in `shutter/core/modes.py` (the Python side owns what each rung does; the
//! daemon only shows the table and copies the defaults into a job's params).

use serde_json::{json, Map, Value};
use std::path::Path;

pub const MODES: [&str; 2] = ["fast", "low_memory"];
pub const DEFAULT_MODE: &str = "fast";
pub const DEFAULT_EFFICIENCY: f64 = 0.5;

pub struct Rung {
    pub at: f64,
    pub id: &'static str,
    pub label: &'static str,
    pub detail: &'static str,
}

pub const LADDER: [Rung; 5] = [
    Rung { at: 0.00, id: "resident", label: "Full speed", detail: "Same as Fast: the whole model stays in memory." },
    Rung { at: 0.25, id: "split", label: "Slightly lower memory", detail: "Each model block runs in 3 smaller pieces: about 2 GB less memory, about 2% slower." },
    Rung { at: 0.50, id: "staged", label: "Lower memory", detail: "Also works through the frames in 4 chunks, and frees the main model before the final decode." },
    Rung { at: 0.75, id: "stream_wide", label: "Very low memory", detail: "Reads the model from the SSD as it goes, keeping 8 blocks in memory at a time." },
    Rung { at: 1.00, id: "stream_tight", label: "Lowest memory", detail: "Reads the model from the SSD as it goes, keeping only 2 blocks in memory: lowest memory, slowest." },
];

/// Accepts "fast", "low_memory" and the spellings "low-memory" / "low memory".
pub fn normalize_mode(s: &str) -> Option<&'static str> {
    match s.trim().to_ascii_lowercase().replace(['-', ' '], "_").as_str() {
        "fast" => Some("fast"),
        "low_memory" | "lowmemory" | "low" => Some("low_memory"),
        _ => None,
    }
}

/// Efficiency clamped to 0..1 and rounded to 4 decimals (a value from a browser slider can
/// be 0.30000000000000004). NaN gives the default.
pub fn clamp_efficiency(x: f64) -> f64 {
    if !x.is_finite() {
        return DEFAULT_EFFICIENCY;
    }
    (x.clamp(0.0, 1.0) * 10_000.0).round() / 10_000.0
}

/// The highest rung whose `at` is <= `efficiency`.
pub fn rung_for(efficiency: f64) -> &'static Rung {
    let e = clamp_efficiency(efficiency);
    LADDER.iter().rev().find(|r| r.at <= e + 1e-9).unwrap_or(&LADDER[0])
}

/// The rung a mode runs at before any fitting: Fast plans the resident rung (and may step
/// down at plan time); Low memory takes the rung of its efficiency.
pub fn rung_id(mode: &str, efficiency: f64) -> &'static str {
    if mode == "low_memory" {
        rung_for(efficiency).id
    } else {
        LADDER[0].id
    }
}

/// Measured numbers from `measured.json` (`{model: {size: {rung: {peak_bytes, seconds,
/// identical}}}}`): the (model, size) with the most measured rungs, ties broken by name.
/// Missing file, bad JSON or a wrong shape give None.
pub fn read_measured(home: &Path) -> Option<(String, String, Map<String, Value>)> {
    let text = std::fs::read_to_string(home.join("measured.json")).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let mut best: Option<(usize, String, String, Map<String, Value>)> = None;
    let mut models: Vec<(&String, &Value)> = v.as_object()?.iter().collect();
    models.sort_by(|a, b| a.0.cmp(b.0));
    for (model, sizes) in models {
        let Some(sizes) = sizes.as_object() else { continue };
        let mut sizes: Vec<(&String, &Value)> = sizes.iter().collect();
        sizes.sort_by(|a, b| a.0.cmp(b.0));
        for (size, rungs) in sizes {
            let Some(rungs) = rungs.as_object() else { continue };
            let n = LADDER.iter().filter(|r| rungs.get(r.id).map(Value::is_object).unwrap_or(false)).count();
            if n > 0 && best.as_ref().map(|b| n > b.0).unwrap_or(true) {
                best = Some((n, model.clone(), size.clone(), rungs.clone()));
            }
        }
    }
    best.map(|(_, m, s, r)| (m, s, r))
}

/// The ladder as JSON, with `peak_bytes` / `seconds` / `identical` filled in from
/// measured.json where present. Returns the ladder and the (model, size) it was measured on.
pub fn ladder_json(home: &Path) -> (Value, Option<Value>) {
    let measured = read_measured(home);
    let rows: Vec<Value> = LADDER
        .iter()
        .map(|r| {
            let mut row = json!({"at": r.at, "id": r.id, "label": r.label, "detail": r.detail, "identical": true});
            if let Some((_, _, m)) = &measured {
                if let Some(e) = m.get(r.id) {
                    for k in ["peak_bytes", "seconds"] {
                        if let Some(x) = e.get(k).and_then(Value::as_f64).filter(|x| x.is_finite() && *x >= 0.0) {
                            row[k] = if k == "peak_bytes" { json!(x as u64) } else { json!(x) };
                        }
                    }
                    if let Some(b) = e.get("identical").and_then(Value::as_bool) {
                        row["identical"] = json!(b);
                    }
                }
            }
            row
        })
        .collect();
    let used = measured.map(|(m, s, _)| json!({"model": m, "size": s}));
    (Value::Array(rows), used)
}

/// Write `default_mode` / `default_efficiency` into config.json atomically, keeping every
/// other key (known or not) as it was. A config.json that is not a JSON object is left
/// alone and reported.
pub fn persist(path: &Path, mode: &str, efficiency: f64) -> Result<(), String> {
    let mut map = match std::fs::read_to_string(path) {
        Ok(t) if t.trim().is_empty() => Map::new(),
        Ok(t) => match serde_json::from_str::<Value>(&t) {
            Ok(Value::Object(m)) => m,
            Ok(_) => return Err(format!("{} is not a JSON object; not changing it", path.display())),
            Err(e) => return Err(format!("{} is not valid JSON ({e}); not changing it", path.display())),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Map::new(),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    map.insert("default_mode".into(), json!(mode));
    map.insert("default_efficiency".into(), json!(efficiency));
    let text = serde_json::to_string_pretty(&Value::Object(map)).map_err(|e| e.to_string())? + "\n";
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let tmp = path.with_extension(format!("json.tmp.{}", std::process::id()));
    std::fs::write(&tmp, text).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot replace {}: {e}", path.display())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rungs_and_modes() {
        assert_eq!(rung_for(0.0).id, "resident");
        assert_eq!(rung_for(0.24).id, "resident");
        assert_eq!(rung_for(0.25).id, "split");
        assert_eq!(rung_for(0.5).id, "staged");
        assert_eq!(rung_for(0.7499).id, "staged");
        assert_eq!(rung_for(0.75).id, "stream_wide");
        assert_eq!(rung_for(1.0).id, "stream_tight");
        assert_eq!(rung_for(7.0).id, "stream_tight");
        assert_eq!(rung_for(-1.0).id, "resident");
        assert_eq!(rung_id("fast", 1.0), "resident");
        assert_eq!(rung_id("low_memory", 0.6), "staged");
        assert_eq!(clamp_efficiency(0.30000000000000004), 0.3);
        assert_eq!(clamp_efficiency(f64::NAN), DEFAULT_EFFICIENCY);
        assert_eq!(normalize_mode("Low-Memory"), Some("low_memory"));
        assert_eq!(normalize_mode("fast"), Some("fast"));
        assert_eq!(normalize_mode("turbo"), None);
    }

    #[test]
    fn measured_enrichment_tolerates_bad_files() {
        let d = tempfile::tempdir().unwrap();
        let (l, used) = ladder_json(d.path());
        assert!(used.is_none() && l[0].get("peak_bytes").is_none());
        std::fs::write(d.path().join("measured.json"), "{not json").unwrap();
        assert!(ladder_json(d.path()).1.is_none());
        std::fs::write(d.path().join("measured.json"), "[1,2]").unwrap();
        assert!(ladder_json(d.path()).1.is_none());
        std::fs::write(
            d.path().join("measured.json"),
            r#"{"a": {"720": {"resident": {"peak_bytes": 100, "seconds": 10}}, "bad": 3},
                "b": {"480": {"resident": {"peak_bytes": 50, "seconds": 5.5, "identical": true},
                              "staged": {"peak_bytes": 20, "seconds": "x"}}}}"#,
        )
        .unwrap();
        let (l, used) = ladder_json(d.path());
        assert_eq!(used.unwrap(), json!({"model": "b", "size": "480"}));
        assert_eq!(l[0]["peak_bytes"], 50);
        assert_eq!(l[0]["seconds"], 5.5);
        assert_eq!(l[0]["identical"], true);
        assert_eq!(l[2]["peak_bytes"], 20);
        assert!(l[2].get("seconds").is_none(), "a non-number is dropped");
        assert!(l[1].get("peak_bytes").is_none());
    }

    #[test]
    fn persist_keeps_unknown_keys() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("config.json");
        std::fs::write(&p, r#"{"port": 9000, "future": {"x": [1, 2]}}"#).unwrap();
        persist(&p, "low_memory", 0.75).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(v, json!({"port": 9000, "future": {"x": [1, 2]}, "default_mode": "low_memory", "default_efficiency": 0.75}));
        std::fs::write(&p, "[1]").unwrap();
        assert!(persist(&p, "fast", 0.5).is_err());
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "[1]");
        let fresh = d.path().join("new/config.json");
        persist(&fresh, "fast", 0.5).unwrap();
        assert!(fresh.is_file());
        assert_eq!(std::fs::read_dir(d.path()).unwrap().count(), 2, "no temp file left behind");
    }
}
