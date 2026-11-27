//! `~/.shutter/config.json` (moved with SHUTTER_HOME). The SAME file and keys as the
//! Python reference (`shutter/core/config.py`); unknown keys are ignored. The daemon writes
//! the file only for POST /v1/settings (`default_mode`, `default_efficiency`), atomically
//! and keeping every other key, so nothing is lost. Keys added by the daemon: `worker_cmd`,
//! `worker_cwd`, `cancel_grace_s`, `shutdown_grace_s`, `watch_interval_s`,
//! `sample_interval_s`, `sse_heartbeat_s`, `worker_start_timeout_s`,
//! `estimate_timeout_s`, `omlx_timeout_s`, `static_dir`, and the test-only `test_probe_file`.

use crate::util::{expand_user, home_dir};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub home: PathBuf,
    pub path: PathBuf,
    pub host: String,
    pub port: u16,
    pub db_path: PathBuf,
    pub cache_dir: PathBuf,
    pub work_dir: PathBuf,
    pub output_dir: PathBuf,
    pub keep_work_dirs: bool,
    pub gpu_lock_path: PathBuf,
    pub gpu_lock_dirs: Vec<PathBuf>,
    pub busy_process_pattern: Option<String>,
    /// None = automatic: clamp(total / 8, 2 GB, 8 GB) of this Mac's memory.
    pub headroom_gb: Option<f64>,
    pub swap_growth_mb: f64,
    pub max_fraction: Option<f64>,
    pub memory_poll_s: f64,
    pub idle_unload_s: f64,
    pub omlx_url: Option<String>,
    pub omlx_api_key: Option<String>,
    pub omlx_timeout_s: f64,
    pub protected_process_pattern: Option<String>,
    pub protected_process_min_gb: f64,
    pub api_key: Option<String>,
    pub dummy_model: bool,
    pub models: Map<String, Value>,
    /// Engine-wide mode defaults (docs/MODES.md), changed through POST /v1/settings.
    pub default_mode: String,
    pub default_efficiency: f64,
    /// POST /v1/files and /v1/images/edits: the largest accepted file, MB.
    pub max_upload_mb: f64,
    /// /v1/images/*: how long a request waits for its jobs before they are cancelled.
    pub images_timeout_s: f64,
    // daemon-only keys
    pub worker_cmd: Vec<String>,
    pub worker_cwd: Option<PathBuf>,
    pub cancel_grace_s: f64,
    pub shutdown_grace_s: f64,
    pub watch_interval_s: f64,
    pub sample_interval_s: f64,
    pub sse_heartbeat_s: f64,
    pub worker_start_timeout_s: f64,
    pub estimate_timeout_s: f64,
    pub static_dir: Option<PathBuf>,
    pub test_probe_file: Option<PathBuf>,
    /// Problems found while reading the file (wrong types); defaults were used instead.
    pub warnings: Vec<String>,
}

pub fn default_home() -> PathBuf {
    match std::env::var("SHUTTER_HOME") {
        Ok(s) if !s.is_empty() => expand_user(&s),
        _ => home_dir().join(".shutter"),
    }
}

fn default_busy_pattern() -> Option<String> {
    Some("h3_generate.py|zig_fasth3|run_mps_avatar|sd-cli".into())
}

impl Config {
    /// Defaults for `home` (no file read).
    pub fn defaults(home: PathBuf) -> Config {
        Config {
            path: home.join("config.json"),
            host: "127.0.0.1".into(),
            port: 8484,
            db_path: home.join("jobs.db"),
            cache_dir: home.join("cache"),
            work_dir: home.join("work"),
            output_dir: home.join("outputs"),
            keep_work_dirs: false,
            gpu_lock_path: home.join("gpu.lock"),
            gpu_lock_dirs: Vec::new(),
            busy_process_pattern: default_busy_pattern(),
            headroom_gb: None,
            swap_growth_mb: 256.0,
            max_fraction: None,
            memory_poll_s: 15.0,
            idle_unload_s: 900.0,
            omlx_url: Some("http://127.0.0.1:8000".into()),
            omlx_api_key: None,
            omlx_timeout_s: 1.5,
            protected_process_pattern: Some("omlx|splash|strata".into()),
            protected_process_min_gb: 20.0,
            api_key: None,
            dummy_model: true,
            models: Map::new(),
            default_mode: crate::modes::DEFAULT_MODE.into(),
            default_efficiency: crate::modes::DEFAULT_EFFICIENCY,
            max_upload_mb: 2048.0,
            images_timeout_s: 3600.0,
            worker_cmd: vec!["python3".into(), "-m".into(), "shutter.worker".into()],
            worker_cwd: None,
            cancel_grace_s: 30.0,
            shutdown_grace_s: 5.0,
            watch_interval_s: 2.0,
            sample_interval_s: 5.0,
            sse_heartbeat_s: 15.0,
            worker_start_timeout_s: 120.0,
            estimate_timeout_s: 120.0,
            static_dir: None,
            test_probe_file: None,
            warnings: Vec::new(),
            home,
        }
    }

    /// Read config.json from SHUTTER_HOME (default ~/.shutter). Missing file = defaults.
    /// A file that is not a JSON object is an error (the Python reference refuses it too).
    pub fn load() -> Result<Config, String> {
        let home = default_home();
        let path = home.join("config.json");
        Self::load_from(&path, home)
    }

    pub fn load_from(path: &Path, home: PathBuf) -> Result<Config, String> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let data = if text.trim().is_empty() {
            Value::Object(Map::new())
        } else {
            serde_json::from_str::<Value>(&text)
                .map_err(|e| format!("{}: invalid JSON: {e}", path.display()))?
        };
        let Value::Object(map) = data else {
            return Err(format!("{}: expected a JSON object", path.display()));
        };
        let mut cfg = Self::from_map(&map, home);
        cfg.path = path.to_path_buf();
        Ok(cfg)
    }

    pub fn from_map(m: &Map<String, Value>, default_home: PathBuf) -> Config {
        let mut w = Vec::new();
        let home = match m.get("home").and_then(Value::as_str) {
            Some(h) if !h.is_empty() => expand_user(h),
            _ => default_home,
        };
        let mut c = Config::defaults(home.clone());
        let resolve = |p: &str| -> PathBuf {
            let p = expand_user(p);
            if p.is_absolute() {
                p
            } else {
                home.join(p)
            }
        };

        macro_rules! num {
            ($key:literal, $field:expr) => {
                match m.get($key) {
                    None | Some(Value::Null) => {}
                    Some(v) => match v.as_f64() {
                        Some(x) if x.is_finite() => $field = x,
                        _ => w.push(format!("{}: expected a number, using {}", $key, $field)),
                    },
                }
            };
        }
        macro_rules! opt_str {
            ($key:literal, $field:expr) => {
                match m.get($key) {
                    None => {}
                    Some(Value::Null) => $field = None,
                    Some(Value::String(s)) if s.is_empty() => $field = None,
                    Some(Value::String(s)) => $field = Some(s.clone()),
                    Some(_) => w.push(format!("{}: expected a string or null", $key)),
                }
            };
        }
        macro_rules! path {
            ($key:literal, $field:expr) => {
                match m.get($key) {
                    Some(Value::String(s)) if !s.is_empty() => $field = resolve(s),
                    None | Some(Value::Null) | Some(Value::String(_)) => {}
                    Some(_) => w.push(format!("{}: expected a path string", $key)),
                }
            };
        }
        macro_rules! opt_path {
            ($key:literal, $field:expr) => {
                match m.get($key) {
                    Some(Value::String(s)) if !s.is_empty() => $field = Some(resolve(s)),
                    None | Some(Value::Null) | Some(Value::String(_)) => {}
                    Some(_) => w.push(format!("{}: expected a path string", $key)),
                }
            };
        }
        macro_rules! boolean {
            ($key:literal, $field:expr) => {
                match m.get($key) {
                    None | Some(Value::Null) => {}
                    Some(Value::Bool(b)) => $field = *b,
                    Some(_) => w.push(format!("{}: expected true or false", $key)),
                }
            };
        }

        match m.get("host") {
            Some(Value::String(s)) if !s.is_empty() => c.host = s.clone(),
            None | Some(Value::Null) => {}
            Some(_) => w.push("host: expected a string".into()),
        }
        match m.get("port") {
            None | Some(Value::Null) => {}
            Some(v) => match v.as_u64() {
                Some(p) if p <= 65535 => c.port = p as u16,
                _ => w.push("port: expected an integer 0..65535".into()),
            },
        }
        path!("db_path", c.db_path);
        path!("cache_dir", c.cache_dir);
        path!("work_dir", c.work_dir);
        path!("output_dir", c.output_dir);
        path!("gpu_lock_path", c.gpu_lock_path);
        boolean!("keep_work_dirs", c.keep_work_dirs);
        match m.get("gpu_lock_dirs") {
            None | Some(Value::Null) => {}
            Some(Value::Array(a)) => {
                c.gpu_lock_dirs = a
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(|s| crate::util::normalize(&resolve_abs(s)))
                    .collect()
            }
            Some(_) => w.push("gpu_lock_dirs: expected a list of paths".into()),
        }
        opt_str!("busy_process_pattern", c.busy_process_pattern);
        match m.get("headroom_gb") {
            None | Some(Value::Null) => {}
            Some(v) => match v.as_f64() {
                Some(x) if x.is_finite() && x >= 0.0 => c.headroom_gb = Some(x),
                _ => w.push("headroom_gb: expected a number of GB or null (automatic), using automatic".into()),
            },
        }
        num!("swap_growth_mb", c.swap_growth_mb);
        match m.get("max_fraction") {
            None | Some(Value::Null) => {}
            Some(v) => match v.as_f64() {
                Some(x) => c.max_fraction = Some(x),
                None => w.push("max_fraction: expected a number or null".into()),
            },
        }
        num!("memory_poll_s", c.memory_poll_s);
        num!("idle_unload_s", c.idle_unload_s);
        opt_str!("omlx_url", c.omlx_url);
        opt_str!("omlx_api_key", c.omlx_api_key);
        num!("omlx_timeout_s", c.omlx_timeout_s);
        opt_str!("protected_process_pattern", c.protected_process_pattern);
        num!("protected_process_min_gb", c.protected_process_min_gb);
        opt_str!("api_key", c.api_key);
        boolean!("dummy_model", c.dummy_model);
        match m.get("models") {
            None | Some(Value::Null) => {}
            Some(Value::Object(o)) => c.models = o.clone(),
            Some(_) => w.push("models: expected an object".into()),
        }
        match m.get("default_mode") {
            None | Some(Value::Null) => {}
            Some(Value::String(s)) if crate::modes::normalize_mode(s).is_some() => {
                c.default_mode = crate::modes::normalize_mode(s).unwrap_or("fast").into()
            }
            Some(_) => w.push("default_mode: expected \"fast\" or \"low_memory\", using fast".into()),
        }
        match m.get("default_efficiency") {
            None | Some(Value::Null) => {}
            Some(v) => match v.as_f64() {
                Some(x) if x.is_finite() => c.default_efficiency = crate::modes::clamp_efficiency(x),
                _ => w.push(format!("default_efficiency: expected a number from 0 to 1, using {}", c.default_efficiency)),
            },
        }
        match m.get("worker_cmd") {
            None | Some(Value::Null) => {}
            Some(Value::Array(a)) if !a.is_empty() && a.iter().all(Value::is_string) => {
                c.worker_cmd = a.iter().filter_map(Value::as_str).map(String::from).collect()
            }
            Some(Value::String(s)) if !s.trim().is_empty() => {
                c.worker_cmd = s.split_whitespace().map(String::from).collect()
            }
            Some(_) => w.push("worker_cmd: expected a non-empty list of strings".into()),
        }
        num!("max_upload_mb", c.max_upload_mb);
        num!("images_timeout_s", c.images_timeout_s);
        opt_path!("worker_cwd", c.worker_cwd);
        num!("cancel_grace_s", c.cancel_grace_s);
        num!("shutdown_grace_s", c.shutdown_grace_s);
        num!("watch_interval_s", c.watch_interval_s);
        num!("sample_interval_s", c.sample_interval_s);
        num!("sse_heartbeat_s", c.sse_heartbeat_s);
        num!("worker_start_timeout_s", c.worker_start_timeout_s);
        num!("estimate_timeout_s", c.estimate_timeout_s);
        opt_path!("static_dir", c.static_dir);
        opt_path!("test_probe_file", c.test_probe_file);
        c.warnings = w;
        c
    }

    /// The explicit headroom, or None when it is automatic (see `auto_headroom`).
    pub fn headroom_bytes(&self) -> Option<u64> {
        self.headroom_gb.map(|g| (g.max(0.0) * crate::util::GB) as u64)
    }

    pub fn max_upload_bytes(&self) -> usize {
        (self.max_upload_mb.max(0.0) * crate::util::MIB) as usize
    }

    pub fn uploads_dir(&self) -> PathBuf {
        self.work_dir.join("uploads")
    }

    pub fn swap_growth_bytes(&self) -> u64 {
        // 1024 was the default 0.2.3 - 0.2.14 wrote into config.json; 0.3.0 reads it as the new 256 MB default
        let mb = if self.swap_growth_mb == 1024.0 { 256.0 } else { self.swap_growth_mb };
        (mb.max(0.0) * crate::util::MIB) as u64
    }

    pub fn protected_process_min_bytes(&self) -> u64 {
        (self.protected_process_min_gb.max(0.0) * crate::util::GB) as u64
    }

    /// The model's directory: `models.<id>.dir`, default `<home>/models/<id>`.
    pub fn model_dir(&self, model_id: &str) -> PathBuf {
        let d = self
            .models
            .get(model_id)
            .and_then(|s| s.get("dir"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .unwrap_or_else(|| format!("models/{model_id}"));
        let p = expand_user(&d);
        if p.is_absolute() {
            p
        } else {
            self.home.join(p)
        }
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        for d in [&self.home, &self.cache_dir, &self.work_dir, &self.output_dir] {
            std::fs::create_dir_all(d)?;
        }
        for f in [&self.db_path, &self.gpu_lock_path] {
            if let Some(p) = f.parent() {
                std::fs::create_dir_all(p)?;
            }
        }
        Ok(())
    }
}

/// gpu_lock_dirs are abspath'd against the current dir in Python (`os.path.abspath`).
fn resolve_abs(s: &str) -> PathBuf {
    crate::util::abspath(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_and_overrides() {
        let m = json!({
            "port": 9000, "headroom_gb": 4, "busy_process_pattern": null,
            "db_path": "x/jobs.db", "worker_cmd": ["python3", "w.py"], "unknown_key": {"a": 1},
            "gpu_lock_dirs": ["/tmp/a.lock.d"], "max_fraction": 0.5, "cancel_grace_s": 0.5,
            "idle_unload_s": "bad"
        });
        let c = Config::from_map(m.as_object().unwrap(), PathBuf::from("/h"));
        assert_eq!(c.port, 9000);
        assert_eq!(c.headroom_bytes(), Some(4 * 1024 * 1024 * 1024));
        assert!(c.busy_process_pattern.is_none());
        assert_eq!(c.db_path, PathBuf::from("/h/x/jobs.db"));
        assert_eq!(c.gpu_lock_path, PathBuf::from("/h/gpu.lock"));
        assert_eq!(c.worker_cmd, vec!["python3", "w.py"]);
        assert_eq!(c.gpu_lock_dirs, vec![PathBuf::from("/tmp/a.lock.d")]);
        assert_eq!(c.max_fraction, Some(0.5));
        assert_eq!(c.cancel_grace_s, 0.5);
        assert_eq!(c.idle_unload_s, 900.0);
        assert_eq!(c.warnings.len(), 1);
        assert_eq!(c.model_dir("m"), PathBuf::from("/h/models/m"));
        assert_eq!(c.swap_growth_bytes(), 256 * 1024 * 1024);
        let auto = Config::from_map(json!({"headroom_gb": null}).as_object().unwrap(), PathBuf::from("/h"));
        assert_eq!(auto.headroom_bytes(), None);
        assert!(auto.warnings.is_empty());
        assert_eq!(Config::defaults(PathBuf::from("/h")).headroom_bytes(), None);
        let m = Config::from_map(json!({"default_mode": "low_memory", "default_efficiency": 3}).as_object().unwrap(), PathBuf::from("/h"));
        assert_eq!((m.default_mode.as_str(), m.default_efficiency), ("low_memory", 1.0));
        let bad = Config::from_map(json!({"default_mode": "turbo", "default_efficiency": "x"}).as_object().unwrap(), PathBuf::from("/h"));
        assert_eq!((bad.default_mode.as_str(), bad.default_efficiency, bad.warnings.len()), ("fast", 0.5, 2));
    }

    #[test]
    fn load_rejects_non_object() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("config.json");
        std::fs::write(&p, "[1]").unwrap();
        assert!(Config::load_from(&p, d.path().into()).is_err());
        std::fs::write(&p, "").unwrap();
        assert!(Config::load_from(&p, d.path().into()).is_ok());
    }
}
