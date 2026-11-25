//! Small helpers shared by every module: time, ids, number formatting, JSON access.

use serde_json::{Map, Value};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Log a line to stderr; never panics (a closed stderr must not kill the daemon).
#[macro_export]
macro_rules! log {
    ($($t:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), $($t)*);
    }};
}

pub const GB: f64 = 1024.0 * 1024.0 * 1024.0;
pub const MIB: f64 = 1024.0 * 1024.0;

/// Seconds since the Unix epoch, like Python's `time.time()`.
pub fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Bytes as a human GB figure with one decimal, e.g. "12.3 GB" (same as Python `fmt_gb`).
pub fn fmt_gb(n: f64) -> String {
    format!("{:.1} GB", n / GB)
}

/// 12 lowercase hex characters, like `uuid.uuid4().hex[:12]`.
pub fn new_id() -> String {
    let mut buf = [0u8; 6];
    let ok = std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut buf))
        .is_ok();
    if !ok {
        // Fallback: time and pid mixed; uniqueness is still enforced by the store.
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let x = (t as u64) ^ ((std::process::id() as u64) << 32) ^ (t >> 64) as u64;
        buf.copy_from_slice(&x.to_le_bytes()[..6]);
    }
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

/// `~` expansion using $HOME.
pub fn expand_user(p: &str) -> PathBuf {
    if p == "~" {
        return home_dir();
    }
    if let Some(rest) = p.strip_prefix("~/") {
        return home_dir().join(rest);
    }
    PathBuf::from(p)
}

pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// Absolute path (like `os.path.abspath(os.path.expanduser(p))`): lexical, no symlinks.
pub fn abspath(p: &str) -> PathBuf {
    let p = expand_user(p);
    let p = if p.is_absolute() {
        p
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")).join(p)
    };
    normalize(&p)
}

/// Lexically remove `.` and `..` components.
pub fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Python `repr()` of a str (good enough for the messages we mirror).
pub fn py_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(q);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

// ---- JSON access helpers ----------------------------------------------------------

pub fn jstr<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::as_str)
}

pub fn jf64(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}

pub fn ji64(v: &Value, k: &str) -> Option<i64> {
    v.get(k).and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|f| f as i64)))
}

pub fn obj() -> Map<String, Value> {
    Map::new()
}

/// Seconds as f64 to a std Duration, clamped to sane bounds (negative or NaN -> 0).
pub fn secs(s: f64) -> std::time::Duration {
    if !s.is_finite() || s <= 0.0 {
        std::time::Duration::from_millis(0)
    } else {
        std::time::Duration::from_secs_f64(s.min(10.0 * 365.0 * 86400.0))
    }
}

/// Is `pid` alive? Some(false) when it certainly is not.
pub fn pid_alive(pid: i32) -> Option<bool> {
    if pid <= 0 {
        return None;
    }
    // SAFETY: kill with signal 0 only checks for existence/permission.
    let r = unsafe { libc::kill(pid, 0) };
    if r == 0 {
        return Some(true);
    }
    match std::io::Error::last_os_error().raw_os_error() {
        Some(libc::ESRCH) => Some(false),
        Some(libc::EPERM) => Some(true),
        _ => None,
    }
}

/// Python-style round to `n` decimals for JSON output.
pub fn round_to(x: f64, n: i32) -> f64 {
    let m = 10f64.powi(n);
    (x * m).round() / m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_12_hex() {
        let a = new_id();
        let b = new_id();
        assert_eq!(a.len(), 12);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }

    #[test]
    fn gb_and_repr() {
        assert_eq!(fmt_gb(12.0 * GB), "12.0 GB");
        assert_eq!(fmt_gb(0.25 * GB), "0.2 GB");
        assert_eq!(py_repr("nope"), "'nope'");
        assert_eq!(py_repr("it's"), "\"it's\"");
    }

    #[test]
    fn normalize_paths() {
        assert_eq!(normalize(Path::new("/a/b/../c/./d")), PathBuf::from("/a/c/d"));
    }
}
