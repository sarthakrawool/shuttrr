//! Step progress parsed from a stage text, a running job's step rate, and the clients seen
//! recently (GET /v1/clients). Same rules as `shutter/core/stats.py`.

use crate::util::{now, round_to};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

pub const CLIENT_WINDOW_S: f64 = 600.0;
pub const THROUGHPUT_WINDOW_S: f64 = 3600.0;
pub const MAX_CLIENT_LEN: usize = 120;

const STEP_WORDS: [&str; 6] = ["denoise", "denoising", "step", "steps", "sampling", "sample"];

/// (step, steps) from "denoise 3/8" or "segment 1/2 denoise 3/8"; None without a counter.
pub fn parse_step(stage: &str) -> Option<(u64, u64)> {
    let lower = stage.to_ascii_lowercase();
    let toks: Vec<&str> = lower.split_whitespace().collect();
    for w in toks.windows(2) {
        let word = w[0].trim_matches(|c: char| !c.is_ascii_alphanumeric());
        if !STEP_WORDS.contains(&word) {
            continue;
        }
        // "3/8", possibly followed by punctuation such as "3/8,"
        let frac: String = w[1].chars().take_while(|c| c.is_ascii_digit() || *c == '/').collect();
        if let Some((a, b)) = frac.split_once('/') {
            if let (Ok(a), Ok(b)) = (a.parse::<u64>(), b.parse::<u64>()) {
                if b > 0 && a <= b {
                    return Some((a, b));
                }
            }
        }
    }
    // "denoise 3 / 8" (spaced)
    for w in toks.windows(4) {
        let word = w[0].trim_matches(|c: char| !c.is_ascii_alphanumeric());
        if STEP_WORDS.contains(&word) && w[2] == "/" {
            if let (Ok(a), Ok(b)) = (w[1].parse::<u64>(), w[3].parse::<u64>()) {
                if b > 0 && a <= b {
                    return Some((a, b));
                }
            }
        }
    }
    None
}

/// Steps per second of one running job; restarting counters (a new segment) keep counting.
#[derive(Debug, Default)]
pub struct StepRate {
    base: u64,
    last: Option<(u64, u64)>,
    first: Option<(Instant, u64)>,
    last_t: Option<Instant>,
    total: u64,
}

impl StepRate {
    pub fn update(&mut self, step: u64, steps: u64) {
        self.update_at(step, steps, Instant::now())
    }

    pub fn update_at(&mut self, step: u64, steps: u64, t: Instant) {
        if let Some((ls, lsteps)) = self.last {
            if step < ls {
                self.base += lsteps;
            }
        }
        self.last = Some((step, steps));
        let n = self.base + step;
        if self.first.is_none() {
            self.first = Some((t, n));
        }
        if n != self.total || self.last_t.is_none() {
            self.last_t = Some(t);
        }
        self.total = n;
    }

    pub fn snapshot(&self) -> Value {
        let rate = match (self.first, self.last_t) {
            (Some((t0, n0)), Some(t1)) if t1 > t0 && self.total > n0 => {
                Some(round_to((self.total - n0) as f64 / (t1 - t0).as_secs_f64(), 4))
            }
            _ => None,
        };
        json!({"step": self.last.map(|l| l.0), "steps": self.last.map(|l| l.1), "steps_per_s": rate})
    }
}

/// The submitter's client name: the body's `client`, else X-Shutter-Client, else the
/// User-Agent's first word; trimmed to 120 characters.
pub fn client_label(body: Option<&str>, header: Option<&str>, user_agent: Option<&str>) -> Option<String> {
    for v in [body, header].into_iter().flatten() {
        let t = v.trim();
        if !t.is_empty() {
            return Some(t.chars().take(MAX_CLIENT_LEN).collect());
        }
    }
    user_agent
        .and_then(|u| u.split_whitespace().next())
        .map(|w| w.chars().take(MAX_CLIENT_LEN).collect())
}

#[derive(Debug, Clone)]
struct ClientRow {
    first_seen: f64,
    last_seen: f64,
    times: Vec<f64>,
    total: u64,
}

/// Clients seen recently, in memory (a restart forgets them).
pub struct ClientTracker {
    window_s: f64,
    rows: Mutex<HashMap<String, ClientRow>>,
}

impl Default for ClientTracker {
    fn default() -> Self {
        ClientTracker { window_s: CLIENT_WINDOW_S, rows: Mutex::new(HashMap::new()) }
    }
}

impl ClientTracker {
    fn with<R>(&self, f: impl FnOnce(&mut HashMap<String, ClientRow>) -> R) -> R {
        let mut g = self.rows.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut g)
    }

    fn row<'a>(rows: &'a mut HashMap<String, ClientRow>, c: &str, t: f64) -> &'a mut ClientRow {
        rows.entry(c.to_string()).or_insert(ClientRow { first_seen: t, last_seen: t, times: vec![], total: 0 })
    }

    fn prune(&self, rows: &mut HashMap<String, ClientRow>, t: f64) {
        let cut = t - self.window_s;
        rows.retain(|_, r| {
            r.times.retain(|x| *x >= cut);
            r.last_seen >= cut
        });
    }

    /// A request was submitted by `client`.
    pub fn submitted(&self, client: Option<&str>) {
        let Some(c) = client.filter(|c| !c.is_empty()) else { return };
        let t = now();
        self.with(|rows| {
            let r = Self::row(rows, c, t);
            r.last_seen = t;
            r.times.push(t);
            r.total += 1;
            self.prune(rows, t);
        });
    }

    /// Any API call carrying X-Shutter-Client: refresh last_seen only.
    pub fn seen(&self, client: Option<&str>) {
        let Some(c) = client.filter(|c| !c.is_empty()) else { return };
        let t = now();
        self.with(|rows| Self::row(rows, c, t).last_seen = t);
    }

    pub fn list(&self) -> Value {
        let t = now();
        let mut out: Vec<Value> = self.with(|rows| {
            self.prune(rows, t);
            rows.iter()
                .map(|(c, r)| {
                    json!({"client": c, "requests": r.times.len(), "requests_total": r.total,
                           "first_seen": r.first_seen, "last_seen": r.last_seen,
                           "idle_s": round_to((t - r.last_seen).max(0.0), 1)})
                })
                .collect()
        });
        out.sort_by(|a, b| {
            b["last_seen"].as_f64().unwrap_or(0.0).partial_cmp(&a["last_seen"].as_f64().unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal)
        });
        json!({"window_s": self.window_s as i64, "clients": out})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn steps_from_stage_text() {
        assert_eq!(parse_step("denoise 3/8"), Some((3, 8)));
        assert_eq!(parse_step("segment 1/2 denoise 3/8"), Some((3, 8)));
        assert_eq!(parse_step("denoise 2/5 (stream_wide)"), Some((2, 5)));
        assert_eq!(parse_step("Step 4 / 10"), Some((4, 10)));
        assert_eq!(parse_step("segment 1/2 decode"), None);
        assert_eq!(parse_step("loading"), None);
        assert_eq!(parse_step("denoise 9/8"), None);
    }

    #[test]
    fn step_rate_counts_across_segments() {
        let t0 = Instant::now();
        let mut r = StepRate::default();
        assert_eq!(r.snapshot()["steps_per_s"], Value::Null);
        r.update_at(1, 4, t0);
        r.update_at(4, 4, t0 + Duration::from_secs(3));
        r.update_at(2, 4, t0 + Duration::from_secs(5)); // next segment: 4 + 2
        let s = r.snapshot();
        assert_eq!(s["step"], 2);
        assert_eq!(s["steps"], 4);
        assert_eq!(s["steps_per_s"], 1.0); // (6 - 1) / 5 s
    }

    #[test]
    fn clients() {
        assert_eq!(client_label(Some(" comfyui/0.46 "), None, None).as_deref(), Some("comfyui/0.46"));
        assert_eq!(client_label(None, Some("x"), Some("curl/8")).as_deref(), Some("x"));
        assert_eq!(client_label(Some(""), None, Some("curl/8.1 more")).as_deref(), Some("curl/8.1"));
        assert_eq!(client_label(None, None, None), None);
        let t = ClientTracker::default();
        t.submitted(Some("a"));
        t.submitted(Some("a"));
        t.seen(Some("b"));
        t.submitted(None);
        let l = t.list();
        let rows = l["clients"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        let a = rows.iter().find(|r| r["client"] == "a").unwrap();
        assert_eq!(a["requests"], 2);
        let b = rows.iter().find(|r| r["client"] == "b").unwrap();
        assert_eq!(b["requests"], 0);
        assert_eq!(l["window_s"], 600);
    }
}
