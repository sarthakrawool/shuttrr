//! Jobs and their sqlite persistence (port of `core/queue.py` JobStore). Same schema and
//! the same JSON blob as the Python reference, so both can read one `jobs.db`.

use crate::util::{jf64, jstr, new_id, now};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Map, Value};

pub const STATES: [&str; 6] = ["queued", "waiting_memory", "running", "done", "failed", "cancelled"];
pub const REQUEUE_MESSAGE: &str = "requeued after restart";

pub fn is_terminal(state: &str) -> bool {
    matches!(state, "done" | "failed" | "cancelled")
}

#[derive(Debug, Clone, PartialEq)]
pub struct Job {
    pub id: String,
    pub state: String,
    pub created_at: f64,
    pub started_at: Option<f64>,
    pub finished_at: Option<f64>,
    pub progress: f64,
    pub stage: String,
    pub message: String,
    pub result: Map<String, Value>,
    pub node: String,
    /// Free text the submitter sent (e.g. "comfyui/0.46"); None when unknown.
    pub client: Option<String>,
    /// The normalized JobRequest dict (model, prompt, negative_prompt, width, height, frames,
    /// duration_s, fps, steps, guidance, seed, inputs, output, priority, params).
    pub request: Value,
}

impl Job {
    pub fn new(request: Value) -> Job {
        Job {
            id: new_id(),
            state: "queued".into(),
            created_at: now(),
            started_at: None,
            finished_at: None,
            progress: 0.0,
            stage: String::new(),
            message: String::new(),
            result: Map::new(),
            node: "local".into(),
            client: None,
            request,
        }
    }

    pub fn priority(&self) -> i64 {
        self.request.get("priority").and_then(Value::as_i64).unwrap_or(0)
    }

    pub fn model(&self) -> &str {
        jstr(&self.request, "model").unwrap_or("")
    }

    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id, "state": self.state, "created_at": self.created_at,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "progress": self.progress, "stage": self.stage, "message": self.message,
            "result": self.result, "node": self.node, "client": self.client, "request": self.request,
        })
    }

    pub fn from_json(d: &Value) -> Option<Job> {
        Some(Job {
            id: jstr(d, "id")?.to_string(),
            state: jstr(d, "state")?.to_string(),
            created_at: jf64(d, "created_at")?,
            started_at: jf64(d, "started_at"),
            finished_at: jf64(d, "finished_at"),
            progress: jf64(d, "progress").unwrap_or(0.0),
            stage: jstr(d, "stage").unwrap_or("").to_string(),
            message: jstr(d, "message").unwrap_or("").to_string(),
            result: d.get("result").and_then(Value::as_object).cloned().unwrap_or_default(),
            node: jstr(d, "node").unwrap_or("local").to_string(),
            client: jstr(d, "client").map(String::from),
            request: d.get("request").cloned().unwrap_or_else(|| json!({})),
        })
    }
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    state       TEXT NOT NULL,
    priority    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS jobs_sched ON jobs(state, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at DESC);
";

pub struct Store {
    db: Connection,
}

fn e(err: rusqlite::Error) -> String {
    err.to_string()
}

impl Store {
    /// Open (":memory:" for tests). Jobs left `running` become `queued` again.
    pub fn open(path: &str) -> Result<Store, String> {
        let db = if path == ":memory:" { Connection::open_in_memory() } else { Connection::open(path) }.map_err(e)?;
        db.busy_timeout(std::time::Duration::from_secs(5)).map_err(e)?;
        if path != ":memory:" {
            // journal_mode returns a row; query it rather than execute.
            let _: Option<String> = db.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0)).optional().map_err(e)?;
            db.execute_batch("PRAGMA synchronous=NORMAL;").map_err(e)?;
        }
        db.execute_batch(SCHEMA).map_err(e)?;
        let s = Store { db };
        s.recover()?;
        Ok(s)
    }

    fn recover(&self) -> Result<(), String> {
        let rows: Vec<String> = {
            let mut st = self.db.prepare("SELECT data FROM jobs WHERE state = 'running'").map_err(e)?;
            let it = st.query_map([], |r| r.get::<_, String>(0)).map_err(e)?;
            it.filter_map(Result::ok).collect()
        };
        for data in rows {
            let Some(mut job) = serde_json::from_str::<Value>(&data).ok().as_ref().and_then(Job::from_json) else {
                continue;
            };
            job.state = "queued".into();
            job.message = REQUEUE_MESSAGE.into();
            job.started_at = None;
            job.progress = 0.0;
            job.stage = String::new();
            self.update(&job)?;
        }
        Ok(())
    }

    pub fn add(&self, job: &Job) -> Result<(), String> {
        self.db
            .execute(
                "INSERT INTO jobs (id, state, priority, created_at, data) VALUES (?1,?2,?3,?4,?5)",
                params![job.id, job.state, job.priority(), job.created_at, job.to_json().to_string()],
            )
            .map_err(|err| match err {
                rusqlite::Error::SqliteFailure(f, _) if f.code == rusqlite::ErrorCode::ConstraintViolation => {
                    format!("job {} already exists", job.id)
                }
                other => other.to_string(),
            })?;
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<Option<Job>, String> {
        let row: Option<String> = self
            .db
            .query_row("SELECT data FROM jobs WHERE id = ?1", params![id], |r| r.get(0))
            .optional()
            .map_err(e)?;
        Ok(row.and_then(|d| serde_json::from_str::<Value>(&d).ok()).as_ref().and_then(Job::from_json))
    }

    /// Write the job's fields. Err when the job is unknown.
    pub fn update(&self, job: &Job) -> Result<(), String> {
        let n = self
            .db
            .execute(
                "UPDATE jobs SET state=?1, priority=?2, created_at=?3, data=?4 WHERE id=?5",
                params![job.state, job.priority(), job.created_at, job.to_json().to_string(), job.id],
            )
            .map_err(e)?;
        if n == 0 {
            return Err(format!("no job {}", job.id));
        }
        Ok(())
    }

    fn rows(&self, sql: &str, args: Vec<rusqlite::types::Value>) -> Result<Vec<Job>, String> {
        let mut st = self.db.prepare(sql).map_err(e)?;
        let it = st
            .query_map(rusqlite::params_from_iter(args), |r| r.get::<_, String>(0))
            .map_err(e)?;
        Ok(it
            .filter_map(Result::ok)
            .filter_map(|d| serde_json::from_str::<Value>(&d).ok())
            .filter_map(|v| Job::from_json(&v))
            .collect())
    }

    /// Jobs newest first, optionally filtered by state.
    pub fn list(&self, states: Option<&[String]>, limit: i64) -> Result<Vec<Job>, String> {
        let mut sql = "SELECT data FROM jobs".to_string();
        let mut args: Vec<rusqlite::types::Value> = Vec::new();
        if let Some(st) = states {
            if st.is_empty() {
                return Ok(vec![]);
            }
            sql.push_str(&format!(" WHERE state IN ({})", vec!["?"; st.len()].join(",")));
            args.extend(st.iter().map(|s| rusqlite::types::Value::Text(s.clone())));
        }
        sql.push_str(" ORDER BY created_at DESC, rowid DESC LIMIT ?");
        args.push(rusqlite::types::Value::Integer(limit));
        self.rows(&sql, args)
    }

    /// Highest priority, then oldest, among queued and waiting_memory jobs.
    pub fn next_queued(&self) -> Result<Option<Job>, String> {
        Ok(self
            .rows(
                "SELECT data FROM jobs WHERE state IN ('queued', 'waiting_memory') \
                 ORDER BY priority DESC, created_at ASC, rowid ASC LIMIT 1",
                vec![],
            )?
            .into_iter()
            .next())
    }

    /// Jobs in `state` whose finished_at is at or after `ts` (throughput numbers).
    pub fn finished_since(&self, ts: f64) -> Result<Vec<Job>, String> {
        self.rows(
            "SELECT data FROM jobs WHERE state = 'done' AND json_extract(data, '$.finished_at') >= ?1              ORDER BY created_at DESC LIMIT 10000",
            vec![rusqlite::types::Value::Real(ts)],
        )
    }

    /// Number of jobs per state (every state present).
    pub fn counts(&self) -> Result<Map<String, Value>, String> {
        let mut out = Map::new();
        for s in STATES {
            out.insert(s.into(), json!(0));
        }
        let mut st = self.db.prepare("SELECT state, COUNT(*) FROM jobs GROUP BY state").map_err(e)?;
        let it = st
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            .map_err(e)?;
        for (s, n) in it.filter_map(Result::ok) {
            out.insert(s, json!(n));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(model: &str, prio: i64) -> Job {
        Job::new(json!({"model": model, "priority": prio, "params": {}}))
    }

    #[test]
    fn priority_then_oldest() {
        let s = Store::open(":memory:").unwrap();
        let mut a = job("a", 0);
        a.created_at = 1.0;
        let mut b = job("b", 5);
        b.created_at = 2.0;
        let mut c = job("c", 5);
        c.created_at = 3.0;
        for j in [&a, &b, &c] {
            s.add(j).unwrap();
        }
        assert!(s.add(&a).unwrap_err().contains("already exists"));
        assert_eq!(s.next_queued().unwrap().unwrap().id, b.id);
        b.state = "waiting_memory".into();
        s.update(&b).unwrap();
        assert_eq!(s.next_queued().unwrap().unwrap().id, b.id, "waiting jobs keep their turn");
        b.state = "done".into();
        s.update(&b).unwrap();
        assert_eq!(s.next_queued().unwrap().unwrap().id, c.id);
        let l = s.list(None, 10).unwrap();
        assert_eq!(l.iter().map(|j| j.model()).collect::<Vec<_>>(), vec!["c", "b", "a"]);
        assert_eq!(s.list(Some(&["done".into()]), 10).unwrap().len(), 1);
        assert_eq!(s.list(Some(&[]), 10).unwrap().len(), 0);
        let counts = s.counts().unwrap();
        assert_eq!(counts["queued"], 2);
        assert_eq!(counts["done"], 1);
        assert_eq!(counts["running"], 0);
        let mut ghost = job("g", 0);
        ghost.id = "nope".into();
        assert!(s.update(&ghost).is_err());
    }

    #[test]
    fn requeue_on_reopen() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("jobs.db");
        let p = p.to_str().unwrap();
        let mut j = job("x", 0);
        {
            let s = Store::open(p).unwrap();
            s.add(&j).unwrap();
            j.state = "running".into();
            j.started_at = Some(5.0);
            j.progress = 0.5;
            j.stage = "denoise 3/8".into();
            s.update(&j).unwrap();
        }
        let s = Store::open(p).unwrap();
        let back = s.get(&j.id).unwrap().unwrap();
        assert_eq!(back.state, "queued");
        assert_eq!(back.message, REQUEUE_MESSAGE);
        assert_eq!(back.started_at, None);
        assert_eq!(back.progress, 0.0);
        assert_eq!(back.stage, "");
        assert_eq!(back.request, j.request);
        assert!(s.get("missing").unwrap().is_none());
    }

    #[test]
    fn json_roundtrip() {
        let mut j = job("m", 3);
        j.result.insert("output".into(), json!("/x.mp4"));
        let back = Job::from_json(&j.to_json()).unwrap();
        assert_eq!(back, j);
    }
}
