//! The worker child process: spawn in its own process group, the JSON-lines pipe, the
//! stderr tail and kill-the-whole-tree. Nothing a worker does on its pipes can make the
//! daemon panic: garbage lines are logged and skipped, EOF and crashes become messages.

use crate::config::Config;
use crate::protocol::{self, Event};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, watch};

pub const TAIL_LINES: usize = 200;
const MAX_LINE: usize = 32 * 1024 * 1024;

/// What the running job hears from the worker.
#[derive(Debug)]
pub enum JobMsg {
    Progress { progress: f64, stage: String },
    Done(Value),
    Failed { error: String, cancelled: bool, memory_refused: bool },
    Log { level: String, message: String },
    /// The worker process ended (after its stdout was drained). Text describes the exit.
    Exited(String),
}

#[derive(Default)]
struct Inner {
    ready: Option<oneshot::Sender<(String, i64, i64)>>,
    estimates: HashMap<String, oneshot::Sender<Value>>,
    pool_waiters: Vec<oneshot::Sender<Value>>,
    released_waiters: Vec<oneshot::Sender<(Vec<String>, Vec<String>)>>,
    models_waiters: Vec<oneshot::Sender<Value>>,
    job: Option<(String, mpsc::UnboundedSender<JobMsg>)>,
    tail: VecDeque<String>,
    last_pool: Option<(Value, Instant)>,
    last_models: Option<Value>,
    last_error_log: Option<String>,
    garbage_lines: u64,
}

impl Inner {
    fn push_tail(&mut self, line: String) {
        if self.tail.len() == TAIL_LINES {
            self.tail.pop_front();
        }
        let mut line = line;
        if line.len() > 4000 {
            let mut cut = 4000;
            while !line.is_char_boundary(cut) {
                cut -= 1;
            }
            line.truncate(cut);
            line.push_str(" ...");
        }
        self.tail.push_back(line);
    }
}

pub struct Worker {
    pub pid: i32,
    pub version: String,
    pub started: Instant,
    pub started_at: f64,
    stdin: tokio::sync::Mutex<Option<ChildStdin>>,
    inner: Arc<Mutex<Inner>>,
    exit_rx: watch::Receiver<Option<String>>,
}

fn lock(m: &Mutex<Inner>) -> std::sync::MutexGuard<'_, Inner> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn describe_status(st: &std::process::ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;
    if let Some(c) = st.code() {
        format!("exit code {c}")
    } else if let Some(s) = st.signal() {
        let name = match s {
            libc::SIGKILL => "SIGKILL",
            libc::SIGTERM => "SIGTERM",
            libc::SIGSEGV => "SIGSEGV",
            libc::SIGABRT => "SIGABRT",
            libc::SIGBUS => "SIGBUS",
            libc::SIGINT => "SIGINT",
            _ => "",
        };
        if name.is_empty() {
            format!("signal {s}")
        } else {
            format!("signal {s} ({name})")
        }
    } else {
        "unknown exit".into()
    }
}

/// Why the worker could not be started: the job's error message (complete, ready to show)
/// and the worker's stderr tail when there is one.
#[derive(Debug, Clone)]
pub struct StartError {
    pub message: String,
    pub tail: Option<String>,
}

impl std::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

fn start_err(why: impl std::fmt::Display) -> StartError {
    StartError { message: format!("could not start the worker: {why}"), tail: None }
}

pub const WORKER_CMD_HINT: &str = "Set worker_cmd in config.json to a Python that has numpy and mlx.";

/// The line that says why a worker died at start-up: its own `log` error if it sent one,
/// else the exception line of a Python traceback (`ModuleNotFoundError: ...`), else the
/// last non-empty stderr line that is not traceback scaffolding.
pub fn meaningful_failure_line(error_log: Option<&str>, tail: &[String]) -> Option<String> {
    let clean = |s: &str| -> String {
        let mut t = s.trim();
        for p in ["worker failed to start:", "error:"] {
            if t.len() >= p.len() && t[..p.len()].eq_ignore_ascii_case(p) {
                t = t[p.len()..].trim_start();
            }
        }
        t.trim_end_matches('.').to_string()
    };
    if let Some(e) = error_log.filter(|e| !e.trim().is_empty()) {
        return Some(clean(e));
    }
    let lines: Vec<&str> = tail
        .iter()
        .map(|l| l.trim_end())
        .filter(|l| !l.trim().is_empty() && !l.starts_with("stdout (ignored"))
        .collect();
    let is_exc = |l: &str| {
        let head = l.split(':').next().unwrap_or("");
        !l.starts_with(' ')
            && l.contains(':')
            && !head.is_empty()
            && head.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '.')
            && (head.ends_with("Error") || head.ends_with("Exception") || head.ends_with("Exit") || head.ends_with("Interrupt"))
    };
    if let Some(l) = lines.iter().rev().find(|l| is_exc(l)) {
        return Some(clean(l));
    }
    lines
        .iter()
        .rev()
        .find(|l| !l.starts_with(' ') && !l.starts_with("Traceback (most recent call last)"))
        .map(|l| clean(l))
}

impl Worker {
    /// Start the worker and wait for `ready`. Errors carry a plain-language reason (for a
    /// start-up failure: the worker's own `log` text, else its last stderr line).
    pub async fn spawn(cfg: &Config, cwd: Option<&std::path::Path>) -> Result<Arc<Worker>, StartError> {
        let argv = &cfg.worker_cmd;
        let Some(prog) = argv.first() else {
            return Err(start_err("worker_cmd is empty"));
        };
        let mut cmd = Command::new(prog);
        cmd.args(&argv[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("PYTHONUNBUFFERED", "1")
            .env("SHUTTER_HOME", &cfg.home)
            .env("SHUTTER_DAEMON_PID", std::process::id().to_string())
            .kill_on_drop(false);
        if let Some(d) = cwd {
            cmd.current_dir(d);
        }
        // SAFETY: setsid is async-signal-safe; it puts the worker in its own session and
        // process group (pgid == pid), so killpg reaches every process it starts.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = cmd.spawn().map_err(|e| start_err(format!("cannot run {prog:?}: {e}. {WORKER_CMD_HINT}")))?;
        let pid = child.id().map(|p| p as i32).unwrap_or(-1);
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().ok_or_else(|| start_err("worker has no stdout"))?;
        let stderr = child.stderr.take().ok_or_else(|| start_err("worker has no stderr"))?;
        let inner = Arc::new(Mutex::new(Inner::default()));
        let (ready_tx, ready_rx) = oneshot::channel();
        lock(&inner).ready = Some(ready_tx);
        let (exit_tx, exit_rx) = watch::channel(None);

        // stdout: protocol lines only
        let inn = inner.clone();
        let (eof_tx, eof_rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let mut r = BufReader::new(stdout);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match r.read_until(b'\n', &mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                if buf.len() > MAX_LINE {
                    lock(&inn).push_tail("stdout: [line too long, ignored]".into());
                    continue;
                }
                let line = String::from_utf8_lossy(&buf);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                dispatch(&inn, line);
            }
            let _ = eof_tx.send(());
            // stdout closed: a worker that keeps running without its protocol channel is
            // useless (and could hold a job forever), so its group is killed after 2 s.
            tokio::time::sleep(Duration::from_secs(2)).await;
            if pid > 0 && crate::util::pid_alive(pid) == Some(true) {
                lock(&inn).push_tail("stdout closed but the worker kept running; stopped it".into());
                // SAFETY: plain syscalls on our own child's process group.
                unsafe {
                    libc::killpg(pid, libc::SIGKILL);
                }
            }
        });
        // stderr: kept as the log tail
        let inn = inner.clone();
        tokio::spawn(async move {
            let mut r = BufReader::new(stderr);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match r.read_until(b'\n', &mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                let line = String::from_utf8_lossy(&buf).trim_end().to_string();
                lock(&inn).push_tail(line);
            }
        });
        // waiter: reap, drain stdout, then tell everyone
        let inn = inner.clone();
        tokio::spawn(async move {
            let desc = match child.wait().await {
                Ok(st) => describe_status(&st),
                Err(e) => format!("wait failed: {e}"),
            };
            // a grandchild may keep stdout open; do not wait for it forever
            let _ = tokio::time::timeout(Duration::from_secs(2), eof_rx).await;
            let mut g = lock(&inn);
            if let Some((_, tx)) = g.job.take() {
                let _ = tx.send(JobMsg::Exited(desc.clone()));
            }
            g.ready = None;
            g.estimates.clear();
            g.pool_waiters.clear();
            g.released_waiters.clear();
            g.models_waiters.clear();
            drop(g);
            let _ = exit_tx.send(Some(desc));
        });

        let w = Worker {
            pid,
            version: String::new(),
            started: Instant::now(),
            started_at: crate::util::now(),
            stdin: tokio::sync::Mutex::new(stdin),
            inner,
            exit_rx,
        };
        let _ = w.send(&protocol::op_hello()).await;
        let timeout = crate::util::secs(cfg.worker_start_timeout_s);
        let exit = w.exit_rx.clone();
        let outcome = tokio::select! {
            r = ready_rx => r.ok(),
            _ = tokio::time::sleep(timeout) => None,
            _ = exited(exit) => None,
        };
        match outcome {
            Some((version, _pid, proto)) => {
                if proto != protocol::PROTOCOL {
                    w.kill();
                    return Err(start_err(format!(
                        "the worker speaks protocol {proto}, this daemon speaks {}",
                        protocol::PROTOCOL
                    )));
                }
                let mut w = w;
                w.version = version;
                Ok(Arc::new(w))
            }
            None => {
                let exited = w.exit_status();
                if exited.is_none() {
                    w.kill();
                    let _ = w.wait_exit(Duration::from_secs(3)).await;
                    let mut e = start_err(format!(
                        "the worker did not say it was ready within {} s ({})",
                        cfg.worker_start_timeout_s,
                        w.last_line().unwrap_or_else(|| "no output".into())
                    ));
                    e.tail = Some(w.tail().join("\n"));
                    return Err(e);
                }
                // Exited before `ready`: lead with a plain sentence (usually a Python
                // without numpy or mlx); the stderr tail goes to the job's log_tail.
                let why = w.start_failure_text();
                let status = exited.unwrap_or_default();
                let mut tail = w.tail();
                tail.push(format!("worker {status}"));
                Err(StartError {
                    message: format!("The model worker could not start: {why}. {WORKER_CMD_HINT}"),
                    tail: Some(tail.join("\n")),
                })
            }
        }
    }

    fn start_failure_text(&self) -> String {
        let g = lock(&self.inner);
        let tail: Vec<String> = g.tail.iter().cloned().collect();
        meaningful_failure_line(g.last_error_log.as_deref(), &tail)
            .unwrap_or_else(|| "it exited before it was ready, without saying why".into())
    }

    pub fn last_line(&self) -> Option<String> {
        lock(&self.inner).tail.iter().rev().find(|l| !l.trim().is_empty()).cloned()
    }

    pub fn alive(&self) -> bool {
        self.exit_rx.borrow().is_none()
    }

    pub fn exit_status(&self) -> Option<String> {
        self.exit_rx.borrow().clone()
    }

    /// Wait until the process has exited (true) or the timeout passed (false).
    pub async fn wait_exit(&self, timeout: Duration) -> bool {
        let rx = self.exit_rx.clone();
        tokio::time::timeout(timeout, exited(rx)).await.is_ok()
    }

    /// SIGKILL the worker's whole process group.
    pub fn kill(&self) {
        if self.pid > 0 {
            // SAFETY: plain syscalls on our own child's process group / pid.
            unsafe {
                libc::killpg(self.pid, libc::SIGKILL);
                libc::kill(self.pid, libc::SIGKILL);
            }
        }
    }

    pub async fn send(&self, v: &Value) -> Result<(), String> {
        let line = protocol::encode(v);
        let mut g = self.stdin.lock().await;
        let Some(stdin) = g.as_mut() else { return Err("worker stdin is closed".into()) };
        let r = async {
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await
        };
        match tokio::time::timeout(Duration::from_secs(10), r).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                *g = None;
                Err(format!("cannot write to the worker: {e}"))
            }
            Err(_) => Err("writing to the worker timed out".into()),
        }
    }

    /// Ask for a memory estimate. Err texts are user-facing.
    pub async fn estimate(&self, id: &str, request: &Value, timeout: Duration) -> Result<Value, EstimateError> {
        let (tx, rx) = oneshot::channel();
        lock(&self.inner).estimates.insert(id.to_string(), tx);
        if let Err(e) = self.send(&protocol::op_estimate(id, request)).await {
            lock(&self.inner).estimates.remove(id);
            return Err(EstimateError::Died(e));
        }
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(_)) => Err(EstimateError::Died(self.exit_status().unwrap_or_else(|| "worker exited".into()))),
            Err(_) => {
                lock(&self.inner).estimates.remove(id);
                Err(EstimateError::Timeout)
            }
        }
    }

    pub async fn stats(&self, timeout: Duration) -> Option<Value> {
        let (tx, rx) = oneshot::channel();
        lock(&self.inner).pool_waiters.push(tx);
        self.send(&protocol::op_stats()).await.ok()?;
        tokio::time::timeout(timeout, rx).await.ok()?.ok()
    }

    pub async fn release(&self, key: Option<&str>, timeout: Duration) -> Option<(Vec<String>, Vec<String>)> {
        let (tx, rx) = oneshot::channel();
        lock(&self.inner).released_waiters.push(tx);
        self.send(&protocol::op_release(key)).await.ok()?;
        tokio::time::timeout(timeout, rx).await.ok()?.ok()
    }

    pub async fn models(&self, timeout: Duration) -> Option<Value> {
        let (tx, rx) = oneshot::channel();
        lock(&self.inner).models_waiters.push(tx);
        self.send(&protocol::op_models()).await.ok()?;
        tokio::time::timeout(timeout, rx).await.ok()?.ok()
    }

    pub fn last_pool(&self) -> Option<(Value, Instant)> {
        lock(&self.inner).last_pool.clone()
    }

    pub fn last_models(&self) -> Option<Value> {
        lock(&self.inner).last_models.clone()
    }

    pub fn garbage_lines(&self) -> u64 {
        lock(&self.inner).garbage_lines
    }

    /// Route progress/done/failed/log for `id` to the returned channel; clears the tail
    /// (the log tail is per job). If the worker is already gone, Exited arrives at once.
    pub fn attach_job(&self, id: &str) -> mpsc::UnboundedReceiver<JobMsg> {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut g = lock(&self.inner);
        g.tail.clear();
        if let Some(desc) = self.exit_rx.borrow().clone() {
            let _ = tx.send(JobMsg::Exited(desc));
        } else {
            g.job = Some((id.to_string(), tx));
        }
        rx
    }

    pub fn detach_job(&self) {
        lock(&self.inner).job = None;
    }

    pub fn tail(&self) -> Vec<String> {
        lock(&self.inner).tail.iter().cloned().collect()
    }

    /// `shutdown`, then SIGKILL the group after `grace`.
    pub async fn stop(&self, grace: Duration) {
        if self.alive() {
            let _ = self.send(&protocol::op_shutdown()).await;
            if !self.wait_exit(grace).await {
                self.kill();
            }
        }
        // stray grandchildren left in the group go too
        self.kill();
        let _ = self.wait_exit(Duration::from_secs(3)).await;
        *self.stdin.lock().await = None;
    }
}

/// Resolves once the worker has exited (never holds a watch guard across an await).
async fn exited(mut rx: watch::Receiver<Option<String>>) {
    loop {
        if rx.borrow().is_some() {
            return;
        }
        if rx.changed().await.is_err() {
            return;
        }
    }
}

#[derive(Debug)]
pub enum EstimateError {
    Timeout,
    Died(String),
}

fn dispatch(inner: &Mutex<Inner>, line: &str) {
    let mut g = lock(inner);
    let ev = match protocol::decode(line) {
        Ok(ev) => ev,
        Err(why) => {
            g.garbage_lines += 1;
            let shown: String = line.chars().take(300).collect();
            g.push_tail(format!("stdout (ignored, {why}): {shown}"));
            return;
        }
    };
    let job_matches = |g: &Inner, id: &str| g.job.as_ref().map(|(j, _)| j == id).unwrap_or(false);
    match ev {
        Event::Ready { version, pid, protocol } => {
            if let Some(tx) = g.ready.take() {
                let _ = tx.send((version, pid, protocol));
            }
        }
        Event::Estimate { id, msg } => {
            if let Some(tx) = g.estimates.remove(&id) {
                let _ = tx.send(msg);
            }
        }
        Event::Progress { id, progress, stage } => {
            if job_matches(&g, &id) {
                if let Some((_, tx)) = &g.job {
                    let _ = tx.send(JobMsg::Progress { progress, stage });
                }
            }
        }
        Event::Done { id, result } => {
            if job_matches(&g, &id) {
                if let Some((_, tx)) = g.job.take() {
                    let _ = tx.send(JobMsg::Done(result));
                }
            }
        }
        Event::Failed { id, error, cancelled, memory_refused } => {
            if job_matches(&g, &id) {
                if let Some((_, tx)) = g.job.take() {
                    let _ = tx.send(JobMsg::Failed { error, cancelled, memory_refused });
                }
            } else if let Some(tx) = g.estimates.remove(&id) {
                // a failure for an id we only asked an estimate for
                let _ = tx.send(serde_json::json!({"id": id, "error": error}));
            }
        }
        Event::Pool(v) => {
            g.last_pool = Some((v.clone(), Instant::now()));
            for tx in g.pool_waiters.drain(..) {
                let _ = tx.send(v.clone());
            }
        }
        Event::Released { keys, deferred } => {
            for tx in g.released_waiters.drain(..) {
                let _ = tx.send((keys.clone(), deferred.clone()));
            }
        }
        Event::Models(v) => {
            g.last_models = Some(v.clone());
            for tx in g.models_waiters.drain(..) {
                let _ = tx.send(v.clone());
            }
        }
        Event::Log { level, message, id } => {
            g.push_tail(format!("[{level}] {message}"));
            if level == "error" || level == "critical" {
                g.last_error_log = Some(message.clone());
            }
            let for_job = match &id {
                Some(i) => job_matches(&g, i),
                None => g.job.is_some(),
            };
            if for_job {
                if let Some((_, tx)) = &g.job {
                    let _ = tx.send(JobMsg::Log { level, message });
                }
            }
        }
        Event::Unknown(_) => {}
    }
}

#[cfg(test)]
mod start_failure_tests {
    use super::meaningful_failure_line;

    #[test]
    fn picks_the_reason_out_of_a_traceback() {
        let tail: Vec<String> = [
            "Traceback (most recent call last):",
            "  File \"<frozen runpy>\", line 198, in _run_module_as_main",
            "  File \"/x/shutter/worker.py\", line 12, in <module>",
            "    import numpy as np",
            "ModuleNotFoundError: No module named 'numpy'",
            "",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(meaningful_failure_line(None, &tail).unwrap(), "ModuleNotFoundError: No module named 'numpy'");
        assert_eq!(meaningful_failure_line(Some("worker failed to start: no mlx."), &tail).unwrap(), "no mlx");
        assert_eq!(meaningful_failure_line(None, &["python3: not found".to_string()]).unwrap(), "python3: not found");
        assert_eq!(meaningful_failure_line(None, &[]), None);
    }
}
