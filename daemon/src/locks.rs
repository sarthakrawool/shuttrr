//! GPU locks shared with other tools on the Mac.
//!
//! * `GpuLock`: an exclusive `flock` on a lock file (port of `memory.GpuLock`). The holder
//!   writes `{"pid","label","time"}`; the file is truncated on release, never deleted.
//! * `DirLocks`: `mkdir` locks (the `gpulock.sh` convention, port of `core/dirlock.py`),
//!   taken in order, all or nothing; only directories this process created are removed.

use crate::util::{now, pid_alive};
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

pub struct GpuLock {
    pub path: PathBuf,
    pub label: String,
    file: Option<File>,
}

fn flock(f: &File, op: libc::c_int) -> std::io::Result<()> {
    // SAFETY: flock on a valid open fd.
    let r = unsafe { libc::flock(f.as_raw_fd(), op) };
    if r == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

fn would_block(e: &std::io::Error) -> bool {
    matches!(e.raw_os_error(), Some(c) if c == libc::EWOULDBLOCK || c == libc::EAGAIN || c == libc::EACCES)
}

impl GpuLock {
    pub fn new(path: PathBuf, label: String) -> Self {
        GpuLock { path, label, file: None }
    }

    pub fn held(&self) -> bool {
        self.file.is_some()
    }

    /// One non-blocking attempt. Ok(true) when we hold it (now or already).
    pub fn try_acquire(&mut self) -> Result<bool, String> {
        if self.file.is_some() {
            return Ok(true);
        }
        if let Some(p) = self.path.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        // std opens with O_CLOEXEC, so the worker child never inherits the lock.
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.path)
            .map_err(|e| format!("cannot open GPU lock {}: {e}", self.path.display()))?;
        if let Err(e) = flock(&f, libc::LOCK_EX | libc::LOCK_NB) {
            if would_block(&e) {
                return Ok(false);
            }
            return Err(format!("cannot lock {}: {e}", self.path.display()));
        }
        let info = json!({"pid": std::process::id(), "label": self.label, "time": now()}).to_string();
        let _ = f.set_len(0);
        let _ = f.seek(SeekFrom::Start(0));
        let _ = f.write_all(info.as_bytes());
        let _ = f.sync_all();
        self.file = Some(f);
        Ok(true)
    }

    pub fn release(&mut self) {
        if let Some(f) = self.file.take() {
            let _ = f.set_len(0);
            let _ = flock(&f, libc::LOCK_UN);
        }
    }

    /// Who holds the lock: the holder's JSON, or None when nobody holds it.
    pub fn holder(&self) -> Option<Value> {
        let locked = if self.file.is_some() {
            true
        } else {
            let f = File::open(&self.path).ok()?;
            match flock(&f, libc::LOCK_SH | libc::LOCK_NB) {
                Ok(()) => {
                    let _ = flock(&f, libc::LOCK_UN);
                    false
                }
                Err(_) => true,
            }
        };
        if !locked {
            return None;
        }
        let mut raw = String::new();
        if let Ok(f) = File::open(&self.path) {
            let mut buf = Vec::new();
            let _ = f.take(4096).read_to_end(&mut buf);
            raw = String::from_utf8_lossy(&buf).into_owned();
        }
        match serde_json::from_str::<Value>(&raw) {
            Ok(Value::Object(o)) => Some(json!({
                "pid": o.get("pid").cloned().unwrap_or(Value::Null),
                "label": o.get("label").cloned().unwrap_or(Value::Null),
                "time": o.get("time").cloned().unwrap_or(Value::Null),
            })),
            _ => Some(json!({"pid": null, "label": null, "time": null,
                             "raw": raw.chars().take(200).collect::<String>()})),
        }
    }
}

impl Drop for GpuLock {
    fn drop(&mut self) {
        self.release();
    }
}

// ---- mkdir locks ----------------------------------------------------------------------

const OWNER_FILES: [&str; 3] = ["owner", "owner_speed", "pid"];

pub fn read_owner(path: &Path) -> Vec<(&'static str, Option<String>)> {
    OWNER_FILES
        .iter()
        .map(|name| {
            let v = File::open(path.join(name)).ok().and_then(|f| {
                let mut buf = Vec::new();
                f.take(200).read_to_end(&mut buf).ok()?;
                let s = String::from_utf8_lossy(&buf).trim().to_string();
                (!s.is_empty()).then_some(s)
            });
            (*name, v)
        })
        .collect()
}

fn owner_get(info: &[(&'static str, Option<String>)], k: &str) -> Option<String> {
    info.iter().find(|(n, _)| *n == k).and_then(|(_, v)| v.clone())
}

/// Human description of who holds the lock dir `path`.
pub fn describe_holder(path: &Path) -> String {
    let info = read_owner(path);
    let who = owner_get(&info, "owner").or_else(|| owner_get(&info, "owner_speed"));
    let pid = owner_get(&info, "pid");
    let mut parts = vec![who.clone().unwrap_or_else(|| "an unknown owner".into())];
    if let Some(pid) = pid {
        parts.push(format!("pid {pid}"));
        let alive = pid.parse::<i32>().ok().and_then(pid_alive);
        if alive == Some(false) {
            parts.push(
                "that process is gone; the lock may be stale, check it and remove it by hand if so".into(),
            );
        }
    } else if who.is_none() {
        parts.push("no owner file; it may be stale, check it before removing it by hand".into());
    }
    parts.join(", ")
}

pub struct DirLocks {
    pub paths: Vec<PathBuf>,
    pub owner: String,
    created: Vec<PathBuf>,
}

impl DirLocks {
    pub fn new(paths: Vec<PathBuf>, owner: &str) -> Self {
        DirLocks { paths, owner: owner.into(), created: Vec::new() }
    }

    pub fn held(&self) -> bool {
        !self.created.is_empty()
    }

    pub fn ours(&self, p: &Path) -> bool {
        self.created.iter().any(|c| c == p)
    }

    /// One attempt at every lock. None on success, else why not (and nothing held).
    pub fn try_acquire(&mut self) -> Option<String> {
        if !self.created.is_empty() && self.created.len() == self.paths.len() {
            return None;
        }
        self.release();
        for path in self.paths.clone() {
            if let Some(parent) = path.parent() {
                if !parent.is_dir() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        self.release();
                        return Some(format!("cannot take GPU lock {}: {e}", path.display()));
                    }
                }
            }
            match std::fs::create_dir(&path) {
                Ok(()) => {
                    self.created.push(path.clone());
                    for (name, value) in [("owner", self.owner.clone()), ("pid", std::process::id().to_string())] {
                        let _ = std::fs::write(path.join(name), format!("{value}\n"));
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let reason = format!("GPU lock {} is held by {}", path.display(), describe_holder(&path));
                    self.release();
                    return Some(reason);
                }
                Err(e) => {
                    self.release();
                    return Some(format!("cannot take GPU lock {}: {e}", path.display()));
                }
            }
        }
        None
    }

    /// Remove the dirs this instance created, newest first. Never touches others: a dir
    /// whose pid file names another process is left alone.
    pub fn release(&mut self) {
        let me = std::process::id().to_string();
        for path in std::mem::take(&mut self.created).into_iter().rev() {
            let info = read_owner(&path);
            match owner_get(&info, "pid") {
                Some(p) if p != me => continue,
                _ => {
                    let _ = std::fs::remove_dir_all(&path);
                }
            }
        }
    }

    /// Every configured lock that exists and is not ours, with its owner files.
    pub fn holders(&self) -> Vec<Value> {
        self.paths
            .iter()
            .filter(|p| !self.ours(p) && p.is_dir())
            .map(|p| {
                let info = read_owner(p);
                json!({"path": p.to_string_lossy(), "owner": owner_get(&info, "owner"),
                       "owner_speed": owner_get(&info, "owner_speed"), "pid": owner_get(&info, "pid"),
                       "label": describe_holder(p)})
            })
            .collect()
    }
}

impl Drop for DirLocks {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn flock_excludes_another_process() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("gpu.lock");
        let mut a = GpuLock::new(p.clone(), "test-a".into());
        assert!(a.try_acquire().unwrap());
        let h = a.holder().unwrap();
        assert_eq!(h["label"], "test-a");
        // another process (flock(1) or python) cannot take it while we hold it
        let script = format!(
            "import fcntl,sys\nf=open({:?},'a')\ntry:\n fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);print('got')\nexcept OSError:\n print('busy')",
            p.to_string_lossy()
        );
        let out = Command::new("python3").args(["-c", &script]).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "busy");
        a.release();
        assert!(a.holder().is_none());
        let out = Command::new("python3").args(["-c", &script]).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "got");
        // and while another process holds it, we cannot
        let hold = format!(
            "import fcntl,time,sys\nf=open({:?},'w')\nfcntl.flock(f,fcntl.LOCK_EX);f.write('junk');f.flush();print('ok',flush=True);time.sleep(3)",
            p.to_string_lossy()
        );
        let mut child = Command::new("python3").args(["-c", &hold]).stdout(std::process::Stdio::piped()).spawn().unwrap();
        let mut line = [0u8; 3];
        child.stdout.as_mut().unwrap().read_exact(&mut line).unwrap();
        let mut b = GpuLock::new(p.clone(), "test-b".into());
        assert!(!b.try_acquire().unwrap());
        let h = b.holder().unwrap();
        assert_eq!(h["raw"], "junk");
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(b.try_acquire().unwrap());
    }

    #[test]
    fn dirlocks_all_or_nothing_and_never_remove_others() {
        let d = tempfile::tempdir().unwrap();
        let a = d.path().join("a.lock.d");
        let b = d.path().join("q/b.lock.d");
        let mut l = DirLocks::new(vec![a.clone(), b.clone()], "shutter");
        assert!(l.try_acquire().is_none());
        assert!(a.join("owner").exists() && b.is_dir());
        assert_eq!(std::fs::read_to_string(a.join("pid")).unwrap().trim(), std::process::id().to_string());
        // a second holder (another process's view) is refused and holds nothing
        let mut other = DirLocks::new(vec![a.clone()], "other");
        let r = other.try_acquire().unwrap();
        assert!(r.contains("is held by shutter, pid"), "{r}");
        l.release();
        assert!(!a.exists() && !b.exists());

        // someone else holds b: we take nothing, and a is not left behind
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(b.join("owner_speed"), "studio fast\n").unwrap();
        std::fs::write(b.join("pid"), "999999\n").unwrap();
        let r = l.try_acquire().unwrap();
        assert!(r.contains("held by studio fast, pid 999999, that process is gone"), "{r}");
        assert!(!a.exists());
        assert_eq!(l.holders().len(), 1);
        l.release();
        assert!(b.is_dir(), "never remove a lock we do not own");

        // a lock dir replaced by someone else while we held it is left alone
        std::fs::remove_dir_all(&b).unwrap();
        assert!(l.try_acquire().is_none());
        std::fs::write(a.join("pid"), "1\n").unwrap();
        l.release();
        assert!(a.is_dir() && !b.exists());
        let r = DirLocks::new(vec![d.path().join("x.lock.d")], "s");
        std::fs::create_dir(d.path().join("x.lock.d")).unwrap();
        assert!(describe_holder(&d.path().join("x.lock.d")).contains("no owner file"));
        drop(r);
    }

    #[test]
    fn dirlocks_between_two_processes() {
        let d = tempfile::tempdir().unwrap();
        let a = d.path().join("studio-gpu.lock.d");
        // the other process uses the shell convention: mkdir + owner file
        let st = Command::new("sh")
            .args(["-c", &format!("mkdir {0} && echo render > {0}/owner && echo $$ > {0}/pid", a.display())])
            .status()
            .unwrap();
        assert!(st.success());
        let mut l = DirLocks::new(vec![a.clone()], "shutter");
        let r = l.try_acquire().unwrap();
        assert!(r.contains("held by render"), "{r}");
        std::fs::remove_dir_all(&a).unwrap();
        assert!(l.try_acquire().is_none());
        let st = Command::new("sh").args(["-c", &format!("mkdir {} 2>/dev/null", a.display())]).status().unwrap();
        assert!(!st.success(), "the other process must not get the lock we hold");
    }
}
