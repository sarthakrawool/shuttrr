"""The Shutter model worker: a child process of the Rust daemon (`shutterd`) that only
runs models. Protocol: docs/WORKER_PROTOCOL.md (JSON lines, stdin -> ops, stdout -> events).

    python3 -m shutter.worker

Process layout
  * stdout carries ONLY protocol lines. At start-up the real stdout fd is duplicated for
    the protocol writer and fd 1 (and sys.stdout) is pointed at stderr, so a stray print
    from model code, a C library or a child process cannot corrupt the stream. stdin is
    likewise moved to a private fd and fd 0 becomes /dev/null, so a child process (ffmpeg,
    ...) cannot swallow protocol bytes.
  * A reader thread parses ops. Pure ops (hello, estimate, models, stats) and `cancel`
    are answered on the reader thread at once. `run`, `release` and `shutdown` go to the
    main thread, which runs jobs one at a time (MLX/Metal work stays on the main thread).
    While a job runs, queued `release` ops are carried out at the job's next
    report()/check_cancelled() call, i.e. between steps, never concurrently with GPU work.
    Components the running job uses (its adapter's `pool_keys`) are marked in use, so a
    release of them is deferred to the job's end and reported then.
  * EOF on stdin, `shutdown`, SIGTERM or SIGINT: cancel the running job (if any), release
    every component and exit 0. A bad message never ends the process.
"""

from __future__ import annotations

import collections
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from shutter.core.stats import parse_step

PROTOCOL = 1
PROGRESS_MIN_INTERVAL_S = 0.25   # at most ~4 progress events per second
IDLE_TICK_S = 0.25


# ---------------------------------------------------------------------------- stdio


class _Stdio:
    """Owns the protocol fds. After `capture()`, fd 1 and sys.stdout write to stderr."""

    def __init__(self) -> None:
        self.out_fd: Optional[int] = None
        self.in_fd: Optional[int] = None

    def capture(self) -> None:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        self.out_fd = os.dup(1)
        os.dup2(2, 1)
        os.set_inheritable(self.out_fd, False)
        sys.stdout = sys.stderr
        self.in_fd = os.dup(0)
        os.set_inheritable(self.in_fd, False)
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        os.close(devnull)
        try:
            sys.stdin = open(os.devnull, "r")
        except Exception:
            pass


def _clean(v: Any) -> Any:
    """Make a value strict-JSON safe (no NaN/Infinity, numpy scalars, Paths...)."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_clean(x) for x in v]
    if isinstance(v, os.PathLike):
        return os.fspath(v)
    item = getattr(v, "item", None)  # numpy / mlx scalars
    if callable(item):
        try:
            return _clean(item())
        except Exception:
            pass
    tolist = getattr(v, "tolist", None)
    if callable(tolist):
        try:
            return _clean(tolist())
        except Exception:
            pass
    return str(v)


def _log_err(msg: str) -> None:
    try:
        sys.stderr.write(f"[shutter.worker] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _process_peak_rss() -> int:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r) if sys.platform == "darwin" else int(r) * 1024  # Linux reports KiB
    except Exception:
        return 0


def _mlx() -> Any:
    """mlx.core if model code already imported it (never import it just for stats)."""
    return sys.modules.get("mlx.core")


# ---------------------------------------------------------------------------- worker


class _PendingRun:
    def __init__(self, msg: dict[str, Any]):
        self.msg = msg
        self.id = str(msg.get("id") or "")


class Worker:
    def __init__(self, write_fd: int, read_fd: int, *, config: Any = None,
                 probe: Any = None, protected: Any = None):
        from shutter.core.cache import ContentCache
        from shutter.core.config import Config
        from shutter.core.memory import MemoryGuard, OmlxService, ProcessRssService, default_probe
        from shutter.core.pool import ModelPool
        from shutter.core.registry import Registry

        self._wfd = write_fd
        self._rfd = read_fd
        self._wlock = threading.Lock()
        self.broken = False

        self.config = config or Config.load()
        self.config.ensure_dirs()
        self.probe = probe or default_probe()
        if protected is None:
            protected = []
            if self.config.omlx_url:
                protected.append(OmlxService(self.config.omlx_url, api_key=self.config.omlx_api_key))
            if self.config.protected_process_pattern:
                protected.append(ProcessRssService(self.config.protected_process_pattern,
                                                   self.config.protected_process_min_bytes))
        self.guard = MemoryGuard(self.probe, protected=list(protected),
                                 headroom_bytes=self.config.headroom_bytes,
                                 swap_growth_bytes=self.config.swap_growth_bytes,
                                 max_fraction=self.config.max_fraction)
        self.pool = ModelPool(self.guard, on_event=self._on_pool_event)
        self.cache = ContentCache(self.config.cache_dir, self.config.cache_max_bytes)
        self.registry = Registry(self.config)

        self._cond = threading.Condition()
        self._inbox: collections.deque = collections.deque()   # run / release ops
        self._stop = threading.Event()
        self._stop_reason = ""
        self._cancel = threading.Event()
        self._running_id: Optional[str] = None
        self._pool_quiet = 0
        self._deferred: set[str] = set()

    # ------------------------------------------------------------------ output

    def send(self, obj: dict[str, Any]) -> None:
        try:
            line = json.dumps(_clean(obj), ensure_ascii=False, separators=(",", ":"),
                              allow_nan=False)
        except Exception as e:  # pragma: no cover - _clean makes this unreachable
            line = json.dumps({"ev": "log", "level": "error", "message": f"unencodable event: {e}"})
        data = (line + "\n").encode("utf-8")
        failed: Optional[BaseException] = None
        with self._wlock:
            if self.broken:
                return
            try:
                view = memoryview(data)
                while view:
                    n = os.write(self._wfd, view)
                    view = view[n:]
            except OSError as e:
                self.broken = True
                failed = e
        if failed is not None:  # outside the write lock: request_stop takes _cond
            _log_err(f"protocol stream closed ({failed}); shutting down")
            self.request_stop("stdout closed")

    def log(self, level: str, message: str, **extra: Any) -> None:
        self.send({"ev": "log", "level": level, "message": message, **extra})

    def send_ready(self) -> None:
        from shutter import __version__
        self.send({"ev": "ready", "version": __version__, "pid": os.getpid(), "protocol": PROTOCOL})

    def send_pool(self) -> None:
        st = self.pool.stats()
        self.send({"ev": "pool", "total_bytes": st["total_bytes"], "entries": [
            {"key": e["key"], "size_bytes": e["size_bytes"], "idle_s": round(e["idle_s"], 1),
             "pinned": e["pinned"], "in_use": e.get("in_use", False),
             "residency": e.get("residency", "resident")} for e in st["entries"]]})

    def _on_pool_event(self, event: Any) -> None:
        if self._pool_quiet:
            return
        if getattr(event, "kind", "") in ("model.loaded", "model.unloaded"):
            self.send_pool()

    # ------------------------------------------------------------------ control

    def request_stop(self, reason: str) -> None:
        if not self._stop.is_set():
            self._stop_reason = reason
            self._stop.set()
        self._cancel.set()
        with self._cond:
            self._cond.notify_all()

    def _on_signal(self, signum: int, _frame: Any) -> None:
        # Runs on the main thread between bytecodes: only set flags (no locks held here
        # could be re-entered, request_stop's notify happens in the reader path too).
        if not self._stop.is_set():
            self._stop_reason = f"signal {signum}"
            self._stop.set()
        self._cancel.set()

    # ------------------------------------------------------------------ reader thread

    def reader(self) -> None:
        buf = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = os.read(self._rfd, 65536)
                except InterruptedError:
                    continue
                except OSError as e:
                    _log_err(f"stdin read failed: {e}")
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._handle_line(line)
            if buf.strip() and not self._stop.is_set():
                self._handle_line(buf)
        finally:
            self.request_stop("stdin closed")

    def _handle_line(self, raw: bytes) -> None:
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return
        try:
            msg = json.loads(text)
        except ValueError:
            _log_err(f"ignored a line that is not JSON: {text[:200]!r}")
            self.log("warning", f"worker ignored a line that is not JSON: {text[:200]}")
            return
        if not isinstance(msg, dict):
            self.log("warning", "worker ignored a message that is not a JSON object")
            return
        op = msg.get("op")
        try:
            self._dispatch(op, msg)
        except Exception as e:  # never die on a message
            _log_err(f"op {op!r} failed: {traceback.format_exc()}")
            self.log("error", f"worker could not handle op {op!r}: {type(e).__name__}: {e}")

    def _dispatch(self, op: Any, msg: dict[str, Any]) -> None:
        if op == "hello":
            self.send_ready()
        elif op == "estimate":
            self._op_estimate(msg)
        elif op == "models":
            self._op_models()
        elif op == "stats":
            self.send_pool()
        elif op == "cancel":
            self._op_cancel(str(msg.get("id") or ""))
        elif op in ("run", "release"):
            with self._cond:
                self._inbox.append(_PendingRun(msg) if op == "run" else msg)
                self._cond.notify_all()
        elif op == "shutdown":
            self.request_stop("shutdown requested")
        # unknown ops: ignored silently (forward compatibility)

    # ------------------------------------------------------------------ pure ops

    def _op_estimate(self, msg: dict[str, Any]) -> None:
        from shutter.types import JobRequest
        mid = msg.get("id")
        try:
            req = msg.get("request")
            if not isinstance(req, dict):
                raise ValueError("estimate needs a request object")
            request = JobRequest.from_dict(req)
            adapter = self.registry.adapter_for(request.model)
            est = adapter.estimate(_prepared(adapter, request))
            self.send({"ev": "estimate", "id": mid, "peak": int(est.peak), "steady": int(est.steady),
                       "disk_scratch": int(est.disk_scratch), "note": est.note or ""})
        except Exception as e:
            self.send({"ev": "estimate", "id": mid, "error": _error_text(e)})

    def _op_models(self) -> None:
        rows = []
        for r in self.registry.list():
            row = {k: r.get(k) for k in ("id", "family", "kind", "dir", "installed")}
            if r.get("error"):
                row["error"] = r["error"]
            rows.append(row)
        self.send({"ev": "models", "models": rows})

    def _op_cancel(self, job_id: str) -> None:
        if job_id and job_id == self._running_id:
            self._cancel.set()
            return
        found = False
        with self._cond:
            for item in list(self._inbox):
                if isinstance(item, _PendingRun) and item.id == job_id:
                    self._inbox.remove(item)
                    found = True
                    break
        if found:
            self.send({"ev": "failed", "id": job_id, "error": "cancelled before it started",
                       "cancelled": True})
        # unknown or already finished: nothing to do

    # ------------------------------------------------------------------ main thread

    def _do_release(self, msg: dict[str, Any]) -> None:
        key = msg.get("key")
        before = set(self.pool.keys())
        self._pool_quiet += 1
        try:
            if key is None:
                self.pool.release_all()
            else:
                self.pool.release(str(key))
        finally:
            self._pool_quiet -= 1
        after = set(self.pool.keys())
        keys = sorted(before - after)
        targets = after if key is None else ({str(key)} & after)
        deferred = sorted(k for k in targets if self.pool.in_use(k))
        self._deferred.update(deferred)
        ev: dict[str, Any] = {"ev": "released", "keys": keys}
        if deferred:
            ev["deferred"] = deferred
        self.send(ev)
        self.send_pool()

    def _take_releases(self) -> list[dict[str, Any]]:
        with self._cond:
            rel = [m for m in self._inbox if isinstance(m, dict)]
            for m in rel:
                self._inbox.remove(m)
        return rel

    def serve(self) -> int:
        self.send_ready()
        while True:
            item = None
            with self._cond:
                if not self._inbox and not self._stop.is_set():
                    self._cond.wait(IDLE_TICK_S)
                if self._stop.is_set():
                    break
                if self._inbox:
                    item = self._inbox.popleft()
            if item is None:
                self._idle_tick()
                continue
            try:
                if isinstance(item, _PendingRun):
                    self._run(item)
                else:
                    self._do_release(item)
            except Exception:  # the loop must never die
                _log_err(traceback.format_exc())
        # stopping: tell the daemon about runs it will never get, then free everything
        with self._cond:
            pending = [i for i in self._inbox if isinstance(i, _PendingRun)]
            self._inbox.clear()
        for p in pending:
            self.send({"ev": "failed", "id": p.id, "error": "worker shutting down", "cancelled": True})
        self._pool_quiet += 1
        try:
            self.pool.release_all()
        except Exception:
            _log_err(traceback.format_exc())
        return 0

    def _idle_tick(self) -> None:
        ttl = self.config.idle_unload_s
        if ttl and ttl > 0:
            try:
                self.pool.evict_idle(ttl)
            except Exception:
                _log_err(traceback.format_exc())

    # ------------------------------------------------------------------ run

    def _run(self, item: _PendingRun) -> None:
        from shutter.adapters.base import RunContext
        from shutter.types import (LEGACY_INPUT_KEYS, REQUEST_FIELDS, Cancelled, Job, JobRequest, JobState,
                                   MemoryRefused)

        msg = item.msg
        jid = item.id
        t0 = time.perf_counter()
        self._cancel.clear()
        if self._stop.is_set():
            self._cancel.set()
        self._running_id = jid
        acquired: list[str] = []
        prog = {"last_sent": -1e9, "pending": None, "value": 0.0, "stage": ""}

        def flush_progress(force: bool = False) -> None:
            p = prog["pending"]
            if p is None:
                return
            now = time.monotonic()
            if force or now - prog["last_sent"] >= PROGRESS_MIN_INTERVAL_S:
                prog["pending"] = None
                prog["last_sent"] = now
                ev = {"ev": "progress", "id": jid, "progress": p[0], "stage": p[1]}
                step = parse_step(p[1])
                if step is not None:
                    ev["step"], ev["steps"] = step
                self.send(ev)

        def service() -> None:
            for rel in self._take_releases():
                try:
                    self._do_release(rel)
                except Exception:
                    _log_err(traceback.format_exc())

        def report(progress: float, stage: str = "") -> None:
            try:
                p = min(1.0, max(0.0, float(progress)))
            except (TypeError, ValueError):
                p = prog["value"]
            p = max(p, prog["value"])  # progress never goes backwards
            prog["value"] = p
            if stage:
                prog["stage"] = str(stage)
            prog["pending"] = (round(p, 4), prog["stage"])
            flush_progress()
            service()

        def check_cancelled() -> None:
            flush_progress()
            service()
            if self._cancel.is_set():
                raise Cancelled("worker shutting down" if self._stop.is_set() else "cancelled")

        try:
            job_d = msg.get("job")
            if not isinstance(job_d, dict):
                raise ValueError("run needs a job object")
            if isinstance(job_d.get("request"), dict):
                request = JobRequest.from_dict(job_d["request"])
            else:  # tolerate a bare request in place of a Job dict
                request = JobRequest.from_dict({k: v for k, v in job_d.items()
                                                if k in REQUEST_FIELDS + LEGACY_INPUT_KEYS})
            if not request.model:
                raise ValueError("job.request.model is required")
            job = Job(request=request, id=str(job_d.get("id") or jid or ""))
            if not jid:
                jid = job.id
                self._running_id = jid
            job.state = JobState.RUNNING
            job.started_at = time.time()
            job.created_at = float(job_d.get("created_at") or job.started_at)

            adapter = self.registry.adapter_for(request.model)
            work_dir = Path(os.path.expanduser(str(msg.get("work_dir") or "")) or
                            os.path.join(self.config.work_dir, job.id))
            work_dir.mkdir(parents=True, exist_ok=True)
            if request.output:
                request.output = os.path.abspath(os.path.expanduser(request.output))
            else:
                ext = getattr(adapter, "extension", None) or ("png" if getattr(adapter, "kind", "video") == "image" else "mp4")
                request.output = os.path.join(self.config.output_dir, f"{job.id}.{ext}")
            Path(request.output).parent.mkdir(parents=True, exist_ok=True)

            keys_fn = getattr(adapter, "pool_keys", None)
            if callable(keys_fn):
                try:
                    for k in list(keys_fn(request) or []):
                        self.pool.acquire(k)
                        acquired.append(k)
                except Exception:
                    _log_err(f"pool_keys failed: {traceback.format_exc()}")

            mx = _mlx()
            if mx is not None:
                try:
                    getattr(mx, "reset_peak_memory", lambda: None)()
                except Exception:
                    pass

            job.request = _prepared(adapter, request)
            ctx = RunContext(
                models=self.pool, cache=self.cache, work_dir=work_dir, report=report,
                check_cancelled=check_cancelled, memory_sample=lambda: self.probe.sample().to_dict(),
                settings={"job_id": job.id, "output": request.output, "home": self.config.home,
                          "headroom_bytes": self.guard.headroom_bytes, "sleep": None},
            )
            result = adapter.run(job, ctx) or {}
            if self._cancel.is_set():  # adapter swallowed the Cancelled
                raise Cancelled("worker shutting down" if self._stop.is_set() else "cancelled")
            result = dict(result)
            result.setdefault("output", request.output)
            result.setdefault("seconds", round(time.perf_counter() - t0, 3))
            result["peak_bytes"] = self._peak_bytes(adapter, result)
            flush_progress(force=True)
            self._cleanup_work_dir(work_dir, request.output, result)
            self.send({"ev": "done", "id": jid, "result": result})
        except Cancelled as e:
            flush_progress(force=True)
            self.send({"ev": "failed", "id": jid, "error": str(e) or "cancelled", "cancelled": True})
        except MemoryRefused as e:
            flush_progress(force=True)
            self.send({"ev": "failed", "id": jid, "error": e.reason, "cancelled": False,
                       "memory_refused": True})
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                self.request_stop(type(e).__name__)
            flush_progress(force=True)
            tb = traceback.format_exc()
            _log_err(tb)
            self.log("error", "\n".join(tb.rstrip().splitlines()[-12:]), id=jid)
            self.send({"ev": "failed", "id": jid, "error": _error_text(e), "cancelled": False})
        finally:
            self._running_id = None
            self._pool_quiet += 1
            try:
                for k in acquired:
                    self.pool.relinquish(k)
                if self.config.idle_unload_s <= 0:
                    self.pool.evict_idle(0)
            finally:
                self._pool_quiet -= 1
            gone = sorted(k for k in self._deferred if k not in self.pool)
            self._deferred.clear()
            if gone:
                self.send({"ev": "released", "keys": gone})
            service()
            self.send_pool()

    def _peak_bytes(self, adapter: Any, result: dict[str, Any]) -> int:
        for v in (result.get("peak_bytes"), result.get("peak_memory_bytes"),
                  getattr(adapter, "peak_memory_bytes", None)):
            if callable(v):
                try:
                    v = v()
                except Exception:
                    v = None
            try:
                if v is not None and int(v) > 0:
                    return int(v)
            except (TypeError, ValueError):
                pass
        mx = _mlx()
        if mx is not None:
            try:
                fn = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", None),
                                                                      "get_peak_memory", None)
                v = int(fn()) if fn else 0
                if v > 0:
                    return v
            except Exception:
                pass
        return _process_peak_rss()

    def _cleanup_work_dir(self, work_dir: Path, output: str, result: dict[str, Any]) -> None:
        """Like the reference engine: remove the scratch dir after success unless
        keep_work_dirs is set or the output (or any path in the result) lives inside it."""
        if self.config.keep_work_dirs:
            return
        try:
            wd = str(work_dir.resolve()) + os.sep
            paths = [output] + [v for v in result.values() if isinstance(v, str)]
            if not any(str(Path(p).resolve()).startswith(wd) for p in paths if p and os.sep in p):
                shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            _log_err(traceback.format_exc())


def _prepared(adapter: Any, request: Any) -> Any:
    """adapter.prepare(request) (core fields mirrored into params, see adapters.base); the request
    itself when the adapter has no prepare."""
    fn = getattr(adapter, "prepare", None)
    return fn(request) if callable(fn) else request


def _error_text(e: BaseException) -> str:
    from shutter.core.registry import AdapterUnavailable, UnknownModel
    if isinstance(e, (UnknownModel, AdapterUnavailable)):
        return str(e)
    text = str(e)
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


# ---------------------------------------------------------------------------- entry


def main() -> int:
    from shutter.core.config import add_private_bin
    add_private_bin()
    stdio = _Stdio()
    stdio.capture()
    assert stdio.out_fd is not None and stdio.in_fd is not None
    try:
        worker = Worker(stdio.out_fd, stdio.in_fd)
    except Exception as e:
        # Cannot even start (bad config...): say why on both channels and exit non-zero.
        tb = traceback.format_exc()
        _log_err(tb)
        line = json.dumps({"ev": "log", "level": "error",
                           "message": f"worker failed to start: {type(e).__name__}: {e}"}) + "\n"
        try:
            os.write(stdio.out_fd, line.encode())
        except OSError:
            pass
        return 1
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, worker._on_signal)
        except (ValueError, OSError):  # pragma: no cover
            pass
    t = threading.Thread(target=worker.reader, name="shutter-worker-reader", daemon=True)
    t.start()
    code = worker.serve()
    try:
        sys.stderr.flush()
    except Exception:
        pass
    return code


if __name__ == "__main__":
    sys.exit(main())
