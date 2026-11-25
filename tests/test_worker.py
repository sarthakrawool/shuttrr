"""Drive the real model worker (`python3 -m shutter.worker`) over pipes, as shutterd does.
Protocol: docs/WORKER_PROTOCOL.md."""

import json
import os
import queue
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

NOISY_ADAPTER = textwrap.dedent('''
    import os, subprocess, sys
    from shutter.adapters.dummy import DummyAdapter

    class NoisyAdapter(DummyAdapter):
        family = "noisy"

        def run(self, job, ctx):
            print("hello from print")
            sys.stdout.write("partial line without newline")
            sys.stdout.flush()
            os.write(1, b"raw write to fd 1\\n")
            subprocess.run(["echo", "child process output"])
            return super().run(job, ctx)
''')


class WorkerProc:
    def __init__(self, home: Path, extra_path: Path):
        env = dict(os.environ)
        env["SHUTTER_HOME"] = str(home)
        env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(extra_path), env.get("PYTHONPATH", "")])
        env["PYTHONUNBUFFERED"] = "1"
        self.stderr_path = home / "worker.stderr"
        self._stderr = open(self.stderr_path, "wb")
        self.p = subprocess.Popen([sys.executable, "-m", "shutter.worker"], cwd=str(home),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self._stderr, env=env)
        self.lines: list[str] = []       # every raw stdout line, for validation
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.seen: list[dict] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in self.p.stdout:
            line = raw.decode("utf-8")
            self.lines.append(line)
            try:
                self.q.put(json.loads(line))
            except ValueError:
                self.q.put({"ev": "__corrupt__", "line": line})

    def send(self, obj=None, raw=None):
        data = raw if raw is not None else json.dumps(obj)
        self.p.stdin.write((data + "\n").encode())
        self.p.stdin.flush()

    def expect(self, pred, timeout=10.0):
        """Return the next message matching pred (messages before it are kept in seen)."""
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise AssertionError(f"no matching message; seen: {self.seen[-20:]}\n"
                                     f"stderr: {self.stderr_path.read_text()[-2000:]}")
            try:
                m = self.q.get(timeout=left)
            except queue.Empty:
                continue
            assert m.get("ev") != "__corrupt__", f"stdout corrupted: {m['line']!r}"
            self.seen.append(m)
            if pred(m):
                return m

    def ev(self, name, timeout=10.0, **match):
        return self.expect(lambda m: m.get("ev") == name and all(m.get(k) == v for k, v in match.items()),
                           timeout)

    def collect_job(self, jid, timeout=10.0):
        """All messages for one job up to its done/failed."""
        out = []
        while True:
            m = self.expect(lambda m: m.get("id") == jid, timeout)
            out.append(m)
            if m["ev"] in ("done", "failed"):
                return out

    def wait(self, timeout=10.0):
        try:
            return self.p.wait(timeout)
        finally:
            self._stderr.close()

    def close(self):
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait(5)
        self._stderr.close()

    def assert_clean_stdout(self):
        for line in self.lines:
            assert line.endswith("\n")
            m = json.loads(line)
            assert isinstance(m, dict) and "ev" in m, line


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    h.mkdir()
    (h / "config.json").write_text(json.dumps({
        "omlx_url": None, "protected_process_pattern": None, "busy_process_pattern": None,
        "headroom_gb": 0, "idle_unload_s": 900,
        "models": {"noisy": {"adapter": "noisy_adapter:NoisyAdapter"}},
    }))
    (tmp_path / "noisy_adapter.py").write_text(NOISY_ADAPTER)
    return h


@pytest.fixture
def worker(home):
    w = WorkerProc(home, home.parent)
    w.ev("ready")  # sent proactively at start
    yield w
    w.close()


def run_msg(jid, home, model="dummy", **params):
    p = {"steps": 5, "step_s": 0.02, "component_gb": 0.001, "peak_gb": 0.001}
    p.update(params)
    return {"op": "run", "id": jid, "work_dir": str(home / "work" / jid),
            "job": {"id": jid, "state": "queued", "created_at": time.time(),
                    "request": {"model": model, "prompt": "hi", "params": p}}}


def test_ready_on_start_and_after_hello(worker):
    worker.send({"op": "hello"})
    r = worker.ev("ready")
    assert r["protocol"] == 1 and r["pid"] == worker.p.pid and r["version"]


def test_models_lists_dummy(worker):
    worker.send({"op": "models"})
    rows = {r["id"]: r for r in worker.ev("models")["models"]}
    assert rows["dummy"]["family"] == "dummy" and rows["dummy"]["installed"] is True
    assert set(rows["dummy"]) >= {"id", "family", "kind", "dir", "installed"}
    assert "noisy" in rows


def test_estimate(worker):
    worker.send({"op": "estimate", "id": "e1", "request": {"model": "dummy", "params": {"peak_gb": 2}}})
    e = worker.ev("estimate", id="e1")
    assert e["peak"] == 2 * 1024**3 and e["steady"] > 0 and "disk_scratch" in e and "note" in e
    worker.send({"op": "estimate", "id": "e2", "request": {"model": "nope"}})
    e = worker.ev("estimate", id="e2")
    assert "unknown model" in e["error"]


def test_run_to_done(worker, home):
    worker.send(run_msg("j1", home, steps=12, step_s=0.03))
    msgs = worker.collect_job("j1")
    done = msgs[-1]
    assert done["ev"] == "done"
    prog = [m["progress"] for m in msgs if m["ev"] == "progress"]
    assert prog and prog == sorted(prog) and prog[-1] == 1.0
    assert len(prog) <= 0.36 * 4 + 3   # throttled to ~4/s
    res = done["result"]
    assert Path(res["output"]).is_file()
    assert json.loads(Path(res["output"]).read_text())["steps"] == 12
    assert res["seconds"] > 0 and isinstance(res["peak_bytes"], int) and res["peak_bytes"] > 0
    assert not (home / "work" / "j1").exists()       # scratch removed after success
    pool = worker.ev("pool")
    assert [e["key"] for e in pool["entries"]] == ["dummy/dit"]
    worker.assert_clean_stdout()


def test_cancel_mid_run(worker, home):
    worker.send(run_msg("j2", home, steps=500, step_s=0.01))
    worker.ev("progress", id="j2")
    worker.send({"op": "stats"})             # answered while the job runs
    worker.ev("pool")
    worker.send({"op": "cancel", "id": "j2"})
    f = worker.expect(lambda m: m.get("id") == "j2" and m["ev"] in ("done", "failed"))
    assert f["ev"] == "failed" and f["cancelled"] is True


def test_error_param_fails_with_message(worker, home):
    worker.send(run_msg("j3", home, fail="boom at step", fail_at=1))
    msgs = worker.collect_job("j3")
    f = msgs[-1]
    assert f["ev"] == "failed" and f["cancelled"] is False and "boom at step" in f["error"]
    logs = [m for m in msgs if m["ev"] == "log"]
    assert logs and "Traceback" not in f["error"] and "boom at step" in logs[-1]["message"]
    # the worker is still alive and serving
    worker.send(run_msg("j3b", home))
    assert worker.collect_job("j3b")[-1]["ev"] == "done"


def test_stray_prints_do_not_corrupt_stdout(worker, home):
    worker.send(run_msg("j4", home, model="noisy"))
    assert worker.collect_job("j4")[-1]["ev"] == "done"
    worker.send({"op": "shutdown"})
    assert worker.wait() == 0
    time.sleep(0.05)
    worker.assert_clean_stdout()
    err = worker.stderr_path.read_text()
    for s in ("hello from print", "partial line without newline", "raw write to fd 1",
              "child process output"):
        assert s in err


def test_release_drops_warm_component(worker, home):
    worker.send(run_msg("j5", home))
    assert worker.collect_job("j5")[-1]["ev"] == "done"
    worker.ev("pool")
    worker.send({"op": "release", "key": "dummy/dit"})
    r = worker.ev("released")
    assert r["keys"] == ["dummy/dit"]
    pool = worker.ev("pool")
    assert pool["entries"] == [] and pool["total_bytes"] == 0


def test_release_during_run_waits_for_job_end(worker, home):
    worker.send(run_msg("j6", home, steps=30, step_s=0.01))
    worker.expect(lambda m: m["ev"] == "pool" and m["entries"])     # component loaded
    worker.send({"op": "release", "key": None})
    r = worker.ev("released")
    assert r["keys"] == [] and r["deferred"] == ["dummy/dit"]
    assert worker.expect(lambda m: m.get("id") == "j6" and m["ev"] in ("done", "failed"))["ev"] == "done"
    assert worker.ev("released")["keys"] == ["dummy/dit"]
    assert worker.ev("pool")["entries"] == []


def test_bad_json_and_unknown_op_are_ignored(worker):
    worker.send(raw="{not json")
    worker.send(raw="[1, 2]")
    worker.send({"op": "frobnicate", "x": 1})
    worker.send({"no_op": True})
    worker.send({"op": "stats"})
    pool = worker.ev("pool")
    assert pool["entries"] == []
    assert worker.p.poll() is None
    worker.assert_clean_stdout()


def test_shutdown_exits_zero(worker):
    worker.send({"op": "shutdown"})
    assert worker.wait() == 0


def test_eof_exits_zero(worker):
    worker.p.stdin.close()
    assert worker.wait() == 0


def test_sigterm_mid_run_exits_zero(worker, home):
    worker.send(run_msg("j7", home, steps=500, step_s=0.01))
    worker.ev("progress", id="j7")
    worker.p.send_signal(signal.SIGTERM)
    assert worker.wait() == 0
    time.sleep(0.05)
    f = worker.expect(lambda m: m.get("id") == "j7" and m["ev"] in ("done", "failed"))
    assert f["ev"] == "failed" and f["cancelled"] is True
