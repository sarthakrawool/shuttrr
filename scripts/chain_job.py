#!/usr/bin/env python3
"""Run Shutter jobs for scripts/mac_test_chain.sh and record what happened. Standard library + shutter only (no web
packages: the LongCat venv may not have starlette/uvicorn).

    chain_job.py --via daemon --url http://127.0.0.1:8494 --jobs jobs.json --results DIR
    chain_job.py --via worker --python PY --home SHUTTER_HOME --jobs jobs.json --results DIR \
                 [--lock-dirs DIR1:DIR2] [--swap-max-mb 300] [--min-level 30]

jobs.json: [{"name": "T1", "request": {JobRequest...}, "warmup": false}, ...], run in order in ONE daemon/worker
session (so a second job sees the first one's warm weights and caches). For each job DIR/<name>.job.json gets:
name, via, request, state ("done" | "failed" | "cancelled" | "killed"), result, error, wall_s, machine peak
(used = total - available, sampled every 2 s by this helper), the daemon's own peak_memory when available, swap
at start / peak, the lowest memory level seen (macOS kern.memorystatus_level).

--via daemon: POST /v1/jobs, poll GET /v1/jobs/<id>. The daemon owns the GPU locks, the memory guard and the
              swap watchdog (its config sets swap_growth_mb).
--via worker: start `PY -m shutter.worker` (SHUTTER_HOME, PYTHONPATH from the environment), take the mkdir GPU
              locks like ~/longcat-mlx/speed/gpulock.sh, send `run` ops over the JSON-lines protocol, and act as
              the watchdog: kill the worker's process group if swap grows by more than --swap-max-mb or the memory
              level stays under --min-level for two checks.
Exit 0 when every non-warmup job is done, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

TERMINAL = ("done", "failed", "cancelled")


def log(msg: str) -> None:
    sys.stderr.write(f"[chain_job {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


# ----------------------------------------------------------------------------- machine sampling


def swap_mb() -> Optional[float]:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
            m = re.search(r"used = ([0-9.]+)M", out)
            return float(m.group(1)) if m else None
        with open("/proc/meminfo") as f:
            d = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
        return (d.get("SwapTotal", 0) - d.get("SwapFree", 0)) / 1024.0
    except Exception:
        return None


def mem_level() -> Optional[int]:
    if sys.platform != "darwin":
        return None
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True,
                                  text=True).stdout.strip())
    except Exception:
        return None


def parse_vm_stat(text: str) -> Optional[int]:
    """App memory in bytes from `vm_stat` output: anonymous + wired + compressor pages. This is what a job really
    holds; it leaves out file-backed pages (the file cache, memory-mapped weights), which macOS gives back on
    demand. Same definition as ~/ImaginationStudio/tools/memsample.sh, so the numbers compare directly."""
    m = re.search(r"page size of (\d+) bytes", text)
    page = int(m.group(1)) if m else 16384

    def pages(label: str) -> Optional[int]:
        mm = re.search(r"^" + re.escape(label) + r":\s+(\d+)\.?\s*$", text, re.M)
        return int(mm.group(1)) if mm else None

    parts = [pages("Anonymous pages"), pages("Pages wired down"), pages("Pages occupied by compressor")]
    if any(v is None for v in parts):
        return None
    return sum(parts) * page


def used_bytes() -> Optional[int]:
    """Memory in use on the machine. macOS: app memory (see parse_vm_stat); elsewhere total - available."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            v = parse_vm_stat(out)
            if v is not None:
                return v
        from shutter.core.memory import default_probe

        s = default_probe().sample()
        return int(s.total - s.available)
    except Exception:
        return None


class Sampler(threading.Thread):
    """Every `interval` s: machine used bytes, swap, memory level; optional kill rule."""

    def __init__(self, interval: float = 2.0, swap_max_mb: Optional[float] = None, min_level: Optional[int] = None,
                 on_kill=None):
        super().__init__(daemon=True)
        self.interval, self.swap_max_mb, self.min_level, self.on_kill = interval, swap_max_mb, min_level, on_kill
        self.stop_ev = threading.Event()
        self.swap0 = swap_mb()
        self.swap_peak = self.swap0
        self.used_start = used_bytes()      # what the machine already used before this job
        self.used_peak = self.used_start
        self.level_min = mem_level()
        self.killed: Optional[str] = None
        self._low = 0

    def run(self) -> None:
        while not self.stop_ev.wait(self.interval):
            s, u, lv = swap_mb(), used_bytes(), mem_level()
            if s is not None:
                self.swap_peak = s if self.swap_peak is None else max(self.swap_peak, s)
            if u is not None:
                self.used_peak = u if self.used_peak is None else max(self.used_peak, u)
            if lv is not None:
                self.level_min = lv if self.level_min is None else min(self.level_min, lv)
            if self.on_kill is None or self.killed:
                continue
            if self.swap_max_mb is not None and s is not None and self.swap0 is not None \
                    and s - self.swap0 > self.swap_max_mb:
                self.killed = f"swap grew {s - self.swap0:.0f} MB (> {self.swap_max_mb:.0f})"
            if self.min_level is not None and lv is not None:
                self._low = self._low + 1 if lv < self.min_level else 0
                if self._low >= 2:
                    self.killed = f"memory level {lv} < {self.min_level} twice"
            if self.killed:
                log(f"WATCHDOG: {self.killed}: stopping the job")
                self.on_kill(self.killed)

    def stop(self) -> dict[str, Any]:
        self.stop_ev.set()
        added = (None if self.used_start is None or self.used_peak is None else max(0, self.used_peak - self.used_start))
        return {"machine_peak_used_bytes": self.used_peak, "machine_start_used_bytes": self.used_start,
                "machine_added_bytes": added, "swap_start_mb": self.swap0, "swap_peak_mb": self.swap_peak,
                "swap_growth_mb": (None if self.swap0 is None or self.swap_peak is None
                                   else round(self.swap_peak - self.swap0, 1)),
                "memory_level_min": self.level_min, "watchdog": self.killed}


# ----------------------------------------------------------------------------- daemon


def http(method: str, url: str, body: Optional[dict] = None, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {url}: HTTP {e.code}: {e.read()[:500]!r}") from None


def run_daemon(url: str, job: dict, timeout_s: float) -> dict:
    sampler = Sampler()
    sampler.start()
    t0 = time.time()
    rec: dict[str, Any] = {"via": "daemon"}
    try:
        j = http("POST", f"{url}/v1/jobs", job["request"])
        jid = j["id"]
        rec["job_id"] = jid
        last = ""
        while True:
            time.sleep(2.0)
            j = http("GET", f"{url}/v1/jobs/{jid}")
            msg = f"{j.get('state')} {j.get('progress', 0):.2f} {j.get('stage', '')} {j.get('message', '')}"
            if msg != last:
                log(f"{job['name']}: {msg}")
                last = msg
            if j.get("state") in TERMINAL:
                break
            if time.time() - t0 > timeout_s:
                http("DELETE", f"{url}/v1/jobs/{jid}")
                rec["error"] = f"timed out after {timeout_s:.0f} s (cancelled)"
                break
        rec.update(state=j.get("state"), result=j.get("result"), error=rec.get("error") or j.get("message"),
                   job=j)
        pm = (j.get("result") or {}).get("peak_memory") or {}
        rec["daemon_peak_used_bytes"] = pm.get("used_bytes")
    except Exception as e:
        rec.update(state="failed", error=f"{type(e).__name__}: {e}")
    rec["wall_s"] = round(time.time() - t0, 1)
    rec.update(sampler.stop())
    return rec


# ----------------------------------------------------------------------------- worker


class DirLocks:
    """mkdir locks in order, like gpulock.sh (waits 15 s between tries, owner files, removed on exit)."""

    def __init__(self, dirs: list[str], owner: str = "shutter-test"):
        self.dirs = [os.path.expanduser(d) for d in dirs if d]
        self.owner = owner
        self.held: list[str] = []

    def __enter__(self):
        for d in self.dirs:
            Path(d).parent.mkdir(parents=True, exist_ok=True)
            waited = 0
            while True:
                try:
                    os.mkdir(d)
                    break
                except FileExistsError:
                    if waited % 300 == 0:
                        owner = ""
                        for f in ("owner", "owner_speed"):
                            try:
                                owner = Path(d, f).read_text().strip() or owner
                            except OSError:
                                pass
                        log(f"waiting for lock {d} (owner {owner or '?'})")
                    time.sleep(15)
                    waited += 15
            self.held.append(d)
            name = "owner_speed" if d.endswith(".render.lock") else "owner"
            Path(d, name).write_text(self.owner + "\n")
            if name == "owner_speed":
                Path(d, "pid").write_text(f"{os.getpid()}\n")
        return self

    def __exit__(self, *exc):
        import shutil

        for d in reversed(self.held):
            shutil.rmtree(d, ignore_errors=True)
        self.held.clear()


class WorkerSession:
    def __init__(self, python: str):
        env = dict(os.environ)
        self.p = subprocess.Popen([python, "-m", "shutter.worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  env=env, start_new_session=True)
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()
        self.wait_for(lambda e: e.get("ev") == "ready", 300, "worker did not start (no ready event)")

    def _read(self):
        for line in self.p.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            with self.cond:
                self.events.append(ev)
                self.cond.notify_all()
        with self.cond:
            self.events.append({"ev": "_eof"})
            self.cond.notify_all()

    def send(self, obj: dict) -> None:
        self.p.stdin.write((json.dumps(obj) + "\n").encode())
        self.p.stdin.flush()

    def wait_for(self, pred, timeout: float, what: str, start: int = 0):
        deadline = time.time() + timeout
        i = start
        with self.cond:
            while True:
                while i < len(self.events):
                    ev = self.events[i]
                    i += 1
                    if ev.get("ev") == "_eof":
                        raise RuntimeError(f"worker exited ({what})")
                    if pred(ev):
                        return ev, i
                left = deadline - time.time()
                if left <= 0:
                    raise TimeoutError(what)
                self.cond.wait(min(left, 5.0))

    def kill(self) -> None:
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.send({"op": "shutdown"})
            self.p.wait(30)
        except Exception:
            self.kill()


def run_worker_job(sess: WorkerSession, job: dict, timeout_s: float, swap_max_mb, min_level) -> dict:
    jid = f"{job['name']}-{int(time.time())}"
    rec: dict[str, Any] = {"via": "worker", "job_id": jid}
    killed = {}

    def on_kill(reason):
        killed["reason"] = reason
        sess.kill()

    sampler = Sampler(swap_max_mb=swap_max_mb, min_level=min_level, on_kill=on_kill)
    sampler.start()
    t0 = time.time()
    start = len(sess.events)
    try:
        sess.send({"op": "estimate", "id": jid, "request": job["request"]})
        est, _ = sess.wait_for(lambda e: e.get("ev") == "estimate" and e.get("id") == jid, 120, "estimate", start)
        rec["estimate"] = est
        work = os.path.join(os.environ.get("SHUTTER_HOME", "."), "work", jid)
        sess.send({"op": "run", "id": jid, "job": {"id": jid, "request": job["request"]}, "work_dir": work})
        last_p = -1.0
        i = start
        while True:
            ev, i = sess.wait_for(lambda e: e.get("id") == jid and e.get("ev") in ("progress", "done", "failed", "log"),
                                  timeout_s, "job", i)
            if ev["ev"] == "progress":
                if ev["progress"] - last_p >= 0.02:
                    log(f"{job['name']}: {ev['progress']:.2f} {ev.get('stage', '')}")
                    last_p = ev["progress"]
                continue
            if ev["ev"] == "log":
                log(f"{job['name']}: worker log: {ev.get('message', '')[:300]}")
                continue
            break
        if ev["ev"] == "done":
            rec.update(state="done", result=ev.get("result"))
        else:
            rec.update(state="cancelled" if ev.get("cancelled") else "failed", error=ev.get("error"),
                       memory_refused=ev.get("memory_refused", False))
    except Exception as e:
        rec.update(state="killed" if killed else "failed",
                   error=(f"stopped to protect the Mac: {killed['reason']}" if killed else f"{type(e).__name__}: {e}"))
    rec["wall_s"] = round(time.time() - t0, 1)
    rec.update(sampler.stop())
    return rec


# ----------------------------------------------------------------------------- chain bookkeeping


def cmd_config(a) -> int:
    """Write <home>/config.json for the test chain and validate it with Shutter's own loaders."""
    home = Path(os.path.expanduser(a.home))
    home.mkdir(parents=True, exist_ok=True)
    sdir = os.path.abspath(os.path.expanduser(a.shutter_dir))
    cfg: dict[str, Any] = {
        "port": int(a.port),
        "gpu_lock_dirs": [os.path.expanduser(d) for d in a.lock_dirs.split(":") if d],
        "worker_cmd": ["/usr/bin/env", f"SHUTTER_HOME={home}", f"PYTHONPATH={sdir}", "PYTHONUNBUFFERED=1",
                       os.path.expanduser(a.python), "-m", "shutter.worker"],
        "worker_cwd": sdir,
        "swap_growth_mb": float(a.swap_max_mb),
        "idle_unload_s": 3600,
        "keep_work_dirs": False,
        "dummy_model": True,
        "models": {},
    }
    if a.model == "longcat":
        cfg["models"][a.model_id] = {
            "adapter": "longcat-avatar", "dir": os.path.expanduser(a.weights),
            "options": {"code_dir": os.path.expanduser(a.code_dir), "weights_dir": os.path.expanduser(a.weights),
                        "measured_path": str(home / "measured.json")}}
    path = home / "config.json"
    path.write_text(json.dumps(cfg, indent=1))
    from shutter.core.config import Config
    from shutter.core.registry import Registry

    os.environ["SHUTTER_HOME"] = str(home)
    c = Config.load()
    assert os.path.realpath(c.home) == os.path.realpath(str(home)), (c.home, home)
    reg = Registry(c)
    mid = a.model_id if a.model == "longcat" else "dummy"
    spec = reg.spec(mid)
    rows = {r["id"]: r for r in reg.list()}
    print(json.dumps({"config": str(path), "model": mid, "adapter": spec["adapter"], "dir": spec["dir"],
                      "installed": rows.get(mid, {}).get("installed"), "worker_cmd": cfg["worker_cmd"],
                      "gpu_lock_dirs": cfg["gpu_lock_dirs"]}))
    return 0


def cmd_request(a) -> int:
    params: dict[str, Any] = {"mode": a.mode}
    if a.rung:
        params["rung"] = a.rung
        params["efficiency"] = {"resident": 0.0, "split": 0.25, "staged": 0.5, "stream_wide": 0.75,
                                "stream_tight": 1.0}[a.rung]
    out_dir = Path(a.out_dir)
    params["save_latents"] = str(out_dir / "latents")
    if a.model == "dummy":
        params.update(steps=4, step_s=0.05, peak_gb=2.0, component_gb=0.2)
        if a.step in os.environ.get("CHAIN_DUMMY_FAIL", "").split(","):   # orchestration tests: make this job fail
            params.update(fail="injected failure", fail_at=1)
        req = {"model": "dummy", "prompt": "test chain", "seed": 42, "duration_s": 3.72,
               "output": str(out_dir / f"{a.step}.json"), "params": params}
    else:
        prompt = Path(a.prompt_file).read_text().strip()      # like scripts/generate.py
        params.update(size=a.size, upscale_to="off", zoom_cancel="off")
        req = {"model": a.model_id, "prompt": prompt, "image": a.image, "audio": a.audio, "duration_s": 3.72,
               "seed": 42, "framing": "full", "output": str(out_dir / f"{a.step}.mp4"), "params": params}
    Path(a.out).write_text(json.dumps(req, indent=1))
    return 0


def _cmp_mod():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import compare_clips  # noqa: E402

    return compare_clips


def cmd_finish(a) -> int:
    """<step>.json from the job record + comparisons + swap numbers."""
    rec = json.loads(Path(a.job).read_text()) if Path(a.job).exists() else {"state": "failed", "error": "no job record"}
    res = rec.get("result") or {}
    cc = _cmp_mod()
    compares: dict[str, Any] = {}
    lat = None
    if res.get("latents_dir"):
        cand = Path(res["latents_dir"]) / "latents_denorm.npy"
        lat = str(cand) if cand.exists() else None
    out_mp4 = res.get("output") if str(res.get("output", "")).endswith(".mp4") else None
    for label, ref, kind in (("ref_latents", a.ref_latents, "lat"), ("ref_mp4", a.ref_mp4, "mp4"),
                             ("base_latents", a.base_latents, "lat"), ("base_mp4", a.base_mp4, "mp4")):
        if not ref:
            continue
        mine = lat if kind == "lat" else out_mp4
        if not Path(ref).exists():
            compares[label] = {"skipped": f"reference {ref} not found"}
            continue
        if not mine:
            compares[label] = {"skipped": "this run produced no " + ("latents" if kind == "lat" else "mp4")}
            continue
        try:
            compares[label] = cc.compare_latents(mine, ref) if kind == "lat" else cc.compare_clips(mine, ref)
        except Exception as e:
            compares[label] = {"error": f"{type(e).__name__}: {e}", "identical": False}
    done = [c for c in compares.values() if "identical" in c]
    identical = all(c["identical"] for c in done) if done else None
    lat_done = [c for k, c in compares.items() if k.endswith("latents") and "identical" in c]
    sb, sa = (float(a.swap_before) if a.swap_before not in ("", None) else None,
              float(a.swap_after) if a.swap_after not in ("", None) else None)
    step = {"step": a.step, "title": a.title, "state": rec.get("state"), "error": rec.get("error"),
            "via": rec.get("via"), "mode": res.get("mode"), "rung": res.get("rung"),
            "stepped_down_from": res.get("stepped_down_from"), "seconds": res.get("seconds"),
            "wall_s": rec.get("wall_s"), "seconds_per_step": res.get("seconds_per_step"),
            "step_seconds": res.get("step_seconds"), "stage_seconds": res.get("stage_seconds"),
            "stage_peak_bytes": res.get("stage_peak_bytes"), "peak_bytes": res.get("peak_bytes"),
            "estimate_peak_bytes": res.get("estimate_peak_bytes"), "size_key": res.get("size_key"),
            "machine_peak_used_bytes": rec.get("daemon_peak_used_bytes") or rec.get("machine_peak_used_bytes"),
            "helper_machine_peak_used_bytes": rec.get("machine_peak_used_bytes"),
            "machine_start_used_bytes": rec.get("machine_start_used_bytes"),
            "machine_added_bytes": rec.get("machine_added_bytes"),
            "stream": res.get("stream"), "dit_releases": res.get("dit_releases"), "dit_reloads": res.get("dit_reloads"),
            "compare": compares, "identical": identical,
            "latents_identical": (all(c["identical"] for c in lat_done) if lat_done else None),
            "must_match": bool(a.must_match), "swap_before_mb": sb, "swap_after_mb": sa,
            "swap_delta_mb": None if sb is None or sa is None else round(sa - sb, 1),
            "output": res.get("output"), "result": res, "job": rec}
    # ERROR = the job itself did not finish (the chain runs the step again next time); FAIL = it finished but
    # did not match its reference (a real result, kept)
    if rec.get("state") != "done":
        step["status"] = "ERROR"
    else:
        step["status"] = "ok" if (identical is not False or not a.must_match) else "FAIL"
    Path(a.out).write_text(json.dumps(step, indent=1))
    print(json.dumps({k: step[k] for k in ("step", "status", "rung", "seconds", "peak_bytes", "identical",
                                           "swap_delta_mb")}))
    return 0 if step["status"] == "ok" else 1


def _gb(v) -> str:
    return "" if v in (None, "") else f"{float(v) / 1024 ** 3:.1f}"


def _step_results(rdir: Path) -> list[dict]:
    """The <step>.json files written by `finish` (not the jobs/job/request/warm-up files beside them)."""
    out = []
    for p in sorted(Path(rdir).glob("T*.json")):
        if p.name.count(".") != 1:   # T1.job.json, T1T2.jobs.json, T2.warmup.request.json, ...
            continue
        try:
            r = json.loads(p.read_text())
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("step"):
            out.append(r)
    return out


def cmd_summary(a) -> int:
    rdir = Path(a.results)
    rows = _step_results(rdir)
    order = {k: i for i, k in enumerate(a.order.split(","))} if a.order else {}
    rows.sort(key=lambda r: order.get(r.get("step"), 999))
    lines = ["# Shutter test chain", "", f"Results in `{rdir}`, written {time.strftime('%Y-%m-%d %H:%M')}.", "",
             "Peaks are MLX peaks per stage (GB = GiB); machine = whole-machine used memory peak during the job "
             "(daemon samples when the daemon ran it, else this chain's own 2 s samples).", "",
             "| step | what | rung | status | seconds | s/step | peak GB (encode / load / denoise / decode / post) "
             "| MLX peak GB | machine peak GB | identical | swap delta MB |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sp = r.get("stage_peak_bytes") or {}
        enc = max([v for k, v in sp.items() if k in ("prep", "encode")] or [None]) if sp else None
        stages = " / ".join(_gb(x) or "-" for x in (enc, sp.get("load") or sp.get("reload"), sp.get("denoise"),
                                                     sp.get("decode"), sp.get("post")))
        steps = r.get("step_seconds") or []
        sps = f"{sum(steps) / len(steps):.1f}" if steps else (r.get("seconds_per_step") or "")
        ident = {True: "yes", False: "NO", None: "-"}[r.get("identical")]
        rung = (r.get("rung") or "") + (f" (from {r['stepped_down_from']})" if r.get("stepped_down_from") else "")
        lines.append(f"| {r.get('step')} | {r.get('title', '')} | {rung} | {r.get('status')} | {r.get('seconds') or ''} "
                     f"| {sps} | {stages} | {_gb(r.get('peak_bytes'))} | {_gb(r.get('machine_peak_used_bytes'))} "
                     f"| {ident} | {'' if r.get('swap_delta_mb') is None else r['swap_delta_mb']} |")
    lines += ["", "## Comparisons", ""]
    for r in rows:
        for k, c in (r.get("compare") or {}).items():
            if "skipped" in c:
                lines.append(f"- {r['step']} {k}: skipped ({c['skipped']})")
            elif "error" in c and "max_abs_diff" not in c:
                lines.append(f"- {r['step']} {k}: error {c['error']}")
            else:
                extra = (f", frames equal {c.get('frames_equal')}/{c.get('frames_a')}, PSNR {c.get('psnr_db')}"
                         if "frames_a" in c else "")
                lines.append(f"- {r['step']} {k}: identical={c.get('identical')}, max abs diff "
                             f"{c.get('max_abs_diff')}{extra} vs `{c.get('b')}`")
    for r in rows:
        if r.get("stream"):
            s = r["stream"]
            lines.append(f"- {r['step']} streaming: window {s.get('window')}, read {_gb(s.get('bytes_read'))} GB in "
                         f"{s.get('read_seconds')} s, compute waited {s.get('wait_seconds')} s on I/O, host copies "
                         f"{s.get('convert_seconds')} s, LoRA swaps {len(s.get('lora_swaps') or [])}")
        if r.get("error"):
            lines.append(f"- {r['step']} error: {r['error']}")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def cmd_measured(a) -> int:
    """measured.json (docs/MODES.md shape) from the step results: per size and rung the highest MLX peak and the
    fastest wall time seen, and whether the outputs matched their reference."""
    sys.path.insert(0, os.path.abspath(os.path.expanduser(a.shutter_dir)))
    from shutter.core import modes as M

    best: dict[tuple, dict] = {}
    for r in _step_results(Path(a.results)):
        if r.get("state") != "done" or not r.get("rung") or not r.get("peak_bytes") or not r.get("size_key"):
            continue
        k = (r["size_key"], r["rung"])
        e = best.setdefault(k, {"peak_bytes": 0, "seconds": float("inf"), "identical": None, "steps": []})
        e["peak_bytes"] = max(e["peak_bytes"], int(r["peak_bytes"]))
        e["seconds"] = min(e["seconds"], float(r.get("seconds") or float("inf")))
        if r.get("identical") is not None:
            e["identical"] = bool(r["identical"]) if e["identical"] is None else (e["identical"] and r["identical"])
        e["steps"].append(r["step"])
    for out in [p for p in a.out.split(",") if p]:
        for (size, rung), e in best.items():
            M.record_measurement(a.model_id, size, rung, peak_bytes=e["peak_bytes"], seconds=e["seconds"],
                                 identical=e["identical"], path=out, extra={"steps": e["steps"], "kind": "mlx_peak"})
        print(f"wrote {out}: {len(best)} entries")
    return 0


# ----------------------------------------------------------------------------- main


def cmd_run(a) -> int:
    jobs = json.loads(Path(a.jobs).read_text())
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    if a.via == "daemon":
        for job in jobs:
            log(f"{job['name']}: submitting to the daemon" + (" (warm-up)" if job.get("warmup") else ""))
            rec = {"name": job["name"], "request": job["request"], "warmup": bool(job.get("warmup")),
                   **run_daemon(a.url.rstrip("/"), job, a.timeout)}
            (out / f"{job['name']}.job.json").write_text(json.dumps(rec, indent=1))
            log(f"{job['name']}: {rec['state']} in {rec['wall_s']} s" + (f": {rec.get('error')}" if rec.get("error") else ""))
            if rec["state"] != "done" and not job.get("warmup"):
                ok = False
                break
        return 0 if ok else 1
    locks = DirLocks([d for d in a.lock_dirs.split(":") if d])
    with locks:
        sess = WorkerSession(a.python)
        try:
            for job in jobs:
                log(f"{job['name']}: running in the worker" + (" (warm-up)" if job.get("warmup") else ""))
                rec = {"name": job["name"], "request": job["request"], "warmup": bool(job.get("warmup")),
                       **run_worker_job(sess, job, a.timeout, a.swap_max_mb, a.min_level)}
                (out / f"{job['name']}.job.json").write_text(json.dumps(rec, indent=1))
                log(f"{job['name']}: {rec['state']} in {rec['wall_s']} s"
                    + (f": {rec.get('error')}" if rec.get("error") else ""))
                if rec["state"] != "done" and not job.get("warmup"):
                    ok = False
                    break
                if rec["state"] == "killed":
                    ok = False
                    break
        finally:
            sess.close()
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--via", choices=("daemon", "worker"), required=True)
    r.add_argument("--jobs", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--url", default="http://127.0.0.1:8494")
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--lock-dirs", default="")
    r.add_argument("--swap-max-mb", type=float, default=300.0)
    r.add_argument("--min-level", type=int, default=30)
    r.add_argument("--timeout", type=float, default=4 * 3600)
    c = sub.add_parser("config")
    for k, d in (("--home", None), ("--python", sys.executable), ("--shutter-dir", "."), ("--model", "longcat"),
                 ("--model-id", "longcat-avatar-1.6-8bit"), ("--code-dir", ""), ("--weights", ""),
                 ("--lock-dirs", ""), ("--port", "8494"), ("--swap-max-mb", "300")):
        c.add_argument(k, default=d, required=d is None)
    q = sub.add_parser("request")
    for k, d in (("--step", None), ("--model", "longcat"), ("--model-id", "longcat-avatar-1.6-8bit"),
                 ("--size", "480x832"), ("--mode", "fast"), ("--rung", ""), ("--out-dir", None), ("--out", None),
                 ("--image", ""), ("--audio", ""), ("--prompt-file", "")):
        q.add_argument(k, default=d, required=d is None)
    f = sub.add_parser("finish")
    for k, d in (("--step", None), ("--title", ""), ("--job", None), ("--out", None), ("--ref-latents", ""),
                 ("--ref-mp4", ""), ("--base-latents", ""), ("--base-mp4", ""), ("--swap-before", ""),
                 ("--swap-after", "")):
        f.add_argument(k, default=d, required=d is None)
    f.add_argument("--must-match", action="store_true")
    s_ = sub.add_parser("summary")
    s_.add_argument("--results", required=True)
    s_.add_argument("--out", required=True)
    s_.add_argument("--order", default="")
    m = sub.add_parser("measured")
    m.add_argument("--results", required=True)
    m.add_argument("--out", required=True, help="comma-separated measured.json paths")
    m.add_argument("--model-id", required=True)
    m.add_argument("--shutter-dir", default=".")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "config": cmd_config, "request": cmd_request, "finish": cmd_finish,
            "summary": cmd_summary, "measured": cmd_measured}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
