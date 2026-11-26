#!/usr/bin/env python3
"""Measure a Shuttrr Wan build on this Mac: peak memory and time at each memory step, and whether every step gives
the same frames.

    python scripts/wan_check.py ~/.shutter/models/Wan2.2-TI2V-5B-Shuttrr-8bit --model-id wan2.2-ti2v-5b \\
        --sizes 704x1280 --frames 121 --steps 40 --rungs resident,staged,stream_wide,stream_tight \\
        --out ~/shutter-test/wan

Each memory step runs in a fresh Python process, so each peak is that step's own (MLX peak memory, reset before the
run). Frames are written as .npy (and .mp4 when ffmpeg is here) under --out/<size>/<step>/ and compared by hash.
The results go to measured.json (shutter.core.modes.record_measurement, the numbers the Wan adapter prefers over
its formulas) and a "checked_on" entry is added to the build's shuttrr.json. A report is printed and written to
--out/report.json.

GPU lock: it refuses to start while another heavy job holds the GPU. By default it takes Shuttrr's own lock file
and the shared lock folders (the gpulock.sh convention: ~/studio-gpu.lock.d and the studio render lock on a Mac),
all or nothing, and says who holds a lock it cannot take. With --gpulock PATH it runs itself under that lock script
instead (``PATH shuttr-wan-check -- python wan_check.py ...``), which takes the shared folders; Shuttrr's own lock
file is still checked. On a Mac it also refuses when the memory level says a big model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GB = 1024**3
RUNGS = ["resident", "split", "staged", "stream_wide", "stream_tight"]
LOCKED_ENV = "SHUTTER_WAN_CHECK_LOCKED"       # set when running under --gpulock
MIN_MEMORY_LEVEL = 78                          # as gpulock.sh: below this something big is loaded


# ----------------------------------------------------------------------------- arguments


def parse_size(text: str) -> tuple[int, int]:
    """"480x832" -> (height, width), the Shuttrr "HxW" convention."""
    try:
        h, w = text.lower().split("x")
        return int(h), int(w)
    except ValueError:
        raise argparse.ArgumentTypeError(f"size {text!r} is not HEIGHTxWIDTH, e.g. 480x832") from None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Measure a Shuttrr Wan build at each memory step.")
    ap.add_argument("build", type=Path, help="folder of a build made by `shutter convert`")
    ap.add_argument("--model-id", required=True, help="the id the build is configured under (measured.json key)")
    ap.add_argument("--sizes", default="480x832", help="comma-separated HEIGHTxWIDTH, e.g. 480x832,720x1280")
    ap.add_argument("--frames", type=int, default=None, help="frames (4n+1); default: the variant's")
    ap.add_argument("--steps", type=int, default=None, help="denoising steps; default: the variant's")
    ap.add_argument("--rungs", default=",".join(RUNGS), help="comma-separated memory steps, in order")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default="A red fox trots through fresh snow at dawn, soft light, cinematic.")
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--vae-tile", action="store_true",
                    help="tiled VAE decode at every step (less memory; recorded as a separate measurement)")
    ap.add_argument("--encode-once", action="store_true",
                    help="encode the prompts once in their own process and give every step the same embeddings "
                         "(the engine caches them too); saves reading the 11 GB text encoder per step on small Macs")
    ap.add_argument("--image", default=None, help="image for image-to-video builds")
    ap.add_argument("--out", type=Path, default=Path("~/shutter-test/wan"))
    ap.add_argument("--measured", type=Path, default=None, help="measured.json to update (default: Shuttrr's)")
    ap.add_argument("--no-record", action="store_true", help="do not write measured.json or shuttrr.json")
    ap.add_argument("--no-npy", action="store_true", help="do not keep the frames as .npy (hashes still compared)")
    ap.add_argument("--cache-limit-gb", type=float, default=2.0, help="MLX buffer cache limit (as the adapter)")
    ap.add_argument("--gpulock", default=None, help="lock script to run under, e.g. ~/longcat-mlx/speed/gpulock.sh")
    ap.add_argument("--lock-file", default=None, help="Shuttrr's GPU lock file (default: from Shuttrr's config)")
    ap.add_argument("--lock-dirs", default=None, help="colon-separated shared lock folders (default: the config's "
                    "gpu_lock_dirs, as the engine; e.g. ~/studio-gpu.lock.d for gpulock.sh); empty string for none")
    ap.add_argument("--no-lock", action="store_true", help="take no lock (only when nothing else can use the GPU)")
    ap.add_argument("--min-memory-level", type=int, default=MIN_MEMORY_LEVEL,
                    help=f"refuse to start below this kern.memorystatus_level (default {MIN_MEMORY_LEVEL}, as gpulock.sh "
                    "on a big Mac; a 16 GB Mac with a browser open sits near 60, so use a lower value there)")
    ap.add_argument("--timeout", type=float, default=6 * 3600, help="seconds allowed for one step")
    # internal / tests
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--prompt-embeds", default=None, help=argparse.SUPPRESS)  # .npz with prompt / negative
    return ap


# ----------------------------------------------------------------------------- machine


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def total_memory_bytes() -> int:
    if sys.platform == "darwin":
        v = _sysctl("hw.memsize")
        return int(v) if v.isdigit() else 0
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {"arch": platform.machine(), "system": platform.system(),
                            "memory_gb": round(total_memory_bytes() / GB, 1)}
    if sys.platform == "darwin":
        info["model"] = _sysctl("hw.model")
        info["chip"] = _sysctl("machdep.cpu.brand_string")
        info["macos"] = platform.mac_ver()[0]
    return info


def mlx_version() -> str:
    try:
        from importlib.metadata import version
        return version("mlx")
    except Exception:
        try:
            import mlx.core as mx
            return str(getattr(mx, "__version__", "?"))
        except Exception:
            return "?"


# ----------------------------------------------------------------------------- locks


class Locks:
    """Shuttrr's flock file plus the shared mkdir folders, taken all or nothing (what the engine takes)."""

    def __init__(self, lock_file: Optional[str], lock_dirs: Optional[list[str]], take_dirs: bool = True):
        from shutter.core.config import Config
        cfg = None
        if lock_file is None or lock_dirs is None:
            try:
                cfg = Config.load()
            except Exception:
                cfg = Config()
        self.lock_file = lock_file if lock_file is not None else cfg.gpu_lock_path
        if lock_dirs is None:
            # exactly what the engine takes: the configured folders, none when the config lists none (gpulock.sh's
            # folders belong to the Mac Studio's tools; on another Mac taking them only leaves folders behind)
            lock_dirs = list(cfg.gpu_lock_dirs or [])
        self.lock_dirs = lock_dirs if take_dirs else []
        self._flock = None
        self._dirs = None

    def acquire(self) -> Optional[str]:
        """None when held, else why not (nothing is held then)."""
        from shutter.core.dirlock import DirLocks
        from shutter.core.memory import GpuLock
        self._flock = GpuLock(self.lock_file, label=f"wan_check (pid {os.getpid()})")
        if not self._flock.try_acquire(0):
            h = self._flock.holder() or {}
            self._flock = None
            who = h.get("label") or (f"pid {h['pid']}" if h.get("pid") else "another process")
            return f"the GPU lock {self.lock_file} is held by {who}"
        if self.lock_dirs:
            self._dirs = DirLocks(self.lock_dirs, owner="shutter wan_check")
            reason = self._dirs.try_acquire()
            if reason:
                self._dirs = None
                self.release()
                return reason
        return None

    def release(self) -> None:
        if self._dirs is not None:
            self._dirs.release()
            self._dirs = None
        if self._flock is not None:
            self._flock.release()
            self._flock = None


def memory_level_problem(minimum: int = MIN_MEMORY_LEVEL) -> Optional[str]:
    if sys.platform != "darwin" or minimum <= 0:
        return None
    v = _sysctl("kern.memorystatus_level")
    if v.isdigit() and int(v) < minimum:
        return (f"the memory level is {v} (below {minimum}): something big is loaded. Free it and run "
                "this again.")
    return None


# ----------------------------------------------------------------------------- one step (child process)


class _Pool:
    """Minimal ModelPoolLike: holds what is kept, frees on release."""

    def __init__(self):
        self.items: dict[str, Any] = {}

    def get(self, key, loader, size_bytes, *, keep=True, pinned=False, residency=None):
        if key in self.items:
            return self.items[key]
        obj = loader()
        if keep:
            self.items[key] = obj
        return obj

    def release(self, key):
        self.items.pop(key, None)

    def release_all(self, prefix=""):
        for k in [k for k in self.items if k.startswith(prefix)]:
            self.items.pop(k)

    def keys(self):
        return list(self.items)


def frames_hash(frames) -> str:
    import numpy as np
    a = np.ascontiguousarray(frames)
    h = hashlib.sha256(f"{a.dtype}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def encode_once(spec: dict[str, Any]) -> dict[str, Any]:
    """--encode-once: the prompt and negative prompt as the pipeline encodes them, saved as float32 .npz."""
    import mlx.core as mx
    import numpy as np

    from shutter.adapters.wan import WanAdapter
    from shutter.families.wan.pipeline import NEG_PROMPT, WanPipeline
    from shutter.types import JobRequest
    if spec.get("cpu"):
        mx.set_default_device(mx.cpu)
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    t0 = time.time()
    build = Path(spec["build"])
    ad = WanAdapter(spec["model_id"], build)
    s = ad.settings_for(JobRequest(model=spec["model_id"], prompt=spec["prompt"],
                                   negative_prompt=spec.get("negative_prompt", ""), width=64, height=64))
    pipe = WanPipeline(build, ad.manifest)
    pos = pipe.encode_prompt(s.prompt)
    neg = pipe.encode_prompt(s.negative_prompt or NEG_PROMPT)
    out = Path(spec["out_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, prompt=np.array(pos.astype(mx.float32)), negative=np.array(neg.astype(mx.float32)))
    return {"rung": "encode", "ok": True, "peak_bytes": int(mx.get_peak_memory()), "seconds": round(time.time() - t0, 2),
            "file": str(out)}


def run_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one memory step in this process and return its numbers. `spec`: build, model_id, height, width, frames,
    steps, seed, prompt, negative_prompt, image, rung, out_dir, npy, cache_limit_gb, cpu, prompt_embeds."""
    import mlx.core as mx

    from shutter.adapters.longcat import StageMeter, write_mp4
    from shutter.adapters.wan import WanAdapter
    from shutter.families.wan.pipeline import ModelSource, WanPipeline
    from shutter.types import JobRequest
    if spec.get("cpu"):
        mx.set_default_device(mx.cpu)
    if spec.get("cache_limit_gb") is not None and hasattr(mx, "set_cache_limit"):
        mx.set_cache_limit(int(float(spec["cache_limit_gb"]) * GB))
    build = Path(spec["build"])
    ad = WanAdapter(spec["model_id"], build)
    req = JobRequest(model=spec["model_id"], prompt=spec["prompt"], negative_prompt=spec.get("negative_prompt", ""),
                     width=spec["width"], height=spec["height"], frames=spec.get("frames"),
                     steps=spec.get("steps"), seed=int(spec["seed"]),
                     inputs={"image": spec["image"]} if spec.get("image") else None,
                     params={"vae_tile": True} if spec.get("vae_tile") else {})
    s = ad.settings_for(req)
    pipe_cls = WanPipeline
    if spec.get("prompt_embeds"):
        import numpy as np
        emb = np.load(spec["prompt_embeds"])

        class _Fixed(WanPipeline):          # tests: no tokenizer / text encoder
            def encode_prompt(self, text):
                return mx.array(emb["prompt"] if text == s.prompt else emb["negative"])
        pipe_cls = _Fixed
    meter = StageMeter(mx)
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    meter.start("prep")
    t0 = time.perf_counter()
    pipe = pipe_cls(build, ad.manifest, cache=None, meter=meter)
    models = ModelSource(build, ad.manifest, spec["model_id"], _Pool(), spec["rung"],
                         nocache=sys.platform == "darwin")
    frames, info = pipe.generate(s, models, keep_warm=True)
    seconds = time.perf_counter() - t0
    meter.close()
    # the meter resets MLX's peak at every stage, so get_peak_memory() alone is only the last stage's (the decode);
    # the request's peak is the largest stage peak
    stage_peaks = [v for v in (meter.summary().get("stage_peak_bytes") or {}).values() if v]
    last = int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else 0
    peak = max([last] + [int(v) for v in stage_peaks]) or (meter.summary()["peak_bytes"] or 0)
    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    files = []
    if spec.get("npy", True):
        import numpy as np
        np.save(out / "frames.npy", frames)
        files.append(str(out / "frames.npy"))
    if shutil.which("ffmpeg"):
        write_mp4(frames, None, out / "clip.mp4", fps=int(ad.cfg.get("sample_fps", 16)))
        files.append(str(out / "clip.mp4"))
    summ = meter.summary()
    return {"rung": spec["rung"], "ok": True, "peak_bytes": peak, "seconds": round(seconds, 2),
            "hash": frames_hash(frames), "frames": int(frames.shape[0]), "size": info["size"],
            "steps": s.steps, "timings": info["timings"], "expert_loads": info["expert_loads"],
            "notes": info["notes"], "stage_peak_bytes": summ["stage_peak_bytes"], "files": files}


# ----------------------------------------------------------------------------- the loop (parent process)


RESULT_MARK = "WAN_CHECK_RESULT "


def subprocess_runner(spec: dict[str, Any], timeout: float, script: Optional[str] = None) -> dict[str, Any]:
    """Run one step in a fresh Python process; its peak is its own."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        spec_file = f.name
    cmd = [sys.executable, script or str(Path(__file__).resolve()), str(spec["build"]), "--model-id",
           spec["model_id"], "--child", spec_file]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"rung": spec["rung"], "ok": False, "error": f"stopped after {timeout:.0f} s"}
    finally:
        os.unlink(spec_file)
    for line in reversed(p.stdout.splitlines()):
        if line.startswith(RESULT_MARK):
            return json.loads(line[len(RESULT_MARK):])
    tail = (p.stderr or p.stdout).strip().splitlines()[-12:]
    return {"rung": spec["rung"], "ok": False, "error": f"exit code {p.returncode}", "log": tail}


def compare(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Which steps match the first successful one (resident when it ran)."""
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return {"reference": None, "identical": False, "same": {}}
    ref = ok[0]
    same = {r["rung"]: r["hash"] == ref["hash"] for r in ok}
    return {"reference": ref["rung"], "identical": all(same.values()) and len(ok) == len(results), "same": same}


def checked_on_entry(size: str, frames: int, steps: int, results: list[dict[str, Any]], cmp: dict[str, Any],
                     machine: Optional[dict[str, Any]] = None, mlx: Optional[str] = None) -> dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    e = {"date": time.strftime("%Y-%m-%d"), "machine": machine or machine_info(), "mlx": mlx or mlx_version(),
         "size": size, "frames": frames, "steps": steps,
         "peaks_gb": {r["rung"]: round(r["peak_bytes"] / GB, 2) for r in ok},
         "seconds": {r["rung"]: r["seconds"] for r in ok}, "identical": bool(cmp["identical"])}
    failed = [r["rung"] for r in results if not r.get("ok")]
    if failed:
        e["failed"] = failed
    return e


def append_checked_on(build: Path, entry: dict[str, Any]) -> None:
    """Add `entry` to the build's shuttrr.json "checked_on" list (atomic write)."""
    f = Path(build) / "shuttrr.json"
    man = json.loads(f.read_text())
    man.setdefault("checked_on", []).append(entry)
    fd, tmp = tempfile.mkstemp(prefix=".shuttrr.", dir=str(f.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(man, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, f)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def check(a: argparse.Namespace, runner: Optional[Callable[[dict[str, Any], float], dict[str, Any]]] = None,
          log: Callable[[str], None] = print) -> dict[str, Any]:
    """Every size x every step; record; return the report."""
    from shutter.core import modes as M
    runner = runner or subprocess_runner
    build = Path(a.build).expanduser()
    man = json.loads((build / "shuttrr.json").read_text())
    rungs = [r.strip() for r in a.rungs.split(",") if r.strip()]
    bad = [r for r in rungs if r not in RUNGS]
    if bad:
        raise SystemExit(f"unknown memory step {bad[0]!r}; the steps are {', '.join(RUNGS)}")
    out_root = Path(a.out).expanduser()
    machine, mlx = machine_info(), mlx_version()
    report: dict[str, Any] = {"build": str(build), "model_id": a.model_id, "label": man.get("label"),
                              "machine": machine, "mlx": mlx, "sizes": {}}
    if a.encode_once and not a.prompt_embeds:
        log("encoding the prompts once")
        r = runner({"build": str(build), "model_id": a.model_id, "prompt": a.prompt, "rung": "encode",
                    "negative_prompt": a.negative_prompt, "cpu": a.cpu, "encode_only": True,
                    "out_file": str(out_root / "prompt_embeds.npz")}, a.timeout)
        if not r.get("ok"):
            raise SystemExit(f"encoding the prompts failed: {r.get('error')} {r.get('log', '')}")
        log(f"prompts encoded: peak {r['peak_bytes'] / GB:.2f} GB, {r['seconds']:.1f} s")
        report["prompt_encode"] = {"peak_bytes": r["peak_bytes"], "seconds": r["seconds"]}
        a.prompt_embeds = r["file"]
    for size_text in [x.strip() for x in a.sizes.split(",") if x.strip()]:
        h, w = parse_size(size_text)
        results = []
        for rung in rungs:
            spec = {"build": str(build), "model_id": a.model_id, "height": h, "width": w, "frames": a.frames,
                    "steps": a.steps, "seed": a.seed, "prompt": a.prompt, "negative_prompt": a.negative_prompt,
                    "image": a.image, "rung": rung, "out_dir": str(out_root / f"{h}x{w}" / rung),
                    "npy": not a.no_npy, "cache_limit_gb": a.cache_limit_gb, "cpu": a.cpu,
                    "prompt_embeds": a.prompt_embeds, "vae_tile": a.vae_tile}
            log(f"{h}x{w} {rung}: running")
            r = runner(spec, a.timeout)
            if r.get("ok"):
                log(f"{h}x{w} {rung}: peak {r['peak_bytes'] / GB:.2f} GB, {r['seconds']:.1f} s, "
                    f"{r['frames']} frames, hash {r['hash'][:12]}")
            else:
                log(f"{h}x{w} {rung}: FAILED ({r.get('error')})")
                for line in r.get("log", []):
                    log("    " + line)
            results.append(r)
            # keep what is finished if the run is stopped later (records are written per size, below)
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "report.json").write_text(json.dumps(dict(report, running={f"{h}x{w}": results}), indent=1))
        cmp = compare(results)
        ok = [r for r in results if r.get("ok")]
        # the generated size after alignment (what the adapter looks measurements up by)
        sz = f"{ok[0]['size'][1]}x{ok[0]['size'][0]}" if ok else M.size_key(h, w)
        frames = ok[0]["frames"] if ok else a.frames
        steps = ok[0]["steps"] if ok else a.steps
        entry = checked_on_entry(sz, frames, steps, results, cmp, machine, mlx)
        if a.vae_tile:
            entry["vae_tile"] = True
        if not a.no_record and ok:
            for r in ok:
                M.record_measurement(a.model_id, sz, r["rung"], peak_bytes=r["peak_bytes"], seconds=r["seconds"],
                                     identical=cmp["same"].get(r["rung"]), path=a.measured,
                                     extra={"frames": frames, "steps": steps, "hash": r["hash"][:16],
                                            **({"vae_tile": True} if a.vae_tile else {})})
            append_checked_on(build, entry)
        report["sizes"][sz] = {"results": results, "compare": cmp, "checked_on": entry}
        log(f"{sz}: {'all steps give the same frames' if cmp['identical'] else 'NOT identical: ' + json.dumps(cmp['same'])}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "report.json").write_text(json.dumps(report, indent=1))
    return report


# ----------------------------------------------------------------------------- main


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if a.child:
        spec = json.loads(Path(a.child).read_text())
        res = encode_once(spec) if spec.get("encode_only") else run_one(spec)
        print(RESULT_MARK + json.dumps(res), flush=True)
        return 0
    if not (Path(a.build).expanduser() / "shuttrr.json").exists():
        print(f"{a.build} is not a Shuttrr build (no shuttrr.json).", file=sys.stderr)
        return 2
    if a.gpulock and not os.environ.get(LOCKED_ENV):
        script = str(Path(a.gpulock).expanduser())
        if not os.access(script, os.X_OK):
            print(f"the lock script {script} is not there or not executable.", file=sys.stderr)
            return 2
        env = dict(os.environ, **{LOCKED_ENV: "1"})
        cmd = [script, "shuttr-wan-check", "--", sys.executable, str(Path(__file__).resolve()), *argv]
        return subprocess.call(cmd, env=env)
    try:
        [parse_size(x) for x in a.sizes.split(",") if x.strip()]
    except argparse.ArgumentTypeError as e:
        print(str(e), file=sys.stderr)
        return 2
    bad = [r for r in a.rungs.split(",") if r.strip() and r.strip() not in RUNGS]
    if bad:
        print(f"unknown memory step {bad[0]!r}; the steps are {', '.join(RUNGS)}", file=sys.stderr)
        return 2
    # a stop (kill / Ctrl-C) must still release the locks below and stop the running step: turn SIGTERM / SIGHUP
    # into SystemExit, so the `finally` runs and subprocess.run kills its child
    def _stop(signum, frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _stop)
    locks = None
    if not a.no_lock:
        # --cpu is the tests' tiny CPU run: the machine's memory level says nothing about it
        problem = None if a.cpu else memory_level_problem(a.min_memory_level)
        if problem:
            print("Not starting: " + problem, file=sys.stderr)
            return 3
        dirs = None if a.lock_dirs is None else [d for d in a.lock_dirs.split(":") if d]
        # under the lock script the shared folders are already ours; Shuttrr's own lock file is still taken
        locks = Locks(a.lock_file, dirs, take_dirs=not os.environ.get(LOCKED_ENV))
        reason = locks.acquire()
        if reason:
            print(f"Not starting: {reason}. Wait for that job to finish.", file=sys.stderr)
            return 3
    try:
        report = check(a)
    finally:
        if locks is not None:
            locks.release()
    ok = all(v["compare"]["identical"] for v in report["sizes"].values())
    print(f"report: {Path(a.out).expanduser() / 'report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
