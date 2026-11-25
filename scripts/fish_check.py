#!/usr/bin/env python3
"""Check the Fish Drama 3 adapter on a real Mac: same audio as the plain command, lower memory, warm second take.

Run it under the shared GPU lock, with the Drama 3 venv:

    ~/longcat-mlx/speed/gpulock.sh shuttr-fish -- ~/drama3-venv/bin/python scripts/fish_check.py \\
        --ref-audio voice.wav --ref-text "what the clip says" --text "One line to speak." \\
        --baseline ~/models/drama3-mem/ab/llw/l1_s5.wav --seed 5 --out ~/shutter-test/fish

It runs the same line twice through the adapter (first load, then warm) plus a two-take job, and prints one JSON
report: seconds, frames a second, MLX peak memory, whole-machine peak over the idle level, swap change, and
whether the first take's audio equals --baseline sample for sample.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shutter.adapters.base import RunContext  # noqa: E402
from shutter.adapters.drama3 import Drama3Adapter  # noqa: E402
from shutter.core.cache import ContentCache  # noqa: E402
from shutter.types import GB, Job, JobRequest  # noqa: E402


def used_bytes() -> int:
    """anon + wired + compressor, the number that decides swap (macOS)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split(" ")[0])
    get = lambda name: int(next(l for l in out.splitlines() if l.startswith(name)).split(":")[1].strip(" ."))  # noqa: E731
    return page * (get("Anonymous pages") + get("Pages wired down") + get("Pages occupied by compressor"))


def swap_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split("M")[0])


class Pool:
    def __init__(self):
        self.items = {}

    def get(self, key, loader, size_bytes, *, keep=True, **kw):
        if key not in self.items:
            self.items[key] = loader()
        return self.items[key]

    def release(self, key):
        self.items.pop(key, None)

    def release_all(self, prefix=""):
        self.items.clear()


def samples(path) -> bytes:
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/models/drama-3-mlx")
    ap.add_argument("--src", default="~/models/drama3-mem/code/src")
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--baseline", help="WAV of the same line and seed from `python -m tensorfold.drama3`")
    ap.add_argument("--verify", action="store_true", help="as the baseline was made (speech check and redraws)")
    ap.add_argument("--out", default="~/shutter-test/fish")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    adapter = Drama3Adapter("fish-drama3", Path(a.model).expanduser(), {"tensorfold_src": a.src})
    pool, cache = Pool(), ContentCache(out / "cache", max_bytes=GB)
    idle, swap0 = used_bytes(), swap_mb()
    peak = [idle]
    stop = threading.Event()

    def watch():
        while not stop.wait(0.2):
            peak[0] = max(peak[0], used_bytes())

    threading.Thread(target=watch, daemon=True).start()

    def run(name: str, **params):
        req = JobRequest(model="fish-drama3", prompt=a.text, seed=a.seed, inputs={"audio": a.ref_audio},
                         output=str(out / f"{name}.wav"),
                         params={"ref_text": a.ref_text, "verify": a.verify, "save_codes": True, **params})
        ctx = RunContext(models=pool, cache=cache, work_dir=out, report=lambda f, s: print(f"  {f:4.2f} {s}", flush=True),
                         check_cancelled=lambda: None, memory_sample=lambda: {})
        started = time.perf_counter()
        r = adapter.run(Job(id=name, request=req), ctx)
        r["wall_s"] = round(time.perf_counter() - started, 2)
        return r

    report = {"estimate_peak_gb": round(adapter.estimate(JobRequest(model="m", prompt=a.text,
                                                                     params={"verify": a.verify})).peak / GB, 1)}
    for name, params in (("cold", {}), ("warm", {}), ("two_takes", {"takes": 2})):
        r = run(name, **params)
        report[name] = {k: r[k] for k in ("wall_s", "load_s", "warm", "reference", "audio_s", "decode_wall_s",
                                          "realtime_factor", "mlx_peak_bytes") if k in r}
        report[name]["frames_per_s"] = [t["frames_per_s"] for t in r["takes"]]
        report[name]["prefill_s"] = [t["prefill_s"] for t in r["takes"]]
    stop.set()
    report["same_audio_cold_warm"] = samples(out / "cold.wav") == samples(out / "warm.wav")
    if a.baseline:
        report["same_audio_as_baseline"] = samples(out / "cold.wav") == samples(Path(a.baseline).expanduser())
    report["machine_idle_gb"] = round(idle / GB, 1)
    report["machine_peak_over_idle_gb"] = round((peak[0] - idle) / GB, 1)
    report["swap_change_mb"] = round(swap_mb() - swap0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
