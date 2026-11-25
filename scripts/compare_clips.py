#!/usr/bin/env python3
"""Compare two clips frame by frame (or two latent arrays) and say whether they are identical.

    compare_clips.py A.mp4 B.mp4 [--min-psnr DB] [--json OUT]
    compare_clips.py --latents A.npy B.npy [--json OUT]

Clips are decoded with ffmpeg to raw RGB24 (streamed one frame at a time, so 720p clips need little memory) and
compared: frame counts, frames equal, max abs pixel difference, PSNR over all frames and of the worst frame.
Latents (.npy, e.g. latents_denorm.npy from speedlab/run_e2e.py or a Shutter job with save_latents) are compared
exactly: shape, dtype, max abs difference.

Exit code 0 only if identical (or, with --min-psnr, if every frame count matches and PSNR >= DB), 1 if they
differ, 2 on an error (missing file, ffmpeg failure). The JSON report is printed as the last line.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def probe(path: str) -> tuple[int, int]:
    exe = shutil.which("ffprobe") or "ffprobe"
    r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0:s=x", path], capture_output=True, text=True)
    if r.returncode != 0 or "x" not in r.stdout:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr.strip()}")
    w, h = r.stdout.strip().splitlines()[0].split("x")[:2]
    return int(w), int(h)


class Frames:
    def __init__(self, path: str):
        self.w, self.h = probe(path)
        self.size = self.w * self.h * 3
        exe = shutil.which("ffmpeg") or "ffmpeg"
        self.p = subprocess.Popen([exe, "-v", "error", "-i", path, "-map", "0:v:0", "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)

    def next(self):
        buf = self.p.stdout.read(self.size)
        if not buf or len(buf) < self.size:
            return None
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def close(self) -> int:
        try:
            self.p.stdout.close()
        except Exception:
            pass
        return self.p.wait()


def psnr(mse: float) -> float:
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def compare_clips(a: str, b: str) -> dict:
    fa, fb = Frames(a), Frames(b)
    rep = {"a": a, "b": b, "size_a": [fa.w, fa.h], "size_b": [fb.w, fb.h]}
    if (fa.w, fa.h) != (fb.w, fb.h):
        fa.close(), fb.close()
        rep.update(identical=False, error="different frame sizes")
        return rep
    n_a = n_b = equal = 0
    max_abs = 0
    se_total = 0.0
    worst = float("inf")
    worst_i = None
    while True:
        x, y = fa.next(), fb.next()
        if x is None or y is None:
            n_a += x is not None
            n_b += y is not None
            while x is not None:
                x = fa.next()
                n_a += x is not None
            while y is not None:
                y = fb.next()
                n_b += y is not None
            break
        n_a += 1
        n_b += 1
        d = np.abs(x.astype(np.int16) - y.astype(np.int16))
        m = int(d.max())
        if m == 0:
            equal += 1
        max_abs = max(max_abs, m)
        se = float(np.square(d, dtype=np.float64).sum())
        se_total += se
        p = psnr(se / d.size)
        if p < worst:
            worst, worst_i = p, n_a - 1
    rc = (fa.close(), fb.close())
    if any(rc):
        raise RuntimeError(f"ffmpeg exited with {rc}")
    n = min(n_a, n_b)
    mse = se_total / (n * fa.size) if n else float("nan")
    rep.update(frames_a=n_a, frames_b=n_b, frames_equal=equal, max_abs_diff=max_abs,
               psnr_db=None if not n else (None if mse == 0 else round(psnr(mse), 3)),
               worst_frame_psnr_db=None if worst in (float("inf"),) else round(worst, 3), worst_frame=worst_i,
               identical=bool(n_a == n_b and n_a > 0 and equal == n_a))
    return rep


def compare_latents(a: str, b: str) -> dict:
    x, y = np.load(a), np.load(b)
    rep = {"a": a, "b": b, "shape_a": list(x.shape), "shape_b": list(y.shape), "dtype_a": str(x.dtype),
           "dtype_b": str(y.dtype)}
    if x.shape != y.shape:
        rep.update(identical=False, error="different shapes")
        return rep
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    rep.update(max_abs_diff=float(d.max()) if d.size else 0.0, mean_abs_diff=float(d.mean()) if d.size else 0.0,
               elements_equal=int((x == y).sum()), elements=int(x.size),
               identical=bool(x.dtype == y.dtype and np.array_equal(x, y)))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--latents", action="store_true", help="compare two .npy arrays exactly")
    ap.add_argument("--min-psnr", type=float, default=None, help="pass when PSNR >= this (clips only)")
    ap.add_argument("--json", default=None, help="also write the report here")
    a = ap.parse_args(argv)
    for p in (a.a, a.b):
        if not Path(p).exists():
            print(json.dumps({"error": f"missing {p}", "identical": False}))
            return 2
    try:
        rep = compare_latents(a.a, a.b) if a.latents else compare_clips(a.a, a.b)
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}", "identical": False}))
        return 2
    ok = rep.get("identical", False)
    if not ok and a.min_psnr is not None and not a.latents and rep.get("frames_a") == rep.get("frames_b"):
        p = rep.get("psnr_db")
        ok = p is not None and p >= a.min_psnr
        rep["min_psnr"] = a.min_psnr
    rep["pass"] = bool(ok)
    line = json.dumps(rep)
    if a.json:
        Path(a.json).write_text(line + "\n")
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
