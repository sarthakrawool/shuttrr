#!/usr/bin/env python3
"""Shutter speed benchmark (Apple GPU): dense vs block-sparse self-attention at the LongCat-Avatar token counts.

Self-contained: needs only mlx + numpy (Python >= 3.11) and `block_sparse.py` (from --kernels-dir, default
./kernels next to this file, else ../shutter/kernels).

For each size (default 37,440 tokens = 480x832 and 86,592 tokens = 704x1312, 24 latent frames, 32 heads x 128,
bf16) it builds the production layout of one block's self-attention: noise queries (23 frames) attending to
[1 conditioning frame + 23 noise frames]; the conditioning frame's attention to itself is identical in both paths
and left out. Then it times
  dense:   one SDPA per head (LC_SDPA_HEADS=1, the production default), all 32 heads;
  sparse:  shutter.kernels.block_sparse at each --sparsities value (conditioning frame kept as a dense prefix),
and reports seconds per block, per 48-block step (extrapolated), the speed-up, and the error vs the dense output on
these random inputs (cosine, PSNR; random inputs are the worst case for block selection: quality on the real model
must be judged on real activations / videos). Timing: warm-up, then the median of k runs, each ended by
mx.eval + mx.synchronize. Writes JSON to --out and prints a table.

GPU conventions as scripts/bench_kernels.py: refuses to run without --yes, refuses without Metal, caps MLX memory
(--memory-limit-gb), stops starting new measurements after --budget-s, and is meant to run under the GPU lock:
  ~/longcat-mlx/speed/gpulock.sh shutter-speed -- ~/longcat-mlx/code/.venv/bin/python ~/shutter-bench/bench_speed.py --yes --out ~/shutter-bench/bench/speed.json
(copy scripts/bench_speed.py to ~/shutter-bench/ and shutter/kernels/block_sparse.py to ~/shutter-bench/kernels/).

--smoke: tiny shapes on the CPU device (no GPU, no lock, no --yes needed); only checks that the script works.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

RUN_CMD = ("~/longcat-mlx/speed/gpulock.sh shutter-speed -- ~/longcat-mlx/code/.venv/bin/python "
           "~/shutter-bench/bench_speed.py --yes --out ~/shutter-bench/bench/speed.json")

# tokens -> (T, H, W) latent patch grid, 24 latent frames (93 video frames), 1 conditioning latent
SIZES = {37440: (24, 30, 52), 86592: (24, 44, 82)}
SMOKE_SIZES = {6 * 8 * 12: (6, 8, 12), 5 * 10 * 6: (5, 10, 6)}
DEPTH = 48
COND = 1


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog="Run with:\n  " + RUN_CMD,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="required (except --smoke): confirm this may use the GPU")
    ap.add_argument("--out", type=Path, help="JSON output path")
    ap.add_argument("--smoke", action="store_true", help="tiny shapes on the CPU device: tests the script only")
    ap.add_argument("--kernels-dir", type=Path, default=None, help="folder with block_sparse.py")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)), help="token counts, from: " + ", ".join(map(str, SIZES)))
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--sparsities", default="0.5,0.75,0.9375")
    ap.add_argument("--chunk", default="4,4,4")
    ap.add_argument("--head-chunk", type=int, default=1, help="heads per sparse gather/SDPA group")
    ap.add_argument("--max-gather-mb", type=float, default=1024.0, help="gathered K+V per group")
    ap.add_argument("--compiled", action="store_true", help="also time the sparse path through mx.compile")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--memory-limit-gb", type=float, default=40.0)
    ap.add_argument("--budget-s", type=float, default=900.0, help="stop starting new measurements after this")
    return ap.parse_args()


def load_block_sparse(kdir):
    here = Path(__file__).resolve().parent
    for d in ([kdir] if kdir else [here / "kernels", here.parent / "shutter" / "kernels"]):
        if d and (d / "block_sparse.py").exists():
            spec = importlib.util.spec_from_file_location("shutter_bench_block_sparse", d / "block_sparse.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m, str(d)
    return None, None


class Timer:
    def __init__(self, mx, reps, warm, budget_s):
        self.mx, self.reps, self.warm, self.budget = mx, reps, warm, budget_s
        self.t0 = time.perf_counter()
        self.skipped = []

    def over(self, what):
        if time.perf_counter() - self.t0 > self.budget:
            self.skipped.append(what)
            return True
        return False

    def sync(self):
        if hasattr(self.mx, "synchronize"):
            self.mx.synchronize()

    def time(self, fn):
        mx = self.mx
        out = None
        for _ in range(self.warm):
            out = fn()
            mx.eval(out)
            self.sync()
        ts = []
        for _ in range(self.reps):
            t = time.perf_counter()
            out = fn()
            mx.eval(out)
            self.sync()
            ts.append(time.perf_counter() - t)
        return statistics.median(ts), out

    def free(self):
        mx = self.mx
        (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()

    def peak_gb(self, reset=False):
        mx = self.mx
        try:
            v = (getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory)() / 1e9
            if reset:
                (getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory)()
            return v
        except Exception:
            return None


def metrics(mx, got, ref):
    g, r = got.astype(mx.float32).reshape(-1), ref.astype(mx.float32).reshape(-1)
    mse = float(mx.mean((g - r) ** 2))
    peak = float(mx.max(mx.abs(r)))
    cos = float(mx.sum(g * r) / (mx.sqrt(mx.sum(g * g)) * mx.sqrt(mx.sum(r * r)) + 1e-30))
    return {"cosine": cos, "psnr_db": None if mse == 0 else 10 * math.log10(peak * peak / mse),
            "max_abs": float(mx.max(mx.abs(g - r)))}


def bench_size(mx, bs, tm: Timer, tokens, grid, a, sparsities, chunk, dtype):
    T, H, W = grid
    tpf = H * W
    nq, nk = (T - COND) * tpf, T * tpf
    Hd, D = a.heads, a.head_dim
    scale = D ** -0.5
    mx.random.seed(tokens)
    q = mx.random.normal((1, Hd, nq, D)).astype(dtype)
    k = mx.random.normal((1, Hd, nk, D)).astype(dtype)
    v = mx.random.normal((1, Hd, nk, D)).astype(dtype)
    mx.eval(q, k, v)
    tm.peak_gb(reset=True)
    row = {"tokens": tokens, "grid": list(grid), "noise_queries": nq, "keys": nk, "cond_prefix": COND * tpf,
           "heads": Hd, "head_dim": D, "dtype": str(dtype)}

    def dense():
        return mx.concatenate([mx.fast.scaled_dot_product_attention(q[:, h:h + 1], k[:, h:h + 1], v[:, h:h + 1],
                                                                    scale=scale) for h in range(Hd)], axis=1)

    s_d, ref = tm.time(dense)
    flops = 4 * nq * nk * D * Hd
    row.update({"dense_s": s_d, "dense_tflops": flops / s_d / 1e12, "dense_s_per_step": s_d * DEPTH,
                "dense_peak_gb": tm.peak_gb(reset=True)})
    print(f"  {tokens} tokens: dense {s_d:.3f} s/block ({row['dense_tflops']:.1f} TFLOP/s)", flush=True)
    rows = []
    for sp in sparsities:
        if tm.over(f"{tokens} sparsity {sp}"):
            break
        kw = dict(sparsity=sp, chunk=chunk, scale=scale, k_prefix=COND * tpf, head_chunk=a.head_chunk,
                  max_gather_bytes=int(a.max_gather_mb * 2 ** 20))
        g = (T - COND, H, W)
        r = {"sparsity": sp}
        try:
            pat = bs.pattern(g, k_prefix=COND * tpf, sparsity=sp, chunk=chunk)
            r.update({"density": pat["density"], "score_fraction": pat["score_fraction"],
                      "kept_per_q_chunk": pat["kept_per_q_chunk"], "q_chunks": pat["q_chunks"]})
            s, out = tm.time(lambda: bs.block_sparse_attention(q, k, v, g, **kw))
            r.update({"s": s, "speedup": s_d / s, "s_per_step": s * DEPTH, "peak_gb": tm.peak_gb(reset=True),
                      **metrics(mx, out, ref)})
            del out
            if a.compiled:
                f = mx.compile(lambda q_, k_, v_: bs.block_sparse_attention(q_, k_, v_, g, **kw))
                s_c, out = tm.time(lambda: f(q, k, v))
                r.update({"s_compiled": s_c, "speedup_compiled": s_d / s_c})
                del out
            print(f"    sparsity {sp}: {s:.3f} s/block, x{r['speedup']:.2f} vs dense, scores {pat['score_fraction']:.3f}"
                  f" of dense, cosine {r['cosine']:.4f}", flush=True)
        except Exception as e:      # OOM, unsupported mask, etc.: record and continue
            r.update({"error": f"{type(e).__name__}: {e}"[:1500], "trace": traceback.format_exc()[-1500:]})
            print(f"    sparsity {sp}: FAILED {r['error'][:200]}", flush=True)
        rows.append(r)
        tm.free()
    row["sparse"] = rows
    del q, k, v, ref
    tm.free()
    return row


def table(res):
    L = [f"\n{'tokens':>7} {'sparsity':>8} {'s/block':>8} {'s/step':>7} {'speedup':>7} {'scores':>7} {'cosine':>7} "
         f"{'PSNR':>6} {'peak GB':>7}"]
    for row in res["rows"]:
        L.append(f"{row['tokens']:>7} {'dense':>8} {row['dense_s']:8.3f} {row['dense_s_per_step']:7.1f} {'1.00':>7} "
                 f"{'1.000':>7} {'':>7} {'':>6} {row['dense_peak_gb'] or 0:7.2f}")
        for r in row["sparse"]:
            if "error" in r:
                L.append(f"{row['tokens']:>7} {r['sparsity']:8.4f} FAILED: {r['error'][:80]}")
                continue
            L.append(f"{row['tokens']:>7} {r['sparsity']:8.4f} {r['s']:8.3f} {r['s_per_step']:7.1f} {r['speedup']:7.2f} "
                     f"{r['score_fraction']:7.3f} {r['cosine']:7.4f} {r['psnr_db'] or 0:6.1f} {r['peak_gb'] or 0:7.2f}")
    return "\n".join(L)


def main():
    a = parse()
    if not a.yes and not a.smoke:
        print("Refusing to run without --yes (this uses the GPU for several minutes and up to "
              f"{a.memory_limit_gb:.0f} GB).\nRun under the GPU lock:\n  " + RUN_CMD)
        return 2
    if a.out is None:
        print("--out PATH is required")
        return 2
    import mlx.core as mx

    if a.smoke:
        mx.set_default_device(mx.cpu)
    elif not mx.metal.is_available():
        print("Metal is not available: this benchmark only runs on an Apple GPU (use --smoke to test the script).")
        return 3
    try:
        (getattr(mx, "set_memory_limit", None) or mx.metal.set_memory_limit)(int(a.memory_limit_gb * 1024 ** 3))
    except Exception:
        pass
    bs, kdir = load_block_sparse(a.kernels_dir)
    if bs is None:
        print("block_sparse.py not found (pass --kernels-dir)")
        return 4
    sizes = SMOKE_SIZES if a.smoke else {int(t): SIZES[int(t)] for t in a.sizes.split(",") if t.strip()}
    if a.smoke:
        a.heads, a.head_dim, a.reps = 2, 32, 1
    sparsities = [float(s) for s in a.sparsities.split(",")]
    chunk = tuple(int(c) for c in a.chunk.split(","))
    tm = Timer(mx, a.reps, 0 if a.smoke else 1, a.budget_s)
    dtype = mx.float32 if a.smoke else mx.bfloat16
    res = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
           "mlx": getattr(mx, "__version__", "?"), "platform": platform.platform(), "smoke": a.smoke,
           "kernels_dir": kdir, "chunk": list(chunk), "sparsities": sparsities, "depth": DEPTH,
           "layout": "noise queries x (1 cond frame prefix + noise frames), per block, all heads",
           "note": "random inputs: worst case for block selection; quality must be judged on real activations"}
    try:
        info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
        res["device"] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in info.items()}
    except Exception as e:
        res["device"] = str(e)
    res["rows"] = []
    for tokens, grid in sizes.items():
        if tm.over(f"size {tokens}"):
            continue
        try:
            res["rows"].append(bench_size(mx, bs, tm, tokens, grid, a, sparsities, chunk, dtype))
        except Exception as e:
            res.setdefault("errors", []).append({"tokens": tokens, "error": f"{type(e).__name__}: {e}"[:1500]})
            print(f"  {tokens}: FAILED {e}", flush=True)
        tm.free()
    res["elapsed_s"] = time.perf_counter() - tm.t0
    res["skipped_for_budget"] = tm.skipped
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(table(res))
    print(f"\nelapsed {res['elapsed_s']:.1f} s, JSON: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
