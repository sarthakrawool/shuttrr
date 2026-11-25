#!/usr/bin/env python3
"""Shutter kernel benchmark (Apple GPU). Self-contained: needs only mlx + numpy, Python >= 3.11.

Measures on the Mac's GPU, in well under 2 minutes and under 8 GB:
  (a) dense matmul throughput, bf16 / fp16 / fp32, at the shapes that matter
      ([N,128] x [128,N'] attention tiles, [N,4096] x [4096,4096] and the FFN shape)
  (b) 8-bit QuantizedLinear (mx.quantized_matmul, group 64) vs a bf16 Linear of the same shape (dequant cost),
      and a rank-128 runtime-LoRA side path at the qkv shape
  (c) mx.fast.scaled_dot_product_attention, head dim 128, N in {8K, 16K, 32K} per head, extrapolated N^2 to the
      86,592-token step (83K noise queries x 86.6K keys + 3.6K cond x cond, 32 heads, 48 blocks)
  (d) the custom kernels (adaln_mod, norm_rope, qlinear8, flash_attn) vs their MLX versions, with max-abs-diff
      and bit-identity checks against both the port's ops and the production fast path

Timing: warm-up runs, then the median of k runs, each ended by mx.eval + mx.synchronize.
Writes JSON to --out and prints a table. Refuses to run without --yes.

The kernels are loaded from --kernels-dir (default: ./kernels next to this file, else ../shutter/kernels):
copy shutter/kernels/{adaln_mod,norm_rope,qlinear8,flash_attn}.py into a folder next to this script.

Run on the Mac (through the GPU lock):
  ~/longcat-mlx/speed/gpulock.sh shutter-bench -- ~/longcat-mlx/code/.venv/bin/python ~/shutter-bench/bench_kernels.py --yes --out ~/shutter-bench/bench/kernels.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

RUN_CMD = ("~/longcat-mlx/speed/gpulock.sh shutter-bench -- ~/longcat-mlx/code/.venv/bin/python "
           "~/shutter-bench/bench_kernels.py --yes --out ~/shutter-bench/bench/kernels.json")

KERNELS = ("adaln_mod", "norm_rope", "qlinear8", "flash_attn")

# the production step (flops.Config defaults): 704x1312, 93 frames, num_cond_latents = 1
STEP_TOKENS = 86592
STEP_COND = 3608
STEP_HEADS = 32
STEP_DEPTH = 48
MEASURED_SDPA_PER_BLOCK = {"per_head": 2.084, "one_call": 3.00}  # REPORT.md, s per block


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 epilog="Run with:\n  " + RUN_CMD,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="required: confirm this may use the GPU (~1-2 min, <8 GB)")
    ap.add_argument("--out", type=Path, required=False, help="JSON output path")
    ap.add_argument("--kernels-dir", type=Path, default=None, help="folder with the kernel .py files")
    ap.add_argument("--quick", action="store_true", help="smaller shapes, fewer repeats")
    ap.add_argument("--budget-s", type=float, default=100.0, help="stop starting new measurements after this")
    ap.add_argument("--only", default="abcd", help="sections to run, e.g. 'ad'")
    ap.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)  # CPU, tiny shapes: tests the script
    return ap.parse_args()


# ------------------------------------------------------------------------------------------- utilities
class Bench:
    def __init__(self, mx, budget_s, quick):
        self.mx = mx
        self.t0 = time.perf_counter()
        self.budget = budget_s
        self.quick = quick
        self.k = 3 if quick else 5
        self.warm = 1 if quick else 2
        self.skipped = []
        self.scale = 1

    def n(self, x):
        return max(128, x // self.scale)

    def over(self, what):
        if time.perf_counter() - self.t0 > self.budget:
            self.skipped.append(what)
            return True
        return False

    def sync(self):
        if hasattr(self.mx, "synchronize"):
            self.mx.synchronize()

    def time(self, fn, k=None, warm=None):
        """Median seconds of fn() (its result evaluated and the GPU synchronised)."""
        mx = self.mx
        for _ in range(self.warm if warm is None else warm):
            mx.eval(fn())
            self.sync()
        ts = []
        for _ in range(self.k if k is None else k):
            t = time.perf_counter()
            mx.eval(fn())
            self.sync()
            ts.append(time.perf_counter() - t)
        return statistics.median(ts)

    def free(self):
        self.mx.clear_cache() if hasattr(self.mx, "clear_cache") else self.mx.metal.clear_cache()

    def peak_gb(self):
        mx = self.mx
        f = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        return f() / 1e9


def maxdiff(mx, a, b):
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    return {"max_abs_diff": float(mx.max(d)), "frac_exact": float(mx.mean(d == 0)), "bit_identical": bool(mx.all(d == 0))}


def load_kernels(kdir: Path | None):
    here = Path(__file__).resolve().parent
    cands = [kdir] if kdir else [here / "kernels", here.parent / "shutter" / "kernels"]
    for d in cands:
        if d and all((d / f"{k}.py").exists() for k in KERNELS):
            mods = {}
            for k in KERNELS:
                spec = importlib.util.spec_from_file_location(f"shutter_bench_{k}", d / f"{k}.py")
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                mods[k] = m
            return mods, str(d)
    return {}, None


# --------------------------------------------------------------------------------------------- sections
def sec_matmul(b: Bench, res):
    mx = b.mx
    out = []
    Ns = (8192, 16384) if b.quick else (8192, 16384, 32768)
    dts = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
    for name, dt in dts.items():
        # attention-score tiles [N,128] x [128,N']
        for n, n2 in ((b.n(8192), b.n(8192)), (b.n(16384), b.n(16384))):
            if b.over(f"matmul tile {name} {n}"):
                break
            a = mx.random.normal((n, 128)).astype(dt)
            c = mx.random.normal((128, n2)).astype(dt)
            mx.eval(a, c)
            s = b.time(lambda: a @ c)
            out.append({"kind": "attn_tile", "dtype": name, "shape": f"[{n},128]x[128,{n2}]", "s": s,
                        "flops": 2 * n * n2 * 128, "tflops": 2 * n * n2 * 128 / s / 1e12})
            del a, c
            b.free()
        # linear [N,4096] x [4096,4096]
        for n in map(b.n, Ns):
            if b.over(f"matmul linear {name} {n}") or (name == "fp32" and n > 16384):
                continue
            a = mx.random.normal((n, 4096)).astype(dt)
            w = (mx.random.normal((4096, 4096)) * 0.02).astype(dt)
            mx.eval(a, w)
            s = b.time(lambda: a @ w)
            f = 2 * n * 4096 * 4096
            out.append({"kind": "linear", "dtype": name, "shape": f"[{n},4096]x[4096,4096]", "s": s, "flops": f,
                        "tflops": f / s / 1e12})
            del a, w
            b.free()
        # FFN shape
        if not b.over(f"matmul ffn {name}"):
            n = b.n(16384)
            a = mx.random.normal((n, 4096)).astype(dt)
            w = (mx.random.normal((4096, 11008)) * 0.02).astype(dt)
            mx.eval(a, w)
            s = b.time(lambda: a @ w)
            f = 2 * n * 4096 * 11008
            out.append({"kind": "ffn", "dtype": name, "shape": f"[{n},4096]x[4096,11008]", "s": s, "flops": f,
                        "tflops": f / s / 1e12})
            del a, w
            b.free()
    res["matmul"] = out
    best = {}
    for r in out:
        best[r["dtype"]] = max(best.get(r["dtype"], 0.0), r["tflops"])
    res["peak_matmul_tflops"] = best
    res["peak_matmul_flops"] = best.get("bf16", 0.0) * 1e12


def sec_quant(b: Bench, res):
    mx = b.mx
    out = []
    n = b.n(8192 if b.quick else 16384)
    shapes = (("attn.proj", 4096, 4096), ("attn.qkv", 4096, 12288), ("ffn.w1", 4096, 11008), ("ffn.w2", 11008, 4096))
    for name, k, o in shapes:
        if b.over(f"quant {name}"):
            break
        x = mx.random.normal((n, k)).astype(mx.bfloat16)
        wf = (mx.random.normal((o, k)) * 0.02).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(wf, group_size=64, bits=8)
        mx.eval(x, wf, wq, sc, bi)
        f = 2 * n * k * o
        s_q = b.time(lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=8))
        s_d = b.time(lambda: x @ wf.T)
        out.append({"layer": name, "shape": f"[{n},{k}]->{o}", "s_q8": s_q, "s_bf16": s_d,
                    "tflops_q8": f / s_q / 1e12, "tflops_bf16": f / s_d / 1e12, "q8_over_bf16": s_q / s_d})
        del x, wf, wq, sc, bi
        b.free()
    # runtime LoRA side path at the qkv shape (rank 128, fused down, 3 ups), as runtime_lora.LoRALinear does it
    if not b.over("lora"):
        r = 128
        x = mx.random.normal((n, 4096)).astype(mx.bfloat16)
        down = (mx.random.normal((3 * r, 4096)) * 0.01).astype(mx.bfloat16)
        ups = [(mx.random.normal((4096, r)) * 0.01).astype(mx.bfloat16) for _ in range(3)]
        y = mx.random.normal((n, 3 * 4096)).astype(mx.bfloat16)
        mx.eval(x, down, ups, y)

        def lora():
            h = x @ down.T
            hs = mx.split(h, 3, axis=-1)
            d = mx.concatenate([hi @ up.T for hi, up in zip(hs, ups)], axis=-1)
            return y + (d * 0.5).astype(y.dtype)

        s = b.time(lora)
        f = 2 * n * 4096 * 3 * r + 2 * n * r * 3 * 4096
        bytes_ = n * 4096 * 2 + n * 3 * r * 2 * 2 + n * 3 * 4096 * 2 * 3
        out.append({"layer": "lora.attn.qkv (r=128)", "shape": f"[{n},4096]", "s_q8": None, "s_bf16": s,
                    "tflops_bf16": f / s / 1e12, "gbps_min_traffic": bytes_ / s / 1e9})
        del x, down, ups, y
        b.free()
    res["quant"] = out


def sec_sdpa(b: Bench, res):
    mx = b.mx
    out = []
    Ns = (8192, 16384) if b.quick else (8192, 16384, 32768)
    pairs_step = (STEP_TOKENS - STEP_COND) * STEP_TOKENS + STEP_COND * STEP_COND
    for n in map(b.n, Ns):
        if b.over(f"sdpa {n}"):
            break
        q, k, v = (mx.random.normal((1, 1, n, 128)).astype(mx.bfloat16) for _ in range(3))
        mx.eval(q, k, v)
        s = b.time(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5))
        f = 4 * n * n * 128
        per_block = s / (n * n) * pairs_step * STEP_HEADS
        out.append({"heads": 1, "n": n, "s": s, "tflops": f / s / 1e12, "extrapolated_s_per_block": per_block,
                    "extrapolated_s_per_step": per_block * STEP_DEPTH})
        del q, k, v
        b.free()
    if not b.over("sdpa 32 heads"):
        n = b.n(8192)
        q, k, v = (mx.random.normal((1, 32, n, 128)).astype(mx.bfloat16) for _ in range(3))
        mx.eval(q, k, v)
        s = b.time(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5))
        s_ph = b.time(lambda: mx.concatenate([mx.fast.scaled_dot_product_attention(
            q[:, h:h + 1], k[:, h:h + 1], v[:, h:h + 1], scale=128 ** -0.5) for h in range(32)], axis=1))
        f = 32 * 4 * n * n * 128
        out.append({"heads": 32, "n": n, "s": s, "tflops": f / s / 1e12, "s_per_head_calls": s_ph,
                    "tflops_per_head_calls": f / s_ph / 1e12})
        del q, k, v
        b.free()
    res["sdpa"] = out
    res["sdpa_measured_per_block_s"] = MEASURED_SDPA_PER_BLOCK


def sec_kernels(b: Bench, res, mods):
    mx = b.mx
    out = []

    def run(name, f):
        if b.over(f"kernel {name}"):
            return
        try:
            r = f()
            r["name"] = name
            r["ok"] = True
        except Exception as e:  # compile errors etc.: record and continue
            r = {"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"[:2000],
                 "trace": traceback.format_exc()[-2000:]}
        out.append(r)
        b.free()
        print(f"  {name}: {json.dumps({k: v for k, v in r.items() if k != 'trace'})[:300]}", flush=True)

    if "adaln_mod" in mods:
        m = mods["adaln_mod"]

        def f():
            r = {"selftest": m.selftest()}
            x, sh, sc = m.make_inputs(1, 24, b.n(1024), 4096)
            mx.eval(x, sh, sc)
            port, fast = m.reference(x, sh, sc), m.reference_fast(x, sh, sc)
            got = m.kernel(x, sh, sc)
            r["vs_port"], r["vs_fast"] = maxdiff(mx, got, port), maxdiff(mx, got, fast)
            cf = mx.compile(m.reference_fast)
            r["s_kernel"] = b.time(lambda: m.kernel(x, sh, sc))
            r["s_mlx_fast_compiled"] = b.time(lambda: cf(x, sh, sc))
            r["s_mlx_port_eager"] = b.time(lambda: m.reference(x, sh, sc))
            r["shape"] = list(x.shape)
            r["gbps_kernel_min_traffic"] = x.size * 4 / r["s_kernel"] / 1e9
            return r
        run("adaln_mod", f)

    if "norm_rope" in mods:
        m = mods["norm_rope"]

        def f():
            r = {"selftest": m.selftest()}
            grid = (4, 44, 82) if b.scale == 1 else (1, 4, 8)
            qkv, wq, wk = m.make_inputs(grid)
            tables = m.rope_tables(grid)
            mx.eval(qkv, wq, wk, tables)
            port = m.reference(qkv, wq, wk, grid)
            fast = m.reference_fast(qkv, wq, wk, grid)
            got = m.kernel(qkv, wq, wk, grid, tables=tables)
            r["vs_port"] = maxdiff(mx, mx.stack(got), mx.stack(port))
            r["vs_fast"] = maxdiff(mx, mx.stack(got), mx.stack(fast))
            cf = mx.compile(lambda a, b_, c: m.reference_fast(a, b_, c, grid))
            r["s_kernel"] = b.time(lambda: m.kernel(qkv, wq, wk, grid, tables=tables))
            r["s_mlx_fast_compiled"] = b.time(lambda: cf(qkv, wq, wk))
            r["s_mlx_port_eager"] = b.time(lambda: m.reference(qkv, wq, wk, grid))
            r["tokens"] = qkv.shape[1]
            r["gbps_kernel_min_traffic"] = qkv.size * 4 / r["s_kernel"] / 1e9
            r["extrapolated_s_per_block_kernel"] = r["s_kernel"] * STEP_TOKENS / qkv.shape[1]
            r["extrapolated_s_per_block_mlx_compiled"] = r["s_mlx_fast_compiled"] * STEP_TOKENS / qkv.shape[1]
            return r
        run("norm_rope", f)

    if "qlinear8" in mods:
        m = mods["qlinear8"]

        def f():
            r = {"selftest": m.selftest()}
            x, w, s, bi, bias = m.make_inputs(b.n(8192), 4096, 4096)
            mx.eval(x, w, s, bi, bias)
            ref, mlxp = m.reference(x, w, s, bi, bias), m.mlx_path(x, w, s, bi, bias)
            got = m.kernel(x, w, s, bi, bias)
            r["vs_fp32_reference"], r["vs_mlx_quantized_matmul"] = maxdiff(mx, got, ref), maxdiff(mx, got, mlxp)
            r["mlx_vs_fp32_reference"] = maxdiff(mx, mlxp, ref)
            fl = 2 * x.shape[0] * 4096 * 4096
            r["s_kernel"] = b.time(lambda: m.kernel(x, w, s, bi, bias))
            r["s_mlx"] = b.time(lambda: m.mlx_path(x, w, s, bi, bias))
            r["tflops_kernel"], r["tflops_mlx"] = fl / r["s_kernel"] / 1e12, fl / r["s_mlx"] / 1e12
            return r
        run("qlinear8", f)

    if "flash_attn" in mods:
        m = mods["flash_attn"]

        def f():
            r = {"selftest": m.selftest()}
            rows = []
            for n in map(b.n, (4096,) if b.quick else (4096, 8192)):
                q, k, v = m.make_inputs(1, 1, n, n)
                mx.eval(q, k, v)
                ref = m.reference(q, k, v)
                got = m.kernel(q, k, v)
                fl = 4 * n * n * 128
                s_k = b.time(lambda: m.kernel(q, k, v), k=3)
                s_m = b.time(lambda: m.reference(q, k, v), k=3)
                rows.append({"n": n, **maxdiff(mx, got, ref), "s_kernel": s_k, "s_mlx_sdpa": s_m,
                             "tflops_kernel": fl / s_k / 1e12, "tflops_mlx": fl / s_m / 1e12})
            r["rows"] = rows
            return r
        run("flash_attn", f)
    res["kernels"] = out


# ------------------------------------------------------------------------------------------------ table
def table(res):
    L = []
    if "matmul" in res:
        L.append("\n(a) dense matmul")
        L.append(f"{'kind':10} {'dtype':5} {'shape':28} {'ms':>9} {'TFLOP/s':>8}")
        for r in res["matmul"]:
            L.append(f"{r['kind']:10} {r['dtype']:5} {r['shape']:28} {r['s'] * 1e3:9.2f} {r['tflops']:8.1f}")
        L.append(f"peak (best measured): {res['peak_matmul_tflops']}")
    if "quant" in res:
        L.append("\n(b) 8-bit quantized_matmul vs bf16 matmul")
        for r in res["quant"]:
            if r.get("s_q8") is None:
                L.append(f"{r['layer']:24} {r['shape']:16} bf16 {r['s_bf16'] * 1e3:8.2f} ms "
                         f"{r['tflops_bf16']:6.1f} TFLOP/s, >= {r['gbps_min_traffic']:.0f} GB/s")
            else:
                L.append(f"{r['layer']:24} {r['shape']:16} q8 {r['s_q8'] * 1e3:8.2f} ms ({r['tflops_q8']:5.1f} TF) "
                         f"bf16 {r['s_bf16'] * 1e3:8.2f} ms ({r['tflops_bf16']:5.1f} TF) q8/bf16 {r['q8_over_bf16']:.2f}")
    if "sdpa" in res:
        L.append("\n(c) SDPA, head dim 128")
        for r in res["sdpa"]:
            if r["heads"] == 1:
                L.append(f"1 head  N={r['n']:6} {r['s'] * 1e3:8.2f} ms {r['tflops']:6.1f} TF -> 86.6K step: "
                         f"{r['extrapolated_s_per_block']:.2f} s/block, {r['extrapolated_s_per_step']:.0f} s/step")
            else:
                L.append(f"32 heads N={r['n']:6} one call {r['s'] * 1e3:8.2f} ms ({r['tflops']:.1f} TF), per-head calls "
                         f"{r['s_per_head_calls'] * 1e3:8.2f} ms ({r['tflops_per_head_calls']:.1f} TF)")
        L.append(f"measured in the model: {res['sdpa_measured_per_block_s']} s/block")
    if "kernels" in res:
        L.append("\n(d) custom kernels")
        for r in res["kernels"]:
            if not r.get("ok"):
                L.append(f"{r['name']:11} FAILED: {r.get('error', '')[:200]}")
                continue
            keys = [k for k in r if k.startswith("s_") or k.startswith("tflops") or k.startswith("vs_")]
            L.append(f"{r['name']:11} " + ", ".join(
                f"{k}={r[k]['max_abs_diff']:.3g}{' (bit-identical)' if r[k]['bit_identical'] else ''}" if isinstance(r[k], dict)
                else f"{k}={r[k]:.4g}" for k in keys))
            for row in r.get("rows", []):
                L.append(f"            N={row['n']}: kernel {row['s_kernel'] * 1e3:.1f} ms ({row['tflops_kernel']:.1f} TF) vs "
                         f"SDPA {row['s_mlx_sdpa'] * 1e3:.1f} ms ({row['tflops_mlx']:.1f} TF), max diff {row['max_abs_diff']:.3g}")
    return "\n".join(L)


def main():
    a = parse()
    if not a.yes:
        print("Refusing to run without --yes (this uses the GPU for ~1-2 minutes, < 8 GB).\nRun with:\n  " + RUN_CMD)
        return 2
    if a.out is None:
        print("--out PATH is required")
        return 2
    import mlx.core as mx
    if not mx.metal.is_available() and not a.dry_run:
        print("Metal is not available: this benchmark only runs on an Apple GPU.")
        return 3
    try:
        (getattr(mx, "set_memory_limit", None) or mx.metal.set_memory_limit)(8 * 1024 ** 3)
    except Exception:
        pass
    mx.random.seed(0)
    b = Bench(mx, a.budget_s, a.quick)
    if a.dry_run:
        b.scale, b.k, b.warm = 64, 1, 0
    res = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
           "mlx": getattr(mx, "__version__", "?"), "platform": platform.platform()}
    try:
        info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
        res["device"] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in info.items()}
    except Exception as e:
        res["device"] = str(e)
    mods, kdir = load_kernels(a.kernels_dir)
    if a.dry_run and not mx.metal.is_available():  # exercise the script's paths with the MLX references
        stand_in = {"adaln_mod": lambda m: lambda x, sh, sc, eps=1e-6: m.reference_fast(x, sh, sc, eps),
                    "norm_rope": lambda m: lambda qkv, wq, wk, grid, heads=32, eps=1e-6, tables=None:
                        m.reference_fast(qkv, wq, wk, grid, heads, eps),
                    "qlinear8": lambda m: m.reference,
                    "flash_attn": lambda m: lambda q, k, v, scale=None: m.reference_tiled(q, k, v, scale)}
        for name, m in mods.items():
            m.kernel = stand_in[name](m)
    res["kernels_dir"] = kdir
    for sec, fn, label in (("a", sec_matmul, "matmul"), ("b", sec_quant, "quantized"), ("c", sec_sdpa, "sdpa"),
                           ("d", lambda bb, rr: sec_kernels(bb, rr, mods), "kernels")):
        if sec not in a.only:
            continue
        if sec == "d" and not mods:
            print("kernels not found (pass --kernels-dir); skipping (d)")
            res["kernels_error"] = "kernel files not found"
            continue
        print(f"[{time.perf_counter() - b.t0:5.1f}s] {label}", flush=True)
        try:
            fn(b, res)
        except Exception as e:
            res[f"{label}_error"] = f"{type(e).__name__}: {e}"
            print(f"  {label} failed: {e}", flush=True)
        b.free()
    res["elapsed_s"] = time.perf_counter() - b.t0
    res["peak_memory_gb"] = b.peak_gb()
    res["skipped_for_budget"] = b.skipped
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(table(res))
    print(f"\nelapsed {res['elapsed_s']:.1f} s, MLX peak {res['peak_memory_gb']:.2f} GB, JSON: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
