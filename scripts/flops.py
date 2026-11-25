#!/usr/bin/env python3
"""Analytic FLOP / byte count for one LongCat-Video-Avatar 1.6 DiT step, and the speed ceiling.

    python3 scripts/flops.py                          # both production resolutions, assumed peaks
    python3 scripts/flops.py --bench-json bench/kernels.json   # use the measured matmul peak
    python3 scripts/flops.py --size 704x1312 --text-tokens 200 --lora-rank 128

Peak FLOP/s and DRAM bandwidth of the M5 Max are NOT asserted here: they are either read from the benchmark
JSON (scripts/bench_kernels.py, key `peak_matmul_flops`) or shown for a few ASSUMED values, labelled as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shutter.kernels.flops import (  # noqa: E402
    MEASURED_704_BLOCK, MEASURED_STEP_S, MEASURED_STEP_S_ORIGINAL, OP_CLASSES, Config, attention_traffic,
    ceiling, measured_class_rates, step_counts, upper_bound,
)

ASSUMED_PEAKS_TFLOPS = (60, 80, 120)
ASSUMED_BW_GBS = (400, 550, 800)


def fmt(x, unit=""):
    for p, s in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= p:
            return f"{x / p:7.2f} {s}{unit}"
    return f"{x:7.2f} {unit}"


def print_counts(cfg: Config):
    sc = step_counts(cfg)
    t, h, w = cfg.grid
    print(f"\n=== {cfg.height}x{cfg.width}, {cfg.frames} frames -> latent grid {t}x{h}x{w} = {cfg.tokens:,} tokens "
          f"({cfg.noise_tokens:,} noise + {cfg.cond_tokens:,} cond), hidden {cfg.hidden}, {cfg.depth} blocks, "
          f"{cfg.heads} heads x {cfg.head_dim}, FFN {cfg.ffn_hidden}, text {cfg.text_tokens}, "
          f"audio {cfg.audio_tokens}/frame, LoRA r{cfg.lora_rank} x {len(cfg.lora_targets)} ===")
    print(f"{'op class':14} {'MACs/block':>14} {'FLOPs/block':>14} {'FLOPs/step':>14} {'share':>7} "
          f"{'bytes/step*':>13} {'FLOP/B':>8}")
    for c in OP_CLASSES:
        b, s = sc["per_block"][c], sc["per_step"][c]
        inten = s["flops"] / s["bytes"] if s["bytes"] else float("inf")
        print(f"{c:14} {fmt(b['macs']):>14} {fmt(b['flops'], 'F'):>14} {fmt(s['flops'], 'F'):>14} "
              f"{100 * sc['share'][c]:6.2f}% {fmt(s['bytes'], 'B'):>13} {inten:8.0f}")
    bt, st = sc["per_block_total"], sc["per_step_total"]
    print(f"{'TOTAL':14} {fmt(bt['macs']):>14} {fmt(bt['flops'], 'F'):>14} {fmt(st['flops'], 'F'):>14} "
          f"{100.0:6.2f}% {fmt(st['bytes'], 'B'):>13}")
    print(f"  matmul FLOPs/step {fmt(st['matmul_flops'], 'F')}, elementwise {fmt(st['elem_flops'], 'F')} "
          f"({100 * st['elem_flops'] / st['flops']:.2f}%).  *bytes = compulsory traffic, one kernel per op, bf16 "
          f"activations, 8-bit weights (+ scale/bias per 64); attention counted as a fused kernel.")
    return sc


def print_ceiling(cfg: Config, measured_s: float, peaks, bws, label):
    print(f"\n  ceiling at the measured {measured_s:.1f} s/step ({label})")
    print(f"  {'peak (matmul)':24} {'achieved':>12} {'util':>6} {'all-at-peak':>12} {'roofline (BW)':>26}")
    for name, p in peaks:
        c0 = ceiling(measured_s, p, cfg)
        roofs = []
        for bw_name, bw in bws:
            c = ceiling(measured_s, p, cfg, bandwidth=bw)
            roofs.append(f"{c['roofline_s']:.1f}s@{bw_name}")
        flag = "  (achieved > assumed peak: assumption too low)" if c0["utilisation"] > 1 else ""
        print(f"  {name:24} {fmt(c0['achieved_flops'], 'F/s'):>12} {100 * c0['utilisation']:5.0f}% "
              f"{c0['best_case_s']:10.1f} s  {', '.join(roofs)}{flag}")


def print_attention(cfg: Config, bws):
    print(f"\n  self-attention memory traffic, one block ({cfg.heads} heads, {cfg.tokens:,} tokens)")
    a2 = attention_traffic(cfg, score_bytes=2)
    print(f"  scores per head: {a2['entries_per_head_full']:.3e} entries for N x N "
          f"({a2['pairs_per_head']:.3e} with the cond/noise split); x {cfg.heads} heads per block")
    rows = [("naive, fp32 scores", attention_traffic(cfg, score_bytes=4)["naive_bytes"]),
            ("naive, bf16 scores", a2["naive_bytes"])]
    for bq in (32, 64, 128):
        rows.append((f"fused, K/V re-read per {bq}-row tile", attention_traffic(cfg, bq=bq)["fused_streamed_bytes"]))
    rows.append(("fused, compulsory (Q,K,V,O once)", a2["fused_compulsory_bytes"]))
    hdr = "".join(f"{'@' + n:>12}" for n, _ in bws)
    print(f"  {'variant':38} {'bytes/block':>12} {'bytes/step':>12} {'FLOP/B':>8} {hdr}   (time per step if DRAM-bound)")
    for name, by in rows:
        ts = "".join(f"{by * cfg.depth / bw:11.1f}s" for _, bw in bws)
        print(f"  {name:38} {fmt(by, 'B'):>12} {fmt(by * cfg.depth, 'B'):>12} {a2['flops'] / by:8.1f} {ts}")


def print_measured(cfg: Config, peaks):
    print("\n  measured per-block components at 704x1312 (REPORT.md, eager, M5 Max) -> achieved rate")
    for r in measured_class_rates(cfg):
        if "achieved_flops" in r:
            utl = "  ".join(f"{100 * r['achieved_flops'] / p:4.0f}% of {n.split()[0]}" for n, p in peaks)
            print(f"  {r['part']:38} {r['seconds']:6.3f} s  {fmt(r['achieved_flops'], 'F/s'):>12}   {utl}")
        else:
            print(f"  {r['part']:38} {r['seconds']:6.3f} s  {fmt(r['achieved_bytes_per_s'], 'B/s'):>12} "
                  f"(compulsory traffic / time: the MLX chain moves far more)")


def print_bounds(cfg: Config, peaks):
    sc = step_counts(cfg)
    sdpa = sc["per_block"]["self_attn_qk"]["matmul_flops"] + sc["per_block"]["self_attn_av"]["matmul_flops"]
    rate = sdpa / MEASURED_704_BLOCK["sdpa_per_head"]
    share = MEASURED_704_BLOCK["sdpa_per_head"] * cfg.depth / MEASURED_STEP_S[(704, 1312)]
    print(f"\n  Amdahl bound for a better self-attention kernel (SDPA = {100 * share:.0f}% of the 164 s step, "
          f"running at {fmt(rate, 'F/s')}):")
    for name, p in peaks:
        u = min(rate / p, 1.0)
        print(f"  {name:24} SDPA at {100 * u:4.0f}% of peak -> best overall speed-up "
              f"{upper_bound(share, u):.2f}x ({MEASURED_STEP_S[(704, 1312)] / upper_bound(share, u):.0f} s/step)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", action="append", help="HxW (repeatable); default 704x1312 and 480x832")
    ap.add_argument("--frames", type=int, default=93)
    ap.add_argument("--text-tokens", type=int, default=512)
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--num-cond-latents", type=int, default=1)
    ap.add_argument("--peak-tflops", type=float, action="append", help="assumed matmul peak (repeatable)")
    ap.add_argument("--bandwidth-gbs", type=float, action="append", help="assumed DRAM bandwidth (repeatable)")
    ap.add_argument("--bench-json", type=Path, help="bench_kernels.py output: use its measured bf16 peak")
    ap.add_argument("--step-seconds", type=float, help="measured s/step for a non-default size")
    ap.add_argument("--json", type=Path, help="also write the counts as JSON")
    a = ap.parse_args()

    peaks = [(f"{p:g} TF/s ASSUMED", p * 1e12) for p in (a.peak_tflops or ASSUMED_PEAKS_TFLOPS)]
    if a.bench_json:
        j = json.loads(a.bench_json.read_text())
        if j.get("peak_matmul_flops"):
            peaks.insert(0, (f"{j['peak_matmul_flops'] / 1e12:.1f} TF/s MEASURED", j["peak_matmul_flops"]))
    bws = [(f"{b:g}GB/s?", b * 1e9) for b in (a.bandwidth_gbs or ASSUMED_BW_GBS)]
    print("Peaks and bandwidths marked ASSUMED / '?' are assumptions, not M5 Max specs; measure with "
          "scripts/bench_kernels.py and pass --bench-json.")

    sizes = a.size or ["704x1312", "480x832"]
    out = {}
    for s in sizes:
        hh, ww = (int(v) for v in s.lower().split("x"))
        cfg = replace(Config(), height=hh, width=ww, frames=a.frames, text_tokens=a.text_tokens,
                      lora_rank=a.lora_rank, num_cond_latents=a.num_cond_latents)
        sc = print_counts(cfg)
        meas = a.step_seconds if a.step_seconds else MEASURED_STEP_S.get((hh, ww))
        if meas:
            print_ceiling(cfg, meas, peaks, bws, "measured on the M5 Max, per-head SDPA default")
        if (hh, ww) in MEASURED_STEP_S_ORIGINAL:
            print_ceiling(cfg, MEASURED_STEP_S_ORIGINAL[(hh, ww)], peaks[:1], bws[:1], "original path, one SDPA call")
        print_attention(cfg, bws)
        if (hh, ww) == (704, 1312):
            print_measured(cfg, peaks)
            print_bounds(cfg, peaks)
        out[s] = {"tokens": cfg.tokens, "per_step_total": sc["per_step_total"], "share": sc["share"],
                  "per_step": sc["per_step"], "per_block": sc["per_block"]}
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
