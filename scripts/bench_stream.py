#!/usr/bin/env python3
"""Measure what SSD weight streaming would cost for a DiT on this Mac (read-only, no GPU).

    python scripts/bench_stream.py /path/to/model_dir --sec-per-step 90 --yes

Prints per-block sizes, sequential read throughput of whole blocks (first read: cold
unless the files were read recently; second read: warm page cache) and the projected
streaming overhead per denoise step. Nothing is written; only os.pread is used.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from shutter.core.stream import SafetensorsIndex, group_names_by_block  # noqa: E402

MB = 1024 * 1024


def read_blocks(index, groups):
    t0 = time.perf_counter()
    nbytes = 0
    for g in groups:
        for name in g:
            nbytes += index.read(name).nbytes
    return nbytes, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", help="directory holding the DiT *.safetensors shards")
    ap.add_argument("--glob", default="*.safetensors", help="shard file pattern (default %(default)s)")
    ap.add_argument("--pattern", default=r"(^|\.)blocks\.(\d+)\.", help="regex whose last group is the block number")
    ap.add_argument("--blocks", type=int, default=0, help="read only the first N blocks (default: all)")
    ap.add_argument("--sec-per-step", type=float, default=90.0, help="GPU seconds per denoise step")
    ap.add_argument("--steps", type=int, default=8, help="denoise steps per clip")
    ap.add_argument("--yes", action="store_true", help="actually read the data")
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(a.model_dir, a.glob)))
    if not paths:
        print(f"no files matching {a.glob} in {a.model_dir}", file=sys.stderr)
        return 1
    index = SafetensorsIndex(paths)
    groups, rest = group_names_by_block(index.names(), a.pattern)
    if not groups:
        print("no block tensors found; adjust --pattern", file=sys.stderr)
        return 1
    sizes = [index.nbytes(g) for g in groups]
    print(f"{len(paths)} file(s), {len(index)} tensors, {index.nbytes() / MB:,.0f} MB total")
    print(f"{len(groups)} blocks: min {min(sizes) / MB:,.1f} MB, max {max(sizes) / MB:,.1f} MB, "
          f"mean {sum(sizes) / len(sizes) / MB:,.1f} MB; always-resident rest {index.nbytes(rest) / MB:,.1f} MB")
    for i, s in enumerate(sizes):
        print(f"  block {i:3d}: {s / MB:9.1f} MB  ({len(groups[i])} tensors)")

    sel = groups[: a.blocks] if a.blocks > 0 else groups
    sel_bytes = sum(index.nbytes(g) for g in sel)
    if not a.yes:
        print(f"\nwould read {len(sel)} block(s) 3 times: {3 * sel_bytes / MB:,.0f} MB from disk/cache "
              f"(peak RAM ~{max(sizes) / MB:,.0f} MB). Re-run with --yes to measure.")
        return 2

    if sys.platform == "darwin":
        index.set_nocache(True)  # first pass: do not fill the page cache
    b1, t1 = read_blocks(index, sel)
    index.set_nocache(False)
    read_blocks(index, sel)  # populate the page cache
    b2, t2 = read_blocks(index, sel)
    index.close()
    cold = b1 / max(t1, 1e-9)
    warm = b2 / max(t2, 1e-9)
    print(f"\nfirst read : {b1 / MB:,.0f} MB in {t1:.2f}s = {cold / MB:,.0f} MB/s (cold unless recently read)")
    print(f"warm read  : {b2 / MB:,.0f} MB in {t2:.2f}s = {warm / MB:,.0f} MB/s (page cache)")

    n = len(groups)
    compute = a.sec_per_step / n
    for label, bw in (("cold", cold), ("warm", warm)):
        reads = [s / bw for s in sizes]
        # steady state: block i+1 is read while block i computes; only the excess stalls.
        # The first block of the first step is not overlapped (counted once per clip).
        stall = sum(max(0.0, r - compute) for r in reads)
        per_clip = stall * a.steps + reads[0]
        pct = 100.0 * per_clip / (a.sec_per_step * a.steps)
        print(f"{label}: read {max(reads):.3f}s/block vs compute {compute:.2f}s/block -> "
              f"~{per_clip:.1f}s per {a.steps}-step clip ({pct:.2f}% overhead, "
              f"excl. host->MLX copy)")
    print(f"memory: ~{2 * max(sizes) / MB:,.0f} MB for window=2 instead of {sum(sizes) / MB:,.0f} MB of blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
