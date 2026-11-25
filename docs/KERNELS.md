# Custom Metal kernels: where the time goes and what a kernel can still win

Model: LongCat-Video-Avatar 1.6 DiT (48 blocks, hidden 4096, 32 heads x 128, SwiGLU FFN 11008, 8-bit
`QuantizedLinear` group 64, bf16 activations, runtime distill LoRA rank 128 on 7 linears per block),
93 frames -> 24 latent frames, 1 conditioning (reference) frame. Numbers come from
`python3 scripts/flops.py` (analytic) and the M5 Max profile in shutter-ref `speed/exact/REPORT.md` (measured).

## FLOPs per step (one DiT pass; the distilled model runs one pass per step)

| op class | 704x1312 (86,592 tok) | share | 480x832 (37,440 tok) | share |
|---|---|---|---|---|
| self-attn Q K^T | 2.83 PF | 35.1% | 0.53 PF | 25.4% |
| self-attn P V | 2.83 PF | 35.1% | 0.53 PF | 25.4% |
| softmax (elementwise) | 0.06 PF | 0.7% | 0.01 PF | 0.5% |
| FFN (SwiGLU, 3 linears) | 1.12 PF | 13.9% | 0.49 PF | 23.3% |
| qkv linear | 0.42 PF | 5.2% | 0.18 PF | 8.7% |
| text cross-attn (all) | 0.30 PF | 3.8% | 0.13 PF | 6.3% |
| audio cross-attn (all) | 0.27 PF | 3.3% | 0.12 PF | 5.6% |
| attn proj | 0.14 PF | 1.7% | 0.06 PF | 2.9% |
| LoRA (r128, 7 targets) | 0.09 PF | 1.1% | 0.04 PF | 1.9% |
| norms, RoPE, modulation, other | <0.002 PF | <0.1% | <0.001 PF | <0.1% |
| **total** | **8.07 PF** | | **2.09 PF** | |

Text tokens are counted at the 512 pad length (upper bound; packing keeps only the real prompt), which
changes nothing visible. Self-attention is noise queries (82,984) x all keys plus cond x cond, as in production.

**Achieved:** 8.07 PF / 164.0 s = **49.2 TFLOP/s** at 704x1312; 2.09 PF / 45.7 s = **45.6 TFLOP/s** at 480x832
(38.5 TFLOP/s for the original 209.7 s path).

Per component (one block, 704x1312, measured eager):

| part | s/block | achieved |
|---|---|---|
| SDPA, one call | 3.00 | 39.3 TFLOP/s |
| **SDPA, per head (default)** | **2.084** | **56.6 TFLOP/s** |
| qkv q8 matmul | 0.16 | 54.5 TFLOP/s (dequant + matmul throughput, directly) |
| attn proj q8 | 0.06 | 48.4 TFLOP/s |
| FFN q8 + LoRA | 0.57 | 42.9 TFLOP/s |
| text / audio cross-attn | 0.21 / 0.19 | ~30 TFLOP/s |
| qkv LoRA alone | 0.11 | **5.0 TFLOP/s** (skinny rank-128 matmuls: bandwidth bound) |
| q/k RMSNorm + RoPE (fp32) | 0.27 | 21 GB/s of *compulsory* traffic (the MLX chain moves far more) |

Already measured and not helpful: fp16 SDPA = bf16 (3.01 s), fp32 SDPA 1.76x slower (5.27 s), fp16 q8 matmul
2x slower, query chunking slower, fused per-head norm+RoPE+SDPA in MLX ops 5% slower when compiled.

## Attention memory traffic (one block, 32 heads, 704x1312)

N x N = 7.5e9 scores per head (7.2e9 with the cond split), 2.3e11 per block.

| variant | bytes/block | bytes/step | FLOP/byte |
|---|---|---|---|
| naive, fp32 scores written + read back, P written + read | 2.77 TB | 133 TB | 43 |
| naive, bf16 scores | 1.85 TB | 89 TB | 64 |
| fused, K/V re-read from DRAM by every 32-row query tile | 3.69 TB | 177 TB | 32 |
| fused, 64-row tiles | 1.84 TB | 89 TB | 64 |
| fused, 128-row tiles | 0.92 TB | 44 TB | 128 |
| fused, compulsory (Q, K, V, O once) | 2.8 GB | 136 GB | 41,567 |

At an assumed 400 / 550 / 800 GB/s, naive bf16 would take 222 / 161 / 111 s per step on traffic alone; the
compulsory traffic is 0.2-0.3 s. With D = 128, a tiled kernel's K/V streaming costs 512/BQ bytes per score,
the same as naive bf16 scores at BQ = 64: what saves a flash kernel is that concurrent query tiles hit the
same K/V blocks in the GPU caches. MLX's SDPA is already such a kernel (working memory 0-1.4 GB at 86.6K
tokens), so "don't materialise the scores" is already done; the question is only how close it runs to peak.

## What the benchmark tells us (`scripts/bench_kernels.py`)

1. **The bf16 matmul peak P** (best of [N,4096]x[4096,4096], FFN shape, [N,128]x[128,N'] tiles). Feed it
   back: `python3 scripts/flops.py --bench-json bench/kernels.json`.
2. **q8 / bf16 ratio** per layer shape: if q8 is within ~10% of bf16, `qlinear8` cannot win on the matmul.
3. **SDPA rate at 8K-32K tokens** extrapolated to the step, vs the measured 2.084 s/block (checks the model).
4. **Each kernel vs its MLX version**: time, max-abs-diff, bit-identical yes/no, vs both the port's ops and
   the production fast path (`mx.fast.rms_norm`, `mx.fast.layer_norm`, compiled).

## Honest upper bound

SDPA is 61% of the 164 s step (2.084 s x 48) at 56.6 TFLOP/s. Amdahl, if a perfect attention kernel ran
at P:

| P (bf16 matmul peak) | SDPA at | best step | best overall gain |
|---|---|---|---|
| 60 TFLOP/s (assumed) | 94% | 158 s | 1.04x |
| 80 TFLOP/s (assumed) | 71% | 135 s | 1.22x |
| 120 TFLOP/s (assumed) | 47% | 111 s | 1.48x |

If every op of the step ran at P: 134 s (P = 60) / 101 s (80) / 67 s (120). The q8 qkv matmul already
reaches 54.5 TFLOP/s, so unless the benchmark finds a dense peak well above ~60-65 TFLOP/s, **the whole model
is within ~1.2x of the hardware limit and no kernel can give more than that**. Real headroom, largest first:

- **q/k RMSNorm + RoPE** (6% eager, 0.27 s/block): compulsory traffic is 5.7 GB/block, ~10-15 ms at
  several hundred GB/s. `norm_rope` could save up to ~0.25 s/block, ~5-7% of the step, if the compiled
  block has not already fused most of it (the benchmark compares against the compiled fast path).
- **Runtime LoRA** (4% of the step, running at 5 TFLOP/s): fuse the LoRA up-projection into the q8 matmul
  epilogue, or bake it (already evaluated: ~4.5%). At most ~4%.
- **Self-attention**: only if P >> 60 TFLOP/s (see table).
- Cross-attention (8.5%, ~30 TFLOP/s): at most ~4%. adaLN modulation (1%): nothing worth a kernel.

Algorithmic changes are an order of magnitude larger than any kernel (ROI crop 2.48x per step, REPORT C).

## Exactness

Shuttrr's rule: default-on speed-ups must be bit-identical. **None of these kernels is guaranteed
bit-identical**, so all are opt-in:

| kernel | vs MLX path | why |
|---|---|---|
| `adaln_mod` | close (~1 bf16 ulp) | mean/variance reduction order differs from MLX's reduce / layer_norm |
| `norm_rope` | close (~1 bf16 ulp); RoPE arithmetic identical | RMS reduction order (simd_sum) differs |
| `qlinear8` | close | accumulation order differs from MLX's qmm |
| `flash_attn` | close (bf16 level) | online softmax rescaling + tile order |
| `block_sparse` | **approximation** (sparsity 0: exact) | drops key chunks; see "Opt-in speed techniques" below |

The benchmark reports `bit_identical` on random data; one with max diff 0 on real activations
(through `speedlab/cmp.py`) could be promoted. All accumulate in fp32.

## Order to try

1. Run the benchmark; get P. If P <= ~65 TFLOP/s, stop working on attention and matmul kernels.
2. `norm_rope` (largest realistic win, simplest kernel, also removes the transposed qkv copy and the fp32
   temporaries that set the 14 GB working memory).
3. LoRA epilogue fused into the q8 matmul (start from `qlinear8`; only worth it if `qlinear8` gets near
   `mx.quantized_matmul`, which it is not expected to do at first).
4. `flash_attn` only if P is much larger than 56.6 TFLOP/s: bf16 K/V tiles, BQ = 64, double buffering.
5. `adaln_mod`: last, ~1%.

## Kernels (shutter/kernels/)

Written with `mx.fast.metal_kernel`, **not yet run on Metal** (this container has CPU-only MLX). Each has a
pure-MLX `reference` (the model's ops, used when Metal is unavailable), `kernel`, and `selftest()`.
API used (MLX 0.32.4, from the installed package's docstrings and generated-source strings; the online
docs pages were truncated by the fetcher): `metal_kernel(name, input_names, output_names, source, header,
ensure_row_contiguous=True, atomic_outputs=False, compile_options)`, called with
`inputs, template=[(name, dtype|int|bool)], grid (threads, dispatchThreads), threadgroup, output_shapes,
output_dtypes, init_value, verbose`; attributes such as `thread_position_in_grid` are added when the body
uses them.

Run on the Mac (copy `scripts/bench_kernels.py` and `shutter/kernels/*.py` into `~/shutter-bench/` and
`~/shutter-bench/kernels/`):

    ~/longcat-mlx/speed/gpulock.sh shutter-bench -- ~/longcat-mlx/code/.venv/bin/python ~/shutter-bench/bench_kernels.py --yes --out ~/shutter-bench/bench/kernels.json

## Opt-in speed techniques (NOT bit-identical)

Selected per job with `params.speed` (absent = off = the reference path, untouched):

    {"speed": {"block_sparse": {"sparsity": 0.9375, "chunk": [4, 4, 4], "keep_cond": true},
               "step_cache":   {"threshold": 0.1, "warmup": 1}, "strict": false}}

`true` instead of an object means the defaults shown. The job result gets `identical: false` and a `speed`
section: what was requested, whether each option was applied or refused (and why), the attention patterns (query /
key chunks, kept and skipped chunk pairs per head, `score_fraction` = attention scores computed / dense) and the
dense fallbacks with their reasons, the step-cache counts (computed / skipped calls, which steps were skipped, the
per-step relative L1 of the indicator, steps that never reached the cache). `strict: true` makes a refusal an error.
Code: `shutter/kernels/block_sparse.py`, `shutter/core/step_cache.py`, LongCat glue
`shutter/adapters/longcat_speed.py`, tests `tests/test_speed.py`, GPU timing `scripts/bench_speed.py`.

### Block-sparse self-attention (`block_sparse`)

LongCat-Video's BSA (`flash_attn_bsa_3d`, base-model config: sparsity 0.9375, 4x4x4 chunks for Q and K) re-done in
MLX ops: mean-pool Q and K per 4x4x4 chunk of the latent patch grid, score every (query chunk, key chunk) pair with
the pooled dot product, keep the top `ceil((1 - sparsity) x n)` key chunks per query chunk, gather them and run the
fused dense SDPA on [64 queries] x [kept keys]. **The Avatar 1.5/1.6 config has `enable_bsa: false`**: the model was
trained and distilled dense, so this is an approximation at every sparsity > 0 (it is exact at 0, where the dense
SDPA is called directly: tested bit-identical, also through the adapter hook with `mx.compile` on and off).

Avatar specifics and how they are handled:

| situation | handling |
|---|---|
| noise queries attend to [conditioning/reference frame(s) + noise frames] | the conditioning tokens are a dense prefix, kept for every query chunk (`keep_cond`, default on: the reference frame carries the identity); the noise frames are chunked |
| sizes not divisible by 4 (23 noise frames; 30x52 and 44x82 patches) | zero-padded chunks, padded keys masked out of the softmax, padded queries dropped (upstream assumes divisible grids) |
| conditioning frames attending to themselves (1 frame, or ref + 4 cond in continuation) | dense (below `min_q_tokens`, default 4096) |
| continuation KV cache (`kv_cache.forward_noise`) | supported: it calls the same `_process_attn`; the cached ref + cond keys are the dense prefix; the cache itself is built densely |
| Reference-Skip attention (continuation with `mask_frame_range > 0`) | dense for those calls (its masked query slices do not line up with chunk grids); reported |
| `LC_ATTN_FUSED=1`, `LC_TWIN>0` (bypass `_process_attn`), SSD streaming rungs | refused: job runs without it, reason in the result |

What it computes at the real sizes (`block_sparse.pattern`, per head, first segment, cond frame kept):

| size | query chunks | kept key chunks per query chunk at 0.5 / 0.75 / 0.9375 | scores vs dense |
|---|---|---|---|
| 480x832 (37,440 tokens) | 624 (+25 prefix) | 337 / 181 / 64 | 0.641 / 0.344 / 0.122 |
| 704x1312 (86,592 tokens) | 1386 (+57 prefix) | 750 / 404 / 144 | 0.593 / 0.319 / 0.114 |

The always-kept prefix and the padding keep the score fraction above 1 - sparsity.

**Measured quality vs dense** (`block_sparse.quality()`, CPU, fp32, 2 heads x 64, synthetic inputs; PSNR with
peak = max |dense output|). `random` = i.i.d. Q/K/V (near-uniform attention: the worst case); `structured` = Q and K
share smooth low-frequency features of the 3D position plus noise (local, content-dependent attention);
`peaked` = the same with 1.2x sharper logits:

| inputs | grid | sparsity 0.5 | 0.75 | 0.9375 |
|---|---|---|---|---|
| random | 8x16x16 | 16.2 dB, cos 0.710 | 11.5 dB, 0.505 | 4.7 dB, 0.256 |
| structured | 8x16x16 | 20.4 dB, cos 0.9968 | 14.9 dB, 0.9926 | 10.7 dB, 0.9850 |
| peaked | 8x16x16 | 37.1 dB, cos 0.99989 | 32.6 dB, 0.99974 | 29.3 dB, 0.99935 |
| random | 6x10x14 + 1 cond frame | 17.5 dB, cos 0.759 | 13.0 dB, 0.563 | 9.9 dB, 0.430 |
| structured | 6x10x14 + 1 cond frame | 23.1 dB, cos 0.9980 | 18.3 dB, 0.9956 | 15.0 dB, 0.9732 |
| peaked | 6x10x14 + 1 cond frame | 41.8 dB, cos 0.99996 | 38.1 dB, 0.99992 | 19.6 dB, 0.9787 |

The error swings by 20 dB with how concentrated attention is, so these numbers bracket, they do not predict, the
real model. Per-layer error on real activations (capture q/k/v of a few blocks with `speedlab/cmp.py`-style hooks)
and the decoded video against the dense reference are what decide whether any sparsity is usable for Avatar.

**Speed: not measured.** This container has CPU-only MLX. The pure-MLX version pays a gather of every kept K/V chunk
per query chunk (bounded per group by `max_gather_mb`) and runs SDPA on 64-row query tiles, so the time saved is
not the score fraction: it depends on gather bandwidth and on how well MLX's SDPA runs short query tiles. Run
`scripts/bench_speed.py` on the Mac (under the GPU lock, as `bench_kernels.py`); it times dense (per-head SDPA, the
production default) against each sparsity at 37,440 and 86,592 tokens, 32 heads x 128, bf16, and writes JSON:

    ~/longcat-mlx/speed/gpulock.sh shutter-speed -- ~/longcat-mlx/code/.venv/bin/python ~/shutter-bench/bench_speed.py --yes --out ~/shutter-bench/bench/speed.json

If the gather dominates, the next step is a Metal kernel that walks a per-query-tile list of kept key chunks
(flash-attention over an index list, no gathered copy), starting from `flash_attn.py`.

### Step caching (`step_cache`, TeaCache, arXiv 2411.19108)

Per DiT call: indicator = block 0's timestep-modulated input (`modulate_fp32(mod_norm_attn, x, shift_msa,
scale_msa)`, exactly what the block computes); accumulate the relative L1 change of the indicator between steps
(optionally through a fitted polynomial, `coefficients`); while the sum is under `threshold`, skip all 48 blocks
and add the previous computed step's residual (block-stack output minus input, kept in fp32) to the input. The
first `warmup` steps and the last step always compute; the cache is invalidated when the runtime-LoRA multipliers
change (1.6's DMD -> OPSD switch); each call position in a step (e.g. the audio-guidance pass) has its own cache.
`threshold: 0` never skips and is tested identical to no caching (real tree, real `_denoise` loop, compile on/off).
Memory: one fp32 residual per call position, 4 B x tokens x 4096 (0.6 GB at 37,440 tokens, 1.4 GB at 86,592).

Be realistic about LongCat: Avatar 1.6 runs a **distilled 8-step schedule** with large sigma jumps, so the
modulated input changes a lot every step; with warm-up and the last step fixed only 6 of 8 steps are even
candidates, and at the thresholds TeaCache uses (0.1-0.3 on 30-50 step samplers) little or nothing is expected to
be skipped; each skipped step there is also a much larger error than on a long schedule. **Continuation segments
are never cached**: they run through the KV-cache path, which calls the compiled block functions directly
(reported as `steps_without_cached_calls`); the same holds for the first segment with `LC_FIRST_KV=1`. The helper
is model-agnostic (`TeaCache.run(x_in, indicator, stack_fn)`); it is aimed at long-schedule image/video models
added later. No skip counts or quality on the real LongCat model have been measured yet.

## Chip-level options

### M5 GPU neural accelerators (Metal 4 tensor APIs)

M5-family GPUs add a matrix-multiply unit ("Neural Accelerator") to every GPU core. Software reaches it through
Metal 4's tensor APIs: `MTLTensor` resources and, inside shaders, the Metal Performance Primitives
(`#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`, `mpp::tensor_ops::matmul2d` executed
cooperatively by a SIMD-group or threadgroup on tensor views of device / threadgroup memory). It only matters when
all of these hold, which `scripts/chip_probe.sh` reports (chip name, macOS version, MLX version, Metal device info,
a 4096^2 matmul rate in fp16 / bf16 / fp32):

- an M5-family chip (the measured profile in this file is from an M5 Max);
- macOS 26.2 or newer (the first release that exposes the accelerators to Metal, per Apple's MLX material);
- an MLX build whose matmul / attention kernels use the tensor APIs on M5 (check the release notes of the MLX
  version the probe prints; ours were written against MLX 0.32.4 on Linux, CPU only).

Vendor figures, not measured by us: Apple states the M5 GPU delivers over 4x the peak GPU compute for AI of M4, and
Apple's MLX write-up reports up to ~4x faster time-to-first-token for LLMs on M5 vs M4 (compute-bound prefill;
token generation, which is bandwidth-bound, gains much less). Our DiT step is compute-bound (dense matmuls and
attention at 45-57 TFLOP/s achieved), so it is the kind of workload these figures are about, but **no speed-up for
Shuttrr has been measured**.

What a port of our kernels would involve:

- Our `mx.fast.metal_kernel` kernels (`flash_attn`, `qlinear8`) use plain MSL with SIMD-group 8x8 matrix ops, which
  do not use the accelerators. A port rewrites their inner products as `matmul2d` on cooperative tensors: Q.K^T and
  P.V tiles in `flash_attn` (online softmax stays SIMD code), the dequantized-int8 x bf16 tile product in
  `qlinear8` (dequantize to bf16 in threadgroup memory first, unless the tensor ops take int8 directly: unverified),
  fp32 accumulation, tile shapes chosen for what `matmul2d` supports.
- First check, before any kernel work: whether `mx.fast.metal_kernel` can compile Metal 4 code at all (language
  version, the MPP header) on the Mac's macOS / MLX; a 20-line test kernel answers it. If it cannot, the route is a
  custom MLX primitive / extension built against the macOS 26 SDK, a much larger job.
- Keep the current kernels as the fallback for non-M5 machines; dispatch on the probe result.
- Block-sparse attention would be the natural first tensor-API kernel: one flash-attention loop over a per-tile list
  of kept key chunks, which also removes the gathered K/V copies of the pure-MLX version.
- Exactness: accelerator matmuls accumulate in a different order, so all of this stays opt-in like every kernel here.

What can only be decided after the probe (and `bench_kernels.py` / `bench_speed.py`) runs on the Mac:

1. Not M5-family, or macOS < 26.2: nothing at chip level applies; stop here.
2. M5 + macOS 26.2+: does the installed MLX already use the accelerators? Compare the probe's bf16 / fp16 matmul
   rate and `bench_kernels.py`'s peak P with the numbers in this file (56.6 TFLOP/s SDPA, 54.5 TFLOP/s q8 qkv). If
   they are unchanged, the profile above was taken without the accelerators and upgrading macOS / MLX is the first,
   free, experiment (re-measure; it is not bit-identical to the old build either, so the reference outputs must be
   regenerated on the new build).
3. If MLX's matmul uses them but its SDPA or `quantized_matmul` does not (compare section (c) and (b) rates with
   the matmul peak), that op is where a tensor-API port pays off; the "Honest upper bound" table must be redone
   with the new P, because a much higher P makes attention and FFN kernels worth more than this file concludes.
4. Faster dense compute makes block-sparse relatively less attractive (gathers are bandwidth-bound and do not get
   faster), so `bench_speed.py` must be re-run on the final macOS / MLX before choosing any sparsity.
