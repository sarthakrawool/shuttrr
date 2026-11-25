# metal4_bench: do M5 TensorOps beat what MLX already gets?

A standalone Swift + Metal benchmark that times one GEMM, `C[M,N] = A[M,K] @ B[K,N]`, three
ways on the default Metal device. It verifies the TensorOps results against the plain kernel
and a CPU reference, and writes JSON.

**Status: not compiled or run by the author.** It was written on Linux, with no Swift and no
Metal. Every API call is taken from Apple's documents or from MLX's shipped kernels (sources
below), but the first build on the Mac may still need the small fixes listed under
[Likely compile errors](#likely-compile-errors).

## Read this first: MLX already uses the accelerators

The research done for this benchmark mostly answers the question already:

* **MLX added "Support for Neural Accelerators on M5 (macOS >= 26.2)" in v0.30.0**
  (PR #2772, "Add Neural Accelerator Support"). The kernels are called "NAX" in MLX.
* In **MLX 0.32.4**, `matmul` (steel GEMM: fused, split-K, segmented, gather and thin
  variants), `scaled_dot_product_attention` (head dims 64/96/128/256/512, plus 72 and 80
  padded to 96), quantized matmul and gated-delta all go through `mpp::tensor_ops::matmul2d`.
  This happens when `is_nax_available()` is true: the OS is macOS 26.2 or newer, and the
  GPU is generation 17 or newer (generation 18 for the `p` architecture suffix).
* The PyPI wheel `mlx_metal-0.32.4-py3-none-macosx_26_0_arm64.whl` really contains those
  kernels. Its `mlx.metallib` has about 7,900 `*_nax_*` symbols, for example
  `steel_gemm_splitk_nax_nn_bfloat16_float32_bm64_bn64_bk256_wm2_wn2`,
  `steel_attention_*` built from `steel_attention_nax.metal`, and
  `mpp::tensor_ops::matmul2d_descriptor<16,32,16,false,false,true,mode 1>` in the mangled
  names. This was checked by downloading the wheel and running `strings` on it.
* fp32 matmul also uses TensorOps by default: `MLX_ENABLE_TF32` defaults to 1, and the NAX
  path is taken for float32 when it is set.
* Our own numbers point the same way. Apple lists the M5 Max 40-core GPU at about
  **16.6 TFLOP/s FP32** for ordinary shader ALUs. fp16 simdgroup matrix math runs at about
  the same rate, which caps a plain simdgroup_matrix kernel at roughly 17 TFLOP/s. Our
  measured MLX rates (bf16 57.9, fp16 53.8, fp32 39.8 TFLOP/s) are 2.4 to 3.5 times that
  cap. That is only possible on the neural accelerators.
* **Estimated TensorOps peak (unverified):** an independent A19 benchmark measured about
  1,024 fp16 FLOP per GPU core per clock on the accelerators, against 256 on the SIMD pipe.
  It measured about twice that for int8. Applied to M5 Max (40 cores, about 1.62 GHz
  implied by 16.6 TFLOP/s ÷ (40 × 256)), this gives **about 66 TFLOP/s fp16/bf16** and about
  130 TOPS int8. MLX's 57.9 bf16 would then be about 87% of peak, and its 53.8 fp16 about 81%.
* **Newer MLX:** **0.32.5** appeared on PyPI on 2026-11-25. It adds a `macosx_27_0` wheel and
  `mx.metal.is_nax_available()`. It also rewrites a lot of SDPA (NAX "split" kernels for
  short query blocks, d512 changes) and quantized code. Its GitHub release page could not be
  fetched yet. Nothing in the diff points to large gains for long dense GEMMs or long
  non-causal attention, but upgrading is free, so measure it first (see below).

So the realistic headroom for a hand port is the gap between MLX and the TensorOps peak,
probably 10 to 25%. It is not the "4-8x" Apple quotes, because that figure compares against
simdgroup_matrix code. This benchmark measures that gap directly on our shapes.

## What it runs

| label | what it is |
|---|---|
| `a_simdgroup` | Plain MSL GEMM with `simdgroup_matrix` 8x8 and float fragments, in the style of MLX's pre-M5 "steel" kernel: 64x64x32 threadgroup tile staged in threadgroup memory, 2x2 simdgroups of 32x32, fp32 accumulation, Morton tile order. No accelerators. This is the "what plain Metal gets" baseline. It is reasonably tuned, not MLX's exact kernel. |
| `b_tensorops_basic` | The basic form from the MSL 4.1 spec example and Apple's tech talk. One `matmul2d` per threadgroup, `execution_simdgroups<4>`, 64x64 output tile, dynamic K (the op loops over K itself), raster order, result written straight to device memory as fp16/bf16. |
| `c_tensorops_tuned` | Apple's recommended structure from the MPP Programming Guide, section 2.3. One `matmul2d` per simdgroup (`execution_simdgroup`), K blocked at BK=128 (or 64 when K is not a multiple of 128), an fp32 destination cooperative tensor, a threadgroup barrier every 128 K, static slice extents, Morton threadgroup order (and one raster run for comparison). Configs: simdgroup tile 32x32 with 2x2 simdgroups (the guide's starting point), 32x32 with 4x4 (MLX's 128x128 threadgroup tile), and 64x64 with 2x2. |
| int8 `c_tensorops_tuned` | Same as `c`, `char x char -> int32` (MSL spec 4.1 Table 7.3). Square shape only, reported as GOP/s in the `gflops` field. **Experimental.** |

Shapes:

* `square`: 4096 x 4096 x 4096, the same as our MLX 57.9 / 53.8 TFLOP/s measurement.
* `qk`: 37,440 x 128 x 37,440 (attention scores Q·Kᵀ, head dim 128), done in query tiles
  of 4,160 rows (9 dispatches per iteration, `--qk-tile`). Buffers are about 330 MB.
* `pv`: 37,440 x 37,440 x 128 (P·V), done in tiles of 12,480 rows (`--pv-tile`). Buffers are
  about 945 MB. This shape is added because attention is both products. It has small N
  (128), so it has few threadgroups; that is real, and MLX's SDPA fuses it differently.

Only aligned shapes are implemented: rows a multiple of the threadgroup tile, N likewise,
K a multiple of BK. Variants that do not fit are reported as `skipped`. For example, 37,440 is
not a multiple of 128, so the 128x128-tile configs are skipped on the attention shapes.

Timing: each case warms up for at least 2 iterations and about 0.5 s. That warm-up also
sizes a batch so that one command buffer takes about 0.15 s. The case then runs batches
until `--seconds` (default 5) of wall time have passed.
`gflops = 2·M·N·K·iterations / Σ(gpuEndTime − gpuStartTime)`. `gflops_best_batch` is the
best single batch. Data is random in [-1, 1).

Verification runs first, on 256x256x512 and 128x192x128 for each dtype (and 256x256x512 for
int8). Every variant is compared with a CPU double-precision reference built from the
rounded inputs. An element passes when `|err| ≤ rtol·max|ref| + rtol·|ref|`, with rtol = 2e-3
for fp16 and 1.6e-2 for bf16; int8 must match exactly.
Every variant is also compared with the simdgroup result (`max_abs_diff_vs_simdgroup`). The
timing records carry `verified: true/false`.

Safety: the program refuses to run without `--yes`. All allocations go through one counter
capped at `--max-gb`, which defaults to 8 and cannot go higher. The largest case uses
about 1 GB. Buffers are freed after each shape. The program changes no system setting
(no sysctl, no wired limit, nothing written outside `--out`). `build_and_run.sh` writes only
into this folder.

## Run it on the Mac

```bash
bash ~/shutter/scripts/metal4_bench/build_and_run.sh --yes
```

This checks the SDK, the MPP header and (if the `metal` CLI is installed) the offline shader
compile. It then builds with `swiftc -O` and runs
`~/longcat-mlx/speed/gpulock.sh shutter-metal4 -- ./metal4_bench --yes --out results/metal4_bench_<time>.json`.
If the lock wrapper is missing it runs without it, with a warning. Without `--yes` it only
builds and prints the command. Extra arguments are passed on: for example `--quick` (1 s per
case, square only, a smoke test of about 1 minute) or `--shapes qk,pv --dtypes bf16`. A full
run is 2 dtypes x 3 shapes x up to 6 variants x about 5.5 s, plus int8, so **about 4
minutes**.

Do the free experiment first, because it may settle the question without any kernel work:

```bash
# in a scratch venv, not the production one
python -m venv /tmp/mlx0325 && /tmp/mlx0325/bin/pip install -q 'mlx==0.32.5'
/tmp/mlx0325/bin/python -c "import mlx.core as mx; print(mx.__version__, mx.metal.is_nax_available())"
~/longcat-mlx/speed/gpulock.sh shutter-probe -- bash ~/shutter/scripts/chip_probe.sh /tmp/mlx0325/bin/python
```

`is_nax_available()` exists only from 0.32.5 on. On 0.32.4 the 57.9 TFLOP/s itself is the
evidence. For direct proof, run Instruments → Metal System Trace with the "Neural
Accelerator utilization" counter pinned, as Apple's tech talk shows.

## How to read the result

The JSON has `compile` (per library: ok, or the compiler's error text), `verification`,
`results` (one record per shape × dtype × variant) and `summary`. For each shape and dtype,
`summary` gives the best of each kind, `speedup_best_tensorops_vs_simdgroup`, and
`best_tensorops_vs_mlx_57_9_class_reference`, which is the best TensorOps rate divided by
our MLX 4096² number of the same dtype (bf16 57.9, fp16 53.8). Records also carry
`ratio_vs_mlx_4096_matmul`.

What to expect, and what each outcome means:

1. **`a_simdgroup` at ≤ about 17 TFLOP/s, `c` at 50-65 TFLOP/s.** This is the expected
   result. It confirms that TensorOps is the only path to these rates, and that MLX is
   already on it.
2. **Best `c` for square bf16 clearly above 57.9 (for example ≥ 64).** MLX's GEMM leaves
   room. Check MLX 0.32.5 first; if it still trails, the gap is real.
3. **Best `c` ≤ MLX.** MLX's kernels are at least as good as Apple's recipe. Stop: no kernel
   port.
4. **`b` far below `c`.** That is the effect of Apple's tiling, K-blocking and Morton advice.
   The simple form is not how to write a production kernel.
5. **Attention shapes (`qk`, `pv`) vs square.** QKᵀ with K=128 writes an fp16 output of
   37,440² (about 2.8 GB per head) at 256 FLOP per output element. A memory bandwidth of
   about 0.5-0.6 TB/s caps that near 64-77 TFLOP/s. That is why real attention fuses QKᵀ,
   softmax and PV and never writes S. Read `qk`/`pv` as the matmul ceiling for a fused
   flash-attention inner loop, not as attention speed. Our SDPA at 56.6 TFLOP/s *including
   softmax* is already near that ceiling.

### Go / no-go threshold for porting Shuttrr kernels

The **port is a go only if** the best verified TensorOps variant here is **≥ 15% faster than
MLX on the shape that matters**:

* for attention: the best of `qk`/`pv` in bf16 at ≥ 1.15 × 56.6 ≈ **65 TFLOP/s**;
* for dense or quantized linears: square bf16 at ≥ 1.15 × 57.9 ≈ **67 TFLOP/s**;

and the same comparison with MLX 0.32.5 still shows the gap.

Reasoning:

* (a) The step already runs at about 49 TFLOP/s overall. A kernel that is X% faster in
  isolation gives less end to end (Amdahl). Fused attention loses part of any matmul gain
  to softmax and SIMD bookkeeping.
* (b) A port has real costs: Metal 4 only, opt-in exactness (different accumulation
  order, so reference outputs must be regenerated), fallback kernels kept for non-M5
  machines, and maintenance against MLX releases that keep improving the same paths.
* (c) Run-to-run noise on thermally throttling laptops is several percent, so anything
  under about 10% is not a clear signal.
* (d) If the estimated peak of about 66 TFLOP/s is right, a 15% gain over 57.9 is barely
  possible at all. A go therefore needs evidence that the peak is higher than estimated.

Below the threshold the answer is **no-go**: keep MLX's built-ins and spend the effort
elsewhere.

One exception is worth checking regardless of the benchmark. Any Shuttrr kernel written
with `mx.fast.metal_kernel` on simdgroup_matrix (our `flash_attn`, `qlinear8`) runs on the
SIMD pipe, capped at about 17 TFLOP/s on this chip, while MLX's own SDPA and
quantized_matmul run on the accelerators. Wherever Shuttrr uses its custom kernel on the M5
path instead of the MLX built-in, the built-in should win; `bench_kernels.py` section (d)
measures exactly that. If a custom kernel is still needed (for example block-sparse
attention), there is a cheaper route than a C++ extension. MLX 0.32.4 compiles
`mx.fast.metal_kernel` sources with MSL 4.0 (macOS 26) or 4.1 (macOS 27); this is verified
in `mlx/backend/metal/device.cpp`. So `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`
in the kernel `header=` should work. That is unverified at runtime.

## API details: verified vs not

**Verified from Apple documents:**

* MSL spec 4.1, chapter 7 + 2.22:
  * the header `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` and `<metal_tensor>`;
  * namespace `mpp::tensor_ops`;
  * `matmul2d_descriptor(M, N, K = dynamic_length_v<int>, transpose_left, transpose_right, relaxed_precision, mode)`;
  * `mode::multiply` / `mode::multiply_accumulate`;
  * `matmul2d<desc, Scope>`, `execution_simdgroup` and `execution_simdgroups<N>` (N = 1 or simdgroups_per_threadgroup);
  * `run(left, right, dest)`, `get_destination_cooperative_tensor<L, R, Elem>()`, `get_left/right_input_cooperative_tensor` (right: OS 26.1);
  * `tensor<device T, dextents<int,2>, tensor_inline>(ptr, extents, strides)` and `slice` / `slice<...>` with `dynamic_extent`;
  * the dtype table: half×half→half/float and char×char→int since Metal 4; bfloat×bfloat→bfloat/float since OS 26.1; uchar, int4 since OS 26.4; int2, fp4, fp8 since Metal 4.1;
  * `-std=metal4.0`;
  * the rule that every thread of the scope must call `run`.
* MPP Programming Guide (2026-03-16):
  * extents listed innermost first, `{K, SM}` for A with strides `{1, lda}`;
  * a 2x2 simdgroup threadgroup tile, SM = SN = 32 and BK = 128 as starting points for 16-bit types on M5;
  * barriers in the K loop, Morton order on a 1D grid, static extents for full tiles;
  * threadgroup-memory staging "isn't necessary".
* Apple docs (JSON API): `MTLTensor`, `MTLLanguageVersion.version4_0` / `.version4_1`, `MTLGPUFamily.metal4` / `.apple10` exist (macOS 26.0).

**Verified from code Apple's toolchain compiles (MLX 0.32.4 `steel_gemm_thin_nax.h`, `nax.h`):**

* The exact spellings used here:
  * `metal::tensor<device T, metal::dextents<int32_t,2>, metal::tensor_inline>`, `metal::array<int32_t,2>` strides;
  * `tA.template slice<BK, TSM>(k, 0)`, `metal::dynamic_extent`;
  * `op.template get_destination_cooperative_tensor<decltype(slice), decltype(slice), float>()`;
  * `get_capacity()`, `operator[]`, `.store(tensor_slice)`, `.load(...)`;
  * the descriptor with `relaxed_precision = true` and `multiply_accumulate`;
  * `threadgroup_barrier(mem_flags::mem_none)` in the K loop;
  * bfloat instantiations.
* Runtime compilation with `newLibrary(source:)` + language version 4.0 of sources that
  `#include` the MPP header. MLX's JIT build strips the SDK header from its preamble and
  leaves the `#include` for the runtime compiler.
* The simdgroup fragment lane layout (`frag_coord`) is the same as MLX's `BaseMMAFrag::get_coord`.

**Not verified (only these could make the build or the numbers wrong):**

1. Nothing was compiled. The Swift and MSL here have never seen a compiler.
2. `execution_simdgroups<4>` together with the two-argument `matmul2d_descriptor(64, 64)` and
   dynamic-extent slices (variant `b`). This is spelled as in the spec's example, but MLX uses
   only `execution_simdgroup`.
3. The int8 path: `tensor<device char,...>` with an `int` cooperative destination stored to
   `tensor<device int,...>`. The dtype is in the spec table, but no compiled example was found.
4. Whether `half x half -> half` (variant `b`, device destination) accumulates internally in
   fp32. The spec does not say. Verification will show it (larger error at K = 512 if not).
5. 64x64 simdgroup tiles (`sg64x64`) may spill registers or fail to build a pipeline. The
   guide warns about oversized simdgroup tiles. Such a failure shows up as an error for that
   variant only.
6. The estimated peak (about 66 TFLOP/s fp16), the M5 Max clock and the int8 ratio. These
   are extrapolated from an A19 measurement by a third party plus Apple's 16.6 TFLOP/s FP32
   figure.
7. The SDK path of the MPP header (the build script only searches for it), and whether the
   offline `metal` CLI accepts `-std=metal4.0` with this file. That check is optional.
8. `MTLGPUFamily.apple10` as the "M5-class" test is an inference. The benchmark only warns on it.

## Likely compile errors

**Swift (`swiftc`)**

* `'version4_0' is only available in macOS 26.0 or newer` / `type 'MTLGPUFamily' has no member 'metal4'`
  The SDK is older than 26 or the target is too low. The script passes
  `-target arm64-apple-macos26.0`; update Xcode or the CLT.
* Swift 6 concurrency errors about global variables (`var 'allocated' is not
  concurrency-safe`, main-actor isolation). Build with `-swift-version 5`, as the script
  does. If a toolchain ignores that, wrap the top-level code in `@MainActor`, or mark the
  globals `nonisolated(unsafe) var`.
* `cannot find 'Float16' in scope`: this only happens on Intel; the benchmark needs Apple silicon.
* `ambiguous use of ...` around `String(format:)` with `%@`: replace it with string
  interpolation.
* `MTLCompileOptions.mathMode` unavailable: delete that line. It only selects the safe math
  mode.

**Shader (reported in the JSON under `compile`, or by the offline check)**

* `'MetalPerformancePrimitives/MetalPerformancePrimitives.h' file not found`: the OS/SDK
  lacks TensorOps (needs macOS 26). With the CLI, add `-mmacosx-version-min=26.2`.
* `no member named 'execution_simdgroups' in namespace 'metal'`: try `mpp::execution_simdgroups<4>`
  or `execution_simdgroups<4>` after `using namespace mpp;`, as in the spec example. The same
  applies to `execution_simdgroup`.
* Errors in `mpp_basic` about the descriptor: replace `matmul2d_descriptor(64, 64)` with
  `matmul2d_descriptor(64, 64, 0)` (the spec example's form) or with
  `matmul2d_descriptor(64, 64, metal::dynamic_length_v<int>)`.
* `no matching constructor for tensor` / extents: use `extents_t(a, b)` with `int32_t`
  arguments (as written). Do not mix `int` and `uint`, because `tg` is `uint`.
* `missing 'template' keyword prior to dependent template name 'slice'`: add `.template`
  (already used everywhere in the templates).
* Anything in `mpp_i8`: delete the `BENCH_MPP_INT8` section or run with `--no-int8`; fp16 and
  bf16 are unaffected, because each section is compiled separately.
* `threadgroup variable ... must be declared in kernel scope` in `gemm_simd`: the arrays are
  already at function scope; move them to the top of the function if a compiler disagrees.
* Pipeline creation fails, or `maxTotalThreadsPerThreadgroup` is below the needed count, for
  the `w4x4` (512 threads) or `sg64x64` configs: register pressure. That variant reports
  an error; the others still run.

## Sources

Read for this benchmark:

* Apple tech talk 111432 (M5 / A19 GPU neural accelerators and TensorOps):
  https://developer.apple.com/videos/play/tech-talks/111432 . Key points: TensorOps flow,
  Morton dispatch "almost 7x" over simdgroup_matrix at 4096², bfloat (OS 26.1),
  cooperative inputs (26.3) and int4/int8 (26.4) per the captions, and the Neural
  Accelerator counter in Instruments.
* Metal Performance Primitives Programming Guide (PDF, 2026-03-16):
  https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf
* Metal Shading Language Specification 4.1 (PDF, 2026-06-04), sections 2.22 and 7:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
* Apple docs: MTLTensor https://developer.apple.com/documentation/metal/mtltensor ;
  Machine learning passes https://developer.apple.com/documentation/metal/machine-learning-passes ;
  MTLLanguageVersion.version4_0 https://developer.apple.com/documentation/metal/mtllanguageversion/version4_0
  (read through the developer.apple.com/tutorials/data/documentation/... JSON endpoints).
* Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU":
  https://machinelearning.apple.com/research/exploring-llms-mlx-m5 . It says that MLX uses
  TensorOps and MPP, that it "requires macOS 26.2 or later", and it reports 3.3-4.1x
  time-to-first-token and 3.8x FLUX vs M4. It does not name an MLX version and does not
  mention attention.
* MLX source, tags v0.29.0 / v0.30.0 / v0.32.4 / v0.32.5 and master
  (https://github.com/ml-explore/mlx):
  * `mlx/backend/metal/device.cpp` `is_nax_available()`;
  * `matmul.cpp`, `scaled_dot_product_attention.cpp`, `quantized.cpp` dispatch;
  * `kernels/steel/gemm/nax.h`, `kernels/steel/gemm/kernels/steel_gemm_thin_nax.h`;
  * `mlx/backend/metal/CMakeLists.txt` (NAX needs Metal 4, SDK ≥ 26.2 and deployment target ≥ 26.2);
  * `mlx/utils.h` (`MLX_ENABLE_TF32` default 1).
* MLX v0.30.0 release notes: https://github.com/ml-explore/mlx/releases/tag/v0.30.0
* PyPI: https://pypi.org/pypi/mlx/json and https://pypi.org/pypi/mlx-metal/0.32.4/json (versions and
  upload dates, wheel tags). The `macosx_26_0` 0.32.4 wheel was downloaded and inspected.
* T. Zakharko, "Investigating the GPU Neural Accelerators on Apple A19/M5":
  https://tzakharko.github.io/apple-neural-accelerators-benchmark . It measured about 1,024
  fp16 FLOP/core/clock, about 2x that for int8, single-simdgroup 32x32 tiles as best, and the
  same speed with an fp16 or fp32 accumulator. It is third-party and covers A19, not M5 Max.
* M5 Max FP32 peak of 16.6 TFLOP/s: https://flopper.io/gpu/apple-m5-max/spec-sheet (it
  cites Apple's newsroom).
