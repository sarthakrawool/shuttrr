"""qlinear8: fused 8-bit affine dequantise + matmul, y = x @ dequant(W)^T (+ bias).

Weight layout (mlx.nn.QuantizedLinear / mx.quantize, mode "affine", bits 8, group size 64; checked on CPU in
tests/test_kernels.py against mx.dequantize):

    w      uint32 [out, in/4]   four 8-bit values per word, element j in bits 8*(j%4) .. 8*(j%4)+7
                                (little endian, so viewed as uint8 the row is simply [out, in])
    scales [out, in/64]          one per group of 64 consecutive input elements of a row (dtype of the model)
    biases [out, in/64]
    W[o, j] = scales[o, j/64] * q[o, j] + biases[o, j/64]

Kernel: 32x32 output tile per threadgroup, 4 simdgroups (each a 16x16 sub-tile as 2x2 simdgroup 8x8 fp32
accumulators), K stepped in 32-wide slices. Each slice of W is read as uint8 from DRAM, dequantised in
registers and stored to threadgroup memory as fp32; no dequantised copy of W is ever written to DRAM.

Honest expectation: mx.quantized_matmul is already exactly this kind of kernel (MLX's qmm loads packed
tiles, dequantises into threadgroup memory and runs simdgroup MMA; it never materialises W either), tuned by
the MLX authors per GPU family. REPORT.md measures the q8 qkv matmul at 0.16 s = ~54 TFLOP/s, close to what
dense bf16 matmuls reach. A first hand-written kernel like this one, with fp32 staging and no double
buffering, should be expected to be SLOWER, probably by 2-5x. The only routes to a win are (a) an epilogue
that fuses something MLX cannot (the runtime LoRA up-projection add, gate/residual), (b) tile shapes tuned
for the M5 that MLX does not pick. The benchmark decides; treat this file as the starting point for (a).

Exactness: different accumulation order from MLX's qmm, so NOT bit-identical. Opt-in only.
"""

from __future__ import annotations

import mlx.core as mx

GROUP = 64
BITS = 8


def metal_available() -> bool:
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


# ------------------------------------------------------------------------------------------- reference
def dequantize_ref(w: mx.array, scales: mx.array, biases: mx.array, group_size: int = GROUP) -> mx.array:
    """Explicit 8-bit affine dequantisation from the uint32 packing (same values as mx.dequantize)."""
    out, packed = w.shape
    k = packed * 4
    q = w.view(mx.uint8).reshape(out, k // group_size, group_size).astype(scales.dtype)
    return (q * scales[..., None] + biases[..., None]).reshape(out, k)


def reference(x, w, scales, biases, bias=None, group_size: int = GROUP):
    """x @ dequant(W)^T accumulated in fp32, cast to x.dtype, + bias (as nn.QuantizedLinear adds it)."""
    wd = dequantize_ref(w, scales, biases, group_size).astype(mx.float32)
    y = (x.astype(mx.float32) @ wd.T).astype(x.dtype)
    return y + bias if bias is not None else y


def mlx_path(x, w, scales, biases, bias=None, group_size: int = GROUP):
    """What nn.QuantizedLinear does."""
    y = mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=BITS)
    return y + bias if bias is not None else y


# ---------------------------------------------------------------------------------------------- kernel
_HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
"""

_SOURCE = """
    // template: T (x dtype), M, K, NO (out features); K % 64 == 0
    constexpr int BM = 32, BN = 32, BK = 32, GS = 64, NT = 128;
    threadgroup float Xs[BM * BK];
    threadgroup float Ws[BK * BN];
    threadgroup float Cs[BM * BN];
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    int col0 = int(threadgroup_position_in_grid.x) * BN;
    int row0 = int(threadgroup_position_in_grid.y) * BM;
    int sm = int(sg / 2) * 16;
    int sn = int(sg % 2) * 16;
    const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);

    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = int(tid); i < BM * BK; i += NT) {
            int r = i / BK, c = i % BK;
            int gr = row0 + r;
            Xs[r * BK + c] = gr < M ? static_cast<float>(x[size_t(gr) * K + k0 + c]) : 0.0f;
        }
        for (int i = int(tid); i < BN * BK; i += NT) {
            int n = i / BK, c = i % BK;           // consecutive threads read consecutive bytes of one row
            int gn = col0 + n;
            float val = 0.0f;
            if (gn < NO) {
                int kk = k0 + c;
                size_t g = size_t(gn) * (K / GS) + kk / GS;
                float q = static_cast<float>(wb[size_t(gn) * K + kk]);
                val = static_cast<float>(scales[g]) * q + static_cast<float>(biases[g]);
            }
            Ws[c * BN + n] = val;                 // stored transposed: Ws is [BK, BN]
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, Xs + (sm) * BK + kk, BK);
            simdgroup_load(a1, Xs + (sm + 8) * BK + kk, BK);
            simdgroup_load(b0, Ws + kk * BN + sn, BN);
            simdgroup_load(b1, Ws + kk * BN + sn + 8, BN);
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc00, Cs + (sm) * BN + sn, BN);
    simdgroup_store(acc01, Cs + (sm) * BN + sn + 8, BN);
    simdgroup_store(acc10, Cs + (sm + 8) * BN + sn, BN);
    simdgroup_store(acc11, Cs + (sm + 8) * BN + sn + 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < BM * BN; i += NT) {
        int r = i / BN, c = i % BN;
        int gr = row0 + r, gc = col0 + c;
        if (gr < M && gc < NO) y[size_t(gr) * NO + gc] = static_cast<T>(Cs[i]);
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="shutter_qlinear8",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_SOURCE,
            header=_HEADER,
        )
    return _KERNEL


def kernel(x, w, scales, biases, bias=None, group_size: int = GROUP):
    assert group_size == GROUP, "kernel is written for group size 64"
    *lead, K = x.shape
    NO = w.shape[0]
    assert w.shape[1] * 4 == K and K % 64 == 0
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    out = _kernel()(
        inputs=[x2, w, scales, biases],
        template=[("T", x.dtype), ("M", M), ("K", K), ("NO", NO)],
        grid=(((NO + 31) // 32) * 128, (M + 31) // 32, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(M, NO)],
        output_dtypes=[x.dtype],
    )[0].reshape(*lead, NO)
    return out + bias if bias is not None else out


def qlinear8(x, w, scales, biases, bias=None, use_kernel: bool | None = None):
    if use_kernel is None:
        use_kernel = metal_available()
    return (kernel if use_kernel else reference)(x, w, scales, biases, bias)


BIT_IDENTICAL = False


def make_inputs(M=256, K=512, NO=384, seed=0, dtype=mx.bfloat16):
    k = mx.random.split(mx.random.key(seed), 3)
    wf = (mx.random.normal((NO, K), key=k[0]) * 0.02).astype(dtype)
    w, s, b = mx.quantize(wf, group_size=GROUP, bits=BITS)
    x = mx.random.normal((M, K), key=k[1]).astype(dtype)
    bias = (mx.random.normal((NO,), key=k[2]) * 0.01).astype(dtype)
    return x, w, s, b, bias


def selftest(M=256, K=512, NO=384) -> dict:
    x, w, s, b, bias = make_inputs(M, K, NO)
    ref = reference(x, w, s, b, bias)
    ran = metal_available()
    got = kernel(x, w, s, b, bias) if ran else mlx_path(x, w, s, b, bias)
    d = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
    return {"name": "qlinear8", "ran_kernel": ran, "max_abs_diff": float(mx.max(d)),
            "max_rel_diff": float(mx.max(d) / mx.max(mx.abs(ref.astype(mx.float32)))),
            "frac_exact": float(mx.mean(d == 0)), "bit_identical": bool(mx.all(d == 0))}
