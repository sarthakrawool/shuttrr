"""adaln_mod: fused AdaLN modulation, `modulate_fp32(LayerNorm_FP32(no affine), x, shift, scale)`.

Model math (blocks.py `modulate_fp32` + `LayerNorm_FP32`, eps 1e-6), per row of C channels:

    x32 = float(x); mean = mean(x32); var = mean((x32 - mean)^2)
    y   = ((x32 - mean) * rsqrt(var + eps)) * (scale + 1) + shift      # all fp32
    out = bf16(y)

x is [B, T, S, C] (S tokens per latent frame), shift/scale are fp32 [B, T, 1, C] (one row per frame).

MLX path traffic: astype fp32 (write 4 B/elem), layer_norm (read 4, write 4), modulate+cast (read 4, write 2),
plus the bf16 read: ~18 B/elem. The kernel reads bf16 x once and writes bf16 once (4 B/elem): one
threadgroup per row, the row is held in registers between the mean, variance and output passes.

Exactness: NOT bit-identical to the MLX path in general (reduction order of the mean and variance differs
from MLX's reduce / layer_norm kernels; outputs agree to about one bf16 ulp). Opt-in only.

Self-contained (imports only mlx) so scripts/bench_kernels.py can load it from a copied folder.
"""

from __future__ import annotations

import mlx.core as mx

EPS = 1e-6


def metal_available() -> bool:
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


# ------------------------------------------------------------------------------------------- reference
def reference(x: mx.array, shift: mx.array, scale: mx.array, eps: float = EPS) -> mx.array:
    """The model's ops, verbatim (blocks.LayerNorm_FP32 without affine, then modulate_fp32)."""
    dtype = x.dtype
    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    var = mx.mean((x32 - mean) ** 2, axis=-1, keepdims=True)
    x32 = (x32 - mean) * mx.rsqrt(var + eps)
    x32 = x32 * (scale + 1) + shift
    return x32.astype(dtype)


def reference_fast(x: mx.array, shift: mx.array, scale: mx.array, eps: float = EPS) -> mx.array:
    """The production fast path (fast.py FastLayerNorm): mx.fast.layer_norm in fp32, then modulate."""
    x32 = mx.fast.layer_norm(x.astype(mx.float32), None, None, eps)
    return (x32 * (scale + 1) + shift).astype(x.dtype)


# ---------------------------------------------------------------------------------------------- kernel
TPG = 256  # threads per row; C must be a multiple of TPG (4096 / 256 = 16 values per thread)

_SOURCE = """
    // one threadgroup per row; template: T (x dtype), C (channels), TPG (threads), S (rows per frame)
    constexpr int PER = C / TPG;
    constexpr int NSG = TPG / 32;
    threadgroup float red[NSG];
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    const device T* xr = x + size_t(row) * C;
    const device float* sh = shift + size_t(row / S) * C;
    const device float* sc = scale + size_t(row / S) * C;
    device T* outr = out + size_t(row) * C;

    float v[PER];
    float s = 0.0f;
    for (int i = 0; i < PER; i++) {
        v[i] = static_cast<float>(xr[tid + i * TPG]);
        s += v[i];
    }
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < NSG; i++) tot += red[i];
    float mean = tot / float(C);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q = 0.0f;
    for (int i = 0; i < PER; i++) {
        float d = v[i] - mean;
        q += d * d;
    }
    q = simd_sum(q);
    if (lane == 0) red[sg] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qt = 0.0f;
    for (int i = 0; i < NSG; i++) qt += red[i];
    float r = metal::precise::rsqrt(qt / float(C) + eps[0]);

    for (int i = 0; i < PER; i++) {
        int c = tid + i * TPG;
        float y = ((v[i] - mean) * r) * (sc[c] + 1.0f) + sh[c];
        outr[c] = static_cast<T>(y);
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="shutter_adaln_mod",
            input_names=["x", "shift", "scale", "eps"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _KERNEL


def kernel(x: mx.array, shift: mx.array, scale: mx.array, eps: float = EPS) -> mx.array:
    """x [B, T, S, C]; shift/scale fp32 [B, T, 1, C] (or [B, T, C])."""
    B, T, S, C = x.shape
    assert C % TPG == 0, f"C={C} must be a multiple of {TPG}"
    sh = shift.astype(mx.float32).reshape(B * T, C)
    sc = scale.astype(mx.float32).reshape(B * T, C)
    rows = B * T * S
    out = _kernel()(
        inputs=[x, sh, sc, mx.array([eps], dtype=mx.float32)],
        template=[("T", x.dtype), ("C", C), ("TPG", TPG), ("S", S)],
        grid=(rows * TPG, 1, 1),
        threadgroup=(TPG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out[0]


def adaln_mod(x, shift, scale, eps: float = EPS, use_kernel: bool | None = None):
    if use_kernel is None:
        use_kernel = metal_available()
    return kernel(x, shift, scale, eps) if use_kernel else reference(x, shift, scale, eps)


BIT_IDENTICAL = False  # expected; the benchmark reports the measured value


def make_inputs(B=1, T=4, S=512, C=4096, seed=0):
    k = mx.random.split(mx.random.key(seed), 3)
    x = (mx.random.normal((B, T, S, C), key=k[0]) * 3 + 0.5).astype(mx.bfloat16)
    shift = mx.random.normal((B, T, 1, C), key=k[1]) * 0.1
    scale = mx.random.normal((B, T, 1, C), key=k[2]) * 0.1
    return x, shift, scale


def selftest(B=1, T=4, S=512, C=4096) -> dict:
    """Kernel vs reference on random data (reference vs fast reference when Metal is unavailable)."""
    x, shift, scale = make_inputs(B, T, S, C)
    ref = reference(x, shift, scale)
    ran = metal_available()
    got = kernel(x, shift, scale) if ran else reference_fast(x, shift, scale)
    d = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
    return {"name": "adaln_mod", "ran_kernel": ran, "max_abs_diff": float(mx.max(d)),
            "frac_exact": float(mx.mean(d == 0)), "bit_identical": bool(mx.all(d == 0))}
