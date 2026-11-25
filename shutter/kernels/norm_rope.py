"""norm_rope: fused q/k RMSNorm (fp32) + 3D RoPE (fp32) + head transpose, straight from the qkv linear output.

Model math (attention.py Attention.__call__, blocks.RMSNorm_FP32 eps 1e-6, rope_3d.RotaryPositionalEmbedding):

    qkv = x @ Wqkv               # [B, N, 3*C] -> [B, N, 3, H, D] -> [3, B, H, N, D]
    q   = bf16(f32(q) * rsqrt(mean(f32(q)^2) + eps)) * w_q          # weight applied in bf16
    q   = bf16(f32(q) * cos + rotate_half(f32(q)) * sin)            # same for k with w_k
    v   unchanged (only transposed)

The model uses RMSNorm (not LayerNorm) for q/k. The kernel reads one (token, q|k|v, head) row of D = 128
values with one simdgroup (4 values per lane, so each RoPE pair (2i, 2i+1) stays inside one lane), and
writes q, k, v directly in the [B, H, N, D] layout SDPA wants. One pass: 2 B/elem read + 2 B/elem written for
q, k and v, versus the MLX chain (transposed copy, fp32 cast, norm, cast, weight, fp32 cast, 3 RoPE products,
cast) which REPORT.md measures at 0.27 s/block eager and ~6 GB of fp32 temporaries.

The RoPE tables are the ones fast.py's CachedRope uses: cos and a "signed" sin (-sin on even slots, +sin on
odd slots), so out[2i] = x[2i]*cos + x[2i+1]*(-sin), out[2i+1] = x[2i+1]*cos + x[2i]*sin, which is the same
products and sum as `x*cos + rotate_half(x)*sin`.

Exactness: the elementwise RoPE part is the same arithmetic; the RMS reduction order (simd_sum over 32 lanes
of 4-element partial sums) differs from MLX's mean / mx.fast.rms_norm, so a few outputs can differ by one
bf16 ulp. NOT guaranteed bit-identical; opt-in unless the benchmark shows max diff 0 on real activations.
"""

from __future__ import annotations

import mlx.core as mx

EPS = 1e-6


def metal_available() -> bool:
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


# ------------------------------------------------------------------------------------------ rope tables
def rope_freqs(grid: tuple[int, int, int], head_dim: int = 128, base: float = 10000.0) -> mx.array:
    """rope_3d.RotaryPositionalEmbedding._precompute_freqs without frame_index/num_ref (tokens, head_dim)."""
    T, H, W = grid
    dim_t = head_dim - 4 * (head_dim // 6)
    dim_h = 2 * (head_dim // 6)
    dim_w = 2 * (head_dim // 6)

    def axis(n, dim):
        idx = mx.arange(0, dim, 2, dtype=mx.float32)[: dim // 2]
        f = 1.0 / (base ** (idx / dim))
        g = mx.arange(0, n, dtype=mx.float32)
        return mx.repeat(g[:, None] * f[None, :], 2, axis=-1)

    ft, fh, fw = axis(T, dim_t), axis(H, dim_h), axis(W, dim_w)
    f = mx.concatenate([
        mx.broadcast_to(ft[:, None, None, :], (T, H, W, dim_t)),
        mx.broadcast_to(fh[None, :, None, :], (T, H, W, dim_h)),
        mx.broadcast_to(fw[None, None, :, :], (T, H, W, dim_w)),
    ], axis=-1)
    return f.reshape(T * H * W, head_dim)


def rope_tables(grid, head_dim: int = 128):
    """(cos, signed sin), fp32 [tokens, head_dim], as fast.rope_tables."""
    f = rope_freqs(grid, head_dim)
    return mx.cos(f), mx.sin(f) * mx.array([-1.0, 1.0] * (head_dim // 2))


# ------------------------------------------------------------------------------------------- reference
def _rotate_half(x):
    *p, d = x.shape
    x = x.reshape(*p, d // 2, 2)
    return mx.stack([-x[..., 1], x[..., 0]], axis=-1).reshape(*p, d)


def _rms_port(x, w, eps):
    x32 = x.astype(mx.float32)
    n = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return n.astype(x.dtype) * w


def reference(qkv: mx.array, w_q: mx.array, w_k: mx.array, grid, heads: int = 32, eps: float = EPS):
    """The port's ops verbatim (RMSNorm_FP32 + RotaryPositionalEmbedding). qkv [B, N, 3*C]."""
    B, N, C3 = qkv.shape
    D = C3 // 3 // heads
    t = qkv.reshape(B, N, 3, heads, D).transpose(2, 0, 3, 1, 4)
    q, k, v = t[0], t[1], t[2]
    q, k = _rms_port(q, w_q, eps), _rms_port(k, w_k, eps)
    f = rope_freqs(grid, D)
    cos, sin = mx.cos(f)[None, None], mx.sin(f)[None, None]
    dt = q.dtype
    q32, k32 = q.astype(mx.float32), k.astype(mx.float32)
    q = (q32 * cos + _rotate_half(q32) * sin).astype(dt)
    k = (k32 * cos + _rotate_half(k32) * sin).astype(dt)
    return q, k, v


def reference_fast(qkv, w_q, w_k, grid, heads: int = 32, eps: float = EPS):
    """The production fast path: mx.fast.rms_norm (FastRMSNorm) + CachedRope's rotate."""
    B, N, C3 = qkv.shape
    D = C3 // 3 // heads
    t = qkv.reshape(B, N, 3, heads, D).transpose(2, 0, 3, 1, 4)
    q, k, v = t[0], t[1], t[2]
    q = mx.fast.rms_norm(q.astype(mx.float32), None, eps).astype(q.dtype) * w_q
    k = mx.fast.rms_norm(k.astype(mx.float32), None, eps).astype(k.dtype) * w_k
    cos, sin = rope_tables(grid, D)

    def rot(x):
        *p, d = x.shape
        x32 = x.astype(mx.float32)
        sw = x32.reshape(*p, d // 2, 2)[..., ::-1].reshape(*p, d)
        return (x32 * cos + sw * sin).astype(x.dtype)

    return rot(q), rot(k), v


# ---------------------------------------------------------------------------------------------- kernel
_SOURCE = """
    // template: T (dtype), NTOK (tokens per batch item), H (heads), D (head dim), ROWS (B*N*3*H)
    constexpr int PER = D / 32;           // values per lane (4 for D = 128); must be even
    uint gid = thread_position_in_grid.x;
    uint row = gid / 32;
    uint lane = gid % 32;
    if (row >= uint(ROWS)) return;        // whole simdgroups exit together (row is uniform in a simdgroup)
    uint h = row % H;
    uint s = (row / H) % 3;               // 0 = q, 1 = k, 2 = v
    uint bn = row / (3 * H);
    uint n = bn % NTOK;
    uint b = bn / NTOK;
    uint d0 = lane * PER;
    const device T* xp = qkv + size_t(row) * D + d0;
    size_t o = ((size_t(b) * H + h) * NTOK + n) * D + d0;

    float v[PER];
    for (int i = 0; i < PER; i++) v[i] = static_cast<float>(xp[i]);

    if (s == 2) {
        for (int i = 0; i < PER; i++) vo[o + i] = static_cast<T>(v[i]);
        return;
    }
    float ss = 0.0f;
    for (int i = 0; i < PER; i++) ss += v[i] * v[i];
    ss = simd_sum(ss);
    float r = metal::precise::rsqrt(ss / float(D) + eps[0]);

    const device T* w = (s == 0) ? wq : wk;
    const device float* cr = cosv + size_t(n) * D + d0;
    const device float* sr = sinv + size_t(n) * D + d0;
    float z[PER];
    for (int i = 0; i < PER; i++) {
        T nb = static_cast<T>(v[i] * r);                               // norm.astype(bf16)
        T wb = static_cast<T>(static_cast<float>(nb) * static_cast<float>(w[d0 + i]));  // * weight in bf16
        z[i] = static_cast<float>(wb);
    }
    device T* dst = (s == 0) ? qo : ko;
    for (int i = 0; i < PER; i += 2) {
        float a = z[i] * cr[i];
        float bb = z[i + 1] * sr[i];
        float c = z[i + 1] * cr[i + 1];
        float e = z[i] * sr[i + 1];
        dst[o + i] = static_cast<T>(a + bb);
        dst[o + i + 1] = static_cast<T>(c + e);
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="shutter_norm_rope",
            input_names=["qkv", "wq", "wk", "cosv", "sinv", "eps"],
            output_names=["qo", "ko", "vo"],
            source=_SOURCE,
            # products are written as separate statements; "safe" math keeps them un-contracted where the
            # compiler honours it (no FMA fusion), matching MLX's separate multiply and add kernels.
            compile_options={"math_mode": "safe"},
        )
    return _KERNEL


def kernel(qkv, w_q, w_k, grid, heads: int = 32, eps: float = EPS, tables=None):
    B, N, C3 = qkv.shape
    D = C3 // 3 // heads
    assert D % 64 == 0, "D must give an even number of values per lane (D % 64 == 0)"
    assert w_q.dtype == qkv.dtype and w_k.dtype == qkv.dtype, "norm weights expected in the activation dtype"
    cos, sin = tables if tables is not None else rope_tables(grid, D)
    rows = B * N * 3 * heads
    shape = (B, heads, N, D)
    outs = _kernel()(
        inputs=[qkv, w_q, w_k, cos, sin, mx.array([eps], dtype=mx.float32)],
        template=[("T", qkv.dtype), ("NTOK", N), ("H", heads), ("D", D), ("ROWS", rows)],
        grid=(rows * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape, shape, shape],
        output_dtypes=[qkv.dtype] * 3,
    )
    return outs[0], outs[1], outs[2]


def norm_rope(qkv, w_q, w_k, grid, heads: int = 32, eps: float = EPS, use_kernel: bool | None = None):
    if use_kernel is None:
        use_kernel = metal_available()
    return (kernel if use_kernel else reference)(qkv, w_q, w_k, grid, heads, eps)


BIT_IDENTICAL = False  # expected; the benchmark reports the measured value


def make_inputs(grid=(2, 8, 16), heads=32, D=128, B=1, seed=0):
    N = grid[0] * grid[1] * grid[2]
    k = mx.random.split(mx.random.key(seed), 3)
    qkv = (mx.random.normal((B, N, 3 * heads * D), key=k[0]) * 2).astype(mx.bfloat16)
    w_q = (1 + 0.1 * mx.random.normal((D,), key=k[1])).astype(mx.bfloat16)
    w_k = (1 + 0.1 * mx.random.normal((D,), key=k[2])).astype(mx.bfloat16)
    return qkv, w_q, w_k


def selftest(grid=(2, 8, 16), heads=32) -> dict:
    qkv, w_q, w_k = make_inputs(grid, heads)
    ref = reference(qkv, w_q, w_k, grid, heads)
    ran = metal_available()
    got = kernel(qkv, w_q, w_k, grid, heads) if ran else reference_fast(qkv, w_q, w_k, grid, heads)
    ds = [mx.abs(g.astype(mx.float32) - r.astype(mx.float32)) for g, r in zip(got, ref)]
    return {"name": "norm_rope", "ran_kernel": ran, "max_abs_diff": max(float(mx.max(d)) for d in ds),
            "frac_exact": float(sum(mx.mean(d == 0) for d in ds) / 3),
            "bit_identical": all(bool(mx.all(d == 0)) for d in ds)}
