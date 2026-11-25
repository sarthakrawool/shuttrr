"""flash_attn: tiled attention with an online softmax, so the N x N scores never reach DRAM.

softmax(scale * Q K^T) V per (batch, head); q [B, H, Nq, D], k/v [B, H, Nk, D], D = 128, bf16 in and out.

Design (first prototype)
- One threadgroup = 4 simdgroups = 128 threads owns BQ = 32 query rows (8 per simdgroup) of one head.
- It walks the keys in tiles of BK = 16. Per tile: K and V tiles go to threadgroup memory as fp32
  (2 x 16 x 128 x 4 B = 16 KB); each simdgroup computes its 8 x 16 score block with simdgroup 8x8 MMA
  (Q is held in registers as 16 8x8 fragments), stages it in threadgroup memory (512 B per simdgroup),
  and 4 lanes per query row do the online-softmax update:
      m_new = max(m, rowmax(s));  p = exp2(s - m_new);  l = l * exp2(m - m_new) + rowsum(p)
  The O accumulator (8 x 128 fp32 per simdgroup, 16 fragments in registers) is rescaled by
  diag(exp2(m - m_new)) with one 8x8 MMA per fragment, then O += P V.
- At the end O / l is written as bf16. Scores (fp32) live only in registers/threadgroup memory.

Tile sizes vs Apple GPU threadgroup memory (32 KB per threadgroup): K+V tiles 16 KB (reused for the Q tile
at the start and the O tile at the end), scores 2 KB, diagonal 1 KB, row sums 128 B: ~19 KB, leaving room for
2 threadgroups per core only if the compiler keeps registers in check. BK = 32 would need 32 KB of fp32 K/V
tiles: storing K/V tiles as bf16/half (MSL 3.1 has bfloat simdgroup matrices on newer GPUs) is the
obvious next step and halves this; BQ = 64 (8 simdgroups) doubles K/V reuse per tile.

Arithmetic intensity: each K/V tile load serves BQ query rows, so DRAM-level intensity without cache reuse
is ~BQ FLOP/byte (see flops.attention_traffic); BQ = 32 gives ~32 FLOP/B, so concurrent threadgroups
sharing K/V blocks through the caches is what keeps this compute-bound.

Precision: QK^T, softmax statistics and PV all accumulate in fp32; exp2 with scale*log2(e) folded into the
score. Same precision plan as MLX's SDPA, but the summation order differs (online rescaling per key tile).

Exactness: NOT bit-identical to mx.fast.scaled_dot_product_attention (different tile order and online
rescaling). Numerically close (bf16-level). Opt-in only. mx.fast.scaled_dot_product_attention is itself
already a tiled flash-style kernel that never writes the scores (REPORT: SDPA working memory 0-1.4 GB at
86.6K tokens), so this prototype competes with a tuned kernel and is expected to be slower at first.
"""

from __future__ import annotations

import math
import os

import mlx.core as mx


def _keep_fp32_exact():
    """MLX 0.32 on GPUs with neural accelerators (Apple M5, `nax` kernels) runs float32 matmul and float32
    mx.fast.scaled_dot_product_attention as TF32 by default (MLX_ENABLE_TF32 defaults to 1): ~1e-3 relative
    error, e.g. a 64x128 @ 128x64 fp32 matmul off by ~0.03 at |y| ~ 40. Every fp32 reference / exactness claim
    here assumes real fp32, so turn TF32 off process-wide unless the user set MLX_ENABLE_TF32 themselves.
    It has to be the process-wide setting: the value is read when a graph is evaluated (not built), and an
    mx.compile'd function is evaluated outside any scope we could open. bf16 / fp16 work is unaffected."""
    cfg = getattr(mx, "config", None)
    if cfg is not None and hasattr(cfg, "update") and "MLX_ENABLE_TF32" not in os.environ:
        cfg.update("MLX_ENABLE_TF32", 0)


_keep_fp32_exact()


BQ = 32
BK = 16


def metal_available() -> bool:
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


# ------------------------------------------------------------------------------------------- references
def reference(q, k, v, scale=None):
    """The model path: mx.fast.scaled_dot_product_attention."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def reference_naive(q, k, v, scale=None):
    """Materialised scores in fp32 (what a naive implementation writes to memory)."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    s = (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)) * scale
    p = mx.softmax(s, axis=-1)
    return (p @ v.astype(mx.float32)).astype(q.dtype)


def reference_tiled(q, k, v, scale=None, bk: int = 256):
    """The kernel's algorithm in MLX ops (online softmax over key tiles, fp32). Validates the math on CPU."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    c = scale * math.log2(math.e)
    q32 = q.astype(mx.float32)
    *lead, nq, d = q.shape
    m = mx.full((*lead, nq, 1), -mx.inf, dtype=mx.float32)
    l = mx.zeros((*lead, nq, 1), dtype=mx.float32)
    o = mx.zeros((*lead, nq, d), dtype=mx.float32)
    nk = k.shape[-2]
    for k0 in range(0, nk, bk):
        kt = k[..., k0:k0 + bk, :].astype(mx.float32)
        vt = v[..., k0:k0 + bk, :].astype(mx.float32)
        s = (q32 @ kt.swapaxes(-1, -2)) * c
        m_new = mx.maximum(m, mx.max(s, axis=-1, keepdims=True))
        p = mx.power(2.0, s - m_new)
        alpha = mx.power(2.0, m - m_new)
        l = l * alpha + mx.sum(p, axis=-1, keepdims=True)
        o = o * alpha + p @ vt
        m = m_new
    return (o / l).astype(q.dtype)


# ---------------------------------------------------------------------------------------------- kernel
_HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
"""

_SOURCE = """
    // template: T (dtype), NQ, NK (query / key lengths), D (head dim, multiple of 8), H (heads)
    constexpr int BQ_ = 32, BK_ = 16, NT = 128, ND = D / 8;
    threadgroup float KV[2 * BK_ * D];          // K tile | V tile; Q tile at start, O tile at the end
    threadgroup float Ss[4 * 8 * BK_];          // per-simdgroup 8 x BK score block
    threadgroup float Dg[4 * 64];               // per-simdgroup 8 x 8 diagonal (row rescale)
    threadgroup float Ls[BQ_];
    threadgroup float* Ks = KV;
    threadgroup float* Vs = KV + BK_ * D;

    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int q0 = int(threadgroup_position_in_grid.x) * BQ_;
    size_t bh = size_t(threadgroup_position_in_grid.z) * H + threadgroup_position_in_grid.y;
    const device T* qh = q + bh * size_t(NQ) * D;
    const device T* kh = k + bh * size_t(NK) * D;
    const device T* vh = v + bh * size_t(NK) * D;
    device T* oh = o + bh * size_t(NQ) * D;
    float c = scale[0] * 1.4426950408889634f;   // scale * log2(e)

    // stage the Q tile (32 x D) and load this simdgroup's 8 rows into registers
    for (int i = int(tid); i < BQ_ * D; i += NT) {
        int r = i / D, cc = i % D;
        KV[i] = (q0 + r < NQ) ? static_cast<float>(qh[size_t(q0 + r) * D + cc]) : 0.0f;
    }
    threadgroup float* Ssg = Ss + sg * 8 * BK_;
    threadgroup float* Dsg = Dg + sg * 64;
    for (int i = int(lane); i < 64; i += 32) Dsg[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 qf[ND];
    simdgroup_float8x8 of[ND];
    for (int d = 0; d < ND; d++) {
        simdgroup_load(qf[d], KV + (sg * 8) * D + d * 8, D);
        of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int r = int(lane) / 4;            // this lane's query row inside the simdgroup
    int c0 = (int(lane) % 4) * 4;     // and its 4 score columns
    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (int k0 = 0; k0 < NK; k0 += BK_) {
        for (int i = int(tid); i < BK_ * D; i += NT) {
            int kr = i / D, cc = i % D;
            bool ok = k0 + kr < NK;
            Ks[i] = ok ? static_cast<float>(kh[size_t(k0 + kr) * D + cc]) : 0.0f;
            Vs[i] = ok ? static_cast<float>(vh[size_t(k0 + kr) * D + cc]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S (8 x 16) = Q (8 x D) K^T (D x 16), as two 8x8 blocks
        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int d = 0; d < ND; d++) {
            simdgroup_float8x8 kt0, kt1;
            simdgroup_load(kt0, Ks + 0 * D + d * 8, D, ulong2(0, 0), true);
            simdgroup_load(kt1, Ks + 8 * D + d * 8, D, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s0, qf[d], kt0, s0);
            simdgroup_multiply_accumulate(s1, qf[d], kt1, s1);
        }
        simdgroup_store(s0, Ssg, BK_);
        simdgroup_store(s1, Ssg + 8, BK_);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax: 4 lanes per row, 4 columns each
        float sv[4];
        float mloc = -INFINITY;
        for (int j = 0; j < 4; j++) {
            bool ok = k0 + c0 + j < NK;
            sv[j] = ok ? Ssg[r * BK_ + c0 + j] * c : -INFINITY;
            mloc = max(mloc, sv[j]);
        }
        mloc = max(mloc, simd_shuffle_xor(mloc, ushort(1)));
        mloc = max(mloc, simd_shuffle_xor(mloc, ushort(2)));
        float m_new = max(m_i, mloc);
        float alpha = exp2(m_i - m_new);
        float ps = 0.0f;
        for (int j = 0; j < 4; j++) {
            float p = exp2(sv[j] - m_new);
            ps += p;
            Ssg[r * BK_ + c0 + j] = p;
        }
        ps += simd_shuffle_xor(ps, ushort(1));
        ps += simd_shuffle_xor(ps, ushort(2));
        l_i = l_i * alpha + ps;
        m_i = m_new;
        if (c0 == 0) Dsg[r * 8 + r] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(alpha) O + P V
        simdgroup_float8x8 dm, p0, p1;
        simdgroup_load(dm, Dsg, 8);
        simdgroup_load(p0, Ssg, BK_);
        simdgroup_load(p1, Ssg + 8, BK_);
        for (int d = 0; d < ND; d++) {
            simdgroup_float8x8 v0, v1, t;
            simdgroup_load(v0, Vs + 0 * D + d * 8, D);
            simdgroup_load(v1, Vs + 8 * D + d * 8, D);
            simdgroup_multiply(t, dm, of[d]);
            simdgroup_multiply_accumulate(t, p0, v0, t);
            simdgroup_multiply_accumulate(of[d], p1, v1, t);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O / l -> bf16
    for (int d = 0; d < ND; d++) simdgroup_store(of[d], KV + (sg * 8) * D + d * 8, D);
    if (c0 == 0) Ls[sg * 8 + r] = l_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < BQ_ * D; i += NT) {
        int rr = i / D, cc = i % D;
        if (q0 + rr < NQ) oh[size_t(q0 + rr) * D + cc] = static_cast<T>(KV[i] / Ls[rr]);
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="shutter_flash_attn",
            input_names=["q", "k", "v", "scale"],
            output_names=["o"],
            source=_SOURCE,
            header=_HEADER,
        )
    return _KERNEL


def kernel(q, k, v, scale=None):
    B, H, NQ, D = q.shape
    NK = k.shape[2]
    assert D % 8 == 0 and 2 * BK * D * 4 >= BQ * D * 4
    scale = D ** -0.5 if scale is None else scale
    return _kernel()(
        inputs=[q, k, v, mx.array([scale], dtype=mx.float32)],
        template=[("T", q.dtype), ("NQ", NQ), ("NK", NK), ("D", D), ("H", H)],
        grid=(((NQ + BQ - 1) // BQ) * 128, H, B),
        threadgroup=(128, 1, 1),
        output_shapes=[(B, H, NQ, D)],
        output_dtypes=[q.dtype],
    )[0]


def flash_attn(q, k, v, scale=None, use_kernel: bool | None = None):
    if use_kernel is None:
        use_kernel = metal_available()
    return (kernel if use_kernel else reference)(q, k, v, scale)


BIT_IDENTICAL = False


def make_inputs(B=1, H=2, NQ=200, NK=300, D=128, seed=0):
    k = mx.random.split(mx.random.key(seed), 3)
    q = mx.random.normal((B, H, NQ, D), key=k[0]).astype(mx.bfloat16)
    kk = mx.random.normal((B, H, NK, D), key=k[1]).astype(mx.bfloat16)
    v = mx.random.normal((B, H, NK, D), key=k[2]).astype(mx.bfloat16)
    return q, kk, v


def selftest(B=1, H=2, NQ=200, NK=300, D=128) -> dict:
    q, k, v = make_inputs(B, H, NQ, NK, D)
    ref = reference(q, k, v)
    ran = metal_available()
    got = kernel(q, k, v) if ran else reference_tiled(q, k, v, bk=64)
    d = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
    return {"name": "flash_attn", "ran_kernel": ran, "max_abs_diff": float(mx.max(d)),
            "frac_exact": float(mx.mean(d == 0)), "bit_identical": bool(mx.all(d == 0))}
