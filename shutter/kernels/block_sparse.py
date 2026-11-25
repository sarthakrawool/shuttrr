"""Block-sparse 3D self-attention (opt-in, NOT bit-identical), in the style of LongCat-Video's BSA.

Upstream LongCat-Video ships `flash_attn_bsa_3d` with `sparsity=0.9375`, `chunk_3d_shape_q = chunk_3d_shape_k =
[4, 4, 4]` (base DiT config). The Avatar 1.5/1.6 config has `enable_bsa: false`, i.e. the Avatar model was trained
and distilled with DENSE attention, so turning this on is an approximation, never lossless. See docs/KERNELS.md.

Algorithm (pure MLX, no custom Metal kernel):

1. The query tokens form a 3D grid (T, H, W) and the key tokens another one (T', H, W), both in the model's
   row-major token order. Each grid is cut into chunks of `chunk` = (4, 4, 4) = 64 tokens; grids whose sizes are not
   multiples of the chunk are zero-padded and the padding is masked out (padded keys never receive weight, padded
   queries are dropped). Upstream requires divisible grids; LongCat-Avatar latents are not (23 noise frames,
   30 x 52 or 44 x 82 patches), so padding is needed here.
2. Mean-pool Q and K per chunk (over the valid tokens only), score every (query chunk, key chunk) pair with
   pooled_q . pooled_k, and keep the top `ceil((1 - sparsity) * n_key_chunks)` key chunks per query chunk.
3. Optional dense prefix: `k_prefix` leading key tokens that are not part of the key grid (the Avatar
   conditioning/reference frames that noise queries attend to) are cut into 1D chunks and, with `keep_prefix=True`,
   always kept for every query chunk.
4. For every query chunk, gather its kept key/value chunks and run dense `mx.fast.scaled_dot_product_attention`
   on [64 queries] x [kept keys]. Loops over head groups and query-chunk groups bound the gathered K/V memory.

`sparsity == 0` keeps every chunk; the function then calls the dense SDPA directly (exactly the dense result).
`force_gather=True` runs the sparse machinery anyway (used by the tests to check the gather path).

The cost on the GPU is unmeasured (this container has CPU-only MLX): the gather writes every kept K/V chunk once per
query chunk, so the saving depends on how fast MLX's SDPA runs 64-row query tiles and on gather bandwidth.
`scripts/bench_speed.py` measures it on the Mac.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import numpy as np

try:  # importable without MLX (docs, planning); the functions need it
    import mlx.core as mx
except Exception:  # pragma: no cover - exercised only without MLX
    mx = None

__all__ = ["block_sparse_attention", "dense_reference", "chunk_layout", "keep_count", "to_chunks", "from_chunks",
           "make_inputs", "pattern", "quality", "quality_table", "selftest", "UPSTREAM_SPARSITY", "UPSTREAM_CHUNK"]

UPSTREAM_SPARSITY = 0.9375          # LongCat-Video dit/config.json bsa_params
UPSTREAM_CHUNK = (4, 4, 4)


# ----------------------------------------------------------------------------- static layout (numpy, no MLX)


def chunk_layout(grid: Sequence[int], chunk: Sequence[int]) -> dict[str, Any]:
    """Chunking of a (T, H, W) grid: number of chunks per axis, padding, and the [n_chunks, chunk_size] validity
    mask in chunk order (chunk-major, then the 4x4x4 positions inside a chunk)."""
    T, H, W = (int(v) for v in grid)
    ct, ch, cw = (int(v) for v in chunk)
    if min(T, H, W) < 1 or min(ct, ch, cw) < 1:
        raise ValueError(f"bad grid {grid} or chunk {chunk}")
    nt, nh, nw = -(-T // ct), -(-H // ch), -(-W // cw)
    valid = np.zeros((nt * ct, nh * ch, nw * cw), bool)
    valid[:T, :H, :W] = True
    vc = valid.reshape(nt, ct, nh, ch, nw, cw).transpose(0, 2, 4, 1, 3, 5).reshape(nt * nh * nw, ct * ch * cw)
    return {"grid": (T, H, W), "chunk": (ct, ch, cw), "n": (nt, nh, nw), "n_chunks": nt * nh * nw,
            "chunk_size": ct * ch * cw, "pad": (nt * ct - T, nh * ch - H, nw * cw - W), "valid": vc,
            "padded": bool((~vc).any())}


def _prefix_layout(n_prefix: int, chunk_size: int) -> dict[str, Any]:
    n = -(-n_prefix // chunk_size) if n_prefix else 0
    valid = np.zeros((n * chunk_size,), bool)
    valid[:n_prefix] = True
    return {"n_chunks": n, "valid": valid.reshape(n, chunk_size), "pad": n * chunk_size - n_prefix}


def keep_count(n_candidates: int, sparsity: float) -> int:
    """Key chunks kept per query chunk: ceil((1 - sparsity) * n), at least 1, at most n."""
    if not 0.0 <= float(sparsity) < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    return int(min(n_candidates, max(1, math.ceil((1.0 - float(sparsity)) * n_candidates - 1e-9))))


# ----------------------------------------------------------------------------- token order <-> chunk order


def to_chunks(x, grid, chunk):
    """[B, Hd, T*H*W, D] (row-major t, h, w) -> [B, Hd, n_chunks, chunk_size, D], zero-padded."""
    lay = chunk_layout(grid, chunk)
    B, Hd, N, D = x.shape
    T, H, W = lay["grid"]
    if N != T * H * W:
        raise ValueError(f"{N} tokens do not match grid {lay['grid']}")
    (nt, nh, nw), (ct, ch, cw), (pt, ph, pw) = lay["n"], lay["chunk"], lay["pad"]
    x = x.reshape(B, Hd, T, H, W, D)
    if pt or ph or pw:
        x = mx.pad(x, [(0, 0), (0, 0), (0, pt), (0, ph), (0, pw), (0, 0)])
    x = x.reshape(B, Hd, nt, ct, nh, ch, nw, cw, D).transpose(0, 1, 2, 4, 6, 3, 5, 7, 8)
    return x.reshape(B, Hd, nt * nh * nw, ct * ch * cw, D)


def from_chunks(xc, grid, chunk):
    """Inverse of `to_chunks`, padding dropped: [B, Hd, n_chunks, chunk_size, D] -> [B, Hd, T*H*W, D]."""
    lay = chunk_layout(grid, chunk)
    B, Hd, _, _, D = xc.shape
    T, H, W = lay["grid"]
    (nt, nh, nw), (ct, ch, cw) = lay["n"], lay["chunk"]
    x = xc.reshape(B, Hd, nt, nh, nw, ct, ch, cw, D).transpose(0, 1, 2, 5, 3, 6, 4, 7, 8)
    x = x.reshape(B, Hd, nt * ct, nh * ch, nw * cw, D)
    if lay["padded"]:
        x = x[:, :, :T, :H, :W]
    return x.reshape(B, Hd, T * H * W, D)


def _prefix_chunks(x, n_prefix, chunk_size):
    """[B, Hd, n_prefix, D] -> [B, Hd, n_prefix_chunks, chunk_size, D] (1D chunks, zero-padded)."""
    lay = _prefix_layout(n_prefix, chunk_size)
    B, Hd, _, D = x.shape
    if lay["pad"]:
        x = mx.pad(x, [(0, 0), (0, 0), (0, lay["pad"]), (0, 0)])
    return x.reshape(B, Hd, lay["n_chunks"], chunk_size, D)


# ----------------------------------------------------------------------------- the attention


def dense_reference(q, k, v, scale=None):
    """The dense path the model runs: one fused SDPA."""
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5 if scale is None else scale)


def _pool(xc, valid: np.ndarray):
    """Mean over the valid tokens of each chunk, fp32. xc [B, Hd, n, cs, D], valid [n, cs]."""
    w = mx.array(valid.astype(np.float32))[None, None, :, :, None]
    cnt = mx.array(np.maximum(valid.sum(1), 1).astype(np.float32))[None, None, :, None]
    return (xc.astype(mx.float32) * w).sum(axis=3) / cnt


def pattern(q_grid: Sequence[int], k_grid: Optional[Sequence[int]] = None, *, k_prefix: int = 0,
            sparsity: float = UPSTREAM_SPARSITY, chunk: Sequence[int] = UPSTREAM_CHUNK,
            keep_prefix: bool = True) -> dict[str, Any]:
    """Static description of what `block_sparse_attention` computes for these shapes (no MLX needed): chunk counts,
    kept / skipped (query chunk, key chunk) pairs per batch item and head, and `score_fraction` = attention scores
    computed / dense scores (padding included, so it can exceed 1 - sparsity on small or ragged grids)."""
    k_grid = tuple(q_grid) if k_grid is None else tuple(k_grid)
    ql, kl = chunk_layout(q_grid, chunk), chunk_layout(k_grid, chunk)
    cs = ql["chunk_size"]
    Nq, Nk = math.prod(ql["grid"]), int(k_prefix) + math.prod(kl["grid"])
    pl = _prefix_layout(int(k_prefix), cs)
    npc, nkc, nqc = pl["n_chunks"], kl["n_chunks"], ql["n_chunks"]
    forced = npc if (keep_prefix and npc) else 0
    n_cand = nkc + (0 if forced else npc)
    kk = keep_count(n_cand, sparsity)
    n_sel = forced + kk
    st = {"q_grid": ql["grid"], "k_grid": kl["grid"], "chunk": ql["chunk"], "sparsity": float(sparsity),
          "q_tokens": Nq, "k_tokens": Nk, "q_chunks": nqc, "k_chunks": nkc, "prefix_tokens": int(k_prefix),
          "prefix_chunks": npc, "prefix_kept": bool(forced), "selected_per_q_chunk": kk, "kept_per_q_chunk": n_sel,
          "pairs_total": nqc * (nkc + npc), "pairs_kept": nqc * n_sel, "q_padding_tokens": nqc * cs - Nq,
          "k_padding_tokens": nkc * cs - math.prod(kl["grid"]) + pl["pad"],
          "nothing_skipped": kk == n_cand}
    st["pairs_skipped"] = st["pairs_total"] - st["pairs_kept"]
    st["density"] = st["pairs_kept"] / st["pairs_total"]
    st["score_fraction"] = (nqc * cs * n_sel * cs) / (Nq * Nk)
    return st


def block_sparse_attention(q, k, v, q_grid: Sequence[int], k_grid: Optional[Sequence[int]] = None, *,
                           sparsity: float = UPSTREAM_SPARSITY, chunk: Sequence[int] = UPSTREAM_CHUNK,
                           scale: Optional[float] = None, k_prefix: int = 0, keep_prefix: bool = True,
                           head_chunk: Optional[int] = None, q_chunk_group: Optional[int] = None,
                           max_gather_bytes: Optional[int] = None, force_gather: bool = False,
                           dense_fn: Optional[Callable] = None, return_stats: bool = False):
    """Block-sparse attention. q [B, Hd, Nq, D] with Nq = prod(q_grid); k, v [B, Hd, Nk, D] with
    Nk = k_prefix + prod(k_grid) (k_grid defaults to q_grid). Returns [B, Hd, Nq, D] (and a stats dict).

    head_chunk: heads per gather/SDPA group (None = all); q_chunk_group: query chunks per group (None = all);
    max_gather_bytes: shrink the query-chunk group so the gathered K + V of one group stay under this.
    These only bound memory; the result does not depend on them.
    dense_fn(q, k, v): what to call when nothing is skipped (default: one fused SDPA)."""
    if mx is None:
        raise RuntimeError("block_sparse_attention needs MLX")
    k_grid = tuple(q_grid) if k_grid is None else tuple(k_grid)
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    dense_fn = dense_fn or (lambda a, b, c: dense_reference(a, b, c, scale))
    B, Hd, Nq, D = q.shape
    Nk = k.shape[2]
    stats = pattern(q_grid, k_grid, k_prefix=k_prefix, sparsity=sparsity, chunk=chunk, keep_prefix=keep_prefix)
    ql, kl = chunk_layout(q_grid, chunk), chunk_layout(k_grid, chunk)
    cs = ql["chunk_size"]
    if Nq != math.prod(ql["grid"]):
        raise ValueError(f"q has {Nq} tokens, grid {ql['grid']} has {math.prod(ql['grid'])}")
    if Nk != k_prefix + math.prod(kl["grid"]):
        raise ValueError(f"k has {Nk} tokens, expected prefix {k_prefix} + grid {kl['grid']}")
    pl = _prefix_layout(k_prefix, cs)
    npc, nkc, nqc = pl["n_chunks"], kl["n_chunks"], ql["n_chunks"]
    forced = npc if stats["prefix_kept"] else 0
    n_cand = nkc + (0 if forced else npc)
    kk = stats["selected_per_q_chunk"]
    n_sel = stats["kept_per_q_chunk"]
    if kk == n_cand and not force_gather:
        stats["path"] = "dense (nothing to skip)"
        out = dense_fn(q, k, v)
        return (out, stats) if return_stats else out
    stats["path"] = "block-sparse"

    qc = to_chunks(q, ql["grid"], chunk)                                   # [B, Hd, nqc, cs, D]
    kc, vc = to_chunks(k[:, :, k_prefix:], kl["grid"], chunk), to_chunks(v[:, :, k_prefix:], kl["grid"], chunk)
    valid_k = kl["valid"]
    if npc:
        kc = mx.concatenate([_prefix_chunks(k[:, :, :k_prefix], k_prefix, cs), kc], axis=2)
        vc = mx.concatenate([_prefix_chunks(v[:, :, :k_prefix], k_prefix, cs), vc], axis=2)
        valid_k = np.concatenate([pl["valid"], valid_k], axis=0)
    n_all = npc + nkc                                                      # prefix chunks first, then the grid's

    # selection: pooled scores of every (query chunk, candidate key chunk) pair, top-kk per query chunk
    qp = _pool(qc, ql["valid"])                                            # [B, Hd, nqc, D] fp32
    kp = _pool(kc, valid_k)                                                # [B, Hd, n_all, D]
    if forced:
        scores = qp @ kp[:, :, npc:].swapaxes(-1, -2)                      # [B, Hd, nqc, nkc]
        top = mx.argpartition(-scores, kk - 1, axis=-1)[..., :kk] + npc
        idx = mx.concatenate([mx.broadcast_to(mx.arange(npc, dtype=top.dtype), (B, Hd, nqc, npc)), top], axis=-1)
    else:
        scores = qp @ kp.swapaxes(-1, -2)
        idx = mx.argpartition(-scores, kk - 1, axis=-1)[..., :kk]
    idx = mx.sort(idx.astype(mx.int32), axis=-1)                           # [B, Hd, nqc, n_sel], ascending

    need_mask = bool((~valid_k).any())
    valid_mx = mx.array(valid_k) if need_mask else None
    hc = Hd if not head_chunk else max(1, min(Hd, int(head_chunk)))
    qg = nqc if not q_chunk_group else max(1, min(nqc, int(q_chunk_group)))
    if max_gather_bytes:
        per_chunk = 2 * B * hc * n_sel * cs * D * k.dtype.size          # K + V gathered for one query chunk
        qg = max(1, min(qg, int(max_gather_bytes) // max(1, per_chunk)))
    stats["q_chunk_group"], stats["head_chunk"] = qg, hc
    heads_out = []
    for h0 in range(0, Hd, hc):
        h1 = min(Hd, h0 + hc)
        nh = h1 - h0
        kflat = kc[:, h0:h1].reshape(B * nh * n_all, cs, D)
        vflat = vc[:, h0:h1].reshape(B * nh * n_all, cs, D)
        base = (mx.arange(B * nh, dtype=mx.int32) * n_all).reshape(B, nh, 1, 1)
        groups = []
        for g0 in range(0, nqc, qg):
            g1 = min(nqc, g0 + qg)
            ng = g1 - g0
            ii = idx[:, h0:h1, g0:g1]                                      # [B, nh, ng, n_sel]
            flat = (ii + base).reshape(-1)
            ks = mx.take(kflat, flat, axis=0).reshape(B * nh * ng, 1, n_sel * cs, D)
            vs = mx.take(vflat, flat, axis=0).reshape(B * nh * ng, 1, n_sel * cs, D)
            qs = qc[:, h0:h1, g0:g1].reshape(B * nh * ng, 1, cs, D)
            mask = None
            if need_mask:
                mask = mx.take(valid_mx, ii.reshape(-1), axis=0).reshape(B * nh * ng, 1, 1, n_sel * cs)
            o = mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=scale, mask=mask)
            groups.append(o.reshape(B, nh, ng, cs, D))
        heads_out.append(groups[0] if len(groups) == 1 else mx.concatenate(groups, axis=2))
    oc = heads_out[0] if len(heads_out) == 1 else mx.concatenate(heads_out, axis=1)
    out = from_chunks(oc, ql["grid"], chunk).astype(q.dtype)
    return (out, stats) if return_stats else out


# ----------------------------------------------------------------------------- inputs and quality


def make_inputs(grid, heads=2, d=64, kind="random", seed=0, prefix_frames=0, dtype=None):
    """q, k, v [1, heads, N, d] for a (T, H, W) grid (keys optionally with `prefix_frames` extra leading frames).

    kind="random": i.i.d. normal (attention close to uniform: the worst case for any block selection).
    kind="structured": smooth 3D fields: q and k share low-frequency Fourier features of the token position plus
    noise, scaled so attention concentrates on similar (mostly nearby) positions, and v is a smooth field, roughly
    what DiT self-attention on video latents looks like (local, content-dependent).
    kind="peaked": the same with 1.2x sharper logits (attention more concentrated). The block-sparse error depends
    strongly on how concentrated attention is, so synthetic numbers bracket, but do not predict, the real model."""
    rng = np.random.default_rng(seed)
    T, H, W = grid
    Tk = T + prefix_frames
    if kind == "random":
        q = rng.standard_normal((1, heads, T * H * W, d))
        k = rng.standard_normal((1, heads, Tk * H * W, d))
        v = rng.standard_normal((1, heads, Tk * H * W, d))
    elif kind in ("structured", "peaked"):
        def coords(t0, t1):
            t, h, w = np.meshgrid(np.arange(t0, t1), np.arange(H), np.arange(W), indexing="ij")
            return np.stack([t / max(Tk, 1), h / max(H, 1), w / max(W, 1)], -1).reshape(-1, 3)

        cq, ck = coords(prefix_frames, Tk), coords(0, Tk)
        qs, ks, vs = [], [], []
        for _ in range(heads):
            f = rng.normal(0, 2.5, (d // 2, 3))
            b = rng.uniform(0, 2 * np.pi, d // 2)

            def feat(c):
                a = 2 * np.pi * c @ f.T + b
                return np.concatenate([np.cos(a), np.sin(a)], -1) * np.sqrt(2.0)

            sharp = 1.0 if kind == "structured" else 1.2
            qs.append(sharp * feat(cq) + 0.3 * rng.standard_normal((len(cq), d)))
            ks.append(sharp * feat(ck) + 0.3 * rng.standard_normal((len(ck), d)))
            fv = rng.normal(0, 1.5, (d, 3))
            vs.append(np.sin(2 * np.pi * ck @ fv.T + rng.uniform(0, 2 * np.pi, d)) + 0.1 * rng.standard_normal((len(ck), d)))
        q, k, v = (np.stack(a)[None] for a in (qs, ks, vs))
    else:
        raise ValueError(kind)
    arrs = [mx.array(a.astype(np.float32)) for a in (q, k, v)]
    if dtype is not None:
        arrs = [a.astype(dtype) for a in arrs]
    return tuple(arrs)


def _metrics(got, ref) -> dict[str, float]:
    g = np.asarray(got.astype(mx.float32), dtype=np.float64).ravel()
    r = np.asarray(ref.astype(mx.float32), dtype=np.float64).ravel()
    mse = float(np.mean((g - r) ** 2))
    peak = float(np.max(np.abs(r)))
    psnr = float("inf") if mse == 0 else 10 * math.log10(peak * peak / mse)
    cos = float(np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
    return {"psnr_db": psnr, "cosine": cos, "max_abs": float(np.max(np.abs(g - r))), "rel_l2": float(
        np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-30))}


def quality(sparsities=(0.5, 0.75, UPSTREAM_SPARSITY), kinds=("random", "structured", "peaked"), grid=(8, 16, 16),
            prefix_frames=0, heads=2, d=64, seed=0, chunk=UPSTREAM_CHUNK) -> list[dict[str, Any]]:
    """Block-sparse vs dense on the same inputs (fp32, CPU or GPU): PSNR (peak = max |dense|), cosine, max abs."""
    rows = []
    for kind in kinds:
        q, k, v = make_inputs(grid, heads, d, kind, seed, prefix_frames)
        ref = dense_reference(q, k, v)
        k_grid = tuple(grid)
        for s in sparsities:
            out, st = block_sparse_attention(q, k, v, grid, k_grid, sparsity=s, chunk=chunk,
                                             k_prefix=prefix_frames * grid[1] * grid[2], return_stats=True)
            rows.append({"kind": kind, "grid": list(grid), "prefix_frames": prefix_frames, "sparsity": s,
                         "density": round(st["density"], 4), "score_fraction": round(st["score_fraction"], 4),
                         **{m: round(x, 6) for m, x in _metrics(out, ref).items()}})
    return rows


def quality_table(rows) -> str:
    L = [f"{'kind':10} {'grid':12} {'pre':>3} {'sparsity':>8} {'density':>7} {'scores':>7} {'PSNR dB':>8} "
         f"{'cosine':>8} {'max abs':>8}"]
    for r in rows:
        L.append(f"{r['kind']:10} {'x'.join(map(str, r['grid'])):12} {r['prefix_frames']:>3} {r['sparsity']:8.4f} "
                 f"{r['density']:7.4f} {r['score_fraction']:7.4f} {r['psnr_db']:8.2f} {r['cosine']:8.5f} "
                 f"{r['max_abs']:8.4f}")
    return "\n".join(L)


def selftest() -> dict[str, Any]:
    """Sparsity 0 equals dense exactly; the forced gather path with everything kept matches dense to fp32 level."""
    q, k, v = make_inputs((4, 6, 10), 2, 32, "random", 1, prefix_frames=1)
    ref = dense_reference(q, k, v)
    ex = block_sparse_attention(q, k, v, (4, 6, 10), sparsity=0.0, k_prefix=60)
    forced = block_sparse_attention(q, k, v, (4, 6, 10), sparsity=0.0, k_prefix=60, force_gather=True)
    return {"name": "block_sparse", "sparsity0_bit_identical": bool(mx.array_equal(ex, ref)),
            "gather_all_max_abs": float(mx.max(mx.abs(forced - ref)))}


if __name__ == "__main__":  # pragma: no cover
    print(selftest())
    print(quality_table(quality()))
    print(quality_table(quality(grid=(6, 10, 14), prefix_frames=1)))
