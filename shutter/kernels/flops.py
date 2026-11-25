"""Analytic operation count for one LongCat-Video-Avatar 1.6 DiT step.

Pure Python (no MLX needed). Every number here is arithmetic on the model's shapes; nothing is measured,
except the constants in `MEASURED_*` (copied from the Mac profile, see their comments) which the ceiling
functions take as inputs.

Shapes follow the model code (longcat_video_avatar/models/avatar/longcat_video_dit_avatar.py, attention.py,
blocks.py, rope_3d.py, runtime_lora.py):

- latent grid T = (frames-1)//4 + 1, H/8/2 x W/8/2 tokens per frame (VAE 8x spatial, 4x temporal, patch 1x2x2)
- production runs with num_cond_latents = 1 (the reference frame): the self-attention runs
  noise queries x all keys plus cond queries x cond keys (REPORT: "83K queries x 86.6K keys");
  text and audio cross-attention see only the noise tokens.
- block: adaLN(6C) + audio adaLN(3C) linears on B*T rows; LN-modulate; qkv (C->3C, bias); q/k RMSNorm (fp32);
  3D RoPE (fp32); SDPA 32 heads x 128; proj; gated residual; text cross-attn (pre-LN, q C->C, kv C->2C on text,
  q/k RMSNorm, SDPA, proj); audio cross-attn (pre-LN, q C->C, kv 768->2C on 32 tokens/frame, q/k RMSNorm,
  per-frame SDPA, proj, LN-modulate, gated residual); FFN SwiGLU C->F, F->C with F = 11008; gated residual.
- runtime LoRA (rank 128 by default, `alpha/rank = 64/128` in runtime_lora.py) on 7 targets per block.
  The 7 targets are not listed in the code; the set below (qkv, attn.proj, cross_attn.q_linear,
  cross_attn.kv_linear, ffn.w1/w2/w3) is the one whose parameter count reproduces the 2.52 GB LoRA file
  (13.1 M params/block at rank 128 in fp32, or rank 256 in bf16). Parameterised; small either way.

Conventions: 1 multiply-add (MAC) = 2 FLOPs. "elementwise" FLOPs are rough per-element op counts
(norms, softmax, RoPE, modulation) and are reported separately from matmul FLOPs, because utilisation is
measured against a matmul peak. "bytes" are the compulsory DRAM traffic of each op class if it were one
kernel reading its inputs once and writing its outputs once (bf16 activations, 8-bit weights + one bf16
scale and bias per group of 64). Real MLX traffic is higher (fp32 temporaries, transposed copies); the
attention-specific traffic model is in `attention_traffic`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

OP_CLASSES = (
    "self_attn_qk",  # Q K^T of the self-attention
    "softmax",  # exp/max/sum over the self-attention scores
    "self_attn_av",  # P V of the self-attention
    "qkv_linear",
    "attn_proj",
    "ffn",
    "cross_text",  # whole text cross-attention (linears + SDPA)
    "cross_audio",  # whole audio cross-attention (linears + SDPA)
    "norms",  # q/k RMSNorm (self + cross) and the two pre-cross LayerNorms
    "rope",
    "modulation",  # adaLN linears, LN+scale+shift (x3), gated residuals (x3)
    "lora",
    "other",  # per step, outside the blocks: patch embed, t/caption embed, audio proj, final layer
)

DEFAULT_LORA_TARGETS = ("attn.qkv", "attn.proj", "cross_attn.q_linear", "cross_attn.kv_linear",
                        "ffn.w1", "ffn.w2", "ffn.w3")


def swiglu_hidden(hidden: int, mlp_ratio: float = 4, multiple_of: int = 256) -> int:
    """FeedForwardSwiGLU's internal width: int(2*h*ratio/3) rounded up to multiple_of (4096 -> 11008)."""
    h = int(2 * int(hidden * mlp_ratio) / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


@dataclass
class Config:
    height: int = 704
    width: int = 1312
    frames: int = 93
    hidden: int = 4096
    depth: int = 48
    heads: int = 32
    mlp_ratio: float = 4
    text_tokens: int = 512  # umT5 pad length = upper bound; packing keeps only the real prompt tokens
    audio_tokens: int = 32  # audio context tokens per latent frame (AudioProjModel context_tokens)
    audio_dim: int = 768
    lora_rank: int = 128
    lora_targets: tuple = DEFAULT_LORA_TARGETS
    num_cond_latents: int = 1
    vae_spatial: int = 8
    vae_temporal: int = 4
    patch: tuple = (1, 2, 2)
    in_channels: int = 16
    adaln_dim: int = 512
    act_bytes: int = 2  # bf16
    weight_bits: int = 8
    group_size: int = 64
    cfg_passes: int = 1  # distilled model: one DiT pass per step

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def ffn_hidden(self) -> int:
        return swiglu_hidden(self.hidden, self.mlp_ratio)

    @property
    def latent_frames(self) -> int:
        return (self.frames - 1) // self.vae_temporal + 1

    @property
    def grid(self) -> tuple[int, int, int]:
        pt, ph, pw = self.patch
        return (self.latent_frames // pt, self.height // self.vae_spatial // ph, self.width // self.vae_spatial // pw)

    @property
    def tokens(self) -> int:
        t, h, w = self.grid
        return t * h * w

    @property
    def tokens_per_frame(self) -> int:
        return self.grid[1] * self.grid[2]

    @property
    def cond_tokens(self) -> int:
        return self.num_cond_latents * self.tokens_per_frame

    @property
    def noise_tokens(self) -> int:
        return self.tokens - self.cond_tokens

    @property
    def weight_bytes(self) -> float:
        """Bytes per weight element: the packed value plus its share of one bf16 scale and one bf16 bias."""
        return self.weight_bits / 8 + 2 * 2 / self.group_size

    @property
    def attn_pairs(self) -> int:
        """Query-key pairs per head in the self-attention (noise x all + cond x cond)."""
        return self.noise_tokens * self.tokens + self.cond_tokens * self.cond_tokens


@dataclass
class Op:
    name: str
    cls: str
    macs: float = 0.0  # multiply-adds of matmuls
    elem: float = 0.0  # elementwise FLOPs
    bytes: float = 0.0  # compulsory DRAM traffic

    @property
    def matmul_flops(self) -> float:
        return 2 * self.macs

    @property
    def flops(self) -> float:
        return 2 * self.macs + self.elem


def _linear(cfg: Config, name, cls, m, k, n, bias=True, quantized=True) -> Op:
    a = cfg.act_bytes
    wb = cfg.weight_bytes if quantized else a
    return Op(name, cls, macs=m * k * n, elem=m * n if bias else 0,
              bytes=m * k * a + k * n * wb + m * n * a + (n * a if bias else 0))


def _sdpa(cfg: Config, name, cls_qk, cls_sm, cls_av, nq, nk, pairs=None) -> list[Op]:
    """Fused SDPA over `pairs` query-key pairs per head (default nq*nk)."""
    a, h, d = cfg.act_bytes, cfg.heads, cfg.head_dim
    pairs = nq * nk if pairs is None else pairs
    return [
        Op(f"{name}.qk", cls_qk, macs=h * pairs * d, bytes=h * (nq + nk) * d * a),
        Op(f"{name}.softmax", cls_sm, elem=5 * h * pairs),
        Op(f"{name}.av", cls_av, macs=h * pairs * d, bytes=h * nk * d * a + h * nq * d * a),
    ]


def _elementwise(cfg, name, cls, n_elem, flops_per, tensors_in=1, tensors_out=1, extra_bytes=0.0) -> Op:
    return Op(name, cls, elem=n_elem * flops_per,
              bytes=n_elem * cfg.act_bytes * (tensors_in + tensors_out) + extra_bytes)


def _lora(cfg: Config, name, m, k, outs) -> Op:
    r, a = cfg.lora_rank, cfg.act_bytes
    n_split = len(outs)
    macs = m * k * r * n_split + sum(m * r * o for o in outs)
    w = (r * n_split * k + sum(o * r for o in outs)) * a
    traffic = m * k * a + 2 * m * r * n_split * a + 2 * m * sum(outs) * a + w  # re-read x, h, y += d
    return Op(f"lora.{name}", "lora", macs=macs, elem=2 * m * sum(outs), bytes=traffic)


def block_ops(cfg: Config) -> list[Op]:
    C, F, H, D = cfg.hidden, cfg.ffn_hidden, cfg.heads, cfg.head_dim
    N, Nn, Nc = cfg.tokens, cfg.noise_tokens, cfg.cond_tokens
    T = cfg.latent_frames
    Tn = T - cfg.num_cond_latents
    L, A = cfg.text_tokens, cfg.audio_tokens
    ops: list[Op] = []

    # modulation: adaLN linears on B*T rows (tiny), 3 LN-modulates, 3 gated residuals
    ops.append(_linear(cfg, "adaLN_modulation", "modulation", T, cfg.adaln_dim, 6 * C))
    ops.append(_linear(cfg, "audio_adaLN_modulation", "modulation", Tn, cfg.adaln_dim, 3 * C))
    ops.append(_elementwise(cfg, "modulate.attn", "modulation", N * C, 10))
    ops.append(_elementwise(cfg, "modulate.audio_out", "modulation", Nn * C, 10))
    ops.append(_elementwise(cfg, "modulate.ffn", "modulation", N * C, 10))
    ops.append(_elementwise(cfg, "residual.attn", "modulation", N * C, 2, tensors_in=2))
    ops.append(_elementwise(cfg, "residual.audio", "modulation", N * C, 2, tensors_in=2))
    ops.append(_elementwise(cfg, "residual.ffn", "modulation", N * C, 2, tensors_in=2))

    # self-attention
    ops.append(_linear(cfg, "attn.qkv", "qkv_linear", N, C, 3 * C))
    ops.append(_elementwise(cfg, "attn.qk_rmsnorm", "norms", 2 * N * C, 4))
    ops.append(_elementwise(cfg, "attn.rope", "rope", 2 * N * C, 3,
                            extra_bytes=2 * N * D * 4))  # fp32 cos/sin tables
    ops += _sdpa(cfg, "attn.sdpa", "self_attn_qk", "softmax", "self_attn_av", N, N, pairs=cfg.attn_pairs)
    ops.append(_linear(cfg, "attn.proj", "attn_proj", N, C, C))

    # text cross-attention (noise tokens only when num_cond_latents > 0)
    ops.append(_elementwise(cfg, "pre_crs_attn_norm", "norms", N * C, 7))
    ops.append(_linear(cfg, "cross_attn.q_linear", "cross_text", Nn, C, C))
    ops.append(_linear(cfg, "cross_attn.kv_linear", "cross_text", L, C, 2 * C))
    ops.append(_elementwise(cfg, "cross_attn.qk_rmsnorm", "cross_text", (Nn + L) * C, 4))
    ops += _sdpa(cfg, "cross_attn.sdpa", "cross_text", "cross_text", "cross_text", Nn, L)
    ops.append(_linear(cfg, "cross_attn.proj", "cross_text", Nn, C, C))

    # audio cross-attention: each latent frame's tokens attend to that frame's 32 audio tokens
    ops.append(_elementwise(cfg, "pre_video_crs_attn_norm", "norms", N * C, 7))
    ops.append(_linear(cfg, "audio_cross_attn.q_linear", "cross_audio", Nn, C, C))
    ops.append(_linear(cfg, "audio_cross_attn.kv_linear", "cross_audio", Tn * A, cfg.audio_dim, 2 * C))
    ops.append(_elementwise(cfg, "audio_cross_attn.qk_rmsnorm", "cross_audio", (Nn + Tn * A) * C, 4))
    ops += _sdpa(cfg, "audio_cross_attn.sdpa", "cross_audio", "cross_audio", "cross_audio", Nn, Tn * A,
                 pairs=Nn * A)
    ops.append(_linear(cfg, "audio_cross_attn.proj", "cross_audio", Nn, C, C))

    # FFN (SwiGLU)
    ops.append(_linear(cfg, "ffn.w1", "ffn", N, C, F, bias=False))
    ops.append(_linear(cfg, "ffn.w3", "ffn", N, C, F, bias=False))
    ops.append(_elementwise(cfg, "ffn.silu_mul", "ffn", N * F, 5, tensors_in=2))
    ops.append(_linear(cfg, "ffn.w2", "ffn", N, F, C, bias=False))

    # runtime LoRA (one active LoRA; multiplier-0 LoRAs are skipped by runtime_lora.py)
    shapes = {
        "attn.qkv": (N, C, [C, C, C]), "attn.proj": (N, C, [C]),
        "cross_attn.q_linear": (Nn, C, [C]), "cross_attn.kv_linear": (L, C, [C, C]),
        "cross_attn.proj": (Nn, C, [C]),
        "audio_cross_attn.q_linear": (Nn, C, [C]), "audio_cross_attn.kv_linear": (Tn * A, cfg.audio_dim, [C, C]),
        "audio_cross_attn.proj": (Nn, C, [C]),
        "ffn.w1": (N, C, [F]), "ffn.w3": (N, C, [F]), "ffn.w2": (N, F, [C]),
    }
    if cfg.lora_rank > 0:
        for tgt in cfg.lora_targets:
            m, k, outs = shapes[tgt]
            ops.append(_lora(cfg, tgt, m, k, outs))
    return ops


def step_extra_ops(cfg: Config) -> list[Op]:
    """Per step, outside the 48 blocks (all small)."""
    C, N, T = cfg.hidden, cfg.tokens, cfg.latent_frames
    pt, ph, pw = cfg.patch
    patch_in = cfg.in_channels * pt * ph * pw
    return [
        _linear(cfg, "x_embedder", "other", N, patch_in, C, quantized=False),
        _linear(cfg, "y_embedder", "other", cfg.text_tokens, C, C, quantized=False),
        _linear(cfg, "y_embedder.2", "other", cfg.text_tokens, C, C, quantized=False),
        _linear(cfg, "t_embedder", "other", T, 256, cfg.adaln_dim, quantized=False),
        _elementwise(cfg, "final.modulate", "other", N * C, 10),
        _linear(cfg, "final.linear", "other", N, C, patch_in, quantized=False),
    ]


def _sum_by_class(ops, mult=1.0) -> dict[str, dict[str, float]]:
    out = {c: {"macs": 0.0, "matmul_flops": 0.0, "elem_flops": 0.0, "flops": 0.0, "bytes": 0.0} for c in OP_CLASSES}
    for o in ops:
        d = out[o.cls]
        d["macs"] += o.macs * mult
        d["matmul_flops"] += o.matmul_flops * mult
        d["elem_flops"] += o.elem * mult
        d["flops"] += o.flops * mult
        d["bytes"] += o.bytes * mult
    return out


def step_counts(cfg: Config) -> dict:
    """Per-block and per-step totals by op class, with shares of the step's FLOPs."""
    blk = _sum_by_class(block_ops(cfg))
    step = _sum_by_class(block_ops(cfg), cfg.depth * cfg.cfg_passes)
    extra = _sum_by_class(step_extra_ops(cfg), cfg.cfg_passes)
    for c in OP_CLASSES:
        for k in step[c]:
            step[c][k] += extra[c][k]
    tot = {k: sum(step[c][k] for c in OP_CLASSES) for k in ("macs", "matmul_flops", "elem_flops", "flops", "bytes")}
    btot = {k: sum(blk[c][k] for c in OP_CLASSES) for k in ("macs", "matmul_flops", "elem_flops", "flops", "bytes")}
    share = {c: step[c]["flops"] / tot["flops"] for c in OP_CLASSES}
    return {"config": cfg, "tokens": cfg.tokens, "grid": cfg.grid, "per_block": blk, "per_block_total": btot,
            "per_step": step, "per_step_total": tot, "share": share}


def ceiling(measured_seconds_per_step: float, peak_flops: float, cfg: Config | None = None,
            bandwidth: float | None = None) -> dict:
    """Achieved FLOP/s and utilisation at a measured step time, and the best case if every op ran at peak.

    `peak_flops` is a matmul peak (FLOP/s), to be measured by scripts/bench_kernels.py. Utilisation uses all
    FLOPs (matmul + elementwise). `best_case_s` = all FLOPs at peak; with `bandwidth` (bytes/s) a roofline
    per op class is used instead: sum over classes of max(flops/peak, bytes/bandwidth)."""
    cfg = cfg or Config()
    sc = step_counts(cfg)
    tot = sc["per_step_total"]["flops"]
    achieved = tot / measured_seconds_per_step
    best = tot / peak_flops
    roof = None
    if bandwidth:
        roof = sum(max(d["flops"] / peak_flops, d["bytes"] / bandwidth) for d in sc["per_step"].values())
    return {"flops_per_step": tot, "achieved_flops": achieved, "utilisation": achieved / peak_flops,
            "best_case_s": best, "roofline_s": roof, "max_speedup": measured_seconds_per_step / (roof or best)}


def attention_traffic(cfg: Config | None = None, score_bytes: int = 4, bq: int = 32) -> dict:
    """DRAM bytes for one self-attention (all heads, one block): naive vs tiled/fused.

    naive: S = QK^T is written (score_bytes), read back by softmax, P is written in bf16 and read by PV:
           pairs * heads * (2*score_bytes + 2*2) bytes, plus Q, K, V, O.
    fused, compulsory: Q, K, V read once and O written once (scores never leave the chip).
    fused, streamed: a query tile of `bq` rows streams all of K and V once; if nothing is reused from cache
           every tile re-reads K and V from DRAM: heads * ceil(Nq/bq) * 2 * Nk * D * 2 bytes. The real
           number sits between compulsory and streamed (concurrent tiles share K/V blocks in the GPU caches).
    """
    cfg = cfg or Config()
    a, h, d = cfg.act_bytes, cfg.heads, cfg.head_dim
    N, Nn, Nc = cfg.tokens, cfg.noise_tokens, cfg.cond_tokens
    pairs = cfg.attn_pairs
    qkvo = h * 4 * N * d * a
    naive = h * pairs * (2 * score_bytes + 2 * a) + qkvo
    streamed = h * (math.ceil(Nn / bq) * 2 * N * d * a + math.ceil(Nc / bq) * 2 * Nc * d * a) + h * 2 * N * d * a
    flops = 4 * h * pairs * d
    return {"entries_per_head_full": N * N, "pairs_per_head": pairs, "score_bytes": score_bytes, "bq": bq,
            "naive_bytes": naive, "fused_compulsory_bytes": qkvo, "fused_streamed_bytes": streamed,
            "flops": flops, "intensity_naive": flops / naive, "intensity_streamed": flops / streamed,
            "intensity_compulsory": flops / qkvo}


# --------------------------------------------------------------------------------------------- measured
# Measured on the M5 Max, 704x1312, 93 frames (shutter-ref speed/exact/REPORT.md, 2026-11-25).
# Per-block split, eager components (sum 4.62 s); SDPA single call 3.00 s, per head (current default) 2.084 s.
MEASURED_704_BLOCK = {
    "sdpa": 3.00, "sdpa_per_head": 2.084, "ffn": 0.57, "norm_rope": 0.27, "qkv_linear": 0.27,
    "qkv_q8_only": 0.16, "qkv_lora_only": 0.11, "cross_text": 0.21, "cross_audio": 0.19, "attn_proj": 0.06,
    "adaln_modulate": 0.05, "block_compiled": 4.40, "block_compiled_per_head": 3.45,
}
MEASURED_STEP_S = {(704, 1312): 164.0, (480, 832): 45.7}  # current default path (per-head SDPA)
MEASURED_STEP_S_ORIGINAL = {(704, 1312): 209.7}  # before per-head SDPA
# Not helpful (REPORT): fp16 SDPA 3.01 s (= bf16); fp32 SDPA 5.27 s (1.76x slower); q8 matmul in fp16 0.34 s vs
# 0.16 s bf16; query chunking 2.20-2.25 s vs 2.08 s; fused per-head norm+RoPE+SDPA in MLX ops 5% slower compiled.


def measured_class_rates(cfg: Config | None = None, measured: dict | None = None) -> list[dict]:
    """Achieved FLOP/s per measured component (one block at 704x1312)."""
    cfg = cfg or Config()
    m = measured or MEASURED_704_BLOCK
    blk = {o.name: o for o in block_ops(cfg)}
    by = _sum_by_class(block_ops(cfg))
    sdpa = by["self_attn_qk"]["flops"] + by["self_attn_av"]["flops"]
    sdpa_mm = by["self_attn_qk"]["matmul_flops"] + by["self_attn_av"]["matmul_flops"]
    lora = {o.name: o for o in block_ops(cfg) if o.cls == "lora"}
    ffn_lora = sum(lora[k].matmul_flops for k in lora if k.startswith("lora.ffn"))
    qkv_lora = lora["lora.attn.qkv"].matmul_flops if "lora.attn.qkv" in lora else 0.0
    rows = [
        ("self-attn SDPA, one call", sdpa_mm, m["sdpa"]),
        ("self-attn SDPA, per head (default)", sdpa_mm, m["sdpa_per_head"]),
        ("FFN q8 + LoRA", by["ffn"]["matmul_flops"] + ffn_lora, m["ffn"]),
        ("qkv q8 only", blk["attn.qkv"].matmul_flops, m["qkv_q8_only"]),
        ("qkv LoRA only", qkv_lora, m["qkv_lora_only"]),
        ("attn proj q8", blk["attn.proj"].matmul_flops, m["attn_proj"]),
        ("text cross-attn", by["cross_text"]["matmul_flops"], m["cross_text"]),
        ("audio cross-attn", by["cross_audio"]["matmul_flops"], m["cross_audio"]),
    ]
    out = [{"part": p, "flops": f, "seconds": s, "achieved_flops": f / s} for p, f, s in rows]
    nr = blk["attn.qk_rmsnorm"].bytes + blk["attn.rope"].bytes
    out.append({"part": "qk RMSNorm + RoPE (bytes/s, compulsory bf16)", "flops": 0.0, "seconds": m["norm_rope"],
                "achieved_bytes_per_s": nr / m["norm_rope"]})
    return out


def upper_bound(share: float, achieved_util: float) -> float:
    """Best overall step speed-up from making one part run at 100% of peak: Amdahl with that part
    shrinking to achieved_util of its time. share and achieved_util are fractions."""
    return 1.0 / ((1.0 - share) + share * achieved_util)
