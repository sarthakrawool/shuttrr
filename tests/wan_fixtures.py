"""Tiny random Wan checkpoints in every on-disk layout, for tests (no download, CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TINY = dict(dim=256, ffn_dim=512, num_heads=2, num_layers=3, in_dim=16, out_dim=16, text_dim=4096)


def _rng(seed=0):
    return np.random.default_rng(seed)


def official_dit_weights(cfg=TINY, seed=0, extra=None) -> dict[str, np.ndarray]:
    """Official Wan DiT names and shapes (patch embedding as a Conv3d), random values, float32."""
    r = _rng(seed)
    d, f, i, o, t = cfg["dim"], cfg["ffn_dim"], cfg["in_dim"], cfg["out_dim"], cfg["text_dim"]

    def lin(n_out, n_in, scale=0.02):
        return (r.standard_normal((n_out, n_in)) * scale).astype(np.float32), \
               (r.standard_normal((n_out,)) * scale).astype(np.float32)

    w: dict[str, np.ndarray] = {}
    w["patch_embedding.weight"] = (r.standard_normal((d, i, 1, 2, 2)) * 0.05).astype(np.float32)
    w["patch_embedding.bias"] = np.zeros((d,), np.float32)
    for name, (a, b) in (("text_embedding.0", (d, t)), ("text_embedding.2", (d, d)), ("time_embedding.0", (d, 256)),
                         ("time_embedding.2", (d, d)), ("time_projection.1", (6 * d, d))):
        w[name + ".weight"], w[name + ".bias"] = lin(a, b)
    for k in range(cfg["num_layers"]):
        p = f"blocks.{k}."
        for att in ("self_attn", "cross_attn"):
            for q in "qkvo":
                w[f"{p}{att}.{q}.weight"], w[f"{p}{att}.{q}.bias"] = lin(d, d)
            w[f"{p}{att}.norm_q.weight"] = (1 + 0.1 * r.standard_normal((d,))).astype(np.float32)
            w[f"{p}{att}.norm_k.weight"] = (1 + 0.1 * r.standard_normal((d,))).astype(np.float32)
        w[p + "norm3.weight"] = (1 + 0.1 * r.standard_normal((d,))).astype(np.float32)
        w[p + "norm3.bias"] = (0.1 * r.standard_normal((d,))).astype(np.float32)
        w[p + "ffn.0.weight"], w[p + "ffn.0.bias"] = lin(f, d)
        w[p + "ffn.2.weight"], w[p + "ffn.2.bias"] = lin(d, f)
        w[p + "modulation"] = (r.standard_normal((1, 6, d)) * d ** -0.5).astype(np.float32)
    w["head.head.weight"], w["head.head.bias"] = lin(o * 4, d)
    w["head.modulation"] = (r.standard_normal((1, 2, d)) * d ** -0.5).astype(np.float32)
    if extra:
        w.update(extra)
    return w


def to_comfy(w):
    return {"model.diffusion_model." + k: v for k, v in w.items()}


def to_diffusers(w):
    """Inverse of keys.diffusers_to_official, for tests."""
    import re
    out = {}
    top = {"text_embedding.0.": "condition_embedder.text_embedder.linear_1.",
           "text_embedding.2.": "condition_embedder.text_embedder.linear_2.",
           "time_embedding.0.": "condition_embedder.time_embedder.linear_1.",
           "time_embedding.2.": "condition_embedder.time_embedder.linear_2.",
           "time_projection.1.": "condition_embedder.time_proj.", "head.head.": "proj_out."}
    blk = [(".self_attn.q.", ".attn1.to_q."), (".self_attn.k.", ".attn1.to_k."), (".self_attn.v.", ".attn1.to_v."),
           (".self_attn.o.", ".attn1.to_out.0."), (".self_attn.norm_q.", ".attn1.norm_q."),
           (".self_attn.norm_k.", ".attn1.norm_k."), (".cross_attn.q.", ".attn2.to_q."),
           (".cross_attn.k.", ".attn2.to_k."), (".cross_attn.v.", ".attn2.to_v."),
           (".cross_attn.o.", ".attn2.to_out.0."), (".cross_attn.norm_q.", ".attn2.norm_q."),
           (".cross_attn.norm_k.", ".attn2.norm_k."), (".norm3.", ".norm2."), (".ffn.0.", ".ffn.net.0.proj."),
           (".ffn.2.", ".ffn.net.2.")]
    for k, v in w.items():
        if k == "head.modulation":
            out["scale_shift_table"] = v
            continue
        m = re.match(r"^blocks\.(\d+)\.modulation$", k)
        if m:
            out[f"blocks.{m.group(1)}.scale_shift_table"] = v
            continue
        nk = k
        for a, b in top.items():
            if k.startswith(a):
                nk = b + k[len(a):]
        for a, b in blk:
            if a in nk:
                nk = nk.replace(a, b, 1)
                break
        out[nk] = v
    return out


def t5_weights(n_blocks=2, dim=64, ffn=128, heads=4, vocab=300, seed=1) -> dict[str, np.ndarray]:
    """Tiny umT5 in Wan names (mlx-video's T5Encoder, with gate.0 as in the official checkpoint)."""
    r = _rng(seed)
    w = {"token_embedding.weight": r.standard_normal((vocab, dim)).astype(np.float32),
         "norm.weight": (1 + 0.1 * r.standard_normal((dim,))).astype(np.float32)}
    for i in range(n_blocks):
        p = f"blocks.{i}."
        for q in "qkvo":
            w[f"{p}attn.{q}.weight"] = (r.standard_normal((dim, dim)) * 0.05).astype(np.float32)
        w[p + "norm1.weight"] = np.ones((dim,), np.float32)
        w[p + "norm2.weight"] = np.ones((dim,), np.float32)
        w[p + "ffn.gate.0.weight"] = (r.standard_normal((ffn, dim)) * 0.05).astype(np.float32)
        w[p + "ffn.fc1.weight"] = (r.standard_normal((ffn, dim)) * 0.05).astype(np.float32)
        w[p + "ffn.fc2.weight"] = (r.standard_normal((dim, ffn)) * 0.05).astype(np.float32)
        w[p + "pos_embedding.embedding.weight"] = (r.standard_normal((32, heads)) * 0.1).astype(np.float32)
    return w


def save_st(path: Path, w: dict[str, np.ndarray], bf16: bool = False) -> None:
    import mlx.core as mx
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {k: mx.array(v) for k, v in w.items()}
    if bf16:
        arrs = {k: v.astype(mx.bfloat16) for k, v in arrs.items()}
    mx.save_safetensors(str(path), arrs)


def lora_weights(cfg=TINY, rank=4, seed=3, naming="official", blocks=None) -> dict[str, np.ndarray]:
    r = _rng(seed)
    d = cfg["dim"]
    w = {}
    for k in (blocks if blocks is not None else range(cfg["num_layers"])):
        for mod in ("self_attn.q", "cross_attn.v", "ffn.0"):
            n_out = cfg["ffn_dim"] if mod == "ffn.0" else d
            down = (r.standard_normal((rank, d)) * 0.1).astype(np.float32)
            up = (r.standard_normal((n_out, rank)) * 0.1).astype(np.float32)
            if naming == "kohya":
                base = f"lora_unet_blocks_{k}_{mod.replace('.', '_')}"
            else:
                base = f"diffusion_model.blocks.{k}.{mod}"
            w[base + ".lora_down.weight"] = down
            w[base + ".lora_up.weight"] = up
            w[base + ".alpha"] = np.array(rank / 2, np.float32)
    return w


def write_tokenizer_stub(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tokenizer_config.json").write_text(json.dumps({"stub": True}))
