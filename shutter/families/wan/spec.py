"""What a Wan checkpoint is: its config read from tensor shapes, its variant, defaults, and memory formulas.

Shapes are the truth (a fine-tune's config.json is often missing or copied from elsewhere): ``patch_embedding``
gives dim and in_dim, the head gives out_dim, the blocks give depth and FFN width. Every Wan size uses 128-wide
attention heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from shutter.core.parts import PartSize, StageNeeds

GB = 1024**3
HEAD_DIM = 128

# defaults per known variant (from the Wan reference configs, as mlx-video's presets)
VARIANTS: dict[str, dict[str, Any]] = {
    "wan2.1-t2v-1.3b": dict(label="Wan 2.1 T2V 1.3B", task="t2v", version="2.1", dual=False, steps=50, shift=5.0,
                            guidance=5.0, fps=16, frames=81, size=(480, 832), max_area=0, vae="wan2.1"),
    "wan2.1-t2v-14b": dict(label="Wan 2.1 T2V 14B", task="t2v", version="2.1", dual=False, steps=50, shift=5.0,
                           guidance=5.0, fps=16, frames=81, size=(720, 1280), max_area=0, vae="wan2.1"),
    "wan2.2-t2v-a14b": dict(label="Wan 2.2 T2V A14B", task="t2v", version="2.2", dual=True, steps=40, shift=12.0,
                            guidance=(3.0, 4.0), boundary=0.875, fps=16, frames=81, size=(720, 1280), max_area=0,
                            vae="wan2.1"),
    "wan2.2-i2v-a14b": dict(label="Wan 2.2 I2V A14B", task="i2v", version="2.2", dual=True, steps=40, shift=5.0,
                            guidance=(3.5, 3.5), boundary=0.900, fps=16, frames=81, size=(720, 1280),
                            max_area=704 * 1280, vae="wan2.1"),
    "wan2.2-ti2v-5b": dict(label="Wan 2.2 TI2V 5B", task="ti2v", version="2.2", dual=False, steps=50, shift=5.0,
                           guidance=5.0, fps=24, frames=121, size=(704, 1280), max_area=704 * 1280, vae="wan2.2"),
}

NEG_PROMPT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
              "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
              "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
              "杂乱的背景，三条腿，背景人很多，倒着走")


@dataclass
class WanShape:
    dim: int
    in_dim: int
    out_dim: int
    ffn_dim: int
    num_layers: int
    text_dim: int = 4096
    freq_dim: int = 256
    cross_attn_norm: bool = True
    qk_norm: bool = True
    dual: bool = False

    @property
    def num_heads(self) -> int:
        return self.dim // HEAD_DIM

    def params_per_block(self) -> int:
        d, f = self.dim, self.ffn_dim
        lin = 8 * (d * d + d) + (d * f + f) + (f * d + d)          # 8 attention linears + FFN
        norms = 4 * d + (2 * d if self.cross_attn_norm else 0)       # q/k norms x2, norm3 affine
        return lin + norms + 6 * d                                   # + modulation

    def quantized_params_per_block(self) -> int:
        d, f = self.dim, self.ffn_dim
        return 8 * d * d + 2 * d * f

    def block_bytes(self, quant_bits: Optional[int] = 8, group: int = 64) -> int:
        """One block in memory: quantized linears + bf16 rest (+ fp32 modulation)."""
        p = self.params_per_block()
        if not quant_bits:
            return p * 2 + 6 * self.dim * 2
        q = self.quantized_params_per_block()
        packed = q * quant_bits // 8
        scales = (q // group) * 2 * 2                                 # scales + biases, bf16
        return packed + scales + (p - q) * 2 + 6 * self.dim * 2

    def rest_bytes(self) -> int:
        d = self.dim
        patch = self.in_dim * 4 * d + d
        text = self.text_dim * d + d + d * d + d
        time = self.freq_dim * d + d + d * d + d + d * 6 * d + 6 * d
        head = d * self.out_dim * 4 + self.out_dim * 4 + 2 * d * 2
        return (patch + text + time + head) * 2

    def dit_bytes(self, quant_bits: Optional[int] = 8) -> int:
        return self.rest_bytes() + self.num_layers * self.block_bytes(quant_bits)

    def to_config(self, variant: str) -> dict[str, Any]:
        v = VARIANTS.get(variant, {})
        vae = v.get("vae") or ("wan2.2" if self.out_dim == 48 else "wan2.1")
        return {
            "model_type": v.get("task", "ti2v" if self.in_dim == 48 else ("i2v" if self.in_dim == 36 else "t2v")),
            "model_version": v.get("version", "2.2" if vae == "wan2.2" or self.dual else "2.1"),
            "dim": self.dim, "ffn_dim": self.ffn_dim, "in_dim": self.in_dim, "out_dim": self.out_dim,
            "num_heads": self.num_heads, "num_layers": self.num_layers, "text_dim": self.text_dim,
            "freq_dim": self.freq_dim, "cross_attn_norm": self.cross_attn_norm, "qk_norm": self.qk_norm,
            "vae_z_dim": 48 if vae == "wan2.2" else 16, "vae_stride": [4, 16, 16] if vae == "wan2.2" else [4, 8, 8],
            "dual_model": self.dual, "boundary": v.get("boundary", 0.0),
            "sample_shift": v.get("shift", 5.0), "sample_steps": v.get("steps", 50),
            "sample_guide_scale": list(v["guidance"]) if isinstance(v.get("guidance"), tuple) else v.get("guidance", 5.0),
            "default_size": [704, 1280] if v.get("task") == "ti2v" else [480, 832],
            "sample_fps": v.get("fps", 16), "frame_num": v.get("frames", 81), "max_area": v.get("max_area", 0),
        }


def classify(shape: WanShape) -> str:
    """Variant id for a shape, or "" for a Wan-architecture model of a size we have no preset for."""
    d, i, o = shape.dim, shape.in_dim, shape.out_dim
    if d == 1536 and i == 16 and o == 16 and not shape.dual:
        return "wan2.1-t2v-1.3b"
    if d == 3072 and i == 48 and o == 48 and not shape.dual:
        return "wan2.2-ti2v-5b"
    if d == 5120 and i == 16 and o == 16:
        return "wan2.2-t2v-a14b" if shape.dual else "wan2.1-t2v-14b"
    if d == 5120 and i == 36 and o == 16 and shape.dual:
        return "wan2.2-i2v-a14b"
    return ""


# ----------------------------------------------------------------------------- memory


def latent_geometry(cfg: dict[str, Any], height: int, width: int, frames: int) -> dict[str, int]:
    st = cfg["vae_stride"]
    t = (frames - 1) // st[0] + 1
    h, w = height // st[1], width // st[2]
    tokens = t * (h // 2) * (w // 2)
    return {"t": t, "h": h, "w": w, "tokens": tokens}


def stage_needs(cfg: dict[str, Any], height: int, width: int, frames: int, *, cfg_batch: int = 2,
                cross_kv_cached: bool = True, tiled_decode: bool = False) -> StageNeeds:
    """Working memory per stage. Denoise: the float32 residual stream plus one block's temporaries (q/k/v in bf16,
    two float32 RoPE copies, the FFN hidden), with batch 2 when guidance runs a negative pass."""
    g = latent_geometry(cfg, height, width, frames)
    d, f, L, B = cfg["dim"], cfg["ffn_dim"], g["tokens"], cfg_batch
    per_block = B * L * (4 * d + 3 * 2 * d + 2 * 4 * d + 2 * f + 2 * f // 2)
    resid = B * L * d * 4 * 2                       # x and the block output
    cross = cfg["num_layers"] * 2 * B * 512 * d * 2 if cross_kv_cached else 0
    rope = 2 * L * d * 4
    latents = cfg["vae_z_dim"] * g["t"] * g["h"] * g["w"] * 4 * 4    # latents, noise, prediction, scheduler state
    text = B * 512 * cfg["text_dim"] * 4
    # decode: the Wan VAEs decode one latent frame at a time with a cache; the largest activations are at full
    # resolution (96 channels x 4 frames, float32): the layer's input and output, the norm, the two-frame caches of
    # every full-resolution conv, and one conv step's workspace. Measured with the 2.1 VAE on an M5 (MLX 0.32.5):
    # 4.2 GB at 320x512, 9.7 GB at 480x832 (about 27 KB a pixel, 18 such copies). The 2.2 VAE (48 channels) is
    # not measured yet and keeps the older 6 copies. Tiling caps the size at 512 px.
    hh, ww = (min(height, 512), min(width, 512)) if tiled_decode else (height, width)
    frames_out = 1 + 4 * (g["t"] - 1)
    if cfg["vae_z_dim"] == 16:
        decode = int(hh * ww * 96 * 4 * 4 * 18) + frames_out * height * width * 3 * 4
    else:
        # Wan 2.2 VAE, streamed one frame at a time (vae22.StreamDecoder): the conv caches (two frames per causal
        # conv, ~6.6 KB a pixel) plus one frame's activations and conv workspace; it does not grow with the frame
        # count. Measured on an M5 (MLX 0.32.5), without the 2.07 GB of weights: 2.26 GB at 320x512, 4.21 GB at
        # 480x832 -> 0.9 GB + 8.3 KB a pixel. The finished frames are kept as uint8.
        decode = int(0.9 * GB + hh * ww * 8.3 * 1024) + frames_out * height * width * 3
    encode = 512 * 4096 * 4 * 6
    return StageNeeds(encode=encode, image=decode // 2, denoise=per_block + resid + cross + rope,
                      decode=decode, carried=latents + text)


def part_sizes(cfg: dict[str, Any], quant_bits: Optional[int] = 8, image_encoder: bool = False) -> list[PartSize]:
    shape = WanShape(dim=cfg["dim"], in_dim=cfg["in_dim"], out_dim=cfg["out_dim"], ffn_dim=cfg["ffn_dim"],
                     num_layers=cfg["num_layers"], text_dim=cfg.get("text_dim", 4096),
                     cross_attn_norm=cfg.get("cross_attn_norm", True), dual=cfg.get("dual_model", False))
    dit = PartSize("dit", "denoiser", shape.dit_bytes(quant_bits), shape.num_layers, shape.block_bytes(quant_bits))
    parts = [dit]
    if shape.dual:
        parts = [PartSize("dit_high", "denoiser", dit.total, dit.n_blocks, dit.block),
                 PartSize("dit_low", "denoiser", dit.total, dit.n_blocks, dit.block)]
    # umT5-XXL: 256384 x 4096 embedding + 24 blocks of ~0.39 G params, run in float32 one block at a time
    embed = 256384 * 4096 * 2
    t5_block = (4 * 4096 * 4096 + 3 * 4096 * 10240 + 2 * 4096 + 32 * 64) * 4
    parts.append(PartSize("text_encoder", "text_encoder", embed + 24 * t5_block, 24, t5_block, embed=embed))
    # float32 weights: the Wan 2.1 decoder; the Wan 2.2 decoder (2.07 GB) and its encoder (0.56 GB), as built
    vae_bytes = int((0.51 if cfg["vae_z_dim"] == 16 else 2.07) * GB)
    enc_bytes = int((0.51 if cfg["vae_z_dim"] == 16 else 0.56) * GB)
    parts.append(PartSize("vae", "decoder", vae_bytes))
    if image_encoder:
        parts.append(PartSize("vae_encoder", "image_encoder", enc_bytes))
    return parts
