"""MiniMax-H3-Image model code for MLX: the diffusion transformer (pruned "AdaLN curve" layout with the six PDD
output heads), the Qwen3-VL-32B text layers H3 reads, and the image VAE's ViT decoder.

Adapted from minimax-h3-mlx (mrbizarro, Apache-2.0, commit 7919020: `dit.py`, `video_vae.py`) and checked against
stable-diffusion.cpp's H3-Image code (MIT: `minimax_h3.hpp`, `minimax_h3_vae.hpp`); see NOTICE.md. Tensor names are
the ones in the ComfyUI / GGUF single-file checkpoints that sd.cpp reads, so a converted build loads 1:1.

What differs from the video port (and why):

* `attn.qkv_proj` rows are three contiguous blocks [q | k | v] in the ComfyUI repack (sd.cpp `split_qkv`), not the
  per-head interleaved rows of MiniMax's original release. The VAE decoder's `to_qkv` stays per-head interleaved.
* no audio stream: the image checkpoint has no `audio_patch_proj` / `audio_out`.
* `final_layer.video_out` holds six heads (6 x 96 outputs); each step blends them by the 6-step PDD schedule
  (`pdd_head_weights`, sd.cpp `pdd_head_weights`).
* the network's output is minus the flow velocity (sd.cpp scales it by -1); `MiniMaxH3ImageDiT.velocity` returns
  the velocity the Euler sampler steps along.

The text encoder is the text half of Qwen3-VL-32B. With text only, all three M-RoPE axes carry the same position, so
its rotary embedding is the plain Qwen3 one (`shutter.families.zimage.model.te_rotary`, theta 5e6) and its layers are
the Qwen3 layers Z-Image already has (`EncoderLayer`). H3 reads the un-normalized output of layer 50 (0-based 49).
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
from mlx import nn

#: dtype activations run in
PRECISION = mx.bfloat16

#: the Qwen3-VL-32B text part, as H3 uses it (layers 50-63, the final norm and lm_head are never run)
TE_CONFIG = dict(vocab_size=151936, hidden_size=5120, num_hidden_layers=50, num_attention_heads=64,
                 num_key_value_heads=8, intermediate_size=25600, head_dim=128, rope_theta=5000000.0,
                 rms_norm_eps=1e-6)

DIT_CONFIG = dict(hidden_size=5376, num_layers=50, token_refiner_num_layers=2, num_attention_heads=56,
                  attention_head_dim=128, ffn_hidden_size=14336, latent_channels=24, patch=2, text_dim=5120,
                  time_embed_dim=8, adaln_curve_grid=1025, rope_inv_freq_len=16, pdd_heads=6, norm_eps=1e-5)

#: the model's six-step schedule (sd.cpp `pdd_sigmas`); `--sigmas minimax-h3-image`
PDD_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)

#: sequences longer than this (pictures above ~2900x2900) run the FFN MLP_CHUNK tokens at a time
MLP_CHUNK_ABOVE, MLP_CHUNK = 8192, 4096

#: modality tags: one AdaLN row per (timestep, tag)
TAG_IMAGE, TAG_TEXT = 0, 1
MODALITIES = 3

#: VAE latent statistics (diffusion latents -> VAE latents: z * std + mean) and pixel statistics
LATENTS_MEAN = (0.858090341091156, -0.960659146308899, 1.066164016723633, -0.509032547473907, -0.272758185863495,
                -1.367541432380676, -0.255325496196747, -0.269075542688370, -0.537684082984924, -0.046409729868174,
                0.665737032890320, 0.196901276707649, -0.546060800552368, -0.403534203767776, -0.236830249428749,
                0.259284526109695, -0.301339447498322, 0.211341992020607, -1.120684862136841, 0.358193337917328,
                -0.042251437902451, 0.260482996702194, 0.228640928864479, 0.705603182315826)
LATENTS_STD = (1.222377419471741, 1.276726365089417, 1.683177471160889, 1.754945516586304, 1.563621640205383,
               2.194143533706665, 0.965313792228699, 1.056988596916199, 0.841948926448822, 0.772995293140411,
               1.895593762397766, 0.946841835975647, 0.799680948257446, 0.449889004230499, 0.719739973545075,
               0.693629324436188, 2.961095094680786, 2.769419908523560, 3.049618482589722, 2.108805418014527,
               3.276226282119751, 3.162735700607300, 2.281681299209595, 2.612784385681153)
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def compute_dtype(layer: nn.Module) -> mx.Dtype:
    """The dtype a layer's input should be cast to: the scales' for a quantized layer (its `weight` is packed
    uint32), the weight's otherwise."""
    scales = getattr(layer, "scales", None)
    return scales.dtype if scales is not None else layer.weight.dtype


def lin(layer: nn.Module, x: mx.array) -> mx.array:
    return layer(x.astype(compute_dtype(layer)))


# ---------------------------------------------------------------------------------------------------------------------
# Diffusion transformer
# ---------------------------------------------------------------------------------------------------------------------

def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate-half rotary on the leading `cos.shape[-1]` channels of every head (sd.cpp rope, interleaved=false).
    x: (B, heads, S, head_dim); cos / sin: (S, rot)."""
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    half = rot // 2
    x0, x1 = xr[..., :half], xr[..., half:]
    c = cos[..., :half].astype(x.dtype)
    s = sin[..., :half].astype(x.dtype)
    out = mx.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)
    return out if xp.shape[-1] == 0 else mx.concatenate([out, xp], axis=-1)


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, eps: float):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, 3 * inner, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)

    def __call__(self, x: mx.array, rope: Optional[tuple[mx.array, mx.array]] = None) -> mx.array:
        b, s, _ = x.shape
        qkv = lin(self.qkv_proj, x).reshape(b, s, 3, self.heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if rope is not None:
            q = apply_rotary(q, *rope)
            k = apply_rotary(k, *rope)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
        o = o.transpose(0, 2, 1, 3).reshape(b, s, self.heads * self.head_dim)
        return lin(self.out_proj, o.astype(x.dtype)).astype(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 2 * ffn, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)
        self.ffn = ffn

    def __call__(self, x: mx.array) -> mx.array:
        h = lin(self.fc1, x)
        g, v = h[..., : self.ffn], h[..., self.ffn:]
        return lin(self.fc2, (nn.silu(g) * v).astype(x.dtype)).astype(x.dtype)


class RefinerBlock(nn.Module):
    """The text token refiner: plain pre-norm attention + FFN, no AdaLN, no rotary."""

    def __init__(self, c: dict):
        super().__init__()
        self.norm1 = nn.RMSNorm(c["hidden_size"], eps=c["norm_eps"])
        self.norm2 = nn.RMSNorm(c["hidden_size"], eps=c["norm_eps"])
        self.attn = Attention(c["hidden_size"], c["num_attention_heads"], c["attention_head_dim"], c["norm_eps"])
        self.mlp = FeedForward(c["hidden_size"], c["ffn_hidden_size"])

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TokenRefiner(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.blocks = [RefinerBlock(c) for _ in range(c["token_refiner_num_layers"])]
        self.final_norm = nn.RMSNorm(c["hidden_size"], eps=c["norm_eps"])

    def __call__(self, x: mx.array) -> mx.array:
        for b in self.blocks:
            x = b(x)
        return self.final_norm(x)


class AdaLN(nn.Module):
    """`adaln_proj`: the 8-wide curve embedding -> per (timestep, modality) modulation rows. No SiLU in the pruned
    layout (sd.cpp: apply_silu only without curves)."""

    def __init__(self, time_dim: int, out: int):
        super().__init__()
        self.linear = nn.Linear(time_dim, out, bias=True)


class TransformerBlock(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        hs = c["hidden_size"]
        self.hidden = hs
        self.norm1 = nn.RMSNorm(hs, eps=c["norm_eps"])
        self.norm2 = nn.RMSNorm(hs, eps=c["norm_eps"])
        self.attn = Attention(hs, c["num_attention_heads"], c["attention_head_dim"], c["norm_eps"])
        self.mlp = FeedForward(hs, c["ffn_hidden_size"])
        self.adaln_proj = AdaLN(c["time_embed_dim"], hs * 6 * MODALITIES)

    def modulation(self, temb: mx.array) -> mx.array:
        """(T, 8) -> (T * 3, 6, hidden) float32 rows: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
        gate_mlp] for row t * 3 + tag."""
        m = lin(self.adaln_proj.linear, temb).astype(mx.float32)
        return m.reshape(-1, 6, self.hidden)

    def __call__(self, x: mx.array, temb: mx.array, rows, rope) -> mx.array:
        """`rows`: one AdaLN row per token, as an (S,) array, or as `Segments` (runs of tokens sharing a row): then
        the modulation is applied run by run with (hidden,) vectors instead of an (S, 6, hidden) table: the same
        numbers, but several GB less at 16 MP."""
        dt = x.dtype
        if isinstance(rows, Segments):
            m = self.modulation(temb).astype(dt)                 # (T * 3, 6, hidden)

            def mod(y, ishift, iscale):
                return mx.concatenate([y[:, a:b] * (1 + m[r, iscale]) + m[r, ishift] for a, b, r in rows], axis=1)

            def gate(x, o, ig):
                return mx.concatenate([x[:, a:b] + o[:, a:b] * m[r, ig] for a, b, r in rows], axis=1)

            x = gate(x, self.attn(mod(self.norm1(x), 0, 1), rope), 2)
            h = mod(self.norm2(x), 3, 4)
            n = h.shape[1]
            if n > MLP_CHUNK_ABOVE:          # 16 MP pictures: the 28672-wide FFN in token chunks (rows are independent)
                o = []
                for a in range(0, n, MLP_CHUNK):
                    o.append(self.mlp(h[:, a:a + MLP_CHUNK]))
                    mx.eval(o[-1])
                out = mx.concatenate(o, axis=1)
            else:
                out = self.mlp(h)
            return gate(x, out, 5)
        m = self.modulation(temb)[rows]                         # (S, 6, hidden), one row per token
        sh1, sc1, g1, sh2, sc2, g2 = (m[:, i].astype(dt) for i in range(6))
        h = self.norm1(x) * (1 + sc1) + sh1
        x = x + self.attn(h, rope) * g1
        h = self.norm2(x) * (1 + sc2) + sh2
        return x + self.mlp(h) * g2


class Segments(tuple):
    """((start, end, AdaLN row), ...) covering a packed sequence in order."""

    @classmethod
    def of(cls, rows: list[int]) -> "Segments":
        out, a = [], 0
        for i in range(1, len(rows) + 1):
            if i == len(rows) or rows[i] != rows[a]:
                out.append((a, i, int(rows[a])))
                a = i
        return cls(out)


class FinalLayer(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        hs = c["hidden_size"]
        self.hidden = hs
        self.norm = nn.RMSNorm(hs, eps=c["norm_eps"])
        self.adaln_proj = AdaLN(c["time_embed_dim"], hs * 2)
        patch_dim = c["latent_channels"] * c["patch"] * c["patch"]
        self.video_out = nn.Linear(hs, patch_dim * c["pdd_heads"], bias=True)


def pdd_head_weights(sigma: float, sigma_next: float) -> list[float]:
    """Blend weights of the six output heads for a step sigma -> sigma_next (sd.cpp `pdd_head_weights`): head 0 is
    the base, heads 1..5 are offsets used over the schedule intervals the step covers."""
    g = PDD_SIGMAS

    def nearest(v: float) -> int:
        return min(range(len(g)), key=lambda i: (abs(g[i] - v), i))

    start = min(nearest(sigma), len(g) - 2)
    stop = max(nearest(sigma_next), start + 1)
    w = [0.0] * 6
    w[0] = 1.0
    for i in range(max(start, 1), stop):
        w[i] = (g[i] - g[i + 1]) / (g[start] - g[stop])
    return w


def spatial_axis(dim: int, sqrt_area: float) -> list[float]:
    """One aspect-normalized rotary axis of the image grid (sd.cpp `spatial_axis`): dim // 2 coordinates."""
    count = dim // 2
    ratio = dim / sqrt_area
    return [(i * (ratio / count) + (1.0 - ratio) * 0.5) * 32.0 for i in range(count)]


class MiniMaxH3ImageDiT(nn.Module):
    """The H3-Image transformer. `build_blocks=False` builds everything but the 50-block stack (the part kept in
    memory while the stack streams from the SSD)."""

    def __init__(self, build_blocks: bool = True, **cfg):
        super().__init__()
        c = {**DIT_CONFIG, **cfg}
        self.c = c
        hs = c["hidden_size"]
        patch_dim = c["latent_channels"] * c["patch"] * c["patch"]
        self.video_patch_proj = nn.Linear(patch_dim, hs, bias=True)
        self.condition_proj = nn.Linear(c["text_dim"], hs, bias=True)
        self.adaln_t_table = mx.zeros((c["adaln_curve_grid"], c["time_embed_dim"]), dtype=mx.float32)
        self.rope_inv_freq = mx.zeros((c["rope_inv_freq_len"],), dtype=mx.float32)
        self.token_refiner = TokenRefiner(c)
        if build_blocks:
            self.blocks = [TransformerBlock(c) for _ in range(c["num_layers"])]
        self.final_layer = FinalLayer(c)

    # ---- inputs
    def time_embedding(self, t: mx.array) -> mx.array:
        """Interpolated AdaLN curve coordinates for model timesteps t = 1 - sigma (float32, (T, 8))."""
        grid = self.c["adaln_curve_grid"]
        pos = mx.clip(t.astype(mx.float32), 0.0, 1.0) * (grid - 1)
        lo = mx.minimum(mx.floor(pos).astype(mx.int32), grid - 2)
        f = (pos - lo.astype(mx.float32))[:, None]
        return self.adaln_t_table[lo] + (self.adaln_t_table[lo + 1] - self.adaln_t_table[lo]) * f

    def rope(self, positions: mx.array) -> tuple[mx.array, mx.array]:
        """positions (S, 3) float32 -> cos, sin (S, 96): per axis 16 angles, the three axes concatenated, doubled."""
        a = positions[:, :, None] * self.rope_inv_freq[None, None, :]
        a = a.reshape(positions.shape[0], -1)
        a = mx.concatenate([a, a], axis=-1)
        return mx.cos(a), mx.sin(a)

    def refine_text(self, text: mx.array) -> mx.array:
        """(1, L, 5120) encoder states -> (1, L, 5376) refined text rows (independent of the step)."""
        return self.token_refiner(lin(self.condition_proj, text).astype(mx.float32))

    def patchify(self, latents: mx.array) -> mx.array:
        """(24, H, W) latents -> (H/2 * W/2, 96) rows, channel-major patches (C, ph, pw)."""
        c, h, w = latents.shape
        p = self.c["patch"]
        x = latents.reshape(c, h // p, p, w // p, p).transpose(1, 3, 0, 2, 4)
        return x.reshape((h // p) * (w // p), c * p * p)

    def unpatchify(self, rows: mx.array, h: int, w: int) -> mx.array:
        p, c = self.c["patch"], self.c["latent_channels"]
        x = rows.reshape(h // p, w // p, c, p, p).transpose(2, 0, 3, 1, 4)
        return x.reshape(c, h, w)

    @staticmethod
    def layout(text_len: int, lat_h: int, lat_w: int, refs: tuple = ()) -> tuple[mx.array, int]:
        """Positions for [text | reference pictures | image] (diffusers `build_image_packed_sequence`, sd.cpp
        `build_layout`): text row i at (i, 0, 0); each reference at the next integer time from text_len on, over its own
        aspect-normalized grid; the generated image one time step after the last reference. `refs` = latent (h, w) of
        each reference. Returns ((S, 3) float32 positions, number of generated-image rows)."""
        import numpy as np
        pos = [(float(i), 0.0, 0.0) for i in range(text_len)]
        t = float(text_len)
        for rh, rw in refs:
            sq = math.sqrt(rh * rw)
            pos += [(t, hv, wv) for hv in spatial_axis(rh, sq) for wv in spatial_axis(rw, sq)]
            t += 1.0
        sq = math.sqrt(lat_h * lat_w)
        ha, wa = spatial_axis(lat_h, sq), spatial_axis(lat_w, sq)
        pos += [(t, hv, wv) for hv in ha for wv in wa]
        return mx.array(np.array(pos, dtype=np.float32)), len(ha) * len(wa)

    # ---- one forward
    def velocity(self, latents: mx.array, text_rows: mx.array, sigma: float, sigma_next: float,
                 blocks=None, refs: Optional[list] = None, text_tags: Optional[list] = None,
                 ref_t: float = 0.999) -> mx.array:
        """d(latents)/d(sigma) for the Euler step sigma -> sigma_next. latents (24, H, W) float32; text_rows the
        refined text (1, L, hidden). `blocks` runs the 50-block stack: None = this model's own blocks, else a
        callable (x, temb, rows, rope) -> x (the SSD streamed stack). `refs`: noised reference latents (24, h, w)
        held at model timestep `ref_t`; `text_tags`: the modality tag of every text row (vision-block rows are
        image rows; default all text)."""
        c = self.c
        _, lh, lw = latents.shape
        L = text_rows.shape[1]
        refs = refs or []
        pos, n_img = self.layout(L, lh, lw, tuple((r.shape[1], r.shape[2]) for r in refs))
        rope = self.rope(pos)
        parts = [text_rows.astype(mx.float32)]
        n_ref = 0
        for r in refs:
            pr = self.patchify(r)
            n_ref += pr.shape[0]
            parts.append(lin(self.video_patch_proj, pr).astype(mx.float32)[None])
        img = lin(self.video_patch_proj, self.patchify(latents)).astype(mx.float32)
        # the residual stream stays float32 (sd.cpp computes in float32); matmuls run at the weights' precision
        x = mx.concatenate(parts + [img[None]], axis=1)
        t = 1.0 - min(max(sigma, 1e-6), 1.0)
        tags = list(text_tags) if text_tags is not None else [TAG_TEXT] * L
        if n_ref:
            # timestep row 0: the step's t (text and generated rows), row 1: the references' t
            temb = self.time_embedding(mx.array([t, ref_t], dtype=mx.float32))
            rows = Segments.of(tags + [MODALITIES + TAG_IMAGE] * n_ref + [TAG_IMAGE] * n_img)
        else:
            temb = self.time_embedding(mx.array([t], dtype=mx.float32))
            rows = Segments.of(tags + [TAG_IMAGE] * n_img)           # timestep row 0: row = tag
        if blocks is None:
            for b in self.blocks:
                x = b(x, temb, rows, rope)
        else:
            x = blocks(x, temb, rows, rope)
        # final layer on the generated rows: x + x * scale + shift
        fm = lin(self.final_layer.adaln_proj.linear, temb[:1]).astype(mx.float32).reshape(2, c["hidden_size"])
        h = self.final_layer.norm(x[0, L + n_ref:]).astype(mx.float32)
        h = h + h * fm[1] + fm[0]
        out = lin(self.final_layer.video_out, h)                    # (n_img, 6 * 96)
        dim = out.shape[-1] // c["pdd_heads"]
        w = pdd_head_weights(sigma, sigma_next)
        pred = out[:, :dim] * w[0]
        for i in range(1, c["pdd_heads"]):
            if w[i] != 0.0:
                pred = pred + out[:, i * dim:(i + 1) * dim] * w[i]
        return -self.unpatchify(pred.astype(mx.float32), lh, lw)


# ---------------------------------------------------------------------------------------------------------------------
# Image VAE decoder (ViT)
# ---------------------------------------------------------------------------------------------------------------------

class VAEAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 64):
        super().__init__()
        self.heads, self.head_dim = dim // head_dim, head_dim
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)

    @staticmethod
    def _rms(x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        return (x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + 1e-5)).astype(x.dtype)

    def __call__(self, x: mx.array, rope) -> mx.array:
        b, n, _ = x.shape
        qkv = lin(self.to_qkv, x).reshape(b, n, self.heads, 3, self.head_dim)   # per-head [q, k, v]
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
        q = self._rms(q).transpose(0, 2, 1, 3)
        k = self._rms(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = apply_rotary(q, *rope)
        k = apply_rotary(k, *rope)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
        return lin(self.to_out, o.transpose(0, 2, 1, 3).reshape(b, n, -1))


class VAEFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * dim * mult, bias=True)
        self.w2 = nn.Linear(dim * mult, dim, bias=True)
        self.inner = dim * mult

    def __call__(self, x: mx.array) -> mx.array:
        h = lin(self.w1, x)
        return lin(self.w2, nn.silu(h[..., : self.inner]) * h[..., self.inner:])


class VAEBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-5)
        self.norm2 = nn.RMSNorm(dim, eps=1e-5)
        self.attn = VAEAttention(dim)
        self.ff = VAEFeedForward(dim)
        self.scale1 = mx.zeros((dim,))
        self.scale2 = mx.zeros((dim,))

    def __call__(self, x: mx.array, rope) -> mx.array:
        h = self.norm1(x.astype(mx.float32)).astype(x.dtype)
        x = x + (self.attn(h, rope) * self.scale1).astype(x.dtype)
        h = self.norm2(x.astype(mx.float32)).astype(x.dtype)
        return x + (self.ff(h) * self.scale2).astype(x.dtype)


class VAEDecoder(nn.Module):
    """`decoder.*`: one token per latent pixel, 4 register tokens + 1 zero token, 24 blocks, LayerNorm, 16x16x3
    patches. 2D rotary over (h, w) on 32 of each head's 64 channels, theta 100, coordinates (2y - 1) * 2 pi."""

    def __init__(self, dim: int = 1024, layers: int = 24, latent_channels: int = 24):
        super().__init__()
        self.dim = dim
        self.x_embedder = nn.Linear(latent_channels, dim, bias=True)
        self.register_tokens = mx.zeros((1, 4, dim))
        self.mask_token = mx.zeros((1, 1, dim))        # in the file, unused for single images
        self.transformer_blocks = [VAEBlock(dim) for _ in range(layers)]
        self.norm_out = nn.LayerNorm(dim, eps=1e-5, affine=True)
        self.proj_out = nn.Linear(dim, 3 * 16 * 16, bias=True)

    @staticmethod
    def rope(h: int, w: int, extra: int) -> tuple[mx.array, mx.array]:
        inv = 1.0 / (100.0 ** (mx.arange(8, dtype=mx.float32) / 8.0))          # 8 frequencies per axis
        ys = (2.0 * ((mx.arange(h, dtype=mx.float32) + 0.5) / h) - 1.0) * (2 * math.pi)
        xs = (2.0 * ((mx.arange(w, dtype=mx.float32) + 0.5) / w) - 1.0) * (2 * math.pi)
        py = mx.broadcast_to(ys[:, None], (h, w)).reshape(-1)
        px = mx.broadcast_to(xs[None, :], (h, w)).reshape(-1)
        a = mx.concatenate([py[:, None] * inv[None], px[:, None] * inv[None]], axis=-1)     # (hw, 16)
        a = mx.concatenate([a, mx.zeros((extra, 16), dtype=mx.float32)], axis=0)
        a = mx.concatenate([a, a], axis=-1)                                                  # rotate-half over 32
        return mx.cos(a), mx.sin(a)

    def __call__(self, z: mx.array, dtype=mx.float32) -> mx.array:
        """z (H, W, 24) VAE latents (after post_quant_conv) -> pixels (H*16, W*16, 3) in [0, 1]."""
        h, w, c = z.shape
        tok = lin(self.x_embedder, z.reshape(1, h * w, c)).astype(dtype)
        n = tok.shape[1]
        tok = mx.concatenate([tok, self.register_tokens.astype(dtype), mx.zeros((1, 1, self.dim), dtype=dtype)],
                             axis=1)
        rope = self.rope(h, w, 5)
        for blk in self.transformer_blocks:
            tok = blk(tok, rope)
            mx.eval(tok)
        tok = self.norm_out(tok.astype(mx.float32))
        out = lin(self.proj_out, tok)[0, :n].astype(mx.float32)          # (hw, 3*16*16), (C, p, p) order
        out = out.reshape(h, w, 3, 16, 16).transpose(0, 3, 1, 4, 2).reshape(h * 16, w * 16, 3)
        mean = mx.array(PIXEL_MEAN, dtype=mx.float32)
        std = mx.array(PIXEL_STD, dtype=mx.float32)
        return mx.clip(out * std + mean, 0.0, 1.0)


class VAE(nn.Module):
    """The decode half of the image VAE (`post_quant_conv` 1x1 + `decoder`)."""

    def __init__(self, layers: int = 24, dim: int = 1024):
        super().__init__()
        self.post_quant_conv = nn.Linear(24, 24, bias=True)    # a 1x1 conv, stored [24, 24, 1, 1] -> [24, 24]
        self.decoder = VAEDecoder(dim=dim, layers=layers)

    #: the decoder was trained on 256 px tiles: rotary positions are normalized over a tile's own 16x16 latent grid,
    #: and diffusers decodes in such tiles with >= 64 px overlap blended linearly (vae_tile_size 256,
    #: vae_tile_overlap_min 64). Decoding a bigger grid at once leaves 16 px seams.
    TILE, MIN_OVERLAP = 16, 4

    @staticmethod
    def tile_starts(n: int, tile: int = 16, min_overlap: int = 4) -> tuple[list[int], int]:
        if n <= tile:
            return [0], n
        k = math.ceil((n - min_overlap) / (tile - min_overlap))
        return [round(i * (n - tile) / (k - 1)) for i in range(k)], tile

    @staticmethod
    def _ramp(n: int, lo: int, hi: int):
        import numpy as np
        w = np.ones(n, np.float32)
        if lo > 0:
            w[:lo] = (np.arange(lo) + 0.5) / lo
        if hi > 0:
            w[n - hi:] = np.minimum(w[n - hi:], (np.arange(hi)[::-1] + 0.5) / hi)
        return w

    def decode(self, latents: mx.array, check=None) -> mx.array:
        """Diffusion latents (24, H, W) -> float pixels (H*16, W*16, 3) in [0, 1], decoded in 16x16-latent tiles
        blended over their overlaps (one tile at a time in memory)."""
        import numpy as np
        mean = mx.array(LATENTS_MEAN, dtype=mx.float32)[:, None, None]
        std = mx.array(LATENTS_STD, dtype=mx.float32)[:, None, None]
        z = (latents.astype(mx.float32) * std + mean).transpose(1, 2, 0)          # (H, W, 24)
        z = lin(self.post_quant_conv, z).astype(mx.float32)
        mx.eval(z)
        H, W, _ = z.shape
        ys, th = self.tile_starts(H, self.TILE, self.MIN_OVERLAP)
        xs, tw = self.tile_starts(W, self.TILE, self.MIN_OVERLAP)
        out = np.zeros((H * 16, W * 16, 3), np.float32)
        wsum = np.zeros((H * 16, W * 16, 1), np.float32)
        for i, y in enumerate(ys):
            lo_y = (ys[i - 1] + th - y) * 16 if i > 0 else 0
            hi_y = (y + th - ys[i + 1]) * 16 if i + 1 < len(ys) else 0
            for j, x in enumerate(xs):
                if check is not None:
                    check()
                lo_x = (xs[j - 1] + tw - x) * 16 if j > 0 else 0
                hi_x = (x + tw - xs[j + 1]) * 16 if j + 1 < len(xs) else 0
                img = np.array(self.decoder(z[y:y + th, x:x + tw]))
                wt = self._ramp(th * 16, lo_y, hi_y)[:, None, None] * self._ramp(tw * 16, lo_x, hi_x)[None, :, None]
                out[y * 16:(y + th) * 16, x * 16:(x + tw) * 16] += img * wt
                wsum[y * 16:(y + th) * 16, x * 16:(x + tw) * 16] += wt
        return mx.array(np.clip(out / wsum, 0.0, 1.0))
