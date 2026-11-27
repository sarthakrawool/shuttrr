"""Z-Image Turbo model code for MLX: the Qwen3 prompt encoder, the single-stream DiT and the VAE decoder.

Vendored from mflux 0.24.0 (`mflux/models/z_image/model/...`, MIT licence, see NOTICE.md), trimmed to what text to
image needs: no image encoder, no ControlNet, no LoRA wrappers, no float16 compute option. The arithmetic is the same
op for op, so an mflux-saved build gives the same image here as in mflux.

What Shuttrr changes around it (not in it):

* `TextEncoder` only builds the layers the image model reads. Z-Image uses the second to last hidden state of the
  36-layer Qwen3 encoder, so layer 36 and the final norm are never run (`ENCODER_LAYERS_USED`); the output is the same.
* The layers can be run one at a time (shutter.families.zimage.pipeline), so the prompt encoder never needs more than
  its token embedding plus one layer in memory.
* The DiT's main stack (`layers`) can be streamed from the SSD block by block (shutter.core.stream).
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

#: dtype the weights and activations run in (mflux ModelConfig.precision)
PRECISION = mx.bfloat16

#: Qwen3-4B as Z-Image uses it
TE_CONFIG = dict(vocab_size=151936, hidden_size=2560, num_hidden_layers=36, num_attention_heads=32,
                 num_key_value_heads=8, intermediate_size=9728, head_dim=128, rope_theta=1000000.0,
                 rms_norm_eps=1e-6)
#: hidden_states[-2] of the 36 layers = the output of layer 35
ENCODER_LAYERS_USED = 35


# ---------------------------------------------------------------------------------------------------------------------
# Prompt encoder (Qwen3)
# ---------------------------------------------------------------------------------------------------------------------

class TEAttention(nn.Module):
    def __init__(self, hidden_size: int = 2560, num_attention_heads: int = 32, num_key_value_heads: int = 8,
                 head_dim: int = 128):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None,
                 position_embeddings: Optional[tuple[mx.array, mx.array]] = None) -> mx.array:
        b, n, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(b, n, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(b, n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(b, n, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if position_embeddings is not None:
            q, k = _apply_rotary_pos_emb(q, k, *position_embeddings)
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=2)
            v = mx.repeat(v, self.num_kv_groups, axis=2)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))
        out = scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attention_mask)
        out = mx.transpose(out, axes=(0, 2, 1, 3)).reshape(b, n, -1)
        return self.o_proj(out)


def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array,
                          unsqueeze_dim: int = 2) -> tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class TEMLP(nn.Module):
    def __init__(self, hidden_size: int = 2560, intermediate_size: int = 9728):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 2560, num_attention_heads: int = 32, num_key_value_heads: int = 8,
                 intermediate_size: int = 9728, head_dim: int = 128, rms_norm_eps: float = 1e-6, **_):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = TEAttention(hidden_size, num_attention_heads, num_key_value_heads, head_dim)
        self.mlp = TEMLP(hidden_size, intermediate_size)

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None,
                 position_embeddings: Optional[tuple[mx.array, mx.array]] = None) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), attention_mask, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


def te_rotary(head_dim: int, base: float, seq_len: int, dtype) -> tuple[mx.array, mx.array]:
    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = mx.outer(mx.arange(seq_len, dtype=mx.float32), inv_freq)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb)[None, :, :].astype(dtype), mx.sin(emb)[None, :, :].astype(dtype)


def te_causal_mask(attention_mask: Optional[mx.array], batch: int, seq_len: int, dtype) -> mx.array:
    idx = mx.arange(seq_len, dtype=mx.int32)
    causal = mx.where(idx[:, None] >= idx[None, :], mx.zeros((seq_len, seq_len), dtype=dtype),
                      mx.full((seq_len, seq_len), float("-inf"), dtype=dtype))[None, None, :, :]
    if attention_mask is not None:
        pad = mx.where(attention_mask[:, None, None, :] == 1, mx.zeros((batch, 1, 1, seq_len), dtype=dtype),
                       mx.full((batch, 1, 1, seq_len), float("-inf"), dtype=dtype))
        causal = causal + pad
    return causal


class TextEncoder(nn.Module):
    """Embedding + the first `ENCODER_LAYERS_USED` layers. `__call__` runs them all (resident); the pipeline can also
    run `embed`, then each layer on its own, then `finish` (same ops, one layer in memory at a time)."""

    def __init__(self, n_layers: int = ENCODER_LAYERS_USED, build_layers: bool = True, **cfg):
        super().__init__()
        c = {**TE_CONFIG, **cfg}
        self.cfg = c
        self.embed_tokens = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = [EncoderLayer(**c) for _ in range(n_layers)] if build_layers else []

    def embed(self, input_ids: mx.array, attention_mask: Optional[mx.array]):
        b, n = input_ids.shape
        h = self.embed_tokens(input_ids).astype(mx.float32)
        pos = te_rotary(self.cfg["head_dim"], self.cfg["rope_theta"], n, h.dtype)
        mask = te_causal_mask(attention_mask, b, n, h.dtype)
        return h, mask, pos

    @staticmethod
    def finish(h: mx.array) -> mx.array:
        return h.astype(PRECISION)

    def __call__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        h, mask, pos = self.embed(input_ids, attention_mask)
        for layer in self.layers:
            h = layer(h, attention_mask=mask, position_embeddings=pos)
        return self.finish(h)


# ---------------------------------------------------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------------------------------------------------

class ZImageAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, qk_norm: bool = True, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]
        if qk_norm:
            self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
            self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None,
                 freqs_cis: Optional[mx.array] = None) -> mx.array:
        b, n, _ = hidden_states.shape
        q = self.to_q(hidden_states).reshape(b, n, self.n_heads, self.head_dim)
        k = self.to_k(hidden_states).reshape(b, n, self.n_heads, self.head_dim)
        v = self.to_v(hidden_states).reshape(b, n, self.n_heads, self.head_dim)
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))
        mask = None
        if attention_mask is not None:
            mask = mx.where(attention_mask[:, None, None, :], mx.array(0.0, dtype=q.dtype),
                            mx.array(float("-inf"), dtype=q.dtype))
        out = scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = mx.transpose(out, axes=(0, 2, 1, 3)).reshape(b, n, self.dim)
        return self.to_out[0](out)


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    dtype = x.dtype
    b, n, h, d = x.shape
    x = x.reshape(b, n, h, d // 2, 2)
    freqs_cis = mx.expand_dims(mx.expand_dims(freqs_cis, axis=0), axis=2)
    x_real, x_imag = x[..., 0], x[..., 1]
    cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]
    out = mx.stack([x_real * cos - x_imag * sin, x_real * sin + x_imag * cos], axis=-1)
    return out.reshape(b, n, h, d).astype(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class ZImageTransformerBlock(nn.Module):
    def __init__(self, dim: int = 3840, n_heads: int = 30, norm_eps: float = 1e-5, qk_norm: bool = True):
        super().__init__()
        self.attention = ZImageAttention(dim=dim, n_heads=n_heads, qk_norm=qk_norm, eps=1e-5)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = [nn.Linear(min(dim, 256), 4 * dim, bias=True)]

    def __call__(self, x: mx.array, attn_mask: mx.array, freqs_cis: mx.array, t_emb: mx.array) -> mx.array:
        modulation = mx.expand_dims(self.adaLN_modulation[0](t_emb), axis=1)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(modulation, 4, axis=2)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp
        gate_msa = mx.tanh(gate_msa)
        gate_mlp = mx.tanh(gate_mlp)
        attn_out = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + gate_msa * self.attention_norm2(attn_out)
        ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        return x + gate_mlp * self.ffn_norm2(ffn_out)


class ZImageContextBlock(nn.Module):
    def __init__(self, dim: int = 3840, n_heads: int = 30, norm_eps: float = 1e-5, qk_norm: bool = True):
        super().__init__()
        self.attention = ZImageAttention(dim=dim, n_heads=n_heads, qk_norm=qk_norm, eps=1e-5)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(self, x: mx.array, attn_mask: mx.array, freqs_cis: mx.array) -> mx.array:
        attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + self.attention_norm2(attn_out)
        ffn_out = self.feed_forward(self.ffn_norm1(x))
        return x + self.ffn_norm2(ffn_out)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = [nn.Linear(min(hidden_size, 256), hidden_size, bias=True)]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = mx.expand_dims(1.0 + self.adaLN_modulation[0](nn.silu(c)), axis=1)
        return self.linear(self.norm(x) * scale)


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, mid_size: Optional[int] = None, frequency_embedding_size: int = 256):
        super().__init__()
        mid_size = mid_size or out_size
        self.frequency_embedding_size = frequency_embedding_size
        self.linear1 = nn.Linear(frequency_embedding_size, mid_size, bias=True)
        self.linear2 = nn.Linear(mid_size, out_size, bias=True)

    def __call__(self, t: mx.array) -> mx.array:
        return self.linear2(nn.silu(self.linear1(_timestep_embedding(t, self.frequency_embedding_size))))


def _timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
    return emb


class RopeEmbedder:
    def __init__(self, theta: float = 256.0, axes_dims=(32, 48, 48), axes_lens=(1024, 512, 512)):
        self.axes_dims = list(axes_dims)
        self.freqs_cis = []
        for d, e in zip(axes_dims, axes_lens):
            freqs = 1.0 / (theta ** (mx.arange(0, d, 2, dtype=mx.float32) / d))
            f = mx.outer(mx.arange(e, dtype=mx.float32), freqs)
            self.freqs_cis.append(mx.stack([mx.cos(f), mx.sin(f)], axis=-1))
        mx.eval(*self.freqs_cis)

    def __call__(self, ids: mx.array) -> mx.array:
        return mx.concatenate([self.freqs_cis[i][ids[:, i].astype(mx.int32)] for i in range(len(self.axes_dims))],
                              axis=1)


DIT_CONFIG = dict(patch_size=2, f_patch_size=1, in_channels=16, dim=3840, n_layers=30, n_refiner_layers=2,
                  n_heads=30, norm_eps=1e-5, qk_norm=True, cap_feat_dim=2560, rope_theta=256.0, t_scale=1000.0,
                  axes_dims=(32, 48, 48))


class ZImageTransformer(nn.Module):
    """The DiT. `layers` (the 30-block main stack) can be replaced at run time by a streamed stack: pass
    `main_stack(unified, attn_mask, freqs, t_emb) -> unified` to `__call__`."""

    def __init__(self, build_main: bool = True, **cfg):
        super().__init__()
        c = {**DIT_CONFIG, **cfg}
        dim, n_heads, eps, qk = c["dim"], c["n_heads"], c["norm_eps"], c["qk_norm"]
        self.in_channels = self.out_channels = c["in_channels"]
        self.patch_size, self.f_patch_size, self.dim, self.n_heads, self.t_scale = (
            c["patch_size"], c["f_patch_size"], dim, n_heads, c["t_scale"])
        key = f"{c['patch_size']}-{c['f_patch_size']}"
        embed_dim = c["f_patch_size"] * c["patch_size"] * c["patch_size"] * c["in_channels"]
        self.all_x_embedder = {key: nn.Linear(embed_dim, dim, bias=True)}
        self.all_final_layer = {key: FinalLayer(dim, embed_dim)}
        self.t_embedder = TimestepEmbedder(out_size=min(dim, 256), mid_size=1024)
        self.cap_embedder = [nn.RMSNorm(c["cap_feat_dim"], eps=eps), nn.Linear(c["cap_feat_dim"], dim, bias=True)]
        self.x_pad_token = mx.zeros((1, dim))
        self.cap_pad_token = mx.zeros((1, dim))
        self.noise_refiner = [ZImageTransformerBlock(dim, n_heads, eps, qk) for _ in range(c["n_refiner_layers"])]
        self.context_refiner = [ZImageContextBlock(dim, n_heads, eps, qk) for _ in range(c["n_refiner_layers"])]
        self.layers = [ZImageTransformerBlock(dim, n_heads, eps, qk) for _ in range(c["n_layers"])] if build_main else []
        self.n_layers = c["n_layers"]
        self.rope_embedder = RopeEmbedder(theta=c["rope_theta"], axes_dims=tuple(c["axes_dims"]))

    def __call__(self, x: mx.array, timestep: mx.array, cap_feats: mx.array, main_stack=None) -> mx.array:
        key = f"{self.patch_size}-{self.f_patch_size}"
        if timestep.ndim == 0:
            timestep = timestep.reshape((1,))
        t_emb = self.t_embedder(timestep.astype(mx.float32) * self.t_scale).astype(PRECISION)
        x_emb, cap_emb, x_size, x_pos, cap_pos, x_pad, cap_pad = _patchify(x, cap_feats, self.patch_size,
                                                                           self.f_patch_size)
        x_emb = self.all_x_embedder[key](x_emb)
        x_emb = mx.where(x_pad[:, None], self.x_pad_token, x_emb)
        x_freqs = self.rope_embedder(x_pos)
        x_mask = mx.ones((1, x_emb.shape[0]), dtype=mx.bool_)
        x_emb = mx.expand_dims(x_emb, axis=0)
        for layer in self.noise_refiner:
            x_emb = layer(x=x_emb, attn_mask=x_mask, freqs_cis=x_freqs, t_emb=t_emb)
        cap_emb = self.cap_embedder[1](self.cap_embedder[0](cap_emb))
        cap_emb = mx.where(cap_pad[:, None], self.cap_pad_token, cap_emb)
        cap_freqs = self.rope_embedder(cap_pos)
        cap_mask = mx.ones((1, cap_emb.shape[0]), dtype=mx.bool_)
        cap_emb = mx.expand_dims(cap_emb, axis=0)
        for layer in self.context_refiner:
            cap_emb = layer(x=cap_emb, attn_mask=cap_mask, freqs_cis=cap_freqs)
        x_len = x_emb.shape[1]
        unified = mx.concatenate([x_emb, cap_emb], axis=1)
        freqs = mx.concatenate([x_freqs, cap_freqs], axis=0)
        mask = mx.ones((1, unified.shape[1]), dtype=mx.bool_)
        if main_stack is not None:
            unified = main_stack(unified, mask, freqs, t_emb)
        else:
            for layer in self.layers:
                unified = layer(x=unified, attn_mask=mask, freqs_cis=freqs, t_emb=t_emb)
        unified = self.all_final_layer[key](unified, t_emb)
        out = _unpatchify(unified[0, :x_len], x_size, self.patch_size, self.f_patch_size, self.out_channels)
        return -out


def _coord_grid(size, start=None) -> mx.array:
    start = start or tuple(0 for _ in size)
    axes = [mx.arange(x0, x0 + span, dtype=mx.int32) for x0, span in zip(start, size)]
    return mx.stack(mx.meshgrid(*axes, indexing="ij"), axis=-1)


def _patchify(image: mx.array, cap_feats: mx.array, patch_size: int, f_patch_size: int):
    pH = pW = patch_size
    pF = f_patch_size
    cap_len = cap_feats.shape[0]
    cap_padn = (-cap_len) % 32
    cap_pos = _coord_grid((cap_len + cap_padn, 1, 1), (1, 0, 0)).reshape(-1, 3)
    cap_pad = mx.concatenate([mx.zeros((cap_len,), dtype=mx.bool_), mx.ones((cap_padn,), dtype=mx.bool_)])
    cap = mx.concatenate([cap_feats, mx.repeat(cap_feats[-1:], cap_padn, axis=0)], axis=0) if cap_padn > 0 else cap_feats
    C, F, H, W = image.shape
    Ft, Ht, Wt = F // pF, H // pH, W // pW
    image = image.reshape(C, Ft, pF, Ht, pH, Wt, pW)
    image = mx.transpose(image, axes=(1, 3, 5, 2, 4, 6, 0)).reshape(Ft * Ht * Wt, pF * pH * pW * C)
    img_len = image.shape[0]
    img_padn = (-img_len) % 32
    img_pos = _coord_grid((Ft, Ht, Wt), (cap_len + cap_padn + 1, 0, 0)).reshape(-1, 3)
    if img_padn > 0:
        img_pos = mx.concatenate([img_pos, mx.zeros((img_padn, 3), dtype=mx.int32)], axis=0)
        image = mx.concatenate([image, mx.repeat(image[-1:], img_padn, axis=0)], axis=0)
    img_pad = mx.concatenate([mx.zeros((img_len,), dtype=mx.bool_), mx.ones((img_padn,), dtype=mx.bool_)])
    return image, cap, (F, H, W), img_pos, cap_pos, img_pad, cap_pad


def _unpatchify(x: mx.array, size, patch_size: int, f_patch_size: int, out_channels: int) -> mx.array:
    pH = pW = patch_size
    pF = f_patch_size
    F, H, W = size
    n = (F // pF) * (H // pH) * (W // pW)
    x = x[:n].reshape(F // pF, H // pH, W // pW, pF, pH, pW, out_channels)
    return mx.transpose(x, axes=(6, 0, 3, 1, 4, 2, 5)).reshape(out_channels, F, H, W)


# ---------------------------------------------------------------------------------------------------------------------
# VAE decoder (the Flux 16-channel VAE)
# ---------------------------------------------------------------------------------------------------------------------

def _gn(dims: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, dims=dims, eps=1e-6, affine=True, pytorch_compatible=True)


class _Conv(nn.Module):
    """A 3x3 (or 1x1) conv held as `.conv`, NCHW in and out (the mflux key layout: decoder.conv_in.conv.weight)."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=1, padding=k // 2)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.transpose(self.conv(mx.transpose(x, (0, 2, 3, 1))), (0, 3, 1, 2))


class _NormOut(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.norm = _gn(channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 3, 1))
        return mx.transpose(self.norm(x.astype(mx.float32)).astype(PRECISION), (0, 3, 1, 2))


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_conv_shortcut: bool = False):
        super().__init__()
        self.norm1 = _gn(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _gn(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = (nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
                              if use_conv_shortcut or in_channels != out_channels else None)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 3, 1))
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return mx.transpose(x + h, (0, 3, 1, 2))


class VAEAttention(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.group_norm = _gn(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 3, 1))
        B, H, W, C = x.shape
        h = self.group_norm(x.astype(mx.float32)).astype(PRECISION)
        q, k, v = (mx.transpose(p(h).reshape(B, H * W, 1, C), (0, 2, 1, 3)) for p in (self.to_q, self.to_k, self.to_v))
        scale = 1.0 / mx.sqrt(mx.array(q.shape[-1], dtype=q.dtype))
        h = mx.transpose(scaled_dot_product_attention(q, k, v, scale=scale), (0, 2, 1, 3)).reshape(B, H, W, C)
        return mx.transpose(x + self.to_out[0](h), (0, 3, 1, 2))


class UNetMidBlock(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.attentions = [VAEAttention(channels)]
        self.resnets = [ResnetBlock2D(channels, channels), ResnetBlock2D(channels, channels)]


class UpSampler(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)


class UpDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3, add_upsample: bool = True):
        super().__init__()
        self.resnets = [ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels,
                                      use_conv_shortcut=(i == 0 and in_channels != out_channels))
                        for i in range(num_layers)]
        self.upsamplers = [UpSampler(out_channels)] if add_upsample else None


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = _Conv(16, 512)
        self.mid_block = UNetMidBlock(512)
        self.up_blocks = [UpDecoderBlock(512, 512), UpDecoderBlock(512, 512), UpDecoderBlock(512, 256),
                          UpDecoderBlock(256, 128, add_upsample=False)]
        self.conv_norm_out = _NormOut(128)
        self.conv_out = _Conv(128, 3)


class VAE(nn.Module):
    """The decoder half (the encoder's weights in the file are not loaded). Decoding runs in
    shutter.families.zimage.vae_decode.decode (mflux's low-memory block decoder)."""

    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    spatial_scale = 8
    latent_channels = 16

    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
