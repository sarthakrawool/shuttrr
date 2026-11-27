"""Reference pictures for MiniMax-H3-Image ("image_conditioned" / edit requests).

A reference picture enters the model twice (diffusers `modular_pipelines/minimax_h3_image`, Apache-2.0; checked
against stable-diffusion.cpp, MIT):

* through the prompt encoder: a `Picture i: ` label, then a vision block (`<|vision_start|>`, one `<|image_pad|>` per
  merged 32x32 patch, `<|vision_end|>`) whose rows are the Qwen3-VL vision tower's output. The tower's three
  DeepStack outputs are added to those rows after text layers 1-3, and the rows sit on Qwen3-VL's 3D M-RoPE grid.
  Vision-block rows carry the image modality tag in the transformer.
* through the image VAE's encoder: its latents (a posterior sample, rounded to float16, normalized) are noised to
  t = 0.999 and packed into the transformer's sequence in front of the generated image, at their own latent grid, one
  rotary time step each. They are held at t = 0.999 for every step and never written.

The vision tower (`VisionTower`) follows mlx-vlm's `qwen3_vl/vision.py` (MIT); the encoder (`VAEEncoder`) is the 2D
form of the VAE's causal 3D encoder: a single frame with two causal zero frames in front only ever meets the last
temporal slice of every 3x3x3 kernel, so each conv is a 2D conv with `weight[:, :, -1]`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from mlx import nn

#: Qwen3-VL-32B's vision tower (the one MiniMax-H3 ships with its text encoder)
VISION_CONFIG = dict(depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16, patch_size=16,
                     temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=5120, num_position_embeddings=2304,
                     deepstack_visual_indexes=(8, 16, 24))

#: Qwen3-VL text rotary: interleaved M-RoPE over (t, h, w)
MROPE_SECTION = (24, 20, 20)

#: the reference resize bounds of the released pipeline (short edge, multiple of 32)
VAE_MIN_SHORT, VAE_MAX_SHORT = 1024, 2048
VISION_MIN_SHORT, VISION_MAX_SHORT = 1024, 1024
MULTIPLE = 32
#: model timestep the reference rows are held at
REFERENCE_T = 0.999

VISION_START, VISION_END, IMAGE_PAD = "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"


# ---------------------------------------------------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------------------------------------------------

def bound_short_edge(width: int, height: int, lo: int, hi: int, multiple: int = MULTIPLE) -> tuple[int, int]:
    """diffusers `bound_short_edge`: keep the size when the short edge is within [lo, hi], else scale so it lands on
    the nearer bound, both sides rounded to `multiple`."""
    short = min(width, height)
    if lo <= short <= hi:
        return width, height
    scale = (lo if short < lo else hi) / short
    return (max(multiple, round(width * scale / multiple) * multiple),
            max(multiple, round(height * scale / multiple) * multiple))


def load_reference(path) -> "PIL.Image.Image":
    from PIL import Image, ImageOps
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if min(img.size) < MULTIPLE:
        raise ValueError(f"a reference picture needs at least {MULTIPLE} pixels on each side, got {img.size}")
    return img


def resized(img, lo: int, hi: int):
    from PIL import Image
    w, h = bound_short_edge(*img.size, lo, hi)
    return img if (w, h) == img.size else img.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------------------------------------------------

def vision_patches(img) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Qwen2-VL / Qwen3-VL image processor for a picture whose sides are already multiples of 32: pixels to [-1, 1]
    (mean 0.5, std 0.5), the frame doubled over the temporal patch, patches in merge-window order. Returns
    ((N, 3 * 2 * 16 * 16) float32, (t, h, w) grid in patches)."""
    c = VISION_CONFIG
    p, tp, m = c["patch_size"], c["temporal_patch_size"], c["spatial_merge_size"]
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = x.transpose(2, 0, 1)[None]                                   # (1, C, H, W)
    x = np.concatenate([x] * tp, axis=0)                              # (tp, C, H, W)
    H, W = x.shape[2], x.shape[3]
    gh, gw = H // p, W // p
    x = x.reshape(1, tp, 3, gh // m, m, p, gw // m, m, p)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(x.reshape(gh * gw, 3 * tp * p * p)), (1, gh, gw)


def _rotate_half(x):
    a, b = mx.split(x, 2, axis=-1)
    return mx.concatenate([-b, a], axis=-1)


class VAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x, cos, sin):
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.heads, self.head_dim).transpose(1, 2, 0, 3)   # (3, heads, n, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dt = q.dtype
        q = (q.astype(mx.float32) * cos + _rotate_half(q.astype(mx.float32)) * sin).astype(dt)
        k = (k.astype(mx.float32) * cos + _rotate_half(k.astype(mx.float32)) * sin).astype(dt)
        o = mx.fast.scaled_dot_product_attention(q[None], k[None], v[None], scale=self.head_dim ** -0.5)[0]
        return self.proj(o.transpose(1, 0, 2).reshape(n, -1))


class VMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden, bias=True)
        self.linear_fc2 = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x):
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class VBlock(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.norm1 = nn.LayerNorm(c["hidden_size"], eps=1e-6)
        self.norm2 = nn.LayerNorm(c["hidden_size"], eps=1e-6)
        self.attn = VAttention(c["hidden_size"], c["num_heads"])
        self.mlp = VMLP(c["hidden_size"], c["intermediate_size"])

    def __call__(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class Merger(nn.Module):
    def __init__(self, c: dict, postshuffle: bool):
        super().__init__()
        self.hidden = c["hidden_size"] * c["spatial_merge_size"] ** 2
        self.postshuffle = postshuffle
        self.norm = nn.LayerNorm(self.hidden if postshuffle else c["hidden_size"], eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden, c["out_hidden_size"], bias=True)

    def __call__(self, x):
        x = self.norm(x.reshape(-1, self.hidden)) if self.postshuffle else self.norm(x).reshape(-1, self.hidden)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class PatchEmbed(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.proj = nn.Linear(3 * c["temporal_patch_size"] * c["patch_size"] ** 2, c["hidden_size"], bias=True)


class VisionTower(nn.Module):
    """Qwen3-VL's ViT: patch embedding (a 2x16x16 Conv3d, as one matmul), bilinearly interpolated learned positions,
    27 blocks with 2D rotary, the merger (2x2 patches -> 5120) and three DeepStack mergers after blocks 8, 16, 24."""

    def __init__(self, **cfg):
        super().__init__()
        c = {**VISION_CONFIG, **cfg}
        self.c = c
        self.patch_embed = PatchEmbed(c)
        self.pos_embed = nn.Embedding(c["num_position_embeddings"], c["hidden_size"])
        self.blocks = [VBlock(c) for _ in range(c["depth"])]
        self.merger = Merger(c, postshuffle=False)
        self.deepstack_merger_list = [Merger(c, postshuffle=True) for _ in c["deepstack_visual_indexes"]]

    def _pos_embed(self, gh: int, gw: int) -> mx.array:
        n = int(math.isqrt(self.c["num_position_embeddings"]))
        m = self.c["spatial_merge_size"]
        hi = np.linspace(0, n - 1, gh, dtype=np.float32)
        wi = np.linspace(0, n - 1, gw, dtype=np.float32)
        hf, wf = hi.astype(np.int32), wi.astype(np.int32)
        hc, wc = np.minimum(hf + 1, n - 1), np.minimum(wf + 1, n - 1)
        dh, dw = hi - hf, wi - wf
        idx = [(hf[:, None] * n + wf[None]), (hf[:, None] * n + wc[None]), (hc[:, None] * n + wf[None]),
               (hc[:, None] * n + wc[None])]
        wts = [((1 - dh)[:, None] * (1 - dw)[None]), ((1 - dh)[:, None] * dw[None]), (dh[:, None] * (1 - dw)[None]),
               (dh[:, None] * dw[None])]
        e = self.pos_embed(mx.array(np.stack([i.reshape(-1) for i in idx]).astype(np.int32)))     # (4, gh*gw, D)
        w = mx.array(np.stack([x.reshape(-1) for x in wts]).astype(np.float32)).astype(e.dtype)
        pe = (e * w[:, :, None]).sum(axis=0)
        d = pe.shape[-1]
        return pe.reshape(gh // m, m, gw // m, m, d).transpose(0, 2, 1, 3, 4).reshape(-1, d)

    def _rotary(self, gh: int, gw: int) -> tuple[mx.array, mx.array]:
        m = self.c["spatial_merge_size"]
        hd = self.c["hidden_size"] // self.c["num_heads"]
        dim = hd // 2
        inv = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        rows = (np.arange(gh // m)[:, None, None, None] * m + np.arange(m)[None, None, :, None])
        cols = (np.arange(gw // m)[None, :, None, None] * m + np.arange(m)[None, None, None, :])
        rows = np.broadcast_to(rows, (gh // m, gw // m, m, m)).reshape(-1)
        cols = np.broadcast_to(cols, (gh // m, gw // m, m, m)).reshape(-1)
        f = np.concatenate([np.outer(rows, inv), np.outer(cols, inv)], axis=-1)                 # (N, hd/2)
        f = np.concatenate([f, f], axis=-1).astype(np.float32)                                  # (N, hd)
        return mx.array(np.cos(f))[None], mx.array(np.sin(f))[None]

    def __call__(self, patches: mx.array, grid: tuple[int, int, int]) -> tuple[mx.array, list[mx.array]]:
        _, gh, gw = grid
        dt = self.patch_embed.proj.weight.dtype
        x = self.patch_embed.proj(patches.astype(dt))
        x = x + self._pos_embed(gh, gw).astype(x.dtype)
        cos, sin = self._rotary(gh, gw)
        deep = []
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin)
            if i in self.c["deepstack_visual_indexes"]:
                deep.append(self.deepstack_merger_list[self.c["deepstack_visual_indexes"].index(i)](x))
                mx.eval(deep[-1])
            mx.eval(x)
        return self.merger(x), deep

    @staticmethod
    def remap(name: str, a: mx.array):
        """HF tensor names -> this module (the patch Conv3d becomes a Linear over (C, T, P, P))."""
        name = name[len("visual."):] if name.startswith("visual.") else name
        if name == "patch_embed.proj.weight":
            a = a.reshape(a.shape[0], -1)
        return name, a


# ---------------------------------------------------------------------------------------------------------------------
# Text side: M-RoPE positions for a presentation with vision blocks
# ---------------------------------------------------------------------------------------------------------------------

def mrope_positions(token_ids: list[int], blocks: list[tuple[int, int, int]]) -> np.ndarray:
    """Qwen3-VL `get_rope_index` for images: (3, n) positions. `blocks` = (index of the first <|image_pad|>, merged
    grid h, merged grid w) per vision block. Text rows count up on all three axes; an image's rows sit at
    (start, start + row, start + col) and the text after it resumes at max + 1."""
    n = len(token_ids)
    pos = np.zeros((3, n), dtype=np.float32)
    cur, nxt = 0, 0
    for start, gh, gw in sorted(blocks):
        for i in range(cur, start):
            pos[:, i] = nxt
            nxt += 1
        rr, cc = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        k = gh * gw
        pos[0, start:start + k] = nxt
        pos[1, start:start + k] = nxt + rr.reshape(-1)
        pos[2, start:start + k] = nxt + cc.reshape(-1)
        nxt = nxt + max(gh, gw)
        cur = start + k
    for i in range(cur, n):
        pos[:, i] = nxt
        nxt += 1
    return pos


def mrope_cos_sin(positions: np.ndarray, head_dim: int, theta: float, dtype) -> tuple[mx.array, mx.array]:
    """Interleaved M-RoPE (HF `apply_interleaved_mrope`, section 24/20/20): frequency j comes from the h axis when
    j % 3 == 1 and j < 60, from the w axis when j % 3 == 2 and j < 60, else from t. With text only all axes are equal
    and this is the plain 1D rotary."""
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))       # (64,)
    f = positions[:, :, None].astype(np.float64) * inv[None, None, :]                    # (3, n, 64)
    out = f[0].copy()
    for axis, off in ((1, 1), (2, 2)):
        sl = slice(off, MROPE_SECTION[axis] * 3, 3)
        out[:, sl] = f[axis][:, sl]
    out = out.astype(np.float32)
    emb = np.concatenate([out, out], axis=-1)
    return mx.array(np.cos(emb))[None].astype(dtype), mx.array(np.sin(emb))[None].astype(dtype)


# ---------------------------------------------------------------------------------------------------------------------
# VAE encoder (2D form of the causal 3D CNN)
# ---------------------------------------------------------------------------------------------------------------------

def _reflect1(x: mx.array) -> mx.array:
    """Reflect-pad H and W by 1 (NHWC)."""
    x = mx.concatenate([x[:, 1:2], x, x[:, -2:-1]], axis=1)
    return mx.concatenate([x[:, :, 1:2], x, x[:, :, -2:-1]], axis=2)


class Conv(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, pad: str = "reflect"):
        super().__init__()
        self.weight = mx.zeros((cout, k, k, cin), dtype=mx.float32)
        self.bias = mx.zeros((cout,), dtype=mx.float32)
        self.k, self.stride, self.pad = k, stride, pad

    def __call__(self, x):
        if self.k == 3 and self.pad == "reflect":
            x = _reflect1(x)
        elif self.pad == "down":                                   # bottom/right reflect pad of 1, then stride 2
            x = mx.concatenate([x, x[:, -2:-1]], axis=1)
            x = mx.concatenate([x, x[:, :, -2:-1]], axis=2)
        return mx.conv2d(x, self.weight, stride=self.stride) + self.bias


class Resnet(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin, eps=1e-6, pytorch_compatible=True)
        self.conv1 = Conv(cin, cout, 3)
        self.norm2 = nn.GroupNorm(32, cout, eps=1e-6, pytorch_compatible=True)
        self.conv2 = Conv(cout, cout, 3)
        if cin != cout:
            self.nin_shortcut = Conv(cin, cout, 1, pad="none")

    def __call__(self, x):
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        if "nin_shortcut" in self:
            x = self.nin_shortcut(x)
        return x + h


class Down(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = Conv(c, c, 3, stride=2, pad="down")


class Level(nn.Module):
    def __init__(self, cin: int, cout: int, down: bool):
        super().__init__()
        self.block = [Resnet(cin, cout), Resnet(cout, cout)]
        if down:
            self.downsample = Down(cout)


class VAEEncoder(nn.Module):
    """`encoder` + `quant_conv` of the H3-Image VAE: pixels (H, W, 3) in [0, 1] -> 48 moments at 1/16 size."""

    MULT = (1, 2, 2, 4, 4, 8)
    SPATIAL_DOWN = (2, 2, 2, 2, 1, 1)

    def __init__(self):
        super().__init__()
        ch = 128
        self.conv_in = Conv(3, ch, 3)
        prev = ch
        self.down = []
        for lvl, m in enumerate(self.MULT):
            cur = ch * m
            self.down.append(Level(prev, cur, self.SPATIAL_DOWN[lvl] > 1 or lvl in (1, 2)))
            prev = cur
        self.norm_out = nn.GroupNorm(32, prev, eps=1e-6, pytorch_compatible=True)
        self.conv_out = Conv(prev, 48, 3)
        self.quant_conv = Conv(48, 48, 1, pad="none")

    def __call__(self, pixels: mx.array, check=None) -> mx.array:
        from shutter.families.h3image.model import PIXEL_MEAN, PIXEL_STD
        x = (pixels.astype(mx.float32) - mx.array(PIXEL_MEAN)) / mx.array(PIXEL_STD)
        x = self.conv_in(x[None])
        mx.eval(x)
        for lvl in self.down:
            for b in lvl.block:
                if check is not None:
                    check()
                x = b(x)
                mx.eval(x)
            if "downsample" in lvl:
                x = lvl.downsample.conv(x)
                mx.eval(x)
        x = self.conv_out(nn.silu(self.norm_out(x)))
        return self.quant_conv(x)[0]                                     # (h, w, 48)

    @staticmethod
    def remap(name: str, a: mx.array):
        """`encoder.down.3.block.1.conv1.weight` [out, in, t, kh, kw] -> `down.3.block.1.conv1.weight` [out, kh, kw,
        in] (the last temporal slice); norms and biases unchanged; float32."""
        name = name[len("encoder."):] if name.startswith("encoder.") else name
        if name.endswith(".weight") and a.ndim == 5:
            a = a[:, :, -1].transpose(0, 2, 3, 1)
        return name, a.astype(mx.float32)


#: the released VAE encodes in tiles (diffusers AutoencoderKLMiniMaxH3Image: use_tiling on, 256 px tiles, >= 64 px
#: overlap, blended linearly in latent space); a tile's working memory is ~0.3 GB instead of ~8 GB for 1024x1024
TILE, TILE_OVERLAP, RATIO = 256, 64, 16


def split_tiles(length: int, tile: int = TILE, min_overlap: int = TILE_OVERLAP) -> tuple[list[int], list[int]]:
    """diffusers `_split_tiles`: (tile starts, overlaps) covering `length` exactly with latent-aligned overlaps."""
    if tile >= length:
        return [0], []
    n = math.ceil(length / tile)
    while tile * n - min_overlap * (n - 1) - length < 0:
        n += 1
    overlaps = [min_overlap] * (n - 1)
    remaining = tile * n - sum(overlaps) - length
    for i in range(remaining // RATIO):
        overlaps[i % (n - 1)] += RATIO
    starts = [0]
    for i in range(n - 1):
        starts.append(starts[-1] + tile - overlaps[i])
    return starts, overlaps


def _blend(a: np.ndarray, b: np.ndarray, extent: int, axis: int) -> np.ndarray:
    extent = min(a.shape[axis], b.shape[axis], extent)
    pos = np.arange(extent, dtype=np.float32)
    shape = [1] * a.ndim
    shape[axis] = extent
    wa, wb = (1 - pos / extent).reshape(shape), (pos / extent).reshape(shape)
    sa = [slice(None)] * a.ndim
    sa[axis] = slice(-extent, None)
    sb = [slice(None)] * b.ndim
    sb[axis] = slice(0, extent)
    blended = a[tuple(sa)] * wa + b[tuple(sb)] * wb
    if extent == b.shape[axis]:
        return blended
    sr = [slice(None)] * b.ndim
    sr[axis] = slice(extent, None)
    return np.concatenate([blended, b[tuple(sr)]], axis=axis)


def encode_moments(encoder: VAEEncoder, img, check=None) -> np.ndarray:
    """A PIL picture -> (48, h, w) float32 moments: cropped (top left) to whole 16-pixel cells, encoded in 256 px
    tiles, stitched like diffusers `_stitch_tiles` (each tile blended with the unblended tile above and to its left)."""
    x = np.asarray(img, dtype=np.float32) / 255.0
    H, W = x.shape[0] // RATIO * RATIO, x.shape[1] // RATIO * RATIO
    x = x[:H, :W]
    ys, yo = split_tiles(H)
    xs, xo = split_tiles(W)
    th, tw = min(TILE, H), min(TILE, W)
    tiles = []
    for y in ys:
        row = []
        for x0 in xs:
            m = encoder(mx.array(x[y:y + th, x0:x0 + tw]), check=check)
            mx.eval(m)
            row.append(np.array(m.astype(mx.float32)).transpose(2, 0, 1))        # (48, th/16, tw/16)
        tiles.append(row)
    lyo, lxo = [o // RATIO for o in yo], [o // RATIO for o in xo]
    rows = []
    for i, row in enumerate(tiles):
        out = []
        for j, t in enumerate(row):
            if i > 0:
                t = _blend(tiles[i - 1][j], t, lyo[i - 1], axis=1)
            if j > 0:
                t = _blend(row[j - 1], t, lxo[j - 1], axis=2)
            if i < len(tiles) - 1:
                t = t[:, : -lyo[i], :]
            if j < len(row) - 1:
                t = t[:, :, : -lxo[j]]
            out.append(t)
        rows.append(np.concatenate(out, axis=2))
    return np.concatenate(rows, axis=1)


def reference_latents(moments: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Posterior sample (mean + exp(0.5 * clamp(logvar)) * noise), rounded to float16, cropped to whole 2x2 patches,
    normalized with the latent statistics. Returns (24, h, w) float32."""
    from shutter.families.h3image.model import LATENTS_MEAN, LATENTS_STD
    mean, logvar = moments[:24], np.clip(moments[24:], -30.0, 20.0)
    z = (mean + np.exp(0.5 * logvar) * noise.reshape(mean.shape)).astype(np.float16).astype(np.float32)
    z = z[:, : z.shape[1] // 2 * 2, : z.shape[2] // 2 * 2]
    lm = np.array(LATENTS_MEAN, np.float32)[:, None, None]
    ls = np.array(LATENTS_STD, np.float32)[:, None, None]
    return (z - lm) / ls
