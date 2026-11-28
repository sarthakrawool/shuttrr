"""Text to image with MiniMax-H3-Image on any Mac from 16 GB up.

Stages and what is in memory during each (`RUNGS` is the memory step table):

    encode   the Qwen3-VL-32B text layers H3 reads (50), one layer at a time: ~0.3-0.45 GB at a time, never the 20 GB
             encoder. Only the prompt's rows of the token embedding are read from the SSD. The output (one [n, 5120]
             array) is refined by the DiT's token refiner once and kept for all six steps.
    denoise  the DiT (20B). resident / split / staged: the whole DiT (~22 GB at 8-bit). stream_wide / stream_tight:
             the 50-block stack is read from the SSD as it runs, 8 or 2 blocks in memory at a time (0.43 GB each);
             the rest (embedders, token refiner, AdaLN curve, output heads, ~1 GB) stays.
    decode   the ViT VAE decoder (0.8 GB float16, run in float32). resident and split keep the DiT in memory
             meanwhile (and between requests); staged and the stream steps free it first.

Every step gives the same image, bit for bit: the same bytes go through the same ops in the same order.

Sampling is the model's own: six Euler steps over the PDD sigmas, CFG 1 (one forward per step), the six output heads
blended per step. Initial noise uses the PyTorch CPU generator (`torch.manual_seed(seed); torch.randn(...)`), the same
as stable-diffusion.cpp's `--rng cpu`, so a seed gives the same start as sd-cli.
"""

from __future__ import annotations

import gc
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from shutter.families.zimage import weights as W
from shutter.families.zimage.pipeline import Meter

GB = 1024**3

RUNGS: dict[str, dict[str, Any]] = {
    "resident": {"window": 0, "keep_dit": True},
    "split": {"window": 0, "keep_dit": True},
    "staged": {"window": 0, "keep_dit": False},
    "stream_wide": {"window": 8, "keep_dit": False},
    "stream_tight": {"window": 2, "keep_dit": False},
}

SIZE_MULTIPLE = 32
MIN_SIDE, MAX_SIDE = 256, 2048
MAX_TOKENS = 1024
_BLOCK = re.compile(r"^blocks\.(\d+)\.")


@dataclass
class Settings:
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int = 42
    count: int = 1
    images: list[str] = field(default_factory=list)      # reference pictures (paths), referred to as <Picture i>
    notes: list[str] = field(default_factory=list)
    texture: Any = None              # easyfix.TextureFix for text to image; None = the plain six Euler steps

    @property
    def seeds(self) -> list[int]:
        return [int(self.seed) + i for i in range(max(1, int(self.count)))]


def align(s: Settings) -> Settings:
    """Sizes to multiples of 32 (rounded up, as sd.cpp does), between 256 and 2048."""
    for k in ("width", "height"):
        v = int(getattr(s, k))
        a = min(MAX_SIDE, max(MIN_SIDE, -(-v // SIZE_MULTIPLE) * SIZE_MULTIPLE))
        if a != v:
            s.notes.append(f"{k} {v} is not a multiple of {SIZE_MULTIPLE} (or out of range): used {a}")
        setattr(s, k, a)
    return s


class TorchRandn:
    """`torch.Generator().manual_seed(seed)` on the CPU, drawing `torch.randn` tensors one after another (sd.cpp
    MT19937RNG): mt19937 seeded with init_genrand, 24-bit uniforms, Box-Muller in blocks of 16, the tail redrawn."""

    def __init__(self, seed: int):
        self.rs = np.random.RandomState(int(seed) & 0xFFFFFFFF)

    def _uniforms(self, k: int) -> np.ndarray:
        u = self.rs.randint(0, 2**32, size=k, dtype=np.uint32).astype(np.uint64) & ((1 << 24) - 1)
        return u.astype(np.float32) * np.float32(2.0 ** -24)

    @staticmethod
    def _fill16(d: np.ndarray) -> np.ndarray:
        u1 = np.float32(1.0) - d[:8]
        u2 = d[8:16]
        r = np.sqrt(np.float32(-2.0) * np.log(u1))
        th = np.float32(2.0 * np.pi) * u2
        return np.concatenate([r * np.cos(th), r * np.sin(th)]).astype(np.float32)

    def randn(self, n: int) -> np.ndarray:
        if n < 16:
            raise ValueError("latents are always at least 16 values")
        u = self._uniforms(n)
        out = u.copy()
        for i in range(0, n - 15, 16):
            out[i:i + 16] = self._fill16(u[i:i + 16])
        if n % 16:
            out[n - 16:] = self._fill16(self._uniforms(16))
        return out


def torch_randn(seed: int, n: int) -> np.ndarray:
    """`torch.manual_seed(seed); torch.randn(n)` on the CPU, in numpy."""
    return TorchRandn(seed).randn(n)


class Tokenizer:
    """The Qwen tokenizer; H3 encodes the prompt verbatim, no chat template, no special tokens."""

    def __init__(self, folder: Path):
        from tokenizers import Tokenizer as HFTokenizer
        self.tok = HFTokenizer.from_file(str(Path(folder) / "tokenizer.json"))

    def ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def special(self, token: str) -> int:
        i = self.tok.token_to_id(token)
        if i is None:
            raise ValueError(f"the tokenizer has no {token}")
        return i

    def __call__(self, prompt: str) -> tuple[np.ndarray, bool]:
        ids = self.tok.encode(prompt, add_special_tokens=False).ids
        truncated = len(ids) > MAX_TOKENS
        ids = ids[:MAX_TOKENS] or [self.tok.token_to_id(" ") or 220]
        return np.array(ids, dtype=np.int32), truncated


def read_rows(comp: W.Component, name: str, ids: np.ndarray):
    """Rows `ids` of a 2D tensor (the token embedding), read straight from the shard with F_NOCACHE: ~10 KB per
    token instead of the whole 1.6 GB table."""
    import mlx.core as mx
    from shutter.core.stream import SafetensorsIndex, _pread_into, to_mlx_array
    with SafetensorsIndex([str(comp.folder / comp.weight_map[name])], nocache=True) as ix:
        e = ix.entries[name]
        width = int(np.prod(e.shape[1:]))
        item = e.nbytes // max(1, int(np.prod(e.shape)))
        row_bytes = width * item
        uniq, inv = np.unique(ids, return_inverse=True)
        buf = np.empty((len(uniq), row_bytes), dtype=np.uint8)
        fd = ix._fd(e.file)
        for j, r in enumerate(uniq):
            _pread_into(fd, memoryview(buf[j]), e.offset + int(r) * row_bytes, name)
        np_dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[e.dtype]
        rows = buf.view(np_dt).reshape(len(uniq), width)[inv]
        return to_mlx_array(np.ascontiguousarray(rows), e.dtype)


class H3ImagePipeline:
    """One Shuttrr H3-Image build folder (transformer/, text_encoder/, vae/, tokenizer/, h3image.json). `pool`
    (shutter.core.pool.ModelPool or None) keeps the DiT warm between requests on the resident / split steps."""

    def __init__(self, model_dir: Path, model_id: str = "h3-image", *, pool=None,
                 report: Optional[Callable[[float, str], None]] = None,
                 check_cancelled: Optional[Callable[[], None]] = None, meter: Optional[Meter] = None):
        from shutter.families.h3image.model import DIT_CONFIG, TE_CONFIG
        self.dir = Path(model_dir)
        self.model_id = model_id
        self.pool = pool
        self.report = report or (lambda p, s: None)
        self.check = check_cancelled or (lambda: None)
        self.meter = meter or Meter()
        self.te = W.Component(self.dir / "text_encoder")
        self.dit_c = W.Component(self.dir / "transformer")
        self.vae_c = W.Component(self.dir / "vae")
        info = json.loads((self.dir / "h3image.json").read_text()) if (self.dir / "h3image.json").exists() else {}
        self.dit_cfg = {**DIT_CONFIG, **(info.get("dit_config") or {})}
        self.te_cfg = {**TE_CONFIG, **(info.get("te_config") or {})}
        self.vae_layers = int(info.get("vae_layers") or 24)
        self.vae_dim = int(info.get("vae_dim") or 1024)
        self.te_layers = len({int(m.group(1)) for n in self.te.weight_map
                              for m in [re.match(r"^layers\.(\d+)\.", n)] if m})
        self.n_blocks = len({int(m.group(1)) for n in self.dit_c.weight_map for m in [_BLOCK.match(n)] if m})
        self.stream_stats: dict[str, Any] = {}
        self.stream_io = "direct"      # "direct": SSD reads straight into reusable MLX buffers; "copy": numpy + copy
        self.stream_window: Optional[int] = None   # blocks held while streaming (None: the memory step's own)
        self.steer_stats: list = []        # (block, measured projection, c_plain, c_target) from texture steering

    # ---- sizes (for estimates)
    def part_sizes(self) -> dict[str, int]:
        return {"te_embed_row": 5120 * 2,
                "te_layer": W.part_bytes(self.te, lambda n: n.startswith("layers.0.")),
                "te_layers": self.te_layers,
                "dit": W.part_bytes(self.dit_c, lambda n: True),
                "dit_block": max(W.part_bytes(self.dit_c, lambda n, i=i: n.startswith(f"blocks.{i}."))
                                 for i in range(min(self.n_blocks, 4))),
                "dit_blocks": self.n_blocks,
                "dit_shell": self.shell_bytes(),
                "vae": W.part_bytes(self.vae_c, lambda n: True)}

    # ---- prompt encoder, one layer at a time
    def has_refs_weights(self) -> bool:
        return (self.dir / "vision").is_dir() and (self.dir / "vae_encoder").is_dir()

    def encode(self, prompt: str, vision_images: Optional[list] = None):
        """(1, n, 5120) bfloat16: the output of text layer 50, before the final norm. With reference pictures the
        presentation is `Picture i: <vision block>` per picture, then the prompt (diffusers
        MiniMaxH3ImageEditTextEncoderStep); returns (states, truncated, row tags or None)."""
        import mlx.core as mx
        from shutter.families.zimage.model import EncoderLayer, te_causal_mask, te_rotary
        tok = Tokenizer(self.dir / "tokenizer")
        ids, truncated = tok(prompt)
        if not vision_images:
            n = len(ids)
            h = read_rows(self.te, "embed_tokens.weight", ids)[None].astype(mx.bfloat16)
            pos = te_rotary(self.te_cfg["head_dim"], self.te_cfg["rope_theta"], n, mx.bfloat16)
            tags, vis_rows, deep = None, None, None
        else:
            from shutter.families.h3image import refs as R
            feats, deeps, grids = self.vision(vision_images)
            merge = R.VISION_CONFIG["spatial_merge_size"]
            start, pad, end = tok.special(R.VISION_START), tok.special(R.IMAGE_PAD), tok.special(R.VISION_END)
            all_ids: list[int] = []
            tags = []
            blocks = []
            for i, (f, (_, gh, gw)) in enumerate(zip(feats, grids)):
                label = tok.ids(f"Picture {i + 1}: ")[:MAX_TOKENS]
                all_ids += label
                tags += [1] * len(label)
                k = (gh // merge) * (gw // merge)
                if f.shape[0] != k:
                    raise RuntimeError(f"vision tower gave {f.shape[0]} rows for a {k}-row block")
                all_ids.append(start)
                blocks.append((len(all_ids), gh // merge, gw // merge))
                all_ids += [pad] * k + [end]
                tags += [0] * (k + 2)
            all_ids += [int(i) for i in ids]
            tags += [1] * len(ids)
            n = len(all_ids)
            arr = np.array(all_ids, dtype=np.int32)
            h = read_rows(self.te, "embed_tokens.weight", arr).astype(mx.bfloat16)
            vis_rows = np.concatenate([np.arange(b, b + gh * gw) for b, gh, gw in blocks])
            vr = mx.array(vis_rows.astype(np.int32))
            h[vr] = mx.concatenate(feats, axis=0).astype(mx.bfloat16)
            h = h[None]
            deep = [mx.concatenate([d[j] for d in deeps], axis=0).astype(mx.bfloat16) for j in range(len(deeps[0]))]
            pos = R.mrope_cos_sin(R.mrope_positions(all_ids, blocks), self.te_cfg["head_dim"],
                                  self.te_cfg["rope_theta"], mx.bfloat16)
            del feats, deeps
        mask = te_causal_mask(None, 1, n, mx.bfloat16)
        mx.eval(h, *pos, mask)
        for i in range(self.te_layers):
            self.check()
            layer = EncoderLayer(**self.te_cfg)
            W.load_into(layer, W.strip(self.te.load(self.te.names(f"layers.{i}."), nocache=True), f"layers.{i}."))
            h = layer(h, attention_mask=mask, position_embeddings=pos)
            if deep is not None and i < len(deep):           # DeepStack: vision features added after layers 1-3
                h[:, vr] = h[:, vr] + deep[i][None]
            mx.eval(h)
            del layer
            self.report(0.005 * (i + 1) / self.te_layers * 8, f"reading the prompt ({i + 1}/{self.te_layers})")
        deep = None
        gc.collect()
        mx.clear_cache()
        if vision_images:
            return h, truncated, tags
        return h, truncated

    # ---- reference pictures
    def vision(self, images: list):
        """Qwen3-VL vision tower over each picture (one at a time, full attention within a picture): per picture
        (merged rows (k, 5120), [3 DeepStack rows], (1, gh, gw) patch grid)."""
        import mlx.core as mx
        from shutter.families.h3image import refs as R
        comp = W.Component(self.dir / "vision")
        tower = R.VisionTower()
        raw = comp.load(list(comp.weight_map), nocache=True)
        tower.load_weights([R.VisionTower.remap(k, v) for k, v in raw.items()], strict=True)
        mx.eval(tower.parameters())
        del raw
        feats, deeps, grids = [], [], []
        for img in images:
            self.check()
            patches, grid = R.vision_patches(img)
            f, d = tower(mx.array(patches), grid)
            mx.eval(f, *d)
            feats.append(f)
            deeps.append(d)
            grids.append(grid)
        del tower
        gc.collect()
        mx.clear_cache()
        return feats, deeps, grids

    def ref_moments(self, images: list) -> list[np.ndarray]:
        """VAE encoder moments (48, h, w) of each reference picture."""
        import mlx.core as mx
        from shutter.families.h3image import refs as R
        comp = W.Component(self.dir / "vae_encoder")
        enc = R.VAEEncoder()
        raw = comp.load(list(comp.weight_map), nocache=True)
        enc.load_weights([R.VAEEncoder.remap(k, v) for k, v in raw.items()], strict=True)
        mx.eval(enc.parameters())
        del raw
        out = [R.encode_moments(enc, img, check=self.check) for img in images]
        del enc
        gc.collect()
        mx.clear_cache()
        return out

    # ---- DiT
    def _dit_key(self) -> str:
        return f"h3image/{self.model_id}/dit"

    def _shell_key(self) -> str:
        return f"h3image/{self.model_id}/shell"

    def shell_bytes(self) -> int:
        return W.part_bytes(self.dit_c, lambda n: not _BLOCK.match(n))

    def _load_dit(self, build_blocks: bool):
        import mlx.core as mx
        from shutter.families.h3image.model import MiniMaxH3ImageDiT
        dit = MiniMaxH3ImageDiT(build_blocks=build_blocks, **self.dit_cfg)
        names = [n for n in self.dit_c.weight_map if build_blocks or not _BLOCK.match(n)]
        W.load_into(dit, self.dit_c.load(names, nocache=True))
        mx.eval(dit.parameters())
        return dit

    def load_shell(self):
        return self.pool.get(self._shell_key(), lambda: self._load_dit(False), self.shell_bytes(),
                             residency="streamed")

    def dit(self, rung: str):
        """(transformer, blocks runner or None, release()) for this memory step (see zimage.pipeline.dit)."""
        r = RUNGS[rung]
        if self.pool is not None and self._dit_key() in self.pool:
            return self.pool.get(self._dit_key(), lambda: self._load_dit(True), self.dit_c.total_bytes()), None, \
                (lambda: None)
        shell_pooled = self.pool is not None and self._shell_key() in self.pool
        window = r["window"] or (RUNGS["stream_wide"]["window"] if shell_pooled and not r["keep_dit"] else 0)
        if window:
            dit = self.load_shell() if shell_pooled else self._load_dit(False)
            stack = self._streamed_blocks(int(self.stream_window or window))
            return dit, stack, stack.close
        size = self.dit_c.total_bytes()
        if self.pool is not None and r["keep_dit"]:
            dit = self.pool.get(self._dit_key(), lambda: self._load_dit(True), size)
        else:
            if self.pool is not None:
                self.pool.release(self._dit_key())
            dit = self._load_dit(True)
        return dit, None, (lambda: None)

    def _streamed_blocks(self, window: int):
        """The 50-block stack read from the SSD with `window` blocks in memory. Blocks differ in which matrices are
        quantized (the release keeps a few at 16 bits), so there is one reusable shell per layout."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_unflatten
        from shutter.core.stream import BlockStreamer, SlotStreamer, group_names_by_block
        from shutter.families.h3image.model import TransformerBlock
        index = self.dit_c.tensor_index()
        groups, _ = group_names_by_block(index.names(), pattern=r"^(blocks)\.(\d+)\.")
        # ~22 GB per step, read in a loop: the file cache never helps
        if self.stream_io == "direct":
            streamer = SlotStreamer(index, groups, window=window, readahead=max(1, window - 1), cyclic=True,
                                    nocache=True)
        else:
            streamer = BlockStreamer(index, groups, window=window, readahead=max(1, window - 1), cyclic=True,
                                     nocache=True)
        index.set_nocache(True)
        shells: dict[tuple, Any] = {}
        empty = mx.zeros((0,), dtype=mx.float32)
        cfg = self.dit_cfg
        pipe = self

        def shell_for(local: dict[str, Any]):
            sig = tuple(sorted(k for k in local if k.endswith(".scales")))
            sh = shells.get(sig)
            if sh is None:
                sh = TransformerBlock(cfg)
                quant = {k[: -len(".scales")] for k in sig}

                def pred(path, m, local=local, quant=quant):
                    if path not in quant:
                        return False
                    return W.stored_quant(local, path, m)
                nn.quantize(sh, class_predicate=pred)
                shells[sig] = sh
            return sh

        class Blocks:
            def __call__(self, x, temb, rows, rope, hook=None):
                for i in range(len(streamer)):
                    pipe.check()
                    with streamer.block(i) as tensors:
                        local = {_BLOCK.sub("", n, count=1): a for n, a in tensors.items()}
                        sh = shell_for(local)
                        sh.update(tree_unflatten(list(local.items())), strict=False)
                        x = sh(x, temb, rows, rope)
                        if hook is not None:
                            x = hook(i, x)
                        mx.eval(x)
                        sh.update(tree_unflatten([(k, empty) for k, _ in tree_flatten(sh.parameters())]),
                                  strict=False)
                        tensors = local = None
                return x

            def close(self):
                pipe.stream_stats = streamer.stats()
                streamer.close()
                index.close()

        return Blocks()

    # ---- VAE
    def _load_vae(self):
        import mlx.core as mx
        from mlx.utils import tree_map
        from shutter.families.h3image.model import VAE
        vae = VAE(layers=self.vae_layers, dim=self.vae_dim)
        W.load_into(vae, self.vae_c.load(list(self.vae_c.weight_map), nocache=True))
        vae.update(tree_map(lambda a: a.astype(mx.float32), vae.parameters()))   # float16 weights, float32 compute
        mx.eval(vae.parameters())
        return vae

    # ---- the whole request
    def generate(self, s: Settings, rung: str = "staged",
                 on_image: Optional[Callable[[int, int, np.ndarray], None]] = None) -> list[np.ndarray]:
        import mlx.core as mx
        from shutter.families.h3image.model import PDD_SIGMAS
        s = align(s)
        seeds = s.seeds
        self.meter.mark("encode")
        tags, moments, vis = None, [], None
        if s.images:
            from shutter.families.h3image import refs as R
            if not self.has_refs_weights():
                raise RuntimeError("this MiniMax-H3-Image download has no picture reader yet: download the model again "
                                   "to add it (about 1.5 GB)")
            pics = [R.load_reference(p) for p in s.images]
            self.report(0.0, "reading the reference pictures")
            vis = [R.resized(p, R.VISION_MIN_SHORT, R.VISION_MAX_SHORT) for p in pics]
            moments = self.ref_moments([R.resized(p, R.VAE_MIN_SHORT, R.VAE_MAX_SHORT) for p in pics])
            del pics
        self.report(0.0, "reading the prompt")
        if vis:
            text, truncated, tags = self.encode(s.prompt, vis)
        else:
            text, truncated = self.encode(s.prompt)
        del vis
        if truncated:
            s.notes.append(f"the prompt is longer than {MAX_TOKENS} tokens: the end was cut off")
        self.check()
        self.meter.mark("load")
        self.report(0.05, "loading the image model")
        dit, blocks, release = self.dit(rung)
        steps = len(PDD_SIGMAS) - 1
        lh, lw = s.height // 16, s.width // 16
        latents_out = []
        total = len(seeds) * steps
        done = 0
        try:
            text_rows = dit.refine_text(text)
            mx.eval(text_rows)
            del text
            for seed in seeds:
                refs = None
                if moments:
                    # one generator per request, in the released order: the posterior sample of every reference,
                    # the noise of the generated image, then each reference's t = 0.999 noise
                    from shutter.families.h3image import refs as R
                    g = TorchRandn(seed)
                    zs = [R.reference_latents(m, g.randn(24 * m.shape[1] * m.shape[2])) for m in moments]
                    lat = mx.array(g.randn(24 * lh * lw).reshape(24, lh, lw))
                    sig_r = np.float32(1.0) - np.float32(R.REFERENCE_T)
                    refs = [mx.array(((np.float32(1.0) - sig_r) * z + sig_r * g.randn(z.size).reshape(z.shape))
                                     .astype(np.float32)) for z in zs]
                else:
                    lat = mx.array(torch_randn(seed, 24 * lh * lw).reshape(24, lh, lw))
                fix = s.texture if (s.texture is not None and s.texture.active and not refs) else None
                if fix is not None:
                    from shutter.families.h3image import easyfix as EF
                    plan = EF.parse_plan(fix.plan, steps)
                    # its own generator (seed + 1), so the start noise stays the plain run's
                    gen = TorchRandn(seed + 1)
                    fresh = lambda: mx.array(gen.randn(24 * lh * lw).reshape(24, lh, lw))   # noqa: E731
                for k in range(steps):
                    self.check()
                    self.meter.mark(f"denoise:{seed}:{k}")
                    sig, nxt = PDD_SIGMAS[k], PDD_SIGMAS[k + 1]
                    hook = EF.steering_hook(fix, k, self.steer_stats) if fix is not None else None
                    v = dit.velocity(lat, text_rows, sig, nxt, blocks=blocks, refs=refs, text_tags=tags,
                                     block_hook=hook)
                    if fix is None:
                        lat = lat + v * (nxt - sig)
                    else:
                        lat = EF.step(lat, v, sig, nxt, plan[k], fix, fresh)
                    mx.eval(lat)
                    done += 1
                    self.report(0.08 + 0.84 * done / total,
                                f"step {done} of {total}" if len(seeds) == 1 else
                                f"image {seeds.index(seed) + 1} of {len(seeds)}, step {k + 1} of {steps}")
                latents_out.append(lat)
        finally:
            release()
        del dit, blocks
        gc.collect()
        mx.clear_cache()
        self.meter.mark("decode")
        self.report(0.93, "decoding")
        vae = self._load_vae()
        images = []
        for i, (seed, lat) in enumerate(zip(seeds, latents_out)):
            self.check()
            img = vae.decode(lat, check=self.check)
            img = (np.array(img) * 255).round().astype(np.uint8)
            images.append(img)
            if on_image:
                on_image(i, seed, img)
        del vae, latents_out
        gc.collect()
        mx.clear_cache()
        self.meter.close()
        return images
