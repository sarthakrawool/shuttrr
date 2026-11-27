"""Text to image with Z-Image Turbo on a small Mac.

Stages and what is in memory during each (`RUNGS` is the memory step table):

    encode   the Qwen3 prompt encoder, one layer at a time: the token embedding + one layer (~0.5 GB at 8-bit), never
             the whole 4 GB encoder. Its output (one [n, 2560] array per prompt) is kept; the encoder is gone.
    denoise  the DiT. resident / split / staged: the whole DiT (6.5 GB at 8-bit). stream_wide / stream_tight: the
             DiT's main stack of 30 blocks is read from the SSD as it runs, 8 or 2 blocks in memory at a time; the
             rest (embedders, refiners, final layer, ~0.8 GB) stays.
    decode   the VAE decoder (0.17 GB), one block at a time in row strips (mflux's low-memory decode). resident and
             split keep the DiT in memory meanwhile (and between requests); staged and the stream steps free it first.

Every step gives the same image, bit for bit: the same bytes go through the same ops. Several images in one request
(`count`) share one prompt encoding and one loaded DiT; image i uses seed + i.

The schedule is mflux's for Z-Image Turbo: `steps` linear sigmas from 1 to 1/steps, the resolution-dependent flow
shift (base 0.5 at 256 tokens, max 1.15 at 4096 tokens), Euler steps, no guidance (Turbo is guidance distilled).
`guidance` > 1 runs classifier-free guidance with the negative prompt, as mflux does for the base model.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from shutter.families.zimage import weights as W

GB = 1024**3

#: memory step -> how the pipeline runs (identical images on every step)
RUNGS: dict[str, dict[str, Any]] = {
    "resident": {"window": 0, "keep_dit": True},
    "split": {"window": 0, "keep_dit": True},
    "staged": {"window": 0, "keep_dit": False},
    "stream_wide": {"window": 8, "keep_dit": False},
    "stream_tight": {"window": 2, "keep_dit": False},
}

MAX_TOKENS = 512           # the prompt limit (diffusers / mflux max_sequence_length)
SIZE_MULTIPLE = 16
#: flow shift (mflux ModelConfig defaults; Z-Image Turbo sets requires_sigma_shift)
SHIFT_BASE, SHIFT_MAX, SHIFT_BASE_LEN, SHIFT_MAX_LEN = 0.5, 1.15, 256, 4096


@dataclass
class Settings:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 9
    guidance: float = 0.0
    seed: int = 42
    count: int = 1
    pad_prompt: bool = True          # pad the prompt to 512 tokens as mflux does
    loras: list = field(default_factory=list)   # [{"path", "strength"}] or paths (shutter.families.zimage.lora)
    notes: list[str] = field(default_factory=list)

    @property
    def seeds(self) -> list[int]:
        return [int(self.seed) + i for i in range(max(1, int(self.count)))]


def align(s: Settings) -> Settings:
    """Sizes to multiples of 16 (rounded down, as mflux does), at least 256."""
    for k in ("width", "height"):
        v = int(getattr(s, k))
        a = max(256, SIZE_MULTIPLE * (v // SIZE_MULTIPLE))
        if a != v:
            s.notes.append(f"{k} {v} is not a multiple of {SIZE_MULTIPLE}: used {a}")
        setattr(s, k, a)
    s.steps = max(1, int(s.steps))
    return s


def sigmas(steps: int, width: int, height: int):
    """mflux LinearScheduler for Z-Image Turbo: shifted linear sigmas plus a final 0 (float32)."""
    import mlx.core as mx
    s = mx.linspace(1.0, 1.0 / steps, steps).astype(mx.float32)
    m = (SHIFT_MAX - SHIFT_BASE) / (SHIFT_MAX_LEN - SHIFT_BASE_LEN)
    b = SHIFT_BASE - m * SHIFT_BASE_LEN
    mu = mx.array(m * width * height / 256 + b)
    shifted = mx.exp(mu) / (mx.exp(mu) + (1 / s - 1))
    return mx.concatenate([shifted, mx.zeros(1)])


class Tokenizer:
    """The Qwen tokenizer with Z-Image's chat template (enable_thinking=True, generation prompt), padded to 512."""

    def __init__(self, folder: Path):
        from tokenizers import Tokenizer as HFTokenizer
        self.tok = HFTokenizer.from_file(str(Path(folder) / "tokenizer.json"))
        self.pad_id = self.tok.token_to_id("<|endoftext|>")

    @staticmethod
    def template(prompt: str) -> str:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, prompt: str, pad: bool = True) -> tuple[np.ndarray, np.ndarray, bool]:
        ids = self.tok.encode(self.template(prompt), add_special_tokens=True).ids
        truncated = len(ids) > MAX_TOKENS
        ids = ids[:MAX_TOKENS]
        n = len(ids)
        if pad:
            ids = ids + [self.pad_id] * (MAX_TOKENS - n)
        mask = [1] * n + [0] * (len(ids) - n)
        return np.array([ids], dtype=np.int32), np.array([mask], dtype=np.int32), truncated


class Meter:
    """Seconds and MLX peak per stage."""

    def __init__(self):
        self.records: list[dict[str, Any]] = []
        self._name: Optional[str] = None
        self._t = 0.0

    def mark(self, name: str) -> None:
        import mlx.core as mx
        self.close()
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
        self._name, self._t = name, time.perf_counter()

    def close(self) -> None:
        import mlx.core as mx
        if self._name is None:
            return
        try:
            peak = int(mx.get_peak_memory())
        except Exception:
            peak = 0
        self.records.append({"stage": self._name, "seconds": round(time.perf_counter() - self._t, 3),
                             "peak_bytes": peak})
        self._name = None

    def summary(self) -> dict[str, Any]:
        secs: dict[str, float] = {}
        peaks: dict[str, int] = {}
        for r in self.records:
            k = r["stage"].split(":")[0]
            secs[k] = round(secs.get(k, 0.0) + r["seconds"], 3)
            peaks[k] = max(peaks.get(k, 0), r["peak_bytes"])
        return {"stage_seconds": secs, "stage_peak_bytes": peaks,
                "peak_bytes": max(peaks.values(), default=0)}


class ZImagePipeline:
    """One build folder (mflux layout). `pool` (shutter.core.pool.ModelPool or None) keeps the DiT warm between
    requests on the resident / split steps."""

    def __init__(self, model_dir: Path, model_id: str = "z-image", *, pool=None,
                 report: Optional[Callable[[float, str], None]] = None,
                 check_cancelled: Optional[Callable[[], None]] = None, meter: Optional[Meter] = None):
        self.dir = Path(model_dir)
        self.model_id = model_id
        self.pool = pool
        self.report = report or (lambda p, s: None)
        self.check = check_cancelled or (lambda: None)
        self.meter = meter or Meter()
        self.te = W.Component(self.dir / "text_encoder")
        self.dit_c = W.Component(self.dir / "transformer")
        self.vae_c = W.Component(self.dir / "vae")
        self.stream_stats: dict[str, Any] = {}
        # optional zimage.json in the build: {"dit": {...}, "te": {...}} config overrides (smaller test models)
        import json
        cfg_file = self.dir / "zimage.json"
        over = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
        from shutter.families.zimage.model import DIT_CONFIG, TE_CONFIG
        self.dit_cfg = {**DIT_CONFIG, **(over.get("dit") or {})}
        self.te_cfg = {**TE_CONFIG, **(over.get("te") or {})}
        self.te_layers_used = int(self.te_cfg["num_hidden_layers"]) - 1
        self.lora_info: list[dict[str, Any]] = []

    # ---- sizes (for estimates)
    def part_sizes(self) -> dict[str, int]:
        te_embed = W.part_bytes(self.te, lambda n: n.startswith("embed_tokens."))
        te_layer = W.part_bytes(self.te, lambda n: n.startswith("layers.0."))
        dit_total = W.part_bytes(self.dit_c, lambda n: True)
        dit_block = W.part_bytes(self.dit_c, lambda n: n.startswith("layers.0."))
        n_main = len({W.layer_of(n) for n in self.dit_c.weight_map if W.layer_of(n) is not None})
        vae = W.part_bytes(self.vae_c, lambda n: n.startswith("decoder."))
        return {"te_embed": te_embed, "te_layer": te_layer, "te_layers": self.te_layers_used, "dit": dit_total,
                "dit_block": dit_block, "dit_blocks": n_main, "vae": vae}

    # ---- prompt encoder, one layer at a time
    def encode(self, prompt: str, pad: bool = True):
        import mlx.core as mx
        from shutter.families.zimage.model import EncoderLayer, TextEncoder
        tok = Tokenizer(self.dir / "tokenizer")
        ids, mask, truncated = tok(prompt, pad=pad)
        n_valid = int(mask.sum())
        te = TextEncoder(build_layers=False, **self.te_cfg)
        W.load_into(te, self.te.load(self.te.names("embed_tokens.")))
        h, cmask, pos = te.embed(mx.array(ids), mx.array(mask))
        mx.eval(h, cmask, *pos)
        del te
        for i in range(self.te_layers_used):
            self.check()
            layer = EncoderLayer(**self.te_cfg)
            W.load_into(layer, W.strip(self.te.load(self.te.names(f"layers.{i}.")), f"layers.{i}."))
            h = layer(h, attention_mask=cmask, position_embeddings=pos)
            mx.eval(h)
            del layer
        out = TextEncoder.finish(h)[0, :n_valid, :]
        mx.eval(out)
        del h
        gc.collect()
        mx.clear_cache()
        return out, truncated

    # ---- DiT
    def _dit_key(self) -> str:
        return f"zimage/{self.model_id}/dit"

    def _load_dit(self, build_main: bool):
        import mlx.core as mx
        from shutter.families.zimage.model import ZImageTransformer
        dit = ZImageTransformer(build_main=build_main, **self.dit_cfg)
        names = [n for n in self.dit_c.weight_map if build_main or W.layer_of(n) is None]
        W.load_into(dit, self.dit_c.load(names))
        mx.eval(dit.parameters())
        return dit

    def dit(self, rung: str, loras: Optional[dict] = None):
        """(transformer, main_stack or None, release()) for this memory step, with `loras` (read by
        lora.load_loras) set on it; none clears any a warm model had from an earlier request."""
        from shutter.families.zimage import lora as L
        loras = loras or {}
        r = RUNGS[rung]
        if r["window"]:
            dit = self._load_dit(build_main=False)
            L.apply_resident(dit, loras, include_main=False)
            stack = self._streamed_main(dit, r["window"], loras)
            return dit, stack, (lambda: stack.close())
        size = self.dit_c.total_bytes()
        if self.pool is not None and r["keep_dit"]:
            dit = self.pool.get(self._dit_key(), lambda: self._load_dit(True), size)
        else:
            if self.pool is not None:
                self.pool.release(self._dit_key())   # a warm copy from a resident request: not kept on this step
            dit = self._load_dit(True)
        L.apply_resident(dit, loras)
        return dit, None, (lambda: None)

    def _streamed_main(self, dit, window: int, loras: Optional[dict] = None):
        """The 30-block main stack read from the SSD: two reusable block shells, `window` blocks in memory."""
        import re
        from shutter.core.stream import BlockStreamer, StreamedStack, group_names_by_block
        from shutter.families.zimage import lora as L
        from shutter.families.zimage.model import ZImageTransformerBlock
        loras = loras or {}
        index = self.dit_c.tensor_index()
        groups, _ = group_names_by_block(index.names(), pattern=r"^(layers)\.(\d+)\.")
        streamer = BlockStreamer(index, groups, window=window, readahead=max(1, window - 1), cyclic=True,
                                 nocache=False)
        sample = {k: v for k, v in self.dit_c.load(groups[0]).items()}
        shells = []
        for _ in range(2 if window > 1 else 1):
            c = self.dit_cfg
            sh = ZImageTransformerBlock(c["dim"], c["n_heads"], c["norm_eps"], c["qk_norm"])
            W.load_into(sh, W.strip(sample, "layers.0."))
            L.wrap(sh, L.BLOCK_TARGETS)       # LoRA slots always there (empty: the base output)
            shells.append(sh)
        del sample
        rx = re.compile(r"^layers\.\d+\.")
        stack = StreamedStack(shells, streamer, strip_prefix_fn=lambda n: L.shell_name(rx.sub("", n, count=1)))
        pipe = self

        def apply(shell, h, i):
            L.set_adapters(shell, L.BLOCK_TARGETS, loras, f"layers.{i}.")
            return shell(x=h, attn_mask=mask_[0], freqs_cis=mask_[1], t_emb=mask_[2])

        mask_: list = [None, None, None]

        class Main:
            def __call__(self, unified, mask, freqs, t_emb):
                mask_[:] = [mask, freqs, t_emb]
                return stack.run(unified, apply=apply)

            def close(self):
                pipe.stream_stats = streamer.stats()
                streamer.close()
                index.close()

        return Main()

    # ---- VAE
    def _load_vae(self):
        import mlx.core as mx
        from shutter.families.zimage.model import VAE
        vae = VAE()
        W.load_into(vae, self.vae_c.load(self.vae_c.names("decoder.")))
        mx.eval(vae.parameters())
        return vae

    # ---- the whole request
    def generate(self, s: Settings, rung: str = "staged",
                 on_image: Optional[Callable[[int, int, np.ndarray], None]] = None) -> list[np.ndarray]:
        """uint8 RGB images [h, w, 3], one per seed. `on_image(i, seed, image)` is called as each is decoded."""
        import mlx.core as mx
        from shutter.families.zimage.model import PRECISION
        from shutter.families.zimage.vae_decode import decode as vae_decode
        s = align(s)
        seeds = s.seeds
        cfg = float(s.guidance) > 1.0
        self.meter.mark("encode")
        self.report(0.0, "reading the prompt")
        cap, truncated = self.encode(s.prompt, pad=s.pad_prompt)
        if truncated:
            s.notes.append(f"the prompt is longer than {MAX_TOKENS} tokens: the end was cut off")
        neg = None
        if cfg:
            neg, _ = self.encode(s.negative_prompt if s.negative_prompt.strip() else " ", pad=s.pad_prompt)
        elif s.negative_prompt.strip():
            s.notes.append("negative prompt not used: Z-Image Turbo runs without guidance (set guidance above 1 to "
                           "use it)")
        self.check()
        self.meter.mark("load")
        self.report(0.04, "loading the image model")
        from shutter.families.zimage import lora as L
        loras, self.lora_info = L.load_loras(s.loras)
        dit, main, release = self.dit(rung, loras)
        sig = sigmas(s.steps, s.width, s.height)
        mx.eval(sig)
        latents_out = []
        total = len(seeds) * s.steps
        done = 0
        try:
            for seed in seeds:
                lat = mx.random.normal(shape=[16, 1, s.height // 8, s.width // 8], key=mx.random.key(seed)).astype(PRECISION)
                for t in range(s.steps):
                    self.check()
                    self.meter.mark(f"denoise:{seed}:{t}")
                    sigma_t = sig[t].reshape((1,))
                    timestep = mx.ones_like(sigma_t) - sigma_t
                    noise = dit(x=lat, timestep=timestep, cap_feats=cap, main_stack=main)
                    if neg is not None:
                        nn_ = dit(x=lat, timestep=timestep, cap_feats=neg, main_stack=main)
                        noise = noise + float(s.guidance) * (noise - nn_)
                    dt = (sig[t + 1] - sig[t]).astype(lat.dtype)
                    lat = lat + noise.astype(lat.dtype) * dt
                    mx.eval(lat)
                    done += 1
                    self.report(0.06 + 0.86 * done / total,
                                f"step {done} of {total}" if len(seeds) == 1 else
                                f"image {seeds.index(seed) + 1} of {len(seeds)}, step {t + 1} of {s.steps}")
                latents_out.append(lat)
        finally:
            release()
        if not RUNGS[rung]["keep_dit"] or self.pool is None:
            del dit
            main = None
            gc.collect()
            mx.clear_cache()
        del cap, neg
        self.meter.mark("decode")
        self.report(0.93, "decoding")
        vae = self._load_vae()
        images = []
        for i, (seed, lat) in enumerate(zip(seeds, latents_out)):
            self.check()
            img = vae_decode(vae, mx.squeeze(mx.expand_dims(lat, axis=0), axis=2))
            img = mx.clip(img / 2 + 0.5, 0, 1)
            img = np.array(mx.transpose(img, (0, 2, 3, 1)).astype(mx.float32))[0]
            img = (img * 255).round().astype(np.uint8)
            images.append(img)
            if on_image:
                on_image(i, seed, img)
        del vae, latents_out
        gc.collect()
        mx.clear_cache()
        self.meter.close()
        return images
