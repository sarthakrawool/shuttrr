"""Text/image to video for any Wan build, stage by stage, with Shuttrr's memory steps.

Stages: prompt encode (umT5, one block at a time, float32) -> image encode (I2V / TI2V) -> denoise -> VAE decode.
The denoise loop is mlx-video's (flow matching, UniPC by default, guidance as one batch-2 pass, the A14B expert
switch at the boundary timestep, TI2V's frozen first frame), with model loading handed to `ModelSource`, which
decides per memory step what stays resident, what is released, and what streams from the SSD.

Nothing here changes the arithmetic between memory steps: every step gives the same frames for the same seed
(streamed blocks are bit-identical to resident ones; see loader.py). The only optional, not-identical setting is
`vae_tile` (tiled decode), which is off unless asked for.
"""

from __future__ import annotations

import gc
import hashlib
import html
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from shutter.families.wan import loader as L
from shutter.families.wan.spec import NEG_PROMPT

GB = 1024**3

#: what each memory step does for Wan (window: blocks kept when streaming)
RUNGS: dict[str, dict[str, Any]] = {
    "resident": dict(window=None, keep=True, one_expert=False),
    "split": dict(window=None, keep=True, one_expert=True),
    "staged": dict(window=None, keep=False, one_expert=True),
    "stream_wide": dict(window=8, keep=False, one_expert=True),
    "stream_tight": dict(window=2, keep=False, one_expert=True),
}


def clean_text(text: str) -> str:
    try:
        import ftfy
        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def best_output_size(w: int, h: int, dw: int, dh: int, max_area: int) -> tuple[int, int]:
    """Largest (w, h) aligned to (dw, dh) with the same aspect and w*h <= max_area (mlx-video's rule)."""
    ratio = w / h
    ow = math.sqrt(max_area * ratio)
    oh = max_area / ow
    ow1, oh1 = int(ow // dw * dw), int(max_area / (ow // dw * dw) // dh * dh)
    oh2, ow2 = int(oh // dh * dh), int(max_area / (oh // dh * dh) // dw * dw)
    if abs(ow1 / oh1 - ratio) < abs(ow2 / oh2 - ratio):
        return ow1, oh1
    return ow2, oh2


@dataclass
class Settings:
    prompt: str
    negative_prompt: str = ""
    height: int = 480
    width: int = 832
    frames: int = 81
    steps: int = 40
    guidance: Any = 5.0               # float, or (low-noise, high-noise) for A14B
    shift: float = 5.0
    seed: int = 42
    scheduler: str = "unipc"
    image: Optional[str] = None
    vae_tile: bool = False
    loras: list = field(default_factory=list)    # [{"path", "strength", "expert": "both"|"high"|"low"}]
    notes: list = field(default_factory=list)


def align(cfg: dict[str, Any], s: Settings) -> Settings:
    st, ps = cfg["vae_stride"], (1, 2, 2)
    ah, aw = ps[1] * st[1], ps[2] * st[2]
    h, w = max(ah, s.height // ah * ah), max(aw, s.width // aw * aw)
    if (h, w) != (s.height, s.width):
        s.notes.append(f"size aligned from {s.width}x{s.height} to {w}x{h} (multiples of {aw}x{ah})")
    if cfg.get("max_area") and h * w > cfg["max_area"]:
        w2, h2 = best_output_size(w, h, aw, ah, cfg["max_area"])
        s.notes.append(f"size reduced from {w}x{h} to {w2}x{h2} (this model's largest area)")
        w, h = w2, h2
    f = s.frames
    if (f - 1) % st[0]:
        f2 = (f - 1) // st[0] * st[0] + 1
        s.notes.append(f"frames {f} -> {f2} (Wan needs 4n+1)")
        f = f2
    s.height, s.width, s.frames = h, w, max(1, f)
    return s


class ModelSource:
    """Hands out the DiT expert(s) for the denoise loop according to the memory step, through the engine's pool
    (so the guard admits every load and evicts idle models when memory is short)."""

    def __init__(self, build: Path, manifest: dict[str, Any], model_id: str, pool, rung: str, *,
                 nocache: bool = False, on_block: Optional[Callable[[int, int], None]] = None):
        self.build, self.man, self.model_id, self.pool = Path(build), manifest, model_id, pool
        self.cfg = manifest["config"]
        self.rung = rung
        self.r = RUNGS[rung]
        self.nocache = nocache
        self.on_block = on_block
        self.current: Optional[str] = None
        self.loras: dict[str, dict[str, list]] = {}
        self.reloads = 0
        self._objs: dict[str, Any] = {}       # what this request got from the pool, by key

    def expert_names(self) -> list[str]:
        return [n for n in ("dit_high", "dit_low") if n in self.man["parts"]] or ["dit"]

    def key(self, name: str) -> str:
        w = self.r["window"]
        return f"wan/{self.model_id}/{name}/" + (f"stream{w}" if w else "resident")

    def size(self, name: str) -> int:
        p = self.man["parts"][name]
        w = self.r["window"]
        if w:
            return int(p.get("rest_bytes", 0) + min(w, p.get("n_blocks", 1)) * p.get("block_bytes", 0))
        return int(p.get("total_bytes", 0))

    def _release_others(self, name: str) -> None:
        """Before loading `name`: drop this model's copies this step does not use. That is a copy kept warm by a
        request at another step (a resident DiT while this request streams), and, when the step holds one expert
        at a time, the other expert, even one left warm by the previous request."""
        keep = {self.key(name)} if self.r["one_expert"] else {self.key(n) for n in self.expert_names()}
        prefix = f"wan/{self.model_id}/"
        keys = getattr(self.pool, "keys", None)
        cands = list(keys()) if callable(keys) else [self.key(n) for n in self.expert_names()]
        dropped = False
        for k in cands:
            if k.startswith(prefix) and k not in keep:
                self._release(k)
                dropped = True
        if dropped:
            gc.collect()
            import mlx.core as mx
            mx.clear_cache()

    def _release(self, key: str) -> None:
        """Give `key` back to the pool. A streamed model's reader thread is stopped too: it holds the blocks it
        read ahead, and would keep them (and itself) alive after the pool lets go."""
        m = self._objs.pop(key, None)
        self.pool.release(key)
        blocks = getattr(m, "_streamed_blocks", None) if m is not None else None
        if blocks is not None:
            blocks.close()

    def get(self, name: str):
        if self.current != name:
            self._release_others(name)
        part = self.man["parts"][name]
        w = self.r["window"]

        def loader():
            self.reloads += 1
            if w:
                m = L.load_streamed(self.build, self.cfg, part, window=w, nocache=self.nocache,
                                    loras={}, on_block=self.on_block)
            else:
                m = L.load_resident(self.build, self.cfg, part)
                L.apply_loras_resident(m, {})        # LoRA slots (empty): LoRAs change per request, no reload
            return m

        m = self.pool.get(self.key(name), loader, self.size(name), keep=True,
                          residency="streamed" if w else None)
        self._objs[self.key(name)] = m
        self.current = name
        lo = self.loras.get(name, {})
        if w:
            m.blocks.loras = lo
            m.blocks.on_block = self.on_block
        else:
            L.apply_loras_resident(m, lo)
        return m

    def finish(self, keep_warm: bool) -> None:
        """After denoising: what the step keeps for the next request."""
        if not (keep_warm and self.r["keep"]):
            for n in self.expert_names():
                self._release(self.key(n))
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()


def load_loras(specs: list, expert_names: list[str]) -> dict[str, dict[str, list]]:
    """Per expert: {module: [(A, B, scale)]}."""
    out: dict[str, dict[str, list]] = {n: {} for n in expert_names}
    for spec in specs or []:
        if isinstance(spec, str):
            spec = {"path": spec}
        path = Path(str(spec["path"])).expanduser()
        if not path.exists():
            raise ValueError(f"LoRA file not found: {path}")
        lo = L.read_lora(path, float(spec.get("strength", 1.0)))
        if not lo:
            raise ValueError(f"{path.name} has no Wan LoRA tensors Shuttrr recognises")
        target = spec.get("expert", "both")
        for n in expert_names:
            if target == "both" or n == "dit" or n == f"dit_{target}":
                out[n] = L.merge_loras(out[n], lo)
    return out


def check_lora_shapes(loras: dict[str, dict[str, list]], cfg: dict[str, Any]) -> None:
    d = cfg["dim"]
    for lo in loras.values():
        for mod, items in lo.items():
            for a, b, _ in items:
                want_in = cfg["ffn_dim"] if mod.endswith("ffn.fc2") else d
                if a.shape[1] != want_in:
                    raise ValueError(f"This LoRA was made for another Wan size (layer {mod} expects width {want_in}, "
                                     f"the LoRA has {a.shape[1]}).")


class WanPipeline:
    def __init__(self, build: Path, manifest: dict[str, Any], *, cache=None,
                 report: Callable[[float, str], None] = lambda f, s: None,
                 check_cancelled: Callable[[], None] = lambda: None, meter=None):
        self.build, self.man = Path(build), manifest
        self.cfg = manifest["config"]
        self.cache = cache
        self.report = report
        self.check_cancelled = check_cancelled
        self.meter = meter
        self.timings: dict[str, float] = {}

    def _mark(self, stage: str):
        if self.meter is not None:
            self.meter.mark(stage)

    # ---- prompt
    def _fingerprint(self) -> str:
        src = self.man.get("source", {}).get("fingerprint", "")
        return hashlib.sha1((src + json.dumps(self.man["parts"]["text_encoder"].get("files", []))).encode()).hexdigest()[:16]

    def _text_encoder_dir(self) -> Path:
        """The folder of the text encoder's files as shuttrr.json lists them: text_encoder/ (bf16) in older builds,
        text_encoder_q8/ (8-bit) in the published 1.3B since 0.2.6. Both load the same way."""
        files = (self.man.get("parts", {}).get("text_encoder") or {}).get("files") or []
        first = next((f for f in files if "/" in f), "")
        return self.build / (first.split("/")[0] if first else "text_encoder")

    def encode_prompt(self, text: str):
        import mlx.core as mx
        cleaned = clean_text(text)

        def compute():
            from transformers import AutoTokenizer

            from shutter.core.text_encode import encode_ids, lazy_weights
            from shutter.families.wan.model.text_encoder import T5Encoder
            tok = AutoTokenizer.from_pretrained(str(self.build / "tokenizer"))
            enc = tok(cleaned, max_length=512, padding="max_length", truncation=True, return_tensors="np")
            ids, mask = mx.array(enc["input_ids"]), mx.array(enc["attention_mask"])
            weights = lazy_weights(self._text_encoder_dir(), "model.safetensors.index.json", mx)
            n_blocks = len({k.split(".")[1] for k in weights if k.startswith("blocks.")})
            emb_shape = weights["token_embedding.weight"].shape
            umt5 = T5Encoder(vocab_size=emb_shape[0], dim=emb_shape[1], dim_attn=emb_shape[1],
                             dim_ffn=weights["blocks.0.ffn.fc1.weight"].shape[0],
                             num_heads=weights["blocks.0.pos_embedding.embedding.weight"].shape[1],
                             num_layers=n_blocks, num_buckets=weights["blocks.0.pos_embedding.embedding.weight"].shape[0],
                             shared_pos=False)
            x = encode_ids(umt5, weights, ids, mask, mx, device=mx.default_device(), upcast=mx.float32)
            n = int(mask.sum().item())
            out = x[0, :n]
            mx.eval(out)
            del umt5, weights
            gc.collect()
            mx.clear_cache()
            return {"emb": out}

        if self.cache is not None:
            d = self.cache.get_or_compute("wan-umt5", [self._fingerprint(), cleaned], compute, as_mlx=True)
            return d["emb"]
        return compute()["emb"]

    # ---- image
    def encode_image(self, s: Settings, g: dict[str, int]):
        """(y for I2V-A14B, or (z_img, mask, mask_tokens) for TI2V)."""
        import mlx.core as mx
        import numpy as np
        from PIL import Image
        vae_file = self.build / self.man["parts"]["vae"]["files"][0]
        w = mx.load(str(vae_file))
        if self.cfg["model_type"] == "i2v":
            from shutter.families.wan.model.vae import WanVAE
            img = Image.open(s.image).convert("RGB")
            sc = max(s.width / img.width, s.height / img.height)
            img = img.resize((round(img.width * sc), round(img.height * sc)), Image.LANCZOS)
            x1, y1 = (img.width - s.width) // 2, (img.height - s.height) // 2
            img = img.crop((x1, y1, x1 + s.width, y1 + s.height))
            arr = mx.array(np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0).transpose(2, 0, 1)
            video = mx.concatenate([arr[:, None], mx.zeros((3, s.frames - 1, s.height, s.width))], axis=1)
            vae = WanVAE(z_dim=16, encoder=True)
            vae.load_weights(list(w.items()), strict=False)
            z = vae.encode(video[None])[0]
            mx.eval(z)
            hl, wl = g["h"], g["w"]
            msk = mx.ones((1, s.frames, hl, wl))
            msk = mx.concatenate([msk[:, :1], mx.zeros((1, s.frames - 1, hl, wl))], axis=1)
            msk = mx.concatenate([mx.repeat(msk[:, :1], 4, axis=1), msk[:, 1:]], axis=1)
            msk = msk.reshape(1, msk.shape[1] // 4, 4, hl, wl).transpose(0, 2, 1, 3, 4)[0]
            y = mx.concatenate([msk, z], axis=0)
            mx.eval(y)
            del vae, w
            return y
        from shutter.families.wan.model.i2v_utils import build_i2v_mask, preprocess_image
        from shutter.families.wan.model.vae22 import Wan22VAEEncoder
        img = preprocess_image(s.image, s.width, s.height)
        vae = Wan22VAEEncoder(z_dim=self.cfg["vae_z_dim"])
        vae.load_weights(list(w.items()), strict=False)
        z = vae.encode(img)
        mx.eval(z)
        z = z[0].transpose(3, 0, 1, 2)
        mask, mask_tokens = build_i2v_mask((self.cfg["vae_z_dim"], g["t"], g["h"], g["w"]), (1, 2, 2))
        del vae, w
        return z, mask, mask_tokens

    # ---- decode
    def decode(self, latents, tiled: bool):
        import mlx.core as mx
        import numpy as np
        vae_file = self.build / self.man["parts"]["vae"]["files"][0]
        w = mx.load(str(vae_file))
        if self.cfg["vae_z_dim"] == 48:
            from shutter.families.wan.model.vae22 import Wan22VAEDecoder, denormalize_latents
            vae = Wan22VAEDecoder(z_dim=48)
            vae.load_weights([(k, v) for k, v in w.items() if not k.startswith(("encoder.", "conv1."))], strict=False)
            z = denormalize_latents(latents.transpose(1, 2, 3, 0)[None])
            video = vae.decode_tiled(z) if tiled else vae(z)
            mx.eval(video)
            v = np.array(video[0])
        else:
            from shutter.families.wan.model.vae import WanVAE
            vae = WanVAE(z_dim=16)
            vae.load_weights([(k, v) for k, v in w.items() if not k.startswith(("encoder.", "conv1."))], strict=False)
            video = vae.decode_tiled(latents[None]) if tiled else vae.decode(latents[None])
            mx.eval(video)
            v = np.array(video[0]).transpose(1, 2, 3, 0)
        del vae, w
        gc.collect()
        mx.clear_cache()
        return np.clip((v + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)

    # ---- the whole run
    def generate(self, s: Settings, models: ModelSource, keep_warm: bool = True) -> tuple[Any, dict[str, Any]]:
        import mlx.core as mx
        import numpy as np
        from shutter.families.wan.model.scheduler import (FlowDPMPP2MScheduler, FlowMatchEulerScheduler,
                                                        FlowUniPCScheduler)
        from shutter.families.wan.spec import latent_geometry
        cfg = self.cfg
        dual = cfg.get("dual_model", False)
        s = align(cfg, s)
        g = latent_geometry(cfg, s.height, s.width, s.frames)
        target = (cfg["vae_z_dim"], g["t"], g["h"], g["w"])
        seq_len = g["tokens"]
        gs = s.guidance
        if isinstance(gs, (list, tuple)):
            gs = tuple(float(x) for x in gs)
            cfg_off = all(x <= 1.0 for x in gs)
        else:
            gs = float(gs)
            cfg_off = gs <= 1.0
        experts = models.expert_names()
        models.loras = load_loras(s.loras, experts)
        check_lora_shapes(models.loras, cfg)

        t0 = time.time()
        self._mark("prompt")
        self.report(0.02, "encoding the prompt")
        ctx = self.encode_prompt(s.prompt)
        ctx_null = None if cfg_off else self.encode_prompt(s.negative_prompt or NEG_PROMPT)
        self.timings["prompt_s"] = round(time.time() - t0, 1)
        self.check_cancelled()

        y = z_img = mask = mask_tokens = None
        is_i2v = cfg["model_type"] == "i2v"
        blend = cfg["model_type"] == "ti2v" and bool(s.image)
        if is_i2v and not s.image:
            raise ValueError("this is an image-to-video model: give an image")
        if s.image:
            t1 = time.time()
            self._mark("image")
            self.report(0.05, "encoding the image")
            r = self.encode_image(s, g)
            if is_i2v:
                y = r
            else:
                z_img, mask, mask_tokens = r
            self.timings["image_s"] = round(time.time() - t1, 1)
            gc.collect()
            mx.clear_cache()

        mx.random.seed(int(s.seed))
        np.random.seed(int(s.seed) % (2**32))
        sched = {"euler": FlowMatchEulerScheduler, "dpm++": FlowDPMPP2MScheduler,
                 "unipc": FlowUniPCScheduler}.get(s.scheduler, FlowUniPCScheduler)(
            num_train_timesteps=cfg.get("num_train_timesteps", 1000))
        sched.set_timesteps(s.steps, shift=s.shift)
        noise = mx.random.normal(target)
        latents = (1.0 - mask) * z_img + mask * noise if blend else noise
        boundary = cfg.get("boundary", 0.0) * cfg.get("num_train_timesteps", 1000) if dual else None
        grid = (g["t"], g["h"] // 2, g["w"] // 2)
        batch = 1 if cfg_off else 2
        timesteps = sched.timesteps.tolist()

        emb_cache: dict[str, Any] = {}

        def prepared(name, model):
            """Per expert: embedded context, cross K/V (resident only), RoPE tables."""
            if name in emb_cache:
                return emb_cache[name]
            emb = model.embed_text([ctx] if cfg_off else [ctx, ctx_null])
            streamed = isinstance(model.blocks, L.StreamedBlocks)
            kv = None if streamed else model.prepare_cross_kv(emb)
            rcs = model.prepare_rope([grid] * batch)
            mx.eval(emb, rcs, *([x for pair in kv for x in pair] if kv else []))
            emb_cache[name] = (emb, kv, rcs)
            return emb_cache[name]

        t2 = time.time()
        self._mark("denoise")
        for i, tv in enumerate(timesteps):
            self.check_cancelled()
            name = ("dit_high" if tv >= boundary else "dit_low") if dual else "dit"
            if name not in emb_cache and models.current and models.current != name and models.r["one_expert"]:
                emb_cache.pop(models.current, None)
            model = models.get(name)
            emb, kv, rcs = prepared(name, model)
            if blend:
                tt = mask_tokens * tv
                pad = seq_len - tt.shape[1]
                if pad > 0:
                    tt = mx.concatenate([tt, mx.full((1, pad), tv)], axis=1)
                t_batch = tt if batch == 1 else mx.concatenate([tt, tt], axis=0)
            else:
                t_batch = mx.array([tv] * batch)
            y_arg = None if y is None else [y] * batch
            preds = model([latents] * batch, t=t_batch, context=emb, seq_len=seq_len, cross_kv_caches=kv,
                          y=y_arg, rope_cos_sin=rcs)
            if batch == 1:
                pred = preds[0]
            else:
                scale = (gs[1] if tv >= boundary else gs[0]) if (dual and isinstance(gs, tuple)) else (
                    gs if not isinstance(gs, tuple) else gs[0])
                pred = preds[1] + scale * (preds[0] - preds[1])
            del preds
            latents = sched.step(pred[None], tv, latents[None]).squeeze(0)
            if blend:
                latents = (1.0 - mask) * z_img + mask * latents
            del pred
            mx.eval(latents)
            self.report(0.08 + 0.8 * (i + 1) / len(timesteps), f"denoising, step {i + 1} of {len(timesteps)}")
        self.timings["denoise_s"] = round(time.time() - t2, 1)
        self.timings["seconds_per_step"] = round((time.time() - t2) / max(1, len(timesteps)), 2)
        emb_cache.clear()
        # drop this frame's references too, or the DiT (and its cross K/V) outlive finish() and stay in memory
        # through the decode at the steps that free it
        model = emb = kv = rcs = None  # noqa: F841
        self._mark("release")
        models.finish(keep_warm)
        self.check_cancelled()

        t3 = time.time()
        self._mark("decode")
        self.report(0.9, "decoding")
        frames = self.decode(latents, s.vae_tile)
        self.timings["decode_s"] = round(time.time() - t3, 1)
        info = {"size": [s.width, s.height], "frames": int(frames.shape[0]), "steps": s.steps,
                "guidance": list(gs) if isinstance(gs, tuple) else gs, "shift": s.shift, "seed": s.seed,
                "scheduler": s.scheduler, "notes": s.notes, "timings": dict(self.timings),
                "expert_loads": models.reloads, "loras": [str(Path(str(x.get("path") if isinstance(x, dict) else x)).name)
                                                          for x in s.loras or []]}
        return frames, info
