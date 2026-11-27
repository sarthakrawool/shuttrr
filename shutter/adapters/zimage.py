"""Z-Image Turbo (text to image) on Shuttrr.

Configure a model with the folder of an mflux-saved build (the catalog's `z-image-turbo-8bit` is
mflux-community/z-image-turbo-mflux-q8, pinned):

    "models": {"z-image-turbo-8bit": {"adapter": "z-image", "dir": "~/.shutter/models/Z-Image-Turbo-mflux-q8"}}

Request fields: prompt, negative_prompt (used only with guidance > 1), width / height (multiples of 16), steps
(default 9), guidance (default 0: Turbo is guidance distilled), seed. Params:

  count         images in this request (1..8), seeds seed, seed + 1, ...; they share one prompt encoding and one
                loaded model. The first image is `output`, the others sit next to it as <name>.s<seed>.png
  pad_prompt    pad the prompt to 512 tokens before encoding (default true, as mflux does)
  style         a look preset that adds words around the prompt ("3d-animated"; request_options.IMAGE_STYLES)
  loras         [{"path": file, "strength": 1.0}] or a list of paths: LoRAs applied at run time on the 8-bit
                weights (diffusers / PEFT / kohya / ComfyUI names; shutter/families/zimage/lora.py)
  mode, efficiency, rung    memory steps (docs/MODES.md)

Memory steps (every one gives the same image, bit for bit; see shutter/families/zimage/pipeline.py):
resident / split keep the DiT warm between requests, staged frees it before the decode, stream_wide / stream_tight
read the DiT's 30-block main stack from the SSD with 8 / 2 blocks in memory. The prompt encoder always runs one
layer at a time. Options: default_size [h, w], default_steps, default_guidance, keep_warm (true), cache_limit_gb (1),
default_mode / default_efficiency.

Output: an 8-bit RGB PNG per image.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from shutter.adapters.base import Adapter, RunContext
from shutter.core import modes as M
from shutter.core import parts as P
from shutter.types import GB, Job, JobRequest, MemoryEstimate

MAX_COUNT = 8
#: Macs with less memory than this start Fast mode at the lowest memory step (0.3.0): on a 16 GB Mac with everyday
#: apps open the higher steps made swap grow even when their estimate fit what macOS called available
SMALL_MAC_BYTES = 24 * GB
#: Fast mode picks a step only if its peak plus this margin fits (the app, macOS and other apps keep moving)
FAST_MARGIN_BYTES = int(1.5 * GB)
#: build part sizes (bytes on disk) of mflux-community/z-image-turbo-mflux-q8, used when the folder is not here
DEFAULT_PARTS = {"te_embed": 414_868_480, "te_layer": 112_486_400, "te_layers": 35, "dit": 6_540_000_000,
                 "dit_block": 190_000_000, "dit_blocks": 30, "vae": 98_000_000}


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on") if not isinstance(v, bool) else v


def image_tokens(h: int, w: int) -> int:
    return (h // 16) * (w // 16)


def stage_needs(h: int, w: int, cfg: bool = False) -> P.StageNeeds:
    """Working memory per stage (bytes). Calibrated on a 16 GB M5 MacBook Pro (REPORT.md), rounded up."""
    tokens = image_tokens(h, w) + 544            # + the prompt (padded to a multiple of 32)
    denoise = int(tokens * 3840 * 2 * 26)        # activations of one block, with the attention scores in tiles
    decode = int(h * w * 256 * 2 * 3)            # the widest full-size layers of the strip decode
    encode = int(512 * 2560 * 4 * 12)            # float32 hidden states of 512 tokens, one layer at a time
    return P.StageNeeds(encode=encode, denoise=denoise, decode=decode,
                        carried=int(544 * 2560 * 2 * (2 if cfg else 1)))


class ZImageAdapter(Adapter):
    family = "z-image"
    kind = "image"
    extension = "png"

    def __init__(self, model_id: str, model_dir: Path, options: Optional[dict[str, Any]] = None):
        super().__init__(model_id, model_dir, options)
        self._parts: Optional[dict[str, int]] = None

    def _opt(self, key, default):
        return self.options.get(key, default)

    # ---- build
    def problems(self) -> list[str]:
        out = []
        for sub in ("transformer", "text_encoder", "vae", "tokenizer"):
            if not (self.model_dir / sub).is_dir():
                out.append(f"no {sub}/ folder")
        if not (self.model_dir / "tokenizer" / "tokenizer.json").exists():
            out.append("no tokenizer/tokenizer.json")
        return out

    def can_load(self) -> tuple[bool, str]:
        p = self.problems()
        return (False, "This folder is not a Z-Image build: " + ", ".join(p) + ".") if p else (True, "")

    def parts(self) -> dict[str, int]:
        if self._parts is None:
            try:
                from shutter.families.zimage.pipeline import ZImagePipeline
                self._parts = ZImagePipeline(self.model_dir, self.model_id).part_sizes()
            except Exception:
                self._parts = dict(DEFAULT_PARTS)
        return self._parts

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(label="Z-Image Turbo", variant="z-image-turbo", task="t2i")
        return d

    # ---- request -> settings
    def settings_for(self, request: JobRequest):
        from shutter.families.zimage.pipeline import Settings, align
        p = request.params or {}
        dh, dw = self._opt("default_size", None) or (1024, 1024)
        h = int(request.height or p.get("height") or dh)
        w = int(request.width or p.get("width") or dw)
        gs = request.guidance if request.guidance is not None else p.get("guidance", self._opt("default_guidance", 0.0))
        count = int(p.get("count", 1) or 1)
        if not 1 <= count <= MAX_COUNT:
            raise ValueError(f"count must be between 1 and {MAX_COUNT}")
        from shutter.core.request_options import IMAGE_STYLES, styled_prompt
        style = p.get("style") or None
        if style and style not in IMAGE_STYLES:
            raise ValueError(f"unknown style {style!r} (known: {', '.join(IMAGE_STYLES)})")
        s = Settings(prompt=styled_prompt(request.prompt or "", style), negative_prompt=request.negative_prompt or "", width=w, height=h,
                     steps=int(request.steps or p.get("steps") or self._opt("default_steps", 9)),
                     guidance=float(gs or 0.0), seed=int(request.seed), count=count,
                     pad_prompt=_truthy(p.get("pad_prompt", True)), loras=list(p.get("loras") or []))
        return align(s)

    # ---- memory
    def part_sizes(self) -> list[P.PartSize]:
        ps = self.parts()
        return [P.PartSize("dit", "denoiser", ps["dit"], n_blocks=ps["dit_blocks"], block=ps["dit_block"]),
                P.PartSize("text_encoder", "text_encoder", ps["te_embed"] + ps["te_layers"] * ps["te_layer"],
                           n_blocks=ps["te_layers"], block=ps["te_layer"], embed=ps["te_embed"]),
                P.PartSize("vae", "decoder", ps["vae"])]

    def rung_estimates(self, request: JobRequest, s=None) -> dict[str, MemoryEstimate]:
        s = s or self.settings_for(request)
        needs = stage_needs(s.height, s.width, cfg=s.guidance > 1.0)
        ps = self.part_sizes()
        data = M.load_measured(self._opt("measured_path", None))
        size = M.size_key(s.height, s.width)
        out = {}
        for rid in M.RUNG_IDS:
            e = P.estimate(ps, needs, rid)
            peak = int(e["peak"])
            note = f"z-image {rid}: worst stage {e['worst']} ~{peak / GB:.1f} GB"
            m = M.measured_for(self.model_id, size, rid, data) or self._checked(size, rid)
            if m:
                peak = int(m["peak_bytes"])
                note = f"z-image {rid}: measured {peak / GB:.1f} GB at {size}"
            from shutter.families.zimage.lora import lora_bytes
            lb = lora_bytes(s.loras)
            if lb:
                peak += lb
                note += f" + LoRAs {lb / GB:.2f} GB"
            out[rid] = MemoryEstimate(peak=peak, steady=int(e["steady"]), disk_scratch=int(0.1 * GB), note=note)
        prev = None
        for rid in M.RUNG_IDS:      # a lower memory step never needs more than the one above it
            if prev is not None and out[rid].peak > out[prev].peak:
                out[rid].peak = out[prev].peak
                out[rid].note += f" (capped at {prev})"
            prev = rid
        return out

    def _checked(self, size: str, rid: str) -> Optional[dict[str, Any]]:
        try:
            from shutter.core.catalog import CATALOG
            for c in (CATALOG.get(self.model_id) or {}).get("checked_on") or []:
                g = (c.get("peaks_gb") or {}).get(rid)
                if c.get("size") == size and g is not None:
                    return {"peak_bytes": int(float(g) * GB)}
        except Exception:
            pass
        return None

    def measured_ladder(self) -> Optional[dict[str, Any]]:
        try:
            from shutter.core.catalog import CATALOG
            s = self.settings_for(JobRequest(model=self.model_id, prompt="x"))
            size = M.size_key(s.height, s.width)
            for c in (CATALOG.get(self.model_id) or {}).get("checked_on") or []:
                if c.get("size") == size and c.get("peaks_gb"):
                    rungs = {}
                    for rid, g in c["peaks_gb"].items():
                        row: dict[str, Any] = {"peak_bytes": int(float(g) * GB), "identical": True}
                        sec = (c.get("seconds") or {}).get(rid)
                        if sec is not None:
                            row["seconds"] = float(sec)
                        rungs[rid] = row
                    return {"size": size, "rungs": rungs}
        except Exception:
            return None
        return None

    def mode_defaults(self, ctx: Optional[RunContext] = None) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for src in (self.options, (ctx.settings if ctx is not None else None) or {}):
            if src.get("default_mode"):
                d["mode"] = src["default_mode"]
            if src.get("default_efficiency") is not None:
                d["efficiency"] = src["default_efficiency"]
        return d

    def budget_for(self, request: JobRequest, ctx: Optional[RunContext] = None) -> Optional[int]:
        p = request.params or {}
        if p.get("budget_bytes") is not None:
            return int(p["budget_bytes"])
        try:
            if ctx is not None:
                smp = ctx.memory_sample()
                if isinstance(smp, dict) and smp.get("available"):
                    return int(smp["available"]) - int((ctx.settings or {}).get("headroom_bytes") or 0)
                return None
            from shutter.core.config import Config
            from shutter.core.memory import default_probe
            smp = default_probe().sample().to_dict()
            return int(smp["available"]) - int(Config.load().headroom_for(int(smp.get("total") or 0)))
        except Exception:
            return None

    def _own_pool_bytes(self, ctx: RunContext) -> int:
        prefix = f"zimage/{self.model_id}/"
        try:
            return int(sum(e["size_bytes"] for e in ctx.models.stats()["entries"] if str(e["key"]).startswith(prefix)))
        except Exception:
            return 0

    def total_bytes(self, request: JobRequest, ctx: Optional[RunContext] = None) -> int:
        p = request.params or {}
        if p.get("total_bytes") is not None:          # tests stand in another Mac
            return int(p["total_bytes"])
        try:
            if ctx is not None:
                smp = ctx.memory_sample()
                if isinstance(smp, dict) and smp.get("total"):
                    return int(smp["total"])
            from shutter.core.memory import default_probe
            return int(default_probe().sample().total)
        except Exception:
            return 0

    def choose_rung(self, request: JobRequest, ctx: Optional[RunContext] = None, s=None) -> dict[str, Any]:
        """The memory step for this request. Fast mode: on a Mac under SMALL_MAC_BYTES the lowest step; elsewhere
        the fastest step whose peak plus FAST_MARGIN_BYTES fits the budget, else the lowest. Low memory: the slider's
        step, stepping further down only if it does not fit. A pinned `rung` is used as is. `fits` (whether it may
        start now) compares the chosen step's peak with the budget itself, without the margin."""
        choice = M.resolve_mode(request.params, self.mode_defaults(ctx))
        ests = self.rung_estimates(request, s)
        budget = self.budget_for(request, ctx)
        if budget is not None and ctx is not None and (request.params or {}).get("budget_bytes") is None:
            budget += self._own_pool_bytes(ctx)     # our own warm DiT is reused, not free memory to find
        lowest = M.RUNG_IDS[-1]
        small = 0 < self.total_bytes(request, ctx) < SMALL_MAC_BYTES
        why = ""
        if choice.explicit_rung or not choice.step_down:
            rid = choice.rung
        elif choice.mode == "fast" and small:
            rid, why = lowest, "small Mac"
        elif choice.mode == "fast":
            rid = M.fit_to_budget(ests, None if budget is None else budget - FAST_MARGIN_BYTES, choice.rung)
        else:
            rid = M.fit_to_budget(ests, budget, choice.rung)
        return {"choice": choice, "rung": rid, "estimates": ests, "budget": budget,
                "stepped_down_from": choice.rung if rid != choice.rung else None, "why": why,
                "fits": budget is None or ests[rid].peak <= budget}

    def pool_keys(self, request: JobRequest) -> list[str]:
        return [f"zimage/{self.model_id}/dit"]

    def estimate(self, request: JobRequest) -> MemoryEstimate:
        s = self.settings_for(request)
        pick = self.choose_rung(request, None, s)
        est = pick["estimates"][pick["rung"]]
        est.note += f"; mode {pick['choice'].mode}, step {pick['rung']}"
        if pick["stepped_down_from"]:
            est.note += f" (stepped down from {pick['stepped_down_from']} to fit)"
        return est

    # ---- run
    @staticmethod
    def image_path(output: Path, seeds: list[int], i: int) -> Path:
        return output if i == 0 else output.with_name(f"{output.stem}.s{seeds[i]}{output.suffix or '.png'}")

    def run(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        import mlx.core as mx

        from shutter.adapters.dummy import png_bytes
        from shutter.families.zimage.pipeline import Meter, ZImagePipeline
        req = job.request
        t0 = time.time()
        s = self.settings_for(req)
        work = Path(ctx.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = Path(req.output) if req.output else work / f"{job.id}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lim = self._opt("cache_limit_gb", 1.0)
        if lim is not None and hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(int(float(lim) * GB))
        pick = self.choose_rung(req, ctx, s)
        rid = pick["rung"]
        if not pick["fits"]:
            from shutter.types import MemoryRefused
            raise MemoryRefused(f"{rid} needs ~{pick['estimates'][rid].peak / GB:.1f} GB, budget "
                                f"{pick['budget'] / GB:.1f} GB (what macOS can give right now): waiting for memory")
        if pick["why"] == "small Mac":
            s.notes.append(f"{rid}: Macs under {SMALL_MAC_BYTES // GB} GB use the lowest memory step in Fast mode")
        elif pick["stepped_down_from"]:
            s.notes.append(f"stepped down from {pick['stepped_down_from']} to {rid} to fit memory")
        meter = Meter()
        keep = _truthy(self._opt("keep_warm", True))
        pipe = ZImagePipeline(self.model_dir, self.model_id, pool=ctx.models if keep else None, report=ctx.report,
                              check_cancelled=ctx.check_cancelled, meter=meter)
        seeds = s.seeds
        outputs: list[str] = []

        def save(i: int, seed: int, img) -> None:
            p = self.image_path(out_path, seeds, i)
            p.write_bytes(png_bytes(img))
            outputs.append(str(p))

        pipe.generate(s, rid, on_image=save)
        summ = meter.summary()
        ctx.report(1.0, "done")
        result = {"output": outputs[0], "outputs": outputs, "kind": "image", "width": s.width, "height": s.height,
                  "steps": s.steps, "guidance": s.guidance, "seeds": seeds, "count": len(seeds),
                  "seconds": round(time.time() - t0, 1), "peak_memory_bytes": summ["peak_bytes"],
                  "peak_bytes": summ["peak_bytes"], "mode": pick["choice"].mode,
                  "efficiency": pick["choice"].efficiency, "rung": rid, "identical": True,
                  "estimate_peak_bytes": pick["estimates"][rid].peak, "budget_bytes": pick["budget"],
                  "stages": meter.records, **summ, "size_key": M.size_key(s.height, s.width),
                  "variant": "z-image-turbo", "notes": s.notes}
        if pipe.stream_stats:
            result["stream"] = pipe.stream_stats
        if pipe.lora_info:
            result["loras"] = pipe.lora_info
        if pick["stepped_down_from"]:
            result["stepped_down_from"] = pick["stepped_down_from"]
        return result

    # ---- POST /v1/models/<id>/load
    def warm(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        """Load the DiT the way the default request would (Fast picks the step). On resident / split it stays in the
        pool for the next request; on the lower steps nothing is kept between requests, so there is nothing to load."""
        from shutter.families.zimage.pipeline import RUNGS, ZImagePipeline
        req = job.request
        s = self.settings_for(req)
        ctx.report(0.0, "preparing")
        pick = self.choose_rung(req, ctx, s)
        rid = pick["rung"]
        if not pick["fits"]:
            from shutter.types import MemoryRefused
            raise MemoryRefused(f"{rid} needs ~{pick['estimates'][rid].peak / GB:.1f} GB, budget "
                                f"{pick['budget'] / GB:.1f} GB (what macOS can give right now): waiting for memory")
        if not RUNGS[rid]["keep_dit"]:
            ctx.report(1.0, "ready")
            return {"rung": rid, "residency": "streamed" if RUNGS[rid]["window"] else "resident",
                    "note": "On this memory step the model loads for each request and is freed after it, so there "
                            "is nothing to keep loaded."}
        ctx.report(0.1, "loading model")
        pipe = ZImagePipeline(self.model_dir, self.model_id, pool=ctx.models, check_cancelled=ctx.check_cancelled)
        pipe.dit(rid)
        ctx.report(1.0, "loaded")
        return {"rung": rid, "residency": "resident", "note": "The image model is in memory."}
