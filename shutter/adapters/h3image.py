"""MiniMax-H3-Image (text to image) on Shuttrr.

Configure a model with the folder `python -m shutter.families.h3image.convert` built:

    "models": {"minimax-h3-image": {"adapter": "h3-image", "dir": "~/.shutter/models/MiniMax-H3-Image-Shuttrr"}}

Request fields: prompt, width / height (multiples of 32, 256..2048, default 1024x1024), seed. The model always runs
its own six steps without guidance, so `steps`, `guidance` and `negative_prompt` are not used (a note says so).
Params:

  count         images in this request (1..8), seeds seed, seed + 1, ...; they share one prompt encoding
  style         a look preset that adds words around the prompt ("3d-animated"; request_options.IMAGE_STYLES)
  mode, efficiency, rung    memory steps (docs/MODES.md)

Memory steps (every one gives the same image, bit for bit; shutter/families/h3image/pipeline.py): resident / split
keep the 20B DiT warm (~22 GB at 8-bit), staged frees it before the decode, stream_wide / stream_tight read its
50-block stack from the SSD with 8 / 2 blocks in memory (~1 GB stays). The 32B prompt encoder always runs one layer at
a time. On Macs under 24 GB Fast mode uses the lowest step (as for Z-Image).

Output: an 8-bit RGB PNG per image.

Licence: MiniMax-H3-Image Community License Agreement. The UI must show "MiniMax-H3-Image" (§IV.2); the licence does
not cover the US, the EU, the UK and South Korea (§I.5).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from shutter.adapters.base import RunContext
from shutter.adapters.zimage import MAX_COUNT, SMALL_CACHE_GB, SMALL_MAC_BYTES, ZImageAdapter, _step_used, _truthy
from shutter.core import modes as M
from shutter.core import parts as P
from shutter.types import GB, Job, JobRequest, MemoryEstimate

#: build part sizes (bytes on disk) of the default build (8-bit DiT from the Q8_0 GGUF, Q4_K_M text encoder), used
#: before the folder is here
DEFAULT_PARTS = {"te_embed_row": 10_240, "te_layer": 372_000_000, "te_layers": 50, "dit": 22_400_000_000,
                 "dit_block": 433_000_000, "dit_blocks": 50, "dit_shell": 950_000_000, "vae": 810_000_000}


def image_tokens(h: int, w: int) -> int:
    return (h // 32) * (w // 32)


def stage_needs(h: int, w: int, text_tokens: int = 256) -> P.StageNeeds:
    """Working memory per stage (bytes), rounded up."""
    seq = image_tokens(h, w) + text_tokens
    denoise = int(seq * 520_000)                       # one block's activations at float32 residuals (+ modulation)
    vae_tokens = (h // 16) * (w // 16)
    decode = int(810_000_000 + vae_tokens * 1024 * 4 * 24)   # float32 copy of the decoder + its activations
    encode = int(text_tokens * 25600 * 4 * 4)          # one layer's activations
    return P.StageNeeds(encode=encode, denoise=denoise, decode=decode, carried=int(text_tokens * 5376 * 4 * 2))


class H3ImageAdapter(ZImageAdapter):
    family = "h3-image"
    kind = "image"
    extension = "png"

    # ---- build
    def problems(self) -> list[str]:
        out = []
        for sub in ("transformer", "text_encoder", "vae", "tokenizer"):
            if not (self.model_dir / sub).is_dir():
                out.append(f"no {sub}/ folder")
        if not (self.model_dir / "tokenizer" / "tokenizer.json").exists():
            out.append("no tokenizer/tokenizer.json")
        if not (self.model_dir / "h3image.json").exists():
            out.append("no h3image.json")
        return out

    def can_load(self) -> tuple[bool, str]:
        p = self.problems()
        return (False, "This folder is not a MiniMax-H3-Image build: " + ", ".join(p) + ".") if p else (True, "")

    def parts(self) -> dict[str, int]:
        if self._parts is None:
            try:
                from shutter.families.h3image.pipeline import H3ImagePipeline
                self._parts = H3ImagePipeline(self.model_dir, self.model_id).part_sizes()
            except Exception:
                self._parts = dict(DEFAULT_PARTS)
        return self._parts

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(label="MiniMax-H3-Image", variant="minimax-h3-image", task="t2i",
                 licence="MiniMax-H3-Image Community License Agreement")
        return d

    # ---- request -> settings
    def settings_for(self, request: JobRequest):
        from shutter.core.request_options import IMAGE_STYLES, styled_prompt
        from shutter.families.h3image.pipeline import Settings, align
        p = request.params or {}
        dh, dw = self._opt("default_size", None) or (1024, 1024)
        h = int(request.height or p.get("height") or dh)
        w = int(request.width or p.get("width") or dw)
        count = int(p.get("count", 1) or 1)
        if not 1 <= count <= MAX_COUNT:
            raise ValueError(f"count must be between 1 and {MAX_COUNT}")
        style = p.get("style") or None
        if style and style not in IMAGE_STYLES:
            raise ValueError(f"unknown style {style!r} (known: {', '.join(IMAGE_STYLES)})")
        s = Settings(prompt=styled_prompt(request.prompt or "", style), width=w, height=h, seed=int(request.seed),
                     count=count)
        steps = request.steps or p.get("steps")
        if steps and int(steps) != 6:
            s.notes.append("MiniMax-H3-Image always runs its own 6 steps: steps setting not used")
        gs = request.guidance if request.guidance is not None else p.get("guidance")
        if gs and float(gs) > 1.0:
            s.notes.append("MiniMax-H3-Image runs without guidance: guidance setting not used")
        if (request.negative_prompt or "").strip():
            s.notes.append("negative prompt not used: MiniMax-H3-Image runs without guidance")
        if p.get("loras"):
            raise ValueError("LoRAs are not supported for MiniMax-H3-Image yet")
        return align(s)

    # ---- memory
    def part_sizes(self) -> list[P.PartSize]:
        ps = self.parts()
        dit_rest = ps["dit_shell"]
        block = ps["dit_block"]
        return [P.PartSize("dit", "denoiser", dit_rest + ps["dit_blocks"] * block, n_blocks=ps["dit_blocks"],
                           block=block),
                P.PartSize("text_encoder", "text_encoder", ps["te_layers"] * ps["te_layer"],
                           n_blocks=ps["te_layers"], block=ps["te_layer"], embed=0),
                P.PartSize("vae", "decoder", ps["vae"])]

    def rung_estimates(self, request: JobRequest, s=None) -> dict[str, MemoryEstimate]:
        s = s or self.settings_for(request)
        needs = stage_needs(s.height, s.width)
        ps = self.part_sizes()
        data = M.load_measured(self._opt("measured_path", None))
        size = M.size_key(s.height, s.width)
        out = {}
        for rid in M.RUNG_IDS:
            e = P.estimate(ps, needs, rid)
            peak = int(e["peak"])
            note = f"h3-image {rid}: worst stage {e['worst']} ~{peak / GB:.1f} GB"
            m = M.measured_for(self.model_id, size, rid, data) or self._checked(size, rid)
            if m:
                peak = int(m["peak_bytes"])
                note = f"h3-image {rid}: measured {peak / GB:.1f} GB at {size}"
            out[rid] = MemoryEstimate(peak=peak, steady=int(e["steady"]), disk_scratch=int(0.1 * GB), note=note)
        prev = None
        for rid in M.RUNG_IDS:
            if prev is not None and out[rid].peak > out[prev].peak:
                out[rid].peak = out[prev].peak
                out[rid].note += f" (capped at {prev})"
            prev = rid
        return out

    def _own_pool_bytes(self, ctx: RunContext) -> int:
        prefix = f"h3image/{self.model_id}/"
        try:
            return int(sum(e["size_bytes"] for e in ctx.models.stats()["entries"] if str(e["key"]).startswith(prefix)))
        except Exception:
            return 0

    def pool_keys(self, request: JobRequest) -> list[str]:
        return [f"h3image/{self.model_id}/dit"]

    # ---- run
    def run(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        import mlx.core as mx

        from shutter.adapters.dummy import png_bytes
        from shutter.families.h3image.pipeline import H3ImagePipeline
        from shutter.families.zimage.pipeline import Meter
        req = job.request
        t0 = time.time()
        s = self.settings_for(req)
        work = Path(ctx.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = Path(req.output) if req.output else work / f"{job.id}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        small = 0 < self.total_bytes(req, ctx) < SMALL_MAC_BYTES
        lim = self._opt("cache_limit_gb", SMALL_CACHE_GB if small else 1.0)
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
        _step_used(ctx, rid)
        meter = Meter()
        keep = _truthy(self._opt("keep_warm", True))
        pipe = H3ImagePipeline(self.model_dir, self.model_id, pool=ctx.models if keep else None, report=ctx.report,
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
                  "steps": 6, "guidance": 0.0, "seeds": seeds, "count": len(seeds),
                  "seconds": round(time.time() - t0, 1), "peak_memory_bytes": summ["peak_bytes"],
                  "peak_bytes": summ["peak_bytes"], "mode": pick["choice"].mode,
                  "efficiency": pick["choice"].efficiency, "rung": rid, "identical": True,
                  "estimate_peak_bytes": pick["estimates"][rid].peak, "budget_bytes": pick["budget"],
                  "stages": meter.records, **summ, "size_key": f"{s.height}x{s.width}",
                  "variant": "minimax-h3-image", "notes": s.notes}
        if pipe.stream_stats:
            result["stream"] = pipe.stream_stats
        if pick["stepped_down_from"]:
            result["stepped_down_from"] = pick["stepped_down_from"]
        return result

    # ---- POST /v1/models/<id>/load
    def warm(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        from shutter.families.h3image.pipeline import RUNGS, H3ImagePipeline
        req = job.request
        s = self.settings_for(req)
        ctx.report(0.0, "preparing")
        pick = self.choose_rung(req, ctx, s)
        rid = pick["rung"]
        if not pick["fits"]:
            from shutter.types import MemoryRefused
            raise MemoryRefused(f"{rid} needs ~{pick['estimates'][rid].peak / GB:.1f} GB, budget "
                                f"{pick['budget'] / GB:.1f} GB (what macOS can give right now): waiting for memory")
        _step_used(ctx, rid)
        ctx.report(0.1, "loading model")
        pipe = H3ImagePipeline(self.model_dir, self.model_id, pool=ctx.models, check_cancelled=ctx.check_cancelled)
        res = pick["estimates"].get("resident")
        whole = not RUNGS[rid]["window"] and (rid != "staged" or pick["budget"] is None
                                              or (res is not None and res.peak <= pick["budget"]))
        if not whole:
            pipe.load_shell()
            ctx.report(1.0, "loaded")
            return {"rung": rid, "residency": "streamed",
                    "note": "The small always-loaded part of the image model is in memory; the rest streams from "
                            "the SSD during a request."}
        pipe.dit("resident")
        ctx.report(1.0, "loaded")
        return {"rung": rid, "residency": "resident", "note": "The image model is in memory."}
