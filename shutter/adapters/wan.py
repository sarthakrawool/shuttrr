"""Wan family adapter: runs any Shuttrr Wan build (Wan 2.1 / 2.2, 1.3B to A14B, fine-tunes, LoRAs).

Configure a model with the folder of a build made by `shutter convert`:

    "models": {"wan2.2-ti2v-5b": {"adapter": "wan", "dir": "~/.shutter/models/Wan2.2-TI2V-5B-Shuttrr-8bit"}}

Request fields: prompt, negative_prompt, width/height, frames (4n+1; or duration_s x fps), steps, guidance, seed,
inputs.image (image-to-video; TI2V uses it when given). Params:

  shift         flow shift (default from the variant)
  scheduler     "unipc" (default) | "dpm++" | "euler"
  guidance_high A14B: guidance for the high-noise expert (default from the variant)
  loras         [{"path": file, "strength": 1.0, "expert": "both" | "high" | "low"}] or a list of paths; applied
                without touching the 8-bit weights and changeable per request
  vae_tile      true: tiled decode (less memory at 720p; NOT bit-identical, a slight seam blend)
  mode, efficiency, rung    memory steps (docs/MODES.md)

Memory steps for Wan (all bit-identical): resident keeps every expert warm; split holds one A14B expert at a time;
staged also frees the model before the decode; stream_wide / stream_tight read the blocks from the SSD with 8 / 2
blocks in memory. Estimates come from the build's real part sizes and the activation formulas in
families/wan/spec.py until a measured peak exists in measured.json (written by scripts/wan_check.py).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

from shutter.adapters.base import Adapter, RunContext
from shutter.core import modes as M
from shutter.core import parts as P
from shutter.types import GB, Job, JobRequest, MemoryEstimate



def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on") if not isinstance(v, bool) else v


class WanAdapter(Adapter):
    family = "wan"
    kind = "video"

    def __init__(self, model_id: str, model_dir: Path, options: Optional[dict[str, Any]] = None):
        super().__init__(model_id, model_dir, options)
        self._man: Optional[dict[str, Any]] = None

    # ---- build
    @property
    def manifest(self) -> dict[str, Any]:
        if self._man is None:
            f = self.model_dir / "shuttrr.json"
            if not f.exists():
                raise FileNotFoundError(f"{self.model_dir} is not a Shuttrr Wan build (no shuttrr.json). Convert "
                                        "the checkpoint first: shutter convert <source> <folder>")
            self._man = json.loads(f.read_text())
        return self._man

    @property
    def cfg(self) -> dict[str, Any]:
        return self.manifest["config"]

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        try:
            m = self.manifest
            d.update(label=m.get("label"), variant=m.get("variant"), task=m["config"].get("model_type"),
                     checked_on=m.get("checked_on", []), loras_merged=m.get("loras_merged", []))
        except Exception as e:  # noqa: BLE001
            d["error"] = str(e)
        return d

    def _opt(self, key, default):
        return self.options.get(key, default)

    # ---- request -> settings
    def settings_for(self, request: JobRequest):
        from shutter.families.wan.pipeline import Settings, align
        cfg = self.cfg
        p = request.params or {}
        # defaults: the model entry's options (preset_for gives small Macs 320x512), else the variant's own
        dh, dw = (self._opt("default_size", None) or cfg.get("default_size")
                  or ((704, 1280) if cfg["model_type"] == "ti2v" else (480, 832)))
        h = int(request.height or p.get("height") or dh)
        w = int(request.width or p.get("width") or dw)
        fps = float(request.fps or cfg.get("sample_fps", 16))
        frames = request.frames or (int(round(request.duration_s * fps)) + 1 if request.duration_s else None) \
            or int(self._opt("default_frames", 0) or cfg.get("frame_num", 81))
        gs = request.guidance if request.guidance is not None else p.get("guidance")
        default_gs = cfg.get("sample_guide_scale", 5.0)
        if gs is None:
            gs = tuple(default_gs) if isinstance(default_gs, list) else default_gs
        if cfg.get("dual_model") and not isinstance(gs, (list, tuple)):
            gs = (float(gs), float(p.get("guidance_high", gs)))
        s = Settings(prompt=request.prompt, negative_prompt=request.negative_prompt or "", height=h, width=w,
                     frames=int(frames), steps=int(request.steps or p.get("steps") or self._opt("default_steps", 0)
                               or cfg.get("sample_steps", 40)),
                     guidance=gs, shift=float(p.get("shift", cfg.get("sample_shift", 5.0))), seed=int(request.seed),
                     scheduler=str(p.get("scheduler", "unipc")), image=request.image,
                     vae_tile=_truthy(p.get("vae_tile", self._opt("vae_tile", False))), loras=list(p.get("loras") or []))
        return align(cfg, s)

    # ---- memory
    def part_sizes(self) -> list[P.PartSize]:
        from shutter.families.wan.spec import part_sizes
        man, cfg = self.manifest, self.cfg
        quant = None
        for n in ("dit", "dit_high", "dit_low"):
            if n in man["parts"]:
                quant = (man["parts"][n].get("quant") or {}).get("bits")
        ps = part_sizes(cfg, quant_bits=quant, image_encoder=cfg["model_type"] in ("i2v", "ti2v"))
        # real sizes from the build where known
        for x in ps:
            mp = man["parts"].get(x.name)
            if mp and x.role == "denoiser" and mp.get("total_bytes"):
                x.total, x.block, x.n_blocks = int(mp["total_bytes"]), int(mp["block_bytes"]), int(mp["n_blocks"])
            if x.name == "vae" and man["parts"].get("vae", {}).get("total_bytes"):
                x.total = int(man["parts"]["vae"]["total_bytes"])
        return ps

    def rung_estimates(self, request: JobRequest, s=None) -> dict[str, MemoryEstimate]:
        from shutter.families.wan.spec import stage_needs
        s = s or self.settings_for(request)
        cfg = self.cfg
        batch = 1 if (max(s.guidance) if isinstance(s.guidance, (list, tuple)) else s.guidance) <= 1.0 else 2
        ps = self.part_sizes()
        data = M.load_measured(self._opt("measured_path", None))
        size = M.size_key(s.height, s.width)
        lora_bytes = sum(Path(str(x.get("path") if isinstance(x, dict) else x)).expanduser().stat().st_size
                         for x in s.loras if Path(str(x.get("path") if isinstance(x, dict) else x)).expanduser().exists())
        out = {}
        for rid in M.RUNG_IDS:
            streamed = rid.startswith("stream")
            needs = stage_needs(cfg, s.height, s.width, s.frames, cfg_batch=batch, cross_kv_cached=not streamed,
                                tiled_decode=s.vae_tile)
            e = P.estimate(ps, needs, rid)
            peak = e["peak"] + lora_bytes
            note = f"wan {rid}: worst stage {e['worst']} ~{peak / GB:.1f} GB"
            m = M.measured_for(self.model_id, size, rid, data)
            # a measurement counts only for the same frames and the same decode (tiled peaks are lower)
            if m and m.get("frames", s.frames) == s.frames and bool(m.get("vae_tile", False)) == bool(s.vae_tile):
                peak = int(m["peak_bytes"]) + lora_bytes
                note = f"wan {rid}: measured {m['peak_bytes'] / GB:.1f} GB at {size}"
            else:
                # the build's own check (shuttrr.json checked_on) at the same size and as many or more frames is an
                # upper bound for this request: fewer frames never need more memory
                b = self._checked_peak(size, s.frames, rid, s.vae_tile)
                if b is not None and b[0] < peak:
                    peak = b[0] + lora_bytes
                    note = f"wan {rid}: measured {b[0] / GB:.1f} GB at {size} x {b[1]} frames (an upper bound)"
            out[rid] = MemoryEstimate(peak=int(peak), steady=int(e["steady"]), disk_scratch=int(1 * GB), note=note)
        # a lower memory step never needs more than the step above it
        prev = None
        for rid in M.RUNG_IDS:
            if prev is not None and out[rid].peak > out[prev].peak:
                out[rid].peak = out[prev].peak
                out[rid].note += f" (capped at {prev})"
            prev = rid
        return out

    def _checked_peak(self, size: str, frames: int, rid: str, vae_tile: bool) -> Optional[tuple[int, int]]:
        """(peak bytes, frames) of the smallest measured run in the build's checked_on at this size with at least
        `frames` frames, the same decode and this memory step; None when there is none."""
        best = None
        try:
            for c in self._checked_runs():
                f = int(c.get("frames", 0))
                g = (c.get("peaks_gb") or {}).get(rid)
                if c.get("size") != size or g is None or f < frames or bool(c.get("vae_tile", False)) != bool(vae_tile):
                    continue
                cand = (int(float(g) * GB), f)
                if best is None or f < best[1] or (f == best[1] and cand[0] < best[0]):
                    best = cand
        except Exception:
            return None
        return best

    def _checked_runs(self) -> list[dict[str, Any]]:
        """Measured runs for this build: its own shuttrr.json checked_on, plus the catalog's checked_on for this
        model id (runs measured after the build was published, e.g. on a 16 GB MacBook)."""
        runs = list(self.manifest.get("checked_on") or [])
        try:
            from shutter.core.catalog import CATALOG
            e = CATALOG.get(self.model_id) or {}
            if e.get("adapter") == "wan":
                runs += list(e.get("checked_on") or [])
        except Exception:
            pass
        return runs

    def measured_ladder(self) -> Optional[dict[str, Any]]:
        """Measured peak (and seconds) per memory step for this model's default request on this Mac, from its
        checked runs at exactly that size and frame count; None when it was not measured there."""
        try:
            s = self.settings_for(JobRequest(model=self.model_id, prompt="x"))
        except Exception:
            return None
        size = M.size_key(s.height, s.width)
        best = None
        for c in self._checked_runs():
            if (c.get("size") == size and int(c.get("frames", 0)) == s.frames and not c.get("vae_tile")
                    and c.get("peaks_gb")):
                if best is None or len(c["peaks_gb"]) > len(best["peaks_gb"]):
                    best = c
        if best is None:
            return None
        rungs = {}
        for rid, g in best["peaks_gb"].items():
            row: dict[str, Any] = {"peak_bytes": int(float(g) * GB), "identical": bool(best.get("identical", True))}
            sec = (best.get("seconds") or {}).get(rid)
            if sec is not None:
                row["seconds"] = float(sec)
            rungs[rid] = row
        return {"size": f"{size}, {s.frames} frames", "rungs": rungs}

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

    def choose_rung(self, request: JobRequest, ctx: Optional[RunContext] = None, s=None) -> dict[str, Any]:
        choice = M.resolve_mode(request.params, self.mode_defaults(ctx))
        ests = self.rung_estimates(request, s)
        budget = self.budget_for(request, ctx)
        if budget is not None and ctx is not None and (request.params or {}).get("budget_bytes") is None:
            # this model's own copies in the pool (a Load, or a warm DiT from the last request) are counted in the
            # estimates and are not free memory: the request reuses them, or drops them before loading another
            # copy (ModelSource._release_others). Without this a loaded 5B (5 GB) refused its own request.
            budget += self._own_pool_bytes(ctx)
        rid = M.fit_to_budget(ests, budget, choice.rung) if choice.step_down else choice.rung
        return {"choice": choice, "rung": rid, "estimates": ests, "budget": budget,
                "stepped_down_from": choice.rung if rid != choice.rung else None,
                "fits": budget is None or ests[rid].peak <= budget}

    def _own_pool_bytes(self, ctx: RunContext) -> int:
        prefix = f"wan/{self.model_id}/"
        try:
            return int(sum(e["size_bytes"] for e in ctx.models.stats()["entries"] if str(e["key"]).startswith(prefix)))
        except Exception:
            return 0

    def estimate(self, request: JobRequest) -> MemoryEstimate:
        s = self.settings_for(request)
        if self.cfg["model_type"] == "i2v" and not request.image:
            raise ValueError(f"{self.manifest.get('label', 'this model')} is image-to-video: give inputs.image")
        pick = self.choose_rung(request, None, s)
        est = pick["estimates"][pick["rung"]]
        est.note += f"; mode {pick['choice'].mode}, step {pick['rung']}"
        if pick["stepped_down_from"]:
            est.note += f" (stepped down from {pick['stepped_down_from']} to fit)"
        return est

    # ---- run
    def run(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        import mlx.core as mx

        from shutter.adapters.longcat import StageMeter, write_mp4
        from shutter.families.wan.pipeline import WanPipeline
        req = job.request
        t0 = time.time()
        s = self.settings_for(req)
        work = Path(ctx.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = Path(req.output) if req.output else work / f"{job.id}.mp4"
        if self._opt("cache_limit_gb", 2.0) is not None and hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(int(float(self._opt("cache_limit_gb", 2.0)) * GB))
        pick = self.choose_rung(req, ctx, s)
        rid = pick["rung"]
        if not pick["fits"]:
            from shutter.types import MemoryRefused
            raise MemoryRefused(f"{rid} needs ~{pick['estimates'][rid].peak / GB:.1f} GB, budget "
                                f"{pick['budget'] / GB:.1f} GB (what macOS can give right now): waiting for memory")
        if pick["stepped_down_from"]:
            s.notes.append(f"stepped down from {pick['stepped_down_from']} to {rid} to fit memory")
        meter = StageMeter(mx)
        meter.start("prep")
        pipe = WanPipeline(self.model_dir, self.manifest, cache=ctx.cache, report=ctx.report,
                           check_cancelled=ctx.check_cancelled, meter=meter)
        models = self.model_source(ctx, rid, room=lambda held: self._room_after_denoise(ctx, held, pick, rid))
        frames, info = pipe.generate(s, models, keep_warm=_truthy(self._opt("keep_warm", True)))
        if models.reused:
            info["reused_loaded"] = models.reused
        if models.dropped_held:
            s.notes.append("The loaded model was unloaded before the decode to make room; load it again to keep it.")
        meter.mark("write")
        ctx.report(0.97, "writing mp4")
        write_mp4(frames, None, out_path, fps=int(self.cfg.get("sample_fps", 16)))
        meter.close()
        summ = meter.summary()
        ctx.report(1.0, "done")
        return {"output": str(out_path), "frames": int(frames.shape[0]), "size": info["size"],
                "seconds": round(time.time() - t0, 1), "peak_memory_bytes": summ["peak_bytes"],
                "peak_bytes": summ["peak_bytes"], "mode": pick["choice"].mode,
                "efficiency": pick["choice"].efficiency, "rung": rid,
                "identical": not s.vae_tile, "estimate_peak_bytes": pick["estimates"][rid].peak,
                "budget_bytes": pick["budget"], "stages": meter.records, **summ, "info": info,
                "size_key": M.size_key(s.height, s.width), "variant": self.manifest.get("variant"),
                **({"stepped_down_from": pick["stepped_down_from"]} if pick["stepped_down_from"] else {})}

    # ---- load ahead of a request (POST /v1/models/<id>/load)
    def model_source(self, ctx: RunContext, rid: str, room=None):
        return held_source_class()(self.model_dir, self.manifest, self.model_id, ctx.models, rid,
                               nocache=_truthy(self._opt("stream_nocache", sys.platform == "darwin")), room=room)

    def _room_after_denoise(self, ctx: RunContext, held_bytes: int, pick: dict[str, Any], rid: str) -> bool:
        """At a step that frees the DiT before the decode: may a model loaded on purpose stay? Only when what macOS
        can give right now still covers this step's whole peak (the decode is part of it), so keeping it never
        pushes the decode into swap."""
        try:
            smp = ctx.memory_sample()
            avail = int(smp["available"]) if isinstance(smp, dict) else None
        except Exception:
            avail = None
        if avail is None:
            return False
        return avail >= int(pick["estimates"][rid].peak)

    def warm(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        """Load the DiT the way the default request on this Mac would (Fast mode picks the step), through the same
        pool keys a request uses, so the next request reuses it. Streamed steps load what they keep: the non-block
        part, the first blocks of the window, and the SSD reader thread."""
        from shutter.families.wan.pipeline import RUNGS
        req = job.request
        s = self.settings_for(req)
        ctx.report(0.0, "preparing")
        pick = self.choose_rung(req, ctx, s)
        rid = pick["rung"]
        if not pick["fits"]:
            from shutter.types import MemoryRefused
            raise MemoryRefused(f"{rid} needs ~{pick['estimates'][rid].peak / GB:.1f} GB, budget "
                                f"{pick['budget'] / GB:.1f} GB (what macOS can give right now): waiting for memory")
        models = self.model_source(ctx, rid)
        names = models.expert_names()
        first = names[:1] if RUNGS[rid]["one_expert"] else names   # A14B: the high-noise expert runs first
        for i, n in enumerate(first):
            ctx.check_cancelled()
            ctx.report(0.1 + 0.8 * i / len(first), "loading model")
            models.get(n)
        streamed = bool(RUNGS[rid]["window"])
        ctx.report(1.0, "loaded")
        note = ("The small always-loaded part is in memory; the rest of the model streams from the SSD during a "
                "request." if streamed else "The model is in memory.")
        if RUNGS[rid]["one_expert"] and len(names) > 1:
            note += " This step holds one of the two experts at a time; the other loads when a request needs it."
        return {"rung": rid, "residency": "streamed" if streamed else "resident", "note": note,
                **({"stepped_down_from": pick["stepped_down_from"]} if pick["stepped_down_from"] else {})}

    def can_load(self) -> tuple[bool, str]:
        try:
            self.manifest
        except Exception:
            return False, "This folder is not a Shuttrr Wan build."
        return True, ""


def _held_keys(pool) -> set:
    try:
        return {e["key"] for e in pool.stats()["entries"] if e.get("held")}
    except Exception:
        return set()


def _pool_size(pool, key: str) -> int:
    try:
        return int(next(e["size_bytes"] for e in pool.stats()["entries"] if e["key"] == key))
    except Exception:
        return 0


_HELD_SOURCE = None


def held_source_class():
    """ModelSource that respects a model loaded on purpose (pool entry `held`): a request reuses it, and a step
    that frees the DiT after denoising keeps it when there is room for the decode (`room(held_bytes)`), so it
    stays loaded for the next request. Without a held entry it behaves exactly like ModelSource. Built on first
    use so importing the adapter does not import the pipeline."""
    global _HELD_SOURCE
    if _HELD_SOURCE is not None:
        return _HELD_SOURCE
    from shutter.families.wan.pipeline import ModelSource

    class HeldAwareSource(ModelSource):
        def __init__(self, *a, room=None, **kw):
            super().__init__(*a, **kw)
            self.room = room
            self.reused: list[str] = []
            self.dropped_held: list[str] = []

        def get(self, name):
            key = self.key(name)
            if key not in self.reused and key in self.pool and key in _held_keys(self.pool):
                self.reused.append(key)
            return super().get(name)

        def finish(self, keep_warm: bool) -> None:
            if keep_warm and self.r["keep"]:
                return super().finish(keep_warm)
            held = _held_keys(self.pool)
            mine = [self.key(n) for n in self.expert_names()]
            keep = [k for k in mine if k in held]
            if keep and self.room is not None and not self.room(sum(_pool_size(self.pool, k) for k in keep)):
                self.dropped_held = keep
                keep = []
            for k in mine:
                if k in keep:
                    self._objs.pop(k, None)      # the pool keeps it; this request lets go of its reference
                else:
                    self._release(k)
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass

    _HELD_SOURCE = HeldAwareSource
    return _HELD_SOURCE
