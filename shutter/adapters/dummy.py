"""A fake adapter for tests and for trying Shutter without a model.

Params (all optional, in JobRequest.params):
  peak_gb       memory estimate of the resident (Fast) rung, reported to the guard (default 1.0)
  component_gb  size of the fake component loaded through ctx.models (default 0.5)
  steps         number of fake denoise steps (default 5)
  step_s        seconds per step (default 0.2)
  fail          error text: raise RuntimeError(fail) at step `fail_at` (default 0)
  keep          keep the fake component warm in the pool (default True)
  mode, efficiency, rung   modes (docs/MODES.md); each rung scales the peak down and the step time up the way
                the LongCat ladder does, so the UI's slider can be tried without a model
  budget_bytes  memory budget for Fast mode's step-down (else the run context's available - headroom)
  save_latents  directory: write a deterministic latents_denorm.npy (same for every rung: the rungs are
                "bit-identical"), so the test chain's comparisons can be exercised without a model
  media         true: write a real mp4 (ffmpeg needed) of `frames` (default 8) frames at width x height
                (default 64x64) and `fps` (default 8), with inputs.audio muxed in when given; otherwise the
                output is a small JSON file (the default, no ffmpeg needed)

The "dummy-image" family (DummyImageAdapter, kind "image") writes a real PNG of width x height (default
64x64), a deterministic gradient from the seed and prompt. Configure it like any model:
`"models": {"dummy-image": {"adapter": "dummy-image"}}`. In stream rungs both report their fake
component as "streamed" (GET /v1/memory).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import time
import zlib
from pathlib import Path
from typing import Any, Optional

import numpy as np

from shutter.adapters.base import Adapter, RunContext
from shutter.core import modes as M
from shutter.types import GB, Job, JobRequest, MemoryEstimate

# (peak factor, step time factor) per rung: same shape as LongCat's (REPORT.md numbers, rounded)
RUNG_FACTORS = {"resident": (1.0, 1.0), "split": (0.94, 1.02), "staged": (0.92, 1.03),
                "stream_wide": (0.50, 1.10), "stream_tight": (0.42, 1.18)}


class DummyAdapter(Adapter):
    family = "dummy"
    kind = "video"
    needs_model_dir = False

    def _p(self, req: JobRequest, key: str, default: Any) -> Any:
        return req.params.get(key, self.options.get(key, default))

    def pool_keys(self, request: JobRequest) -> list[str]:
        return [f"{self.model_id}/dit"]

    # ---- modes
    def rung_estimates(self, request: JobRequest) -> dict[str, MemoryEstimate]:
        peak = float(self._p(request, "peak_gb", 1.0))
        comp = float(self._p(request, "component_gb", 0.5))
        out = {}
        for rid in M.RUNG_IDS:
            f = RUNG_FACTORS[rid][0]
            out[rid] = MemoryEstimate(peak=int(peak * f * GB), steady=int(comp * min(1.0, f * 1.2) * GB),
                                      note=f"dummy estimate ({rid})")
        return out

    def _budget(self, request: JobRequest, ctx: Optional[RunContext]) -> Optional[int]:
        b = self._p(request, "budget_bytes", None)
        if b is not None:
            return int(b)
        if ctx is None:
            return None
        try:
            s = ctx.memory_sample() or {}
        except Exception:
            return None
        if not s.get("available"):
            return None
        return int(s["available"]) - int((ctx.settings or {}).get("headroom_bytes") or 0)

    def choose(self, request: JobRequest, ctx: Optional[RunContext] = None) -> dict[str, Any]:
        defaults = {k[len("default_"):]: v for k, v in self.options.items() if k in ("default_mode", "default_efficiency")}
        if ctx is not None:
            for k in ("default_mode", "default_efficiency"):
                if (ctx.settings or {}).get(k) is not None:
                    defaults[k[len("default_"):]] = ctx.settings[k]
        choice = M.resolve_mode(request.params, defaults)
        ests = self.rung_estimates(request)
        budget = self._budget(request, ctx)
        rid = M.fit_to_budget(ests, budget, choice.rung) if choice.step_down else choice.rung
        return {"choice": choice, "rung": rid, "estimates": ests, "budget": budget,
                "stepped_down_from": choice.rung if rid != choice.rung else None}

    def estimate(self, request: JobRequest) -> MemoryEstimate:
        pick = self.choose(request)
        est = pick["estimates"][pick["rung"]]
        if pick["rung"] == "resident" and pick["choice"].mode == "fast":
            est.note = "dummy estimate"
        return est

    def warm(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        """POST /v1/models/<id>/load: put the fake component in the pool, as a request would."""
        req = job.request
        comp = int(float(self._p(req, "component_gb", 0.5)) * GB)
        rid = self.choose(req, ctx)["rung"]
        ctx.report(0.0, "loading")
        residency = "streamed" if rid.startswith("stream") else "resident"
        step_s = float(self._p(req, "load_s", 0.0))
        if step_s > 0:
            (ctx.settings.get("sleep") or time.sleep)(step_s)
        ctx.check_cancelled()
        try:
            ctx.models.get(f"{self.model_id}/dit", lambda: {"weights": "fake", "rung": rid}, comp,
                           residency=residency)
        except TypeError:  # a pool without residency (test doubles)
            ctx.models.get(f"{self.model_id}/dit", lambda: {"weights": "fake", "rung": rid}, comp)
        ctx.report(1.0, "loaded")
        return {"rung": rid, "note": "loaded the test component"}

    def on_step(self, job: Job, step: int, ctx: RunContext) -> None:
        """Hook for tests: called before each step."""

    default_name = "dummy.json"

    def write_output(self, req: JobRequest, out: Path, steps: int, emb0: float, component: bool) -> dict[str, Any]:
        """Write the finished file; returns extra result fields."""
        if _truthy(self._p(req, "media", False)):
            w, h = _size(req, 64, 64)
            n = int(req.frames or self._p(req, "frames", 8))
            fps = float(req.fps or self._p(req, "fps", 8))
            write_test_mp4(out, gradient(w, h, req.seed, req.prompt, n), fps, req.audio)
            return {"width": w, "height": h, "frames": n, "fps": fps, "kind": "video"}
        out.write_text(json.dumps({
            "model": self.model_id, "prompt": req.prompt, "seed": req.seed, "steps": steps,
            "emb0": emb0, "component": component,
        }))
        return {}

    def run(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        req = job.request
        steps = int(self._p(req, "steps", 5))
        fail = self._p(req, "fail", None)
        fail_at = int(self._p(req, "fail_at", 0))
        comp = int(float(self._p(req, "component_gb", 0.5)) * GB)
        keep = bool(self._p(req, "keep", True))
        pick = self.choose(req, ctx)
        rid = pick["rung"]
        step_s = float(self._p(req, "step_s", 0.2)) * RUNG_FACTORS[rid][1]
        stages: list[dict[str, Any]] = []
        clock = time.perf_counter

        t_stage = clock()
        ctx.report(0.0, "loading" if not pick["stepped_down_from"] else f"loading (stepped down to {rid})")
        residency = "streamed" if rid.startswith("stream") else "resident"
        try:
            model = ctx.models.get(f"{self.model_id}/dit", lambda: {"weights": "fake", "rung": rid}, comp, keep=keep,
                                   residency=residency)
        except TypeError:  # a pool without residency (test doubles)
            model = ctx.models.get(f"{self.model_id}/dit", lambda: {"weights": "fake", "rung": rid}, comp, keep=keep)
        emb = ctx.cache.get_or_compute(
            "dummy-prompt", [self.model_id, req.prompt, req.seed],
            lambda: {"emb": np.full((4,), float(len(req.prompt)), dtype=np.float32)},
        )
        peak = pick["estimates"][rid].peak
        stages.append({"stage": "encode", "segment": 1, "seconds": round(clock() - t_stage, 3),
                       "peak_bytes": int(peak * 0.3)})
        sleep = ctx.settings.get("sleep") or time.sleep
        t0 = clock()
        for i in range(steps):
            ctx.check_cancelled()
            self.on_step(job, i, ctx)
            if fail and i >= fail_at:
                raise RuntimeError(str(fail))
            ts = clock()
            if step_s > 0:
                sleep(step_s)
            stages.append({"stage": "step", "segment": 1, "step": i + 1, "seconds": round(clock() - ts, 3),
                           "peak_bytes": int(peak)})
            ctx.report((i + 1) / max(1, steps), f"denoise {i + 1}/{steps}" + ("" if rid == "resident" else f" ({rid})"))
        ctx.check_cancelled()
        if fail and steps <= fail_at:
            raise RuntimeError(str(fail))

        ts = clock()
        out = Path(req.output or (ctx.work_dir / self.default_name))
        out.parent.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {"output": os.fspath(out), "steps": steps, "denoise_s": clock() - t0}
        result.update(self.write_output(req, out, steps, float(emb["emb"][0]), bool(model)))
        lat = self._p(req, "save_latents", None)
        if lat:
            d = Path(os.path.expanduser(str(lat)))
            d.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(int(req.seed))
            arr = rng.standard_normal((1, 16, 3, 4, 4)).astype(np.float32) + float(len(req.prompt))
            np.save(d / "latents_denorm.npy", arr)
            result["latents"] = [str(d / "latents_denorm.npy")]
            result["latents_dir"] = str(d)
        stages.append({"stage": "decode", "segment": 1, "seconds": round(clock() - ts, 3),
                       "peak_bytes": int(peak * (0.4 if rid != "resident" else 0.9))})
        secs: dict[str, float] = {}
        peaks: dict[str, int] = {}
        for r in stages:
            k = "denoise" if r["stage"] == "step" else r["stage"]
            secs[k] = round(secs.get(k, 0.0) + r["seconds"], 3)
            peaks[k] = max(peaks.get(k, 0), r["peak_bytes"])
        result.update({"mode": pick["choice"].mode, "efficiency": pick["choice"].efficiency, "rung": rid,
                       "identical": M.rung(rid).get("identical", True), "stages": stages, "stage_seconds": secs,
                       "stage_peak_bytes": peaks, "step_seconds": [r["seconds"] for r in stages if r["stage"] == "step"],
                       "estimate_peak_bytes": peak, "budget_bytes": pick["budget"], "size_key": "dummy"})
        if pick["stepped_down_from"]:
            result["stepped_down_from"] = pick["stepped_down_from"]
        return result


class DummyImageAdapter(DummyAdapter):
    """Image-kind dummy: writes a PNG (for the OpenAI Images shim and image clients)."""

    family = "dummy-image"
    kind = "image"
    default_name = "dummy.png"

    def write_output(self, req: JobRequest, out: Path, steps: int, emb0: float, component: bool) -> dict[str, Any]:
        w, h = _size(req, 64, 64)
        frame = gradient(w, h, req.seed, req.prompt, 1)[0]
        if req.image and os.path.isfile(req.image):  # an edit: blend in the input's mean colour
            frame = (frame.astype(np.uint16) + _mean_rgb(req.image)) // 2
            frame = frame.astype(np.uint8)
        out.write_bytes(png_bytes(frame))
        return {"width": w, "height": h, "kind": "image"}


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _size(req: JobRequest, dw: int, dh: int) -> tuple[int, int]:
    w = int(req.width or 0)
    h = int(req.height or 0)
    size = req.params.get("size")
    if (not w or not h) and isinstance(size, str) and "x" in size:  # "HxW", like LongCat's params.size
        hh, ww = (int(v) for v in size.lower().split("x", 1))
        w, h = ww, hh
    return (w or dw), (h or dh)


def gradient(w: int, h: int, seed: int, prompt: str, n: int) -> np.ndarray:
    """n deterministic uint8 RGB frames [n, h, w, 3]: a gradient whose tint follows seed and prompt."""
    rng = np.random.default_rng(abs(int(seed)) + 7919 * len(prompt or ""))
    tint = rng.integers(0, 256, size=3)
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    frames = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        shift = 255.0 * i / max(1, n)
        frames[i, :, :, 0] = ((x + shift + tint[0]) % 256).astype(np.uint8)
        frames[i, :, :, 1] = ((y + tint[1]) % 256).astype(np.uint8)
        frames[i, :, :, 2] = np.uint8(tint[2])
    return frames


def png_bytes(rgb: np.ndarray) -> bytes:
    """A PNG (8-bit RGB) from a [h, w, 3] uint8 array, stdlib only."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _mean_rgb(path: str) -> np.ndarray:
    try:
        from PIL import Image  # optional
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0).astype(np.uint16)
    except Exception:
        return np.array([128, 128, 128], dtype=np.uint16)


def write_test_mp4(out: Path, frames: np.ndarray, fps: float, audio: Optional[str] = None) -> None:
    """Encode uint8 RGB frames (and optionally an audio file) as an mp4 with ffmpeg."""
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise RuntimeError("dummy media=true needs ffmpeg on PATH")
    n, h, w, _ = frames.shape
    cmd = [ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-"]
    if audio:
        cmd += ["-i", audio, "-shortest", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)]
    subprocess.run(cmd, input=frames.tobytes(), check=True, capture_output=True)
