"""One Shutter request from start to finish, written against plain callbacks so it runs (and is tested)
without ComfyUI. The node in nodes.py supplies ComfyUI's progress bar, interrupt flag and model
unloading as the callbacks.

Order of work:
  1. GET /v1/memory?model=...  - when `fits` is false, call `free_host_models` (ComfyUI unloads its own
     models in process; POST /free would only apply after the prompt) and record that it did.
  2. Upload the IMAGE / MASK / AUDIO inputs through POST /v1/files; their ids go into `inputs`.
  3. POST /v1/jobs, then poll GET /v1/jobs/{id} every `poll_s` (default 0.25 s), reporting progress.
  4. On interrupt, an exception or task cancellation: DELETE /v1/jobs/{id}, then raise.
  5. GET /v1/jobs/{id}/output, decode a PNG to IMAGE or an mp4 to IMAGE frames (+ AUDIO).
  6. keep_loaded=False: POST /v1/pool/release so the RAM goes back to the nodes that run next.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import convert
from .client import ShutterClient, ShutterError

TERMINAL = ("done", "failed", "cancelled")
_STEP_RE = re.compile(r"(?:denoise|denoising|steps?|sampling|sample)\s+(\d+)\s*/\s*(\d+)", re.I)


class Interrupted(Exception):
    """The host asked to stop; the Shutter job has been cancelled."""


def parse_step(stage: str):
    """(step, steps) from a stage text such as "segment 1/2 denoise 3/8"; None without a counter."""
    m = _STEP_RE.search(stage or "")
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if 0 < b and a <= b else None


@dataclass
class Hooks:
    on_progress: Callable[[float, Optional[int], Optional[int], str], None] = lambda *a: None
    is_interrupted: Callable[[], bool] = lambda: False
    free_host_models: Callable[[], None] = lambda: None


@dataclass
class Result:
    job: Dict[str, Any]
    kind: str                                  # "image", "video" or "other"
    output_path: str                           # local copy of the output file
    images: Optional[np.ndarray] = None        # IMAGE [N,H,W,3] float32 0..1
    audio: Optional[Dict[str, Any]] = None     # AUDIO {"waveform": [1,C,T], "sample_rate"}
    fps: Optional[float] = None
    freed_host_models: bool = False
    memory: Optional[Dict[str, Any]] = None    # the /v1/memory model block read before submitting
    uploads: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _ext(job: Dict[str, Any], data: bytes) -> str:
    out = (job.get("result") or {}).get("output") or ""
    ext = os.path.splitext(out)[1].lower()
    if ext:
        return ext
    if data[:8] == convert.PNG_SIG:
        return ".png"
    if data[4:8] == b"ftyp":
        return ".mp4"
    return ".bin"


async def generate(client: ShutterClient, request: Dict[str, Any], *,
                   image: Any = None, mask: Any = None, audio: Optional[Dict[str, Any]] = None,
                   hooks: Optional[Hooks] = None, poll_s: float = 0.25, free_host_models: bool = True,
                   keep_loaded: bool = True, out_dir: Optional[str] = None,
                   timeout_s: Optional[float] = None) -> Result:
    hooks = hooks or Hooks()
    req = dict(request)
    req["inputs"] = dict(req.get("inputs") or {})
    call = asyncio.to_thread

    # 1. room check
    mem_block, freed = None, False
    q = {k: req.get(k) for k in ("width", "height", "frames", "steps") if req.get(k) is not None}
    try:
        mem = await call(client.memory, req.get("model"), **q)
        mem_block = mem.get("model")
    except ShutterError:
        mem = None
    if free_host_models and mem_block and mem_block.get("fits") is False:
        await call(hooks.free_host_models)
        freed = True

    # 2. inputs
    uploads: Dict[str, Dict[str, Any]] = {}
    if image is not None:
        uploads["image"] = await call(client.upload, convert.image_to_png(image), "comfy_image.png", "image/png")
    if mask is not None:
        m = convert.from_tensor(mask)
        m = m[0] if m.ndim == 3 else m
        png = convert.encode_png((np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8)[..., None])
        uploads["mask"] = await call(client.upload, png, "comfy_mask.png", "image/png")
    if audio is not None:
        uploads["audio"] = await call(client.upload, convert.audio_to_wav(audio), "comfy_audio.wav", "audio/wav")
    for role, up in uploads.items():
        req["inputs"][role] = up["id"]
    if not req["inputs"]:
        req.pop("inputs")

    # 3-4. submit and follow
    job = await call(client.submit, req)
    jid = job["id"]
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    last = -1.0
    try:
        while True:
            if hooks.is_interrupted():
                raise Interrupted(f"interrupted; Shutter job {jid} cancelled")
            if timeout_s is not None and loop.time() - t0 > timeout_s:
                raise TimeoutError(f"Shutter job {jid} took longer than {timeout_s} s")
            job = await call(client.job, jid)
            if job["state"] in TERMINAL:
                break
            p = float(job.get("progress") or 0.0)
            if p != last:
                st = parse_step(job.get("stage", ""))
                hooks.on_progress(p, st[0] if st else None, st[1] if st else None, job.get("stage", ""))
                last = p
            await asyncio.sleep(poll_s)
    except BaseException:
        try:
            await asyncio.shield(call(client.cancel, jid))
        except Exception:
            pass
        raise

    if job["state"] != "done":
        detail = (job.get("result") or {}).get("error_detail")
        raise ShutterError(500, f"job {jid} {job['state']}: {job.get('message') or 'no message'}"
                           + (f" ({detail})" if detail else ""))
    hooks.on_progress(1.0, None, None, job.get("stage", ""))

    # 5. output
    data = await call(client.output, jid)
    ext = _ext(job, data)
    out_dir = out_dir or tempfile.mkdtemp(prefix="shutter_comfy_")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shutter_{jid}{ext}")
    with open(path, "wb") as f:
        f.write(data)
    res = Result(job=job, kind="other", output_path=path, freed_host_models=freed, memory=mem_block,
                 uploads=uploads)
    if ext == ".png":
        res.kind, res.images = "image", convert.png_to_image(data)
    elif ext in (".mp4", ".mov", ".webm"):
        res.kind = "video"
        res.images, res.fps, res.audio = await call(convert.read_video, path)

    # 6. hand the RAM back
    if not keep_loaded:
        try:
            await call(client.release, None)
        except ShutterError:
            pass
    return res
