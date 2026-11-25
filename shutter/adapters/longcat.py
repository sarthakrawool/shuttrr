"""LongCat-Video-Avatar 1.6 adapter (audio-driven talking head: portrait + voice + prompt -> 25 fps video).

The model code is NOT part of Shutter. It is imported at run time from `code_dir` (the user's patched tree,
`~/longcat-mlx/code-exact` on the Mac; plain upstream fedeinfe/longcat-avatar-mlx also works with fewer features).
This adapter drives the mechanisms that tree already has, the way its `scripts/generate.py` does:

  - crop box: `longcat_video_avatar.roi` (`plate_uint8`, `parse_box`, `composite`, 32 px feather), i.e.
    generate a full-height box `x0,x1` (multiples of 16, plate pixels) and composite it on the static plate;
  - room put back at every step: env `LC_BG_ANCHOR=1`, `LC_BG_PERSON=x0,x1` (person box in CROP pixels),
    `LC_BG_MARGIN`, `LC_BG_FEATHER` (read by `AvatarGenerator.generate`, applied inside `_denoise`);
  - speed / memory switches: `LC_SDPA_HEADS`, `LC_LORA_SWAP`, `LC_BLOCK_SPLIT`, `LC_FFN_CHUNKS`, `LC_FIRST_KV`,
    `LC_SEG_FRAMES`; `vae_fp32=False` for the bf16 decode; `Settings(steps=...)`.
Shutter's own code is only what the tree has no equivalent for: the render rule (`resolve_render`), memory/time
estimates, shot planning, progress/cancel hooks, the mp4 mux, and the zoom-cancel upscale (`shutter.addons.reels.upscale`).
The framing / crop box / room lock / upscale options come from the reels add-on (`shutter.addons.reels`) and are
read from `request.params` (`params.framing`; the old top-level `framing` key is folded in there by the core).

When each switch is read (checked in the patched tree, 2026-11-25):
  import time (module globals, read when the module is first imported, then used at call time inside compiled
  block functions):  LC_SDPA_HEADS -> models/attention.py SDPA_HEAD_CHUNK, LC_FFN_CHUNKS -> fast.py FFN_CHUNKS,
                     (LC_ATTN_FUSED, LC_TWIN: experimental, not driven).
  load time (AvatarGenerator._load_dit):  LC_LORA_SWAP -> self.lora_swap; LC_BLOCK_SPLIT -> _mount_fast (also
                     re-read at every LoRA swap re-mount, i.e. during the denoise loop).
  call time (every generate/_denoise call):  LC_SEG_FRAMES, LC_BG_ANCHOR/PERSON/MARGIN/FEATHER, LC_FIRST_KV.
So, per job, the adapter (1) writes all switches into os.environ before importing or calling; (2) overwrites the
two import-time module globals with the job's values (they are plain module attributes read at call time);
(3) keys a warm generator by its load signature (switches + VAE precision + code/weights dirs) and rebuilds it
(DiT reload, fresh mx.compile traces) when a job wants a different signature, because compiled traces bake the
globals in and the LoRA residency depends on LC_LORA_SWAP; (4) clears the LC_BG_* variables after every job and
resets `gen._anchor_parts`: the patched `generate` only sets `_anchor_parts` when LC_BG_ANCHOR=1 and never clears
it, so without the reset a warm generator would keep anchoring the background of later jobs.

Note: in the patched tree the background anchor is applied only when `num_cond == 1`, i.e. in the FIRST segment
of a clip. Continuation segments (beyond 93 frames) are not room-locked; the plan records a note when that applies.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import importlib
import importlib.util
import inspect
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from shutter.adapters.base import Adapter, RunContext
from shutter.core import modes as M
from shutter.addons.reels import Framing, framing_of
from shutter.types import GB, Cancelled, Job, JobRequest, MemoryEstimate, Shot

__all__ = ["LongCatAvatarAdapter", "RenderPlan", "resolve_render", "segment_layout", "segment_seed",
           "estimate_seconds", "estimate_memory", "probe_duration", "probe_image_size", "bucket_480p",
           "full_size_for", "RUNG_SETTINGS", "estimate_rung", "estimate_rung_seconds", "StageMeter"]

FPS = 25
SEG_FRAMES = 93          # upstream NUM_FRAMES; LC_SEG_FRAMES can change it (4k+1)
COND_FRAMES = 13         # upstream NUM_COND_FRAMES
FULL_HW = (704, 1312)    # the user's full-framing size (H, W)
FULL_AREA = FULL_HW[0] * FULL_HW[1]
SEED_SEGMENT_STRIDE = 7919     # upstream: mx.random.key(seed + 7919 * segment + 100003 * attempt)
SEED_ATTEMPT_STRIDE = 100003
DEFAULT_ZOOM_720 = 0.0070      # measured model push-in by the last frame, full frame 720p-class
CLASS_720_PIXELS = 600_000     # renders with H*W at or above this are "720p-class"
BG_KEYS = ("LC_BG_ANCHOR", "LC_BG_PERSON", "LC_BG_MARGIN", "LC_BG_FEATHER")

# Meituan's 480p buckets (ratio = H / W, portrait side); landscape uses the mirrored entry (16:9 -> 480x832).
_BUCKETS_480 = {0.82: (480, 592), 1.0: (576, 576), 1.17: (672, 576), 1.29: (704, 544), 1.35: (736, 544),
                1.5: (768, 512), 1.78: (832, 480)}

# Measured on an Apple M5 Max 128 GB, 8-bit DiT, 8 steps, 93 frames, --vae-tile 64 (NOTES.md 2026-11-25).
MEASURED = {
    "p480": {"hw": (480, 832), "tokens": 37_440, "s_per_step": 45.7, "decode_s": 53.0, "clip_s": 422.0,
             "mlx_peak_gb": 33.8, "machine_gb": 39.7},
    "full": {"hw": (704, 1312), "tokens": 86_592, "s_per_step": 164.0, "decode_s": 172.0, "clip_s": 1513.0,
             "mlx_peak_gb": 36.3, "process_gb": 38.0, "machine_gb": 55.0},
    "crop832_anchor": {"hw": (704, 832), "tokens": 54_912, "clip_s": 810.0, "clip_s_no_anchor": 758.0,
                       "machine_gb": 46.7},
}
PROCESS_OVER_MLX_GB = 38.0 - 36.3       # Python, ffmpeg pipes, numpy frames on top of MLX's peak
ROOM_LOCK_GB = 5.0
ROOM_LOCK_S = 52.0                      # per anchored segment (810 - 758)
LOW_MEMORY_GB = 2.7
VAE_BF16_GB = 3.8                       # 480p: 33.8 -> ~30 GB MLX peak
DEFAULT_RESIDENT_GB = 18.0              # 8-bit DiT + one LoRA + VAE + whisper, used when the weights are not on disk

# ---- modes (docs/MODES.md): rung -> the tree's switches + Shutter's own staged release / weight streaming
RUNG_SETTINGS: dict[str, dict[str, Any]] = {
    "resident":     {"split": False, "ffn": False, "release": "never", "window": None},
    "split":        {"split": True,  "ffn": False, "release": "never", "window": None},
    "staged":       {"split": True,  "ffn": True,  "release": "every", "window": None},
    "stream_wide":  {"split": True,  "ffn": True,  "release": "never", "window": 8},
    "stream_tight": {"split": True,  "ffn": True,  "release": "never", "window": 2},
}
SPLIT_SAVING_GB = 2.0          # REPORT.md prof5: split working memory 12.8 vs 14.8 GB
STAGED_SAVING_GB = 2.7         # split + FFN 4 frame chunks: 12.1 vs 14.8 GB
SPLIT_TIME = 1.023             # +2.3% per block (prof5)
STAGED_TIME = 1.020            # split + FFN 4: 3.52 vs 3.451 s/block
VAE_DECODE_ALONE_GB = 13.8     # fp32 tile-64 decode with nothing else resident (vae2)
STREAM_PREP_GB = 8.5           # MLX peak of the prep stage (prompt + audio encode) with streaming, measured stream_tight 480p
VAE_DECODE_ALONE_BF16_GB = 7.1
DIT_GB = 18.0                  # 8-bit DiT weights (REPORT.md memory table)
LORA_GB = 2.52                 # one LoRA (bf16), the one LC_LORA_SWAP keeps
NONBLOCK_GB = 0.25             # embedders, audio projection, final layer (stay resident when streaming)
DEPTH = 48
RELOAD_S = 20.0                # staged: DiT reload + re-trace per extra segment (to be replaced by measurements)
STREAM_COPY_BPS = 4e9          # host -> MLX copy of a streamed block on the compute thread
STREAM_SYNC_S = 0.005          # one mx.eval per block


# ----------------------------------------------------------------------------- render rule (pure)


@dataclass
class RenderPlan:
    framing: str
    height: int                    # generated height (crop height when a crop box is used)
    width: int                     # generated width
    plate_height: int              # full-frame render size the plate / composite has
    plate_width: int
    crop: Optional[str] = None     # roi spec as the tree's parse_box takes it: "auto" or "x0,x1"
    crop_feather: int = 32
    room_lock: bool = False
    person_box: Optional[str] = None   # "x0,x1" in plate px, or "auto"
    bg_margin: int = 24
    bg_feather: int = 4
    vae_tile: int = 64
    vae_bf16: bool = False
    steps: int = 8
    seg_frames: int = SEG_FRAMES
    low_memory: bool = False       # LC_BLOCK_SPLIT + LC_FFN_CHUNKS
    upscale_to: Optional[tuple[int, int]] = None   # (W, H) edit size
    zoom_cancel: float = 0.0
    tokens: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def latent_frames(self) -> int:
        return (self.seg_frames - 1) // 4 + 1

    @property
    def class_720(self) -> bool:
        return self.plate_height * self.plate_width >= CLASS_720_PIXELS

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["upscale_to"] = list(self.upscale_to) if self.upscale_to else None
        return d


def _round_to(v: float, m: int) -> int:
    return max(m, int(round(v / m)) * m)


def bucket_480p(img_w: int, img_h: int) -> tuple[int, int]:
    """(H, W) 480p bucket for an image, landscape mirrored: 16:9 -> (480, 832), 9:16 -> (832, 480)."""
    r = img_h / img_w
    if r >= 1:
        key = min(_BUCKETS_480, key=lambda k: abs(k - r))
        return _BUCKETS_480[key]
    key = min(_BUCKETS_480, key=lambda k: abs(k - 1 / r))
    h, w = _BUCKETS_480[key]
    return w, h


def full_size_for(img_w: int, img_h: int) -> tuple[int, int]:
    """(H, W) of the 704x1312 class for an image: landscape (W/H >= 1.5) -> 704x1312, portrait (H/W >= 1.5) ->
    1312x704, anything squarer keeps the same pixel count at the image's ratio, multiples of 32."""
    a = img_w / img_h
    if a >= 1.5:
        return FULL_HW
    if a <= 1 / 1.5:
        return FULL_HW[1], FULL_HW[0]
    w = _round_to(math.sqrt(FULL_AREA * a), 32)
    h = _round_to(FULL_AREA / w, 32)
    return h, w


def _parse_hw(s) -> tuple[int, int]:
    """'HxW' (the tree's --size convention, e.g. 480x832) or (H, W)."""
    if isinstance(s, str):
        h, w = s.lower().split("x")
        return int(h), int(w)
    h, w = s
    return int(h), int(w)


def _parse_wh(s) -> tuple[int, int]:
    """'WxH' (edit size convention, e.g. 1920x1080) or (W, H)."""
    if isinstance(s, str):
        w, h = s.lower().split("x")
        return int(w), int(h)
    w, h = s
    return int(w), int(h)


def _parse_crop(spec: str, plate_h: int, plate_w: int, notes: list[str]) -> str:
    """Normalise a crop box to the tree's full-height 'x0,x1' form (multiples of 16), or 'auto'.
    Accepts 'auto', 'x0,x1' or 'w:h:x:y' (plate pixels; the box is always full height)."""
    spec = str(spec).strip()
    if spec == "auto":
        return "auto"
    if ":" in spec:
        w, h, x, _y = (int(float(v)) for v in spec.split(":"))
        if h != plate_h:
            notes.append(f"crop box height {h} ignored: the crop is always full height ({plate_h})")
        x0, x1 = x, x + w
    else:
        x0, x1 = (int(float(v)) for v in spec.split(","))
    a = (max(0, x0) // 16) * 16
    b = min(plate_w, -(-x1 // 16) * 16)
    if (a, b) != (x0, x1):
        notes.append(f"crop box {x0},{x1} snapped to {a},{b} (multiples of 16 inside the frame)")
    if b - a < 64:
        raise ValueError(f"crop box {spec!r} is too narrow")
    return f"{a},{b}"


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def resolve_render(request: JobRequest, image_size: Optional[tuple[int, int]] = None,
                   options: Optional[dict[str, Any]] = None) -> RenderPlan:
    """Pure: the concrete render for a request. `image_size` = (W, H) of the start image (16:9 assumed if None).

    The user's standing rule: SMALL framing (corner card, split screen, picture in picture) renders the 480p
    bucket FULL FRAME (never a crop box, never room lock) and is upscaled; FULL/AUTO renders the 704x1312 class;
    camera zoom is added in the edit, never by the model. Request params:
      size "HxW" (override), crop_box "auto" | "x0,x1" | "w:h:x:y" (FULL only, opt-in), room_lock bool (needs a
      crop box and person_box "x0,x1" plate px or "auto"), upscale_to "WxH" | "off", zoom_cancel "auto" | float |
      "off", steps, vae_tile, vae_bf16, crop_feather, bg_margin, bg_feather.
    """
    o = dict(options or {})
    p = dict(request.params or {})
    notes: list[str] = []
    iw, ih = image_size or (1920, 1080)
    framing = framing_of(request)
    small = framing == Framing.SMALL

    if small:
        plate_h, plate_w = bucket_480p(iw, ih)
    else:
        plate_h, plate_w = full_size_for(iw, ih)
    if p.get("size"):
        h, w = _parse_hw(p["size"])
        if h % 32 or w % 32:
            nh, nw = _round_to(h, 32), _round_to(w, 32)
            notes.append(f"size {h}x{w} rounded to {nh}x{nw} (multiples of 32)")
            h, w = nh, nw
        plate_h, plate_w = h, w

    crop = None
    if p.get("crop_box"):
        if small:
            notes.append("crop_box ignored: small framing always renders the 480p bucket full frame, then upscales")
        else:
            crop = _parse_crop(p["crop_box"], plate_h, plate_w, notes)

    room_lock = _truthy(p.get("room_lock", False))
    person_box = p.get("person_box")
    if room_lock and small:
        notes.append("room_lock ignored: small framing renders full frame (480p push is ~0.02%, needs no fix)")
        room_lock = False
    elif room_lock and crop is None:
        notes.append("room_lock ignored: it is used with a crop box only (full-frame push is cancelled in the upscale)")
        room_lock = False
    if room_lock:
        if not person_box:
            raise ValueError("room_lock needs person_box ('x0,x1' in plate pixels, or 'auto')")
        if person_box != "auto":
            px0, px1 = (int(float(v)) for v in str(person_box).split(","))
            person_box = f"{px0},{px1}"
            if crop and crop != "auto":
                a, b = (int(v) for v in crop.split(","))
                if not (a <= px0 < px1 <= b):
                    raise ValueError(f"person_box {person_box} is not inside the crop box {crop}")
                if px0 - a < 64 or b - px1 < 64:
                    notes.append("person box is close to the crop edge: keep the feathered mask edge away from the "
                                 "person to avoid a halo")

    if crop is None:
        height, width = plate_h, plate_w
    elif crop == "auto":
        height, width = plate_h, min(plate_w, 832)   # measured auto box width; the real one is known at run time
        notes.append("crop box 'auto' (Apple Vision): size estimated as 832 px wide until run time")
    else:
        a, b = (int(v) for v in crop.split(","))
        height, width = plate_h, b - a

    steps = int(p.get("steps", o.get("steps", 8)))
    if steps != 8:
        notes.append(f"{steps} steps: the distill schedule is trained for 8, other counts are not lossless")
    vae_tile = int(p.get("vae_tile", o.get("vae_tile", 64)))
    vae_bf16 = _truthy(p.get("vae_bf16", o.get("vae_bf16", False)))
    if vae_bf16:
        notes.append("vae_bf16: faster, about half the decode memory, NOT bit-exact (46.6 dB vs fp32)")
    seg_frames = int(p.get("seg_frames", o.get("seg_frames", SEG_FRAMES)))
    if (seg_frames - 1) % 4 or seg_frames <= COND_FRAMES:
        raise ValueError("seg_frames must be 4k+1 and > 13")

    plan = RenderPlan(framing=framing.value, height=height, width=width, plate_height=plate_h, plate_width=plate_w,
                      crop=crop, crop_feather=int(p.get("crop_feather", o.get("crop_feather", 32))),
                      room_lock=room_lock, person_box=person_box if room_lock else None,
                      bg_margin=int(p.get("bg_margin", o.get("bg_margin", 24))),
                      bg_feather=int(p.get("bg_feather", o.get("bg_feather", 4))),
                      vae_tile=vae_tile, vae_bf16=vae_bf16, steps=steps, seg_frames=seg_frames,
                      low_memory=_truthy(o.get("low_memory", False)), notes=notes)
    plan.tokens = (height // 16) * (width // 16) * plan.latent_frames

    # zoom cancel: total model push by the last frame, cancelled inside the final upscale (one resample)
    zc = p.get("zoom_cancel", o.get("zoom_cancel", "auto"))
    if isinstance(zc, str) and zc.strip().lower() in ("off", "none", "0", ""):
        zoom = 0.0
    elif isinstance(zc, str) and zc.strip().lower() == "auto":
        zoom = DEFAULT_ZOOM_720 if (crop is None and plan.class_720) else 0.0
    else:
        zoom = float(zc)
    if zoom and crop is not None:
        notes.append("zoom_cancel off: with a crop box the plate is static (use room_lock to hold the crop's camera)")
        zoom = 0.0
    plan.zoom_cancel = zoom

    up = p.get("upscale_to", o.get("edit_size", "1920x1080"))
    if up is None or (isinstance(up, str) and up.strip().lower() in ("off", "none", "")):
        plan.upscale_to = (plate_w, plate_h) if zoom else None
        if zoom:
            notes.append("upscale_to off: zoom cancel still runs, as one resample at the render size")
    else:
        uw, uh = _parse_wh(up)
        if (uw >= uh) != (plate_w >= plate_h):     # orient the edit size like the render
            uw, uh = uh, uw
        plan.upscale_to = (uw, uh)
    if small and plan.upscale_to is None:
        notes.append("small framing without upscale_to: the 480p render is returned as is")
    return plan


# ----------------------------------------------------------------------------- segments and seeds (pure)


def segment_layout(duration_s: float, seg_frames: int = SEG_FRAMES) -> list[tuple[int, int, int]]:
    """Upstream's split of a clip (image start, no start_frames): n_seg = max(1, ceil((d*25 - NF) / step) + 1),
    step = NF - 13. Returns per segment (first_new_frame, new_frames, total_frames_generated); the output is trimmed
    to round(d * 25) frames, so the last segment's kept frames can be fewer."""
    step = seg_frames - COND_FRAMES
    n_seg = max(1, math.ceil((duration_s * FPS - seg_frames) / step) + 1)
    total = int(round(duration_s * FPS))
    out = []
    for s in range(n_seg):
        start = 0 if s == 0 else seg_frames + (s - 1) * step
        new = seg_frames if s == 0 else step
        keep = max(0, min(new, total - start))
        out.append((start, keep, seg_frames))
    return out


def segment_seed(base_seed: int, segment: int, attempt: int = 0) -> int:
    """The integer upstream feeds to mx.random.key for a segment's initial noise:
    `seed + 7919 * segment + 100003 * attempt` (AvatarGenerator.generate, `noise()`)."""
    return base_seed + SEED_SEGMENT_STRIDE * segment + SEED_ATTEMPT_STRIDE * attempt


# ----------------------------------------------------------------------------- estimates (pure)


def estimate_seconds(plan: RenderPlan, n_segments: int = 1) -> dict[str, float]:
    """Per step: s = 45.7 * (tokens / 37440) ** 1.524 (fits 45.7 s @ 37,440 and 164 s @ 86,592 tokens:
    attention makes it super-linear). Decode: 53 * (pixels / 399,360) ** 1.404 (53 s @ 480x832, 172 s @
    704x1312, tiled fp32). bf16 decode about halves it. Room lock: +52 s per anchored (first) segment."""
    a, b = MEASURED["p480"], MEASURED["full"]
    pe = math.log(b["s_per_step"] / a["s_per_step"]) / math.log(b["tokens"] / a["tokens"])
    sps = a["s_per_step"] * (plan.tokens / a["tokens"]) ** pe
    px_a, px_b = a["hw"][0] * a["hw"][1], b["hw"][0] * b["hw"][1]
    de = math.log(b["decode_s"] / a["decode_s"]) / math.log(px_b / px_a)
    dec = a["decode_s"] * (plan.height * plan.width / px_a) ** de
    if plan.vae_bf16:
        dec *= 0.5
    seg = plan.steps * sps + dec
    total = n_segments * seg + (ROOM_LOCK_S if plan.room_lock else 0.0)
    return {"seconds_per_step": round(sps, 1), "decode_s": round(dec, 1), "segment_s": round(seg, 1),
            "total_s": round(total, 1)}


def estimate_memory(plan: RenderPlan, resident_bytes: Optional[int] = None) -> MemoryEstimate:
    """Process peak = MLX peak + 1.7 GB, MLX peak = 31.9 GB + 50.9 KB * tokens (linear fit of the two measured
    MLX peaks: 33.8 GB @ 37,440 tokens, 36.3 GB @ 86,592 tokens; tokens = (H/16)*(W/16)*24 latent frames of the
    generated box). +5 GB with room lock (static-plate latents + anchor), -2.7 GB low_memory, -3.8 GB bf16 VAE.
    Shifted by (actual resident weights - 18 GB) when the weights are on disk. The whole machine adds ~2-17 GB
    on top (39.7 GB at 480p, 46.7 GB crop+anchor, ~55 GB full 720p), which the memory guard sees as used."""
    a, b = MEASURED["p480"], MEASURED["full"]
    slope = (b["mlx_peak_gb"] - a["mlx_peak_gb"]) / (b["tokens"] - a["tokens"])
    mlx = a["mlx_peak_gb"] + slope * (plan.tokens - a["tokens"])
    peak = mlx + PROCESS_OVER_MLX_GB
    if plan.room_lock:
        peak += ROOM_LOCK_GB
    if plan.vae_bf16:
        peak -= VAE_BF16_GB
    low = plan.low_memory
    if low:
        peak -= LOW_MEMORY_GB
    steady = DEFAULT_RESIDENT_GB * GB
    if resident_bytes:
        peak += (resident_bytes - steady) / GB
        steady = resident_bytes
    peak_b = int(max(peak * GB, steady + 2 * GB))
    note = (f"{plan.height}x{plan.width}, {plan.tokens} tokens, peak {peak_b / GB:.1f} GB "
            f"(MLX ~{mlx:.1f} GB)" + (", room lock" if plan.room_lock else "") + (", low memory" if low else ""))
    return MemoryEstimate(peak=peak_b, steady=int(steady), disk_scratch=600 * 1024**2, note=note)


def kv_cache_bytes(plan: RenderPlan, hidden: int = 4096, depth: int = DEPTH) -> int:
    """Continuation segments keep per-block keys/values of the 5 conditioning latents (kv_cache.py):
    depth x (k + v) x cond tokens x hidden x bf16."""
    cond_tokens = (1 + (COND_FRAMES - 1) // 4 + 1) * (plan.height // 16) * (plan.width // 16)
    return depth * 2 * cond_tokens * hidden * 2


def estimate_rung(plan: RenderPlan, rung_id: str, *, resident_bytes: Optional[int] = None,
                  sizes: Optional[dict[str, int]] = None, measured: Optional[dict[str, Any]] = None,
                  n_segments: int = 1, window: Optional[int] = None) -> MemoryEstimate:
    """Process peak for one rung (GB figures from REPORT.md; `measured` = this model/size/rung from measured.json,
    whose `peak_bytes` is the MLX peak of a plain single-segment run, preferred when the plan is plain).

    resident      the measured fast path (estimate_memory)
    split         - 2.0 GB of step working memory
    staged        max(step - 2.7 GB, VAE decode alone 13.8 GB): the DiT is gone before the decode
    stream_*      step - 2.7 - (DiT + LoRA) + non-block + one block in MLX, max with the decode alone, plus the
                  streamer's host buffers (window x block group, numpy, outside MLX's count); the step is NOT
                  shifted by resident_bytes, and the peak is never below the measured prep stage
                  (STREAM_PREP_GB + process overhead)
    Multi-segment clips add the continuation KV cache to every rung (it is activations, kept in all rungs)."""
    M.rung(rung_id)
    s = RUNG_SETTINGS[rung_id]
    win = window if window is not None else s["window"]
    base_plan = dataclasses.replace(plan, low_memory=False)
    base = estimate_memory(base_plan, resident_bytes)
    # Streaming rungs never hold the whole weight folder, so their step is not shifted by the bytes on disk
    # (both LoRAs, the text encoder...); `base.steady` still says what really stays resident.
    peak_gb = (estimate_memory(base_plan).peak if win else base.peak) / GB
    sizes = sizes or {}
    dit_gb = sizes.get("dit", 0) / GB or DIT_GB
    lora_gb = sizes.get("lora", 0) / GB if "lora" in sizes else LORA_GB
    nonblock_gb = sizes.get("nonblock", 0) / GB or NONBLOCK_GB
    group_gb = (sizes.get("block", 0) / GB or (dit_gb - nonblock_gb) / DEPTH) + lora_gb / DEPTH
    decode_gb = (VAE_DECODE_ALONE_BF16_GB if plan.vae_bf16 else VAE_DECODE_ALONE_GB) + PROCESS_OVER_MLX_GB
    host_gb = (win or 0) * group_gb
    plain = not plan.room_lock and not plan.vae_bf16 and plan.crop is None
    notes = []
    if measured and plain and measured.get("peak_bytes"):
        peak_gb = measured["peak_bytes"] / GB + PROCESS_OVER_MLX_GB + host_gb
        notes.append("measured")
    elif rung_id == "split":
        peak_gb -= SPLIT_SAVING_GB
    elif rung_id == "staged":
        peak_gb = max(peak_gb - STAGED_SAVING_GB, decode_gb)
    elif win:
        step = peak_gb - STAGED_SAVING_GB - dit_gb - lora_gb + nonblock_gb + group_gb
        peak_gb = max(step, decode_gb + nonblock_gb) + host_gb
    if n_segments > 1:
        peak_gb += kv_cache_bytes(plan) / GB
        notes.append(f"+{kv_cache_bytes(plan) / GB:.1f} GB continuation KV cache")
    resident = base.steady / GB
    if rung_id == "staged":
        steady = max(0.0, resident - dit_gb - lora_gb)
    elif win:
        steady = max(0.0, resident - dit_gb - lora_gb) + nonblock_gb + host_gb
    else:
        steady = resident
    floor_gb = STREAM_PREP_GB + PROCESS_OVER_MLX_GB if win else 0.0   # the prep stage, measured
    peak_b = int(max(peak_gb, steady + 1, floor_gb) * GB)
    note = (f"{rung_id}: {plan.height}x{plan.width}, {plan.tokens} tokens, peak {peak_b / GB:.1f} GB"
            + (f" (window {win}, {group_gb:.2f} GB/block)" if win else "") + ("; " + ", ".join(notes) if notes else ""))
    return MemoryEstimate(peak=peak_b, steady=int(steady * GB), disk_scratch=600 * 1024**2, note=note)


def estimate_rung_seconds(plan: RenderPlan, rung_id: str, n_segments: int = 1, *,
                          sizes: Optional[dict[str, int]] = None, measured: Optional[dict[str, Any]] = None,
                          window: Optional[int] = None) -> dict[str, float]:
    """estimate_seconds scaled per rung: split +2.3%, staged +2% (+ a reload per extra segment), streaming adds a
    host copy of every block and one eval per block on each step (SSD reads overlap compute)."""
    M.rung(rung_id)
    base = estimate_seconds(plan, n_segments)
    if measured and measured.get("seconds") and n_segments == 1 and plan.crop is None and not plan.room_lock:
        return {**base, "total_s": round(float(measured["seconds"]), 1), "measured": True}
    s = RUNG_SETTINGS[rung_id]
    sps = base["seconds_per_step"]
    if rung_id == "split":
        sps *= SPLIT_TIME
    elif rung_id != "resident":
        sps *= STAGED_TIME
    win = window if window is not None else s["window"]
    if win:
        sizes = sizes or {}
        group = sizes.get("block", 0) or ((DIT_GB - NONBLOCK_GB) / DEPTH + LORA_GB / DEPTH) * GB
        sps += DEPTH * (group / STREAM_COPY_BPS + STREAM_SYNC_S)
    extra = RELOAD_S * (n_segments - 1) if s["release"] == "every" else 0.0
    seg = plan.steps * sps + base["decode_s"]
    total = n_segments * seg + (ROOM_LOCK_S if plan.room_lock else 0.0) + extra
    return {"seconds_per_step": round(sps, 1), "decode_s": base["decode_s"], "segment_s": round(seg, 1),
            "total_s": round(total, 1)}


# ----------------------------------------------------------------------------- probes


def probe_duration(path, runner: Callable = subprocess.run) -> float:
    """Length of an audio file in seconds. With the default runner it needs no ffprobe: soundfile first,
    then librosa (which also reads mp3/m4a through audioread). A custom runner keeps the ffprobe call."""
    if runner is subprocess.run:
        try:
            import soundfile as sf
            info = sf.info(str(path))
            if info.samplerate:
                return float(info.frames) / float(info.samplerate)
        except Exception:
            pass
        try:
            import librosa
            return float(librosa.get_duration(path=str(path)))
        except Exception:
            if not shutil.which("ffprobe"):
                raise
    r = runner([shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def probe_image_size(path, runner: Callable = subprocess.run) -> tuple[int, int]:
    """(width, height) of an image. With the default runner it reads the file with Pillow (no ffprobe)."""
    if runner is subprocess.run:
        try:
            from PIL import Image
            with Image.open(str(path)) as im:
                w, h = im.size
            return int(w), int(h)
        except Exception:
            if not shutil.which("ffprobe"):
                raise
    r = runner([shutil.which("ffprobe") or "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height", "-of", "csv=p=0:s=x", str(path)], capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split("\n")[0].split("x")[:2]
    return int(w), int(h)


# ----------------------------------------------------------------------------- run-time glue (model code)


GEN_MODULE = "longcat_video_avatar.generation"


def import_generation(code_dir: Path):
    """Import the model's generation module from `code_dir` (after the env switches are set). If a module of that
    name was imported from another directory, purge it first so two trees never mix."""
    code_dir = Path(code_dir).expanduser().resolve()
    mod = sys.modules.get(GEN_MODULE)
    if mod is not None and not str(Path(getattr(mod, "__file__", "")).resolve()).startswith(str(code_dir)):
        for name in [n for n in sys.modules if n == "longcat_video_avatar" or n.startswith("longcat_video_avatar.")]:
            del sys.modules[name]
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    importlib.invalidate_caches()
    return importlib.import_module(GEN_MODULE)


def apply_import_time_switches(env: dict[str, str]) -> dict[str, Any]:
    """Overwrite the import-time module globals with the job's values (they are read at call time inside the
    forward). Returns what was set. Missing modules/attributes (plain upstream) are skipped."""
    done = {}
    for modname, attr, key in (("longcat_video_avatar.models.attention", "SDPA_HEAD_CHUNK", "LC_SDPA_HEADS"),
                               ("longcat_video_avatar.fast", "FFN_CHUNKS", "LC_FFN_CHUNKS")):
        m = sys.modules.get(modname)
        if m is not None and hasattr(m, attr) and key in env:
            setattr(m, attr, int(env[key]))
            done[f"{modname}.{attr}"] = int(env[key])
    return done


class StepHook:
    """Progress + cancel, called from the generator subclass at every segment and every denoise step."""

    def __init__(self, ctx: Optional[RunContext], n_segments: int, steps: int, lo: float = 0.0, hi: float = 0.9):
        self.ctx, self.n_seg, self.steps, self.lo, self.hi = ctx, max(1, n_segments), max(1, steps), lo, hi
        self.segment = 0          # 1-based, current segment
        self.step_times: list[float] = []
        self._t_step: Optional[float] = None

    def _units(self, seg: int, step: float) -> float:
        per = self.steps + 1      # steps + decode
        return ((seg - 1) * per + step) / (self.n_seg * per)

    def _report(self, frac: float, stage: str):
        if self.ctx is None:
            return
        self.ctx.check_cancelled()
        self.ctx.report(self.lo + (self.hi - self.lo) * min(1.0, frac), stage)

    def segment_start(self):
        self.segment += 1
        self._t_step = None
        self._report(self._units(self.segment, 0), f"segment {self.segment}/{self.n_seg} denoise 0/{self.steps}")

    def step_start(self, i: int):
        now = time.time()
        if self._t_step is not None:
            self.step_times.append(now - self._t_step)
        self._t_step = now
        self._report(self._units(self.segment, i), f"segment {self.segment}/{self.n_seg} denoise {i + 1}/{self.steps}")

    def segment_denoised(self):
        if self._t_step is not None:
            self.step_times.append(time.time() - self._t_step)
            self._t_step = None
        self._report(self._units(self.segment, self.steps), f"segment {self.segment}/{self.n_seg} decode")

    def segment_done(self, n: int):
        self._report(self._units(n, self.steps + 1), f"segment {n}/{self.n_seg} done")

    @property
    def seconds_per_step(self) -> Optional[float]:
        return round(sum(self.step_times) / len(self.step_times), 2) if self.step_times else None


class StageMeter:
    """Per-stage wall time and MLX peak memory: each `mark(stage)` closes the running stage (seconds,
    `mx.get_peak_memory()` since it began) and starts the next one after `mx.reset_peak_memory()`.
    Stages: prep (audio + prompt features), load / reload (DiT), encode (VAE encode of the reference or
    conditioning frames), step (one denoise step), release (staged: DiT dropped), decode, finish, post."""

    def __init__(self, mx=None, clock: Callable[[], float] = time.perf_counter):
        self.mx = mx
        self.clock = clock
        self.records: list[dict[str, Any]] = []
        self._cur: Optional[tuple[str, dict]] = None
        self._t0 = 0.0

    def _peak(self) -> Optional[int]:
        fn = getattr(self.mx, "get_peak_memory", None) if self.mx is not None else None
        try:
            return int(fn()) if fn else None
        except Exception:
            return None

    def _reset(self) -> None:
        fn = getattr(self.mx, "reset_peak_memory", None) if self.mx is not None else None
        try:
            if fn:
                fn()
        except Exception:
            pass

    def start(self, stage: str, **kw) -> None:
        self._reset()
        self._cur = (stage, kw)
        self._t0 = self.clock()

    def close(self) -> None:
        if self._cur is None:
            return
        stage, kw = self._cur
        rec = {"stage": stage, **kw, "seconds": round(self.clock() - self._t0, 3), "peak_bytes": self._peak()}
        self.records.append(rec)
        self._cur = None

    def mark(self, stage: str, **kw) -> None:
        self.close()
        self.start(stage, **kw)

    def summary(self) -> dict[str, Any]:
        secs: dict[str, float] = {}
        peaks: dict[str, int] = {}
        for r in self.records:
            k = "denoise" if r["stage"] == "step" else r["stage"]
            secs[k] = round(secs.get(k, 0.0) + r["seconds"], 2)
            if r.get("peak_bytes") is not None:
                peaks[k] = max(peaks.get(k, 0), int(r["peak_bytes"]))
        steps = [r["seconds"] for r in self.records if r["stage"] == "step"]
        return {"stage_seconds": secs, "stage_peak_bytes": peaks, "step_seconds": steps,
                "peak_bytes": max(peaks.values()) if peaks else None}


def make_generator_class(gen_mod):
    """Subclass of the tree's AvatarGenerator with hooks, no copied model code:

    - `_denoise` (one call per segment attempt) is wrapped whole: segment start before, "decode" after; when the
      DiT was released (staged rung) it is reloaded first;
    - `_set_loras_for_step(i, st)` is called by the upstream AND patched loop at the top of every step, after the
      previous step's latents were evaluated (`mx.eval` ends each step). It is the only per-step method call in
      the loop, so the override marks the stage meter, calls super() (the patched LoRA swap runs there) and then
      reports progress and checks cancellation. Raising Cancelled there unwinds out of `generate` cleanly.
    - `_load_dit`: the stage meter's load/reload stage; with `_shutter_stream_cfg` set (stream rungs) the DiT is
      built as a streamed DiT (shutter.adapters.longcat_stream) instead of `load_dit`.
    - `_shutter_release_dit()`: drop the DiT, its LoRAs and compiled blocks (staged rung, before the decode).
    The step loop itself is internal to `_denoise`, so a per-step super() delegation is impossible; wrapping the
    DiT call would miss the KV-cache path (`kv_cache.forward_noise` uses the DiT's submodules directly).
    Room lock needs no hook here: the patched `_denoise` applies LC_BG_ANCHOR itself.
    """
    base = gen_mod.AvatarGenerator

    class ShutterAvatarGenerator(base):
        _shutter_hook: Optional[StepHook] = None
        _shutter_sig: tuple = ()
        _shutter_meter: Optional[StageMeter] = None
        _shutter_stream_cfg: Optional[dict] = None
        _shutter_loaded_once: bool = False
        _shutter_releases: int = 0
        _shutter_reloads: int = 0
        _shutter_speed = None               # longcat_speed.SpeedConfig of the running job (opt-in, not bit-identical)
        _shutter_speed_mounted: bool = False

        def _mount_fast(self):
            """The tree's mount (on load and after every LoRA swap), plus the opt-in speed hooks when a job asked
            for them. With no speed options and nothing mounted before, this is exactly the tree's method."""
            sp = self._shutter_speed
            if sp is None and not self._shutter_speed_mounted:
                return super()._mount_fast()
            from shutter.adapters import longcat_speed as LSP

            LSP.uninstall(self.dit)
            self._shutter_speed_mounted = False
            if sp is not None and sp.block_sparse is not None and "block_sparse" not in sp.refused:
                sp.installed["block_sparse"] = LSP.install_attention(self.dit, sp)
                self._shutter_speed_mounted = True
            r = super()._mount_fast()
            if sp is not None and sp.step_cache is not None and "step_cache" not in sp.refused:
                steps = self._shutter_hook.steps if self._shutter_hook is not None else None
                sp.installed["step_cache"] = LSP.install_step_cache(self.dit, sp, steps)
                self._shutter_speed_mounted = True
            return r

        def _shutter_set_speed(self, speed):
            """Set (or clear, with None) the job's speed options. A loaded DiT is re-mounted (fast.remove + mount,
            as the tree's own LoRA swap does) so the hooks and fresh compiled traces apply to this job."""
            if speed is None and self._shutter_speed is None and not self._shutter_speed_mounted:
                return
            self._shutter_speed = speed
            dit = self.dit
            if dit is None or not hasattr(dit, "named_modules") or not hasattr(base, "_mount_fast"):
                return
            fast = sys.modules.get("longcat_video_avatar.fast")
            if fast is not None:
                fast.remove(dit)
            self._mount_fast()

        def _load_dit(self):
            if self.dit is not None:
                return
            meter = self._shutter_meter
            seg = self._shutter_hook.segment if self._shutter_hook is not None else 0
            if meter is not None:
                meter.mark("reload" if self._shutter_loaded_once else "load", segment=seg)
            if self._shutter_loaded_once:
                self._shutter_reloads += 1
            cfg = self._shutter_stream_cfg
            if cfg is None:
                self.lora_keys = []          # the tree appends to it on every load
                super()._load_dit()
            else:
                self._shutter_load_streamed(cfg)
            self._shutter_loaded_once = True
            if meter is not None:
                meter.mark("encode", segment=max(1, seg))

        def _shutter_load_streamed(self, cfg):
            from shutter.adapters import longcat_stream as LS

            LS.install_lora_hook(gen_mod)
            self.lora_keys = list(self.paths.loras)
            self.lora_swap = False              # residency is the streamer's job (LC_LORA_SWAP=1 semantics)
            self.resident = set(self.lora_keys)
            swap = {"dmd", "opsd"} <= set(self.lora_keys)
            initial = [k for k in self.lora_keys if not (swap and k == "opsd")]

            def mount(d):
                self.dit = d
                self._mount_fast()

            self.dit = LS.build_streamed_dit(Path(self.paths.dit), dict(self.paths.loras), window=int(cfg["window"]),
                                             mount=mount, initial_loras=initial, nocache=bool(cfg.get("nocache")))

        def _shutter_release_dit(self):
            dit = self.dit
            if dit is None:
                return
            self.dit = None
            from shutter.adapters import longcat_stream as LS

            LS.release_dit(dit)
            del dit
            self.resident = set()
            self.lora_keys = []
            gc.collect()
            mx = sys.modules.get("mlx.core")
            if mx is not None and hasattr(mx, "clear_cache"):
                mx.clear_cache()
            self._shutter_releases += 1

        def _shutter_stream_stats(self):
            sd = getattr(self.dit, "__dict__", {}).get("_shutter_stream") if self.dit is not None else None
            return sd.stats() if sd is not None else None

        def _denoise(self, *args, **kwargs):
            hook = self._shutter_hook
            if hook is not None:
                hook.segment_start()
            if self.dit is None:            # staged: released before the previous decode
                self._load_dit()
            out = super()._denoise(*args, **kwargs)
            if hook is not None:
                hook.segment_denoised()
            return out

        def _set_loras_for_step(self, i, st):
            meter = self._shutter_meter
            if meter is not None:
                seg = self._shutter_hook.segment if self._shutter_hook is not None else 0
                meter.mark("step", segment=seg, step=i + 1)
            r = super()._set_loras_for_step(i, st)
            sp = self._shutter_speed
            if sp is not None and sp.cache is not None:
                from shutter.adapters import longcat_speed as LSP

                sig = LSP.lora_signature(self.dit)
                if sp.lora_sig is not None and sig != sp.lora_sig:
                    sp.cache.invalidate()          # the model changed (DMD -> OPSD): no residual reuse across it
                    sp.invalidations += 1
                sp.lora_sig = sig
                sp.cache.num_steps = int(getattr(st, "steps", 0) or 0) or sp.cache.num_steps
                sp.cache.new_step(i)
            hook = self._shutter_hook
            if hook is not None:
                hook.step_start(i)
            return r

    return ShutterAvatarGenerator


@contextlib.contextmanager
def decode_hooks(gen_mod, gen, *, release: str = "never", n_segments: int = 1, save_dir: Optional[Path] = None,
                 meter: Optional[StageMeter] = None):
    """Wrap the module-level `vae_decode` that `generate`'s local `decode()` calls once per segment (after the
    segment's latents are evaluated and before the next segment's conditioning frames are encoded), for the job:

    - staged release: `release="every"` drops the DiT before every decode (the decode's peak is the VAE alone;
      the next segment's `_denoise` reloads it), `"last"` only before the last segment's decode;
    - `save_dir`: write `latents_denorm.npy` exactly like speedlab/run_e2e.py (float32 of the denormalized
      latents passed to the decoder; overwritten per segment, so a one-segment clip matches the reference file)
      plus `latents_denorm_seg<N>.npy`;
    - meter stages release / decode / finish."""
    orig = getattr(gen_mod, "vae_decode", None)
    info = {"decodes": 0, "latents": []}
    if orig is None:
        yield info
        return

    def vae_decode(vae, z, *a, **k):
        info["decodes"] += 1
        seg = info["decodes"]
        if release == "every" or (release == "last" and seg >= n_segments):
            if meter is not None:
                meter.mark("release", segment=seg)
            gen._shutter_release_dit()
        if save_dir is not None:
            import numpy as np
            import mlx.core as mx

            save_dir.mkdir(parents=True, exist_ok=True)
            arr = np.asarray(z.astype(mx.float32))
            np.save(save_dir / "latents_denorm.npy", arr)
            np.save(save_dir / f"latents_denorm_seg{seg}.npy", arr)
            info["latents"].append(str(save_dir / f"latents_denorm_seg{seg}.npy"))
        if meter is not None:
            meter.mark("decode", segment=seg)
        out = orig(vae, z, *a, **k)
        if meter is not None:
            meter.mark("finish" if seg >= n_segments else "encode", segment=seg + (0 if seg >= n_segments else 1))
        return out

    gen_mod.vae_decode = vae_decode
    try:
        yield info
    finally:
        gen_mod.vae_decode = orig


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def write_mp4(frames, audio, out: Path, fps: int = FPS, crf: int = 14):
    """Same encode as the tree's scripts/generate.py write_mp4: H.264 crf 14, the given audio file muxed (aac)."""
    import numpy as np

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio:
        cmd += ["-i", str(audio), "-t", f"{len(frames) / fps:.3f}", "-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd.append(str(out))
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {out}")


def write_png(arr, path: Path):
    """uint8 [H, W, 3] -> PNG, with PIL when installed, else ffmpeg."""
    import numpy as np

    try:
        from PIL import Image

        Image.fromarray(np.asarray(arr)).save(path)
        return
    except ImportError:
        pass
    h, w = arr.shape[:2]
    subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                    "rgb24", "-s", f"{w}x{h}", "-i", "-", "-frames:v", "1", str(path)],
                   input=np.ascontiguousarray(arr).tobytes(), check=True)


@contextlib.contextmanager
def env_switches(env: dict[str, Optional[str]]):
    """Set (value) or remove (None) env vars for the job; on exit restore what was there before."""
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ----------------------------------------------------------------------------- the adapter


class LongCatAvatarAdapter(Adapter):
    """Options (registry config): code_dir (default <model_dir>/code; on the Mac ~/longcat-mlx/code-exact),
    weights_dir (default <model_dir>, or <model_dir>/weights when that exists; on the Mac
    ~/longcat-mlx/weights/LongCat-Video-Avatar-1.6-q8), cache_dir (umT5 prompt cache), vae_tile (64), steps (8),
    low_memory (LC_BLOCK_SPLIT=1 + LC_FFN_CHUNKS=ffn_chunks, default 4), vae_bf16, sdpa_heads (True), lora_swap
    (True), first_kv (False), seg_frames (93), memory_limit_gb, cache_limit_gb (2, as measured), keep_warm (True),
    compile (True), edit_size ("1920x1080"), personbox (path to the Apple Vision helper; default
    <code_dir>/speedlab/personbox)."""

    family = "longcat-avatar"
    kind = "video"

    def __init__(self, model_id: str, model_dir: Path, options: Optional[dict[str, Any]] = None,
                 *, duration_probe: Callable = probe_duration, image_probe: Callable = probe_image_size):
        super().__init__(model_id, model_dir, options)
        self._probe_duration = duration_probe
        self._probe_image = image_probe

    # ---- options
    @property
    def code_dir(self) -> Path:
        return Path(self.options.get("code_dir") or self.model_dir / "code").expanduser()

    @property
    def weights_dir(self) -> Path:
        w = self.options.get("weights_dir")
        if w:
            return Path(w).expanduser()
        return self.model_dir / "weights" if (self.model_dir / "weights").is_dir() else self.model_dir

    def _opt(self, key, default):
        return self.options.get(key, default)

    def env_for(self, plan: RenderPlan, person_crop_px: Optional[tuple[int, int]] = None,
                rung_id: Optional[str] = None) -> dict[str, Optional[str]]:
        """All LC_* switches for a job (None = must be absent). With `rung_id`, LC_BLOCK_SPLIT / LC_FFN_CHUNKS
        follow the rung (RUNG_SETTINGS); without it, the legacy `low_memory` option decides."""
        sdpa = self._opt("sdpa_heads", True)
        sdpa_v = "1" if sdpa is True else ("0" if sdpa is False else str(int(sdpa)))
        if rung_id is not None:
            rs = RUNG_SETTINGS[rung_id]
            split, ffn = rs["split"], rs["ffn"]
        else:
            split = ffn = _truthy(self._opt("low_memory", False))
        env: dict[str, Optional[str]] = {
            "LC_SDPA_HEADS": sdpa_v,
            "LC_LORA_SWAP": "1" if _truthy(self._opt("lora_swap", True)) else "0",
            "LC_BLOCK_SPLIT": "1" if split else "0",
            "LC_FFN_CHUNKS": str(int(self._opt("ffn_chunks", 4))) if ffn else "1",
            "LC_FIRST_KV": "1" if _truthy(self._opt("first_kv", False)) else "0",
            "LC_SEG_FRAMES": str(plan.seg_frames),
        }
        for k in BG_KEYS:
            env[k] = None
        if plan.room_lock and person_crop_px is not None:
            env.update({"LC_BG_ANCHOR": "1", "LC_BG_PERSON": f"{person_crop_px[0]},{person_crop_px[1]}",
                        "LC_BG_MARGIN": str(plan.bg_margin), "LC_BG_FEATHER": str(plan.bg_feather)})
        return env

    def load_signature(self, env: dict[str, Optional[str]], stream: Optional[dict] = None) -> tuple:
        """What a warm generator was built with; any difference = rebuild (see module docstring). A streamed
        DiT (stream rungs) is a different build from a resident one."""
        sig = (str(self.code_dir), str(self.weights_dir), env.get("LC_SDPA_HEADS"), env.get("LC_LORA_SWAP"),
               env.get("LC_BLOCK_SPLIT"), env.get("LC_FFN_CHUNKS"), bool(self._opt("compile", True)))
        if stream:
            sig += (("stream", int(stream["window"]), bool(stream.get("nocache"))),)
        return sig

    # ---- modes
    def stream_window(self, rung_id: str) -> Optional[int]:
        w = RUNG_SETTINGS[rung_id]["window"]
        if w is None:
            return None
        key = "stream_window_wide" if rung_id == "stream_wide" else "stream_window_tight"
        return max(1, int(self._opt(key, w)))

    @staticmethod
    def _total_memory(ctx: Optional[RunContext]) -> int:
        try:
            from shutter.core.memory import default_probe
            return int(default_probe().sample().total)
        except Exception:
            return 0

    def stream_cfg(self, rung_id: str) -> Optional[dict]:
        w = self.stream_window(rung_id)
        if w is None:
            return None
        return {"window": w, "nocache": _truthy(self._opt("stream_nocache", sys.platform == "darwin"))}

    def release_policy(self, rung_id: str) -> str:
        pol = RUNG_SETTINGS[rung_id]["release"]
        if pol == "every":
            pol = str(self._opt("staged_release", "every"))
            if pol not in ("every", "last"):
                raise ValueError("staged_release must be 'every' or 'last'")
        return pol

    def mode_defaults(self, ctx: Optional[RunContext] = None) -> dict[str, Any]:
        """Engine-level defaults for a job without `mode`: ctx.settings (default_mode / default_efficiency), then
        options; the legacy `low_memory: true` option means low_memory at 0.5 (staged)."""
        d: dict[str, Any] = {}
        if _truthy(self._opt("low_memory", False)):
            d = {"mode": "low_memory", "efficiency": 0.5}
        for src in (self.options, (ctx.settings if ctx is not None else None) or {}):
            if src.get("default_mode"):
                d["mode"] = src["default_mode"]
            if src.get("default_efficiency") is not None:
                d["efficiency"] = src["default_efficiency"]
        return d

    def sizes(self) -> dict[str, int]:
        """Byte counts from disk (header-only), {} when the weights are absent."""
        out: dict[str, int] = {}
        w = self.weights_dir
        try:
            from shutter.adapters.longcat_stream import dit_stream_sizes

            ds = dit_stream_sizes(w / "dit")
        except Exception:
            ds = None
        if ds:
            out.update({"dit": ds["total"], "nonblock": ds["nonblock"], "block": ds["block_max"]})
        loras = sorted((w / "lora").glob("*.safetensors")) if (w / "lora").is_dir() else []
        if loras:
            out["lora"] = max(f.stat().st_size for f in loras)
        return out

    def rung_estimates(self, request: JobRequest, plan: Optional[RenderPlan] = None,
                       n_segments: Optional[int] = None) -> dict[str, MemoryEstimate]:
        plan = plan or self.resolve(request)
        if n_segments is None:
            n_segments = len(segment_layout(request.duration_s, plan.seg_frames)) if request.duration_s else 1
        resident = self.resident_bytes()
        sizes = self.sizes()
        data = M.load_measured(self._opt("measured_path", None))
        size = M.size_key(plan.height, plan.width)
        out = {}
        for rid in M.RUNG_IDS:
            out[rid] = estimate_rung(plan, rid, resident_bytes=resident, sizes=sizes,
                                     measured=M.measured_for(self.model_id, size, rid, data), n_segments=n_segments,
                                     window=self.stream_window(rid))
        return out

    def _budget_from_sample(self, sample: Any, headroom: Optional[int]) -> Optional[int]:
        if not isinstance(sample, dict) or not sample.get("available"):
            return None
        return int(sample["available"]) - int(headroom or 0)

    def budget_for(self, request: JobRequest, ctx: Optional[RunContext] = None) -> Optional[int]:
        """budget = available - headroom: `params.budget_bytes` (the daemon may pass it), else the run context's
        memory sample and headroom, else (estimate time) this machine's probe and the configured headroom."""
        p = request.params or {}
        if p.get("budget_bytes") is not None:
            return int(p["budget_bytes"])
        if ctx is not None:
            try:
                return self._budget_from_sample(ctx.memory_sample(), (ctx.settings or {}).get("headroom_bytes"))
            except Exception:
                return None
        if not _truthy(self._opt("auto_budget", True)):
            return None
        try:
            from shutter.core.config import Config
            from shutter.core.memory import default_probe

            s = default_probe().sample().to_dict()
            return self._budget_from_sample(s, Config.load().headroom_for(int(s.get("total") or 0)))
        except Exception:
            return None

    def choose_rung(self, request: JobRequest, ctx: Optional[RunContext] = None, plan: Optional[RenderPlan] = None,
                    n_segments: Optional[int] = None) -> dict[str, Any]:
        choice = M.resolve_mode(request.params, self.mode_defaults(ctx))
        ests = self.rung_estimates(request, plan, n_segments)
        budget = self.budget_for(request, ctx)
        rid = M.fit_to_budget(ests, budget, choice.rung) if choice.step_down else choice.rung
        return {"choice": choice, "rung": rid, "estimates": ests, "budget": budget,
                "stepped_down_from": choice.rung if rid != choice.rung else None,
                "fits": budget is None or ests[rid].peak <= budget}

    # ---- contract
    def _image_size(self, request: JobRequest) -> Optional[tuple[int, int]]:
        if request.params.get("image_size"):
            return _parse_wh(request.params["image_size"])
        if request.image and Path(request.image).exists():
            try:
                return self._probe_image(request.image)
            except Exception:
                return None
        return None

    def resolve(self, request: JobRequest) -> RenderPlan:
        return resolve_render(request, self._image_size(request), self.options)

    def resident_bytes(self) -> Optional[int]:
        """8-bit DiT shards + the LoRA(s) that stay attached + VAE + audio encoder, from disk (None if absent)."""
        w = self.weights_dir
        if not (w / "dit").is_dir():
            return None
        total = sum(f.stat().st_size for f in (w / "dit").glob("*.safetensors"))
        loras = sorted((w / "lora").glob("*.safetensors")) if (w / "lora").is_dir() else []
        sizes = [f.stat().st_size for f in loras]
        if sizes:
            total += max(sizes) if _truthy(self._opt("lora_swap", True)) else sum(sizes)
        for sub in ("vae", "audio_encoder"):
            if (w / sub).is_dir():
                total += sum(f.stat().st_size for f in (w / sub).glob("*.safetensors"))
        return total or None

    def estimate(self, request: JobRequest) -> MemoryEstimate:
        """The estimate of the rung this job will run at: fast mode starts at `resident` and steps down the
        ladder only as far as needed to fit `budget = available - headroom`; low_memory starts at the slider's
        rung. When nothing fits, the lowest rung's estimate is returned and the guard makes the job wait."""
        plan = self.resolve(request)
        from shutter.adapters.longcat_speed import parse_speed

        parse_speed((request.params or {}).get("speed"))      # reject bad speed options at submit
        n = len(segment_layout(request.duration_s, plan.seg_frames)) if request.duration_s else 1
        pick = self.choose_rung(request, None, plan, n)
        rid = pick["rung"]
        est = pick["estimates"][rid]
        secs = estimate_rung_seconds(plan, rid, n, sizes=None, window=self.stream_window(rid))
        est.note += f"; mode {pick['choice'].mode}, rung {rid}"
        if pick["stepped_down_from"]:
            est.note += f" (stepped down from {pick['stepped_down_from']} to fit {pick['budget'] / GB:.1f} GB)"
        est.note += f"; ~{secs['total_s']:.0f} s for {n} segment(s) ({secs['seconds_per_step']} s/step)"
        return est

    def warm(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        """POST /v1/models/<id>/load: build the generator for this Mac's default plan through the same cache a
        request uses (_get_generator), and load the DiT the way the chosen step keeps it (streamed steps: only the
        non-block part stays; blocks are read from the SSD during a request)."""
        req = job.request
        plan = self.resolve(req)
        ctx.report(0.0, "preparing")
        base_env = self.env_for(plan)
        with env_switches(base_env):
            gen_mod = import_generation(self.code_dir)
        pick = self.choose_rung(req, ctx, plan, 1)
        rung_id = pick["rung"]
        if not pick["fits"]:
            from shutter.types import MemoryRefused

            raise MemoryRefused(f"{rung_id} needs ~{pick['estimates'][rung_id].steady / GB:.1f} GB to stay loaded, "
                                f"budget {pick['budget'] / GB:.1f} GB (available - headroom): waiting for memory")
        stream = self.stream_cfg(rung_id)
        env = self.env_for(plan, None, rung_id)
        ctx.check_cancelled()
        with env_switches(env):
            apply_import_time_switches(env)
            ctx.report(0.1, "loading model")
            key, gen = self._get_generator(ctx, gen_mod, env, plan, stream)
            if self.release_policy(rung_id) == "never":
                gen._load_dit()
        ctx.report(1.0, "loaded")
        note = ("Lowest memory and Very low memory keep only the small always-loaded part and the video decoder in "
                "memory; the rest of the model is read from the SSD during a request." if stream else
                "The whole model is in memory.")
        return {"rung": rung_id, "note": note}

    def duration(self, request: JobRequest) -> float:
        if request.duration_s:
            return float(request.duration_s)
        if not request.audio:
            raise ValueError("longcat-avatar needs an audio track")
        return self._probe_duration(request.audio)

    def plan(self, request: JobRequest) -> list[Shot]:
        """One shot per segment, all in chain 0 (each continues from the previous 13 frames, so they run in order
        on one node). Shot.seed is the exact integer upstream uses for that segment's noise
        (`segment_seed(request.seed, s)`); `run` passes request.seed as Settings.seed, and the tree adds
        7919 * segment itself, so a run of the planned shots reproduces a full run bit for bit."""
        d = self.duration(request)
        rp = self.resolve(request)
        secs = estimate_seconds(rp, 1)
        shots = []
        for s, (start, keep, nf) in enumerate(segment_layout(d, rp.seg_frames)):
            shots.append(Shot(index=s, start_s=start / FPS, duration_s=keep / FPS, seed=segment_seed(request.seed, s),
                              prompt=request.prompt, chain=0,
                              params={"framing": framing_of(request).value, "segment": s, "base_seed": request.seed, "first_frame": start,
                                      "new_frames": keep, "segment_frames": nf,
                                      "cond_frames": 0 if s == 0 else COND_FRAMES,
                                      "est_seconds": secs["segment_s"]}))
        return shots

    # ---- run
    def stream_resident_bytes(self, window: int) -> int:
        """What a streamed generator keeps: VAE + audio encoder + non-block DiT tensors + the window."""
        sz = self.sizes()
        w = self.weights_dir
        other = 0
        for sub in ("vae", "audio_encoder"):
            if (w / sub).is_dir():
                other += sum(f.stat().st_size for f in (w / sub).glob("*.safetensors"))
        nonblock = sz.get("nonblock") or int(NONBLOCK_GB * GB)
        group = (sz.get("block") or int((DIT_GB - NONBLOCK_GB) / DEPTH * GB)) + int((sz.get("lora") or LORA_GB * GB) / DEPTH)
        return int(other + nonblock + window * group)

    def _get_generator(self, ctx: RunContext, gen_mod, env, plan: RenderPlan, stream: Optional[dict] = None):
        key = f"longcat/{self.model_id}/generator"
        sig = self.load_signature(env, stream)
        cls = make_generator_class(gen_mod)
        vae_fp32 = False if plan.vae_bf16 else plan.vae_tile > 0

        def loader():
            paths = gen_mod.ModelPaths.from_variant(self.weights_dir)
            kw = dict(cache_dir=Path(self._opt("cache_dir", "~/.cache/longcat-avatar-mlx")).expanduser(),
                      vae_tile=plan.vae_tile, compile=bool(self._opt("compile", True)))
            if _accepts(gen_mod.AvatarGenerator.__init__, "vae_fp32"):
                kw["vae_fp32"] = vae_fp32
            g = cls(paths, **kw)
            g._shutter_sig = sig
            g._shutter_vae_fp32 = vae_fp32 if "vae_fp32" in kw else plan.vae_tile > 0
            g._shutter_stream_cfg = dict(stream) if stream else None
            return g

        if stream:
            size = self.stream_resident_bytes(int(stream["window"]))
        else:
            size = self.resident_bytes() or int(DEFAULT_RESIDENT_GB * GB)
        gen = ctx.models.get(key, loader, size)
        if getattr(gen, "_shutter_sig", None) != sig:
            ctx.models.release(key)
            gen = ctx.models.get(key, loader, size)
        # VAE precision / tiling can change per job without touching the DiT
        gen.vae_tile = plan.vae_tile
        if getattr(gen, "_shutter_vae_fp32", vae_fp32) != vae_fp32 and hasattr(gen_mod, "load_vae"):
            gen.vae = gen_mod.load_vae(gen.paths.components, fp32=vae_fp32)
            gen._shutter_vae_fp32 = vae_fp32
        return key, gen

    def _settings(self, gen_mod, request: JobRequest, plan: RenderPlan, size):
        fields = {f.name for f in dataclasses.fields(gen_mod.Settings)}
        kw = {"size": tuple(size), "seed": int(request.seed), "steps": plan.steps}
        for k, v in (request.params or {}).items():   # pass-through of the tree's own Settings fields
            if k in fields and k not in kw and k not in ("resolution",):
                kw[k] = v
        if "steps" not in fields and plan.steps != 8:
            raise RuntimeError("this LongCat tree has no Settings.steps; only 8 steps are supported")
        return gen_mod.Settings(**{k: v for k, v in kw.items() if k in fields})

    def _person_box_auto(self, png: Path) -> tuple[int, int]:
        exe = Path(self._opt("personbox", self.code_dir / "speedlab" / "personbox")).expanduser()
        out = subprocess.run([str(exe), str(png)], capture_output=True, text=True, check=True).stdout.split()
        x0, _, x1, _ = (float(v) for v in out[:4])
        if x0 < 0:
            raise RuntimeError("person_box auto: Apple Vision found no person; pass person_box 'x0,x1'")
        return int(x0), int(math.ceil(x1))

    def speed_for(self, request: JobRequest, plan: RenderPlan, *, stream: bool = False, gen_mod=None,
                  size: Optional[tuple[int, int]] = None):
        """`params.speed` -> longcat_speed.SpeedConfig (opt-in, NOT bit-identical), or None (the default). Records
        what cannot apply to this job (streaming rungs, trees without the hook point) and adds plan notes."""
        from shutter.adapters import longcat_speed as LSP

        speed = LSP.parse_speed((request.params or {}).get("speed"))
        if speed is None:
            return None
        h, w = size or (plan.height, plan.width)
        speed.hw = (h // 16, w // 16)
        LSP.refusals_for(speed, stream=stream, gen_mod=gen_mod)
        if speed.block_sparse is not None and "block_sparse" not in speed.refused:
            b = speed.block_sparse
            plan.notes.append(f"speed.block_sparse: sparsity {b.sparsity}, chunk {'x'.join(map(str, b.chunk))}, "
                              f"keep_cond {b.keep_cond}: {LSP.NOT_IDENTICAL} (the Avatar model was trained dense)")
        if speed.step_cache is not None and "step_cache" not in speed.refused:
            plan.notes.append(f"speed.step_cache: TeaCache threshold {speed.step_cache.threshold}: "
                              f"{LSP.NOT_IDENTICAL}; continuation (KV-cache) steps are not cached")
        for opt, why in speed.refused.items():
            plan.notes.append(f"speed.{opt} not applied: {why}")
        return speed

    def run(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        import numpy as np

        req = job.request
        if not req.image or not req.audio:
            raise ValueError("longcat-avatar needs `image` and `audio`")
        t0 = time.time()
        plan = self.resolve(req)
        duration = self.duration(req)
        shots = self.plan(req)
        work = Path(ctx.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = Path(req.output) if req.output else work / f"{job.id}.mp4"
        raw_path = out_path.with_name(out_path.stem + "-raw.mp4")
        ctx.report(0.0, "preparing")
        ctx.check_cancelled()

        # import-time switches must be in the environment before the first import
        base_env = self.env_for(plan)
        with env_switches(base_env):
            gen_mod = import_generation(self.code_dir)
        has_roi = importlib.util.find_spec("longcat_video_avatar.roi") is not None
        if plan.crop and not has_roi:
            raise RuntimeError(f"crop_box needs the patched tree (longcat_video_avatar.roi) in {self.code_dir}")
        if plan.room_lock and "LC_BG_ANCHOR" not in inspect.getsource(gen_mod):
            raise RuntimeError(f"room_lock needs the patched tree (LC_BG_ANCHOR) in {self.code_dir}")

        import mlx.core as mx

        if self._opt("memory_limit_gb", None):
            mx.set_memory_limit(int(float(self._opt("memory_limit_gb", 0)) * GB))
        if self._opt("cache_limit_gb", 2.0) is not None and hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(int(float(self._opt("cache_limit_gb", 2.0)) * GB))

        image = Path(req.image)
        size = (plan.plate_height, plan.plate_width)
        plate = box = None
        person_crop = None
        if plan.crop:
            roimod = importlib.import_module("longcat_video_avatar.roi")
            plate = roimod.plate_uint8(image, plan.plate_height, plan.plate_width)
            plate_png = work / "plate.png"
            write_png(plate, plate_png)
            box = roimod.parse_box(plan.crop, plate_png, plan.plate_width)
            crop_png = work / "roi.png"
            write_png(np.ascontiguousarray(plate[:, box[0]:box[1]]), crop_png)
            image = crop_png
            size = (plan.plate_height, box[1] - box[0])
            plan.width = size[1]
            plan.tokens = (size[0] // 16) * (size[1] // 16) * plan.latent_frames
            if plan.room_lock:
                if plan.person_box == "auto":
                    person_crop = self._person_box_auto(crop_png)
                else:
                    px0, px1 = (int(v) for v in plan.person_box.split(","))
                    person_crop = (px0 - box[0], px1 - box[0])
                if not (0 <= person_crop[0] < person_crop[1] <= size[1]):
                    raise ValueError(f"person box {person_crop} (crop px) is outside the crop {box}")
                if len(shots) > 1:
                    plan.notes.append("room lock applies to the first segment only (the tree anchors num_cond == 1)")

        # ---- mode / rung (docs/MODES.md): budget = available - headroom, step down only as far as needed
        pick = self.choose_rung(req, ctx, plan, len(shots))
        rung_id = pick["rung"]
        if not pick["fits"]:
            from shutter.types import MemoryRefused

            raise MemoryRefused(f"{rung_id} needs ~{pick['estimates'][rung_id].peak / GB:.1f} GB, budget "
                                f"{pick['budget'] / GB:.1f} GB (available - headroom): waiting for memory")
        if pick["stepped_down_from"]:
            plan.notes.append(f"fast mode stepped down from {pick['stepped_down_from']} to {rung_id} to fit "
                              f"{pick['budget'] / GB:.1f} GB")
            ctx.report(0.005, f"stepping down to {rung_id} to fit memory")
        stream = self.stream_cfg(rung_id)
        release = self.release_policy(rung_id)
        # prompt encode one block at a time (same result, ~2.5 GB instead of 11.4 GB): on for the streaming steps
        # and on Macs where the whole text encoder would not fit; `t5_stream` true/false forces it
        t5_opt = self._opt("t5_stream", None)
        t5_on = _truthy(t5_opt) if t5_opt is not None else (
            bool(stream) or 0 < self._total_memory(ctx) < 24 * GB)
        from shutter.adapters import longcat_t5
        if longcat_t5.install(gen_mod, t5_on) and t5_on:
            plan.notes.append("prompt encoder read one block at a time")
        speed = self.speed_for(req, plan, stream=bool(stream), gen_mod=gen_mod, size=size)
        env = self.env_for(plan, person_crop, rung_id)
        hook = StepHook(ctx, len(shots), plan.steps, lo=0.02, hi=0.9)
        meter = StageMeter(mx)
        meter.start("prep")
        save_latents = req.params.get("save_latents")
        save_dir = None
        if save_latents:
            save_dir = (Path(os.path.expanduser(str(save_latents))) if not isinstance(save_latents, bool)
                        else out_path.parent / (out_path.stem + "-latents"))
        decode_info: dict[str, Any] = {}
        key = None
        gen = None
        log_lines: list[str] = []

        def log(msg):
            log_lines.append(str(msg))

        def on_segment(frames_so_far, n):
            hook.segment_done(n)

        try:
            with env_switches(env):
                apply_import_time_switches(env)
                ctx.report(0.01, "loading model")
                key, gen = self._get_generator(ctx, gen_mod, env, plan, stream)
                gen._anchor_parts = None      # patched tree never clears it (see module docstring)
                gen._anchor = None
                gen._shutter_hook = hook
                gen._shutter_meter = meter
                reloads0, releases0 = gen._shutter_reloads, gen._shutter_releases
                # None (default, or everything refused): no-op unless a previous job mounted hooks
                gen._shutter_set_speed(speed if speed is not None and speed.active else None)
                st = self._settings(gen_mod, req, plan, size)
                kw = {"image": image, "duration": duration, "on_segment": on_segment, "log": log}
                with decode_hooks(gen_mod, gen, release=release, n_segments=len(shots), save_dir=save_dir,
                                  meter=meter) as decode_info:
                    frames, info = gen.generate(Path(req.audio), req.prompt, st, **kw)
                stream_stats = gen._shutter_stream_stats()
                if speed is not None and speed.cache is not None:
                    speed.cache.finish()
        finally:
            if gen is not None:
                gen._shutter_hook = None
                gen._shutter_meter = None
                gen._anchor_parts = None
                gen._anchor = None
        ctx.check_cancelled()
        meter.mark("post")

        if box is not None:
            ctx.report(0.9, "compositing crop")
            write_mp4(frames, req.audio, work / "crop.mp4")
            frames = importlib.import_module("longcat_video_avatar.roi").composite(frames, plate, box[0],
                                                                                    plan.crop_feather)
        ctx.report(0.92, "writing mp4")
        n_frames, (fh, fw) = len(frames), frames.shape[1:3]
        write_mp4(frames, req.audio, raw_path)
        del frames

        post = None
        if plan.upscale_to:
            ctx.check_cancelled()
            ctx.report(0.95, "upscale" + (f" + zoom cancel {plan.zoom_cancel:.4f}" if plan.zoom_cancel else ""))
            from shutter.addons.reels.upscale import upscale

            post = upscale(raw_path, out_path, plan.upscale_to, zoom_cancel=plan.zoom_cancel, fps=FPS,
                           backend=self._opt("upscale_backend", "auto"))
            final_size = list(plan.upscale_to)
        else:
            shutil.move(str(raw_path), str(out_path))
            raw_path = out_path
            final_size = [fw, fh]

        meter.close()
        summ = meter.summary()
        peak = summ["peak_bytes"]
        if not _truthy(self._opt("keep_warm", True)) and key is not None:
            ctx.models.release(key)
        ctx.report(1.0, "done")
        rung_info = M.rung(rung_id)
        res = {"output": str(out_path), "raw_output": str(raw_path), "frames": n_frames, "size": final_size,
               "render_size": [fw, fh], "seconds": round(time.time() - t0, 1), "segments": len(shots),
               "peak_memory_bytes": peak, "peak_bytes": peak, "plan": plan.to_dict(),
               "env": {k: v for k, v in env.items() if v},
               "info": {k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str, list))},
               "mode": pick["choice"].mode, "efficiency": pick["choice"].efficiency, "rung": rung_id,
               "identical": rung_info.get("identical", True) and not (speed is not None and speed.active),
               "estimate_peak_bytes": pick["estimates"][rung_id].peak,
               "budget_bytes": pick["budget"], "stages": meter.records, **summ,
               "dit_releases": gen._shutter_releases - releases0, "dit_reloads": gen._shutter_reloads - reloads0,
               "release_policy": release, "size_key": M.size_key(size[0], size[1])}
        if pick["stepped_down_from"]:
            res["stepped_down_from"] = pick["stepped_down_from"]
        if speed is not None:
            res["speed"] = speed.report(steps_run=len(meter.summary()["step_seconds"]) or None,
                                        depth=len(getattr(gen.dit, "blocks", ())) or DEPTH)
        if stream_stats:
            res["stream"] = stream_stats
        if decode_info.get("latents"):
            res["latents"] = decode_info["latents"]
            res["latents_dir"] = str(save_dir)
        if hook.seconds_per_step is not None:
            res["seconds_per_step"] = hook.seconds_per_step
        if box is not None:
            res["crop_box"] = list(box)
            res["crop_output"] = str(work / "crop.mp4")
        if person_crop is not None:
            res["person_box_crop_px"] = list(person_crop)
        if post:
            res["post"] = post
        res["log_tail"] = log_lines[-20:]
        return res
