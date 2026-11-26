"""Models Shuttrr can fetch by itself, the download that fetches them, and the checks behind `shutter doctor`.

A download is one `shutter download <id>` process. It writes its progress to `<home>/downloads/<id>.json` once a
second, so the engine, the app and the dashboard only ever read a small file. An interrupted download picks up
where it stopped (Hugging Face keeps the partial files). When it finishes, the model is written into config.json
with settings that suit this Mac's memory.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from shutter.core.config import Config
from shutter.core.pinned_files import WAN5B_Q8, WAN13_BF16_TE, WAN13_Q8_TE

GB = 1024**3

__all__ = ["CATALOG", "catalog_rows", "download", "cancel", "read_state", "preset_for", "doctor", "DownloadError",
           "register_build"]

RUNNING = ("downloading", "converting")

#: a Wan build's text encoder: text_encoder/ (bf16) or text_encoder_q8/ (8-bit); either one makes the build complete
WAN_TE = ("text_encoder_q8", "text_encoder")

# Folders that hold the model itself. `code/` is the MLX port the adapter imports.
CATALOG: dict[str, dict[str, Any]] = {
    "longcat-avatar-1.6-8bit": {
        "name": "LongCat Video Avatar 1.6",
        "summary": "Talking video of a person from a photo and a voice.",
        "kind": "video",
        "adapter": "longcat-avatar",
        "repo": "SarthakRawool/LongCat-Video-Avatar-1.6-MLX-8bit-LowMem",
        "folder": "LongCat-Video-Avatar-1.6-MLX-8bit-LowMem",
        "size_gb": 37.5,
        "min_memory_gb": 24,
        # Hard minimum: below 24 GB the lowest step (12.3 GB measured) leaves almost nothing for macOS and apps,
        # so the catalog says "Doesn't fit" whatever the estimate (classify_fit).
        "min_total_gb": 24,
        "needs": ["mlx>=0.31", "safetensors>=0.4", "huggingface_hub>=0.34", "transformers>=4.48", "sentencepiece",
                  "librosa", "Pillow", "imageio", "imageio-ffmpeg", "ftfy", "pyloudnorm"],
        "check_files": ["dit", "text_encoder", "vae", "code/longcat_video_avatar/generation.py"],
    },
}


# Wan 2.1 T2V 1.3B: a published Shuttrr build (8-bit DiT, Wan's own umT5 + tokenizer + VAE), downloaded as is.
# On a Mac with under 24 GB it is the recommended first model ("starter"); `sample` is the request "Try a sample"
# sends (sized to finish in about a minute on a 16 GB M5: measured 54 s, 6.4 GB peak).
WAN13_SHAPE = dict(dim=1536, in_dim=16, out_dim=16, ffn_dim=8960, num_layers=30)
WAN13_CHECKED = [
    {"date": "2026-11-26", "mac": "MacBook Pro M5 (Mac17,2)", "memory_gb": 16, "mlx": "0.32.5",
     "size": "320x512", "frames": 33, "steps": 20, "identical": True,
     "peaks_gb": {"resident": 6.23, "split": 6.23, "staged": 4.61, "stream_wide": 4.67, "stream_tight": 4.67},
     "seconds": {"resident": 166, "split": 153, "staged": 163, "stream_wide": 169, "stream_tight": 160}},
    {"date": "2026-11-26", "mac": "MacBook Pro M5 (Mac17,2)", "memory_gb": 16, "mlx": "0.32.5",
     "size": "480x832", "frames": 33, "steps": 30, "vae_tile": True,
     "peaks_gb": {"staged": 6.93}, "seconds": {"staged": 676},
     "stopped": ["stream_wide", "stream_tight"]},
]
CATALOG["wan2.1-t2v-1.3b-8bit"] = {
    "name": "Wan 2.1 Text to Video 1.3B",
    "summary": "Short video clips from a text prompt.",
    "kind": "video",
    "adapter": "wan", "family": "wan", "variant": "wan2.1-t2v-1.3b",
    "repo": "SarthakRawool/Wan2.1-T2V-1.3B-MLX-8bit-LowMem",
    "folder": "Wan2.1-T2V-1.3B-MLX-8bit-LowMem",
    "size_gb": 12.3,          # GiB, the published repo (set from the pinned file list below)
    "min_memory_gb": 16,
    "shape": WAN13_SHAPE,
    "starter": True,
    "checked_on": WAN13_CHECKED,
    "sample": {"prompt": "A red fox running through fresh snow in a pine forest, morning light, cinematic",
               "width": 512, "height": 320, "frames": 17, "steps": 12, "seed": 42},
    "needs": ["mlx>=0.31", "safetensors>=0.4", "huggingface_hub>=0.34", "transformers>=4.48",
              "sentencepiece", "Pillow", "imageio", "imageio-ffmpeg", "ftfy"],
    "check_files": ["shuttrr.json", "dit", WAN_TE, "tokenizer", "vae"],
    # exactly this revision is fetched and checked; a copy of an earlier one (bf16 text encoder) still counts
    "revision": WAN13_Q8_TE["revision"], "files": WAN13_Q8_TE["files"],
    "legacy": [WAN13_BF16_TE],
}
if WAN13_Q8_TE["files"]:
    CATALOG["wan2.1-t2v-1.3b-8bit"]["size_gb"] = round(sum(f["size"] for f in WAN13_Q8_TE["files"]) / GB, 1)


# Other Wan models. No 8-bit builds of these are published yet, so a download fetches the official repo (bf16 / fp32) into
# <models>/sources/<repo> and then runs the converter into <models>/<id>-shuttrr-8bit. Source sizes are approximate
# (they include the umT5 text encoder and the VAE in their original precision). The built size and the smallest
# Mac are worked out from the model's shape (shutter.families.wan.spec) and the family's memory check, not typed in.
# Wan 2.2 TI2V 5B, the Shuttrr 8-bit build converted from the official repo. Peaks are each request's largest stage
# (denoise or decode), prompt encoded once. Mac Studio runs used an MLX memory limit and 4 steps (the peak does not
# depend on the step count); they say what a bigger Mac needs and are not used for a 16 GB Mac.
_M5 = {"mac": "MacBook Pro M5 (Mac17,2)", "memory_gb": 16, "mlx": "0.32.5"}
WAN5B_CHECKED = [
    {"date": "2026-11-26", **_M5, "size": "320x512", "frames": 25, "steps": 20, "identical": True,
     "peaks_gb": {"staged": 6.34, "stream_wide": 4.34, "stream_tight": 4.34},
     "seconds": {"staged": 72, "stream_wide": 75, "stream_tight": 74},
     "note": "picture too soft at this size; resident keeps the DiT through the decode and swaps a 16 GB Mac"},
    {"date": "2026-11-26", **_M5, "size": "480x832", "frames": 49, "steps": 30, "identical": True,
     "peaks_gb": {"staged": 6.92, "stream_tight": 6.30}, "seconds": {"staged": 445, "stream_tight": 456},
     "note": "8-bit text encoder (text_encoder_q8); with the bf16 encoder: same peaks, 455 / 473 s, clip PSNR 29.1 dB "
             "/ SSIM 0.966 between the two"},
    {"date": "2026-11-26", **_M5, "size": "480x832", "frames": 121, "steps": 30,
     "peaks_gb": {"stream_tight": 6.33}, "seconds": {"stream_tight": 1354}},
    {"date": "2026-11-26", **_M5, "size": "832x480", "frames": 25, "steps": 30, "image": True,
     "peaks_gb": {"stream_tight": 6.85}, "seconds": {"stream_tight": 259}},
    {"date": "2026-11-26", **_M5, "size": "704x1280", "frames": 49, "steps": 20, "vae_tile": True,
     "peaks_gb": {"stream_tight": 5.86}, "seconds": {"stream_tight": 987}},
    # the 16 GB default since the quality check (480x832 looked soft): 704x1280 x 25 frames, 30 steps, tiled decode
    {"date": "2026-11-26", **_M5, "size": "704x1280", "frames": 25, "steps": 30, "vae_tile": True,
     "peaks_gb": {"stream_tight": 5.44}, "seconds": {"stream_tight": 613},
     "note": "denoise 16.4 s/step at 2.5 GB, tiled decode 119 s; other apps open"},
    {"date": "2026-11-26", "mac": "Mac Studio M5 Max (Mac17,14)", "memory_gb": 128, "mlx": "0.32.3",
     "size": "704x1280", "frames": 49, "steps": 50, "peaks_gb": {"resident": 16.32}, "seconds": {"resident": 548},
     "note": "quality check; 30 steps: 341 s, same peak, as sharp to the eye"},
    {"date": "2026-11-26", "mac": "Mac Studio M5 Max (Mac17,14)", "memory_gb": 128, "mlx": "0.32.4",
     "size": "704x1280", "frames": 121, "steps": 4, "identical": True, "memory_limit_gb": {"stream_tight": 16,
                                                                                          "staged": 16, "resident": 32},
     "peaks_gb": {"resident": 16.37, "staged": 11.74, "stream_tight": 11.34},
     "seconds_per_step": {"resident": 27.6, "staged": 27.7, "stream_tight": 28.1}},
]

_WAN_SOURCES: dict[str, dict[str, Any]] = {
    # superseded by the published build above; kept so `shutter download wan2.1-t2v-1.3b` and builds already
    # made from the official repo keep working, but not listed (hidden) unless it is installed or downloading
    "wan2.1-t2v-1.3b": {
        "name": "Wan 2.1 Text to Video 1.3B (converted here)",
        "summary": "The same model, downloaded from the official repo and converted on this Mac.",
        "source_repo": "Wan-AI/Wan2.1-T2V-1.3B", "source_size_gb": 17,
        "shape": WAN13_SHAPE, "checked_on": WAN13_CHECKED, "hidden": True,
    },
    "wan2.2-ti2v-5b": {
        "name": "Wan 2.2 Text and Image to Video 5B",
        "summary": "Video from a text prompt, or from a photo and a prompt.",
        "source_repo": "Wan-AI/Wan2.2-TI2V-5B", "source_size_gb": 34,
        "shape": dict(dim=3072, in_dim=48, out_dim=48, ffn_dim=14336, num_layers=30),
        "checked_on": WAN5B_CHECKED,
        # 16 GB M5: the model's own size, 704x1280, with the tiled decode: 25 frames (1 s at 24 fps), 30 steps
        # peaks at 5.8 GB and takes ~10 min. Below 704x1280 the picture goes soft (measured: 480x832 has about a
        # quarter of the fine detail at any step count), so a smaller size is not offered as the default.
        "small_mac_defaults": (704, 1280, 25, 30),
        "small_mac_options": {"vae_tile": True},
        "small_mac_note": ("This Mac has under {gb} GB of memory, so Wan 2.2 5B makes {h}x{w} clips of {f} frames "
                           "({steps} steps, tiled decode, about 10 minutes on a 16 GB M5) by default. 49 frames "
                           "(2 s) take about 20 minutes; smaller sizes are faster but look soft."),
    },
    "wan2.2-t2v-a14b": {
        "name": "Wan 2.2 Text to Video A14B",
        "summary": "Best-quality video from a text prompt. Very large and slow.",
        "source_repo": "Wan-AI/Wan2.2-T2V-A14B", "source_size_gb": 126,
        "shape": dict(dim=5120, in_dim=16, out_dim=16, ffn_dim=13824, num_layers=40, dual=True),
    },
    "wan2.2-i2v-a14b": {
        "name": "Wan 2.2 Image to Video A14B",
        "summary": "Brings a photo to life, guided by a prompt. Very large.",
        "source_repo": "Wan-AI/Wan2.2-I2V-A14B", "source_size_gb": 126,
        "shape": dict(dim=5120, in_dim=36, out_dim=16, ffn_dim=13824, num_layers=40, dual=True),
    },
}

TEXT_ENCODER_GB = 11      # umT5-XXL in bf16, as the converter writes it


def wan_checkpoint(model_id: str):
    """A Checkpoint for a catalog Wan model, built from its shape alone (for memory checks before a download)."""
    from shutter.families.base import Checkpoint
    from shutter.families.wan.spec import WanShape

    src = _WAN_SOURCES[model_id]
    cfg = WanShape(**src["shape"]).to_config(model_id)
    return Checkpoint(path="", family="wan", variant=model_id, task=cfg["model_type"], config=cfg)


def _wan_numbers(model_id: str) -> tuple[float, int]:
    """(built size in GB, smallest Mac in GB that passes the family memory check at the lowest step)."""
    from shutter.families import family
    from shutter.families.wan.spec import WanShape, part_sizes

    ck = wan_checkpoint(model_id)
    shape = WanShape(**_WAN_SOURCES[model_id]["shape"])
    dit = shape.dit_bytes(8) * (2 if shape.dual else 1)
    vae = sum(p.total for p in part_sizes(ck.config, image_encoder=ck.task in ("i2v", "ti2v"))
              if p.role in ("decoder", "image_encoder"))
    built = (dit + TEXT_ENCODER_GB * GB + vae) / GB
    wan = family("wan")
    min_gb = next(g for g in range(1, 1025) if not wan.check_memory(ck, g * GB))
    return round(built, 1), min_gb


def _wan_entries() -> dict[str, dict[str, Any]]:
    out = {}
    for mid, src in _WAN_SOURCES.items():
        built, min_gb = _wan_numbers(mid)
        out[mid] = {
            "name": src["name"], "summary": src["summary"], "kind": "video", "adapter": "wan",
            "family": "wan", "variant": mid, "shape": dict(src["shape"]),
            "repo": src["source_repo"], "source_repo": src["source_repo"],
            "folder": f"{mid}-shuttrr-8bit",
            "source_folder": f"sources/{src['source_repo'].split('/')[-1]}",
            "needs_conversion": True,
            "size_gb": src["source_size_gb"], "size_approximate": True,
            "built_size_gb": built,
            "min_memory_gb": min_gb,
            "checked_on": list(src.get("checked_on", [])),   # Macs it was measured on; empty until it is
            **({"small_mac_defaults": tuple(src["small_mac_defaults"])} if src.get("small_mac_defaults") else {}),
            **({"small_mac_note": src["small_mac_note"]} if src.get("small_mac_note") else {}),
            **({"small_mac_options": dict(src["small_mac_options"])} if src.get("small_mac_options") else {}),
            "hidden": bool(src.get("hidden")),
            "needs": ["mlx>=0.31", "safetensors>=0.4", "huggingface_hub>=0.34", "transformers>=4.48",
                      "sentencepiece", "Pillow", "imageio", "imageio-ffmpeg", "ftfy"],
            "check_files": ["shuttrr.json", WAN_TE, "tokenizer", "vae"],
        }
    return out


CATALOG.update(_wan_entries())

# Wan 2.2 TI2V 5B is published as a Shuttrr build too (8-bit DiT, 8-bit umT5, Wan 2.2 VAE): a download fetches that
# pinned revision as is, with no conversion. Its shape, measurements and small-Mac defaults stay as above.
_w5 = CATALOG["wan2.2-ti2v-5b"]
for _k in ("needs_conversion", "source_folder", "size_approximate", "built_size_gb"):
    _w5.pop(_k, None)
_w5.update({
    "repo": "SarthakRawool/Wan2.2-TI2V-5B-MLX-8bit-LowMem",
    "folder": "Wan2.2-TI2V-5B-MLX-8bit-LowMem",
    "check_files": ["shuttrr.json", "dit", WAN_TE, "tokenizer", "vae"],
    "revision": WAN5B_Q8["revision"], "files": WAN5B_Q8["files"],
    "size_gb": round(sum(f["size"] for f in WAN5B_Q8["files"]) / GB, 1),
})


class DownloadError(RuntimeError):
    pass


def models_home(cfg: Config) -> Path:
    return Path(cfg.extra.get("models_dir") or os.path.join(cfg.home, "models")).expanduser()


def state_path(cfg: Config, model_id: str) -> Path:
    return Path(cfg.home) / "downloads" / f"{model_id}.json"


def total_memory() -> int:
    try:
        from shutter.core.memory import default_probe
        return int(default_probe().sample().total)
    except Exception:
        return 0


def preset_for(model_id: str, total_bytes: int) -> dict[str, Any]:
    """Settings for this model on a Mac with `total_bytes` of memory: {"options", "settings", "notes"}.
    Only small Macs get anything: Fast mode already steps down to fit on the others."""
    entry = CATALOG[model_id]
    out: dict[str, Any] = {"options": {}, "settings": {}, "notes": []}
    if entry["adapter"] == "longcat-avatar" and 0 < total_bytes < 24 * GB:
        out["options"] = {"vae_bf16": True}
        out["settings"] = {"default_mode": "low_memory", "default_efficiency": 1.0}
        out["notes"] = [
            "This Mac has under 24 GB of memory, so Shuttrr uses Lowest memory and the half-size final decode.",
            "The half-size decode is not bit-identical to the full one (a 46.6 dB match).",
        ]
    elif entry["adapter"] == "wan" and 0 < total_bytes < SMALL_MAC_BYTES:
        h, w, f, steps = small_mac_defaults(model_id)
        out["options"] = {"default_size": [h, w], "default_frames": f, "default_steps": steps,
                          **entry.get("small_mac_options", {})}
        if entry.get("small_mac_note"):
            out["notes"] = [entry["small_mac_note"].format(gb=SMALL_MAC_BYTES // GB, h=h, w=w, f=f, steps=steps)]
        else:
            out["notes"] = [
                f"This Mac has under {SMALL_MAC_BYTES // GB} GB of memory, so Wan makes {h}x{w} clips of {f} frames "
                "by default. Short 480x832 clips work with the tiled decode (vae_tile); 81-frame 480x832 clips need "
                f"a Mac with {SMALL_MAC_BYTES // GB} GB or more.",
            ]
    return out


def small_mac_defaults(model_id: str) -> tuple[int, int, int, int]:
    """(height, width, frames, steps) a Wan model makes by default on a Mac under SMALL_MAC_BYTES: the model's own
    measured choice when it has one (`small_mac_defaults`), else the family's."""
    return tuple(CATALOG[model_id].get("small_mac_defaults") or WAN_SMALL_MAC_DEFAULTS)


SMALL_MAC_BYTES = 24 * GB
WAN_SMALL_MAC_DEFAULTS = (320, 512, 33, 20)     # height, width, frames, steps: measured on a 16 GB M5 (docs/WAN.md)


def _wan_estimates(shape: dict[str, Any], variant: str, h: int, w: int, f: int,
                   tiled: bool = False) -> dict[str, int]:
    """Engine memory estimate (bytes) per memory step for a Wan model of this shape, before it is downloaded."""
    from shutter.core import modes as M
    from shutter.core import parts as P
    from shutter.families.wan.spec import WanShape, part_sizes, stage_needs

    ws = WanShape(**shape)
    cfg = ws.to_config(variant)
    ps = part_sizes(cfg, image_encoder=cfg["model_type"] in ("i2v", "ti2v"))
    needs = stage_needs(cfg, h, w, f, tiled_decode=tiled)
    return {r: int(P.estimate(ps, needs, r)["peak"]) for r in M.RUNG_IDS}


def _longcat_estimates(total_bytes: int) -> dict[str, int]:
    from shutter.adapters import longcat as L
    from shutter.core import modes as M
    from shutter.types import JobRequest

    small = 0 < total_bytes < L.SMALL_MAC_BYTES
    plan = L.resolve_render(JobRequest(model="longcat", params={"size": L.SMALL_MAC_LOAD_SIZE} if small else {}))
    if 0 < total_bytes < SMALL_MAC_BYTES:
        plan.vae_bf16 = True      # preset_for turns the half-size decode on for these Macs
    return {r: int(L.estimate_rung(plan, r).peak) for r in M.RUNG_IDS}


def default_request(model_id: str, total_bytes: int) -> tuple[int, int, int]:
    """(height, width, frames) a Wan model makes by default on a Mac with `total_bytes` of memory."""
    e = CATALOG[model_id]
    if 0 < total_bytes < SMALL_MAC_BYTES:
        h, w, f, _ = small_mac_defaults(model_id)
        return h, w, f
    if e.get("shape", {}).get("in_dim") == 48:     # TI2V 5B: 704x1280
        return 704, 1280, 121
    return 480, 832, 81


# One fit classifier for the app, the dashboard and the CLI (docs/API.md, "fit"). No percentage reserve: a model fits
# when its lowest step's peak fits in this Mac's memory minus what macOS itself keeps (its wired memory, measured on
# this Mac). It never promises a fit because an untested SSD-streaming estimate squeezes it in.
MACOS_BASELINE = 3 * GB          # what macOS keeps for itself when it cannot be measured (wired memory)
MACOS_BASELINE_MIN = int(2.5 * GB)
MACOS_BASELINE_MAX = 6 * GB
EVERYDAY_APPS = 4 * GB           # "Recommended" also leaves this much for a browser and a chat app
MAC_SIZES_GB = (8, 16, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512)
FIT_LABELS = {"recommended": "Recommended", "barely": "Fits but barely", "no_fit": "Doesn't fit", "unknown": ""}
TIGHT = "Tight on this Mac. Close other apps first."
UNTESTED = "Not tested on a Mac this size yet."

# LongCat, measured on an M5 Max (README "Measured results"): MLX peak per memory step, 93 frames, 8 steps.
CATALOG["longcat-avatar-1.6-8bit"]["checked_on"] = [
    {"date": "2026-11-20", "mac": "MacBook Pro M5 Max", "memory_gb": 128, "size": "480x832", "frames": 93,
     "steps": 8, "identical": True, "peaks_gb": {"resident": 31.5, "staged": 12.3, "stream_tight": 12.9}},
    {"date": "2026-11-20", "mac": "MacBook Pro M5 Max", "memory_gb": 128, "size": "704x1312", "frames": 93,
     "steps": 8, "identical": True, "peaks_gb": {"resident": 34.5, "stream_tight": 13.8}},
]


def default_request_any(model_id: str, total_bytes: int) -> tuple[int, int, int]:
    """(height, width, frames) the model makes by default on this Mac, for any catalog adapter."""
    e = CATALOG[model_id]
    if e["adapter"] == "longcat-avatar":
        from shutter.adapters import longcat as L
        h, w = (480, 832) if 0 < total_bytes < L.SMALL_MAC_BYTES else (704, 1312)
        return h, w, 93
    return default_request(model_id, total_bytes)


def _measured_entry(e: dict[str, Any], h: int, w: int, f: int, tiled: bool = False) -> dict[str, Any]:
    """The model's `checked_on` entry for this exact request size and decode (tiled or not), or {}."""
    for c in e.get("checked_on") or []:
        if c.get("size") == f"{h}x{w}" and int(c.get("frames", 0)) == f and bool(c.get("vae_tile")) == bool(tiled):
            return c
    return {}


def _default_tiled(e: dict[str, Any], total_bytes: int) -> bool:
    """Whether a Mac with `total_bytes` decodes with the tiled VAE by default (a small Mac, when the model's preset
    turns it on there)."""
    return bool(0 < total_bytes < SMALL_MAC_BYTES and (e.get("small_mac_options") or {}).get("vae_tile"))


def _measured_peaks(e: dict[str, Any], h: int, w: int, f: int, tiled: bool = False) -> dict[str, int]:
    """Measured peak (bytes) per memory step for this exact request size, from the model's `checked_on`; {} if none."""
    c = _measured_entry(e, h, w, f, tiled)
    return {r: int(float(g) * GB) for r, g in (c.get("peaks_gb") or {}).items()}


def _fast_peak(c: dict[str, Any], peaks: dict[str, int]) -> int:
    """Peak of the step a request runs at full speed: resident when it was measured; otherwise (a model whose
    resident step does not fit the Mac it was measured on) the quickest measured step, by its measured time."""
    if peaks.get("resident"):
        return peaks["resident"]
    secs = {r: float(t) for r, t in (c.get("seconds") or {}).items() if r in peaks}
    if secs:
        return peaks[min(secs, key=secs.get)]
    return 0


def macos_baseline(total_bytes: int) -> int:
    """What macOS itself keeps on this Mac: its wired memory now (clamped to 2.5..6 GB), or 3 GB when the Mac being
    asked about is not this one (or the number cannot be read)."""
    try:
        from shutter.core.memory import default_probe
        s = default_probe().sample()
        if int(s.total) == int(total_bytes) and s.wired:
            return int(min(MACOS_BASELINE_MAX, max(MACOS_BASELINE_MIN, int(s.wired))))
    except Exception:
        pass
    return MACOS_BASELINE


def _mac_size_for(peak: int, floor_gb: int) -> int:
    need = (peak + MACOS_BASELINE) / GB
    return next((g for g in MAC_SIZES_GB if g >= need and g >= floor_gb), max(floor_gb, int(-(-need // 1))))


def classify_fit(model_id: str, total_bytes: int, baseline: Optional[int] = None) -> dict[str, Any]:
    """How a catalog model fits a Mac with `total_bytes` of memory (its total, not what is free right now), at the
    size it would make by default there. usable = total - what macOS keeps (`macos_baseline`, its wired memory):
      recommended  measured at this size, and its full speed peak (resident, or the quickest measured step when
                   resident was not measured) fits in usable with 4 GB to spare for everyday apps; for Wan also
                   measured on a Mac this size or smaller and at most half of the Mac's memory
      barely       its lowest step's peak fits in usable (or it is only an estimate)
      no_fit       its lowest step's peak is more than usable, or the Mac is under the model's smallest Mac
                   (`min_total_gb`, an explicit minimum total memory, wins over any estimate)
    Uses the measured peaks when there are any for this size (and only those), otherwise the engine's estimates.
    Returns {"level", "label", "reason", "detail", "peak_bytes", "measured", "at", "usable_bytes"}."""
    e = CATALOG[model_id]
    if not total_bytes:
        return {"level": "unknown", "label": "", "reason": "This Mac's memory could not be read.",
                "detail": "This Mac's memory could not be read.", "peak_bytes": 0, "measured": False, "at": "",
                "usable_bytes": 0}
    mac_gb = total_bytes / GB
    base = macos_baseline(total_bytes) if baseline is None else int(baseline)
    usable = max(0, int(total_bytes) - base)
    try:
        h, w, f = default_request_any(model_id, total_bytes)
        at = f"{h}x{w}, {f} frames"
    except Exception:
        h = w = f = 0
        at = ""
    # the decode this Mac uses by default (small Macs may get the tiled one from the model's preset)
    tiled = _default_tiled(e, total_bytes)
    entry = _measured_entry(e, h, w, f, tiled) if h else {}
    peaks = _measured_peaks(e, h, w, f, tiled) if h else {}
    measured = bool(peaks)
    if not peaks:
        try:
            if e["adapter"] == "wan":
                peaks = _wan_estimates(e["shape"], e.get("variant", model_id), h, w, f, tiled=tiled)
            elif e["adapter"] == "longcat-avatar":
                peaks = _longcat_estimates(total_bytes)
        except Exception:
            peaks = {}
    if peaks and not measured:
        # an estimate never earns the SSD-streaming steps: those numbers are the least trustworthy, so judge an
        # untested model by its lowest step that keeps the whole model in memory
        peaks = {r: v for r, v in peaks.items() if not r.startswith("stream")} or peaks
    low = min(peaks.values()) if peaks else 0
    fast = _fast_peak(entry, peaks) if measured else peaks.get("resident", 0)
    measured_gb = float(entry.get("memory_gb") or 0)
    floor_gb = int(e.get("min_memory_gb") or 0)

    def out(level: str, reason: str, peak: int) -> dict[str, Any]:
        return {"level": level, "label": FIT_LABELS[level], "reason": reason, "detail": reason,
                "peak_bytes": int(peak), "measured": measured, "at": at, "usable_bytes": usable}

    min_total = int(e.get("min_total_gb") or 0)
    if min_total and total_bytes < min_total * GB:
        return out("no_fit", f"Needs a Mac with {min_total} GB or more. This Mac has {mac_gb:.0f} GB.", low)
    if total_bytes < floor_gb * GB or not peaks or low > usable:
        need = _mac_size_for(low, floor_gb)
        if need <= mac_gb:
            need = next((g for g in MAC_SIZES_GB if g > mac_gb), int(mac_gb) * 2)
        return out("no_fit", f"Needs about {need} GB. This Mac has {mac_gb:.0f} GB.", low)
    # Wan: Recommended only from a measurement on a Mac this size or smaller, with the full speed peak at most half
    # of this Mac's memory (the rule the Wan builds were measured against). Every model: room for everyday apps.
    strict = e["adapter"] == "wan"
    if (measured and fast and fast + EVERYDAY_APPS <= usable
            and (not strict or (fast <= total_bytes // 2 and (not measured_gb or measured_gb <= mac_gb + 0.5)))):
        return out("recommended", "Runs at full speed on this Mac." if fast == peaks.get("resident") else
                   "Runs well on this Mac in a lower memory step.", fast)
    roomy = low + EVERYDAY_APPS <= usable
    if strict and measured and measured_gb > mac_gb + 0.5 and fast and fast + EVERYDAY_APPS <= usable:
        return out("barely", f"Measured only on a {measured_gb:.0f} GB Mac; it should run here, but this size of "
                   "Mac is not tested yet.", fast)
    if not measured:
        return out("barely", UNTESTED if roomy else f"{UNTESTED} {TIGHT}", low)
    if roomy:
        return out("barely", "Runs only in a lower memory step here. It may be slow.", low)
    return out("barely", TIGHT, low)


def memory_line(model_id: str, total_bytes: int) -> str:
    """One line on the memory the model uses on this Mac (not its download size): measured peaks at its default
    request when there are any, else the engine's estimate, marked as such."""
    e = CATALOG[model_id]
    try:
        h, w, f = default_request_any(model_id, total_bytes)
    except Exception:
        return ""
    tiled = _default_tiled(e, total_bytes)
    peaks = _measured_peaks(e, h, w, f, tiled)
    where = ""
    if peaks:
        c = _measured_entry(e, h, w, f, tiled)
        mac = str(c.get("mac") or "").split(" (")[0]
        gb = f" {c['memory_gb']} GB" if c.get("memory_gb") else ""
        where = f" (measured on a{gb} {mac})" if mac else " (measured)"
    else:
        try:
            if e["adapter"] == "wan":
                peaks = _wan_estimates(e["shape"], e.get("variant", model_id), h, w, f, tiled=tiled)
            elif e["adapter"] == "longcat-avatar":
                peaks = _longcat_estimates(total_bytes)
        except Exception:
            peaks = {}
        peaks = {r: v for r, v in peaks.items() if not r.startswith("stream")} or peaks
        where = " (an estimate; not measured yet)"
    if not peaks:
        return ""
    fast, low = peaks.get("resident"), min(peaks.values())
    if fast and fast > low * 1.05:
        return f"Uses about {fast / GB:.1f} GB of memory at full speed, {low / GB:.1f} GB at its lowest step{where}."
    return f"Uses about {low / GB:.1f} GB of memory{where}."


def makes_line(model_id: str, total_bytes: int) -> str:
    """One short line on what the model makes by default on this Mac, e.g. "Makes 320x512 clips on this Mac"."""
    try:
        h, w, _f = default_request_any(model_id, total_bytes)
    except Exception:
        return ""
    if CATALOG[model_id]["adapter"] == "longcat-avatar":
        return f"Makes {'480p' if h < 700 else '720p'} talking videos on this Mac"
    return f"Makes {h}x{w} clips on this Mac"


def fit_for(model_id: str, total_bytes: int, headroom: int = 0) -> dict[str, Any]:
    """Kept for callers of the old name; `headroom` no longer changes the answer (see classify_fit)."""
    return classify_fit(model_id, total_bytes)


def disk_for(entry: dict[str, Any], free: int, done_bytes: int = 0, installed: bool = False) -> dict[str, Any]:
    """Disk space a download needs (what is left to fetch, plus the converted build when there is one, plus room
    to spare) against `free`. `label` is empty when it fits."""
    if installed:
        return {"ok": True, "need_bytes": 0, "label": ""}
    size = int(entry["size_gb"] * GB)
    need = max(0, size - int(done_bytes or 0)) + DISK_HEADROOM
    if entry.get("needs_conversion"):
        need += int(entry.get("built_size_gb", 0) * GB)
    ok = free <= 0 or free >= need
    return {"ok": ok, "need_bytes": need,
            "label": "" if ok else f"Not enough free disk ({need / GB:.0f} GB needed, {free / GB:.0f} GB free)"}


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def read_state(cfg: Config, model_id: str) -> Optional[dict[str, Any]]:
    p = state_path(cfg, model_id)
    try:
        s = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if s.get("state") in RUNNING and not _pid_alive(s.get("pid")):
        s["state"] = "stopped"   # the process died (sleep, reboot, quit): Resume carries on
    return s


def _write_state(cfg: Config, model_id: str, **fields: Any) -> None:
    p = state_path(cfg, model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields.update(id=model_id, updated=time.time())
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(fields))
    os.replace(tmp, p)


def already_downloaded(folder: Path, entry: dict[str, Any], files: Optional[list[dict[str, Any]]]) -> bool:
    """True when the model's main files are there and every file in Hugging Face's list has its exact size (no
    partial file left from a stopped download). Without a file list (offline) this is never True: a download then
    goes through Hugging Face's own resume, which skips finished files. A complete copy of one of the entry's
    `legacy` revisions also counts (it keeps working, so it is not fetched again)."""
    if not is_complete(folder, entry):
        return False
    if any(folder.glob(".cache/huggingface/download/**/*.incomplete")):
        return False
    if files and _sizes_match(folder, files):
        return True
    return legacy_revision(folder, entry) is not None


def legacy_revision(folder: Path, entry: dict[str, Any]) -> Optional[str]:
    """The earlier revision `folder` is a complete copy of (every file of its pinned list, same sizes), else None."""
    for old in entry.get("legacy") or []:
        if old.get("files") and _sizes_match(folder, old["files"]):
            return old.get("revision") or "earlier"
    return None


def _sizes_match(folder: Path, files: list[dict[str, Any]]) -> bool:
    for f in files:
        p = folder / str(f.get("path") or f.get("rfilename") or "")
        size = f.get("size")
        if not p.is_file() or (size is not None and p.stat().st_size != int(size)):
            return False
    return True


def is_complete(folder: Path, entry: dict[str, Any]) -> bool:
    """The model's main files are there. A check_files item that is a tuple is satisfied by any one of its paths."""
    def there(f: Any) -> bool:
        return any((folder / x).exists() for x in f) if isinstance(f, (tuple, list)) else (folder / f).exists()
    return folder.is_dir() and all(there(f) for f in entry["check_files"])


def folder_bytes(folder: Path) -> int:
    """Bytes on disk under `folder`, partial downloads included; Hugging Face's own bookkeeping is left out of
    nothing on purpose: `.cache/huggingface/download/*.incomplete` is where a file grows before it is moved."""
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                st = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            total += st.st_size
    return total


def catalog_rows(cfg: Optional[Config] = None) -> list[dict[str, Any]]:
    """One row per downloadable model: what it is, whether it is installed, and any download in progress."""
    cfg = cfg or Config.load()
    total = total_memory()
    rows = []
    for mid, e in CATALOG.items():
        spec = cfg.models.get(mid) or {}
        folder = Path(os.path.expanduser(spec["dir"])) if spec.get("dir") else models_home(cfg) / e["folder"]
        installed = bool(spec) and is_complete(folder, e)
        st = read_state(cfg, mid)
        if e.get("hidden") and not installed and not (st and st.get("state") in RUNNING):
            continue
        if installed and st and st.get("state") in RUNNING + ("stopped", "failed", "cancelled"):
            st = None if st.get("state") not in RUNNING else st
        try:
            free = shutil.disk_usage(folder if folder.exists() else _existing_parent(folder)).free
        except OSError:
            free = 0
        preset = preset_for(mid, total)
        checked = e.get("checked_on")
        if installed and e.get("adapter") == "wan":
            try:
                checked = json.loads((folder / "shuttrr.json").read_text()).get("checked_on", checked)
            except (OSError, ValueError):
                pass
        fit = fit_for(mid, total, cfg.headroom_for(total) if total else 0)
        disk = disk_for(e, free, (st or {}).get("done_bytes", 0) if (st or {}).get("state") != "converting" else 0,
                        installed)
        rows.append({
            "id": mid, "name": e["name"], "summary": e["summary"], "kind": e["kind"], "source": e["repo"],
            "fit": fit, "disk": disk, "makes": makes_line(mid, total), "memory": memory_line(mid, total),
            "recommended": fit["level"] == "recommended",
            "sample": e.get("sample"),
            "size_bytes": int(e["size_gb"] * GB), "min_memory_bytes": int(e["min_memory_gb"] * GB),
            "fits_this_mac": total == 0 or fit["level"] != "no_fit",
            "dir": str(folder), "installed": installed, "free_bytes": free,
            "notes": preset["notes"], "download": st,
            "needs_conversion": bool(e.get("needs_conversion")),
            "built_size_bytes": int(e.get("built_size_gb", e["size_gb"]) * GB),
            "size_approximate": bool(e.get("size_approximate")),
            "checked_on": checked,
        })
    # Recommended, then Fits but barely, then Doesn't fit (stable within each group)
    order = {"recommended": 0, "barely": 1, "unknown": 1, "no_fit": 2}
    rows.sort(key=lambda r: order.get(r["fit"]["level"], 1))
    return rows


def _existing_parent(p: Path) -> Path:
    p = Path(p)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def register(cfg: Config, model_id: str, folder: Path, total_bytes: Optional[int] = None) -> dict[str, Any]:
    """Write the model into config.json with this Mac's preset. Existing options the user set are kept."""
    e = CATALOG[model_id]
    preset = preset_for(model_id, total_memory() if total_bytes is None else total_bytes)
    cfg = Config.load(cfg.path)   # re-read: the engine may have saved settings while we downloaded
    spec = dict(cfg.models.get(model_id) or {})
    options = dict(preset["options"])
    options.update(spec.get("options") or {})
    spec.update(adapter=e["adapter"], dir=str(folder), options=options)
    cfg.models[model_id] = spec
    for k, v in preset["settings"].items():
        setattr(cfg, k, v)
    cfg.save()
    return preset


def register_build(cfg: Config, model_id: str, folder: Path, adapter: str = "wan") -> dict[str, Any]:
    """Write a model the user built (`shutter convert --add`) into config.json. Options already set are kept."""
    cfg = Config.load(cfg.path)
    spec = dict(cfg.models.get(model_id) or {})
    spec.update(adapter=adapter, dir=str(Path(folder).expanduser().resolve()))
    cfg.models[model_id] = spec
    cfg.save()
    return spec


def _hf_download(repo: str, folder: Path, revision: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download
    # sample clips on the model page are not part of the model
    snapshot_download(repo_id=repo, local_dir=str(folder), revision=revision or None, ignore_patterns=["samples/*"])


def _hf_total(repo: str) -> int:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    return sum(int(s.size or 0) for s in info.siblings)


def _hf_files(repo: str) -> list[dict[str, Any]]:
    """Hugging Face's file list for `repo`: [{"path", "size", "sha256"}]; sha256 is None for small non-LFS files."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    out = []
    for s in info.siblings:
        lfs = getattr(s, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None and not isinstance(lfs, dict) else (lfs or {}).get("sha256")
        out.append({"path": s.rfilename, "size": int(s.size or 0) if s.size is not None else None, "sha256": sha})
    return out


DISK_HEADROOM = 5 * GB   # free space a download leaves on top of the files it still has to fetch


def _sha256(path: Path, on_bytes: Optional[Callable[[int], None]] = None) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            if on_bytes:
                on_bytes(len(chunk))
    return h.hexdigest()


def verify_files(folder: Path, files: Optional[list[dict[str, Any]]], *, hash_files: bool = True,
                 on_bytes: Optional[Callable[[int], None]] = None) -> tuple[list[str], str]:
    """Check a finished download against Hugging Face's file list. Returns (bad paths, how it was checked):
    "sha256" when every LFS file's checksum matched, "sizes" when only sizes could be checked, "files" when no
    list was available (offline), in which case only the model's own check_files were looked at."""
    if not files:
        return [], "files"
    bad: list[str] = []
    hashed = False
    for f in files:
        p = folder / f["path"]
        try:
            size = p.stat().st_size
        except OSError:
            bad.append(f["path"])
            continue
        if f.get("size") is not None and size != int(f["size"]):
            bad.append(f["path"])
            continue
        if hash_files and f.get("sha256"):
            hashed = True
            if _sha256(p, on_bytes) != f["sha256"]:
                bad.append(f["path"])
    return bad, ("sha256" if hashed else "sizes")


def download(model_id: str, cfg: Optional[Config] = None, *, folder: Optional[str] = None,
             on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
             fetch: Optional[Callable[[str, Path], None]] = None,
             total_of: Optional[Callable[[str], int]] = None, poll_s: float = 1.0,
             files_of: Optional[Callable[[str], list[dict[str, Any]]]] = None) -> dict[str, Any]:
    """Fetch `model_id` into the models folder, check it against Hugging Face's file list, then register it.
    Blocks until it is done. `fetch`, `total_of` and `files_of` exist for tests (with `total_of` alone no file
    list is fetched). Raises DownloadError with a plain sentence when it cannot start or fails."""
    if model_id not in CATALOG:
        raise DownloadError(f"unknown model '{model_id}'. Available: {', '.join(CATALOG)}")
    cfg = cfg or Config.load()
    e = CATALOG[model_id]
    dest = Path(os.path.expanduser(folder)) if folder else models_home(cfg) / e["folder"]
    prior = read_state(cfg, model_id)
    if prior and prior.get("state") in RUNNING and int(prior.get("pid", 0)) != os.getpid():
        raise DownloadError(f"{e['name']} is already {prior['state']}")
    if e.get("needs_conversion"):
        return _download_and_convert(model_id, cfg, e, dest, on_progress=on_progress, fetch=fetch,
                                     total_of=total_of, poll_s=poll_s)
    dest.mkdir(parents=True, exist_ok=True)
    files: Optional[list[dict[str, Any]]] = None
    if files_of is not None:
        try:
            files = list(files_of(e["repo"]))
        except Exception:
            files = None
    elif e.get("files"):
        files = list(e["files"])   # the pinned revision's list: no request needed, checksums known ahead
    elif total_of is None:
        try:
            files = list(_hf_files(e["repo"]))
        except Exception:
            files = None   # offline or Hugging Face unreachable: sizes are checked against nothing, said below
    try:
        if total_of is not None:
            total = int(total_of(e["repo"]))
        else:
            total = sum(int(f.get("size") or 0) for f in files or [])
        total = total or int(e["size_gb"] * GB)
    except Exception:
        total = int(e["size_gb"] * GB)
    have = folder_bytes(dest)
    base = dict(pid=os.getpid(), dir=str(dest), total_bytes=total, started=time.time())
    if already_downloaded(dest, e, files):
        # Every file is there with Hugging Face's size: nothing to fetch. Register it (if it was not) and stop.
        preset = register(cfg, model_id, dest)
        old = None if files and _sizes_match(dest, files) else legacy_revision(dest, e)
        detail = "Already downloaded: every file is there with the size Hugging Face lists"
        if old:
            detail += " (an earlier version of this model, which keeps working)"
            base["total_bytes"] = total = folder_bytes(dest)
        done = dict(state="done", done_bytes=total, notes=preset["notes"], verified="sizes", already=True,
                    verified_detail=detail, legacy_revision=old, **base)
        _write_state(cfg, model_id, **done)
        if on_progress:
            on_progress(done)
        return done
    free = shutil.disk_usage(dest).free
    need = max(0, total - have) + DISK_HEADROOM
    if free < need:
        msg = (f"not enough free disk space: {e['name']} needs {need / GB:.0f} GB free "
               f"(what is left to download plus {DISK_HEADROOM // GB} GB to spare), "
               f"this disk has {free / GB:.0f} GB free")
        _write_state(cfg, model_id, state="failed", done_bytes=have, error=msg, **base)
        raise DownloadError(msg)

    stop = threading.Event()
    cancelled = threading.Event()

    def tick() -> dict[str, Any]:
        s = dict(state="downloading", done_bytes=min(total, folder_bytes(dest)), **base)
        _write_state(cfg, model_id, **s)
        if on_progress:
            on_progress(s)
        return s

    def watch() -> None:
        while not stop.wait(poll_s):
            tick()

    def on_term(signum, frame):  # noqa: ARG001
        cancelled.set()
        raise KeyboardInterrupt

    old = None
    if threading.current_thread() is threading.main_thread():
        old = signal.signal(signal.SIGTERM, on_term)
    tick()
    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        if fetch is not None:
            fetch(e["repo"], dest)
        else:
            _hf_download(e["repo"], dest, e.get("revision"))
    except KeyboardInterrupt:
        stop.set()
        _write_state(cfg, model_id, state="cancelled", done_bytes=folder_bytes(dest), **base)
        raise DownloadError("download stopped. Start it again to carry on from where it stopped") from None
    except Exception as ex:
        stop.set()
        msg = _plain_error(ex)
        _write_state(cfg, model_id, state="failed", done_bytes=folder_bytes(dest), error=msg, **base)
        raise DownloadError(msg) from ex
    finally:
        stop.set()
        t.join(timeout=5)
        if old is not None:
            signal.signal(signal.SIGTERM, old)
    if not is_complete(dest, e):
        msg = "the download finished but files are missing. Start it again to fetch the rest"
        _write_state(cfg, model_id, state="failed", done_bytes=folder_bytes(dest), error=msg, **base)
        raise DownloadError(msg)
    # Check every file against Hugging Face's list. Files that do not match are deleted so a restart fetches them.
    checked = [0]

    def hashed(n: int) -> None:
        checked[0] += n
        _write_state(cfg, model_id, state="downloading", phase="verifying", done_bytes=total,
                     verified_bytes=checked[0], **base)

    _write_state(cfg, model_id, state="downloading", phase="verifying", done_bytes=total, verified_bytes=0, **base)
    bad, how = verify_files(dest, files, on_bytes=hashed)
    if bad:
        for rel in bad:
            try:
                (dest / rel).unlink()
            except OSError:
                pass
        msg = (f"{len(bad)} downloaded file{'s' if len(bad) != 1 else ''} did not match Hugging Face's list "
               f"(first: {bad[0]}). They were removed. Start the download again to fetch them")
        _write_state(cfg, model_id, state="failed", done_bytes=folder_bytes(dest), error=msg, **base)
        raise DownloadError(msg)
    _remove_stale(dest, e, files)
    preset = register(cfg, model_id, dest)
    verified = {"sha256": "every file's size and checksum match Hugging Face's list",
                "sizes": "every file's size matches Hugging Face's list (no checksums were published)",
                "files": "Hugging Face could not be reached to check the files, so only the model's main "
                         "files were checked to be there"}[how]
    done = dict(state="done", done_bytes=total, notes=preset["notes"], verified=how, verified_detail=verified,
                **base)
    _write_state(cfg, model_id, **done)
    if on_progress:
        on_progress(done)
    return done


def _remove_stale(dest: Path, e: dict[str, Any], files: Optional[list[dict[str, Any]]]) -> None:
    """After a checked download of the pinned revision: delete files that only an earlier revision had (a stopped
    download of the old version resumed as the new one would otherwise keep, e.g., the 11 GB bf16 text encoder)."""
    if not files or not e.get("legacy"):
        return
    keep = {f["path"] for f in files}
    for old in e["legacy"]:
        for f in old.get("files") or []:
            if f["path"] not in keep:
                try:
                    (dest / f["path"]).unlink()
                except OSError:
                    pass


def _wan_preflight(model_id: str, cfg: Config, e: dict[str, Any], src_dir: Path, dest: Path,
                   total_bytes: int) -> dict[str, Path]:
    """Checks before a big Wan download, so it never runs only to fail at the end. Returns the parts to borrow
    from installed models ({part: folder}). Raises DownloadError with the reason."""
    from shutter.families import family
    from shutter.families.wan import sources

    ck = wan_checkpoint(model_id)
    if total_bytes:
        reasons = family("wan").check_memory(ck, total_bytes)
        if reasons:
            raise DownloadError(f"{e['name']} cannot run on this Mac, so Shuttrr will not download it. "
                                + " ".join(reasons) + f" It needs a Mac with at least {e['min_memory_gb']} GB.")
    # The official repos ship the text encoder and VAE as PyTorch .pth files.
    ck.parts = {}
    ck.missing = list(sources.PARTS)
    have_torch = sources.torch_available()
    borrow = sources.find_parts(ck, sources.candidate_dirs(_model_roots(cfg), exclude=[dest, src_dir]),
                                names=["text_encoder", "tokenizer", "vae"], have_torch=False)
    if not have_torch:
        lacking = [n for n in ("text_encoder", "vae") if n not in borrow]
        if lacking:
            what = " and ".join("the text encoder" if n == "text_encoder" else "the VAE" for n in lacking)
            vae_note = (" Nothing else has the Wan 2.2 VAE that this model needs, so PyTorch is required."
                        if "vae" in lacking and sources.want_vae(ck) == "wan2.2" else
                        " An installed LongCat model or Wan build can supply them instead.")
            raise DownloadError(
                f"{e['name']} comes with {what} as PyTorch (.pth) files, and PyTorch is not installed in "
                f"Shuttrr's Python, so the download could not be converted.{vae_note} Install PyTorch with: "
                f"{sys.executable} -m pip install torch")
    return borrow


def _model_roots(cfg: Config) -> list[Path]:
    roots = [models_home(cfg)]
    for spec in cfg.models.values():
        if spec.get("dir") and str(spec.get("adapter", "")).startswith(("wan", "longcat")):
            roots.append(Path(os.path.expanduser(str(spec["dir"]))))
    return roots


def _download_and_convert(model_id: str, cfg: Config, e: dict[str, Any], dest: Path, *,
                          on_progress: Optional[Callable[[dict[str, Any]], None]],
                          fetch: Optional[Callable[[str, Path], None]],
                          total_of: Optional[Callable[[str], int]], poll_s: float) -> dict[str, Any]:
    """A model with no published Shuttrr build: fetch the source repo, then convert it into `dest`. The convert step
    writes its progress into the same state file with state "converting"."""
    from shutter.families.wan import sources
    from shutter.families.wan.convert import ConvertError, convert

    src_dir = models_home(cfg) / e["source_folder"]
    base0 = dict(pid=os.getpid(), dir=str(dest), source_dir=str(src_dir), started=time.time())
    try:
        borrow = _wan_preflight(model_id, cfg, e, src_dir, dest, total_memory())
    except DownloadError as ex:
        _write_state(cfg, model_id, state="failed", done_bytes=0, total_bytes=int(e["size_gb"] * GB),
                     error=str(ex), **base0)
        raise
    already_built = is_complete(dest, e)
    if not already_built:
        src_dir.mkdir(parents=True, exist_ok=True)
        try:
            total = int((total_of or _hf_total)(e["repo"])) or int(e["size_gb"] * GB)
        except Exception:
            total = int(e["size_gb"] * GB)
        have = folder_bytes(src_dir)
        free = shutil.disk_usage(_existing_parent(dest)).free
        need = max(0, total - have) + int(e["built_size_gb"] * GB) + 3 * GB
        base = dict(base0, total_bytes=total)
        if free < need:
            msg = (f"not enough free disk space: {e['name']} needs {need / GB:.0f} GB more (the original download "
                   f"plus the converted build), this disk has {free / GB:.0f} GB free")
            _write_state(cfg, model_id, state="failed", done_bytes=have, error=msg, **base)
            raise DownloadError(msg)
        _run_with_progress(cfg, model_id, base, "downloading", src_dir, total,
                           lambda: (fetch or _hf_download)(e["repo"], src_dir), on_progress, poll_s)

        # convert
        built_total = int(e["built_size_gb"] * GB)
        cbase = dict(base0, total_bytes=built_total)
        stage = ["Checking the download"]

        def report(line: str) -> None:
            stage[0] = line.strip()

        def run_convert() -> None:
            from shutter.families import inspect as inspect_ckpt
            ck = inspect_ckpt(src_dir)
            if ck is None:
                raise ConvertError("the download is not a Wan model Shuttrr recognises")
            # parts the download lacks, or has only as .pth without PyTorch: from the models found before
            extra = {k: borrow[k] for k in sources.needed(ck) if k in borrow}
            # an interrupted conversion left files with the same names: convert writes over them
            convert(src_dir, dest, bits=8, text_encoder=extra.get("text_encoder"), vae=extra.get("vae"),
                    tokenizer=extra.get("tokenizer"), report=report)

        _run_with_progress(cfg, model_id, cbase, "converting", dest, built_total, run_convert, on_progress,
                           poll_s, stage=stage)
    if not is_complete(dest, e):
        msg = "the conversion finished but the build is incomplete. Start it again to redo the conversion"
        _write_state(cfg, model_id, state="failed", done_bytes=folder_bytes(dest), error=msg,
                     total_bytes=int(e["built_size_gb"] * GB), **base0)
        raise DownloadError(msg)
    register(cfg, model_id, dest)
    notes = [f"Converted to Shuttrr's 8-bit build in {dest}."]
    if src_dir.exists():
        notes.append(f"The original download is in {src_dir} ({folder_bytes(src_dir) / GB:.0f} GB). Shuttrr does "
                     "not need it any more; delete it to free the space.")
    done = dict(state="done", done_bytes=folder_bytes(dest), total_bytes=folder_bytes(dest), notes=notes, **base0)
    _write_state(cfg, model_id, **done)
    if on_progress:
        on_progress(done)
    return done


def _run_with_progress(cfg: Config, model_id: str, base: dict[str, Any], state: str, watch_dir: Path, total: int,
                       work: Callable[[], None], on_progress: Optional[Callable[[dict[str, Any]], None]],
                       poll_s: float, stage: Optional[list[str]] = None) -> None:
    """Run `work` while writing {state, done_bytes} once every `poll_s`; failures become DownloadError and a
    'failed' / 'cancelled' state."""
    from shutter.families.wan.convert import ConvertError

    stop = threading.Event()

    def tick() -> None:
        s = dict(state=state, done_bytes=min(total, folder_bytes(watch_dir)) if watch_dir.exists() else 0, **base)
        if stage:
            s["stage"] = stage[0]
        _write_state(cfg, model_id, **s)
        if on_progress:
            on_progress(s)

    def watch() -> None:
        while not stop.wait(poll_s):
            tick()

    def on_term(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    old = None
    if threading.current_thread() is threading.main_thread():
        old = signal.signal(signal.SIGTERM, on_term)
    tick()
    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        work()
    except KeyboardInterrupt:
        stop.set()
        _write_state(cfg, model_id, state="cancelled", done_bytes=folder_bytes(watch_dir), **base)
        raise DownloadError(f"{'download' if state == 'downloading' else 'conversion'} stopped. Start it again "
                            "to carry on from where it stopped") from None
    except ConvertError as ex:
        stop.set()
        msg = "the conversion failed: " + str(ex).strip()
        _write_state(cfg, model_id, state="failed", done_bytes=folder_bytes(watch_dir), error=msg[:600], **base)
        raise DownloadError(msg) from ex
    except Exception as ex:
        stop.set()
        msg = _plain_error(ex)
        if state == "converting":
            msg = "the conversion failed: " + msg
        _write_state(cfg, model_id, state="failed", done_bytes=folder_bytes(watch_dir), error=msg, **base)
        raise DownloadError(msg) from ex
    finally:
        stop.set()
        t.join(timeout=5)
        if old is not None:
            signal.signal(signal.SIGTERM, old)


def _plain_error(ex: BaseException) -> str:
    text = f"{type(ex).__name__}: {ex}".strip()
    low = text.lower()
    if isinstance(ex, ImportError):
        return "the Hugging Face downloader is not installed. Run the installer again (it adds huggingface_hub)"
    if "connection" in low or "timed out" in low or "name resolution" in low or "offline" in low:
        return "the connection dropped. Start the download again to carry on from where it stopped"
    if "no space" in low:
        return "the disk is full. Free some space, then start the download again"
    return text.splitlines()[0][:300]


def cancel(model_id: str, cfg: Optional[Config] = None) -> bool:
    """Stop a running download (its partial files stay, so it can resume). True if one was running."""
    cfg = cfg or Config.load()
    s = read_state(cfg, model_id)
    if not s or s.get("state") not in RUNNING:
        return False
    try:
        os.kill(int(s["pid"]), signal.SIGTERM)
    except (OSError, ValueError, KeyError):
        return False
    return True


# ---------------------------------------------------------------- doctor


def doctor(cfg: Optional[Config] = None) -> list[dict[str, Any]]:
    """Checks for a first run. Each row: {"check", "ok" (True / False / None = warning), "detail"}."""
    import importlib.util

    from shutter.core.config import add_private_bin

    add_private_bin()
    cfg = cfg or Config.load()
    rows: list[dict[str, Any]] = []

    def add(check: str, ok: Optional[bool], detail: str) -> None:
        rows.append({"check": check, "ok": ok, "detail": detail})

    mac = sys.platform == "darwin" and platform.machine() == "arm64"
    add("Apple silicon Mac", mac, f"{platform.system()} {platform.machine()}"
        + ("" if mac else ". Shuttrr runs models with MLX, which needs an Apple silicon Mac"))
    add("Python 3.11 or newer", sys.version_info >= (3, 11), platform.python_version())
    total = total_memory()
    if total:
        add("Memory", True if total >= 16 * GB else None,
            f"{total / GB:.0f} GB" + ("" if total >= 16 * GB else ". The smallest model needs 16 GB"))
    else:
        add("Memory", None, "could not be read")
    for mod, why in (("mlx", "runs the models"), ("huggingface_hub", "downloads models"),
                     ("transformers", "reads prompts for LongCat")):
        found = importlib.util.find_spec(mod) is not None
        add(f"Python package {mod}", found, why if found else f"missing ({why}). Run the installer again")
    ff = shutil.which("ffmpeg")
    add("ffmpeg", bool(ff), ff if ff else "missing. Run the installer again (it adds its own copy)")
    try:
        free = shutil.disk_usage(_existing_parent(models_home(cfg))).free
        add("Free disk space", True if free >= 45 * GB else None, f"{free / GB:.0f} GB free"
            + ("" if free >= 45 * GB else ". LongCat is a 38 GB download"))
    except OSError:
        pass
    any_model = False
    for row in catalog_rows(cfg):
        any_model = any_model or row["installed"]
        st = (row["download"] or {}).get("state")
        if row["installed"]:
            add(row["name"], True, f"installed in {row['dir']}")
        elif st in RUNNING:
            d = row["download"]
            add(row["name"], None, f"{st}, {d.get('done_bytes', 0) / GB:.1f} of "
                                   f"{d.get('total_bytes', 0) / GB:.1f} GB")
        elif row["needs_conversion"]:
            continue       # optional models that are converted on download: not worth a line until asked for
        else:
            add(row["name"], None, f"not downloaded yet. Run: shutter download {row['id']}")
        for n in row["notes"] if row["installed"] else []:
            add("Note", None, n)
    for mid, spec in cfg.models.items():
        if mid in CATALOG:
            continue
        d = Path(os.path.expanduser(str(spec.get("dir", ""))))
        add(f"Model {mid}", d.is_dir(), str(d) if d.is_dir() else f"folder not found: {d}")
        any_model = any_model or d.is_dir()
    if not any_model:
        add("A model to run", None, "none yet. Shuttrr still starts, with only the built-in test model")
    return rows
