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

GB = 1024**3

__all__ = ["CATALOG", "catalog_rows", "download", "cancel", "read_state", "preset_for", "doctor", "DownloadError",
           "register_build"]

RUNNING = ("downloading", "converting")

# Folders that hold the model itself. `code/` is the MLX port the adapter imports.
CATALOG: dict[str, dict[str, Any]] = {
    "longcat-avatar-1.6-8bit": {
        "name": "LongCat Video Avatar 1.6",
        "summary": "Talking video of a person from one photo and a voice recording.",
        "kind": "video",
        "adapter": "longcat-avatar",
        "repo": "SarthakRawool/LongCat-Video-Avatar-1.6-MLX-8bit-LowMem",
        "folder": "LongCat-Video-Avatar-1.6-MLX-8bit-LowMem",
        "size_gb": 37.5,
        "min_memory_gb": 16,
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
    "summary": "Short video clips from a text prompt. The smallest Wan model: quick, and fits a 16 GB Mac.",
    "kind": "video",
    "adapter": "wan", "family": "wan", "variant": "wan2.1-t2v-1.3b",
    "repo": "SarthakRawool/Wan2.1-T2V-1.3B-MLX-8bit-LowMem",
    "folder": "Wan2.1-T2V-1.3B-MLX-8bit-LowMem",
    "size_gb": 12.3,          # GiB (13.2 GB), the published repo
    "min_memory_gb": 16,
    "shape": WAN13_SHAPE,
    "starter": True,
    "checked_on": WAN13_CHECKED,
    "sample": {"prompt": "A red fox running through fresh snow in a pine forest, morning light, cinematic",
               "width": 512, "height": 320, "frames": 17, "steps": 12, "seed": 42},
    "needs": ["mlx>=0.31", "safetensors>=0.4", "huggingface_hub>=0.34", "transformers>=4.48",
              "sentencepiece", "Pillow", "imageio", "imageio-ffmpeg", "ftfy"],
    "check_files": ["shuttrr.json", "dit", "text_encoder", "tokenizer", "vae"],
}


# Other Wan models. No 8-bit builds of these are published yet, so a download fetches the official repo (bf16 / fp32) into
# <models>/sources/<repo> and then runs the converter into <models>/<id>-shuttrr-8bit. Source sizes are approximate
# (they include the umT5 text encoder and the VAE in their original precision). The built size and the smallest
# Mac are worked out from the model's shape (shutter.families.wan.spec) and the family's memory check, not typed in.
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
        "summary": "720p video from a text prompt, or from a photo and a prompt. One model for both.",
        "source_repo": "Wan-AI/Wan2.2-TI2V-5B", "source_size_gb": 34,
        "shape": dict(dim=3072, in_dim=48, out_dim=48, ffn_dim=14336, num_layers=30),
    },
    "wan2.2-t2v-a14b": {
        "name": "Wan 2.2 Text to Video A14B",
        "summary": "The largest Wan text-to-video model: two 14B experts, the best quality and the slowest. "
                   "A very large download.",
        "source_repo": "Wan-AI/Wan2.2-T2V-A14B", "source_size_gb": 126,
        "shape": dict(dim=5120, in_dim=16, out_dim=16, ffn_dim=13824, num_layers=40, dual=True),
    },
    "wan2.2-i2v-a14b": {
        "name": "Wan 2.2 Image to Video A14B",
        "summary": "Brings a photo to life as a video, guided by a prompt. Two 14B experts; a very large download.",
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
            "hidden": bool(src.get("hidden")),
            "needs": ["mlx>=0.31", "safetensors>=0.4", "huggingface_hub>=0.34", "transformers>=4.48",
                      "sentencepiece", "Pillow", "imageio", "imageio-ffmpeg", "ftfy"],
            "check_files": ["shuttrr.json", "text_encoder", "tokenizer", "vae"],
        }
    return out


CATALOG.update(_wan_entries())


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
        h, w, f, steps = WAN_SMALL_MAC_DEFAULTS
        out["options"] = {"default_size": [h, w], "default_frames": f, "default_steps": steps}
        out["notes"] = [
            f"This Mac has under {SMALL_MAC_BYTES // GB} GB of memory, so Wan makes {h}x{w} clips of {f} frames by "
            "default. Short 480x832 clips work with the tiled decode (vae_tile); 81-frame 480x832 clips need a "
            f"Mac with {SMALL_MAC_BYTES // GB} GB or more.",
        ]
    return out


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
        h, w, f, _ = WAN_SMALL_MAC_DEFAULTS
        return h, w, f
    if e.get("shape", {}).get("in_dim") == 48:     # TI2V 5B: 704x1280
        return 704, 1280, 121
    return 480, 832, 81


def fit_for(model_id: str, total_bytes: int, headroom: int) -> dict[str, Any]:
    """How a catalog model fits this Mac's memory (its total, not what is free right now), from the engine's own
    per-step estimates at the size it would run by default here:
      fits        the fastest step (resident) fits
      low_memory  only a lower step fits (Shuttrr steps down by itself; slower, same frames)
      needs       not even the lowest step fits: needs a Mac with `min_memory_gb`
    `label` is the words to show."""
    e = CATALOG[model_id]
    need_gb = int(e["min_memory_gb"])
    if not total_bytes:
        return {"level": "unknown", "label": "", "detail": "This Mac's memory could not be read."}
    try:
        if e["adapter"] == "wan":
            h, w, f = default_request(model_id, total_bytes)
            est = _wan_estimates(e["shape"], e.get("variant", model_id), h, w, f)
            at = f"{h}x{w}, {f} frames"
        elif e["adapter"] == "longcat-avatar":
            est = _longcat_estimates(total_bytes)
            at = "the default size"
        else:
            est = {}
            at = ""
    except Exception:
        est = {}
        at = ""
    budget = total_bytes - headroom
    if total_bytes < need_gb * GB:
        return {"level": "needs", "label": f"Needs {need_gb} GB",
                "detail": f"Needs a Mac with at least {need_gb} GB of memory; this one has {total_bytes / GB:.0f} GB."}
    if not est:
        return {"level": "fits", "label": "Fits", "detail": f"This Mac has {total_bytes / GB:.0f} GB of memory."}
    fast, low = est.get("resident", 0), min(est.values())
    # a model that needs more than half of this Mac's memory only runs with most other apps closed: say so
    tight = (f" It needs about {low / GB:.0f} GB free when it runs, so close other apps first."
             if low > total_bytes / 2 else "")
    if fast <= budget:
        return {"level": "fits", "label": "Fits",
                "detail": f"Runs at full speed here: about {fast / GB:.1f} GB at {at}.", "peak_bytes": fast}
    if low <= budget:
        return {"level": "low_memory", "label": "Fits in Low memory",
                "detail": f"Runs here in Low memory: about {low / GB:.1f} GB at {at} (the fastest step would "
                          f"need about {fast / GB:.0f} GB).{tight}", "peak_bytes": low}
    more = max(need_gb, int(-(-(low + headroom) // GB)))
    return {"level": "needs", "label": f"Needs {more} GB",
            "detail": f"Even Low memory needs about {low / GB:.0f} GB at {at}; this Mac has "
                      f"{total_bytes / GB:.0f} GB.", "peak_bytes": low}


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
            "label": "" if ok else f"Needs {need / GB:.0f} GB of free disk space ({free / GB:.0f} GB free)"}


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


def is_complete(folder: Path, entry: dict[str, Any]) -> bool:
    return folder.is_dir() and all((folder / f).exists() for f in entry["check_files"])


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
            "fit": fit, "disk": disk,
            "recommended": bool(e.get("starter")) and 0 < total < SMALL_MAC_BYTES and fit["level"] != "needs",
            "sample": e.get("sample"),
            "size_bytes": int(e["size_gb"] * GB), "min_memory_bytes": int(e["min_memory_gb"] * GB),
            "fits_this_mac": total == 0 or fit["level"] != "needs",
            "dir": str(folder), "installed": installed, "free_bytes": free,
            "notes": preset["notes"], "download": st,
            "needs_conversion": bool(e.get("needs_conversion")),
            "built_size_bytes": int(e.get("built_size_gb", e["size_gb"]) * GB),
            "size_approximate": bool(e.get("size_approximate")),
            "checked_on": checked,
        })
    # the recommended starter first, then models that fit, then the rest (stable within each group)
    order = {"fits": 0, "low_memory": 1, "unknown": 1, "needs": 2}
    rows.sort(key=lambda r: (not r["recommended"], order.get(r["fit"]["level"], 1)))
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


def _hf_download(repo: str, folder: Path) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=str(folder))


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
    if files_of is not None or total_of is None:
        try:
            files = list((files_of or _hf_files)(e["repo"]))
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
    free = shutil.disk_usage(dest).free
    need = max(0, total - have) + DISK_HEADROOM
    base = dict(pid=os.getpid(), dir=str(dest), total_bytes=total, started=time.time())
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
        (fetch or _hf_download)(e["repo"], dest)
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
