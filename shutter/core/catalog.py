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

__all__ = ["CATALOG", "catalog_rows", "download", "cancel", "read_state", "preset_for", "doctor", "DownloadError"]

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
    return out


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
    if s.get("state") == "downloading" and not _pid_alive(s.get("pid")):
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
        if installed and st and st.get("state") in ("downloading", "stopped", "failed", "cancelled"):
            st = None if st.get("state") != "downloading" else st
        try:
            free = shutil.disk_usage(folder if folder.exists() else _existing_parent(folder)).free
        except OSError:
            free = 0
        preset = preset_for(mid, total)
        rows.append({
            "id": mid, "name": e["name"], "summary": e["summary"], "kind": e["kind"], "source": e["repo"],
            "size_bytes": int(e["size_gb"] * GB), "min_memory_bytes": int(e["min_memory_gb"] * GB),
            "fits_this_mac": total == 0 or total >= e["min_memory_gb"] * GB,
            "dir": str(folder), "installed": installed, "free_bytes": free,
            "notes": preset["notes"], "download": st,
        })
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


def _hf_download(repo: str, folder: Path) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=str(folder))


def _hf_total(repo: str) -> int:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    return sum(int(s.size or 0) for s in info.siblings)


def download(model_id: str, cfg: Optional[Config] = None, *, folder: Optional[str] = None,
             on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
             fetch: Optional[Callable[[str, Path], None]] = None,
             total_of: Optional[Callable[[str], int]] = None, poll_s: float = 1.0) -> dict[str, Any]:
    """Fetch `model_id` into the models folder, then register it. Blocks until it is done. `fetch` and
    `total_of` exist for tests. Raises DownloadError with a plain sentence when it cannot start or fails."""
    if model_id not in CATALOG:
        raise DownloadError(f"unknown model '{model_id}'. Available: {', '.join(CATALOG)}")
    cfg = cfg or Config.load()
    e = CATALOG[model_id]
    dest = Path(os.path.expanduser(folder)) if folder else models_home(cfg) / e["folder"]
    prior = read_state(cfg, model_id)
    if prior and prior.get("state") == "downloading" and int(prior.get("pid", 0)) != os.getpid():
        raise DownloadError(f"{e['name']} is already downloading")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        total = int((total_of or _hf_total)(e["repo"])) or int(e["size_gb"] * GB)
    except Exception:
        total = int(e["size_gb"] * GB)
    have = folder_bytes(dest)
    free = shutil.disk_usage(dest).free
    need = max(0, total - have) + 3 * GB
    base = dict(pid=os.getpid(), dir=str(dest), total_bytes=total, started=time.time())
    if free < need:
        msg = (f"not enough free disk space: {e['name']} needs {need / GB:.0f} GB more, "
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
    preset = register(cfg, model_id, dest)
    done = dict(state="done", done_bytes=total, notes=preset["notes"], **base)
    _write_state(cfg, model_id, **done)
    if on_progress:
        on_progress(done)
    return done


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
    if not s or s.get("state") != "downloading":
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
        elif st == "downloading":
            d = row["download"]
            add(row["name"], None, f"downloading, {d.get('done_bytes', 0) / GB:.1f} of "
                                   f"{d.get('total_bytes', 0) / GB:.1f} GB")
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
