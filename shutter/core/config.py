"""Shutter configuration: ~/.shutter/config.json (the directory can be moved with the
SHUTTER_HOME environment variable).

Unknown keys are kept and written back by `save()`, so newer and older versions can share
one file. Paths may use `~`; relative paths are relative to the home directory.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = ["Config", "default_home", "auto_headroom", "add_private_bin"]

DEFAULT_GPU_LOCK_DIRS_MAC = ["~/studio-gpu.lock.d", "~/ImaginationStudio/queue/.render.lock"]


def default_home() -> Path:
    env = os.environ.get("SHUTTER_HOME")
    return Path(os.path.expanduser(env)) if env else Path(os.path.expanduser("~/.shutter"))


def add_private_bin() -> None:
    """Append <home>/bin (where the installer puts its own ffmpeg and uv) to PATH when the folder exists,
    so a Mac without Homebrew still finds them. Existing PATH entries keep priority."""
    d = default_home() / "bin"
    if not d.is_dir():
        return
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if str(d) not in parts:
        parts.append(str(d))
        os.environ["PATH"] = os.pathsep.join(parts)


def auto_headroom(total_bytes: int) -> int:
    """Since 0.2.3 there is no automatic reserve: a request may use all the memory macOS can give it right now,
    and the watchdog (core/memory.Watchdog) protects the Mac while it runs. Kept for callers of the old name."""
    return 0


def _machine_total() -> int:
    """This machine's total memory in bytes (0 when it cannot be read)."""
    try:
        from shutter.core.memory import default_probe
        return int(default_probe().sample().total)
    except Exception:
        return 0


#: the 0.2.3 - 0.2.14 default, written into config.json by those versions; read as the 0.3.0 default
LEGACY_SWAP_GROWTH_MB = 1024.0
SWAP_GROWTH_DEFAULT_MB = 256.0

# Fields whose default is derived from `home` (stored as "" = derive).
_DERIVED = {
    "db_path": "jobs.db",
    "cache_dir": "cache",
    "work_dir": "work",
    "output_dir": "outputs",
    "gpu_lock_path": "gpu.lock",
}


@dataclass
class Config:
    home: str = ""
    host: str = "127.0.0.1"
    port: int = 8484
    db_path: str = ""
    cache_dir: str = ""
    cache_max_gb: float = 20.0
    work_dir: str = ""
    output_dir: str = ""
    keep_work_dirs: bool = False
    gpu_lock_path: str = ""
    # mkdir-style locks shared with other tools (gpulock.sh); taken in this order.
    gpu_lock_dirs: list[str] = field(default_factory=list)
    # A running process matching this means another render is in progress: jobs wait.
    busy_process_pattern: Optional[str] = "h3_generate.py|zig_fasth3|run_mps_avatar|sd-cli"
    # None = no reserve (the default since 0.2.3). A number keeps that many GB free on top of every request.
    headroom_gb: Optional[float] = None
    # {model id: {"licence": name, "at": ISO time}}: licences the person accepted before a download (Fish Drama 3)
    accepted_licences: dict = field(default_factory=dict)
    # The watchdog frees caches when swap grows by this much during a request, stops the request if swap keeps
    # growing (another half of this) within 10 s, and stops it at once when swap has grown 3x this (0.3.0: 256 MB,
    # so a request that pushes the Mac into swap is stopped at 768 MB; before it was 1 GB / 2 GB and a run that grew
    # swap by 1.1 GB was let through).
    swap_growth_mb: float = 256.0
    max_fraction: Optional[float] = None
    memory_poll_s: float = 15.0
    idle_unload_s: float = 900.0
    omlx_url: Optional[str] = "http://127.0.0.1:8000"
    omlx_api_key: Optional[str] = None
    protected_process_pattern: Optional[str] = "omlx|splash|strata"
    protected_process_min_gb: float = 20.0
    api_key: Optional[str] = None
    max_upload_mb: float = 2048.0     # POST /v1/files, /v1/images/edits: largest accepted file
    images_timeout_s: float = 3600.0  # /v1/images/*: how long a request waits for its jobs
    dummy_model: bool = True          # offer the built-in "dummy" model for trying the app
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Engine-wide mode defaults (docs/MODES.md), changed through POST /v1/settings.
    default_mode: str = "fast"
    default_efficiency: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)   # unknown keys, kept on save
    path: str = ""                                        # where this config lives

    def __post_init__(self) -> None:
        if not self.home:
            self.home = str(default_home())
        self.home = os.path.expanduser(self.home)
        if not self.path:
            self.path = os.path.join(self.home, "config.json")
        for name, leaf in _DERIVED.items():
            v = getattr(self, name)
            setattr(self, name, self._resolve(v) if v else os.path.join(self.home, leaf))

    def _resolve(self, p: str) -> str:
        p = os.path.expanduser(str(p))
        return p if os.path.isabs(p) else os.path.join(self.home, p)

    # -- derived numbers ------------------------------------------------------------

    @property
    def headroom_auto(self) -> bool:
        return self.headroom_gb is None

    @property
    def headroom_bytes(self) -> int:
        """The headroom in force: the explicit `headroom_gb`, or automatic from this
        machine's total memory (see auto_headroom). Code with its own memory probe (the
        engine) should call `headroom_for(total)` instead."""
        if self.headroom_gb is not None:
            return int(max(0.0, float(self.headroom_gb)) * 1024**3)
        return self.headroom_for(_machine_total())

    def headroom_for(self, total_bytes: int) -> int:
        if self.headroom_gb is not None:
            return int(max(0.0, float(self.headroom_gb)) * 1024**3)
        return auto_headroom(total_bytes)

    @property
    def swap_growth_bytes(self) -> int:
        mb = float(self.swap_growth_mb)
        if mb == LEGACY_SWAP_GROWTH_MB:  # the old default, saved by 0.2.3 - 0.2.14
            mb = SWAP_GROWTH_DEFAULT_MB
        return int(mb * 1024**2)

    @property
    def cache_max_bytes(self) -> int:
        return int(self.cache_max_gb * 1024**3)

    @property
    def protected_process_min_bytes(self) -> int:
        return int(self.protected_process_min_gb * 1024**3)

    def model_dir(self, model_id: str) -> str:
        spec = self.models.get(model_id) or {}
        d = spec.get("dir") or os.path.join("models", model_id)
        return self._resolve(d)

    # -- load / save ----------------------------------------------------------------

    @classmethod
    def _field_names(cls) -> set[str]:
        return {f.name for f in dataclasses.fields(cls)} - {"extra", "path"}

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, home: Optional[str] = None, path: str = "") -> "Config":
        d = dict(d or {})
        names = cls._field_names()
        known = {k: v for k, v in d.items() if k in names}
        extra = {k: v for k, v in d.items() if k not in names}
        if home is not None and not known.get("home"):
            known["home"] = home
        if known.get("models") is None:
            known.pop("models", None)
        if known.get("gpu_lock_dirs") is None:
            known.pop("gpu_lock_dirs", None)
        return cls(**known, extra=extra, path=path)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """Read config.json from SHUTTER_HOME (default ~/.shutter); defaults when missing."""
        home = str(default_home())
        path = os.path.expanduser(path) if path else os.path.join(home, "config.json")
        data: dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if text.strip():
                data = json.loads(text)
                if not isinstance(data, dict):
                    raise ValueError(f"{path}: expected a JSON object")
        return cls.from_dict(data, home=home, path=path)

    def to_dict(self, *, full: bool = False) -> dict[str, Any]:
        """JSON form. Paths still at their home-derived default are left out unless
        `full`, so moving SHUTTER_HOME moves them too."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name in ("extra", "path"):
                continue
            v = getattr(self, f.name)
            if not full and f.name in _DERIVED and v == os.path.join(self.home, _DERIVED[f.name]):
                continue
            if not full and f.name == "home" and Path(v) == default_home():
                continue
            out[f.name] = v
        for k, v in self.extra.items():
            out.setdefault(k, v)
        return out

    def save(self, path: Optional[str] = None) -> str:
        path = os.path.expanduser(path or self.path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
        self.path = path
        return path

    def ensure_dirs(self) -> None:
        for d in (self.home, self.cache_dir, self.work_dir, self.output_dir,
                  os.path.dirname(self.db_path), os.path.dirname(self.gpu_lock_path)):
            if d:
                os.makedirs(d, exist_ok=True)
