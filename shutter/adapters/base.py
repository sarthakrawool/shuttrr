"""The adapter contract. One adapter per model family (LongCat avatar, FastH3, MiniMax H3...).

The engine core knows nothing about any model. It asks an adapter three things:
  1. estimate(request)  -> how much memory will this need?   (memory guard admission)
  2. plan(request)      -> which shots make up this job?      (shot-parallel, exo later)
  3. run(job, ctx)      -> do the work, reporting progress.

Adapters own their weights. To stay warm between jobs they keep components inside
`ctx.models` (a ModelPool, see core/pool.py), which the engine may evict when another
job or the user's own apps need the memory.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from shutter.types import Job, JobRequest, MemoryEstimate, Shot


class ModelPoolLike(Protocol):
    """Subset of core.pool.ModelPool that adapters use."""

    def get(self, key: str, loader: Callable[[], Any], size_bytes: int, *, keep: bool = True) -> Any:
        """Return the cached component `key`, or call loader() after the guard admits
        `size_bytes`. keep=False loads without caching (stage-scoped use). core.pool.ModelPool also
        takes `residency="streamed"` for a component whose weights are streamed from SSD (then
        `size_bytes` is what stays resident); GET /v1/memory reports it."""

    def release(self, key: str) -> None:
        """Drop one component now and return its memory (staged loading: free after use)."""

    def release_all(self, prefix: str = "") -> None: ...


class CacheLike(Protocol):
    """Subset of core.cache.ContentCache: content-addressed arrays on disk."""

    def get_or_compute(self, namespace: str, key_parts: list[Any], compute: Callable[[], dict[str, Any]]) -> dict[str, Any]: ...


@dataclass
class RunContext:
    """Everything an adapter may use while running one job."""

    models: ModelPoolLike
    cache: CacheLike
    work_dir: Path                                  # scratch for this job
    report: Callable[[float, str], None]            # report(progress 0..1, stage text)
    check_cancelled: Callable[[], None]             # raises types.Cancelled
    memory_sample: Callable[[], dict[str, Any]]     # current memory numbers, for logging
    settings: dict[str, Any] = field(default_factory=dict)  # engine-level settings


class Adapter(abc.ABC):
    #: registry id prefix this adapter serves, e.g. "longcat-avatar"
    family: str = ""
    #: "video" | "image"
    kind: str = "video"

    def __init__(self, model_id: str, model_dir: Path, options: Optional[dict[str, Any]] = None):
        self.model_id = model_id
        self.model_dir = Path(model_dir)
        self.options = options or {}

    @abc.abstractmethod
    def estimate(self, request: JobRequest) -> MemoryEstimate: ...

    #: core request fields copied into `params` (under these names) by `prepare` when params do not
    #: set them, for adapters that read their options from params. New adapters may read the fields.
    PARAM_MIRROR = {"negative_prompt": "negative_prompt", "frames": "frames", "fps": "fps", "steps": "steps",
                    "guidance": "guidance"}

    def prepare(self, request: JobRequest) -> JobRequest:
        """The request as this adapter sees it, called by the engine/worker before `estimate` and `run`
        (the stored request is not changed). Default: a copy with the core fields mirrored into params
        (`PARAM_MIRROR`; width + height as `params.size = "HxW"`), explicit params winning."""
        req = request.copy()
        p = req.params
        for field_name, key in self.PARAM_MIRROR.items():
            v = getattr(req, field_name)
            if v not in (None, "") and key not in p:
                p[key] = v
        if req.width and req.height and "size" not in p:
            p["size"] = f"{int(req.height)}x{int(req.width)}"
        return req

    def plan(self, request: JobRequest) -> list[Shot]:
        """Default: the whole job is one shot."""
        return [Shot(index=0, start_s=0.0, duration_s=request.duration_s or 0.0, seed=request.seed,
                     prompt=request.prompt)]

    @abc.abstractmethod
    def run(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        """Run the job to completion and return the result dict (must include 'output')."""

    def describe(self) -> dict[str, Any]:
        return {"id": self.model_id, "family": self.family, "kind": self.kind, "dir": str(self.model_dir)}
