"""Shared contracts for Shutter. Every module imports its data types from here.

Keep this file dependency-free (stdlib only) so it imports on any machine, including
Linux CI without MLX.
"""

from __future__ import annotations

import copy
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

GB = 1024**3


class JobState(str, enum.Enum):
    QUEUED = "queued"          # accepted, waiting for its turn
    WAITING_MEMORY = "waiting_memory"  # its turn, but the memory guard refuses for now
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = {JobState.DONE, JobState.FAILED, JobState.CANCELLED}


@dataclass
class MemoryEstimate:
    """What a job needs, in bytes. `peak` is the whole-process peak the adapter expects
    (MLX active + cache + Python). `disk_scratch` is temporary disk for outputs/caches."""

    peak: int
    steady: int = 0
    disk_scratch: int = 0
    note: str = ""


@dataclass
class Shot:
    """The unit of parallel work. A job is split into shots that can each run on a
    different machine (exo-style, shot-parallel) and be joined afterwards.

    A shot must be reproducible from its own fields alone: same shot + same model
    build = same frames, regardless of which node runs it.
    """

    index: int
    start_s: float
    duration_s: float
    seed: int
    prompt: str
    # Shots in one `chain` continue from each other (e.g. LongCat 93-frame segments that
    # continue from the previous 13 frames) and must run in order on one node.
    chain: int = 0
    params: dict[str, Any] = field(default_factory=dict)


#: The plain inference request fields, in the order they are serialised.
REQUEST_FIELDS = ("model", "prompt", "negative_prompt", "width", "height", "frames", "duration_s", "fps",
                  "steps", "guidance", "seed", "inputs", "output", "priority", "params")
#: Older top-level spellings of `inputs` entries, still accepted on submit.
LEGACY_INPUT_KEYS = ("image", "audio", "video", "mask")
#: Older top-level keys that are folded into `params` (never emitted again).
LEGACY_PARAM_KEYS = ("framing",)


class JobRequest:
    """What a client submits: a plain inference request.

    `model` is a registry id. `inputs` maps a role ("image", "audio", "video", "mask", ...) to a file
    path on this machine (an upload's path from POST /v1/files works too). Everything model specific
    goes in `params`; adapters document the params they read. `request.image` / `.audio` / `.video`
    are shorthands for `inputs[...]`.

    Backward compatibility: top-level `image` / `audio` / `video` / `mask` keys (and constructor
    arguments) go into `inputs`; the old `framing` key goes into `params["framing"]` (read by the
    optional reels add-on, `shutter.addons.reels`). Neither is ever emitted by `to_dict`.
    """

    __slots__ = REQUEST_FIELDS

    def __init__(self, model: str, prompt: str = "", negative_prompt: str = "",
                 width: Optional[int] = None, height: Optional[int] = None, frames: Optional[int] = None,
                 duration_s: Optional[float] = None, fps: Optional[float] = None, steps: Optional[int] = None,
                 guidance: Optional[float] = None, seed: int = 42, inputs: Optional[dict[str, str]] = None,
                 output: Optional[str] = None, priority: int = 0, params: Optional[dict[str, Any]] = None,
                 **legacy: Any):
        self.model = model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.width = width
        self.height = height
        self.frames = frames
        self.duration_s = duration_s
        self.fps = fps
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.inputs = {k: v for k, v in dict(inputs or {}).items() if v is not None}
        self.output = output
        self.priority = priority
        self.params = dict(params or {})
        for k, v in legacy.items():
            if k in LEGACY_INPUT_KEYS:
                if v is not None:
                    self.inputs.setdefault(k, v)
            elif k in LEGACY_PARAM_KEYS:
                if v is not None:
                    self.params.setdefault(k, getattr(v, "value", v))
            else:
                raise TypeError(f"JobRequest got an unexpected keyword argument {k!r}")

    # -- input shorthands (adapters written against the old request read these)
    def _input(self, role: str) -> Optional[str]:
        return self.inputs.get(role)

    def _set_input(self, role: str, value: Optional[str]) -> None:
        if value is None:
            self.inputs.pop(role, None)
        else:
            self.inputs[role] = value

    image = property(lambda self: self._input("image"), lambda self, v: self._set_input("image", v))
    audio = property(lambda self: self._input("audio"), lambda self, v: self._set_input("audio", v))
    video = property(lambda self: self._input("video"), lambda self, v: self._set_input("video", v))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobRequest":
        d = dict(d)
        d.pop("client", None)  # recorded on the Job, not the request
        inputs = dict(d.pop("inputs", None) or {})
        params = dict(d.pop("params", None) or {})
        for k in LEGACY_INPUT_KEYS:
            v = d.pop(k, None)
            if v is not None:
                inputs.setdefault(k, v)
        for k in LEGACY_PARAM_KEYS:
            v = d.pop(k, None)
            if v is not None:
                params.setdefault(k, getattr(v, "value", v))
        extra = {k: d.pop(k) for k in list(d) if k not in REQUEST_FIELDS}
        if extra:  # unknown keys are adapter params (explicit params win)
            params = {**extra, **params}
        return cls(inputs=inputs, params=params, **d)

    def to_dict(self) -> dict[str, Any]:
        return {k: (dict(getattr(self, k)) if k in ("inputs", "params") else getattr(self, k))
                for k in REQUEST_FIELDS}

    def copy(self) -> "JobRequest":
        return JobRequest.from_dict(copy.deepcopy(self.to_dict()))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, JobRequest) and self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={getattr(self, k)!r}" for k in REQUEST_FIELDS
                         if getattr(self, k) not in (None, "", {}) or k == "model")
        return f"JobRequest({body})"


@dataclass
class Job:
    request: JobRequest
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: JobState = JobState.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = 0.0             # 0..1
    stage: str = ""                   # human-readable current stage, e.g. "denoise 3/8"
    message: str = ""                 # last status line or the error text
    result: dict[str, Any] = field(default_factory=dict)  # output path, timings, peak memory
    node: str = "local"               # which machine ran it (for multi-Mac later)
    client: Optional[str] = None      # free text the submitter sent, e.g. "comfyui/0.46"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "result": self.result,
            "node": self.node,
            "client": self.client,
            "request": self.request.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Job":
        return cls(
            request=JobRequest.from_dict(d["request"]),
            id=d["id"],
            state=JobState(d["state"]),
            created_at=d["created_at"],
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            progress=d.get("progress", 0.0),
            stage=d.get("stage", ""),
            message=d.get("message", ""),
            result=d.get("result", {}),
            node=d.get("node", "local"),
            client=d.get("client"),
        )


@dataclass
class Event:
    """Published on the event bus and streamed to clients (SSE)."""

    kind: str                 # job.queued | job.started | job.progress | job.done | job.failed |
                              # job.cancelled | job.waiting_memory | memory.sample | model.loaded |
                              # model.unloaded
    job_id: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "job_id": self.job_id, "data": self.data, "ts": self.ts}


class Cancelled(Exception):
    """Raised inside an adapter when the job was cancelled."""


class MemoryRefused(Exception):
    """Raised when the memory guard will not admit a load or a job."""

    def __init__(self, reason: str, retry: bool = True):
        super().__init__(reason)
        self.reason = reason
        self.retry = retry
