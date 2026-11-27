"""The engine: one process, one GPU worker thread.

The worker takes the next queued job (highest priority, then oldest), asks its adapter for
a memory estimate, and only starts it when the MemoryGuard admits it, no other render is
running, and it holds both the flock `GpuLock` and the shared `mkdir` lock dirs. Otherwise
the job waits in `waiting_memory` with a human reason and is re-checked every
`memory_poll_s`; a newer higher-priority job may overtake it.

While a job runs, MLX is capped at what was free when the job was admitted, and a watchdog
thread reads memory every second: at the first sign of trouble (critical pressure, or swap
growing by 1 GB) it frees caches and idle models; if that does not help it stops the job
"to protect the Mac" at the next step and releases every model. If the job does not reach a
step within `EMERGENCY_S` (10 s) while the Mac is still in trouble, the engine records the job as
stopped and exits with code 75 (the app restarts it): the OS frees the memory at once.
Exceptions never kill the worker.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from shutter.adapters.base import RunContext
from shutter.core.cache import ContentCache
from shutter.core.config import Config
from shutter.core.dirlock import DirLocks
from shutter.core.memory import (
    ACTION_RELIEVE, ACTION_STOP, PRESSURE_CRITICAL, PROTECT_PREFIX, STOP_BODY, STOP_MESSAGE, GpuLock, MemoryGuard, OmlxService, ProcessRssService,
    admission_budget, default_probe, parse_ps,
)
from shutter.core import memory as _memory
from shutter.core.pool import ModelPool
from shutter.core.queue import EventBus, JobStore
from shutter.core.registry import AdapterUnavailable, Registry, UnknownModel
from shutter.core import files as _files
from shutter.core.eta import EtaTracker, decode_estimate_from_history
from shutter.core.stats import ClientTracker, StepRate, parse_step, throughput_from_jobs, THROUGHPUT_WINDOW_S
from shutter.types import (
    GB, TERMINAL_STATES, Cancelled, Event, Job, JobRequest, JobState, MemoryEstimate, MemoryRefused,
)

__all__ = ["Engine", "SubmitError", "UnloadRefused", "LoadRefused", "version", "is_load", "model_name"]

EMERGENCY_S = 10.0          # stop asked, no step boundary reached, Mac still in trouble: exit the engine
EMERGENCY_EXIT_CODE = 75    # the app restarts an engine that exits with this code


def _mlx() -> Any:
    try:
        import mlx.core as mx  # noqa: PLC0415
        return mx
    except Exception:
        return None


def _mlx_active() -> int:
    mx = _mlx()
    if mx is None:
        return 0
    for fn in ("get_active_memory", "metal.get_active_memory"):
        try:
            obj = mx
            for part in fn.split("."):
                obj = getattr(obj, part)
            return int(obj())
        except Exception:
            continue
    return 0
def _mlx_held() -> Optional[int]:
    """Memory MLX holds in this process (active + cache), or None without MLX."""
    mx = _mlx()
    if mx is None:
        return None
    total = 0
    for fn in ("get_active_memory", "get_cache_memory"):
        f = getattr(mx, fn, None)
        if not callable(f):
            return None
        try:
            total += int(f())
        except Exception:
            return None
    return total


RENDER_BUSY_MESSAGE = "another render is running"
MAX_MIDRUN_REFUSALS = 3


def version() -> str:
    try:
        from shutter import __version__
        return __version__
    except Exception:  # pragma: no cover
        return "0.0.0"


class SubmitError(ValueError):
    """Bad job request (unknown model, missing input file, ...)."""


class UnloadRefused(RuntimeError):
    """POST /v1/models/<id>/unload while a request for that model is running."""


class LoadRefused(RuntimeError):
    """POST /v1/models/<id>/load when the memory guard says no right now. `info` carries the numbers the app shows:
    kind ("memory" or "busy"), model_name, peak_gb, headroom_gb, available_gb, short_by_gb, rung, lowest."""

    def __init__(self, reason: str, info: dict[str, Any]):
        super().__init__(reason)
        self.info = info


LOAD_ACTION = "load"


def is_load(request: JobRequest) -> bool:
    """A load request: POST /v1/models/<id>/load queues one of these (params._action = "load"). It goes through the
    same queue, GPU lock and memory guard as any request, but runs adapter.warm() instead of adapter.run()."""
    return (getattr(request, "params", None) or {}).get("_action") == LOAD_ACTION


def load_estimate(adapter: Any, request: JobRequest) -> MemoryEstimate:
    """The guard checks a load against the model's request peak in the current mode (loading a model that then
    cannot run a request would only take memory from other apps); `steady` is what the load keeps in memory."""
    est = adapter.estimate(request)
    return MemoryEstimate(peak=int(est.peak), steady=int(est.steady or est.peak), note=f"load: {est.note}".strip())


def model_name(model_id: str, family: str = "") -> str:
    """A name for people: the catalog's name, "Test model" for the built-in one, else the id."""
    if family == "dummy" or model_id == "dummy":
        return "Test model"
    try:
        from shutter.core.catalog import CATALOG
        if model_id in CATALOG:
            return str(CATALOG[model_id]["name"])
    except Exception:
        pass
    return model_id


def _rung_of(adapter: Any, request: JobRequest) -> Optional[str]:
    for name in ("choose_rung", "choose"):
        fn = getattr(adapter, name, None)
        if callable(fn):
            try:
                return str(fn(request)["rung"])
            except Exception:
                return None
    return None


class _Shutdown(Cancelled):
    pass


def _is_oom(e: BaseException) -> bool:
    """An allocation failure from MLX/Metal (it raises when the memory limit is hit and nothing can be freed)."""
    t = f"{type(e).__name__}: {e}".lower()
    return isinstance(e, MemoryError) or any(k in t for k in (
        "out of memory", "insufficient memory", "memory limit", "failed to allocate", "kiogpucommandbuffercallbackerroroutofmemory"))


NO_LOAD_HINT = "Loads when a request needs it."


class LoadUnsupported(SubmitError):
    """POST /v1/models/<id>/load for a model whose adapter cannot load ahead of a request."""


def plain_error(e: BaseException, name: str = "", loading: bool = False) -> str:
    """A sentence for people for an exception from a model (the app never shows Python's text). The exception
    itself goes to the engine log and the job's result (`error_detail`), which Copy Diagnostics collects."""
    if isinstance(e, (FileNotFoundError, NotADirectoryError)):
        return (f"Some of {name or 'this model'}'s files are missing. Download it again from Setup, or check the "
                f"folder in config.json.")
    if isinstance(e, ImportError):
        return "A part of the engine is missing. Restart Shuttrr; if it keeps happening, reinstall it."
    if isinstance(e, PermissionError):
        return f"Shuttrr may not read {name or 'this model'}'s files. Check the folder's permissions."
    if type(e) in (ValueError, SubmitError) and str(e).strip():
        return str(e).strip()           # raised on purpose with a sentence for people (bad input, wrong model)
    if loading:
        return "Something inside the engine failed while loading it. Copy Diagnostics has the details."
    return "The request stopped because of an internal error. Copy Diagnostics has the details."


def _prepare(adapter: Any, request: JobRequest) -> JobRequest:
    """adapter.prepare(request) (core fields mirrored into params, see adapters.base), or the
    request itself for an adapter without prepare."""
    fn = getattr(adapter, "prepare", None)
    return fn(request) if callable(fn) else request


# ---------------------------------------------------------------- modes (docs/MODES.md)

try:  # the model side owns the ladder; fall back to the table in docs/MODES.md
    from shutter.core.modes import LADDER as _LADDER
except Exception:  # pragma: no cover - only while modes.py is missing or broken
    _LADDER = [
        {"at": 0.00, "id": "resident", "label": "Full speed", "detail": "Same as Fast: the whole model stays in memory."},
        {"at": 0.25, "id": "split", "label": "Slightly lower memory", "detail": "Each model block runs in 3 smaller pieces: about 2 GB less memory, about 2% slower."},
        {"at": 0.50, "id": "staged", "label": "Lower memory", "detail": "Also works through the frames in 4 chunks, and frees the main model before the final decode."},
        {"at": 0.75, "id": "stream_wide", "label": "Very low memory", "detail": "Reads the model from the SSD as it goes, keeping 8 blocks in memory at a time."},
        {"at": 1.00, "id": "stream_tight", "label": "Lowest memory", "detail": "Reads the model from the SSD as it goes, keeping only 2 blocks in memory: lowest memory, slowest."},
    ]

MODES = ("fast", "low_memory")
DEFAULT_MODE = "fast"
DEFAULT_EFFICIENCY = 0.5


def normalize_mode(v: Any) -> Optional[str]:
    """"fast" / "low_memory" (also "low-memory", "low memory"); None for anything else."""
    if not isinstance(v, str):
        return None
    k = v.strip().lower().replace("-", "_").replace(" ", "_")
    return {"fast": "fast", "low_memory": "low_memory", "lowmemory": "low_memory", "low": "low_memory"}.get(k)


def clamp_efficiency(v: Any) -> float:
    """0..1, rounded to 4 decimals; a non-number raises ValueError (bool is not a number)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("efficiency must be a number from 0 to 1")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        return DEFAULT_EFFICIENCY if f != f else (1.0 if f > 0 else 0.0)
    return round(min(1.0, max(0.0, f)), 4)


def rung_id(mode: str, efficiency: float) -> str:
    if mode != "low_memory":
        return _LADDER[0]["id"]
    best = _LADDER[0]
    for r in _LADDER:
        if r["at"] <= efficiency + 1e-9:
            best = r
    return best["id"]


def read_measured(home: str) -> Optional[tuple[str, str, dict[str, Any]]]:
    """measured.json's (model, size, rungs) with the most measured rungs (ties: by name).
    Missing file, bad JSON or a wrong shape give None. Same rule as the daemon."""
    import json as _json
    try:
        with open(os.path.join(home, "measured.json"), encoding="utf-8") as f:
            data = _json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    ids = [r["id"] for r in _LADDER]
    best = None
    for model in sorted(data):
        sizes = data[model]
        if not isinstance(sizes, dict):
            continue
        for size in sorted(sizes):
            rungs = sizes[size]
            if not isinstance(rungs, dict):
                continue
            n = sum(1 for i in ids if isinstance(rungs.get(i), dict))
            if n and (best is None or n > best[0]):
                best = (n, model, size, rungs)
    return (best[1], best[2], best[3]) if best else None


def ladder_json(home: str) -> tuple[list[dict[str, Any]], Optional[dict[str, str]]]:
    measured = read_measured(home)
    rows = []
    for r in _LADDER:
        row = {"at": r["at"], "id": r["id"], "label": r["label"], "detail": r["detail"],
               "identical": r.get("identical", True)}
        e = measured[2].get(r["id"]) if measured else None
        if isinstance(e, dict):
            for k in ("peak_bytes", "seconds"):
                x = e.get(k)
                if isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0 and x == x:
                    row[k] = int(x) if k == "peak_bytes" else float(x)
            if isinstance(e.get("identical"), bool):
                row["identical"] = e["identical"]
        rows.append(row)
    return rows, ({"model": measured[0], "size": measured[1]} if measured else None)


def persist_settings(path: str, mode: str, efficiency: float) -> None:
    """Write default_mode / default_efficiency into config.json atomically, keeping every
    other key as it is in the file (not the whole Config, so the file stays small)."""
    import json as _json
    data: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            data = _json.loads(text)  # ValueError: left alone, reported
            if not isinstance(data, dict):
                raise ValueError(f"{path} is not a JSON object; not changing it")
    data["default_mode"] = mode
    data["default_efficiency"] = efficiency
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


class Engine:
    def __init__(
        self,
        config: Optional[Config] = None,
        probe: Any = None,
        protected: Optional[Iterable[Any]] = None,
        *,
        registry: Optional[Registry] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Optional[Callable[[float], None]] = None,
        ps: Optional[Callable[[], str]] = None,
        sample_interval_s: float = 5.0,
        watch_interval_s: float = 3.0,
        watchdog_interval_s: float = 1.0,
        emergency_s: float = EMERGENCY_S,
        idle_check_s: float = 30.0,
        progress_persist_s: float = 1.0,
    ):
        self.config = config or Config.load()
        self.config.ensure_dirs()
        self.probe = probe or default_probe()
        if protected is None:
            protected = []
            if self.config.omlx_url:
                protected.append(OmlxService(self.config.omlx_url, api_key=self.config.omlx_api_key))
            if self.config.protected_process_pattern:
                protected.append(ProcessRssService(self.config.protected_process_pattern,
                                                   self.config.protected_process_min_bytes))
        self.clock = clock
        self._sleep = sleep
        self._ps = ps or _memory._ps
        self.sample_interval_s = sample_interval_s
        self.watch_interval_s = watch_interval_s
        self.watchdog_interval_s = watchdog_interval_s
        self.emergency_s = emergency_s
        self.idle_check_s = idle_check_s
        self.progress_persist_s = progress_persist_s

        self.store = JobStore(self.config.db_path)
        self.bus = EventBus()
        self.headroom_auto = self.config.headroom_auto
        try:
            total = int(self.probe.sample().total)
        except Exception:
            total = 0
        headroom = self.config.headroom_for(total)
        self.guard = MemoryGuard(self.probe, protected=list(protected),
                                 headroom_bytes=headroom,
                                 swap_growth_bytes=self.config.swap_growth_bytes,
                                 max_fraction=self.config.max_fraction)
        self.pool = ModelPool(self.guard, on_event=self.bus.publish, clock=clock)
        self.cache = ContentCache(self.config.cache_dir, self.config.cache_max_bytes)
        self.registry = registry or Registry(self.config)
        self.gpu_lock = GpuLock(self.config.gpu_lock_path, label=f"shutter (pid {os.getpid()})")
        self.dir_locks = DirLocks(self.config.gpu_lock_dirs or [], owner="shutter")
        self._busy_re = (re.compile(self.config.busy_process_pattern)
                         if self.config.busy_process_pattern else None)

        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._running: Optional[Job] = None
        self._step_info: dict[str, dict[str, Any]] = {}   # job id -> {"rung", "residency"} (adapter's on_step)
        self._cancel = threading.Event()
        self._abort_reason: Optional[str] = None
        self._wait_reasons: dict[str, str] = {}
        self._load_errors: dict[str, str] = {}
        self._load_error_details: dict[str, str] = {}
        self._midrun_refusals: dict[str, int] = {}
        self._last_idle_check = clock()
        self._settings_lock = threading.Lock()
        mode = normalize_mode(self.config.default_mode) or DEFAULT_MODE
        try:
            eff = clamp_efficiency(self.config.default_efficiency)
        except ValueError:
            eff = DEFAULT_EFFICIENCY
        self._settings = {"mode": mode, "efficiency": eff}
        self.worker_errors: list[str] = []
        self.clients = ClientTracker()
        self._steprate = StepRate(clock=clock)

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._worker, name="shutter-gpu-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stopping.set()
        if self._running is not None:
            self._cancel.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    def close(self) -> None:
        self.stop()
        self.pool.release_all()
        self.store.close()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------ public API

    def submit(self, request: JobRequest, client: Optional[str] = None) -> Job:
        if isinstance(request, dict):
            client = client or request.get("client")
            request = JobRequest.from_dict(request)
        if not request.model:
            raise SubmitError("model is required")
        if request.model not in self.registry:
            raise SubmitError(str(UnknownModel(request.model, self.registry.model_ids())))
        if not isinstance(request.inputs, dict):
            raise SubmitError("inputs must be an object: {\"image\": path, \"audio\": path, ...}")
        resolved = {}
        for name, p in request.inputs.items():
            if p is None or p == "":
                continue
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
                raise SubmitError(f"input names are short words such as image, audio, video (got {name!r})")
            if not isinstance(p, str):
                raise SubmitError(f"inputs.{name} must be a file path or an upload id")
            p = _files.resolve_input(self.config.work_dir, p)
            if not os.path.isfile(p):
                raise SubmitError(f"{name} file not found: {p}")
            resolved[name] = p
        request.inputs = resolved
        for k in ("width", "height", "frames", "steps"):
            v = getattr(request, k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v <= 0):
                raise SubmitError(f"{k} must be a positive integer")
        for k in ("fps", "guidance"):
            v = getattr(request, k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0):
                raise SubmitError(f"{k} must be a number >= 0")
        if request.output:
            request.output = os.path.abspath(os.path.expanduser(request.output))
        if request.duration_s is not None and request.duration_s < 0:
            raise SubmitError("duration_s must be >= 0")
        try:
            request.priority = int(request.priority or 0)
            request.seed = int(request.seed)
        except (TypeError, ValueError) as e:
            raise SubmitError(f"priority and seed must be integers ({e})")
        if not isinstance(request.params, dict):
            raise SubmitError("params must be an object")
        cur = dict(self._settings)
        params = dict(request.params)
        if params.get("mode") is None:
            params["mode"] = cur["mode"]
        elif normalize_mode(params["mode"]) is None:
            raise SubmitError("mode must be one of: fast, low_memory")
        else:
            params["mode"] = normalize_mode(params["mode"])
        if params.get("efficiency") is None:
            params["efficiency"] = cur["efficiency"]
        else:
            try:
                params["efficiency"] = clamp_efficiency(params["efficiency"])
            except ValueError as e:
                raise SubmitError(str(e))
        request.params = params
        job = Job(request=request, client=client or None)
        self.store.add(job)
        self.clients.submitted(job.client)
        extra = {"client": job.client} if job.client else {}
        self._publish("job.queued", job, priority=request.priority, model=request.model, **extra)
        self._wake.set()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            if self._running is not None and self._running.id == job_id:
                return self._live_copy(self._running)
        return self.store.get(job_id)

    def list(self, states: Optional[Iterable[JobState]] = None, limit: int = 200) -> list[Job]:
        jobs = self.store.list(states=states, limit=limit)
        with self._lock:
            if self._running is not None:
                jobs = [self._live_copy(self._running) if j.id == self._running.id else j for j in jobs]
        for j in jobs:
            if j.state in (JobState.QUEUED, JobState.WAITING_MEMORY):
                j.expected_s = self._expected_seconds(j)
        return jobs

    def _live_copy(self, job: Job) -> Job:
        """The running job with its time left worked out now (it counts down between steps)."""
        c = self._copy(job)
        tr = getattr(self, "_eta", None)
        if tr is not None and not c.cancel_requested:
            est = tr.estimate(self.clock())
            c.eta_s, c.eta_decode_known = est["eta_s"], est["decode_known"]
        return c

    def _expected_seconds(self, job: Job) -> Optional[float]:
        """How long a queued request should take once it starts, from the catalog's measured runs (the 0.2.10
        estimate for the request's own size, frames, steps and decode). None when nothing is measured for it."""
        r = job.request
        key = (r.model, r.width, r.height, r.frames, r.steps, bool((r.params or {}).get("vae_tile")))
        cache = self.__dict__.setdefault("_expected_cache", {})
        if key in cache:
            return cache[key]
        val: Optional[float] = None
        try:
            from shutter.core import request_options as ro
            body = {k: v for k, v in (("width", r.width), ("height", r.height), ("frames", r.frames),
                                      ("steps", r.steps)) if v}
            if "vae_tile" in (r.params or {}):
                body["vae_tile"] = bool(r.params["vae_tile"])
            val = ro.estimate(self.config, r.model, body).get("seconds")
            val = round(float(val), 1) if val else None
        except Exception:
            val = None
        cache[key] = val
        return val

    def _decode_estimate(self, job: Job) -> Optional[float]:
        """Seconds the decode should take: earlier finished runs of the same request on this Mac first, then the
        catalog's measured run for exactly this request. None when neither exists."""
        try:
            hist = decode_estimate_from_history(self.store.list([JobState.DONE], limit=200), job.request.model,
                                                job.request)
            if hist:
                return hist
            from shutter.core import catalog as cat
            e = cat.CATALOG.get(job.request.model)
            if e and job.request.width and job.request.height and job.request.frames:
                c = cat._measured_entry(e, job.request.height, job.request.width, job.request.frames,
                                        bool((job.request.params or {}).get("vae_tile")))
                d = c.get("decode_s")
                return float(d) if d else None
        except Exception:
            return None
        return None

    def cancel(self, job_id: str) -> Optional[Job]:
        """Cancel a job. Queued/waiting: cancelled at once. Running: the adapter stops at
        its next check_cancelled(). Returns the job (None when unknown)."""
        with self._lock:
            if self._running is not None and self._running.id == job_id:
                self._cancel.set()
                self._running.message = "cancelling"
                self._running.cancel_requested = True
                self._running.eta_s = None
                self._publish("job.cancelling", self._running)
                return self._copy(self._running)
            job = self.store.get(job_id)
            if job is None:
                return None
            if job.state in (JobState.QUEUED, JobState.WAITING_MEMORY):
                job.state = JobState.CANCELLED
                job.finished_at = time.time()
                job.message = "cancelled before it started"
                self.store.update(job)
                self._wait_reasons.pop(job.id, None)
                self._publish("job.cancelled", job)
        self._wake.set()
        return job

    def pause(self) -> None:
        self._paused = True
        self.bus.publish(Event("queue.paused"))

    def resume(self) -> None:
        self._paused = False
        self.bus.publish(Event("queue.resumed"))
        self._wake.set()

    @property
    def paused(self) -> bool:
        return self._paused

    def release(self, key: Optional[str] = None) -> list[str]:
        """Unload one pool entry (or all when key is None). Entries in use by the running
        job are dropped once it lets go. Returns the keys that were unloaded now."""
        before = set(self.pool.keys())
        if key is None:
            self.pool.release_all()
        else:
            self.pool.release(key)
        return sorted(before - set(self.pool.keys()))

    # ------------------------------------------------------------------ load / unload

    def _model_keys(self, model_id: str) -> list[str]:
        return [e["key"] for e in self.pool.stats()["entries"] if self._model_of_key(e["key"]) == model_id]

    def _pending_load(self, model_id: str) -> Optional[Job]:
        for j in self.store.list([JobState.QUEUED, JobState.WAITING_MEMORY, JobState.RUNNING], limit=500):
            if j.request.model == model_id and is_load(j.request):
                return j
        with self._lock:
            r = self._running
            if r is not None and r.request.model == model_id and is_load(r.request):
                return self._copy(r)
        return None

    def load_model(self, model_id: str, client: Optional[str] = None, wait: bool = False) -> Job:
        """POST /v1/models/<id>/load: queue a load (or return the one already pending). Without `wait` the memory
        guard is asked first and a "no" raises LoadRefused with the numbers; with `wait` the load queues and waits
        for memory like any request."""
        if model_id not in self.registry:
            raise SubmitError(str(UnknownModel(model_id, self.registry.model_ids())))
        pending = self._pending_load(model_id)
        if pending is not None:
            return pending
        ok, hint = self._can_load(model_id)
        if not ok:
            name = model_name(model_id, self.registry.spec(model_id).get("adapter", ""))
            raise LoadUnsupported(f"{name} loads when a request needs it; it cannot be loaded ahead of time."
                                  if hint == NO_LOAD_HINT else hint)
        req = JobRequest(model=model_id, params={"_action": LOAD_ACTION, **self._settings})
        if not wait:
            self.check_load(req)
        self._load_errors.pop(model_id, None)
        return self.submit(req, client=client or "load")

    def check_load(self, req: JobRequest) -> None:
        """Ask the guard now, without queueing anything; raise LoadRefused with structured numbers on a no."""
        try:
            adapter = self.registry.adapter_for(req.model)
            family = self.registry.spec(req.model).get("adapter", "")
        except (UnknownModel, AdapterUnavailable) as e:
            raise SubmitError(str(e))
        preq = _prepare(adapter, req)
        est = load_estimate(adapter, preq)
        keys, prefix = self._needed_keys(adapter, preq)
        warm = sum(e["size_bytes"] for e in self.pool.stats()["entries"] if self._is_needed(e["key"], keys, prefix))
        need = max(0, int(est.peak) - warm)
        name = model_name(req.model, family)
        d = self.guard.check(need, name)
        if d.ok:
            return
        rung = _rung_of(adapter, preq)
        avail = int(d.sample.available)
        head = int(self.guard.headroom_bytes)
        memory = "Not enough free memory" in d.reason or "more than the" in d.reason
        info = {"kind": "memory" if memory else "busy", "model": req.model, "model_name": name,
                "peak_gb": round(int(est.peak) / GB, 1), "need_gb": round(need / GB, 1),
                "headroom_gb": round(head / GB, 1), "available_gb": round(avail / GB, 1),
                "free_gb": round(admission_budget(d.sample, head) / GB, 1),
                "short_by_gb": round(max(0, need + head - avail) / GB, 1),
                "rung": rung, "lowest": rung == "stream_tight", "reason": d.reason}
        raise LoadRefused(d.reason, info)

    def unload_model(self, model_id: str) -> list[str]:
        """POST /v1/models/<id>/unload: cancel a pending load, drop the model's pool entries and their load mark.
        Raises UnloadRefused while a request for the model is running."""
        if model_id not in self.registry:
            raise SubmitError(str(UnknownModel(model_id, self.registry.model_ids())))
        with self._lock:
            r = self._running
            if r is not None and r.request.model == model_id:
                what = "loading it" if is_load(r.request) else "running a request for it"
                raise UnloadRefused(f"Shuttrr is {what} right now. Unload it when that is done.")
        for j in self.store.list([JobState.QUEUED, JobState.WAITING_MEMORY], limit=500):
            if j.request.model == model_id and is_load(j.request):
                self.cancel(j.id)
        self._load_errors.pop(model_id, None)
        keys = self._model_keys(model_id)
        before = set(self.pool.keys())
        for k in keys:
            try:
                self.pool.hold(k, False)
            except KeyError:
                pass
            self.pool.release(k)
        return sorted(before - set(self.pool.keys()))

    def _can_load(self, model_id: str) -> tuple[bool, str]:
        """Whether Load works for this model, and a one-line hint for people when it does not."""
        try:
            adapter = self.registry.adapter_for(model_id)
        except (UnknownModel, AdapterUnavailable):
            return False, "This model's engine part could not start. Copy Diagnostics has the details."
        except Exception:
            return False, "This model could not be read. Copy Diagnostics has the details."
        if not callable(getattr(adapter, "warm", None)):
            return False, NO_LOAD_HINT
        fn = getattr(adapter, "can_load", None)
        if callable(fn):
            try:
                ok, hint = fn()
                return bool(ok), str(hint or "")
            except Exception:
                return False, "This model could not be read. Copy Diagnostics has the details."
        return True, ""

    def models(self) -> list[dict[str, Any]]:
        """GET /v1/models: the registry rows plus state (not_loaded, loading, waiting_for_memory, loaded), loaded_gb
        and, while a load waits, the guard's sentence (`waiting`)."""
        rows = self.registry.list()
        entries = self.pool.stats()["entries"]
        with self._lock:
            run = self._copy(self._running) if self._running is not None else None
            run_step = dict(self._step_info.get(run.id) or {}) if run is not None else {}
        for row in rows:
            mid = row["id"]
            mine = [e for e in entries if self._model_of_key(e["key"]) == mid]
            pending = self._pending_load(mid)
            row["loaded_bytes"] = int(sum(e["size_bytes"] for e in mine))
            row["loaded_gb"] = round(row["loaded_bytes"] / GB, 2) if mine else None
            row["held"] = any(e.get("held") for e in mine)
            row["waiting"] = None
            if pending is not None and pending.state == JobState.RUNNING:
                row["state"] = "loading"
            elif pending is not None and pending.state == JobState.WAITING_MEMORY:
                row["state"] = "waiting_for_memory"
                row["waiting"] = pending.message
            elif pending is not None:
                row["state"] = "loading"
            elif run is not None and run.request.model == mid and not is_load(run.request):
                row["state"] = "in_use"
            elif mine:
                row["state"] = "loaded"
            else:
                row["state"] = "not_loaded"
            # in_use: a request runs on this model now, with the memory step it uses (residency streamed = most of
            # the model is read from the SSD during the request); loaded_* still describe what stays afterwards
            row["in_use"] = ({"request": run.id, "rung": run_step.get("rung"),
                              "residency": run_step.get("residency")} if row["state"] == "in_use" else None)
            row["load_request"] = pending.id if pending is not None else None
            row["name"] = model_name(mid, row.get("family", ""))
            row["load_error"] = self._load_errors.get(mid) if row["state"] == "not_loaded" else None
            row["load_error_detail"] = (self._load_error_details.get(mid) or None) if row["load_error"] else None
            res = {e.get("residency") or "resident" for e in mine}
            row["loaded_residency"] = ("streamed" if res == {"streamed"} else "mixed" if len(res) > 1
                                       else "resident") if mine else None
            if row.get("installed"):
                row["can_load"], row["load_hint"] = self._can_load(mid)
            else:
                row["can_load"], row["load_hint"] = False, ""
        return rows

    # ------------------------------------------------------------------ settings

    def settings_brief(self) -> dict[str, Any]:
        cur = dict(self._settings)
        return {"mode": cur["mode"], "efficiency": cur["efficiency"],
                "rung": rung_id(cur["mode"], cur["efficiency"])}

    def settings(self) -> dict[str, Any]:
        """GET /v1/settings: defaults, the rung they pick, the ladder (+ measured numbers)."""
        out = self.settings_brief()
        ladder, used = ladder_json(self.config.home)
        if used is None:
            ladder, used = self._ladder_from_models(ladder)
        out["ladder"] = ladder
        out["measured_for"] = used
        return out

    def _ladder_from_models(self, ladder: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Optional[dict[str, str]]]:
        """Without measured.json: the measured peaks of an installed model's default request on this Mac (its
        build's / the catalog's checked runs), so the memory steps show real numbers instead of none."""
        for row in self.registry.list():
            if not row.get("installed"):
                continue
            try:
                fn = getattr(self.registry.adapter_for(row["id"]), "measured_ladder", None)
                m = fn() if callable(fn) else None
            except Exception:
                m = None
            if not m or not m.get("rungs"):
                continue
            out = []
            for r in ladder:
                r = dict(r)
                e = m["rungs"].get(r["id"]) or {}
                for k in ("peak_bytes", "seconds", "identical"):
                    if k in e:
                        r[k] = e[k]
                out.append(r)
            return out, {"model": row["id"], "size": m["size"]}
        return ladder, None

    def update_settings(self, mode: Any = None, efficiency: Any = None) -> dict[str, Any]:
        """POST /v1/settings. Raises SubmitError (400) on a bad value, OSError/ValueError
        when config.json cannot be written."""
        m = None
        if mode is not None:
            m = normalize_mode(mode)
            if m is None:
                raise SubmitError("mode must be one of: fast, low_memory")
        e = None
        if efficiency is not None:
            try:
                e = clamp_efficiency(efficiency)
            except ValueError as err:
                raise SubmitError(str(err))
        with self._settings_lock:
            nxt = {"mode": m or self._settings["mode"],
                   "efficiency": self._settings["efficiency"] if e is None else e}
            persist_settings(self.config.path, nxt["mode"], nxt["efficiency"])
            self._settings = nxt
            self.config.default_mode, self.config.default_efficiency = nxt["mode"], nxt["efficiency"]
        self.bus.publish(Event("settings.changed", data=self.settings_brief()))
        return self.settings()

    def system(self) -> dict[str, Any]:
        snap = self.guard.snapshot()
        s = snap["sample"]
        pool = self.pool.stats()
        counts = self.store.counts()
        holder = None
        held = self.gpu_lock.held
        try:
            holder = self.gpu_lock.holder()
        except Exception:
            holder = None
        dir_holders = []
        try:
            dir_holders = self.dir_locks.holders()
        except Exception:
            pass
        running = self._running
        return {
            "version": version(),
            "settings": self.settings_brief(),
            "memory": {"total": s["total"], "available": s["available"],
                       "swap_used": s["swap_used"], "pressure": s["pressure"]},
            "guard": {
                "headroom_bytes": self.guard.headroom_bytes,
                "headroom_auto": self.headroom_auto,
                "swap_growth_bytes": self.guard.swap_growth_bytes,
                "protected": [{"name": p["name"], "busy": p["busy"]} for p in snap["protected"]],
            },
            "pool": {
                "total_bytes": pool["total_bytes"],
                "entries": [{"key": e["key"], "size_bytes": e["size_bytes"],
                             "idle_s": round(e["idle_s"], 1), "pinned": e["pinned"],
                             "in_use": e.get("in_use", False),
                             "residency": "streamed" if e.get("residency") == "streamed" else "resident"}
                            for e in pool["entries"]],
            },
            "queue": {
                "queued": counts.get("queued", 0),
                "running": counts.get("running", 0),
                "waiting_memory": counts.get("waiting_memory", 0),
                "done": counts.get("done", 0),
                "failed": counts.get("failed", 0),
                "cancelled": counts.get("cancelled", 0),
                "paused": self._paused,
                "current": running.id if running is not None else None,
            },
            "gpu_lock": {
                "held": bool(held or holder),
                "ours": held,
                "holder": holder,
                "path": self.gpu_lock.path,
                "dirs": [{"path": p, "ours": p in self.dir_locks._created}
                         for p in self.dir_locks.paths],
                "dir_holders": dir_holders,
            },
            "throughput": self.throughput(),
        }

    # ------------------------------------------------------------------ native API additions

    def throughput(self) -> dict[str, Any]:
        """Done requests in the last hour, their mean wall seconds, the running job's step rate."""
        now = time.time()
        try:
            jobs = self.store.finished_since(now - THROUGHPUT_WINDOW_S)
        except Exception:
            jobs = []
        running = self._running
        cur = None
        if running is not None:
            cur = {"id": running.id, "model": running.request.model, **self._steprate.snapshot()}
        return throughput_from_jobs(jobs, cur, now=now)

    def _model_of_key(self, key: str) -> Optional[str]:
        ids = self.registry.model_ids()
        parts = key.split("/")
        for mid in sorted(ids, key=len, reverse=True):
            if mid in parts or key.startswith(mid + "/"):
                return mid
        return None

    def memory(self, model: Optional[str] = None, request: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """GET /v1/memory: what the Mac has, what Shutter keeps back, what Shutter holds (each
        component resident or streamed) and, with `model`, whether that model's request fits now."""
        snap = self.guard.snapshot()
        s = snap["sample"]
        reserve = int(self.guard.headroom_bytes)
        free = max(0, int(s["available"]) - reserve)
        cap = None
        if self.guard.max_fraction is not None and s["total"]:
            cap = int(self.guard.max_fraction * s["total"])
            free = min(free, cap)
        pool = self.pool.stats()
        comps = [{"key": e["key"], "model": self._model_of_key(e["key"]), "size_bytes": int(e["size_bytes"]),
                  "residency": e.get("residency", "resident"), "in_use": bool(e.get("in_use", False)),
                  "pinned": bool(e["pinned"]), "idle_s": round(e["idle_s"], 1)} for e in pool["entries"]]
        out: dict[str, Any] = {
            "total": int(s["total"]), "available": int(s["available"]), "reserve": reserve,
            "reserve_auto": bool(self.headroom_auto), "free_for_models": free, "max_job_bytes": cap,
            "swap": int(s["swap_used"]), "pressure": s["pressure"],
            "held": int(pool["total_bytes"]),
            "resident_bytes": sum(c["size_bytes"] for c in comps if c["residency"] == "resident"),
            "streamed_bytes": sum(c["size_bytes"] for c in comps if c["residency"] == "streamed"),
            "components": comps,
            "protected": [{"name": p["name"], "busy": p["busy"]} for p in snap["protected"]],
        }
        if model:
            out["model"] = self._model_fit(model, request or {}, free, comps)
        return out

    def _model_fit(self, model: str, extra: dict[str, Any], free: int, comps: list[dict[str, Any]]) -> dict[str, Any]:
        row: dict[str, Any] = {"id": model, "need_bytes": None, "warm_bytes": 0, "fits": None,
                               "short_by_bytes": None, "error": None}
        try:
            adapter = self.registry.adapter_for(model)
            req = JobRequest.from_dict({"model": model, **extra})
            params = dict(req.params)
            params.setdefault("mode", self._settings["mode"])
            params.setdefault("efficiency", self._settings["efficiency"])
            req.params = params
            est = adapter.estimate(_prepare(adapter, req))
        except Exception as e:
            row["error"] = str(e) if isinstance(e, (UnknownModel, AdapterUnavailable)) else f"{type(e).__name__}: {e}"
            return row
        keys, prefix = self._needed_keys(adapter, req)
        warm = sum(c["size_bytes"] for c in comps if self._is_needed(c["key"], keys, prefix))
        need = int(est.peak)
        short = max(0, need - warm - free)
        row.update({"need_bytes": need, "warm_bytes": int(warm), "fits": short == 0, "short_by_bytes": short})
        return row

    def save_upload(self, name: Optional[str], data: bytes, content_type: Optional[str] = None) -> dict[str, Any]:
        """POST /v1/files. Raises files.UploadError."""
        return _files.save_upload(self.config.work_dir, name, data, content_type,
                                  max_bytes=int(self.config.max_upload_mb * 1024 * 1024))

    def output_file(self, job_id: str) -> tuple[int, str]:
        """GET /v1/jobs/{id}/output: (200, path) or (status, error message)."""
        job = self.get(job_id)
        if job is None:
            return 404, f"no job {job_id}"
        if job.state != JobState.DONE:
            return 409, f"job {job_id} is {job.state.value}; it has no output yet"
        out = job.result.get("output") or job.request.output
        if not out or not isinstance(out, str):
            return 404, f"job {job_id} has no output file"
        if not os.path.isfile(out):
            return 404, f"the output file of job {job_id} is gone: {out}"
        return 200, out

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _copy(job: Job) -> Job:
        return Job.from_dict(job.to_dict())

    def _publish(self, kind: str, job: Optional[Job] = None, **data: Any) -> None:
        if job is not None:
            data.setdefault("state", job.state.value)
            if job.message and "message" not in data:
                data["message"] = job.message
        self.bus.publish(Event(kind, job_id=job.id if job else None, data=data))

    def _wait(self, seconds: float) -> None:
        """Interruptible sleep: returns early on submit/cancel/resume/stop."""
        if self._sleep is not None:
            self._sleep(seconds)
            self._wake.clear()
            return
        self._wake.wait(max(0.0, seconds))
        self._wake.clear()

    def _render_busy(self) -> Optional[str]:
        if self._busy_re is None:
            return None
        try:
            procs = parse_ps(self._ps())
        except Exception:
            return None
        me = os.getpid()
        for pid, _rss, cmd in procs:
            if pid != me and self._busy_re.search(cmd):
                return RENDER_BUSY_MESSAGE
        return None

    def _needed_keys(self, adapter: Any, request: JobRequest) -> tuple[list[str], str]:
        keys: list[str] = []
        fn = getattr(adapter, "pool_keys", None)
        if callable(fn):
            try:
                keys = list(fn(request) or [])
            except Exception:
                keys = []
        return keys, f"{adapter.model_id}/"

    def _is_needed(self, key: str, keys: list[str], prefix: str) -> bool:
        """A pool entry belongs to this job's model when the adapter lists it in
        `pool_keys(request)`, or the model id is one of the key's '/'-separated parts
        (e.g. "dummy/dit", "longcat/<model id>/generator")."""
        return key in keys or key.startswith(prefix) or prefix[:-1] in key.split("/")

    def _free_unneeded(self, keys: list[str], prefix: str) -> list[str]:
        freed = []
        for e in self.pool.stats()["entries"]:
            if e["pinned"] or e.get("in_use") or self._is_needed(e["key"], keys, prefix):
                continue
            self.pool.release(e["key"])
            freed.append(e["key"])
        return freed

    def _set_waiting(self, job: Job, reason: str) -> None:
        with self._lock:
            cur = self.store.get(job.id)
            if cur is None or cur.state not in (JobState.QUEUED, JobState.WAITING_MEMORY):
                return
            changed = self._wait_reasons.get(job.id) != reason
            if cur.state != JobState.WAITING_MEMORY or cur.message != reason:
                cur.state = JobState.WAITING_MEMORY
                cur.message = reason
                self.store.update(cur)
            self._wait_reasons[job.id] = reason
        if changed:
            self._publish("job.waiting_memory", cur, reason=reason)

    def _finish(self, job: Job, state: JobState, message: str = "", kind: str = "") -> None:
        if is_load(job.request) and state == JobState.FAILED:
            self._load_errors[job.request.model] = message or "the model could not be loaded"
            self._load_error_details[job.request.model] = str((job.result or {}).get("error_detail") or "")
        job.state = state
        job.finished_at = time.time()
        job.message = message
        job.eta_s = None
        if state == JobState.DONE:
            job.progress = 1.0
        self.store.update(job)
        self._wait_reasons.pop(job.id, None)
        data: dict[str, Any] = {}
        if state == JobState.DONE:
            data["result"] = job.result
        self._publish(kind or f"job.{state.value}", job, **data)

    def _idle_maintenance(self, force: bool = False) -> None:
        now = self.clock()
        if force or now - self._last_idle_check >= min(self.idle_check_s, max(1.0, self.config.idle_unload_s)):
            self._last_idle_check = now
            try:
                self.pool.evict_idle(self.config.idle_unload_s)
            except Exception as e:  # pragma: no cover
                self.worker_errors.append(f"evict_idle: {e!r}")
            self._relieve_idle_pressure()

    def _relieve_idle_pressure(self) -> list[str]:
        """With no request running, critical memory pressure unloads idle models (loaded on purpose or not): the
        Mac's other apps come first. Returns the keys unloaded."""
        if self._running is not None:
            return []
        try:
            if self.probe.sample().pressure != PRESSURE_CRITICAL:
                return []
        except Exception:
            return []
        freed = []
        for e in self.pool.stats()["entries"]:
            if e["pinned"] or e.get("in_use"):
                continue
            self.pool.release(e["key"])
            freed.append(e["key"])
        if freed:
            self.bus.publish(Event("memory.relieved", data={"unloaded": freed, "reason": "critical memory pressure"}))
        return freed

    # ------------------------------------------------------------------ worker

    def _worker(self) -> None:
        while not self._stopping.is_set():
            try:
                if self._paused:
                    self._idle_maintenance()
                    self._wait(min(1.0, self.idle_check_s))
                    continue
                job = self.store.next_queued()
                if job is None:
                    self._idle_maintenance()
                    self._wait(min(self.idle_check_s, max(0.05, self.config.idle_unload_s or 1.0)))
                    continue
                self._step(job)
            except Exception:  # the worker must never die
                self.worker_errors.append(traceback.format_exc())
                self._wait(0.5)

    def _step(self, job: Job) -> None:
        req = job.request
        try:
            adapter = self.registry.adapter_for(req.model)
            preq = _prepare(adapter, req)
            est = load_estimate(adapter, preq) if is_load(req) else adapter.estimate(preq)
        except (UnknownModel, AdapterUnavailable) as e:
            self._finish(job, JobState.FAILED, str(e))
            return
        except Exception as e:
            try:
                import sys as _sys
                print(f"shutter: estimate for {job.id} ({req.model}) failed: {type(e).__name__}: {e}\n"
                      f"{traceback.format_exc()[-4000:]}", file=_sys.stderr, flush=True)
            except Exception:
                pass
            job.result = {"error_detail": f"estimate failed: {type(e).__name__}: {e}"[:2000]}
            self._finish(job, JobState.FAILED, plain_error(e, model_name(req.model), is_load(req)))
            return

        # 1. another render running on this Mac?
        busy = self._render_busy()
        if busy:
            self._set_waiting(job, busy)
            self._wait(self.config.memory_poll_s)
            return

        # 2. memory guard (warm components this job reuses are already resident)
        keys, prefix = self._needed_keys(adapter, req)
        warm = sum(e["size_bytes"] for e in self.pool.stats()["entries"]
                   if self._is_needed(e["key"], keys, prefix))
        need = max(0, int(est.peak) - warm)
        label = f"job {job.id} ({req.model})"
        d = self.guard.check(need, label)
        if not d.ok and self._free_unneeded(keys, prefix):
            d = self.guard.check(need, label)
        if not d.ok:
            self._set_waiting(job, d.reason)
            self._wait(self.config.memory_poll_s)
            return

        # 3. GPU locks: flock, then the shared mkdir dirs (all or nothing)
        if not self.gpu_lock.try_acquire(0):
            h = self.gpu_lock.holder() or {}
            who = h.get("label") or (h.get("raw") or "").strip() or "another process"
            self._set_waiting(job, f"GPU busy: {who}")
            self._wait(self.config.memory_poll_s)
            return
        reason = self.dir_locks.try_acquire()
        if reason is not None:
            self.gpu_lock.release()
            self._set_waiting(job, f"GPU busy: {reason}")
            self._wait(self.config.memory_poll_s)
            return

        try:
            with self._lock:
                cur = self.store.get(job.id)
                if cur is None or cur.state not in (JobState.QUEUED, JobState.WAITING_MEMORY):
                    return  # cancelled meanwhile
                job = cur
                job.state = JobState.RUNNING
                job.started_at = time.time()
                job.finished_at = None
                job.progress = 0.0
                job.stage = "starting"
                job.message = ""
                job.result = {}
                self._cancel.clear()
                self._abort_reason = None
                self._running = job
                self.store.update(job)
            self._run(job, adapter, est)
        finally:
            with self._lock:
                self._running = None
                self._step_info.pop(job.id, None)
            self.dir_locks.release()
            self.gpu_lock.release()
            if self.config.idle_unload_s <= 0:
                self.pool.evict_idle(0)
            else:
                self._idle_maintenance(force=True)

    def _run(self, job: Job, adapter: Any, est: Any) -> None:
        req = job.request
        work_dir = Path(self.config.work_dir) / job.id
        work_dir.mkdir(parents=True, exist_ok=True)
        loading = is_load(req)
        if not req.output and not loading:
            ext = getattr(adapter, "extension", None) or ("png" if getattr(adapter, "kind", "video") == "image" else "mp4")
            req.output = os.path.join(self.config.output_dir, f"{job.id}.{ext}")
        if req.output:
            Path(req.output).parent.mkdir(parents=True, exist_ok=True)

        base = self.guard.begin_run()
        peak = {"used": base.used, "swap": base.swap_used, "avail_min": base.available,
                "base_used": base.used, "own": 0}
        self._publish("job.started", job, estimate_peak=int(est.peak), model=req.model)

        stop_sampler = threading.Event()

        def take_sample(publish: bool) -> dict[str, Any]:
            s = self.probe.sample()
            peak["used"] = max(peak["used"], s.used)
            peak["swap"] = max(peak["swap"], s.swap_used)
            peak["avail_min"] = min(peak["avail_min"], s.available)
            if getattr(adapter, "family", "") != "dummy":
                v = _mlx_held()
                if v is not None:
                    peak["own"] = max(peak["own"], v)
            d = s.to_dict()
            if publish:
                self.bus.publish(Event("memory.sample", job_id=job.id, data={
                    "total": s.total, "available": s.available, "swap_used": s.swap_used,
                    "pressure": s.pressure, "used": s.used}))
            return d

        def sampler() -> None:
            while not stop_sampler.wait(self.sample_interval_s):
                try:
                    take_sample(True)
                except Exception:
                    pass

        sampler_t = threading.Thread(target=sampler, name="shutter-mem-sampler", daemon=True)
        sampler_t.start()

        t_start = self.clock()
        state = {"persist": t_start, "publish": 0.0, "watch": t_start, "stage": None}

        self._steprate.reset()
        eta = EtaTracker(self._decode_estimate(job), clock=self.clock)
        self._eta = eta

        def report(progress: float, stage: str = "") -> None:
            p = min(1.0, max(0.0, float(progress)))
            now = self.clock()
            st = parse_step(stage)
            if st is not None:
                self._steprate.update(*st)
            eta.update(stage or "", now)
            est = eta.estimate(now)
            with self._lock:
                job.progress = p
                job.stage = stage or job.stage
                if not job.cancel_requested:
                    job.eta_s = est["eta_s"]
                    job.eta_decode_known = est["decode_known"]
                if now - state["persist"] >= self.progress_persist_s or p >= 1.0:
                    state["persist"] = now
                    self.store.update(job)
            if stage != state["stage"] or now - state["publish"] >= 0.1 or p >= 1.0:
                state["publish"] = now
                state["stage"] = stage
                self._publish("job.progress", job, progress=p, stage=job.stage, eta_s=job.eta_s,
                              eta_decode_known=job.eta_decode_known)

        wd = {"relieve": False, "stop_at": None, "lock": threading.Lock(), "events": []}

        def relieve_now() -> None:
            """Step down without stopping: free MLX's cache and every idle model this job does not use."""
            mx = _mlx()
            try:
                if mx is not None:
                    mx.clear_cache()
            except Exception:
                pass
            try:  # only models this job does not use: dropping its own would make it load them again
                keys, prefix = self._needed_keys(adapter, req)
                self._free_unneeded(keys, prefix)
            except Exception:
                pass

        def own_now() -> Optional[int]:
            if getattr(adapter, "family", "") == "dummy":
                return None
            v = _mlx_held()
            if v is not None:
                peak["own"] = max(peak["own"], v)
            return v

        def watch_once() -> None:
            own = own_now()
            with wd["lock"]:
                action, why = self.guard.watch_step(own_bytes=own)
            if action == ACTION_RELIEVE and not wd["relieve"]:
                wd["relieve"] = True
                wd["events"].append({"t": round(self.clock() - t_start, 1), "action": "relieve", "why": why})
                self._publish("job.memory", job, action="relieve", reason=why)
            elif action == ACTION_STOP and not self._abort_reason:
                self._abort_reason = f"{STOP_BODY} ({why})"
                wd["stop_at"] = self.clock()
                wd["events"].append({"t": round(self.clock() - t_start, 1), "action": "stop", "why": why})
                self._publish("job.memory", job, action="stop", reason=why)
                self._cancel.set()

        def check_cancelled() -> None:
            if wd["relieve"]:
                wd["relieve"] = False
                relieve_now()
            if self._abort_reason:
                raise Cancelled(self._abort_reason)
            if self._cancel.is_set():
                if self._stopping.is_set():
                    raise _Shutdown("engine is shutting down")
                raise Cancelled("cancelled")
            now = self.clock()
            if now - state["watch"] >= self.watch_interval_s:
                state["watch"] = now
                watch_once()
                if self._abort_reason:
                    raise Cancelled(self._abort_reason)

        def emergency_exit() -> None:  # pragma: no cover - ends the process
            msg = PROTECT_PREFIX + (self._abort_reason or STOP_BODY) + " The engine restarted to free the memory."
            try:
                with self._lock:
                    job.state = JobState.FAILED
                    job.message = msg
                    job.finished_at = time.time()
                    job.result = {"watchdog": wd["events"]}
                    self.store.update(job)
            except Exception:
                pass
            try:
                self.dir_locks.release()
            except Exception:
                pass
            import sys as _sys
            print(f"shutter: {msg}", file=_sys.stderr, flush=True)
            os._exit(EMERGENCY_EXIT_CODE)

        stop_watch = threading.Event()

        def watchdog_thread() -> None:
            while not stop_watch.wait(self.watchdog_interval_s):
                try:
                    watch_once()
                    if wd["stop_at"] is not None and self.clock() - wd["stop_at"] >= self.emergency_s:
                        trouble = self.guard.watchdog.trouble(self.probe.sample())
                        if trouble and not stop_watch.is_set():
                            emergency_exit()
                except Exception:
                    pass

        watch_t = threading.Thread(target=watchdog_thread, name="shutter-watchdog", daemon=True)
        watch_t.start()

        # Hard cap: MLX may not grow past what was free when this job was admitted (plus what it already holds).
        mx = _mlx() if not getattr(adapter, "family", "") == "dummy" else None
        prev_limit = None
        cap_bytes = None
        if mx is not None and hasattr(mx, "set_memory_limit"):
            try:
                cap_bytes = _mlx_active() + admission_budget(base, self.guard.headroom_bytes)
                prev_limit = mx.set_memory_limit(int(max(cap_bytes, int(est.peak))))
            except Exception:
                prev_limit = None

        ctx = RunContext(
            models=self.pool, cache=self.cache, work_dir=work_dir, report=report,
            check_cancelled=check_cancelled, memory_sample=lambda: take_sample(False),
            settings={"job_id": job.id, "output": req.output, "home": self.config.home,
                      "headroom_bytes": self.guard.headroom_bytes,
                      "sleep": self._sleep},
            on_step=lambda info: self._step_info.update({job.id: dict(info)}),
        )
        t0 = time.perf_counter()
        outcome = None
        run_job = Job(request=_prepare(adapter, req), id=job.id, state=job.state, created_at=job.created_at,
                      started_at=job.started_at, node=job.node, client=job.client)
        try:
            if loading:
                warm = getattr(adapter, "warm", None)
                if not callable(warm):
                    raise LoadUnsupported(f"{model_name(req.model, getattr(adapter, 'family', ''))} loads when a "
                                          f"request needs it; it cannot be loaded ahead of time.")
                result = warm(run_job, ctx) or {}
                held = self._model_keys(req.model)
                for k in held:
                    self.pool.hold(k)
                result["loaded_keys"] = held
                result["loaded_bytes"] = int(sum(e["size_bytes"] for e in self.pool.stats()["entries"]
                                                 if e["key"] in held))
            else:
                result = adapter.run(run_job, ctx) or {}
            if self._abort_reason:  # adapter swallowed the Cancelled
                raise Cancelled(self._abort_reason)
            if self._cancel.is_set() and not self._stopping.is_set():
                raise Cancelled("cancelled")
            note = ""
            if result.get("stepped_down_from") and result.get("rung"):
                try:
                    from shutter.core.modes import rung as _rung
                    note = (f"Ran in {_rung(str(result['rung']))['label']} (instead of "
                            f"{_rung(str(result['stepped_down_from']))['label']}) to fit the memory free right now.")
                except Exception:
                    note = f"Ran in a lower memory step ({result['rung']}) to fit the memory free right now."
            outcome = (JobState.DONE, note, dict(result))
        except _Shutdown:
            outcome = (JobState.QUEUED, "interrupted by shutdown; it will run again", {})
        except Cancelled:
            if self._abort_reason:
                outcome = (JobState.FAILED, PROTECT_PREFIX + self._abort_reason, {})
            else:
                outcome = (JobState.CANCELLED, "cancelled while running", {})
        except MemoryRefused as e:
            n = self._midrun_refusals.get(job.id, 0) + 1
            self._midrun_refusals[job.id] = n
            if n >= MAX_MIDRUN_REFUSALS:
                outcome = (JobState.FAILED, f"memory refused {n} times mid-run: {e.reason}", {})
            else:
                outcome = (JobState.WAITING_MEMORY, e.reason, {})
        except Exception as e:
            if self._abort_reason:  # e.g. an MLX allocation failed while the watchdog was stopping the job
                outcome = (JobState.FAILED, PROTECT_PREFIX + self._abort_reason, {})
            elif _is_oom(e):
                outcome = (JobState.FAILED, STOP_MESSAGE, {"error_detail": f"{type(e).__name__}: {e}"[:2000]})
            else:
                detail = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()[-4000:]
                try:
                    import sys as _sys
                    print(f"shutter: {'load' if loading else 'job'} {job.id} ({req.model}) failed: {detail}\n{tb}",
                          file=_sys.stderr, flush=True)
                except Exception:
                    pass
                name = model_name(req.model, getattr(adapter, "family", ""))
                outcome = (JobState.FAILED, plain_error(e, name, loading),
                           {"error_detail": detail[:2000], "traceback": tb})
        finally:
            stop_sampler.set()
            stop_watch.set()
            if mx is not None and prev_limit is not None:
                try:
                    mx.set_memory_limit(int(prev_limit))
                except Exception:
                    pass
            try:
                take_sample(False)
            except Exception:
                pass
            self.guard.end_run()

        state_, message, result = outcome
        seconds = time.perf_counter() - t0
        result.setdefault("seconds", round(seconds, 3))
        adapter_peak = result.get("peak_bytes", result.get("peak_memory_bytes"))
        try:
            # the request's own peak: the adapter's figure, else (a failed or stopped run) the most MLX held
            # while it ran, else the Mac's growth; never the whole Mac's used memory (0.3.0 showed 12.8 GB)
            own_peak = peak["own"] if peak["own"] > 0 else max(0, peak["used"] - peak["base_used"])
            result["peak_bytes"] = int(adapter_peak) if adapter_peak is not None else int(own_peak)
        except (TypeError, ValueError):
            result["peak_bytes"] = int(max(0, peak["used"] - peak["base_used"]))
        if peak["own"] > 0:
            result["peak_memory_own_bytes"] = int(peak["own"])
        result["peak_memory"] = {"used_bytes": int(peak["used"]),
                                 "delta_bytes": int(peak["used"] - peak["base_used"]),
                                 "available_min": int(peak["avail_min"]),
                                 "swap_used_max": int(peak["swap"]),
                                 "swap_used_start": int(base.swap_used)}
        result["estimate_peak_bytes"] = int(est.peak)
        result["watchdog"] = {"events": wd["events"], "max_pressure": self.guard.watchdog.max_pressure,
                              "swap_growth_max_bytes": int(self.guard.watchdog.max_swap_growth),
                              "mlx_cap_bytes": int(cap_bytes) if cap_bytes else None}
        if state_ == JobState.DONE and not loading:
            result.setdefault("output", req.output)
        # Give the GPU back (and, after a protect-abort, every model) before the job's
        # final state is visible, so observers never see a finished job holding locks.
        self.dir_locks.release()
        self.gpu_lock.release()
        if state_ == JobState.FAILED and message.startswith(PROTECT_PREFIX):
            self.pool.release_all()
        with self._lock:
            job.result = result
            if state_ in (JobState.QUEUED, JobState.WAITING_MEMORY):
                job.state = state_
                job.message = message
                job.started_at = None
                job.progress = 0.0
                job.stage = ""
                self.store.update(job)
                if state_ == JobState.WAITING_MEMORY:
                    self._wait_reasons[job.id] = message
                    self._publish("job.waiting_memory", job, reason=message)
                else:
                    self._publish("job.requeued", job)
            else:
                self._midrun_refusals.pop(job.id, None)
                self._finish(job, state_, message)
        if state_ == JobState.DONE and not self.config.keep_work_dirs:
            wd = str(work_dir.resolve()) + os.sep
            paths = [req.output] + [v for v in result.values() if isinstance(v, str)]
            if not any(str(Path(p).resolve()).startswith(wd) for p in paths if p and os.sep in p):
                shutil.rmtree(work_dir, ignore_errors=True)
