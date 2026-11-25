"""Small bookkeeping shared by the reference server and the worker: step progress parsed from a
stage text, a running job's step rate, and the clients seen recently (GET /v1/clients).

The Rust daemon implements the same rules (daemon/src/stats.rs); docs/API.md has the shapes.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable, Optional

__all__ = ["parse_step", "StepRate", "ClientTracker", "CLIENT_WINDOW_S", "THROUGHPUT_WINDOW_S",
           "client_label", "throughput_from_jobs"]

CLIENT_WINDOW_S = 600.0       # GET /v1/clients: clients seen in the last 10 minutes
THROUGHPUT_WINDOW_S = 3600.0  # GET /v1/system throughput: requests finished in the last hour
MAX_CLIENT_LEN = 120

_STEP_RE = re.compile(r"\b(?:denoise|denoising|step|steps|sampling|sample)\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def parse_step(stage: Any) -> Optional[tuple[int, int]]:
    """(step, steps) from a stage text such as "denoise 3/8" or "segment 1/2 denoise 3/8"; None when
    the stage carries no step counter."""
    if not isinstance(stage, str):
        return None
    m = _STEP_RE.search(stage)
    if not m:
        return None
    step, steps = int(m.group(1)), int(m.group(2))
    if steps <= 0 or step < 0 or step > steps:
        return None
    return step, steps


class StepRate:
    """Steps per second of one running job. Step counters that restart (a new segment) keep counting
    up, so a multi-segment clip has one rate."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        self.base = 0
        self.last: Optional[tuple[int, int]] = None
        self.first_t: Optional[float] = None
        self.first_n = 0
        self.last_t: Optional[float] = None
        self.total = 0

    def update(self, step: Optional[int], steps: Optional[int]) -> None:
        if step is None or steps is None:
            return
        if self.last is not None and step < self.last[0]:
            self.base += self.last[1]
        self.last = (int(step), int(steps))
        n = self.base + int(step)
        now = self.clock()
        if self.first_t is None:
            self.first_t, self.first_n = now, n
        if n != self.total or self.last_t is None:
            self.last_t = now
        self.total = n

    def snapshot(self) -> dict[str, Any]:
        rate = None
        if self.first_t is not None and self.last_t is not None and self.last_t > self.first_t \
                and self.total > self.first_n:
            rate = round((self.total - self.first_n) / (self.last_t - self.first_t), 4)
        return {"step": self.last[0] if self.last else None,
                "steps": self.last[1] if self.last else None,
                "steps_per_s": rate}


def client_label(body_value: Any, header: Optional[str] = None, user_agent: Optional[str] = None) -> Optional[str]:
    """The submitter's free-text client name: the request's `client` field, else the
    X-Shutter-Client header, else the User-Agent's first word. Trimmed to 120 characters."""
    for v in (body_value, header):
        if isinstance(v, str) and v.strip():
            return v.strip()[:MAX_CLIENT_LEN]
    if isinstance(user_agent, str) and user_agent.strip():
        return user_agent.strip().split()[0][:MAX_CLIENT_LEN]
    return None


class ClientTracker:
    """Clients seen recently, in memory (a restart forgets them). `submitted` counts a request;
    `seen` only refreshes last_seen (any API call carrying X-Shutter-Client)."""

    def __init__(self, window_s: float = CLIENT_WINDOW_S, clock: Callable[[], float] = time.time):
        self.window_s = window_s
        self.clock = clock
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, Any]] = {}

    def _row(self, client: str) -> dict[str, Any]:
        row = self._rows.get(client)
        if row is None:
            row = {"client": client, "first_seen": self.clock(), "last_seen": self.clock(), "times": [],
                   "requests_total": 0}
            self._rows[client] = row
        return row

    def submitted(self, client: Optional[str]) -> None:
        if not client:
            return
        with self._lock:
            row = self._row(client)
            now = self.clock()
            row["last_seen"] = now
            row["times"].append(now)
            row["requests_total"] += 1
            self._prune(now)

    def seen(self, client: Optional[str]) -> None:
        if not client:
            return
        with self._lock:
            self._row(client)["last_seen"] = self.clock()

    def _prune(self, now: float) -> None:
        cut = now - self.window_s
        for k in list(self._rows):
            row = self._rows[k]
            row["times"] = [t for t in row["times"] if t >= cut]
            if row["last_seen"] < cut:
                del self._rows[k]

    def list(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            self._prune(now)
            rows = [{"client": r["client"], "requests": len(r["times"]), "requests_total": r["requests_total"],
                     "first_seen": r["first_seen"], "last_seen": r["last_seen"],
                     "idle_s": round(max(0.0, now - r["last_seen"]), 1)}
                    for r in self._rows.values()]
        rows.sort(key=lambda r: -r["last_seen"])
        return {"window_s": int(self.window_s), "clients": rows}


def throughput_from_jobs(jobs: list[Any], running: Optional[dict[str, Any]], now: Optional[float] = None,
                         window_s: float = THROUGHPUT_WINDOW_S) -> dict[str, Any]:
    """GET /v1/system "throughput": done jobs finished in the window, their mean wall seconds
    (finished_at - started_at), and the running job's step rate."""
    now = time.time() if now is None else now
    durs = []
    for j in jobs:
        fin, st = getattr(j, "finished_at", None), getattr(j, "started_at", None)
        if fin is None or fin < now - window_s:
            continue
        if st is not None and fin >= st:
            durs.append(fin - st)
        else:
            durs.append(None)
    known = [d for d in durs if d is not None]
    return {"window_s": int(window_s), "finished": len(durs),
            "mean_seconds": round(sum(known) / len(known), 3) if known else None,
            "running": running}
