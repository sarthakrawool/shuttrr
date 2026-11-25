"""Job persistence (sqlite) and the in-process event bus.

`JobStore` keeps every `Job` as a JSON blob plus a few indexed columns so the scheduler can
pick the next job cheaply. It survives restarts: jobs that were `running` when the process
died are put back to `queued` on open.

`EventBus` fans `Event`s out to any number of subscribers (SSE streams, the menu bar app,
tests). Each subscriber has its own bounded queue; a slow subscriber loses its oldest
events instead of blocking publishers or growing without limit.
"""

from __future__ import annotations

import collections
import json
import sqlite3
import threading
from typing import Iterable, Optional

from shutter.types import Event, Job, JobState

__all__ = ["JobStore", "EventBus", "Subscription"]

REQUEUE_MESSAGE = "requeued after restart"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    state       TEXT NOT NULL,
    priority    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS jobs_sched ON jobs(state, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at DESC);
"""


class JobStore:
    """sqlite-backed store of `Job`s. Thread-safe (one connection guarded by an RLock).

    `path` may be ":memory:" for tests.
    """

    def __init__(self, path: str = ":memory:"):
        self.path = str(path)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        with self._lock:
            if self.path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(_SCHEMA)
            self._recover()

    # -- internals ---------------------------------------------------------------------

    @staticmethod
    def _row_values(job: Job) -> tuple:
        return (
            job.id,
            job.state.value,
            int(job.request.priority or 0),
            float(job.created_at),
            json.dumps(job.to_dict(), separators=(",", ":")),
        )

    def _recover(self) -> None:
        rows = self._db.execute(
            "SELECT data FROM jobs WHERE state = ?", (JobState.RUNNING.value,)
        ).fetchall()
        if not rows:
            return
        self._db.execute("BEGIN")
        try:
            for (data,) in rows:
                job = Job.from_dict(json.loads(data))
                job.state = JobState.QUEUED
                job.message = REQUEUE_MESSAGE
                job.started_at = None
                job.progress = 0.0
                job.stage = ""
                self._db.execute(
                    "UPDATE jobs SET state=?, priority=?, created_at=?, data=? WHERE id=?",
                    self._row_values(job)[1:] + (job.id,),
                )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    # -- public API --------------------------------------------------------------------

    def add(self, job: Job) -> Job:
        """Insert a new job. Raises ValueError if the id already exists."""
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO jobs (id, state, priority, created_at, data) VALUES (?,?,?,?,?)",
                    self._row_values(job),
                )
            except sqlite3.IntegrityError as e:
                raise ValueError(f"job {job.id} already exists") from e
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            row = self._db.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_dict(json.loads(row[0])) if row else None

    def update(self, job: Job) -> Job:
        """Write the job's current fields. Raises KeyError if the job is unknown."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET state=?, priority=?, created_at=?, data=? WHERE id=?",
                self._row_values(job)[1:] + (job.id,),
            )
            if cur.rowcount == 0:
                raise KeyError(job.id)
        return job

    def list(self, states: Optional[Iterable[JobState]] = None, limit: int = 200) -> list[Job]:
        """Jobs newest first, optionally filtered by state."""
        sql = "SELECT data FROM jobs"
        args: list = []
        if states is not None:
            vals = [JobState(s).value for s in states]
            if not vals:
                return []
            sql += f" WHERE state IN ({','.join('?' * len(vals))})"
            args.extend(vals)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [Job.from_dict(json.loads(r[0])) for r in rows]

    def next_queued(self) -> Optional[Job]:
        """The job that should run next: highest priority, then oldest, among jobs that are
        QUEUED or WAITING_MEMORY."""
        with self._lock:
            row = self._db.execute(
                "SELECT data FROM jobs WHERE state IN (?, ?) "
                "ORDER BY priority DESC, created_at ASC, rowid ASC LIMIT 1",
                (JobState.QUEUED.value, JobState.WAITING_MEMORY.value),
            ).fetchone()
        return Job.from_dict(json.loads(row[0])) if row else None

    def finished_since(self, ts: float, states: Iterable[JobState] = (JobState.DONE,)) -> list[Job]:
        """Jobs in `states` whose finished_at is at or after `ts` (throughput numbers)."""
        vals = [JobState(s).value for s in states]
        if not vals:
            return []
        sql = (f"SELECT data FROM jobs WHERE state IN ({','.join('?' * len(vals))}) "
               "AND json_extract(data, '$.finished_at') >= ? ORDER BY created_at DESC LIMIT 10000")
        with self._lock:
            rows = self._db.execute(sql, vals + [float(ts)]).fetchall()
        return [Job.from_dict(json.loads(r[0])) for r in rows]

    def counts(self) -> dict[str, int]:
        """Number of jobs per state value (every state present, zero when none)."""
        out = {s.value: 0 for s in JobState}
        with self._lock:
            for state, n in self._db.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state"):
                out[state] = n
        return out

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            try:
                self._db.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "JobStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Subscription:
    """A subscriber's bounded queue of events. Drops the oldest event when full."""

    def __init__(self, bus: "EventBus", maxlen: int):
        self._bus = bus
        self._q: collections.deque[Event] = collections.deque(maxlen=maxlen)
        self._cond = threading.Condition()
        self._closed = False
        self.dropped = 0

    def _put(self, event: Event) -> None:
        with self._cond:
            if self._closed:
                return
            if self._q.maxlen is not None and len(self._q) == self._q.maxlen:
                self.dropped += 1
            self._q.append(event)
            self._cond.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[Event]:
        """Next event, waiting up to `timeout` seconds (None = forever). Returns None on
        timeout or once the subscription is closed and drained."""
        with self._cond:
            if not self._q and not self._closed:
                self._cond.wait_for(lambda: self._q or self._closed, timeout=timeout)
            if self._q:
                return self._q.popleft()
            return None

    def get_nowait(self) -> Optional[Event]:
        with self._cond:
            return self._q.popleft() if self._q else None

    def qsize(self) -> int:
        with self._cond:
            return len(self._q)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._bus._unsubscribe(self)

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self):
        while True:
            ev = self.get()
            if ev is None:
                return
            yield ev


class EventBus:
    """Thread-safe publish/subscribe. `maxlen` bounds each subscription's queue and the
    history kept for `recent()`."""

    def __init__(self, maxlen: int = 1000):
        self.maxlen = int(maxlen)
        self._lock = threading.Lock()
        self._subs: list[Subscription] = []
        self._recent: collections.deque[Event] = collections.deque(maxlen=self.maxlen)

    def publish(self, event: Event) -> None:
        with self._lock:
            self._recent.append(event)
            subs = list(self._subs)
        for s in subs:
            s._put(event)

    def subscribe(self, maxlen: Optional[int] = None) -> Subscription:
        sub = Subscription(self, maxlen or self.maxlen)
        with self._lock:
            self._subs.append(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            try:
                self._subs.remove(sub)
            except ValueError:
                pass

    def recent(self, n: int = 50) -> list[Event]:
        """The last `n` published events, oldest first."""
        if n <= 0:
            return []
        with self._lock:
            items = list(self._recent)
        return items[-n:]

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)
