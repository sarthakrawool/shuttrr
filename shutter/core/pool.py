"""ModelPool: keeps model components warm between jobs and gives memory back on demand.

Every load is admitted by the memory guard first. When the guard refuses, the pool evicts
its own least-recently-used, unpinned, idle entries one at a time and asks again; if it
still refuses, `MemoryRefused` propagates and the caller (the engine) parks the job in
`waiting_memory`. Entries that are in use (see `using()` / `acquire()`) are never evicted.
"""

from __future__ import annotations

import contextlib
import gc
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from shutter.types import Event, MemoryRefused

__all__ = ["ModelPool", "PoolEntry", "free_memory", "residency_of"]

RESIDENCIES = ("resident", "streamed")


def residency_of(obj: Any, given: Optional[str] = None) -> str:
    """"streamed" when the component's weights are streamed from SSD, else "resident". `given` wins;
    otherwise the object may say so itself (`obj.residency`), and LongCat's streamed generator is
    recognised by its stream config."""
    if given in RESIDENCIES:
        return given  # type: ignore[return-value]
    r = getattr(obj, "residency", None)
    if r in RESIDENCIES:
        return r
    if isinstance(obj, dict) and obj.get("residency") in RESIDENCIES:
        return obj["residency"]
    try:
        if getattr(obj, "_shutter_stream_cfg", None):
            return "streamed"
    except Exception:  # pragma: no cover - exotic objects
        pass
    return "resident"


def free_memory() -> None:
    """Return freed memory to the system: run the GC and, when MLX is importable, drop
    MLX's buffer cache."""
    gc.collect()
    try:
        import mlx.core as mx  # type: ignore
    except Exception:  # pragma: no cover - mlx missing
        return
    try:
        clear = getattr(mx, "clear_cache", None)
        if clear is None:  # older MLX
            clear = getattr(getattr(mx, "metal", None), "clear_cache", None)
        if clear is not None:
            clear()
    except Exception:  # pragma: no cover - never let cleanup break the caller
        pass


@dataclass
class PoolEntry:
    obj: Any
    size_bytes: int
    last_used: float
    loaded_at: float
    pinned: bool = False
    hits: int = 0
    residency: str = "resident"   # "resident" | "streamed" (weights read from SSD per block)
    # Loaded on purpose (POST /v1/models/<id>/load): the idle timer leaves it alone, but it can still be evicted when
    # another model needs the memory, and an explicit release drops it.
    held: bool = False
    drop_when_idle: bool = field(default=False, repr=False)


class ModelPool:
    """Implements `adapters.base.ModelPoolLike`.

    `guard` needs only `admit(need_bytes, what="")`, raising `MemoryRefused` on refusal.
    `on_event(Event)` receives "model.loaded" / "model.unloaded".
    `clock` is used for last_used / TTL bookkeeping (inject a fake one in tests).
    """

    def __init__(self, guard: Any, on_event: Optional[Callable[[Event], None]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.guard = guard
        self.on_event = on_event
        self.clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, PoolEntry] = {}
        self._inuse: dict[str, int] = {}
        self._loading: dict[str, threading.Lock] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # -- events / freeing --------------------------------------------------------------

    def _emit(self, kind: str, **data: Any) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(Event(kind, data=data))
        except Exception:
            pass  # an event sink must never break model loading

    def _drop(self, key: str, reason: str) -> Optional[PoolEntry]:
        """Remove `key` from the table (caller holds the lock). Returns the entry."""
        entry = self._entries.pop(key, None)
        if entry is not None and reason in ("evicted", "idle"):
            self._evictions += 1
        return entry

    def _after_drop(self, dropped: list[tuple[str, PoolEntry, str]]) -> None:
        """Free memory and publish events for entries already removed (lock not held)."""
        if not dropped:
            return
        for _, entry, _ in dropped:
            # a streamed model's SSD reader thread holds the blocks it read ahead: stop it, or they stay in memory
            blocks = getattr(entry.obj, "_streamed_blocks", None)
            if blocks is not None and callable(getattr(blocks, "close", None)):
                try:
                    blocks.close()
                except Exception:
                    pass
            entry.obj = None
        free_memory()
        for key, entry, reason in dropped:
            self._emit("model.unloaded", key=key, size_bytes=entry.size_bytes, reason=reason)

    # -- in-use refcount ---------------------------------------------------------------

    def acquire(self, key: str) -> None:
        """Mark `key` as in use; it will not be evicted until `relinquish(key)`. May be
        called before the entry is loaded."""
        with self._lock:
            self._inuse[key] = self._inuse.get(key, 0) + 1

    def relinquish(self, key: str) -> None:
        """Undo one `acquire(key)`. A `release()` requested while the entry was in use is
        carried out when the last user lets go."""
        dropped = []
        with self._lock:
            n = self._inuse.get(key, 0)
            if n <= 1:
                self._inuse.pop(key, None)
                entry = self._entries.get(key)
                if entry is not None and entry.drop_when_idle:
                    self._drop(key, "released")
                    dropped.append((key, entry, "released"))
            else:
                self._inuse[key] = n - 1
        self._after_drop(dropped)

    @contextlib.contextmanager
    def using(self, key: str) -> Iterator[None]:
        """Context manager form of acquire/relinquish."""
        self.acquire(key)
        try:
            yield
        finally:
            self.relinquish(key)

    def in_use(self, key: str) -> bool:
        with self._lock:
            return self._inuse.get(key, 0) > 0

    # -- loading -----------------------------------------------------------------------

    def _hit(self, key: str, pinned: bool) -> tuple[bool, Any]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False, None
            entry.last_used = self.clock()
            entry.hits += 1
            entry.drop_when_idle = False
            if pinned:
                entry.pinned = True
            self._hits += 1
            return True, entry.obj

    def _pick_victim(self, exclude: str) -> Optional[str]:
        candidates = [
            (e.last_used, k) for k, e in self._entries.items()
            if not e.pinned and self._inuse.get(k, 0) == 0 and k != exclude
        ]
        if not candidates:
            return None
        return min(candidates)[1]

    def _admit(self, key: str, size_bytes: int) -> None:
        while True:
            try:
                self.guard.admit(size_bytes, key)
                return
            except MemoryRefused:
                with self._lock:
                    victim = self._pick_victim(exclude=key)
                    entry = self._drop(victim, "evicted") if victim is not None else None
                if entry is None:
                    raise
                self._after_drop([(victim, entry, "evicted")])

    def get(self, key: str, loader: Callable[[], Any], size_bytes: int, *, keep: bool = True,
            pinned: bool = False, residency: Optional[str] = None) -> Any:
        """Return the cached component `key`, or admit `size_bytes` through the guard
        (evicting idle LRU entries if refused) and call `loader()`. keep=False loads without
        caching. Raises `MemoryRefused` if memory cannot be found; loader errors propagate
        and nothing is cached."""
        found, obj = self._hit(key, pinned)
        if found:
            return obj
        if not keep:
            with self._lock:
                self._misses += 1
            self._admit(key, size_bytes)
            return loader()

        with self._lock:
            klock = self._loading.setdefault(key, threading.Lock())
        with klock:
            # another thread may have loaded it while we waited
            found, obj = self._hit(key, pinned)
            if found:
                return obj
            with self._lock:
                self._misses += 1
            self._admit(key, size_bytes)
            t0 = time.perf_counter()
            obj = loader()  # outside the pool lock
            load_s = time.perf_counter() - t0
            now = self.clock()
            with self._lock:
                self._entries[key] = PoolEntry(obj=obj, size_bytes=int(size_bytes), last_used=now,
                                               loaded_at=now, pinned=pinned,
                                               residency=residency_of(obj, residency))
        self._emit("model.loaded", key=key, size_bytes=int(size_bytes), load_s=round(load_s, 4))
        return obj

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._entries)

    # -- releasing ---------------------------------------------------------------------

    def release(self, key: str) -> None:
        """Drop `key` now and return its memory. If it is in use, it is dropped when the
        last user lets go. Unknown keys are ignored."""
        dropped = []
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if self._inuse.get(key, 0) > 0:
                entry.drop_when_idle = True
            else:
                self._drop(key, "released")
                dropped.append((key, entry, "released"))
        self._after_drop(dropped)

    def release_all(self, prefix: str = "") -> None:
        """Release every entry whose key starts with `prefix` (pinned ones included)."""
        dropped = []
        with self._lock:
            for key in [k for k in self._entries if k.startswith(prefix)]:
                entry = self._entries[key]
                if self._inuse.get(key, 0) > 0:
                    entry.drop_when_idle = True
                else:
                    self._drop(key, "released")
                    dropped.append((key, entry, "released"))
        self._after_drop(dropped)

    def evict_idle(self, ttl_s: float) -> list[str]:
        """Evict unpinned, unheld, unused entries not touched for at least `ttl_s` seconds."""
        dropped = []
        with self._lock:
            now = self.clock()
            for key, entry in list(self._entries.items()):
                if entry.pinned or entry.held or self._inuse.get(key, 0) > 0:
                    continue
                if now - entry.last_used >= ttl_s:
                    self._drop(key, "idle")
                    dropped.append((key, entry, "idle"))
        self._after_drop(dropped)
        return [k for k, _, _ in dropped]

    def pin(self, key: str) -> None:
        with self._lock:
            if key not in self._entries:
                raise KeyError(key)
            self._entries[key].pinned = True

    def hold(self, key: str, held: bool = True) -> None:
        """Mark an entry as loaded on purpose (kept past the idle timer) or clear the mark."""
        with self._lock:
            if key not in self._entries:
                raise KeyError(key)
            self._entries[key].held = held

    def unpin(self, key: str) -> None:
        with self._lock:
            if key not in self._entries:
                raise KeyError(key)
            self._entries[key].pinned = False

    # -- introspection -----------------------------------------------------------------

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return sum(e.size_bytes for e in self._entries.values())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            entries = [
                {
                    "key": k,
                    "size_bytes": e.size_bytes,
                    "age_s": now - e.loaded_at,
                    "idle_s": now - e.last_used,
                    "pinned": e.pinned,
                    "held": e.held,
                    "in_use": self._inuse.get(k, 0) > 0,
                    "hits": e.hits,
                    "residency": e.residency if e.residency != "resident" else residency_of(e.obj),
                }
                for k, e in sorted(self._entries.items(), key=lambda kv: -kv[1].last_used)
            ]
            return {
                "entries": entries,
                "count": len(entries),
                "total_bytes": sum(e["size_bytes"] for e in entries),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }
