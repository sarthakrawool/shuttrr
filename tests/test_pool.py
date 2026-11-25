import threading
import time

import pytest

from shutter.core import pool as pool_mod
from shutter.core.pool import ModelPool, free_memory
from shutter.types import MemoryRefused

MB = 1024**2


class FakeGuard:
    """Admits while the pool's bytes + need fit in `capacity`."""

    def __init__(self, capacity, pool_ref=None):
        self.capacity = capacity
        self.pool = pool_ref
        self.calls = []

    def admit(self, need_bytes, what=""):
        self.calls.append((need_bytes, what))
        used = self.pool.total_bytes if self.pool else 0
        if used + need_bytes > self.capacity:
            raise MemoryRefused(f"need {need_bytes} for {what}, used {used}")


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_pool(capacity=100 * MB, **kw):
    clock = FakeClock()
    events = []
    guard = FakeGuard(capacity)
    p = ModelPool(guard, on_event=events.append, clock=clock, **kw)
    guard.pool = p
    return p, guard, clock, events


def loader_for(value, counter=None):
    def load():
        if counter is not None:
            counter.append(value)
        return value
    return load


def test_hit_and_miss():
    p, guard, clock, events = make_pool()
    calls = []
    assert p.get("a", loader_for("A", calls), 10 * MB) == "A"
    assert p.get("a", loader_for("A2", calls), 10 * MB) == "A"
    assert calls == ["A"]
    assert len(guard.calls) == 1 and guard.calls[0] == (10 * MB, "a")
    s = p.stats()
    assert s["hits"] == 1 and s["misses"] == 1 and s["total_bytes"] == 10 * MB
    assert s["entries"][0]["hits"] == 1
    assert [e.kind for e in events] == ["model.loaded"]
    assert events[0].data["key"] == "a" and events[0].data["size_bytes"] == 10 * MB


def test_keep_false_not_cached():
    p, guard, *_ = make_pool()
    calls = []
    p.get("a", loader_for("A", calls), 10 * MB, keep=False)
    p.get("a", loader_for("A", calls), 10 * MB, keep=False)
    assert calls == ["A", "A"]
    assert "a" not in p
    assert len(guard.calls) == 2


def test_hit_updates_last_used_and_eviction_order_is_lru():
    p, guard, clock, events = make_pool(capacity=30 * MB)
    p.get("a", loader_for("A"), 10 * MB)
    clock.advance(1)
    p.get("b", loader_for("B"), 10 * MB)
    clock.advance(1)
    p.get("c", loader_for("C"), 10 * MB)
    clock.advance(1)
    p.get("a", loader_for("A"), 10 * MB)  # a is now most recent; b is LRU
    clock.advance(1)
    p.get("d", loader_for("D"), 10 * MB)
    assert set(p.keys()) == {"a", "c", "d"}
    clock.advance(1)
    p.get("e", loader_for("E"), 20 * MB)  # needs two evictions: c then a
    assert set(p.keys()) == {"d", "e"}
    unloaded = [e.data["key"] for e in events if e.kind == "model.unloaded"]
    assert unloaded == ["b", "c", "a"]
    assert p.stats()["evictions"] == 3


def test_evicts_one_at_a_time_and_retries():
    p, guard, clock, _ = make_pool(capacity=40 * MB)
    for k in "abcd":
        p.get(k, loader_for(k), 10 * MB)
        clock.advance(1)
    guard.calls.clear()
    p.get("x", loader_for("x"), 10 * MB)
    # one refusal, one eviction, then admitted
    assert len(guard.calls) == 2
    assert set(p.keys()) == {"b", "c", "d", "x"}


def test_refusal_propagates_when_nothing_evictable():
    p, guard, clock, _ = make_pool(capacity=20 * MB)
    p.get("a", loader_for("A"), 10 * MB, pinned=True)
    with p.using("b"):
        p.get("b", loader_for("B"), 10 * MB)
        called = []
        with pytest.raises(MemoryRefused):
            p.get("c", loader_for("C", called), 5 * MB)
        assert called == []
    assert set(p.keys()) == {"a", "b"}
    assert "c" not in p


def test_refused_even_when_pool_empty():
    p, *_ = make_pool(capacity=10 * MB)
    with pytest.raises(MemoryRefused) as ei:
        p.get("huge", loader_for("H"), 50 * MB)
    assert "huge" in ei.value.reason
    assert p.keys() == []


def test_in_use_never_evicted():
    p, guard, clock, _ = make_pool(capacity=20 * MB)
    p.get("old", loader_for("O"), 10 * MB)
    clock.advance(1)
    p.get("new", loader_for("N"), 10 * MB)
    with p.using("old"):
        assert p.in_use("old")
        p.get("x", loader_for("X"), 10 * MB)  # must evict "new", not the LRU "old"
        assert set(p.keys()) == {"old", "x"}
        assert p.evict_idle(0) == ["x"]
    assert not p.in_use("old")
    assert p.evict_idle(0) == ["old"]


def test_acquire_refcount_nested():
    p, *_ = make_pool()
    p.get("a", loader_for("A"), MB)
    p.acquire("a")
    p.acquire("a")
    p.relinquish("a")
    assert p.in_use("a")
    assert p.evict_idle(0) == []
    p.relinquish("a")
    assert not p.in_use("a")
    assert p.evict_idle(0) == ["a"]


def test_release_while_in_use_is_deferred():
    p, _, _, events = make_pool()
    p.get("a", loader_for("A"), MB)
    with p.using("a"):
        p.release("a")
        assert "a" in p
    assert "a" not in p
    assert events[-1].kind == "model.unloaded" and events[-1].data["reason"] == "released"


def test_pinned_not_evicted_and_unpin():
    p, guard, clock, _ = make_pool(capacity=20 * MB)
    p.get("pin", loader_for("P"), 10 * MB, pinned=True)
    clock.advance(1)
    p.get("b", loader_for("B"), 10 * MB)
    clock.advance(1)
    p.get("c", loader_for("C"), 10 * MB)
    assert set(p.keys()) == {"pin", "c"}
    assert p.evict_idle(0) == ["c"]
    p.unpin("pin")
    assert p.evict_idle(0) == ["pin"]
    p.get("z", loader_for("Z"), MB)
    p.pin("z")
    assert p.stats()["entries"][0]["pinned"] is True
    with pytest.raises(KeyError):
        p.pin("missing")
    with pytest.raises(KeyError):
        p.unpin("missing")


def test_pinned_on_hit():
    p, *_ = make_pool()
    p.get("a", loader_for("A"), MB)
    p.get("a", loader_for("A"), MB, pinned=True)
    assert p.evict_idle(0) == []


def test_loader_failure_caches_nothing():
    p, _, _, events = make_pool()

    def boom():
        raise RuntimeError("bad weights")

    with pytest.raises(RuntimeError):
        p.get("a", boom, MB)
    assert "a" not in p
    assert events == []
    assert p.get("a", loader_for("A"), MB) == "A"  # can retry afterwards


def test_release_and_release_all():
    p, _, _, events = make_pool()
    for k in ("longcat/te", "longcat/dit", "longcat/vae", "other/x"):
        p.get(k, loader_for(k), MB, pinned=(k == "longcat/dit"))
    p.release("longcat/te")
    assert "longcat/te" not in p
    p.release("not-there")  # ignored
    p.release_all("longcat/")
    assert p.keys() == ["other/x"]
    p.release_all()
    assert p.keys() == []
    reasons = {e.data["reason"] for e in events if e.kind == "model.unloaded"}
    assert reasons == {"released"}


def test_evict_idle_ttl_with_fake_clock():
    p, _, clock, events = make_pool()
    p.get("a", loader_for("A"), MB)
    clock.advance(30)
    p.get("b", loader_for("B"), MB)
    clock.advance(20)
    assert p.evict_idle(60) == []
    clock.advance(10)  # a idle 60, b idle 30
    assert p.evict_idle(60) == ["a"]
    p.get("b", loader_for("B"), MB)  # touch b
    clock.advance(59)
    assert p.evict_idle(60) == []
    clock.advance(1)
    assert p.evict_idle(60) == ["b"]
    assert [e.data["reason"] for e in events if e.kind == "model.unloaded"] == ["idle", "idle"]


def test_stats_shape():
    p, _, clock, _ = make_pool()
    p.get("a", loader_for("A"), 3 * MB)
    clock.advance(5)
    p.get("b", loader_for("B"), 4 * MB)
    clock.advance(2)
    s = p.stats()
    assert s["count"] == 2 and s["total_bytes"] == 7 * MB
    by = {e["key"]: e for e in s["entries"]}
    assert by["a"]["age_s"] == 7 and by["a"]["idle_s"] == 7
    assert by["b"]["age_s"] == 2
    assert by["a"]["in_use"] is False and by["a"]["pinned"] is False
    assert s["entries"][0]["key"] == "b"  # most recently used first


def test_free_memory_called_on_release(monkeypatch):
    calls = []
    monkeypatch.setattr(pool_mod, "free_memory", lambda: calls.append(1))
    p, *_ = make_pool()
    p.get("a", loader_for("A"), MB)
    p.release("a")
    assert calls == [1]


def test_free_memory_runs():
    free_memory()  # must not raise with or without mlx


def test_reference_dropped_on_release():
    import weakref

    class Big:
        pass

    p, *_ = make_pool()
    holder = {}

    def load():
        obj = Big()
        holder["ref"] = weakref.ref(obj)
        return obj

    p.get("a", load, MB)
    assert holder["ref"]() is not None
    p.release("a")
    assert holder["ref"]() is None


def test_event_sink_errors_do_not_break_pool():
    def bad_sink(ev):
        raise ValueError("sink down")

    p = ModelPool(FakeGuard(100 * MB), on_event=bad_sink)
    assert p.get("a", loader_for("A"), MB) == "A"
    p.release("a")


def test_no_events_without_sink():
    p = ModelPool(FakeGuard(100 * MB))
    p.get("a", loader_for("A"), MB)
    p.release("a")


def test_concurrent_get_same_key_loads_once():
    p = ModelPool(FakeGuard(100 * MB))
    count = []
    barrier = threading.Barrier(4)

    def slow():
        count.append(1)
        time.sleep(0.05)
        return object()

    results = []

    def worker():
        barrier.wait()
        results.append(p.get("a", slow, MB))

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(count) == 1
    assert len({id(r) for r in results}) == 1


def test_stats_readable_during_load():
    p = ModelPool(FakeGuard(100 * MB))
    started, finish = threading.Event(), threading.Event()

    def slow():
        started.set()
        finish.wait(2)
        return "A"

    t = threading.Thread(target=lambda: p.get("a", slow, MB))
    t.start()
    assert started.wait(2)
    done = []
    reader = threading.Thread(target=lambda: done.append(p.stats()))
    reader.start()
    reader.join(1)
    assert done and done[0]["count"] == 0  # not blocked by the loader
    finish.set()
    t.join(2)
    assert "a" in p
