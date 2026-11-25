import threading
import time

import pytest

from shutter.core.queue import EventBus, JobStore
from shutter.types import Event, Job, JobRequest, JobState


def make_job(priority=0, created_at=None, model="m", **kw):
    job = Job(request=JobRequest(model=model, prompt="p", priority=priority), **kw)
    if created_at is not None:
        job.created_at = created_at
    return job


@pytest.fixture
def store():
    s = JobStore(":memory:")
    yield s
    s.close()


def test_add_get_roundtrip(store):
    job = make_job(priority=3)
    job.request.params = {"steps": 8}
    store.add(job)
    got = store.get(job.id)
    assert got is not None
    assert got.to_dict() == job.to_dict()
    assert store.get("nope") is None


def test_add_duplicate_raises(store):
    job = make_job()
    store.add(job)
    with pytest.raises(ValueError):
        store.add(job)


def test_update(store):
    job = make_job()
    store.add(job)
    job.state = JobState.DONE
    job.progress = 1.0
    job.result = {"output": "/tmp/x.mp4"}
    store.update(job)
    got = store.get(job.id)
    assert got.state == JobState.DONE
    assert got.result == {"output": "/tmp/x.mp4"}
    with pytest.raises(KeyError):
        store.update(make_job())


def test_list_newest_first_and_filter(store):
    jobs = [make_job(created_at=100 + i) for i in range(5)]
    jobs[1].state = JobState.DONE
    jobs[3].state = JobState.FAILED
    for j in jobs:
        store.add(j)
    assert [j.id for j in store.list()] == [j.id for j in reversed(jobs)]
    assert [j.id for j in store.list(limit=2)] == [jobs[4].id, jobs[3].id]
    assert [j.id for j in store.list(states=[JobState.DONE, JobState.FAILED])] == [jobs[3].id, jobs[1].id]
    assert store.list(states=[]) == []
    assert [j.id for j in store.list(states=["queued"])] == [jobs[4].id, jobs[2].id, jobs[0].id]


def test_next_queued_priority_then_oldest(store):
    a = make_job(priority=0, created_at=1)
    b = make_job(priority=5, created_at=3)
    c = make_job(priority=5, created_at=2)
    d = make_job(priority=9, created_at=0)
    d.state = JobState.DONE
    for j in (a, b, c, d):
        store.add(j)
    assert store.next_queued().id == c.id
    c.state = JobState.RUNNING
    store.update(c)
    assert store.next_queued().id == b.id
    b.state = JobState.WAITING_MEMORY
    store.update(b)
    assert store.next_queued().id == b.id  # waiting_memory still considered
    b.state = JobState.CANCELLED
    store.update(b)
    assert store.next_queued().id == a.id
    a.state = JobState.DONE
    store.update(a)
    assert store.next_queued() is None


def test_priority_change_via_update(store):
    a = make_job(priority=0, created_at=1)
    b = make_job(priority=0, created_at=2)
    store.add(a)
    store.add(b)
    assert store.next_queued().id == a.id
    b.request.priority = 10
    store.update(b)
    assert store.next_queued().id == b.id


def test_counts_and_delete(store):
    for s in (JobState.QUEUED, JobState.QUEUED, JobState.DONE):
        store.add(make_job(state=s))
    c = store.counts()
    assert c["queued"] == 2 and c["done"] == 1 and c["running"] == 0
    assert set(c) == {s.value for s in JobState}
    j = store.list(states=[JobState.DONE])[0]
    assert store.delete(j.id) is True
    assert store.delete(j.id) is False
    assert store.counts()["done"] == 0


def test_persistence_across_reopen(tmp_path):
    path = tmp_path / "jobs.db"
    s = JobStore(path)
    job = make_job(priority=2)
    job.state = JobState.DONE
    job.message = "ok"
    s.add(job)
    s.close()
    s2 = JobStore(str(path))
    got = s2.get(job.id)
    assert got.to_dict() == job.to_dict()
    s2.close()


def test_wal_mode(tmp_path):
    s = JobStore(tmp_path / "jobs.db")
    mode = s._db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    s.close()


def test_crash_recovery_requeues_running(tmp_path):
    path = tmp_path / "jobs.db"
    s = JobStore(path)
    running = make_job(priority=1)
    running.state = JobState.RUNNING
    running.started_at = time.time()
    running.progress = 0.4
    done = make_job()
    done.state = JobState.DONE
    s.add(running)
    s.add(done)
    s.close()  # simulate crash: never marked finished

    s2 = JobStore(path)
    got = s2.get(running.id)
    assert got.state == JobState.QUEUED
    assert got.message == "requeued after restart"
    assert got.started_at is None
    assert got.progress == 0.0
    assert s2.get(done.id).state == JobState.DONE
    assert s2.next_queued().id == running.id
    assert s2.counts()["running"] == 0
    s2.close()


def test_threaded_access(store):
    def worker(n):
        for i in range(25):
            j = make_job(priority=i % 3)
            store.add(j)
            j.progress = 0.5
            store.update(j)
            store.get(j.id)
            store.next_queued()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.counts()["queued"] == 100


# -- EventBus -----------------------------------------------------------------------------

def test_publish_subscribe():
    bus = EventBus()
    sub = bus.subscribe()
    bus.publish(Event("job.queued", job_id="a"))
    ev = sub.get(timeout=1)
    assert ev.kind == "job.queued" and ev.job_id == "a"
    assert sub.get(timeout=0.01) is None


def test_multi_subscriber_each_gets_all():
    bus = EventBus()
    s1, s2 = bus.subscribe(), bus.subscribe()
    for i in range(3):
        bus.publish(Event("job.progress", data={"i": i}))
    for s in (s1, s2):
        assert [s.get(timeout=1).data["i"] for _ in range(3)] == [0, 1, 2]


def test_overflow_drops_oldest():
    bus = EventBus(maxlen=3)
    sub = bus.subscribe()
    for i in range(5):
        bus.publish(Event("x", data={"i": i}))
    assert sub.qsize() == 3
    assert sub.dropped == 2
    assert [sub.get(timeout=1).data["i"] for _ in range(3)] == [2, 3, 4]


def test_slow_subscriber_does_not_affect_others():
    bus = EventBus(maxlen=2)
    slow = bus.subscribe()
    fast = bus.subscribe(maxlen=100)
    for i in range(10):
        bus.publish(Event("x", data={"i": i}))
    assert fast.qsize() == 10
    assert slow.qsize() == 2


def test_recent_for_late_joiners():
    bus = EventBus(maxlen=4)
    for i in range(6):
        bus.publish(Event("x", data={"i": i}))
    assert [e.data["i"] for e in bus.recent(2)] == [4, 5]
    assert [e.data["i"] for e in bus.recent(10)] == [2, 3, 4, 5]
    assert bus.recent(0) == []
    late = bus.subscribe()
    assert late.get(timeout=0.01) is None  # subscription only sees new events


def test_close_unsubscribes_and_wakes_waiter():
    bus = EventBus()
    sub = bus.subscribe()
    assert bus.subscriber_count == 1
    result = []

    def waiter():
        result.append(sub.get())  # blocks forever until closed

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    sub.close()
    t.join(timeout=2)
    assert not t.is_alive()
    assert result == [None]
    assert bus.subscriber_count == 0
    bus.publish(Event("x"))
    assert sub.get(timeout=0.01) is None


def test_get_blocks_until_publish():
    bus = EventBus()
    sub = bus.subscribe()
    threading.Timer(0.05, lambda: bus.publish(Event("late"))).start()
    ev = sub.get(timeout=2)
    assert ev is not None and ev.kind == "late"


def test_concurrent_publishers():
    bus = EventBus(maxlen=10000)
    sub = bus.subscribe()

    def pub(n):
        for i in range(500):
            bus.publish(Event("x", data={"n": n, "i": i}))

    threads = [threading.Thread(target=pub, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sub.qsize() == 2000
    seen = {}
    while (ev := sub.get_nowait()) is not None:
        n = ev.data["n"]
        assert ev.data["i"] == seen.get(n, -1) + 1  # per-publisher order preserved
        seen[n] = ev.data["i"]
