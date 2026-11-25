import json
import os
import threading
import time

import pytest

from shutter.adapters.dummy import DummyAdapter
from shutter.core.config import Config
from shutter.core.dirlock import DirLocks
from shutter.core.engine import PROTECT_PREFIX, Engine, SubmitError
from shutter.core.memory import GB, FakeProbe, GpuLock
from shutter.core.queue import JobStore
from shutter.core.registry import Registry
from shutter.types import Job, JobRequest, JobState


def wait_until(pred, timeout=5.0, step=0.005):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(step)
    raise AssertionError("condition not reached in time")


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class GatedAdapter(DummyAdapter):
    """Dummy that blocks on each step until the test lets it go."""

    family = "gated"
    gate = None      # threading.Event
    entered = None   # threading.Event
    step_hook = None

    def on_step(self, job, step, ctx):
        type(self).entered.set()
        if type(self).step_hook:
            type(self).step_hook(job, step, ctx)
        while not type(self).gate.wait(0.005):
            ctx.check_cancelled()


class Busy:
    name = "oMLX"

    def __init__(self):
        self.reason = None

    def busy(self):
        return self.reason


def make_config(tmp_path, **kw):
    kw.setdefault("memory_poll_s", 0.01)
    kw.setdefault("busy_process_pattern", None)
    kw.setdefault("models", {"gated": {"adapter": "gated"}})
    return Config(home=str(tmp_path / "home"), **kw)


@pytest.fixture
def gated():
    GatedAdapter.gate = threading.Event()
    GatedAdapter.entered = threading.Event()
    GatedAdapter.step_hook = None
    yield GatedAdapter
    GatedAdapter.gate.set()


def make_engine(tmp_path, probe=None, protected=(), start=True, **kw):
    engine_kw = {k: kw.pop(k) for k in list(kw) if k in ("clock", "ps", "idle_check_s")}
    cfg = make_config(tmp_path, **kw)
    reg = Registry(cfg, families={"gated": GatedAdapter})
    eng = Engine(cfg, probe=probe or FakeProbe(), protected=list(protected), registry=reg,
                 watch_interval_s=0.0, sample_interval_s=0.01, **engine_kw)
    if start:
        eng.start()
    return eng


def req(model="dummy", **params):
    params.setdefault("step_s", 0)
    params.setdefault("steps", 3)
    return JobRequest(model=model, params=params)


@pytest.fixture
def engines():
    made = []
    yield made
    for e in made:
        e.stop(timeout=5)


def test_job_runs_to_done_with_progress_events(tmp_path, engines):
    eng = make_engine(tmp_path)
    engines.append(eng)
    sub = eng.bus.subscribe()
    job = eng.submit(req(steps=4, prompt="hi"))
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)
    j = eng.get(job.id)
    assert j.progress == 1.0
    assert os.path.exists(j.result["output"])
    for k in ("output", "peak_bytes", "seconds", "peak_memory"):
        assert k in j.result
    assert isinstance(j.result["peak_bytes"], int)
    kinds = []
    while (ev := sub.get(timeout=0.2)) is not None:
        kinds.append(ev)
    names = [e.kind for e in kinds]
    assert names[0] == "job.queued"
    assert names.index("job.started") < names.index("job.done")
    progress = [e for e in kinds if e.kind == "job.progress"]
    assert progress and progress[-1].data["progress"] == 1.0
    assert "stage" in progress[-1].data and "eta_s" in progress[-1].data
    assert "model.loaded" in names
    assert eng.worker_errors == []


def test_priority_order(tmp_path, engines, gated):
    eng = make_engine(tmp_path, start=False)
    engines.append(eng)
    low = eng.submit(JobRequest(model="dummy", priority=0, params={"steps": 1, "step_s": 0}))
    low2 = eng.submit(JobRequest(model="dummy", priority=0, params={"steps": 1, "step_s": 0}))
    high = eng.submit(JobRequest(model="dummy", priority=5, params={"steps": 1, "step_s": 0}))
    order = []
    eng.bus_sub = eng.bus.subscribe()
    eng.start()
    wait_until(lambda: all(eng.get(j.id).state == JobState.DONE for j in (low, low2, high)))
    while (ev := eng.bus_sub.get(timeout=0.1)) is not None:
        if ev.kind == "job.started":
            order.append(ev.job_id)
    assert order == [high.id, low.id, low2.id]


def test_cancel_queued_and_running(tmp_path, engines, gated):
    eng = make_engine(tmp_path)
    engines.append(eng)
    running = eng.submit(req("gated"))
    assert gated.entered.wait(5)
    queued = eng.submit(req())
    c = eng.cancel(queued.id)
    assert c.state == JobState.CANCELLED
    assert eng.get(running.id).state == JobState.RUNNING
    eng.cancel(running.id)
    wait_until(lambda: eng.get(running.id).state == JobState.CANCELLED)
    assert eng.get(queued.id).state == JobState.CANCELLED
    assert eng.cancel("nope") is None
    # GPU lock released after cancel
    assert not eng.gpu_lock.held


def test_memory_refusal_waits_then_runs(tmp_path, engines):
    probe = FakeProbe(available=5 * GB)
    eng = make_engine(tmp_path, probe=probe)
    engines.append(eng)
    sub = eng.bus.subscribe()
    job = eng.submit(req(peak_gb=4))
    wait_until(lambda: eng.get(job.id).state == JobState.WAITING_MEMORY)
    assert "Not enough free memory" in eng.get(job.id).message
    time.sleep(0.05)  # several polls with the same reason
    probe.set(available=40 * GB)
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)
    waits = []
    while (ev := sub.get(timeout=0.1)) is not None:
        if ev.kind == "job.waiting_memory":
            waits.append(ev)
    assert len(waits) == 1  # published once per changed reason


def test_higher_priority_overtakes_waiting_job(tmp_path, engines):
    probe = FakeProbe(available=12 * GB)
    eng = make_engine(tmp_path, probe=probe)
    engines.append(eng)
    big = eng.submit(req(peak_gb=30))
    wait_until(lambda: eng.get(big.id).state == JobState.WAITING_MEMORY)
    small = eng.submit(JobRequest(model="dummy", priority=1, params={"peak_gb": 1, "steps": 1, "step_s": 0}))
    wait_until(lambda: eng.get(small.id).state == JobState.DONE)
    assert eng.get(big.id).state == JobState.WAITING_MEMORY


def test_refusal_frees_unneeded_pool_entries(tmp_path, engines):
    probe = FakeProbe(available=12 * GB)
    eng = make_engine(tmp_path, probe=probe)
    engines.append(eng)
    eng.pool.get("other-model/dit", lambda: object(), 1 * GB)

    def free_on_release(key):
        probe.set(available=40 * GB)

    orig = eng.pool.release

    def release(key):
        orig(key)
        free_on_release(key)

    eng.pool.release = release
    job = eng.submit(req(peak_gb=10))
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)
    assert "other-model/dit" not in eng.pool.keys()


def test_protected_service_blocks_and_unblocks(tmp_path, engines):
    svc = Busy()
    svc.reason = "oMLX has a model loaded (flash)"
    eng = make_engine(tmp_path, protected=[svc])
    engines.append(eng)
    job = eng.submit(req())
    wait_until(lambda: eng.get(job.id).state == JobState.WAITING_MEMORY)
    assert eng.get(job.id).message == svc.reason
    assert eng.system()["guard"]["protected"] == [{"name": "oMLX", "busy": svc.reason}]
    svc.reason = None
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)


def test_swap_growth_aborts_and_releases_pool(tmp_path, engines, gated):
    probe = FakeProbe()
    eng = make_engine(tmp_path, probe=probe)
    engines.append(eng)
    eng.pool.get("other/x", lambda: object(), GB)

    def hook(job, step, ctx):
        probe.set(swap_used=2 * GB)
    gated.step_hook = hook
    job = eng.submit(req("gated"))
    wait_until(lambda: eng.get(job.id).state == JobState.FAILED)
    j = eng.get(job.id)
    assert j.message.startswith(PROTECT_PREFIX) and "Swap grew" in j.message
    assert eng.pool.keys() == []
    assert not eng.gpu_lock.held


def test_gpu_lock_held_by_other_process_waits(tmp_path, engines):
    cfg_lock = str(tmp_path / "home" / "gpu.lock")
    other = GpuLock(cfg_lock, label="other-tool")
    assert other.try_acquire(0)
    try:
        eng = make_engine(tmp_path)
        engines.append(eng)
        job = eng.submit(req())
        wait_until(lambda: eng.get(job.id).state == JobState.WAITING_MEMORY)
        assert eng.get(job.id).message == "GPU busy: other-tool"
        sysinfo = eng.system()["gpu_lock"]
        assert sysinfo["held"] and sysinfo["holder"]["label"] == "other-tool"
    finally:
        other.release()
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)


def test_dir_locks_taken_and_released(tmp_path, engines, gated):
    d1, d2 = str(tmp_path / "studio.lock.d"), str(tmp_path / "q" / ".render.lock")
    eng = make_engine(tmp_path, gpu_lock_dirs=[d1, d2])
    engines.append(eng)
    job = eng.submit(req("gated"))
    assert gated.entered.wait(5)
    assert os.path.isdir(d1) and os.path.isdir(d2)
    assert open(os.path.join(d1, "owner")).read().strip() == "shutter"
    gated.gate.set()
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)
    assert not os.path.exists(d1) and not os.path.exists(d2)


def test_dir_lock_held_by_other_tool_waits(tmp_path, engines):
    d1 = str(tmp_path / "studio.lock.d")
    os.mkdir(d1)
    with open(os.path.join(d1, "owner"), "w") as f:
        f.write("fish\n")
    eng = make_engine(tmp_path, gpu_lock_dirs=[d1])
    engines.append(eng)
    job = eng.submit(req())
    wait_until(lambda: eng.get(job.id).state == JobState.WAITING_MEMORY)
    assert eng.get(job.id).message.startswith("GPU busy:") and "fish" in eng.get(job.id).message
    assert not eng.gpu_lock.held  # flock given back while waiting
    os.rmdir(d1) if not os.listdir(d1) else (os.remove(os.path.join(d1, "owner")), os.rmdir(d1))
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)


def test_busy_render_process_waits(tmp_path, engines):
    state = {"ps": "  101 2048 /usr/bin/python3 h3_generate.py --x\n"}
    eng = make_engine(tmp_path, busy_process_pattern="h3_generate.py|sd-cli",
                      ps=lambda: state["ps"])
    engines.append(eng)
    job = eng.submit(req())
    wait_until(lambda: eng.get(job.id).state == JobState.WAITING_MEMORY)
    assert eng.get(job.id).message == "another render is running"
    state["ps"] = "  101 2048 /bin/zsh\n"
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)


def test_adapter_error_fails_job_but_worker_survives(tmp_path, engines):
    eng = make_engine(tmp_path)
    engines.append(eng)
    bad = eng.submit(req(fail="boom"))
    good = eng.submit(req())
    wait_until(lambda: eng.get(good.id).state == JobState.DONE)
    b = eng.get(bad.id)
    assert b.state == JobState.FAILED and "boom" in b.message
    assert not eng.gpu_lock.held


def test_crash_recovery_requeues_running(tmp_path, engines):
    cfg = make_config(tmp_path)
    cfg.ensure_dirs()
    store = JobStore(cfg.db_path)
    job = Job(request=req(), state=JobState.RUNNING, started_at=time.time())
    store.add(job)
    store.close()
    eng = make_engine(tmp_path, start=False)
    engines.append(eng)
    j = eng.get(job.id)
    assert j.state == JobState.QUEUED and j.message == "requeued after restart"
    eng.start()
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)


def test_idle_unload(tmp_path, engines):
    clock = FakeClock()
    eng = make_engine(tmp_path, clock=clock, idle_unload_s=100, idle_check_s=0.01)
    engines.append(eng)
    job = eng.submit(req())
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)
    assert eng.pool.keys() == ["dummy/dit"]
    clock.t += 50
    time.sleep(0.05)
    assert eng.pool.keys() == ["dummy/dit"]
    clock.t += 60
    wait_until(lambda: eng.pool.keys() == [])


def test_idle_unload_zero_unloads_at_once(tmp_path, engines):
    eng = make_engine(tmp_path, idle_unload_s=0)
    engines.append(eng)
    job = eng.submit(req())
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)
    wait_until(lambda: eng.pool.keys() == [])


def test_warm_model_not_double_counted(tmp_path, engines):
    probe = FakeProbe(available=40 * GB)
    eng = make_engine(tmp_path, probe=probe)
    engines.append(eng)
    eng.submit(req(peak_gb=30, component_gb=20))
    wait_until(lambda: eng.store.counts()["done"] == 1)
    # the 20 GB component is warm and now counted as used memory
    probe.set(available=20 * GB)
    j2 = eng.submit(req(peak_gb=30, component_gb=20))
    wait_until(lambda: eng.get(j2.id).state == JobState.DONE)


def test_submit_validation(tmp_path, engines):
    eng = make_engine(tmp_path, start=False)
    engines.append(eng)
    with pytest.raises(SubmitError, match="unknown model"):
        eng.submit(JobRequest(model="nope"))
    with pytest.raises(SubmitError, match="image file not found"):
        eng.submit(JobRequest(model="dummy", image=str(tmp_path / "missing.png")))
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    j = eng.submit(JobRequest(model="dummy", image=str(img)))
    assert j.request.image == str(img)


def test_pause_resume_and_release(tmp_path, engines):
    eng = make_engine(tmp_path)
    engines.append(eng)
    eng.pause()
    job = eng.submit(req())
    time.sleep(0.05)
    assert eng.get(job.id).state == JobState.QUEUED
    assert eng.system()["queue"]["paused"] is True
    eng.resume()
    wait_until(lambda: eng.get(job.id).state == JobState.DONE)
    assert eng.release() == ["dummy/dit"]
    assert eng.release("missing") == []


def test_stop_requeues_running_job(tmp_path, gated):
    eng = make_engine(tmp_path)
    job = eng.submit(req("gated"))
    assert gated.entered.wait(5)
    eng.stop(timeout=5)
    assert eng.get(job.id).state == JobState.QUEUED


def test_broken_adapter_module_does_not_break_registry(tmp_path):
    cfg = make_config(tmp_path, models={"x": {"adapter": "broken"}, "lc": {"adapter": "longcat-avatar"}})
    reg = Registry(cfg, families={"broken": "shutter.adapters.does_not_exist:Nope"})
    rows = {r["id"]: r for r in reg.list()}
    assert rows["x"]["error"] and "ModuleNotFoundError" in rows["x"]["error"]
    assert rows["dummy"]["installed"] and rows["dummy"]["error"] is None
    assert "lc" in rows
    eng = Engine(cfg, probe=FakeProbe(), protected=[], registry=reg)
    try:
        eng.start()
        j = eng.submit(JobRequest(model="x"))
        wait_until(lambda: eng.get(j.id).state == JobState.FAILED)
        assert "cannot be loaded" in eng.get(j.id).message
    finally:
        eng.stop()


def test_config_roundtrip_keeps_unknown_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("SHUTTER_HOME", str(tmp_path / "h"))
    path = tmp_path / "h" / "config.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"port": 9000, "future_key": {"a": 1}, "models": {"m": {"adapter": "dummy"}}}))
    cfg = Config.load()
    assert cfg.port == 9000 and cfg.host == "127.0.0.1"
    assert cfg.db_path == str(tmp_path / "h" / "jobs.db")
    assert cfg.gpu_lock_dirs == [] and cfg.idle_unload_s == 900
    cfg.save()
    data = json.loads(path.read_text())
    assert data["future_key"] == {"a": 1} and data["port"] == 9000
    assert "db_path" not in data


def test_pool_key_ownership(tmp_path, engines):
    eng = make_engine(tmp_path, start=False)
    engines.append(eng)
    keys, prefix = eng._needed_keys(eng.registry.adapter_for("dummy"), req())
    assert eng._is_needed("dummy/dit", keys, prefix)
    assert eng._is_needed("longcat/dummy/generator", [], prefix)
    assert not eng._is_needed("other/dit", [], prefix)
