"""POST /v1/models/<id>/load and /unload, the load mark that survives the idle timer, and the state field."""

import pytest

from shutter.core.engine import LoadRefused, UnloadRefused, is_load
from shutter.core.memory import GB, FakeProbe
from shutter.types import JobState

from tests.test_engine import gated, make_engine, req, wait_until  # noqa: F401  (fixture re-export)


def row(eng, mid="dummy"):
    return next(r for r in eng.models() if r["id"] == mid)


def test_models_rows_have_state_not_loaded(tmp_path):
    eng = make_engine(tmp_path, start=False)
    r = row(eng)
    assert r["state"] == "not_loaded" and r["loaded_gb"] is None and r["waiting"] is None


def test_load_goes_through_the_queue_and_marks_the_model_loaded(tmp_path):
    eng = make_engine(tmp_path)
    try:
        job = eng.load_model("dummy")
        assert is_load(job.request)
        # a second press while it is pending returns the same request
        assert eng.load_model("dummy").id == job.id or eng.get(job.id).state == JobState.DONE
        wait_until(lambda: eng.get(job.id).state == JobState.DONE)
        r = row(eng)
        assert r["state"] == "loaded" and r["held"] is True
        assert r["loaded_gb"] == pytest.approx(0.5)
        done = eng.get(job.id)
        assert "output" not in done.result and done.result["loaded_keys"] == ["dummy/dit"]
    finally:
        eng.stop(timeout=5)


def test_loaded_model_survives_the_idle_timer_until_unloaded(tmp_path):
    eng = make_engine(tmp_path)
    try:
        job = eng.load_model("dummy")
        wait_until(lambda: eng.get(job.id).state == JobState.DONE)
        assert eng.pool.evict_idle(0) == []                 # held: the idle timer leaves it
        assert row(eng)["state"] == "loaded"
        released = eng.unload_model("dummy")
        assert released == ["dummy/dit"]
        assert row(eng)["state"] == "not_loaded"
        # a request's own warm component (not loaded on purpose) still goes on idle
        j = eng.submit(req())
        wait_until(lambda: eng.get(j.id).state == JobState.DONE)
        wait_until(lambda: "dummy/dit" in eng.pool)
        assert eng.pool.evict_idle(0) == ["dummy/dit"]
    finally:
        eng.stop(timeout=5)


def test_held_entry_can_still_be_evicted_for_another_model(tmp_path):
    eng = make_engine(tmp_path, start=False)
    eng.pool.get("dummy/dit", lambda: "w", int(0.5 * GB))
    eng.pool.hold("dummy/dit")
    victim = eng.pool._pick_victim(exclude="other/dit")
    assert victim == "dummy/dit"


def test_load_waits_for_memory_with_the_guard_sentence(tmp_path):
    probe = FakeProbe(total=64 * GB, available=GB // 2)
    eng = make_engine(tmp_path, probe=probe)
    try:
        job = eng.load_model("dummy", wait=True)
        wait_until(lambda: eng.get(job.id).state == JobState.WAITING_MEMORY)
        r = row(eng)
        assert r["state"] == "waiting_for_memory"
        assert "Not enough free memory" in r["waiting"] and "macOS can give" in r["waiting"]
        # unload cancels the waiting load
        eng.unload_model("dummy")
        assert eng.get(job.id).state == JobState.CANCELLED
        assert row(eng)["state"] == "not_loaded"
    finally:
        eng.stop(timeout=5)


def test_unload_is_refused_while_a_request_for_the_model_runs(tmp_path, gated):  # noqa: F811
    eng = make_engine(tmp_path)
    try:
        j = eng.submit(req("gated", steps=2))
        assert gated.entered.wait(5)
        with pytest.raises(UnloadRefused, match="running a request"):
            eng.unload_model("gated")
        gated.gate.set()
        wait_until(lambda: eng.get(j.id).state == JobState.DONE)
        eng.unload_model("gated")          # fine once it is done
    finally:
        gated.gate.set()
        eng.stop(timeout=5)


def test_http_routes(tmp_path, gated):  # noqa: F811
    from starlette.testclient import TestClient

    from shutter.server.api import create_app

    eng = make_engine(tmp_path)
    try:
        c = TestClient(create_app(eng))
        r = c.post("/v1/models/dummy/load", json={})
        assert r.status_code == 202 and r.json()["request"]["request"]["params"]["_action"] == "load"
        wait_until(lambda: row(eng)["state"] == "loaded")
        m = {x["id"]: x for x in c.get("/v1/models").json()["models"]}
        assert m["dummy"]["state"] == "loaded" and m["dummy"]["loaded_gb"] == pytest.approx(0.5)
        r = c.post("/v1/models/dummy/unload")
        assert r.status_code == 200 and r.json()["model"]["state"] == "not_loaded"
        assert c.post("/v1/models/nope/load").status_code == 404
        j = eng.submit(req("gated", steps=2))
        assert gated.entered.wait(5)
        r = c.post("/v1/models/gated/unload")
        assert r.status_code == 409 and "running a request" in r.json()["error"]
        gated.gate.set()
        wait_until(lambda: eng.get(j.id).state == JobState.DONE)
    finally:
        gated.gate.set()
        eng.stop(timeout=5)


def test_load_fails_fast_with_structured_numbers_when_memory_is_short(tmp_path):
    probe = FakeProbe(total=64 * GB, available=GB // 2)
    eng = make_engine(tmp_path, probe=probe)
    try:
        with pytest.raises(LoadRefused) as ei:
            eng.load_model("dummy")
        info = ei.value.info
        assert info["kind"] == "memory" and info["model_name"] == "Test model"
        assert info["peak_gb"] == pytest.approx(1.0) and info["headroom_gb"] == 0
        assert info["available_gb"] == pytest.approx(0.5) and info["free_gb"] == pytest.approx(0.5)
        assert info["short_by_gb"] == pytest.approx(0.5)
        assert info["lowest"] is False
        assert row(eng)["state"] == "not_loaded"           # nothing queued
        assert not eng.list()
    finally:
        eng.stop(timeout=5)


def test_pool_release_flips_the_model_back_to_not_loaded(tmp_path):
    eng = make_engine(tmp_path)
    try:
        job = eng.load_model("dummy")
        wait_until(lambda: eng.get(job.id).state == JobState.DONE)
        assert row(eng)["state"] == "loaded"
        eng.release("dummy/dit")                            # the In memory section's Unload
        assert row(eng)["state"] == "not_loaded" and row(eng)["held"] is False
        assert "dummy/dit" not in eng.pool
    finally:
        eng.stop(timeout=5)


def test_unload_route_empties_the_in_memory_entries_of_that_model(tmp_path):
    eng = make_engine(tmp_path)
    try:
        job = eng.load_model("dummy")
        wait_until(lambda: eng.get(job.id).state == JobState.DONE)
        assert any(e["key"] == "dummy/dit" for e in eng.memory()["components"])
        eng.unload_model("dummy")
        assert not any(c["model"] == "dummy" for c in eng.memory()["components"])
        assert eng.system()["pool"]["entries"] == []
    finally:
        eng.stop(timeout=5)


def test_failed_load_is_reported_on_the_row(tmp_path):
    eng = make_engine(tmp_path, start=False)
    from shutter.adapters import dummy as D

    def boom(self, job, ctx):
        raise RuntimeError("the weights are damaged")
    orig = D.DummyAdapter.warm
    D.DummyAdapter.warm = boom
    try:
        eng.start()
        job = eng.load_model("dummy")
        wait_until(lambda: eng.get(job.id).state == JobState.FAILED)
        r = row(eng)
        assert r["state"] == "not_loaded" and "the weights are damaged" in r["load_error"]
    finally:
        D.DummyAdapter.warm = orig
        eng.stop(timeout=5)
