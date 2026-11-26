"""Load / reuse / unload for Wan builds (POST /v1/models/<id>/load): the adapter's warm() loads the DiT the way the
chosen memory step keeps it, a request reuses it instead of loading again, a step that frees the DiT before the decode
keeps a loaded model only when there is room, and unloading a streamed model stops its SSD reader.

Tiny random builds on the CPU (tests/test_wan_pipeline.py helpers); the prompt encode and the decode are stubbed.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tests.test_wan_pipeline import (  # noqa: E402,F401  (fixtures re-exported)
    FRAMES, H, STEPS, W, LogPool, fixed_prompt, latents_out, make_build, make_ctx,
)
from shutter.adapters.wan import WanAdapter  # noqa: E402
from shutter.types import GB, Job, JobRequest  # noqa: E402


def _load_req(model="tiny", eff=None):
    p = {"_action": "load", "budget_bytes": 1000 * GB}
    if eff is not None:
        p.update(mode="low_memory", efficiency=eff)
    return JobRequest(model=model, params=p)


def _req(model="tiny", eff=None, out=None):
    p = {"budget_bytes": 1000 * GB}
    if eff is not None:
        p.update(mode="low_memory", efficiency=eff)
    return JobRequest(model=model, prompt="a red fox", width=W, height=H, frames=FRAMES, steps=STEPS, seed=3,
                      output=out, params=p)


def _adapter(build, tmp_path, mid="tiny"):
    opts = {"measured_path": str(tmp_path / "measured.json"), "stream_nocache": False,
            "default_size": [H, W], "default_frames": FRAMES, "default_steps": STEPS}
    return WanAdapter(mid, build, opts)


@pytest.fixture
def no_mp4(monkeypatch):
    from shutter.adapters import longcat

    def fake(frames, audio, out, fps=16, crf=14):
        from pathlib import Path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"mp4")
    monkeypatch.setattr(longcat, "write_mp4", fake)


def _hold_all(pool):
    for k in pool.keys():
        pool.hold(k)


def test_load_resident_then_request_reuses_it(tmp_path, fixed_prompt, latents_out, no_mp4):
    build = make_build(tmp_path, None)
    ad = _adapter(build, tmp_path)
    pool = LogPool()
    ctx, reports = make_ctx(tmp_path, pool)
    res = ad.warm(Job(request=_load_req()), ctx)
    assert res["rung"] == "resident" and res["residency"] == "resident"
    assert pool.loads() == ["wan/tiny/dit/resident"] and reports[-1] == (1.0, "loaded")
    _hold_all(pool)                                   # what the engine does after warm()
    out = ad.run(Job(request=_req(out=str(tmp_path / "a.mp4"))), ctx)
    assert out["info"]["expert_loads"] == 0 and out["info"]["reused_loaded"] == ["wan/tiny/dit/resident"]
    assert pool.loads() == ["wan/tiny/dit/resident"]  # no second load
    assert "wan/tiny/dit/resident" in pool and pool.stats()["entries"][0]["held"]


def test_load_then_run_gives_the_same_frames_as_a_cold_run(tmp_path, fixed_prompt, latents_out, monkeypatch):
    from shutter.adapters import longcat
    frames = {}

    def keep(f, audio, out, fps=16, crf=14):
        from pathlib import Path
        frames[Path(out).stem] = f
        Path(out).write_bytes(b"mp4")
    monkeypatch.setattr(longcat, "write_mp4", keep)
    build = make_build(tmp_path, None)
    ad = _adapter(build, tmp_path)
    cold, _ = make_ctx(tmp_path, LogPool())
    ad.run(Job(request=_req(out=str(tmp_path / "cold.mp4"))), cold)
    warm_pool = LogPool()
    ctx, _ = make_ctx(tmp_path, warm_pool)
    ad.warm(Job(request=_load_req()), ctx)
    _hold_all(warm_pool)
    ad.run(Job(request=_req(out=str(tmp_path / "warm.mp4"))), ctx)
    assert np.array_equal(frames["cold"], frames["warm"])


@pytest.mark.parametrize("eff,key", [(0.75, "wan/tiny/dit/stream8"), (1.0, "wan/tiny/dit/stream2")])
def test_load_streamed_keeps_the_reader_and_unload_stops_it(tmp_path, fixed_prompt, latents_out, no_mp4, eff, key):
    build = make_build(tmp_path, None)
    ad = _adapter(build, tmp_path)
    pool = LogPool()
    ctx, _ = make_ctx(tmp_path, pool)
    res = ad.warm(Job(request=_load_req(eff=eff)), ctx)
    assert res["residency"] == "streamed" and "SSD" in res["note"]
    assert pool.loads() == [key] and pool.stats()["entries"][0]["residency"] == "streamed"
    _hold_all(pool)
    out = ad.run(Job(request=_req(eff=eff, out=str(tmp_path / "s.mp4"))), ctx)
    # a streamed step frees the DiT before the decode, but a loaded model stays when there is room
    assert out["info"]["expert_loads"] == 0 and key in pool and pool.loads() == [key]
    pool.release(key)                                 # what Unload does
    deadline = time.time() + 5
    while time.time() < deadline and any(t.name == "shutter-stream" and t.is_alive() for t in threading.enumerate()):
        time.sleep(0.05)
    assert key not in pool
    assert not any(t.name == "shutter-stream" and t.is_alive() for t in threading.enumerate())


def test_staged_drops_a_loaded_model_when_the_decode_has_no_room(tmp_path, fixed_prompt, latents_out, no_mp4):
    build = make_build(tmp_path, None)
    ad = _adapter(build, tmp_path)
    pool = LogPool()
    ctx, _ = make_ctx(tmp_path, pool)
    ad.warm(Job(request=_load_req(eff=0.5)), ctx)
    _hold_all(pool)
    monkey = {"avail": 1}                              # macOS has (almost) nothing left for the decode
    ctx.memory_sample = lambda: {"available": monkey["avail"], "total": 64 * GB}
    r = _req(eff=0.5, out=str(tmp_path / "x.mp4"))
    r.params["budget_bytes"] = 1000 * GB
    out = ad.run(Job(request=r), ctx)
    assert pool.keys() == [] and any("unloaded before the decode" in n for n in out["info"]["notes"])


def test_dual_expert_load_at_a_one_expert_step_loads_the_high_noise_expert(tmp_path):
    build = make_build(tmp_path, None, dual=True, name="dual")
    ad = _adapter(build, tmp_path, "dual")
    pool = LogPool()
    ctx, _ = make_ctx(tmp_path, pool)
    res = ad.warm(Job(request=_load_req("dual", eff=0.25)), ctx)
    assert res["rung"] == "split" and pool.loads() == ["wan/dual/dit_high/resident"]
    assert "one of the two experts" in res["note"]
    pool2 = LogPool()
    ctx2, _ = make_ctx(tmp_path, pool2)
    ad.warm(Job(request=_load_req("dual")), ctx2)       # Fast, room for everything: both experts
    assert sorted(pool2.loads()) == ["wan/dual/dit_high/resident", "wan/dual/dit_low/resident"]


def test_load_refused_when_even_the_lowest_step_does_not_fit(tmp_path):
    from shutter.types import MemoryRefused
    build = make_build(tmp_path, None)
    ad = _adapter(build, tmp_path)
    pool = LogPool()
    ctx, _ = make_ctx(tmp_path, pool, available=1 << 20)
    r = JobRequest(model="tiny", params={"_action": "load"})
    with pytest.raises(MemoryRefused):
        ad.warm(Job(request=r), ctx)
    assert pool.log == []


def test_catalog_measured_runs_lower_the_estimate(tmp_path, monkeypatch):
    """The catalog's checked_on (measured on a 16 GB Mac after publishing) counts as a measured peak too."""
    from shutter.core import catalog as cat
    build = make_build(tmp_path, None)
    ad = _adapter(build, tmp_path, "tiny")
    req = _req()
    before = ad.rung_estimates(req)["resident"].peak
    monkeypatch.setitem(cat.CATALOG, "tiny", {"adapter": "wan", "checked_on": [
        {"size": f"{H}x{W}", "frames": FRAMES, "peaks_gb": {"resident": 0.001, "staged": 0.0005}}]})
    est = ad.rung_estimates(req)
    assert est["resident"].peak == int(0.001 * GB) < before and "measured" in est["resident"].note
    assert est["staged"].peak == int(0.0005 * GB)


# ----------------------------------------------------------------------------- through the engine


def _engine(tmp_path, build, probe=None, mid="tiny"):
    from shutter.core.config import Config
    from shutter.core.engine import Engine
    from shutter.core.memory import FakeProbe
    from shutter.core.registry import Registry
    cfg = Config(home=str(tmp_path / "home"), memory_poll_s=0.01, busy_process_pattern=None,
                 protected_process_pattern=None, omlx_url=None, dummy_model=False,
                 models={mid: {"adapter": "wan", "dir": str(build),
                                  "options": {"default_size": [H, W], "default_frames": FRAMES,
                                              "default_steps": STEPS, "stream_nocache": False,
                                              "measured_path": str(tmp_path / "m.json")}}})
    eng = Engine(cfg, probe=probe or FakeProbe(total=64 * GB, available=48 * GB), protected=[],
                 registry=Registry(cfg), watch_interval_s=0.0, sample_interval_s=0.01)
    eng.start()
    return eng


def _wait(fn, t=60):
    end = time.time() + t
    while time.time() < end:
        if fn():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def _row(eng):
    return next(r for r in eng.models() if r["id"] == "tiny")


def test_engine_load_run_unload(tmp_path, fixed_prompt, latents_out, no_mp4):
    from shutter.types import JobState
    build = make_build(tmp_path, None)
    eng = _engine(tmp_path, build)
    try:
        r = _row(eng)
        assert r["state"] == "not_loaded" and r["can_load"] is True and r["load_hint"] == ""
        job = eng.load_model("tiny")
        _wait(lambda: eng.get(job.id).state in (JobState.DONE, JobState.FAILED))
        assert eng.get(job.id).state == JobState.DONE, eng.get(job.id).message
        r = _row(eng)
        assert r["state"] == "loaded" and r["held"] and r["loaded_residency"] == "resident" and r["loaded_bytes"] > 0
        # run, then the model is still loaded and was not loaded twice
        j = eng.submit(_req())
        _wait(lambda: eng.get(j.id).state in (JobState.DONE, JobState.FAILED))
        done = eng.get(j.id)
        assert done.state == JobState.DONE, done.message
        assert done.result["info"]["expert_loads"] == 0 and _row(eng)["state"] == "loaded"
        assert eng.unload_model("tiny") == ["wan/tiny/dit/resident"]
        assert _row(eng)["state"] == "not_loaded"
        # run first, then Load: the request's warm copy becomes the loaded one (no second load)
        j = eng.submit(_req())
        _wait(lambda: eng.get(j.id).state == JobState.DONE)
        job = eng.load_model("tiny")
        _wait(lambda: eng.get(job.id).state == JobState.DONE)
        assert _row(eng)["held"] is True
    finally:
        eng.stop(timeout=10)


def test_engine_streamed_load_shows_streamed(tmp_path, fixed_prompt, latents_out, no_mp4):
    from shutter.types import JobState
    build = make_build(tmp_path, None)
    eng = _engine(tmp_path, build)
    try:
        eng.update_settings(mode="low_memory", efficiency=1.0)
        job = eng.load_model("tiny")
        _wait(lambda: eng.get(job.id).state in (JobState.DONE, JobState.FAILED))
        assert eng.get(job.id).state == JobState.DONE, eng.get(job.id).message
        assert _row(eng)["loaded_residency"] == "streamed"
        eng.unload_model("tiny")
        _wait(lambda: not any(t.name == "shutter-stream" and t.is_alive() for t in threading.enumerate()), 5)
    finally:
        eng.stop(timeout=10)


def test_engine_idle_critical_pressure_unloads_loaded_models(tmp_path, fixed_prompt, latents_out):
    from shutter.core.memory import FakeProbe
    from shutter.types import JobState
    build = make_build(tmp_path, None)
    probe = FakeProbe(total=64 * GB, available=48 * GB)
    eng = _engine(tmp_path, build, probe)
    try:
        job = eng.load_model("tiny")
        _wait(lambda: eng.get(job.id).state == JobState.DONE)
        assert _row(eng)["state"] == "loaded"
        probe.set(pressure="critical")
        assert eng._relieve_idle_pressure() == ["wan/tiny/dit/resident"]
        assert _row(eng)["state"] == "not_loaded"
    finally:
        eng.stop(timeout=10)


def test_engine_settings_ladder_shows_measured_peaks_of_an_installed_model(tmp_path, monkeypatch):
    from shutter.core import catalog as cat
    build = make_build(tmp_path, None)
    monkeypatch.setitem(cat.CATALOG, "tiny", {"adapter": "wan", "checked_on": [
        {"size": f"{H}x{W}", "frames": FRAMES, "identical": True, "peaks_gb": {"resident": 6.23, "staged": 4.61},
         "seconds": {"resident": 166, "staged": 163}}]})
    eng = _engine(tmp_path, build)
    try:
        st = eng.settings()
        assert st["measured_for"] == {"model": "tiny", "size": f"{H}x{W}, {FRAMES} frames"}
        rows = {r["id"]: r for r in st["ladder"]}
        assert rows["resident"]["peak_bytes"] == int(6.23 * GB) and rows["staged"]["seconds"] == 163.0
        assert "peak_bytes" not in rows["stream_tight"]
    finally:
        eng.stop(timeout=10)


def _as_q8_encoder_layout(build):
    """Rename text_encoder/ to text_encoder_q8/ the way the 0.2.6 catalog revision ships it."""
    import json
    (build / "text_encoder").rename(build / "text_encoder_q8")
    man = json.loads((build / "shuttrr.json").read_text())
    te = man["parts"]["text_encoder"]
    te["files"] = [f.replace("text_encoder/", "text_encoder_q8/", 1) for f in te.get("files", [])]
    (build / "shuttrr.json").write_text(json.dumps(man))


@pytest.mark.parametrize("layout", ["bf16_encoder_709d9c4f", "q8_encoder_a3029232"])
def test_catalog_wan13_id_loads_with_either_encoder_revision(tmp_path, fixed_prompt, latents_out, no_mp4, layout):
    """Load works for the catalog id of Wan 2.1 1.3B whichever revision is on disk: the earlier one with the bf16
    text encoder (text_encoder/) and the 0.2.6 one (text_encoder_q8/). Load only touches the DiT, so the encoder
    folder must not matter; the row says Loaded with measured bytes and Unload frees it."""
    from shutter.types import JobState
    mid = "wan2.1-t2v-1.3b-8bit"
    build = make_build(tmp_path, None)
    if layout.startswith("q8"):
        _as_q8_encoder_layout(build)
    eng = _engine(tmp_path, build, mid=mid)
    try:
        row = next(r for r in eng.models() if r["id"] == mid)
        assert row["can_load"] is True and row["load_hint"] == ""
        job = eng.load_model(mid)
        _wait(lambda: eng.get(job.id).state in (JobState.DONE, JobState.FAILED))
        assert eng.get(job.id).state == JobState.DONE, eng.get(job.id).message
        row = next(r for r in eng.models() if r["id"] == mid)
        assert row["state"] == "loaded" and row["held"] and row["loaded_bytes"] > 0
        assert not row.get("load_error")
        assert eng.unload_model(mid)
        assert next(r for r in eng.models() if r["id"] == mid)["state"] == "not_loaded"
    finally:
        eng.stop(timeout=10)


def test_load_error_text_is_never_a_python_exception(tmp_path, fixed_prompt, latents_out, monkeypatch):
    """A failure inside warm() reaches the app as a plain sentence, not "RuntimeError: ..."."""
    from shutter.types import JobState
    build = make_build(tmp_path, None)
    eng = _engine(tmp_path, build)

    def boom(self, job, ctx):
        raise RuntimeError("tiny cannot be loaded ahead of a request; it loads when a request needs it")
    monkeypatch.setattr(WanAdapter, "warm", boom)
    try:
        job = eng.load_model("tiny")
        _wait(lambda: eng.get(job.id).state in (JobState.DONE, JobState.FAILED))
        j = eng.get(job.id)
        assert j.state == JobState.FAILED
        row = _row(eng)
        for text in (j.message or "", row.get("load_error") or ""):
            assert "RuntimeError" not in text and "Traceback" not in text
    finally:
        eng.stop(timeout=10)
