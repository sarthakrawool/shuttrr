"""Shuttrr 0.2.11: time left from measured steps (core/eta.py), cancel points inside the DiT and the decode
(families/wan/model/interrupt.py), and the engine's cancel flags."""

from __future__ import annotations

import time

import pytest

from shutter.core.eta import EtaTracker, decode_estimate_from_history
from shutter.core.stats import parse_step


def test_parse_step_reads_the_wan_stage_text():
    assert parse_step("denoising, step 3 of 20") == (3, 20)
    assert parse_step("denoise 3/8") == (3, 8)


def test_no_number_before_the_first_step():
    e = EtaTracker(decode_estimate=18.0)
    e.update("encoding the prompt", 0.0)
    assert e.estimate(3.0)["eta_s"] is None


def test_steps_plus_decode_and_countdown():
    e = EtaTracker(decode_estimate=18.0)
    e.update("encoding the prompt", 0.0)
    e.update("denoising, step 1 of 20", 6.0)          # first step 6 s (includes the load)
    assert e.estimate(6.0)["eta_s"] == pytest.approx(19 * 6.0 + 18.0)
    e.update("denoising, step 2 of 20", 11.0)         # step-to-step: 5 s
    assert e.estimate(11.0)["eta_s"] == pytest.approx(18 * 5.0 + 18.0)
    # counts down inside the step in progress
    assert e.estimate(13.0)["eta_s"] == pytest.approx(18 * 5.0 + 18.0 - 2.0)
    e.update("decoding", 101.0)
    assert e.estimate(111.0)["eta_s"] == pytest.approx(8.0)
    assert e.estimate(130.0)["eta_s"] == 0.0


def test_unknown_decode_is_flagged():
    e = EtaTracker(decode_estimate=None)
    e.update("x", 0.0)
    e.update("denoising, step 1 of 4", 2.0)
    r = e.estimate(2.0)
    assert r["eta_s"] == pytest.approx(6.0) and r["decode_known"] is False
    e.update("decoding", 8.0)
    assert e.estimate(9.0)["eta_s"] is None


def test_decode_history_matches_the_same_request():
    from shutter.types import Job, JobRequest, JobState

    def done(w, decode, model="m"):
        j = Job(request=JobRequest(model=model, width=w, height=320, frames=33))
        j.state = JobState.DONE
        j.result = {"info": {"timings": {"decode_s": decode}}}
        return j
    jobs = [done(512, 20.0), done(512, 18.0), done(832, 60.0), done(512, 19.0), done(512, 99.0, model="other")]
    want = JobRequest(model="m", width=512, height=320, frames=33)
    assert decode_estimate_from_history(jobs, "m", want) == 19.0
    assert decode_estimate_from_history(jobs, "m", JobRequest(model="m", width=100, height=100, frames=9)) is None


def test_interrupt_poll_raises_through_the_check():
    from shutter.families.wan.model import interrupt

    class Stop(Exception):
        pass

    def boom():
        raise Stop()
    interrupt.set_check(boom)
    try:
        with pytest.raises(Stop):
            interrupt.poll()
    finally:
        interrupt.set_check(None)
    interrupt.poll()      # no check set: nothing happens


def test_job_round_trips_the_new_fields():
    from shutter.types import Job, JobRequest
    j = Job(request=JobRequest(model="m"), eta_s=12.5, eta_decode_known=True, expected_s=90.0,
            cancel_requested=True)
    k = Job.from_dict(j.to_dict())
    assert (k.eta_s, k.eta_decode_known, k.expected_s, k.cancel_requested) == (12.5, True, 90.0, True)
    old = Job.from_dict({k2: v for k2, v in j.to_dict().items()
                         if k2 not in ("eta_s", "eta_decode_known", "expected_s", "cancel_requested")})
    assert old.eta_s is None and old.cancel_requested is False


def test_engine_cancel_marks_cancelling_then_cancelled_not_failed(tmp_path):
    from shutter.core.config import Config
    from shutter.core.engine import Engine
    from shutter.core.memory import FakeProbe
    from shutter.core.registry import Registry
    from shutter.types import GB, JobRequest, JobState
    cfg = Config(home=str(tmp_path / "h"), memory_poll_s=0.01, busy_process_pattern=None,
                 protected_process_pattern=None, omlx_url=None, dummy_model=True)
    eng = Engine(cfg, probe=FakeProbe(total=64 * GB, available=48 * GB), protected=[], registry=Registry(cfg),
                 watch_interval_s=0.0, sample_interval_s=0.01)
    eng.start()
    try:
        run = eng.submit(JobRequest(model="dummy", prompt="x", params={"steps": 400, "step_s": 0.02}))
        queued = eng.submit(JobRequest(model="dummy", prompt="y", params={"steps": 5}))
        end = time.time() + 10
        while eng.get(run.id).state != JobState.RUNNING and time.time() < end:
            time.sleep(0.01)
        while eng.get(run.id).eta_s is None and time.time() < end:     # measured after a couple of steps
            time.sleep(0.02)
        assert eng.get(run.id).eta_s is not None
        c = eng.cancel(run.id)
        assert c.cancel_requested and c.eta_s is None
        q = eng.cancel(queued.id)
        assert q.state == JobState.CANCELLED                            # queued: at once
        t0 = time.time()
        while eng.get(run.id).state == JobState.RUNNING and time.time() - t0 < 5:
            time.sleep(0.01)
        assert eng.get(run.id).state == JobState.CANCELLED
        assert time.time() - t0 < 1.0
    finally:
        eng.stop(timeout=10)
