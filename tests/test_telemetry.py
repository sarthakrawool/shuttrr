"""Help The Dev Improve: off by default, nothing collected or sent while off, scrubbed issues, weekly usage, the
crash marker, and the API the app uses."""

from __future__ import annotations

import time

import pytest
from starlette.testclient import TestClient

from shutter.core import telemetry as T
from shutter.core.config import Config
from shutter.core.engine import Engine
from shutter.core.memory import FakeProbe
from shutter.server.api import create_app


class Sink:
    def __init__(self, ok=True):
        self.sent = []
        self.ok = ok

    def __call__(self, url, payload):
        self.sent.append((url, payload))
        if self.ok is None:
            raise OSError("network down")
        return self.ok


def _tel(tmp_path, sink, clock=None, enabled=False):
    cfg = Config(home=str(tmp_path / "home"))
    cfg.save()
    cfg.telemetry_enabled = enabled
    return T.Telemetry(cfg, engine_version="0.3.3", post=sink, clock=clock or time.time)


def test_off_by_default_and_nothing_is_collected_or_sent(tmp_path):
    sink = Sink()
    t = _tel(tmp_path, sink)
    assert Config().telemetry_enabled is False and t.enabled is False
    t.record_job("z-image-turbo-8bit", True)
    assert t.record_issue("request_failed", "ValueError", "boom") is False
    t.engine_started()
    assert t.send_now(force=True)["sent"] is False
    assert sink.sent == [] and t.state.get("usage", {}) == {} and not t.state.get("issues")


def test_updated_install_without_the_setting_sends_nothing(tmp_path):
    # a config.json written by an older Shuttrr has no telemetry keys at all: it must load as off and not asked
    import json
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"port": 8484, "models": {}}))
    cfg = Config.load(str(home / "config.json"))
    assert cfg.telemetry_enabled is False and cfg.telemetry_asked is False
    sink = Sink()
    t = T.Telemetry(cfg, engine_version="0.3.3", post=sink)
    t.record_job("z-image-turbo-8bit", True)
    t.record_issue("request_failed", "ValueError", "boom")
    t.engine_started()
    assert t.send_now(force=True)["sent"] is False and sink.sent == []


def test_usage_is_weekly_and_custom_models_are_anonymous(tmp_path):
    now = [1_000_000.0]
    sink = Sink()
    t = _tel(tmp_path, sink, clock=lambda: now[0], enabled=True)
    for done in (True, True, False):
        t.record_job("z-image-turbo-8bit", done)
    t.record_job("my-private-finetune", True)
    assert t.send_now()["sent"] is False             # less than a week of data: not due yet
    now[0] += T.WEEK_S + 1
    r = t.send_now()
    assert r["sent"] and len(sink.sent) == 1
    p = sink.sent[0][1]
    assert p["usage"] == {"z-image-turbo-8bit": {"requests": 3, "finished": 2}, "custom": {"requests": 1,
                                                                                         "finished": 1}}
    assert set(p) >= {"schema", "engine_version", "macos", "chip", "memory_gb"}
    assert not any(k in p for k in ("id", "install_id", "user", "hostname", "serial"))
    assert t.status()["last_payload"] == p and t.state["usage"] == {}


def test_issues_are_scrubbed_and_deduplicated(tmp_path):
    sink = Sink()
    t = _tel(tmp_path, sink, enabled=True)
    home = str(T.Path.home())
    trace = (f'Traceback:\n  File "{home}/.shutter/engine/lib/python3.11/site-packages/shutter/core/engine.py", '
             f'line 10, in run\n  File "{home}/Desktop/cat photo.png", line 1\nOSError: {home}/secret/x.safetensors')
    for _ in range(2):
        t.record_issue("request_failed", "OSError", f'cannot open "{home}/Pictures/holiday at the beach 2026.png" '
                       "for a long prompt about my family", trace, model_id="minimax-h3-image-8bit")
    assert len(t.state["issues"]) == 1 and t.state["issues"][0]["count"] == 2
    t.send_now(force=True)
    sent = str(sink.sent[0][1])
    assert home not in sent and "holiday" not in sent and "cat photo" not in sent and "secret" not in sent
    assert "shutter/core/engine.py" in sent and "minimax-h3-image-8bit" in sent


def test_network_failure_keeps_data_and_never_raises(tmp_path):
    sink = Sink(ok=None)
    t = _tel(tmp_path, sink, enabled=True)
    t.record_issue("request_failed", "ValueError", "boom")
    assert t.send_now(force=True)["sent"] is False
    assert len(t.state["issues"]) == 1 and t.status()["last_payload"] is None


def test_unclean_exit_is_reported_once_when_on(tmp_path):
    sink = Sink()
    t = _tel(tmp_path, sink, enabled=True)
    t.engine_started()
    t2 = _tel(tmp_path, sink, enabled=True)            # the next start finds the marker: last run did not stop
    t2.engine_started()
    assert [i["kind"] for i in t2.state["issues"]] == ["engine_unexpected_exit"]
    t2.engine_stopped()
    t3 = _tel(tmp_path, sink, enabled=True)
    t3.engine_started()
    assert [i["kind"] for i in t3.state["issues"]] == ["engine_unexpected_exit"]   # no new one after a clean stop


def test_turning_off_forgets_unsent_data(tmp_path):
    t = _tel(tmp_path, Sink(), enabled=True)
    t.record_job("z-image-turbo-8bit", True)
    t.record_issue("request_failed", "E", "m")
    out = t.set(enabled=False)
    assert out["enabled"] is False and out["asked"] is True
    assert t.state["usage"] == {} and t.state["issues"] == []
    assert Config.load(t.config.path).telemetry_enabled is False


@pytest.fixture
def engine(tmp_path):
    cfg = Config(home=str(tmp_path / "home"), memory_poll_s=0.01, busy_process_pattern=None)
    eng = Engine(cfg, probe=FakeProbe(), protected=[], watch_interval_s=0, sample_interval_s=0.01)
    eng.telemetry._post = Sink()
    eng.start()
    yield eng
    eng.stop(timeout=5)


def _wait(pred, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    raise AssertionError("timeout")


def test_api_and_engine_counting(engine):
    c = TestClient(create_app(engine))
    s = c.get("/v1/telemetry").json()
    assert s["enabled"] is False and s["asked"] is False and s["what"] and s["never"]
    # off: a finished request is not counted, an app crash is not kept
    jid = c.post("/v1/jobs", json={"model": "dummy", "prompt": "x", "steps": 1}).json()["id"]
    _wait(lambda: c.get(f"/v1/jobs/{jid}").json()["state"] == "done")
    assert c.post("/v1/telemetry/app-crash", json={"exception": "EXC_BAD_ACCESS", "frames": ["a"]}).json() == \
        {"queued": False}
    assert engine.telemetry.state.get("usage", {}) == {} and engine.telemetry._post.sent == []
    assert c.post("/v1/telemetry", json={"asked": True}).json()["asked"] is True
    # on: counted, app crash kept, send now posts one payload with both
    assert c.post("/v1/telemetry", json={"enabled": True}).json()["enabled"] is True
    jid = c.post("/v1/jobs", json={"model": "dummy", "prompt": "x", "steps": 1}).json()["id"]
    _wait(lambda: c.get(f"/v1/jobs/{jid}").json()["state"] == "done")
    _wait(lambda: engine.telemetry.state.get("usage"))
    assert c.post("/v1/telemetry/app-crash", json={"exception": "EXC_BAD_ACCESS", "signal": "SIGSEGV",
                                                   "frames": ["EngineStore.refresh"]}).json() == {"queued": True}
    out = c.post("/v1/telemetry", json={"send_now": True}).json()
    assert out["send"]["sent"] is True
    p = engine.telemetry._post.sent[-1][1]
    assert p["usage"]["custom"]["requests"] == 1 or "dummy" in p["usage"]
    assert p["issues"][0]["kind"] == "app_crash"
    assert c.get("/v1/telemetry").json()["last_payload"] == p
    assert c.post("/v1/telemetry", json={"enabled": "yes"}).status_code == 400
