import json
import time

import pytest
from starlette.testclient import TestClient

from shutter.core.config import Config
from shutter.core.engine import Engine
from shutter.core.memory import GB, FakeProbe
from shutter.server.api import check_bind, create_app, is_loopback


def wait_until(pred, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.005)
    raise AssertionError("timeout")


@pytest.fixture
def engine(tmp_path):
    cfg = Config(home=str(tmp_path / "home"), memory_poll_s=0.01, busy_process_pattern=None)
    eng = Engine(cfg, probe=FakeProbe(), protected=[], watch_interval_s=0, sample_interval_s=0.01)
    eng.start()
    yield eng
    eng.stop(timeout=5)


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine))


def test_system_shape(client):
    s = client.get("/v1/system").json()
    assert set(s) >= {"version", "memory", "guard", "pool", "queue", "gpu_lock"}
    assert set(s["memory"]) >= {"total", "available", "swap_used", "pressure"}
    assert s["guard"]["headroom_bytes"] == 0 and s["guard"]["protected"] == []
    assert set(s["pool"]) >= {"total_bytes", "entries"}
    assert set(s["queue"]) >= {"queued", "running", "waiting_memory", "done", "failed", "cancelled", "paused"}
    assert s["queue"]["paused"] is False
    assert set(s["gpu_lock"]) >= {"held", "holder"}


def test_models(client):
    models = client.get("/v1/models").json()["models"]
    ids = {m["id"]: m for m in models}
    assert ids["dummy"]["installed"] is True


def test_submit_get_list_and_pool(client, engine):
    r = client.post("/v1/jobs", json={"model": "dummy", "prompt": "x", "steps": 2,
                                      "params": {"step_s": 0}})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["request"]["steps"] == 2 and "steps" not in job["request"]["params"]  # a core field now
    wait_until(lambda: client.get(f"/v1/jobs/{job['id']}").json()["state"] == "done")
    j = client.get(f"/v1/jobs/{job['id']}").json()
    assert "output" in j["result"] and "peak_bytes" in j["result"] and "seconds" in j["result"]
    lst = client.get("/v1/jobs?limit=5&state=done").json()["jobs"]
    assert [x["id"] for x in lst] == [job["id"]]
    assert client.get("/v1/jobs?state=queued").json()["jobs"] == []
    pool = client.get("/v1/system").json()["pool"]
    assert pool["entries"][0]["key"] == "dummy/dit"
    assert client.post("/v1/pool/release", json={"key": None}).json() == {"released": ["dummy/dit"]}
    assert client.post("/v1/pool/release").json() == {"released": []}


@pytest.mark.parametrize("body,needle", [
    (b"not json", "JSON"),
    (b"[1]", "object"),
    (b"{}", "model"),
    (json.dumps({"model": "nope"}).encode(), "unknown model"),
    (json.dumps({"model": "dummy", "framing": 3}).encode(), "framing"),
    (json.dumps({"model": "dummy", "width": "64"}).encode(), "width"),
    (json.dumps({"model": "dummy", "steps": 0}).encode(), "steps"),
    (json.dumps({"model": "dummy", "inputs": ["x"]}).encode(), "inputs"),
    (json.dumps({"model": "dummy", "inputs": {"image": "/no/such.png"}}).encode(), "image file not found"),
    (json.dumps({"model": "dummy", "seed": "x"}).encode(), "seed"),
    (json.dumps({"model": "dummy", "params": []}).encode(), "params"),
    (json.dumps({"model": "dummy", "audio": "/no/such.wav"}).encode(), "audio file not found"),
])
def test_submit_400(client, body, needle):
    r = client.post("/v1/jobs", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert needle in r.json()["error"]


def test_bad_query_and_404(client):
    assert client.get("/v1/jobs?state=bogus").status_code == 400
    assert client.get("/v1/jobs?limit=x").status_code == 400
    assert client.get("/v1/jobs/nope").status_code == 404
    assert client.delete("/v1/jobs/nope").status_code == 404


def test_cancel_and_pause(client, engine):
    assert client.post("/v1/queue/pause").json() == {"paused": True}
    job = client.post("/v1/jobs", json={"model": "dummy"}).json()
    assert client.get("/v1/system").json()["queue"]["paused"] is True
    r = client.delete(f"/v1/jobs/{job['id']}")
    assert r.json()["state"] == "cancelled"
    assert client.post("/v1/queue/resume").json() == {"paused": False}


def test_sse_sends_recent_events_first(client, engine):
    job = client.post("/v1/jobs", json={"model": "dummy", "params": {"steps": 1, "step_s": 0}}).json()
    wait_until(lambda: engine.get(job["id"]).state.value == "done")
    events = []
    with client.stream("GET", "/v1/events?max_events=3") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert len(events) == 3
    assert events[0]["kind"] == "job.queued" and events[0]["job_id"] == job["id"]


def test_sse_live_event(client, engine):
    import threading
    events = []
    # TestClient may buffer the body, so publish from another thread once subscribed
    threading.Thread(target=lambda: (wait_until(lambda: engine.bus.subscriber_count > 0),
                                     engine.pause())).start()
    with client.stream("GET", "/v1/events?recent=0&max_events=1") as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert events[0]["kind"] == "queue.paused"


def test_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200 and "EventSource" in r.text and "<title>Shuttrr</title>" in r.text


def test_api_key_for_non_loopback(engine):
    c = TestClient(create_app(engine, api_key="s3cret"))  # client host is "testclient"
    assert c.get("/v1/system").status_code == 401
    assert c.get("/v1/system", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/v1/system?key=s3cret").status_code == 200


def test_bind_rules():
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0")
    check_bind("127.0.0.1", None)
    with pytest.raises(ValueError):
        check_bind("0.0.0.0", None)
    check_bind("0.0.0.0", "key")


def test_no_reserve_by_default(tmp_path):
    from shutter.core.config import auto_headroom
    assert auto_headroom(16 * GB) == 0 and auto_headroom(64 * GB) == 0
    # 16 GB Mac with 4 GB free: a 4 GB job starts, nothing is held back
    cfg = Config(home=str(tmp_path / "home"), memory_poll_s=0.01, busy_process_pattern=None)
    assert cfg.headroom_gb is None and cfg.headroom_auto
    eng = Engine(cfg, probe=FakeProbe(total=16 * GB, available=4 * GB), protected=[],
                 watch_interval_s=0, sample_interval_s=0.01)
    s = TestClient(create_app(eng)).get("/v1/system").json()
    assert s["guard"]["headroom_bytes"] == 0 and s["guard"]["headroom_auto"] is True
    assert s["guard"]["swap_growth_bytes"] == 1 * GB
    assert eng.guard.check(4 * GB, "a 4 GB job").ok
    assert not eng.guard.check(int(4.1 * GB), "a 4.1 GB job").ok
    # an explicit number still wins
    cfg2 = Config(home=str(tmp_path / "home2"), headroom_gb=3, busy_process_pattern=None)
    eng2 = Engine(cfg2, probe=FakeProbe(total=16 * GB, available=4 * GB), protected=[])
    s = TestClient(create_app(eng2)).get("/v1/system").json()
    assert s["guard"]["headroom_bytes"] == 3 * GB and s["guard"]["headroom_auto"] is False
    assert not eng2.guard.check(2 * GB, "a 2 GB job").ok


def test_default_config_writes_automatic_headroom(tmp_path):
    cfg = Config(home=str(tmp_path / "home"))
    path = cfg.save()
    data = json.loads(open(path).read())
    assert "headroom_gb" in data and data["headroom_gb"] is None
    assert Config.load(path).headroom_auto


# ---------------------------------------------------------------- modes / settings


def test_settings_defaults_and_ladder(client):
    s = client.get("/v1/settings").json()
    assert (s["mode"], s["efficiency"], s["rung"]) == ("fast", 0.5, "resident")
    assert [r["id"] for r in s["ladder"]] == ["resident", "split", "staged", "stream_wide", "stream_tight"]
    assert [r["at"] for r in s["ladder"]] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all(r["label"] and r["detail"] and "peak_bytes" not in r for r in s["ladder"])
    assert s["measured_for"] is None
    assert client.get("/v1/system").json()["settings"] == {"mode": "fast", "efficiency": 0.5, "rung": "resident"}


def test_settings_post_validates_persists_and_publishes(client, engine):
    for body, msg in [({"mode": "turbo"}, "mode must be"), ({"mode": 1}, "mode must be"),
                      ({"efficiency": "0.5"}, "efficiency must be"), ({"efficiency": True}, "efficiency must be"),
                      ([1], "body must be a JSON object")]:
        r = client.post("/v1/settings", json=body)
        assert r.status_code == 400 and r.json()["error"].startswith(msg), body
    assert client.post("/v1/settings", content=b"{nope").status_code == 400
    with open(engine.config.path, "w") as f:
        json.dump({"port": 8484, "some_future_key": {"x": 1}}, f)
    sub = engine.bus.subscribe(maxlen=50)
    r = client.post("/v1/settings", json={"mode": "low_memory", "efficiency": 0.8})
    assert r.status_code == 200
    v = r.json()
    assert (v["mode"], v["efficiency"], v["rung"]) == ("low_memory", 0.8, "stream_wide") and len(v["ladder"]) == 5
    ev = None
    while True:
        e = sub.get(1.0)
        assert e is not None
        if e.kind == "settings.changed":
            ev = e
            break
    assert ev.data == {"mode": "low_memory", "efficiency": 0.8, "rung": "stream_wide"}
    sub.close()
    assert client.post("/v1/settings", json={"efficiency": 7}).json()["efficiency"] == 1.0
    v = client.post("/v1/settings", json={"efficiency": 0.30000000000000004}).json()
    assert (v["mode"], v["efficiency"], v["rung"]) == ("low_memory", 0.3, "split")
    data = json.loads(open(engine.config.path).read())
    assert data == {"port": 8484, "some_future_key": {"x": 1}, "default_mode": "low_memory", "default_efficiency": 0.3}
    assert Config.load(engine.config.path).default_mode == "low_memory"


def test_submit_copies_mode_defaults(client, engine):
    engine.pause()
    client.post("/v1/settings", json={"mode": "low_memory", "efficiency": 0.25})
    j = client.post("/v1/jobs", json={"model": "dummy", "params": {"steps": 1}}).json()
    assert j["request"]["params"] == {"steps": 1, "mode": "low_memory", "efficiency": 0.25}
    j = client.post("/v1/jobs", json={"model": "dummy", "params": {"mode": "fast"}}).json()
    assert j["request"]["params"] == {"mode": "fast", "efficiency": 0.25}
    j = client.post("/v1/jobs", json={"model": "dummy", "mode": "low-memory", "efficiency": 2}).json()
    assert j["request"]["params"] == {"mode": "low_memory", "efficiency": 1.0}
    for p in ({"mode": "turbo"}, {"efficiency": "hi"}):
        assert client.post("/v1/jobs", json={"model": "dummy", "params": p}).status_code == 400


def test_settings_ladder_measured_json(client, engine):
    import os
    path = os.path.join(engine.config.home, "measured.json")
    with open(path, "w") as f:
        f.write("{broken")
    assert client.get("/v1/settings").json()["measured_for"] is None
    with open(path, "w") as f:
        json.dump({"a": {"720": {"resident": {"peak_bytes": 1, "seconds": 1}}},
                   "longcat": {"704x1312": {"resident": {"peak_bytes": 40 * GB, "seconds": 100.0, "identical": True},
                                            "stream_tight": {"peak_bytes": 9 * GB, "seconds": "x"}}}}, f)
    s = client.get("/v1/settings").json()
    assert s["measured_for"] == {"model": "longcat", "size": "704x1312"}
    assert s["ladder"][0]["peak_bytes"] == 40 * GB and s["ladder"][0]["seconds"] == 100.0
    assert s["ladder"][4]["peak_bytes"] == 9 * GB and "seconds" not in s["ladder"][4]


def test_daemon_ladder_matches_modes_py():
    """The Rust daemon keeps a built-in copy of the ladder: it must match shutter/core/modes.py."""
    import re
    from pathlib import Path
    from shutter.core.engine import _LADDER
    src = (Path(__file__).resolve().parents[1] / "daemon" / "src" / "modes.rs").read_text()
    rows = re.findall(r'Rung \{ at: ([\d.]+), id: "([^"]+)", label: "([^"]+)", detail: "((?:[^"\\]|\\.)*)" \}', src)
    assert [(float(a), i, l, d.replace('\\"', '"')) for a, i, l, d in rows] == \
        [(r["at"], r["id"], r["label"], r["detail"]) for r in _LADDER]


def test_catalog_lists_downloadable_models(client):
    r = client.get("/v1/catalog")
    assert r.status_code == 200
    row = {m["id"]: m for m in r.json()["models"]}["longcat-avatar-1.6-8bit"]
    assert row["installed"] is False and row["size_bytes"] > 30 * 2**30
    assert client.post("/v1/catalog/nope/download").status_code == 404
    assert client.delete("/v1/catalog/longcat-avatar-1.6-8bit/download").status_code == 200


def test_catalog_refuses_a_model_that_does_not_fit(client, monkeypatch):
    from shutter.core import catalog as cat
    monkeypatch.setattr(cat, "total_memory", lambda: 16 * 2**30)
    monkeypatch.setattr(cat, "macos_baseline", lambda total: cat.MACOS_BASELINE)   # not this Mac's live wired memory
    r = client.post("/v1/catalog/wan2.2-t2v-a14b/download")
    assert r.status_code == 409 and "Needs a" in r.text
    rows = {m["id"]: m for m in client.get("/v1/catalog").json()["models"]}
    assert rows["wan2.2-t2v-a14b"]["fit"]["level"] == "no_fit"
    assert rows["wan2.1-t2v-1.3b-8bit"]["fit"]["level"] == "recommended"


def test_licensed_model_needs_acceptance_before_download(client, monkeypatch):
    """Fish Drama 3: without accept_licence=1 the download is refused with the licence; nothing starts."""
    from shutter.core import catalog as cat
    monkeypatch.setattr(cat, "total_memory", lambda: 128 * 2**30)
    started = []
    import subprocess
    real = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda args, *a, **k: started.append(args) if "download" in args
                        else real(args, *a, **k))
    r = client.post("/v1/catalog/fish-drama3-8bit/download")
    assert r.status_code == 409 and "still uploading" in r.json()["error"]          # files not all on HF yet
    monkeypatch.setattr(cat, "remote_ready", lambda mid, **k: True)
    r = client.post("/v1/catalog/fish-drama3-8bit/download")
    assert r.status_code == 409 and r.json()["code"] == "licence_required"
    assert r.json()["licence"]["name"].startswith("Fish Audio Community License") and not started
