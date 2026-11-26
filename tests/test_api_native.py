"""The native API additions on the Python reference server: /v1/memory, /v1/files,
/v1/jobs/{id}/output, the `client` field, /v1/clients and the throughput numbers. The daemon
serves the same shapes (daemon/tests/daemon.rs)."""

import hashlib
import time

import pytest
from starlette.testclient import TestClient

from shutter.addons.reels import Framing, framing_of
from shutter.core.config import Config
from shutter.core.engine import Engine
from shutter.core.memory import GB, FakeProbe
from shutter.core.pool import ModelPool, residency_of
from shutter.core.stats import ClientTracker, StepRate, client_label, parse_step
from shutter.server.api import create_app
from shutter.types import Job, JobRequest


def wait_until(pred, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.005)
    raise AssertionError("timeout")


@pytest.fixture
def engine(tmp_path):
    cfg = Config(home=str(tmp_path / "home"), memory_poll_s=0.01, busy_process_pattern=None, max_upload_mb=0.01)
    probe = FakeProbe(total=64 * GB, available=48 * GB)
    eng = Engine(cfg, probe=probe, protected=[], watch_interval_s=0, sample_interval_s=0.01)
    eng.start()
    yield eng
    eng.stop(timeout=5)


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine))


def done(client, jid):
    wait_until(lambda: client.get(f"/v1/jobs/{jid}").json()["state"] in ("done", "failed", "cancelled"))
    return client.get(f"/v1/jobs/{jid}").json()


# ---------------------------------------------------------------- the request type

def test_request_is_plain_and_folds_legacy_keys():
    r = JobRequest.from_dict({"model": "m", "image": "/a.png", "audio": "/b.wav", "framing": "small",
                              "steps": 4, "cfg": 2, "client": "x", "params": {"cfg": 3}})
    d = r.to_dict()
    assert list(d) == ["model", "prompt", "negative_prompt", "width", "height", "frames", "duration_s", "fps",
                       "steps", "guidance", "seed", "inputs", "output", "priority", "params"]
    assert d["inputs"] == {"image": "/a.png", "audio": "/b.wav"}
    assert d["params"] == {"framing": "small", "cfg": 3}   # explicit params win; client is not a param
    assert "framing" not in d and "image" not in d
    assert r.image == "/a.png" and r.audio == "/b.wav" and r.video is None
    r.image = "/c.png"
    assert r.inputs["image"] == "/c.png"
    # explicit inputs win over the legacy keys; the old constructor keywords still work
    assert JobRequest.from_dict({"model": "m", "image": "/x", "inputs": {"image": "/y"}}).image == "/y"
    r2 = JobRequest(model="m", image="/i.png", framing=Framing.FULL)
    assert r2.inputs == {"image": "/i.png"} and r2.params == {"framing": "full"}
    assert JobRequest.from_dict(r.to_dict()) == r
    with pytest.raises(TypeError):
        JobRequest(model="m", nope=1)
    j = Job(request=r, client="comfyui/0.46")
    assert Job.from_dict(j.to_dict()).client == "comfyui/0.46"


def test_prepare_mirrors_core_fields_into_params(engine):
    ad = engine.registry.adapter_for("dummy")
    req = JobRequest(model="dummy", steps=3, width=64, height=32, guidance=4.5, params={"steps": 9})
    p = ad.prepare(req)
    assert p.params["steps"] == 9 and p.params["guidance"] == 4.5 and p.params["size"] == "32x64"
    assert "guidance" not in req.params   # the stored request is untouched


def test_reels_framing_lives_in_params():
    assert framing_of(JobRequest(model="m")) is Framing.AUTO
    assert framing_of(JobRequest.from_dict({"model": "m", "framing": "small"})) is Framing.SMALL
    with pytest.raises(ValueError):
        framing_of(JobRequest(model="m", params={"framing": "wide"}))


def test_core_does_not_import_the_addon():
    import subprocess
    import sys
    code = ("import sys, shutter.types, shutter.core.engine, shutter.server.api, shutter.worker; "
            "print([m for m in sys.modules if m.startswith('shutter.addons')])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# ---------------------------------------------------------------- stats helpers

def test_parse_step_and_rate():
    assert parse_step("denoise 3/8") == (3, 8)
    assert parse_step("segment 1/2 denoise 3/8") == (3, 8)
    assert parse_step("Step 4 / 10") == (4, 10)
    assert parse_step("segment 1/2 decode") is None and parse_step(None) is None
    t = [0.0]
    r = StepRate(clock=lambda: t[0])
    r.update(1, 4)
    t[0] = 3.0
    r.update(4, 4)
    t[0] = 5.0
    r.update(2, 4)
    assert r.snapshot() == {"step": 2, "steps": 4, "steps_per_s": 1.0}


def test_client_tracker_window():
    now = [1000.0]
    c = ClientTracker(window_s=600, clock=lambda: now[0])
    c.submitted("a")
    c.seen("b")
    now[0] += 700
    c.submitted("c")
    rows = c.list()["clients"]
    assert [r["client"] for r in rows] == ["c"]
    assert client_label(None, None, "curl/8.1 x") == "curl/8.1"


def test_pool_residency():
    class Guard:
        def admit(self, need, what=""):
            pass

    p = ModelPool(Guard())
    p.get("a", lambda: object(), 10)
    p.get("b", lambda: {"w": 1}, 20, residency="streamed")

    class Gen:
        _shutter_stream_cfg = {"window": 2}

    p.get("c", Gen, 30)
    res = {e["key"]: e["residency"] for e in p.stats()["entries"]}
    assert res == {"a": "resident", "b": "streamed", "c": "streamed"}
    assert residency_of(object()) == "resident"


# ---------------------------------------------------------------- /v1/memory

def test_memory_shape_components_and_fit(client, engine):
    m = client.get("/v1/memory").json()
    assert m["total"] == 64 * GB and m["available"] == 48 * GB
    assert m["reserve"] == 0 and m["free_for_models"] == 48 * GB
    assert m["swap"] == 0 and m["pressure"] == "normal" and m["components"] == [] and "model" not in m
    for k in ("held", "resident_bytes", "streamed_bytes", "protected", "reserve_auto", "max_job_bytes"):
        assert k in m
    j = client.post("/v1/jobs", json={"model": "dummy", "params": {"steps": 1, "step_s": 0, "component_gb": 2,
                                                                    "mode": "low_memory", "efficiency": 1.0}}).json()
    assert done(client, j["id"])["state"] == "done"
    m = client.get("/v1/memory").json()
    c = m["components"][0]
    assert c["key"] == "dummy/dit" and c["model"] == "dummy" and c["residency"] == "streamed"
    assert c["size_bytes"] == 2 * GB and m["streamed_bytes"] == 2 * GB and m["resident_bytes"] == 0
    assert client.get("/v1/system").json()["pool"]["entries"][0]["residency"] == "streamed"
    m = client.get("/v1/memory?model=dummy&steps=3").json()["model"]
    assert m["need_bytes"] == GB and m["warm_bytes"] == 2 * GB and m["fits"] is True and m["short_by_bytes"] == 0
    client.post("/v1/pool/release", json={"key": None})
    engine.probe.set(available=GB // 2)
    m = client.get("/v1/memory?model=dummy").json()
    assert m["free_for_models"] == GB // 2
    assert m["model"]["fits"] is False and m["model"]["short_by_bytes"] == GB // 2
    assert "unknown model" in client.get("/v1/memory?model=nope").json()["model"]["error"]
    assert client.get("/v1/memory?width=x").status_code == 400


# ---------------------------------------------------------------- /v1/files and output

def test_files_inputs_and_output(client, engine):
    r = client.post("/v1/files?name=start.png", content=b"\x89PNGraw", headers={"content-type": "image/png"})
    assert r.status_code == 201, r.text
    up = r.json()
    assert set(up) == {"id", "path", "name", "size", "content_type", "sha256", "created_at"}
    assert up["name"] == "start.png" and up["size"] == 7 and up["content_type"] == "image/png"
    assert up["sha256"] == hashlib.sha256(b"\x89PNGraw").hexdigest()
    assert "/work/uploads/" in up["path"] and open(up["path"], "rb").read() == b"\x89PNGraw"
    r = client.post("/v1/files", files={"file": ("voice.wav", b"RIFFxxxx", "audio/wav")}, data={"purpose": "input"})
    assert r.status_code == 201, r.text
    up2 = r.json()
    assert up2["name"] == "voice.wav" and up2["content_type"] == "audio/wav"
    assert client.post("/v1/files", content=b"").status_code == 400
    assert client.post("/v1/files?name=big.bin", content=b"x" * 20000).status_code == 413
    assert client.post("/v1/files", files={"x": (None, "y")}).status_code == 400  # multipart without a file

    j = client.post("/v1/jobs", json={"model": "dummy", "inputs": {"image": up["id"], "audio": up2["path"]},
                                      "params": {"steps": 1, "step_s": 0}}).json()
    assert j["request"]["inputs"] == {"image": up["path"], "audio": up2["path"]}
    fin = done(client, j["id"])
    r = client.get(f"/v1/jobs/{j['id']}/output")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert f"{j['id']}.mp4" in r.headers["content-disposition"]
    assert r.content == open(fin["result"]["output"], "rb").read()

    client.post("/v1/queue/pause")
    q = client.post("/v1/jobs", json={"model": "dummy"}).json()
    r = client.get(f"/v1/jobs/{q['id']}/output")
    assert r.status_code == 409 and "is queued" in r.json()["error"]
    assert client.get("/v1/jobs/nope/output").status_code == 404
    import os
    os.remove(fin["result"]["output"])
    r = client.get(f"/v1/jobs/{j['id']}/output")
    assert r.status_code == 404 and "gone" in r.json()["error"]


# ---------------------------------------------------------------- clients and throughput

def test_clients_and_throughput(client, engine):
    t = client.get("/v1/system").json()["throughput"]
    assert t == {"window_s": 3600, "finished": 0, "mean_seconds": None, "running": None}
    assert client.get("/v1/clients").json() == {"window_s": 600, "clients": []}
    a = client.post("/v1/jobs", json={"model": "dummy", "client": "comfyui/0.46",
                                      "params": {"steps": 2, "step_s": 0.02}}).json()
    assert a["client"] == "comfyui/0.46"
    done(client, a["id"])
    b = client.post("/v1/jobs", json={"model": "dummy", "params": {"steps": 8, "step_s": 0.1}},
                    headers={"x-shutter-client": "my-script"}).json()
    assert b["client"] == "my-script"
    client.get("/v1/system", headers={"x-shutter-client": "watcher/1"})
    wait_until(lambda: (client.get("/v1/system").json()["throughput"]["running"] or {}).get("steps_per_s"))
    run = client.get("/v1/system").json()["throughput"]["running"]
    assert run["id"] == b["id"] and run["steps"] == 8 and run["step"] >= 1 and 1.0 < run["steps_per_s"] < 30
    done(client, b["id"])
    t = client.get("/v1/system").json()["throughput"]
    assert t["finished"] == 2 and t["mean_seconds"] > 0 and t["running"] is None
    rows = {r["client"]: r for r in client.get("/v1/clients").json()["clients"]}
    assert rows["comfyui/0.46"]["requests"] == 1 and rows["my-script"]["requests"] == 1
    assert rows["watcher/1"]["requests"] == 0
    assert set(rows["my-script"]) == {"client", "requests", "requests_total", "first_seen", "last_seen", "idle_s"}
    assert client.get(f"/v1/jobs/{a['id']}").json()["client"] == "comfyui/0.46"
