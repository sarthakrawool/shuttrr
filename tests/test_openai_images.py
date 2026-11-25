"""The OpenAI Images shim on the Python reference server, with the dummy adapter in image mode."""

import base64
import io
import time

import pytest
from starlette.testclient import TestClient

from shutter.core.config import Config
from shutter.core.engine import Engine
from shutter.core.memory import FakeProbe
from shutter.server.api import create_app

FAST = {"steps": 1, "step_s": 0}


@pytest.fixture
def engine(tmp_path):
    cfg = Config(home=str(tmp_path / "home"), memory_poll_s=0.01, busy_process_pattern=None,
                 models={"dummy-image": {"adapter": "dummy-image"}})
    eng = Engine(cfg, probe=FakeProbe(), protected=[], watch_interval_s=0, sample_interval_s=0.01)
    eng.start()
    yield eng
    eng.stop(timeout=5)


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine))


def png_size(b: bytes):
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")


def test_models_list_image_kind(client):
    ids = {m["id"]: m for m in client.get("/v1/models").json()["models"]}
    assert ids["dummy-image"]["kind"] == "image" and ids["dummy"]["kind"] == "video"


def test_generations_b64(client, engine):
    r = client.post("/v1/images/generations", json={"model": "dummy-image", "prompt": "a red fox", "n": 2,
                                                    "size": "64x48", "params": FAST},
                    headers={"x-shutter-client": "open-webui"})
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["created"] > 0 and v["size"] == "64x48" and v["output_format"] == "png" and len(v["data"]) == 2
    jobs = [engine.get(j) for j in v["shutter"]["jobs"]]
    for row, job in zip(v["data"], jobs):
        png = base64.b64decode(row["b64_json"])
        assert png_size(png) == (64, 48)
        assert png == open(job.result["output"], "rb").read() and job.result["output"].endswith(".png")
        assert row["revised_prompt"] == "a red fox" and job.client == "open-webui"
        assert job.request.width == 64 and job.request.height == 48
    assert jobs[1].request.seed == jobs[0].request.seed + 1
    # two seeds, two images
    assert v["data"][0]["b64_json"] != v["data"][1]["b64_json"]


def test_generations_default_model_url_and_pil_decodes(client):
    r = client.post("/v1/images/generations", json={"prompt": "x", "response_format": "url", "seed": 7, "params": FAST})
    assert r.status_code == 200, r.text
    url = r.json()["data"][0]["url"]
    assert url.endswith("/output")
    got = client.get(url[url.index("/v1/"):])
    assert got.status_code == 200 and got.headers["content-type"] == "image/png"
    PIL = pytest.importorskip("PIL.Image")
    assert PIL.open(io.BytesIO(got.content)).size == (64, 64)


@pytest.mark.parametrize("body,status,needle", [
    ({"model": "dummy", "prompt": "x"}, 400, "video model"),
    ({"model": "dummy-image"}, 400, "prompt is required"),
    ({"model": "nope", "prompt": "x"}, 404, "unknown model"),
    ({"model": "dummy-image", "prompt": "x", "n": 11}, 400, "n must be"),
    ({"model": "dummy-image", "prompt": "x", "size": "huge"}, 400, "size"),
    ({"model": "dummy-image", "prompt": "x", "output_format": "jpeg"}, 400, "png"),
    ({"model": "dummy-image", "prompt": "x", "stream": True}, 400, "stream"),
])
def test_generation_errors_in_openai_shape(client, body, status, needle):
    r = client.post("/v1/images/generations", json=body)
    assert r.status_code == status, r.text
    e = r.json()["error"]
    assert needle in e["message"] and set(e) == {"message", "type", "param", "code"}


def test_failed_job_is_a_server_error(client):
    r = client.post("/v1/images/generations", json={"model": "dummy-image", "prompt": "x",
                                                    "params": {"fail": "boom", "steps": 1}})
    assert r.status_code == 500 and r.json()["error"]["type"] == "server_error"
    assert "boom" in r.json()["error"]["message"]


def test_no_image_model_configured(tmp_path):
    cfg = Config(home=str(tmp_path / "h2"), memory_poll_s=0.01, busy_process_pattern=None)
    eng = Engine(cfg, probe=FakeProbe(), protected=[], watch_interval_s=0)
    c = TestClient(create_app(eng))
    r = c.post("/v1/images/generations", json={"prompt": "x"})
    assert r.status_code == 400 and "no image model" in r.json()["error"]["message"]


def red_png():
    from shutter.adapters.dummy import png_bytes
    import numpy as np
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    a[..., 0] = 255
    return png_bytes(a)


def test_edits_multipart_json_and_file_id(client, engine):
    src = red_png()
    r = client.post("/v1/images/edits", data={"prompt": "make it blue", "model": "dummy-image",
                                              "params": '{"steps": 1, "step_s": 0}', "size": "32x32"},
                    files=[("image", ("in.png", src, "image/png")), ("mask", ("m.png", src, "image/png"))])
    assert r.status_code == 200, r.text
    job = engine.get(r.json()["shutter"]["jobs"][0])
    assert "/work/uploads/" in job.request.inputs["image"] and job.request.inputs["image"].endswith("in.png")
    assert open(job.request.inputs["image"], "rb").read() == src
    assert job.request.inputs["mask"].endswith("m.png")
    assert png_size(base64.b64decode(r.json()["data"][0]["b64_json"])) == (32, 32)

    data_url = "data:image/png;base64," + base64.b64encode(src).decode()
    r = client.post("/v1/images/edits", json={"prompt": "p", "images": [{"image_url": data_url}], "params": FAST})
    assert r.status_code == 200, r.text
    job = engine.get(r.json()["shutter"]["jobs"][0])
    assert open(job.request.inputs["image"], "rb").read() == src

    up = client.post("/v1/files?name=a.png", content=src).json()
    r = client.post("/v1/images/edits", json={"prompt": "p", "images": [{"file_id": up["id"]}], "params": FAST})
    assert r.status_code == 200, r.text
    assert engine.get(r.json()["shutter"]["jobs"][0]).request.inputs["image"] == up["path"]

    r = client.post("/v1/images/edits", json={"prompt": "p"})
    assert r.status_code == 400 and "image is required" in r.json()["error"]["message"]
    r = client.post("/v1/images/edits", json={"prompt": "p", "images": [{"image_url": "https://x/y.png"}]})
    assert r.status_code == 400 and "data: URL" in r.json()["error"]["message"]


def test_disconnect_or_timeout_cancels_the_jobs(client, engine):
    engine.config.images_timeout_s = 0.3
    r = client.post("/v1/images/generations", json={"model": "dummy-image", "prompt": "slow",
                                                    "params": {"steps": 50, "step_s": 0.1}})
    assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"
    jid = r.json()["error"]["message"].split("jobs ")[1].split(" ")[0]
    t0 = time.monotonic()
    while engine.get(jid).state.value not in ("cancelled", "failed") and time.monotonic() - t0 < 10:
        time.sleep(0.02)
    assert engine.get(jid).state.value == "cancelled"
