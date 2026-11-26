"""Chip-aware defaults, expected times, request options, estimates and the person's own saved defaults."""

import pytest
from starlette.testclient import TestClient

from shutter.core import catalog as cat
from shutter.core import request_options as ro
from shutter.core.chip import tier_of
from shutter.core.config import Config
from shutter.core.memory import GB

M = "wan2.2-ti2v-5b"
BASE = {"name": "Apple M5", "tier": "base", "gpu_cores": 10}
MAX = {"name": "Apple M5 Max", "tier": "max", "gpu_cores": 40}


def chip(monkeypatch, info):
    monkeypatch.setattr(cat, "chip_info", lambda: info)


def test_tier_of():
    assert tier_of("Apple M5") == "base" and tier_of("Apple M1") == "base"
    assert tier_of("Apple M4 Pro") == "pro" and tier_of("Apple M3 Max") == "max"
    assert tier_of("Apple M2 Ultra") == "ultra"
    assert tier_of("Intel(R) Core(TM) i9") == "" and tier_of("") == ""


def test_16gb_default_ignores_the_chip(monkeypatch):
    for info in (BASE, MAX):
        chip(monkeypatch, info)
        assert cat.default_request(M, 16 * GB) == (704, 1280, 25)
        assert cat.preset_for(M, 16 * GB)["options"]["default_steps"] == 30


def test_base_chip_with_24gb_or_more_gets_a_2s_clip(monkeypatch):
    chip(monkeypatch, BASE)
    for g in (24, 32, 64):
        assert cat.default_request(M, g * GB) == (704, 1280, 49)
        p = cat.preset_for(M, g * GB)
        assert p["options"] == {"default_size": [704, 1280], "default_frames": 49, "default_steps": 30,
                                "vae_tile": True}
        assert "Apple M5" in p["notes"][0]
        f = cat.classify_fit(M, g * GB, baseline=3 * GB)
        assert f["level"] == "recommended" and f["measured"] and f["peak_bytes"] <= g * GB // 2
        t = cat.expected_time(M, g * GB)
        assert t["measured"] and 1200 < t["seconds"] < 1300 and t["label"] == "About 21 min on this Mac"


def test_pro_max_with_24gb_or_more_keep_the_reference_recipe(monkeypatch):
    chip(monkeypatch, MAX)
    for g in (24, 32):
        assert cat.default_request(M, g * GB) == (704, 1280, 121)
        assert cat.preset_for(M, g * GB)["options"] == {}
        assert cat.classify_fit(M, g * GB, baseline=3 * GB)["level"] == "barely"
    t = cat.expected_time(M, 128 * GB)
    assert not t["measured"] and "(estimate)" in t["label"]
    assert 1400 < t["seconds"] < 1600           # 27.6 s/step x 50 + 125 s decode on the M5 Max it was measured on


def test_time_scales_with_gpu_cores(monkeypatch):
    chip(monkeypatch, {"name": "Apple M4", "tier": "base", "gpu_cores": 20})
    t = cat.expected_time(M, 16 * GB)
    assert not t["measured"] and abs(t["seconds"] - 613 / 2) < 2


def test_makes_line_says_the_time(monkeypatch):
    chip(monkeypatch, BASE)
    assert cat.makes_line(M, 16 * GB) == "Makes 1 s 704x1280 clips, about 10 min each on this Mac"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("SHUTTER_HOME", str(tmp_path / "home"))
    c = Config.load()
    c.ensure_dirs()
    c.models[M] = {"adapter": "wan", "dir": str(tmp_path / "nowhere"),
                   "options": cat.preset_for(M, 16 * GB)["options"]}
    c.save()
    return c


def test_request_options_and_estimate(cfg, monkeypatch):
    chip(monkeypatch, BASE)
    o = ro.request_options(cfg, M, 16 * GB)
    assert o["editable"] and o["defaults"] == o["mac_defaults"]
    assert o["defaults"] == {"width": 1280, "height": 704, "frames": 25, "steps": 30, "guidance": 5.0, "shift": 5.0,
                             "fps": 24.0, "vae_tile": True}
    assert o["limits"]["size_multiple"] == 32 and o["sizes"][0]["label"] == "1280x704 (native)"
    assert o["estimate"]["label"] == "About 10 min on this Mac" and not o["estimate"]["issues"]
    e = ro.estimate(cfg, M, {"frames": 50, "width": 1000, "height": 704}, 16 * GB)
    fields = {i["field"]: i for i in e["issues"]}
    assert fields["frames"]["fix"] == 49 and fields["width"]["fix"] == 992
    big = ro.estimate(cfg, M, {"frames": 121, "vae_tile": False}, 16 * GB)
    assert big["warnings"] and "(estimate)" in big["label"]


def test_save_and_reset_defaults(cfg, monkeypatch):
    chip(monkeypatch, BASE)
    ro.save_defaults(cfg, M, {"frames": 49, "steps": 40})
    o = ro.request_options(Config.load(), M, 16 * GB)
    assert o["user_defaults"] == {"frames": 49, "steps": 40} and o["defaults"]["frames"] == 49
    assert o["mac_defaults"]["frames"] == 25
    # a re-download keeps the person's values and refreshes the rest
    cat.register(Config.load(), M, cfg.models[M]["dir"] and __import__("pathlib").Path(cfg.models[M]["dir"]), 16 * GB)
    assert Config.load().models[M]["options"]["default_frames"] == 49
    ro.reset_defaults(Config.load(), M, 16 * GB)
    o = ro.request_options(Config.load(), M, 16 * GB)
    assert o["user_defaults"] == {} and o["defaults"]["frames"] == 25
    with pytest.raises(ro.OptionsError):
        ro.save_defaults(Config.load(), M, {"width": 1280})


def test_api_endpoints(cfg, monkeypatch):
    from shutter.core.engine import Engine
    from shutter.core.memory import FakeProbe
    from shutter.server.api import create_app
    chip(monkeypatch, BASE)
    monkeypatch.setattr(cat, "total_memory", lambda: 16 * GB)
    eng = Engine(Config.load(), probe=FakeProbe(), protected=[], watch_interval_s=0, sample_interval_s=0.01)
    c = TestClient(create_app(eng))
    r = c.get(f"/v1/models/{M}/request-options")
    assert r.status_code == 200 and r.json()["defaults"]["frames"] == 25
    assert c.post(f"/v1/models/{M}/estimate", json={"frames": 49}).json()["seconds"] > 1000
    r = c.put(f"/v1/models/{M}/defaults", json={"frames": 49, "guidance": 4.5})
    assert r.status_code == 200 and r.json()["defaults"]["guidance"] == 4.5
    assert eng.config.models[M]["options"]["default_guidance"] == 4.5
    assert c.delete(f"/v1/models/{M}/defaults").json()["user_defaults"] == {}
    assert c.get("/v1/models/nope/request-options").status_code == 404
    assert c.put(f"/v1/models/{M}/defaults", json={"steps": "x"}).status_code == 400
