"""Modes (docs/MODES.md): the ladder, mode resolution, fit-to-budget, measured.json, and how the LongCat and dummy
adapters map rungs to switches, step down in Fast mode, release the DiT before the decode (staged) and report
per-stage times and peaks."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from shutter.adapters import longcat as L
from shutter.adapters.base import RunContext
from shutter.adapters.dummy import DummyAdapter
from shutter.core import modes as M
from shutter.addons.reels import Framing
from shutter.types import GB, Job, JobRequest, MemoryEstimate


# ----------------------------------------------------------------------------- ladder / resolution


def test_ladder_matches_the_spec():
    assert [(r["at"], r["id"]) for r in M.LADDER] == [
        (0.0, "resident"), (0.25, "split"), (0.5, "staged"), (0.75, "stream_wide"), (1.0, "stream_tight")]
    for r in M.LADDER:
        assert set(r) >= {"at", "id", "label", "detail", "identical"}
        assert r["identical"] is True       # every rung is bit-identical to the production compiled path


@pytest.mark.parametrize("eff,rid", [(0, "resident"), (0.1, "resident"), (0.25, "split"), (0.49, "split"),
                                     (0.5, "staged"), (0.74, "staged"), (0.75, "stream_wide"),
                                     (0.99, "stream_wide"), (1.0, "stream_tight"), (5, "stream_tight"),
                                     (-1, "resident")])
def test_rung_for(eff, rid):
    assert M.rung_for(eff)["id"] == rid


def test_rung_for_rejects_garbage():
    with pytest.raises(ValueError):
        M.rung_for("lots")
    with pytest.raises(ValueError):
        M.rung_for(float("nan"))


def test_resolve_mode():
    c = M.resolve_mode({}, {})
    assert (c.mode, c.rung, c.step_down) == ("fast", "resident", True)
    c = M.resolve_mode({"mode": "low_memory", "efficiency": 0.8}, {"mode": "fast"})
    assert (c.mode, c.rung, c.efficiency) == ("low_memory", "stream_wide", 0.8)
    c = M.resolve_mode({}, {"default_mode": "low_memory", "default_efficiency": 0.3})
    assert (c.mode, c.rung) == ("low_memory", "split")
    c = M.resolve_mode({"mode": "low-memory"}, {})
    assert c.rung == "staged"                     # low memory without a slider value: 0.5
    c = M.resolve_mode({"rung": "stream_tight"}, {})
    assert c.rung == "stream_tight" and c.explicit_rung and not c.step_down
    with pytest.raises(ValueError):
        M.resolve_mode({"mode": "turbo"}, {})
    with pytest.raises(ValueError):
        M.resolve_mode({"rung": "nope"}, {})


def test_fit_to_budget():
    est = {"resident": MemoryEstimate(peak=40 * GB), "split": {"peak_bytes": 38 * GB}, "staged": 36 * GB,
           "stream_wide": MemoryEstimate(peak=18 * GB), "stream_tight": MemoryEstimate(peak=15 * GB)}
    assert M.fit_to_budget(est, 50 * GB) == "resident"
    assert M.fit_to_budget(est, 37 * GB) == "staged"
    assert M.fit_to_budget(est, 16 * GB) == "stream_tight"
    assert M.fit_to_budget(est, 10 * GB) == "stream_tight"        # nothing fits: lowest rung, the guard refuses
    assert M.fit_to_budget(est, None, "split") == "split"
    assert M.fit_to_budget(est, 50 * GB, "stream_wide") == "stream_wide"   # never steps up
    assert M.fit_to_budget({"resident": 40 * GB, "stream_tight": 15 * GB}, 20 * GB) == "stream_tight"


def test_measured_json_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SHUTTER_HOME", str(tmp_path / "home"))
    assert M.measured_path() == tmp_path / "home" / "measured.json"
    assert M.load_measured() == {}
    M.record_measurement("lc", "480x832", "split", peak_bytes=31 * GB, seconds=430.04, identical=True)
    M.record_measurement("lc", "480x832", "stream_tight", peak_bytes=14 * GB, seconds=480, identical=True)
    data = json.loads((tmp_path / "home" / "measured.json").read_text())
    assert data["lc"]["480x832"]["split"]["peak_bytes"] == 31 * GB
    assert data["lc"]["480x832"]["split"]["seconds"] == 430.0 and data["lc"]["480x832"]["split"]["identical"]
    assert M.measured_for("lc", "480x832", "split")["peak_bytes"] == 31 * GB
    assert M.measured_for("lc", "704x1312", "split") is None
    lad = M.ladder_with_measurements("lc", "480x832")
    assert lad[1]["peak_bytes"] == 31 * GB and "peak_bytes" not in lad[0]
    (tmp_path / "home" / "measured.json").write_text("{broken")
    assert M.load_measured() == {}


# ----------------------------------------------------------------------------- LongCat estimates per rung


def req(framing="full", **params):
    return JobRequest(model="lc", prompt="a man talks", image="img.png", audio="vo.wav",
                      framing=Framing(framing), params=params)


def test_rung_estimates_order_and_reasoning():
    plan = L.resolve_render(req(size="480x832"), None)
    e = {r: L.estimate_rung(plan, r) for r in M.RUNG_IDS}
    gb = {r: e[r].peak / GB for r in e}
    assert gb["resident"] == pytest.approx(33.8 + 1.7, abs=0.2)
    assert gb["resident"] - gb["split"] == pytest.approx(2.0, abs=0.01)
    assert gb["resident"] - gb["staged"] == pytest.approx(2.7, abs=0.01)
    assert gb["resident"] > gb["split"] > gb["staged"] > gb["stream_wide"] > gb["stream_tight"]
    # streamed: the VAE decode alone (13.8 GB) is the floor
    assert gb["stream_tight"] >= L.VAE_DECODE_ALONE_GB
    assert gb["stream_wide"] - gb["stream_tight"] == pytest.approx(6 * (L.DIT_GB - L.NONBLOCK_GB + L.LORA_GB) / 48,
                                                                    abs=0.05)
    full = L.resolve_render(req(), None)
    assert L.estimate_rung(full, "staged").peak / GB == pytest.approx(36.3 + 1.7 - 2.7, abs=0.2)
    # measured values win for a plain plan
    m = L.estimate_rung(plan, "split", measured={"peak_bytes": int(30 * GB)})
    assert m.peak / GB == pytest.approx(30 + 1.7, abs=0.01) and "measured" in m.note
    # continuation KV cache in multi-segment clips
    two = L.estimate_rung(plan, "stream_tight", n_segments=2)
    assert (two.peak - e["stream_tight"].peak) == pytest.approx(L.kv_cache_bytes(plan), abs=2)
    s = {r: L.estimate_rung_seconds(plan, r)["total_s"] for r in M.RUNG_IDS}
    assert s["resident"] < s["split"] < s["stream_wide"] <= s["stream_tight"]


def test_stream_tight_480p_bf16_ignores_weights_on_disk():
    """Lowest memory at 480p with the half-size decode measures ~8.5 GB MLX in prep (+1.7 GB process): the estimate
    stays near 10.2 GB however big the downloaded folder is (both LoRAs, text encoder), and never under the floor."""
    plan = L.resolve_render(req(size="480x832", vae_bf16=True), None)
    assert plan.vae_bf16 and (plan.height, plan.width) == (480, 832)
    floor = L.STREAM_PREP_GB + L.PROCESS_OVER_MLX_GB
    for resident in (None, int(18 * GB), int(22 * GB), int(24 * GB)):   # old layout ... both LoRAs on disk
        e = L.estimate_rung(plan, "stream_tight", resident_bytes=resident)
        assert 9.5 <= e.peak / GB <= 11.5, (resident, e.note)
        assert e.peak / GB >= floor - 1e-6
    # resident / split / staged keep the shift by the real weights
    assert (L.estimate_rung(plan, "staged", resident_bytes=int(22 * GB)).peak
            - L.estimate_rung(plan, "staged").peak) / GB == pytest.approx(4.0, abs=0.01)


def test_adapter_estimate_steps_down_to_fit_the_budget():
    ad = L.LongCatAvatarAdapter("lc", Path("/nonexistent"), {"measured_path": "/nonexistent/m.json"})
    r = req(size="480x832")
    r.duration_s = 3.72
    r.params["budget_bytes"] = int(100 * GB)
    e = ad.estimate(r)
    assert "rung resident" in e.note and e.peak / GB == pytest.approx(35.5, abs=0.2)
    r.params["budget_bytes"] = int(33 * GB)
    e = ad.estimate(r)
    assert "rung staged" in e.note and "stepped down from resident" in e.note
    r.params["budget_bytes"] = int(20 * GB)
    assert "rung stream_wide" in ad.estimate(r).note
    r.params.update(mode="low_memory", efficiency=1.0, budget_bytes=int(100 * GB))
    assert "rung stream_tight" in ad.estimate(r).note


# ----------------------------------------------------------------------------- LongCat run() on a fake tree


FAKE = '''
import math, os, pathlib
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

FPS = 25
NUM_FRAMES = 93
NUM_COND_FRAMES = 13
EVENTS = []

def load_vae(components, fp32=False):
    return {"fp32": fp32}

def vae_decode(vae, z, tile=0, overlap=12):
    EVENTS.append(("decode", z.shape))
    return np.zeros((z.shape[2], 8, 8, 3), np.uint8)

@dataclass
class ModelPaths:
    components: pathlib.Path
    dit: pathlib.Path
    version: str = "1.6"
    loras: dict = field(default_factory=dict)
    distilled: bool = True

    @classmethod
    def from_variant(cls, folder, version=None):
        folder = pathlib.Path(folder)
        return cls(components=folder, dit=folder / "dit", loras={"dmd": "d", "opsd": "o"})

@dataclass
class Settings:
    resolution: str = "480p"
    size: Optional[tuple] = None
    seed: int = 42
    steps: int = 8
    mix_steps: int = 4

class AvatarGenerator:
    def __init__(self, paths, cache_dir=None, vae_tile=0, vae_overlap=12, compile=True, vae_fp32=None):
        self.paths, self.vae_tile, self.vae_overlap, self.compile = paths, vae_tile, vae_overlap, compile
        self.vae = load_vae(paths.components)
        self.dit = None
        self.lora_keys = []

    def _load_dit(self):
        if self.dit is not None:
            return
        self.dit = {"split": os.environ.get("LC_BLOCK_SPLIT"), "ffn": os.environ.get("LC_FFN_CHUNKS")}
        self.lora_keys.extend(["dmd", "opsd"])
        EVENTS.append(("load_dit", dict(self.dit), list(self.lora_keys)))

    def _set_loras_for_step(self, i, st):
        EVENTS.append(("step", i, self.dit is not None))

    def _denoise(self, latents, num_cond, text, text_mask, neg, neg_mask, audio, st, num_ref_latents=None, log=print):
        for i in range(st.steps):
            self._set_loras_for_step(i, st)
        return latents

    def generate(self, audio_path, prompt, st, image=None, start_frames=None, duration=None, on_segment=None,
                 segment_check=None, log=print):
        import mlx.core as mx
        H, W = st.size
        NF = int(os.environ.get("LC_SEG_FRAMES", NUM_FRAMES))
        step = NF - NUM_COND_FRAMES
        n_seg = max(1, math.ceil((duration * FPS - NF) / step) + 1)
        self._load_dit()
        out = []
        for s in range(n_seg):
            lat = self._denoise(mx.zeros((1, 4, 2, 2, 2)), 1, None, None, None, None, None, st)
            fr = vae_decode(self.vae, lat + s, self.vae_tile, self.vae_overlap)
            n = NF if s == 0 else step
            out.append(np.zeros((n, H, W, 3), np.uint8) + 40 * s)
            if on_segment:
                on_segment(np.concatenate(out), s + 1)
        return np.concatenate(out)[: int(round(duration * FPS))], {"segments": n_seg}
'''


@pytest.fixture
def fake_code(tmp_path):
    root = tmp_path / "code"
    pkg = root / "longcat_video_avatar"
    (pkg / "models").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "models" / "__init__.py").write_text("")
    (pkg / "models" / "attention.py").write_text("import os\nSDPA_HEAD_CHUNK = int(os.environ.get('LC_SDPA_HEADS', '1'))\n")
    (pkg / "fast.py").write_text("import os\nFFN_CHUNKS = int(os.environ.get('LC_FFN_CHUNKS', '1'))\n")
    (pkg / "generation.py").write_text(textwrap.dedent(FAKE))
    yield root
    for name in [n for n in sys.modules if n.startswith("longcat_video_avatar")]:
        del sys.modules[name]
    if str(root.resolve()) in sys.path:
        sys.path.remove(str(root.resolve()))


class Pool:
    def __init__(self):
        self.items, self.loads, self.released = {}, [], []

    def get(self, key, loader, size_bytes, *, keep=True):
        if key not in self.items:
            self.loads.append((key, size_bytes))
            self.items[key] = loader()
        return self.items[key]

    def release(self, key):
        self.released.append(key)
        self.items.pop(key, None)

    def release_all(self, prefix=""):
        for k in list(self.items):
            self.release(k)


@pytest.fixture
def media(tmp_path):
    img = tmp_path / "img.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x180",
                    "-frames:v", "1", str(img)], check=True)
    wav = tmp_path / "vo.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=220:r=48000:d=4.2",
                    str(wav)], check=True)
    return img, wav


def _ctx(tmp_path, pool, available=None, headroom=0):
    reports = []
    return RunContext(models=pool, cache=None, work_dir=tmp_path / "work", report=lambda p, s: reports.append((p, s)),
                      check_cancelled=lambda: None,
                      memory_sample=lambda: ({"available": available} if available else {}),
                      settings={"headroom_bytes": headroom}), reports


def _job(media, tmp_path, **params):
    img, wav = media
    r = JobRequest(model="lc", prompt="hi", image=str(img), audio=str(wav), framing=Framing.FULL,
                   output=str(tmp_path / "out" / "clip.mp4"), seed=42,
                   params={"size": "64x128", "upscale_to": "off", "zoom_cancel": "off", **params})
    return Job(request=r)


def events():
    return sys.modules["longcat_video_avatar.generation"].EVENTS


@pytest.mark.parametrize("rung,split,ffn,releases", [
    ("resident", "0", "1", 0), ("split", "1", "1", 0), ("staged", "1", "4", 2)])
def test_rung_switches_and_staged_release(fake_code, media, tmp_path, rung, split, ffn, releases):
    ad = L.LongCatAvatarAdapter("lc", tmp_path / "model", {"code_dir": str(fake_code),
                                                          "measured_path": str(tmp_path / "none.json")})
    pool = Pool()
    ctx, reports = _ctx(tmp_path, pool)
    res = ad.run(_job(media, tmp_path, mode="low_memory", efficiency=M.rung(rung)["at"]), ctx)   # 2 segments
    assert res["mode"] == "low_memory" and res["rung"] == rung and "stepped_down_from" not in res
    assert res["identical"] is True
    assert res["env"]["LC_BLOCK_SPLIT"] == split and res["env"]["LC_FFN_CHUNKS"] == ffn
    loads = [e for e in events() if e[0] == "load_dit"]
    assert loads[0][1] == {"split": split, "ffn": ffn}
    assert res["dit_releases"] == releases and res["dit_reloads"] == (1 if releases else 0)
    if releases:
        # the DiT was gone during each decode and reloaded for segment 2; LoRA keys not duplicated on reload
        assert len(loads) == 2 and loads[1][2] == ["dmd", "opsd"]
        steps = [e for e in events() if e[0] == "step"]
        assert all(alive for _, _, alive in steps)
    stages = [s["stage"] for s in res["stages"]]
    assert stages[0] == "prep" and stages[1] == "load" and stages[-1] == "post"
    assert stages.count("step") == 16 and stages.count("decode") == 2
    assert ("release" in stages) == bool(releases) and ("reload" in stages) == bool(releases)
    assert set(res["stage_seconds"]) >= {"prep", "load", "denoise", "decode", "post"}
    assert len(res["step_seconds"]) == 16
    assert res["peak_bytes"] is not None and res["peak_bytes"] == max(res["stage_peak_bytes"].values())
    assert all(s["peak_bytes"] is not None for s in res["stages"])


def test_fast_mode_steps_down_and_refuses_when_nothing_fits(fake_code, media, tmp_path):
    from shutter.types import MemoryRefused

    ad = L.LongCatAvatarAdapter("lc", tmp_path / "model", {"code_dir": str(fake_code),
                                                          "measured_path": str(tmp_path / "none.json")})
    pool = Pool()
    job = _job(media, tmp_path)
    job.request.duration_s = 3.0
    ests = ad.rung_estimates(job.request, ad.resolve(job.request), 1)
    assert ests["split"].peak > ests["staged"].peak
    avail = ests["staged"].peak + int(6 * GB) + 1          # staged fits, split does not
    ctx, reports = _ctx(tmp_path, pool, available=avail, headroom=int(6 * GB))
    res = ad.run(job, ctx)
    assert res["mode"] == "fast" and res["rung"] == "staged" and res["stepped_down_from"] == "resident"
    assert res["budget_bytes"] == ests["staged"].peak + 1 and res["estimate_peak_bytes"] <= res["budget_bytes"]
    assert any("stepping down to staged" in s for _, s in reports)
    assert any("stepped down" in n for n in res["plan"]["notes"])
    ctx, _ = _ctx(tmp_path, pool, available=int(10 * GB), headroom=int(6 * GB))
    with pytest.raises(MemoryRefused):
        ad.run(job, ctx)
    ctx, _ = _ctx(tmp_path, pool, available=int(200 * GB))
    assert ad.run(job, ctx)["rung"] == "resident"


def test_save_latents_like_run_e2e(fake_code, media, tmp_path):
    ad = L.LongCatAvatarAdapter("lc", tmp_path / "model", {"code_dir": str(fake_code),
                                                          "measured_path": str(tmp_path / "none.json")})
    ctx, _ = _ctx(tmp_path, Pool())
    res = ad.run(_job(media, tmp_path, save_latents=str(tmp_path / "lat")), ctx)
    a = np.load(tmp_path / "lat" / "latents_denorm.npy")
    assert a.dtype == np.float32 and float(a.max()) == 1.0          # last segment's latents (overwritten)
    assert len(res["latents"]) == 2 and res["latents_dir"] == str(tmp_path / "lat")


def test_stream_rungs_use_a_streamed_build(fake_code, media, tmp_path):
    ad = L.LongCatAvatarAdapter("lc", tmp_path / "model", {"code_dir": str(fake_code), "stream_window_tight": 3})
    assert ad.stream_cfg("resident") is None and ad.stream_cfg("staged") is None
    assert ad.stream_cfg("stream_wide")["window"] == 8 and ad.stream_cfg("stream_tight")["window"] == 3
    env = ad.env_for(L.resolve_render(req(size="64x128"), None), rung_id="stream_tight")
    assert env["LC_BLOCK_SPLIT"] == "1" and env["LC_FFN_CHUNKS"] == "4"
    s1 = ad.load_signature(env, ad.stream_cfg("stream_tight"))
    s2 = ad.load_signature(env, ad.stream_cfg("stream_wide"))
    assert s1 != s2 != ad.load_signature(env)
    # the pool is asked for the streamed footprint, not the whole DiT
    assert ad.stream_resident_bytes(2) < 3 * GB < ad.stream_resident_bytes(8) < L.DIT_GB * GB


def test_stage_meter_summary():
    t = iter([0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 9.5])
    m = L.StageMeter(None, clock=lambda: next(t))
    m.start("prep")
    m.mark("step", step=1)
    m.mark("step", step=2)
    m.mark("decode")
    m.close()
    s = m.summary()
    assert s["stage_seconds"] == {"prep": 1.0, "denoise": 3.0, "decode": 5.5}
    assert s["step_seconds"] == [2.0, 1.0] and s["peak_bytes"] is None


# ----------------------------------------------------------------------------- dummy adapter


class _Cache:
    def get_or_compute(self, ns, parts, compute):
        return compute()


def _dctx(tmp_path, available=None, headroom=0):
    return RunContext(models=Pool(), cache=_Cache(), work_dir=tmp_path, report=lambda p, s: None,
                      check_cancelled=lambda: None,
                      memory_sample=lambda: ({"available": available} if available else {}),
                      settings={"headroom_bytes": headroom, "sleep": lambda s: None})


def test_dummy_modes(tmp_path):
    ad = DummyAdapter("dummy", tmp_path)
    r = JobRequest(model="dummy", prompt="p", output=str(tmp_path / "o.json"), params={"peak_gb": 10, "steps": 3})
    assert ad.estimate(r).peak == 10 * GB and ad.estimate(r).note == "dummy estimate"
    res = ad.run(Job(request=r), _dctx(tmp_path))
    assert (res["mode"], res["rung"]) == ("fast", "resident") and "stepped_down_from" not in res
    assert len(res["step_seconds"]) == 3 and {"encode", "denoise", "decode"} <= set(res["stage_seconds"])
    res = ad.run(Job(request=r), _dctx(tmp_path, available=int(10.5 * GB), headroom=int(6 * GB)))
    assert res["rung"] == "stream_tight" and res["stepped_down_from"] == "resident"
    r2 = JobRequest(model="dummy", prompt="p", params={"peak_gb": 10, "mode": "low_memory", "efficiency": 0.6,
                                                       "save_latents": str(tmp_path / "lat")})
    assert ad.estimate(r2).peak == int(10 * 0.92 * GB)
    res2 = ad.run(Job(request=r2), _dctx(tmp_path))
    assert res2["rung"] == "staged"
    a = np.load(tmp_path / "lat" / "latents_denorm.npy")
    r3 = JobRequest(model="dummy", prompt="p", params={"rung": "stream_wide", "save_latents": str(tmp_path / "l3")})
    ad.run(Job(request=r3), _dctx(tmp_path))
    assert np.array_equal(a, np.load(tmp_path / "l3" / "latents_denorm.npy"))     # identical across rungs
    r4 = JobRequest(model="dummy", params={"peak_gb": 10, "budget_bytes": int(9.3 * GB)})
    assert ad.estimate(r4).peak == int(10 * 0.92 * GB)
