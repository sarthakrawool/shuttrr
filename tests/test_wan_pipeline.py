"""Wan generation end to end on the CPU with tiny random builds: every memory step gives the same frames, runtime
LoRAs, the two-expert (A14B-style) switch, and the adapter (estimate + run).

The DiTs and the Wan 2.1 VAE decoder hold random weights; the prompt encode is replaced by a fixed random embedding
(transformers is not needed), except in the adapter test, which runs the real block-by-block umT5 path on a tiny
text encoder with a stand-in tokenizer.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tests import wan_fixtures as F  # noqa: E402
from tests.test_wan_family import _repo  # noqa: E402
from shutter.core.pool import ModelPool  # noqa: E402
from shutter.families import inspect as inspect_ckpt  # noqa: E402
from shutter.families.wan import pipeline as WP  # noqa: E402
from shutter.families.wan.convert import convert  # noqa: E402

# CPU by default (bit-identical checks); SHUTTER_TEST_GPU=1 runs the same tests on Metal
if not __import__("os").environ.get("SHUTTER_TEST_GPU"):
    mx.set_default_device(mx.cpu)

RUNG_IDS = list(WP.RUNGS)
H, W, FRAMES, STEPS = 16, 16, 5, 2        # Wan 2.1: multiples of 16 (VAE stride 8 x patch 2); 4n+1 frames


# ----------------------------------------------------------------------------- helpers


class Guard:
    def __init__(self):
        self.calls = []

    def admit(self, need_bytes, what=""):
        self.calls.append((need_bytes, what))


class LogPool(ModelPool):
    """The engine's pool with a log of loads and releases, and of what else was in memory at each load."""

    def __init__(self):
        super().__init__(Guard())
        self.log: list[tuple[str, str]] = []
        self.held_at_load: dict[str, list[str]] = {}

    def get(self, key, loader, size_bytes, *, keep=True, pinned=False, residency=None):
        def logged():
            self.log.append(("load", key))
            self.held_at_load.setdefault(key, []).append(sorted(self.keys()))
            return loader()
        return super().get(key, logged, size_bytes, keep=keep, pinned=pinned, residency=residency)

    def release(self, key):
        if key in self:
            self.log.append(("release", key))
        super().release(key)

    def loads(self):
        return [k for a, k in self.log if a == "load"]


def random_wan21_vae(seed=0) -> dict:
    """MLX-layout weights of a fresh WanVAE(z_dim=16) with random convolutions (the module starts with zeros)."""
    from mlx.utils import tree_flatten

    from shutter.families.wan.model.vae import WanVAE
    r = np.random.default_rng(seed)
    out = {}
    for k, a in tree_flatten(WanVAE(z_dim=16).parameters()):
        if k.endswith("weight") and a.ndim >= 2:
            a = mx.array((r.standard_normal(a.shape) / np.sqrt(np.prod(a.shape[1:]))).astype(np.float32))
        out[k] = a
    return out


@pytest.fixture(scope="session")
def vae_file(tmp_path_factory) -> Path:
    f = tmp_path_factory.mktemp("vae") / "model.safetensors"
    mx.save_safetensors(str(f), random_wan21_vae())
    return f


def make_build(tmp: Path, vae: Path | None, cfg=F.TINY, dual=False, boundary=0.9, name="build", seed=0) -> Path:
    """Convert a tiny official-layout repo (8-bit) and put a real random VAE in it."""
    src = tmp / f"src-{name}"
    if dual:
        root = _repo(src, cfg=cfg)
        (root / "diffusion_pytorch_model.safetensors").unlink()
        F.save_st(root / "high_noise_model" / "diffusion_pytorch_model.safetensors",
                  F.official_dit_weights(cfg, seed=seed + 11))
        F.save_st(root / "low_noise_model" / "diffusion_pytorch_model.safetensors",
                  F.official_dit_weights(cfg, seed=seed + 12))
    else:
        root = _repo(src, cfg=cfg)
    out = tmp / name
    convert(root, out, bits=8, report=lambda s: None)
    if vae is not None:
        shutil.copyfile(vae, out / "vae" / "model.safetensors")
    if dual:
        man = json.loads((out / "shuttrr.json").read_text())
        man["config"]["boundary"] = boundary
        (out / "shuttrr.json").write_text(json.dumps(man))
    return out


def manifest(build: Path) -> dict:
    return json.loads((build / "shuttrr.json").read_text())


@pytest.fixture
def fixed_prompt(monkeypatch):
    """encode_prompt -> a fixed random [L, 4096] embedding per text (no tokenizer, no umT5)."""
    def enc(self, text):
        seed = sum(map(ord, text)) % 1000
        return mx.array(np.random.default_rng(seed).standard_normal((7, 4096)).astype(np.float32))
    monkeypatch.setattr(WP.WanPipeline, "encode_prompt", enc)


@pytest.fixture
def latents_out(monkeypatch):
    """decode -> the final latents themselves (float32, so equality is exact). The real decode is covered by
    test_every_memory_step_same_frames; this keeps the other tests fast."""
    monkeypatch.setattr(WP.WanPipeline, "decode", lambda self, latents, tiled: np.array(latents.astype(mx.float32)))


def run(build, rung, pool=None, model_id="tiny", keep_warm=True, **kw):
    pool = pool if pool is not None else LogPool()
    man = manifest(build)
    pipe = WP.WanPipeline(build, man)
    src = WP.ModelSource(build, man, model_id, pool, rung)
    s = WP.Settings(prompt=kw.pop("prompt", "a red fox in snow"), height=kw.pop("height", H),
                    width=kw.pop("width", W), frames=kw.pop("frames", FRAMES), steps=kw.pop("steps", STEPS),
                    seed=kw.pop("seed", 7), **kw)
    frames, info = pipe.generate(s, src, keep_warm=keep_warm)
    return frames, info, pool


# ----------------------------------------------------------------------------- tests


def test_align():
    cfg = {"vae_stride": [4, 8, 8]}
    s = WP.align(cfg, WP.Settings(prompt="x", height=470, width=830, frames=80))
    assert (s.height, s.width, s.frames) == (464, 816, 77) and len(s.notes) == 2
    s = WP.align({"vae_stride": [4, 16, 16]}, WP.Settings(prompt="x", height=704, width=1280, frames=121))
    assert (s.height, s.width, s.frames) == (704, 1280, 121) and not s.notes


def test_every_memory_step_same_frames(tmp_path, vae_file, fixed_prompt):
    """(a) the real pipeline, real VAE decode: all five steps give bit-identical frames for one seed."""
    build = make_build(tmp_path, vae_file)
    out = {}
    for rung in RUNG_IDS:
        frames, info, pool = run(build, rung)
        assert frames.shape == (FRAMES, H, W, 3) and frames.dtype == np.uint8
        assert info["expert_loads"] == 1
        out[rung] = frames
        streamed = rung.startswith("stream")
        key = f"wan/tiny/dit/{'stream%d' % WP.RUNGS[rung]['window'] if streamed else 'resident'}"
        assert pool.loads() == [key]
        assert (key in pool) == (rung in ("resident", "split"))      # kept warm only by resident / split
    ref = out["resident"]
    assert len(np.unique(ref)) > 10, "the random VAE should give a non-trivial picture"
    for rung, f in out.items():
        assert np.array_equal(f, ref), f"{rung} differs from resident"
    other, _, _ = run(build, "resident", seed=8)
    assert not np.array_equal(other, ref)                         # the seed matters


def test_warm_pool_and_step_changes(tmp_path, fixed_prompt, latents_out):
    """A warm resident DiT is reused; a request at a streamed step first drops the warm resident copy."""
    build = make_build(tmp_path, None)
    pool = LogPool()
    a, info1, _ = run(build, "resident", pool)
    b, info2, _ = run(build, "resident", pool)
    assert info1["expert_loads"] == 1 and info2["expert_loads"] == 0 and np.array_equal(a, b)
    c, _, _ = run(build, "stream_tight", pool)
    assert np.array_equal(a, c)
    assert pool.held_at_load["wan/tiny/dit/stream2"] == [[]]       # resident copy released before streaming
    assert pool.keys() == []
    # the streamed model's background reader stopped with it (it would otherwise keep read-ahead blocks alive)
    assert not any(t.name == "shutter-stream" and t.is_alive() for t in threading.enumerate())


def test_runtime_lora(tmp_path, fixed_prompt, latents_out):
    """(b) a LoRA changes the output, identically at every step, and switching it off restores the base output
    without reloading the model."""
    build = make_build(tmp_path, None)
    lf = tmp_path / "style.safetensors"
    F.save_st(lf, F.lora_weights())
    lora = [{"path": str(lf), "strength": 0.8}]
    pool = LogPool()
    base, _, _ = run(build, "resident", pool)
    with_lora = {}
    for rung in RUNG_IDS:
        f, info, _ = run(build, rung, pool if rung == "resident" else None, loras=lora)
        with_lora[rung] = f
        assert info["loras"] == ["style.safetensors"]
    assert not np.array_equal(with_lora["resident"], base)
    for rung, f in with_lora.items():
        assert np.array_equal(f, with_lora["resident"]), f"{rung} with LoRA differs from resident"
    again, info, _ = run(build, "resident", pool)
    assert info["expert_loads"] == 0 and np.array_equal(again, base)
    # a LoRA for another Wan width is refused in plain words
    wide = tmp_path / "wide.safetensors"
    F.save_st(wide, F.lora_weights(cfg=dict(F.TINY, dim=128, ffn_dim=256)))
    with pytest.raises(ValueError, match="another Wan size"):
        run(build, "resident", loras=[str(wide)])
    with pytest.raises(ValueError, match="not found"):
        run(build, "resident", loras=[str(tmp_path / "nope.safetensors")])


def test_dual_expert(tmp_path, fixed_prompt, latents_out):
    """(c) A14B-style build: classified as two experts; split / staged hold one at a time; same output."""
    build = make_build(tmp_path, None, dual=True, boundary=0.9, name="dual")
    ck = inspect_ckpt(tmp_path / "src-dual" / "repo-official")
    assert ck.ok and set(ck.parts) >= {"dit_high", "dit_low"} and ck.config["dual_model"], ck.problems
    man = manifest(build)
    assert {"dit_high", "dit_low"} <= set(man["parts"]) and "dit" not in man["parts"]
    # shift 5, 4 steps: timesteps 999, 937 (high noise) then 833, 624 (low noise) with the boundary at 900
    out = {}
    hi, lo = "wan/tiny/dit_high/", "wan/tiny/dit_low/"
    for rung in RUNG_IDS:
        f, info, pool = run(build, rung, steps=4)
        out[rung] = f
        assert info["expert_loads"] == 2, rung
        loads = pool.loads()
        assert [k.split("/")[2] for k in loads] == ["dit_high", "dit_low"], loads
        if rung == "resident":
            assert pool.held_at_load[loads[1]] == [[loads[0]]]      # both experts kept
            assert sorted(pool.keys()) == sorted(loads)
        else:
            assert pool.held_at_load[loads[1]] == [[]], rung        # high released before low was loaded
            assert ("release", loads[0]) in pool.log
            assert pool.log.index(("release", loads[0])) < pool.log.index(("load", loads[1]))
    for rung, f in out.items():
        assert np.array_equal(f, out["resident"]), f"{rung} differs from resident"
    # the experts really differ: an all-low-noise run (boundary above every timestep) gives another output
    man["config"]["boundary"] = 1.0
    (build / "shuttrr.json").write_text(json.dumps(man))
    f_low, info, pool = run(build, "split", steps=4)
    assert [k.split("/")[2] for k in pool.loads()] == ["dit_low"]
    assert not np.array_equal(f_low, out["resident"])
    # split, second request on a warm pool: the low expert left warm is released before the high one loads
    man["config"]["boundary"] = 0.9
    (build / "shuttrr.json").write_text(json.dumps(man))
    pool = LogPool()
    run(build, "split", pool, steps=4)
    assert pool.keys() == [lo + "resident"]
    f2, _, _ = run(build, "split", pool, steps=4)
    assert pool.held_at_load[hi + "resident"][1] == []
    assert np.array_equal(f2, out["resident"])
    # A LoRA on the high-noise expert only, streamed == resident
    lf = tmp_path / "hi.safetensors"
    F.save_st(lf, F.lora_weights())
    lora = [{"path": str(lf), "expert": "high"}]
    a, _, _ = run(build, "resident", steps=4, loras=lora)
    b, _, _ = run(build, "stream_tight", steps=4, loras=lora)
    assert np.array_equal(a, b) and not np.array_equal(a, out["resident"])


def test_ti2v_text_only(tmp_path, fixed_prompt, latents_out):
    """TI2V-5B-style build (48 latent channels, VAE stride 16), text only, steps agree. The Wan 2.2 VAE itself is
    too big to build at random here (its decoder is ~0.6 G parameters), so the decode is replaced."""
    cfg = dict(F.TINY, in_dim=48, out_dim=48)
    src = _repo(tmp_path / "src", cfg=cfg, vae_z=48)
    build = tmp_path / "ti2v"
    man = convert(src, build, bits=8, report=lambda s: None)
    assert man["config"]["model_type"] == "ti2v" and man["config"]["vae_z_dim"] == 48
    a, info, _ = run(build, "resident", height=32, width=32)
    assert a.shape == (48, 2, 2, 2)
    b, _, _ = run(build, "stream_tight", height=32, width=32)
    assert np.array_equal(a, b)
    c, info, _ = run(build, "resident", height=40, width=40)
    assert any("aligned" in n for n in info["notes"]) and info["size"] == [32, 32]


# ----------------------------------------------------------------------------- adapter


def fake_transformers(monkeypatch):
    """A stand-in `transformers` with an AutoTokenizer that maps characters to ids (vocab 300)."""
    mod = types.ModuleType("transformers")

    class Tok:
        def __call__(self, text, max_length=512, padding="max_length", truncation=True, return_tensors="np"):
            ids = [(ord(c) % 290) + 5 for c in text][:max_length]
            n = len(ids)
            ids = ids + [0] * (max_length - n)
            return {"input_ids": np.array([ids]), "attention_mask": np.array([[1] * n + [0] * (max_length - n)])}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            assert Path(path).is_dir()
            return Tok()

    mod.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", mod)


def make_ctx(tmp, pool=None, available=64 << 30):
    from shutter.adapters.base import RunContext
    from shutter.core.cache import ContentCache
    reports = []
    ctx = RunContext(models=pool or LogPool(), cache=ContentCache(tmp / "cache"), work_dir=tmp / "work",
                     report=lambda f, s: reports.append((f, s)), check_cancelled=lambda: None,
                     memory_sample=lambda: {"available": available, "total": 2 * available})
    return ctx, reports


def test_adapter_estimates(tmp_path):
    """(d) estimates: sane numbers, ordered by step. A tiny build's DiT is negligible, so its order is only >=;
    a manifest with A14B sizes (no weights needed to estimate) gives the strict order."""
    from shutter.adapters.wan import WanAdapter
    from shutter.types import GB, JobRequest
    build = make_build(tmp_path, None)
    opts = {"measured_path": str(tmp_path / "measured.json")}
    ad = WanAdapter("tiny", build, opts)
    req = JobRequest(model="tiny", prompt="x", width=W, height=H, frames=FRAMES, steps=STEPS,
                     params={"budget_bytes": 1000 * GB})
    est = ad.estimate(req)
    assert 1 * GB < est.peak < 20 * GB and "step resident" in est.note
    e = {r: v.peak for r, v in ad.rung_estimates(req).items()}
    assert e["resident"] >= e["split"] >= e["staged"] >= e["stream_wide"] >= e["stream_tight"] > 0

    from shutter.families.wan.spec import WanShape
    big = tmp_path / "a14b"
    big.mkdir()
    cfg = WanShape(5120, 16, 16, 13824, 40, dual=True).to_config("wan2.2-t2v-a14b")
    blk, rest = 370 << 20, 600 << 20
    part = {"role": "denoiser", "files": [], "n_blocks": 40, "block_bytes": blk, "rest_bytes": rest,
            "total_bytes": rest + 40 * blk, "quant": {"bits": 8, "group_size": 64}}
    (big / "shuttrr.json").write_text(json.dumps({
        "family": "wan", "variant": "wan2.2-t2v-a14b", "label": "Wan 2.2 T2V A14B", "config": cfg,
        "parts": {"dit_high": part, "dit_low": part, "text_encoder": {"files": []}, "vae": {"files": []},
                  "tokenizer": {"files": []}}}))
    ad = WanAdapter("a14b", big, opts)
    req = JobRequest(model="a14b", prompt="x", width=832, height=480, frames=81, params={"budget_bytes": 1000 * GB})
    e = {r: v.peak for r, v in ad.rung_estimates(req).items()}
    assert e["resident"] >= e["split"] >= e["staged"] > e["stream_wide"] > e["stream_tight"], e
    assert e["resident"] > e["split"]                              # two experts vs one
    assert 30 * GB < e["resident"] < 45 * GB and 3 * GB < e["stream_tight"] < 15 * GB, e
    # fast mode steps down to fit the budget; low_memory picks by the slider
    pick = ad.choose_rung(JobRequest(model="a14b", prompt="x", width=832, height=480, frames=81,
                                     params={"budget_bytes": int(e["stream_wide"]) + 1}))
    assert pick["rung"] == "stream_wide" and pick["stepped_down_from"] == "resident"
    pick = ad.choose_rung(JobRequest(model="a14b", prompt="x", params={"mode": "low_memory", "efficiency": 1.0,
                                                                       "budget_bytes": 1000 * GB}))
    assert pick["rung"] == "stream_tight"
    # a measured peak wins over the formula
    from shutter.core import modes as M
    M.record_measurement("a14b", "480x832", "staged", peak_bytes=int(9.5 * GB), seconds=100, identical=True,
                         path=opts["measured_path"], extra={"frames": 81})
    assert ad.rung_estimates(req)["staged"].peak == int(9.5 * GB)


def test_adapter_run(tmp_path, vae_file, monkeypatch):
    """(d) adapter.run with a RunContext: real prompt encode (tiny umT5, cached), real decode, mp4 written (or the
    writer stubbed when there is no ffmpeg)."""
    from shutter.adapters import longcat
    from shutter.adapters.wan import WanAdapter
    from shutter.types import Job, JobRequest
    fake_transformers(monkeypatch)
    build = make_build(tmp_path, vae_file, name="t64")
    # the converter refuses a text width other than umT5-XXL's 4096, so swap in a DiT whose text input matches the
    # tiny umT5 (width 64), converted the same way
    from shutter.families.wan.convert import convert_dit
    f64 = tmp_path / "dit64.safetensors"
    F.save_st(f64, F.official_dit_weights(dict(F.TINY, text_dim=64)))
    shutil.rmtree(build / "dit")
    man = manifest(build)
    man["parts"]["dit"] = convert_dit([str(f64)], "official", build / "dit", bits=8, group=64, report=lambda s: None)
    man["config"]["text_dim"] = 64
    (build / "shuttrr.json").write_text(json.dumps(man))
    written = {}
    if shutil.which("ffmpeg") is None:
        def fake_write(frames, audio, out, fps=16, crf=14):
            written.update(frames=frames, fps=fps)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"mp4")
        monkeypatch.setattr(longcat, "write_mp4", fake_write)
    ad = WanAdapter("tiny64", build, {"measured_path": str(tmp_path / "measured.json"), "stream_nocache": False})
    ctx, reports = make_ctx(tmp_path)
    req = JobRequest(model="tiny64", prompt="a red fox", width=W, height=H, frames=FRAMES, steps=STEPS, seed=3,
                     params={"mode": "low_memory", "efficiency": 1.0})
    res = ad.run(Job(request=req), ctx)
    assert Path(res["output"]).exists() and res["rung"] == "stream_tight" and res["identical"] is True
    assert res["frames"] == FRAMES and res["size"] == [W, H] and res["size_key"] == f"{H}x{W}"
    assert reports[-1] == (1.0, "done") and any("denoising" in s for _, s in reports)
    assert [r["stage"] for r in res["stages"]] == ["prep", "prompt", "denoise", "release", "decode", "write"]
    st = ctx.cache.stats()
    assert st["namespaces"]["wan-umt5"]["entries"] == 2 and st["misses"] == 2      # prompt + negative prompt
    if written:
        assert written["frames"].shape == (FRAMES, H, W, 3) and written["fps"] == 16
    # the same request at resident and at stream_tight: same frames; the prompt now comes from the cache
    frames = {}

    def keep(frames_, audio, out, fps=16, crf=14):
        frames[Path(out).stem] = frames_
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"mp4")
    monkeypatch.setattr(longcat, "write_mp4", keep)
    for i, eff in enumerate((0.0, 1.0)):
        r = JobRequest(model="tiny64", prompt="a red fox", width=W, height=H, frames=FRAMES, steps=STEPS, seed=3,
                       output=str(tmp_path / f"out{i}.mp4"), params={"mode": "low_memory", "efficiency": eff})
        ad.run(Job(request=r), ctx)
    assert np.array_equal(frames["out0"], frames["out1"])
    assert ctx.cache.stats()["hits"] == 4
    # not enough memory for even the lowest step: refused before loading anything
    from shutter.types import MemoryRefused
    ctx2, _ = make_ctx(tmp_path, available=1 << 30)
    with pytest.raises(MemoryRefused):
        ad.run(Job(request=req), ctx2)
    assert ctx2.models.log == []


def test_adapter_small_mac_defaults(tmp_path):
    """The model entry's options set the default size / frames / steps (catalog.preset_for gives Macs under
    24 GB 320x512 x 33 frames); an explicit request still wins."""
    from shutter.adapters.wan import WanAdapter
    from shutter.types import JobRequest
    build = make_build(tmp_path, None)
    ad = WanAdapter("tiny", build, {"default_size": [320, 512], "default_frames": 33, "default_steps": 20})
    s = ad.settings_for(JobRequest(model="tiny", prompt="x"))
    assert (s.height, s.width, s.frames, s.steps) == (320, 512, 33, 20)
    s = ad.settings_for(JobRequest(model="tiny", prompt="x", width=832, height=480, frames=17, steps=12))
    assert (s.height, s.width, s.frames, s.steps) == (480, 832, 17, 12)
    plain = WanAdapter("tiny", build, {}).settings_for(JobRequest(model="tiny", prompt="x"))
    assert (plain.height, plain.width) != (320, 512) or plain.frames != 33
