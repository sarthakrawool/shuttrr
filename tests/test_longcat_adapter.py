"""LongCat adapter: render rule, estimates, planning, and run() against a fake model tree (and, where it runs on
CPU without weights, the real patched tree read-only from /home/claude/ref/shutter-ref)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from shutter.adapters import longcat as L
from shutter.adapters.base import RunContext
from shutter.addons.reels import Framing
from shutter.types import GB, Cancelled, Job, JobRequest

REAL_CODE = Path("/home/claude/ref/shutter-ref/longcat-mlx-code")


def req(framing="auto", **params):
    return JobRequest(model="longcat-avatar-1.6-8bit", prompt="a man talks", image="img.png", audio="vo.wav",
                      framing=Framing(framing), params=params)


# ----------------------------------------------------------------------------- resolve_render


def test_small_is_480p_full_frame_and_ignores_crop():
    p = L.resolve_render(req("small", crop_box="832:480:100:0", room_lock=True, person_box="10,20"), (1920, 1080))
    assert (p.height, p.width) == (480, 832)
    assert p.crop is None and not p.room_lock
    assert any("crop_box ignored" in n for n in p.notes)
    assert any("room_lock ignored" in n for n in p.notes)
    assert p.zoom_cancel == 0.0
    assert p.upscale_to == (1920, 1080)
    assert p.tokens == 37_440


def test_small_portrait_bucket():
    p = L.resolve_render(req("small"), (1080, 1920))
    assert (p.height, p.width) == (832, 480)
    assert p.upscale_to == (1080, 1920)


@pytest.mark.parametrize("framing", ["full", "auto"])
def test_full_is_704x1312_with_auto_zoom_cancel(framing):
    p = L.resolve_render(req(framing), (1920, 1080))
    assert (p.height, p.width) == (704, 1312)
    assert p.zoom_cancel == pytest.approx(0.007)
    assert p.upscale_to == (1920, 1080)
    assert p.vae_tile == 64 and p.steps == 8
    assert p.tokens == 86_592


def test_full_portrait_and_square():
    p = L.resolve_render(req("full"), (1080, 1920))
    assert (p.height, p.width) == (1312, 704)
    q = L.resolve_render(req("full"), (1000, 1000))
    assert q.height % 32 == 0 and q.width % 32 == 0
    assert abs(q.height * q.width - 704 * 1312) / (704 * 1312) < 0.05


def test_crop_box_forms_and_zoom_off():
    p = L.resolve_render(req("full", crop_box="832:704:240:0"), (1920, 1080))
    assert p.crop == "240,1072" and (p.height, p.width) == (704, 832)
    assert p.zoom_cancel == 0.0
    q = L.resolve_render(req("full", crop_box="250,1070"), (1920, 1080))
    assert q.crop == "240,1072" and any("snapped" in n for n in q.notes)
    r = L.resolve_render(req("full", crop_box="auto"), (1920, 1080))
    assert r.crop == "auto" and r.width == 832


def test_room_lock_rules():
    with pytest.raises(ValueError):
        L.resolve_render(req("full", crop_box="240,1072", room_lock=True), (1920, 1080))
    with pytest.raises(ValueError):
        L.resolve_render(req("full", crop_box="240,1072", room_lock=True, person_box="100,500"), (1920, 1080))
    p = L.resolve_render(req("full", crop_box="240,1072", room_lock=True, person_box="410,883"), (1920, 1080))
    assert p.room_lock and p.person_box == "410,883" and p.zoom_cancel == 0.0
    q = L.resolve_render(req("full", room_lock=True, person_box="410,883"), (1920, 1080))
    assert not q.room_lock and any("crop box only" in n for n in q.notes)
    assert q.zoom_cancel == pytest.approx(0.007)


def test_zoom_and_upscale_overrides():
    assert L.resolve_render(req("full", zoom_cancel="off"), None).zoom_cancel == 0
    assert L.resolve_render(req("full", zoom_cancel=0.004), None).zoom_cancel == 0.004
    p = L.resolve_render(req("full", upscale_to="off"), None)
    assert p.upscale_to == (1312, 704)        # zoom cancel still needs its one resample
    q = L.resolve_render(req("full", upscale_to="off", zoom_cancel="off"), None)
    assert q.upscale_to is None
    r = L.resolve_render(req("full", size="480x832"), None)
    assert (r.height, r.width) == (480, 832) and r.zoom_cancel == 0.0   # 480p-class: no auto zoom cancel
    s = L.resolve_render(req("full", size="500x830"), None)
    assert (s.height, s.width) == (512, 832)
    assert L.resolve_render(req("full", upscale_to="3840x2160"), None).upscale_to == (3840, 2160)


def test_steps_and_bf16_notes():
    p = L.resolve_render(req("full", steps=6, vae_bf16=True), None)
    assert p.steps == 6 and p.vae_bf16
    assert any("not lossless" in n for n in p.notes) and any("NOT bit-exact" in n for n in p.notes)
    d = p.to_dict()
    assert d["upscale_to"] == [1920, 1080] and d["steps"] == 6


# ----------------------------------------------------------------------------- segments, seeds, plan


def test_segment_layout_matches_upstream():
    assert len(L.segment_layout(3.72)) == 1
    assert len(L.segment_layout(93 / 25)) == 1
    assert len(L.segment_layout(94 / 25)) == 2
    lay = L.segment_layout(10.0)
    assert len(lay) == 3                          # ceil((250 - 93) / 80) + 1
    assert [s for s, _, _ in lay] == [0, 93, 173]
    assert sum(k for _, k, _ in lay) == 250
    assert len(L.segment_layout(10.0, seg_frames=133)) == 2   # ceil((250-133)/120)+1


def test_plan_probes_duration_and_derives_seeds():
    calls = []

    def probe(path):
        calls.append(path)
        return 10.0

    ad = L.LongCatAvatarAdapter("lc", Path("/nonexistent"), duration_probe=probe)
    r = req("full")
    r.seed = 42
    shots = ad.plan(r)
    assert calls == ["vo.wav"]
    assert [s.seed for s in shots] == [42, 42 + 7919, 42 + 2 * 7919]
    assert {s.chain for s in shots} == {0}
    assert [s.index for s in shots] == [0, 1, 2]
    assert shots[1].params["cond_frames"] == 13 and shots[0].params["cond_frames"] == 0
    assert sum(s.duration_s for s in shots) == pytest.approx(10.0)
    assert shots[1].start_s == pytest.approx(93 / 25)
    r.duration_s = 3.0
    assert len(ad.plan(r)) == 1 and len(calls) == 1


# ----------------------------------------------------------------------------- estimates


def test_memory_estimates_match_measurements():
    full = L.estimate_memory(L.resolve_render(req("full"), None))
    assert full.peak / GB == pytest.approx(38.0, abs=0.3)          # measured process footprint
    small = L.estimate_memory(L.resolve_render(req("small"), None))
    assert small.peak / GB == pytest.approx(33.8 + 1.7, abs=0.3)
    lock = L.estimate_memory(L.resolve_render(req("full", crop_box="240,1072", room_lock=True,
                                                  person_box="410,883"), None))
    nolock = L.estimate_memory(L.resolve_render(req("full", crop_box="240,1072"), None))
    assert (lock.peak - nolock.peak) / GB == pytest.approx(5.0, abs=0.01)
    low = L.estimate_memory(L.resolve_render(req("full"), None, {"low_memory": True}))
    assert (full.peak - low.peak) / GB == pytest.approx(2.7, abs=0.01)
    assert full.steady == int(L.DEFAULT_RESIDENT_GB * GB)
    shifted = L.estimate_memory(L.resolve_render(req("full"), None), resident_bytes=int(20 * GB))
    assert (shifted.peak - full.peak) / GB == pytest.approx(2.0, abs=0.01)


def test_time_estimates_match_measurements():
    full = L.estimate_seconds(L.resolve_render(req("full"), None))
    assert full["seconds_per_step"] == pytest.approx(164, abs=0.5)
    assert full["decode_s"] == pytest.approx(172, abs=0.5)
    assert full["total_s"] == pytest.approx(1513, rel=0.03)
    small = L.estimate_seconds(L.resolve_render(req("small"), None))
    assert small["total_s"] == pytest.approx(422, rel=0.02)
    crop = L.estimate_seconds(L.resolve_render(req("full", crop_box="240,1072"), None))
    assert crop["total_s"] == pytest.approx(758, rel=0.05)
    lock = L.estimate_seconds(L.resolve_render(req("full", crop_box="240,1072", room_lock=True,
                                                   person_box="410,883"), None))
    assert lock["total_s"] - crop["total_s"] == pytest.approx(52, abs=0.2)
    assert L.estimate_seconds(L.resolve_render(req("small"), None), 3)["total_s"] > 3 * 400


def test_adapter_estimate_note():
    ad = L.LongCatAvatarAdapter("lc", Path("/nonexistent"))
    r = req("full")
    r.duration_s = 10.0
    e = ad.estimate(r)
    assert "3 segment" in e.note and e.peak > e.steady > 0


def test_import_without_mlx_or_model_code():
    code = ("import sys; sys.modules['mlx'] = None; sys.modules['mlx.core'] = None; "
            "import shutter.adapters.longcat as L; assert 'longcat_video_avatar' not in sys.modules; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


# ----------------------------------------------------------------------------- fake model tree


FAKE_GENERATION = '''
import math, os, pathlib
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

FPS = 25
NUM_FRAMES = 93
NUM_COND_FRAMES = 13
EVENTS = []          # (what, payload) for the tests

def load_vae(components, fp32=False):
    EVENTS.append(("load_vae", fp32))
    return {"fp32": fp32}

def load_image(path, H, W):
    return np.zeros((1, 3, 1, H, W), np.float32) + 0.5

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
    audio_scale: float = 1.0

class AvatarGenerator:
    def __init__(self, paths, cache_dir=None, vae_tile=0, vae_overlap=12, compile=True, vae_fp32=None):
        self.paths, self.cache_dir, self.vae_tile, self.compile = paths, cache_dir, vae_tile, compile
        self.vae = load_vae(paths.components, fp32=(vae_tile > 0) if vae_fp32 is None else vae_fp32)
        self.dit = None
        EVENTS.append(("init", vae_tile, vae_fp32))

    def _load_dit(self):
        if self.dit is not None:
            return
        self.dit = object()
        self.lora_swap = os.environ.get("LC_LORA_SWAP", "1") == "1"
        self.block_split = os.environ.get("LC_BLOCK_SPLIT", "0") == "1"
        EVENTS.append(("load_dit", self.lora_swap, self.block_split))

    def _set_loras_for_step(self, i, st):
        EVENTS.append(("lora", i))

    def _denoise(self, latents, num_cond, text, text_mask, neg, neg_mask, audio, st, num_ref_latents=None, log=print):
        for i in range(st.steps):
            self._set_loras_for_step(i, st)
            log(f"    step {i + 1}/{st.steps}")
        return latents

    def generate(self, audio_path, prompt, st, image=None, start_frames=None, duration=None, on_segment=None,
                 segment_check=None, log=print):
        H, W = st.size
        NF = int(os.environ.get("LC_SEG_FRAMES", NUM_FRAMES))
        step = NF - NUM_COND_FRAMES
        n_seg = max(1, math.ceil((duration * FPS - NF) / step) + 1)
        from longcat_video_avatar.models import attention
        from longcat_video_avatar import fast
        EVENTS.append(("generate", {k: v for k, v in os.environ.items() if k.startswith("LC_")}, st.seed, st.steps,
                       (H, W), str(image), attention.SDPA_HEAD_CHUNK, fast.FFN_CHUNKS,
                       getattr(self, "_anchor_parts", None) is not None))
        self._load_dit()
        if os.environ.get("LC_BG_ANCHOR") == "1":
            self._anchor_parts = ("z_bg", "mask")      # like the patched tree: set, never cleared
        out = []
        for s in range(n_seg):
            log(f"  segment {s + 1}/{n_seg}")
            self._denoise(None, 1 if s == 0 else 5, None, None, None, None, None, st)
            n = NF if s == 0 else step
            fr = np.zeros((n, H, W, 3), np.uint8)
            fr[..., 0] = (np.arange(n) % 250)[:, None, None]
            fr[..., 1] = 100
            out.append(fr)
            if on_segment:
                on_segment(np.concatenate(out), s + 1)
        frames = np.concatenate(out)[: int(round(duration * FPS))]
        return frames, {"height": H, "width": W, "segments": n_seg}
'''

FAKE_ROI = '''
import numpy as np

def plate_uint8(image, H, W):
    p = np.zeros((H, W, 3), np.uint8)
    p[..., 2] = 200
    return p

def parse_box(spec, plate_png, W):
    if spec == "auto":
        return (16, 16 + 64)
    a, b = (int(v) for v in spec.split(","))
    return a, b

def composite(crop_frames, plate, x0, feather=32):
    F, H, w, _ = crop_frames.shape
    out = np.broadcast_to(plate, (F,) + plate.shape).copy()
    out[:, :, x0:x0 + w] = crop_frames
    return out
'''


@pytest.fixture
def fake_code(tmp_path):
    root = tmp_path / "code"
    pkg = root / "longcat_video_avatar"
    (pkg / "models").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "models" / "__init__.py").write_text("")
    (pkg / "models" / "attention.py").write_text(
        "import os\nSDPA_HEAD_CHUNK = int(os.environ.get('LC_SDPA_HEADS', '1'))\n")
    (pkg / "fast.py").write_text("import os\nFFN_CHUNKS = int(os.environ.get('LC_FFN_CHUNKS', '1'))\n")
    (pkg / "generation.py").write_text(textwrap.dedent(FAKE_GENERATION))
    (pkg / "roi.py").write_text(textwrap.dedent(FAKE_ROI))
    yield root
    for name in [n for n in sys.modules if n.startswith("longcat_video_avatar")]:
        del sys.modules[name]
    if str(root.resolve()) in sys.path:
        sys.path.remove(str(root.resolve()))


class FakePool:
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
        for k in [k for k in self.items if k.startswith(prefix)]:
            self.release(k)


def make_ctx(tmp_path, pool=None, cancel_at=None):
    reports = []

    def report(p, stage):
        reports.append((p, stage))

    def check():
        if cancel_at is not None and sum("denoise" in s for _, s in reports) >= cancel_at:
            raise Cancelled()

    ctx = RunContext(models=pool or FakePool(), cache=None, work_dir=tmp_path / "work", report=report,
                     check_cancelled=check, memory_sample=lambda: {})
    return ctx, reports


@pytest.fixture
def media(tmp_path):
    img = tmp_path / "img.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x180", "-frames:v",
                    "1", str(img)], check=True)
    wav = tmp_path / "vo.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=220:r=48000:d=4.2",
                    str(wav)], check=True)
    return img, wav


def gen_events():
    return sys.modules["longcat_video_avatar.generation"].EVENTS


def job_for(media, tmp_path, framing="full", **params):
    img, wav = media
    r = JobRequest(model="lc", prompt="hi", image=str(img), audio=str(wav), framing=Framing(framing),
                   output=str(tmp_path / "out" / "clip.mp4"), seed=7, params=params)
    return Job(request=r)


def test_run_with_fake_tree(fake_code, media, tmp_path):
    ad = L.LongCatAvatarAdapter("lc", tmp_path / "model", {"code_dir": str(fake_code)})
    job = job_for(media, tmp_path, size="64x128", upscale_to="256x128")
    ctx, reports = make_ctx(tmp_path)
    res = ad.run(job, ctx)
    # 4.2 s of audio -> 105 frames -> 2 segments
    assert res["segments"] == 2 and res["frames"] == 105
    steps = [s for _, s in reports if "denoise" in s and not s.endswith(" 0/8")]
    assert len(steps) == 2 * 8
    assert "segment 2/2 denoise 8/8" in steps
    assert [p for p, _ in reports] == sorted(p for p, _ in reports)
    assert reports[-1] == (1.0, "done")
    assert Path(res["output"]).exists() and Path(res["raw_output"]).exists()
    assert res["size"] == [256, 128] and res["render_size"] == [128, 64]
    assert res["plan"]["zoom_cancel"] == 0.0           # 64x128 is not 720p-class
    assert res["seconds_per_step"] is not None
    gen = [e for e in gen_events() if e[0] == "generate"][0]
    env, seed, nsteps, hw = gen[1], gen[2], gen[3], gen[4]
    assert seed == 7 and nsteps == 8 and hw == (64, 128)       # base seed: the tree adds 7919 * segment itself
    assert env["LC_SDPA_HEADS"] == "1" and env["LC_LORA_SWAP"] == "1" and env["LC_BLOCK_SPLIT"] == "0"
    assert env["LC_SEG_FRAMES"] == "93" and "LC_BG_ANCHOR" not in env
    assert not any(k.startswith("LC_BG") for k in os.environ)
    # audio muxed
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "csv=p=0", res["output"]],
                           capture_output=True, text=True).stdout
    assert "audio" in probe


def test_run_cancel_between_steps(fake_code, media, tmp_path):
    ad = L.LongCatAvatarAdapter("lc", tmp_path / "model", {"code_dir": str(fake_code)})
    job = job_for(media, tmp_path, size="64x128", upscale_to="off", zoom_cancel="off")
    ctx, reports = make_ctx(tmp_path, cancel_at=5)
    with pytest.raises(Cancelled):
        ad.run(job, ctx)
    assert sum("denoise" in s for _, s in reports) == 5
    assert not Path(job.request.output).exists()
    assert not any(k.startswith("LC_BG") for k in os.environ)


def test_warm_generator_reused_and_rebuilt_on_switch_change(fake_code, media, tmp_path):
    pool = FakePool()
    ad = L.LongCatAvatarAdapter("lc", tmp_path / "model", {"code_dir": str(fake_code)})
    for _ in range(2):
        ctx, _ = make_ctx(tmp_path, pool)
        ad.run(job_for(media, tmp_path, size="64x128", upscale_to="off", zoom_cancel="off"), ctx)
    assert len(pool.loads) == 1 and pool.loads[0][0] == "longcat/lc/generator"
    low = L.LongCatAvatarAdapter("lc", tmp_path / "model", {"code_dir": str(fake_code), "low_memory": True,
                                                            "ffn_chunks": 3})
    ctx, _ = make_ctx(tmp_path, pool)
    low.run(job_for(media, tmp_path, size="64x128", upscale_to="off", zoom_cancel="off"), ctx)
    assert len(pool.loads) == 2 and pool.released == ["longcat/lc/generator"]
    ev = gen_events()
    assert [e for e in ev if e[0] == "load_dit"][-1] == ("load_dit", True, True)
    last = [e for e in ev if e[0] == "generate"][-1]
    assert last[1]["LC_FFN_CHUNKS"] == "3" and last[7] == 3     # import-time global overwritten too
    # vae precision switch reloads only the VAE
    ctx, _ = make_ctx(tmp_path, pool)
    low.run(job_for(media, tmp_path, size="64x128", upscale_to="off", zoom_cancel="off", vae_bf16=True), ctx)
    assert len(pool.loads) == 2 and ("load_vae", False) in ev
    # keep_warm False releases after the job
    cold = L.LongCatAvatarAdapter("lc", tmp_path / "model", {"code_dir": str(fake_code), "keep_warm": False,
                                                             "low_memory": True, "ffn_chunks": 3})
    ctx, _ = make_ctx(tmp_path, pool)
    cold.run(job_for(media, tmp_path, size="64x128", upscale_to="off", zoom_cancel="off", vae_bf16=True), ctx)
    assert "longcat/lc/generator" not in pool.items


def test_crop_room_lock_env_and_stale_anchor_reset(fake_code, media, tmp_path):
    pool = FakePool()
    ad = L.LongCatAvatarAdapter("lc", tmp_path / "model", {"code_dir": str(fake_code)})
    ctx, _ = make_ctx(tmp_path, pool)
    job = job_for(media, tmp_path, size="128x256", crop_box="64,192", room_lock=True, person_box="96,160",
                  upscale_to="off")
    res = ad.run(job, ctx)
    gen = [e for e in gen_events() if e[0] == "generate"][-1]
    env = gen[1]
    assert env["LC_BG_ANCHOR"] == "1" and env["LC_BG_PERSON"] == "32,96"      # crop pixels
    assert env["LC_BG_MARGIN"] == "24" and env["LC_BG_FEATHER"] == "4"
    assert gen[4] == (128, 128) and gen[5].endswith("roi.png")
    assert res["crop_box"] == [64, 192] and res["person_box_crop_px"] == [32, 96]
    assert res["render_size"] == [256, 128]                                    # composited on the plate
    assert Path(res["crop_output"]).exists()
    assert any("first segment only" in n for n in res["plan"]["notes"])
    assert not any(k.startswith("LC_BG") for k in os.environ)
    # next job on the warm generator: no stale anchor
    ctx, _ = make_ctx(tmp_path, pool)
    ad.run(job_for(media, tmp_path, size="64x128", upscale_to="off", zoom_cancel="off"), ctx)
    last = [e for e in gen_events() if e[0] == "generate"][-1]
    assert last[8] is False and "LC_BG_ANCHOR" not in last[1]


def test_switch_code_dir_purges_modules(fake_code, tmp_path):
    mod = L.import_generation(fake_code)
    assert str(fake_code.resolve()) in mod.__file__
    other = tmp_path / "other"
    (other / "longcat_video_avatar").mkdir(parents=True)
    (other / "longcat_video_avatar" / "__init__.py").write_text("")
    (other / "longcat_video_avatar" / "generation.py").write_text("WHO = 'other'\n")
    try:
        mod2 = L.import_generation(other)
        assert mod2.WHO == "other"
    finally:
        sys.path.remove(str(other.resolve()))


# ----------------------------------------------------------------------------- the real patched tree (CPU, no weights)


@pytest.fixture
def real_gen():
    if not REAL_CODE.is_dir():
        pytest.skip("patched LongCat tree not available")
    pytest.importorskip("mlx.core")
    mod = L.import_generation(REAL_CODE)
    yield mod
    for name in [n for n in sys.modules if n.startswith("longcat_video_avatar")]:
        del sys.modules[name]
    if str(REAL_CODE.resolve()) in sys.path:
        sys.path.remove(str(REAL_CODE.resolve()))


def _bare_real_generator(mod, hook):
    import mlx.core as mx

    cls = L.make_generator_class(mod)
    g = object.__new__(cls)            # no weights: skip __init__, give it what _denoise touches
    g.paths = mod.ModelPaths(components=Path("."), dit=Path("."), loras={})
    g.lora_keys = []
    g.dit = lambda latents, ts, *a, **k: mx.zeros_like(latents)
    g._anchor = None
    g._shutter_hook = hook
    return g


def test_real_tree_signatures(real_gen):
    assert L._accepts(real_gen.AvatarGenerator.__init__, "vae_fp32")
    assert "steps" in {f.name for f in __import__("dataclasses").fields(real_gen.Settings)}
    import inspect

    src = inspect.getsource(real_gen)
    for key in ("LC_BG_ANCHOR", "LC_BG_PERSON", "LC_BG_MARGIN", "LC_BG_FEATHER", "LC_SEG_FRAMES", "LC_LORA_SWAP",
                "LC_BLOCK_SPLIT", "LC_FIRST_KV"):
        assert key in src
    assert real_gen.NUM_FRAMES == L.SEG_FRAMES and real_gen.NUM_COND_FRAMES == L.COND_FRAMES
    assert "7919 * seg + 100003 * attempt" in src       # segment_seed mirrors this


def test_real_denoise_hooks_progress_and_cancel(real_gen, tmp_path):
    import mlx.core as mx

    ctx, reports = make_ctx(tmp_path)
    hook = L.StepHook(ctx, n_segments=1, steps=8)
    g = _bare_real_generator(real_gen, hook)
    lat = mx.random.normal((1, 16, 3, 4, 4))
    out = g._denoise(lat, 1, None, None, None, None, None, real_gen.Settings(steps=8))
    stages = [s for _, s in reports]
    assert stages[0] == "segment 1/1 denoise 0/8"
    assert [s for s in stages if "denoise" in s][1:] == [f"segment 1/1 denoise {i}/8" for i in range(1, 9)]
    assert stages[-1] == "segment 1/1 decode"
    assert len(hook.step_times) == 8
    assert out.shape == lat.shape

    ctx2, reports2 = make_ctx(tmp_path, cancel_at=4)
    g2 = _bare_real_generator(real_gen, L.StepHook(ctx2, 1, 8))
    with pytest.raises(Cancelled):
        g2._denoise(lat, 1, None, None, None, None, None, real_gen.Settings(steps=8))
    assert sum("denoise" in s for _, s in reports2) == 4


def test_real_import_time_switches(real_gen):
    import importlib

    att = sys.modules["longcat_video_avatar.models.attention"]
    # fast.py is imported lazily by _load_dit, so on a first job the env var alone decides; once imported (warm
    # process) only the module global can change it
    assert "longcat_video_avatar.fast" not in sys.modules
    fast = importlib.import_module("longcat_video_avatar.fast")
    old = (att.SDPA_HEAD_CHUNK, fast.FFN_CHUNKS)
    try:
        done = L.apply_import_time_switches({"LC_SDPA_HEADS": "0", "LC_FFN_CHUNKS": "4"})
        assert att.SDPA_HEAD_CHUNK == 0 and fast.FFN_CHUNKS == 4 and len(done) == 2
    finally:
        att.SDPA_HEAD_CHUNK, fast.FFN_CHUNKS = old


def test_real_roi_accepts_what_the_adapter_passes(real_gen, tmp_path):
    import importlib

    roi = importlib.import_module("longcat_video_avatar.roi")
    p = L.resolve_render(req("full", crop_box="832:704:240:0"), (1920, 1080))
    assert roi.parse_box(p.crop, tmp_path / "unused.png", p.plate_width) == (240, 1072)
    plate = np.full((64, 256, 3), 50, np.uint8)
    crop = np.full((3, 64, 128, 3), 200, np.uint8)
    out = roi.composite(crop, plate, 64, 32)
    assert out.shape == (3, 64, 256, 3) and out[0, 0, 0, 0] == 50 and out[0, 0, 128, 0] == 200
