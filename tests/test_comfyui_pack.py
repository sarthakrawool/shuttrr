"""The ComfyUI node pack's client, conversions and runner, against the Python reference server with the
dummy adapters. ComfyUI itself is not installed: the last tests drive nodes.py through a hand-written
stand-in for `comfy_api.latest` / `comfy.*`, which checks the wiring only, not ComfyUI's real behaviour."""

import asyncio
import importlib
import io as _io
import shutil
import socket
import sys
import threading
import time
import types
import wave
from pathlib import Path

import numpy as np
import pytest
import uvicorn

from shutter.core.config import Config
from shutter.core.engine import Engine
from shutter.core.memory import GB, FakeProbe
from shutter.server.api import create_app

PACK = Path(__file__).resolve().parents[1] / "integrations" / "comfyui"
sys.path.insert(0, str(PACK))

from shutter_comfy import convert, runner  # noqa: E402
from shutter_comfy.client import ShutterClient, ShutterError  # noqa: E402

FAST = {"steps": 2, "step_s": 0}
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not on PATH")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live(tmp_path):
    cfg = Config(home=str(tmp_path / "home"), memory_poll_s=0.01, busy_process_pattern=None,
                 models={"dummy-image": {"adapter": "dummy-image"}})
    eng = Engine(cfg, probe=FakeProbe(total=64 * GB, available=48 * GB), protected=[], watch_interval_s=0,
                 sample_interval_s=0.01)
    eng.start()
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(eng), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    t0 = time.monotonic()
    while not server.started:
        assert time.monotonic() - t0 < 10
        time.sleep(0.01)
    yield ShutterClient(f"http://127.0.0.1:{port}"), eng
    server.should_exit = True
    t.join(5)
    eng.stop(5)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- conversions (no server)

def test_image_round_trip_and_stdlib_png_decoder(monkeypatch):
    rng = np.random.default_rng(0)
    u8 = rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)
    img = convert.frames_to_image(u8)
    assert img.shape == (1, 5, 7, 3) and img.dtype == np.float32 and 0 <= img.min() and img.max() <= 1
    png = convert.image_to_png(img)
    assert np.array_equal(convert.decode_png(png), u8)
    # Pillow writes adaptive filters (sub/up/average/paeth); the stdlib fallback must read them all
    PIL = pytest.importorskip("PIL.Image")
    yy, xx = np.mgrid[0:20, 0:30]
    smooth = np.stack([xx * 8, yy * 12, (xx + yy) * 5], -1).astype(np.uint8)
    pil_pngs = []
    for a in (u8, smooth, np.dstack([smooth, smooth[..., :1]])):
        b = _io.BytesIO()
        PIL.fromarray(a).save(b, format="PNG", optimize=True)
        pil_pngs.append((a, b.getvalue()))
    from shutter.adapters.dummy import png_bytes
    monkeypatch.setitem(sys.modules, "PIL", None)
    assert np.array_equal(convert.decode_png(png), u8)
    assert np.array_equal(convert.decode_png(png_bytes(smooth)), smooth)
    for a, b in pil_pngs:
        assert np.array_equal(convert.decode_png(b), a)
    assert convert.frames_to_image(np.zeros((2, 3, 4, 4), np.uint8)).shape == (2, 3, 4, 3)
    with pytest.raises(ValueError):
        convert.frames_to_image(np.zeros((3, 4), np.uint8))


def _png_with_every_filter(a):
    """A PNG whose rows cycle through filter types 0..4 (encoded by hand)."""
    import struct
    import zlib
    h, w, c = a.shape
    rows, prev = [], np.zeros(w * c, np.int32)
    for y in range(h):
        cur, f = a[y].reshape(-1).astype(np.int32), y % 5
        left = np.concatenate([np.zeros(c, np.int32), cur[:-c]])
        ul = np.concatenate([np.zeros(c, np.int32), prev[:-c]])
        if f == 0:
            pred = np.zeros_like(cur)
        elif f == 1:
            pred = left
        elif f == 2:
            pred = prev
        elif f == 3:
            pred = (left + prev) // 2
        else:
            pa, pb, pc = np.abs(prev - ul), np.abs(left - ul), np.abs(left + prev - 2 * ul)
            pred = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, ul))
        rows.append(bytes([f]) + ((cur - pred) & 0xFF).astype(np.uint8).tobytes())
        prev = cur

    def chunk(k, b):
        return struct.pack(">I", len(b)) + k + b + struct.pack(">I", zlib.crc32(k + b) & 0xFFFFFFFF)

    return (convert.PNG_SIG + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def test_stdlib_png_decoder_all_filters(monkeypatch):
    a = np.random.default_rng(1).integers(0, 256, (10, 6, 3), dtype=np.uint8)
    png = _png_with_every_filter(a)
    PIL = pytest.importorskip("PIL.Image")
    assert np.array_equal(np.asarray(PIL.open(_io.BytesIO(png))), a)   # the hand encoder is right
    monkeypatch.setitem(sys.modules, "PIL", None)
    assert np.array_equal(convert.decode_png(png), a)


def test_audio_wav_round_trip():
    t = np.linspace(0, 1, 800, dtype=np.float32)
    wf = np.stack([np.sin(2 * np.pi * 5 * t), 0.5 * np.cos(2 * np.pi * 3 * t)])[None]
    wav = convert.audio_to_wav({"waveform": wf, "sample_rate": 8000})
    with wave.open(_io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (2, 8000, 800)
    back = convert.wav_to_audio(wav)
    assert back["sample_rate"] == 8000 and back["waveform"].shape == (1, 2, 800)
    assert np.abs(back["waveform"] - wf).max() < 1e-3


def test_parse_step_matches_the_server_rule():
    from shutter.core.stats import parse_step
    for s in ["denoise 3/8", "segment 1/2 denoise 3/8", "Step 4 / 10", "segment 1/2 decode", "", "denoise 9/8"]:
        assert runner.parse_step(s) == parse_step(s), s


# ---------------------------------------------------------------- client

def test_client_basics(live):
    c, eng = live
    ids = {m["id"]: m["kind"] for m in c.models()}
    assert ids["dummy-image"] == "image" and ids["dummy"] == "video"
    up = c.upload(b"\x89PNGabc", 'a "b".png', "image/png")
    assert up["size"] == 7 and up["content_type"] == "image/png" and Path(up["path"]).read_bytes() == b"\x89PNGabc"
    m = c.memory("dummy-image", width=64, height=64)
    assert m["model"]["fits"] is True and "free_for_models" in m
    with pytest.raises(ShutterError) as e:
        c.submit({"model": "nope"})
    assert e.value.status == 404 or e.value.status == 400
    assert c.cancel("doesnotexist") is None
    assert {r["client"] for r in eng.clients.list()["clients"]} >= {"comfyui-shutter/0.2.0"}
    with pytest.raises(ShutterError) as e:
        ShutterClient("http://127.0.0.1:1", timeout=1).models()
    assert e.value.status == 0 and "cannot reach" in str(e.value)


# ---------------------------------------------------------------- runner

def test_generate_image(live, tmp_path):
    c, eng = live
    seen = []
    hooks = runner.Hooks(on_progress=lambda p, s, n, st: seen.append((p, s, n)))
    res = run(runner.generate(c, {"model": "dummy-image", "prompt": "fox", "width": 40, "height": 24,
                                  "seed": 3, "params": {"steps": 4, "step_s": 0.05}},
                              hooks=hooks, poll_s=0.01, out_dir=str(tmp_path)))
    assert res.kind == "image" and res.images.shape == (1, 24, 40, 3) and res.images.dtype == np.float32
    assert 0.0 <= res.images.min() and res.images.max() <= 1.0
    assert Path(res.output_path).read_bytes() == Path(res.job["result"]["output"]).read_bytes()
    assert res.freed_host_models is False and res.memory["fits"] is True
    ps = [p for p, _, _ in seen]
    assert ps == sorted(ps) and ps[-1] == 1.0 and len(ps) >= 2
    assert any(n == 4 for _, _, n in seen)        # step counters parsed from "denoise k/4"
    assert eng.get(res.job["id"]).client == "comfyui-shutter/0.2.0"


def test_image_input_goes_through_files(live, tmp_path):
    c, eng = live
    red = np.zeros((1, 8, 8, 3), np.float32)
    red[..., 0] = 1.0
    mask = np.ones((1, 8, 8), np.float32)
    res = run(runner.generate(c, {"model": "dummy-image", "prompt": "edit", "params": FAST},
                              image=red, mask=mask, poll_s=0.01, out_dir=str(tmp_path)))
    inputs = eng.get(res.job["id"]).request.inputs
    assert set(inputs) == {"image", "mask"} and "/work/uploads/" in inputs["image"]
    assert np.array_equal(convert.png_to_image(Path(inputs["image"]).read_bytes()), red)
    assert res.uploads["image"]["path"] == inputs["image"]


@needs_ffmpeg
def test_generate_video_with_audio(live, tmp_path):
    c, eng = live
    sr = 16000
    tone = np.sin(np.linspace(0, 2 * np.pi * 440, sr, dtype=np.float32))[None, None] * 0.5
    res = run(runner.generate(c, {"model": "dummy", "prompt": "clip", "width": 32, "height": 16, "frames": 6,
                                  "fps": 6, "params": {"steps": 1, "step_s": 0, "media": True}},
                              audio={"waveform": tone, "sample_rate": sr}, poll_s=0.01, out_dir=str(tmp_path)))
    assert res.kind == "video" and res.output_path.endswith(".mp4")
    assert res.images.shape == (6, 16, 32, 3) and res.images.dtype == np.float32
    assert res.fps == pytest.approx(6.0)
    assert res.audio is not None and res.audio["waveform"].ndim == 3 and res.audio["sample_rate"] > 0
    assert eng.get(res.job["id"]).request.inputs["audio"].endswith(".wav")


def test_interrupt_cancels_the_job(live, tmp_path):
    c, eng = live
    started = {}

    def interrupted():
        started.setdefault("t", time.monotonic())
        return time.monotonic() - started["t"] > 0.3

    with pytest.raises(runner.Interrupted) as e:
        run(runner.generate(c, {"model": "dummy", "prompt": "slow", "params": {"steps": 100, "step_s": 0.05}},
                            hooks=runner.Hooks(is_interrupted=interrupted), poll_s=0.02, out_dir=str(tmp_path)))
    jid = str(e.value).split("job ")[1].split(" ")[0]
    t0 = time.monotonic()
    while eng.get(jid).state.value not in ("cancelled", "failed", "done") and time.monotonic() - t0 < 10:
        time.sleep(0.02)
    assert eng.get(jid).state.value == "cancelled"


def test_task_cancellation_cancels_the_job(live, tmp_path):
    c, eng = live

    async def go():
        t = asyncio.create_task(runner.generate(c, {"model": "dummy", "prompt": "slow",
                                                    "params": {"steps": 100, "step_s": 0.05}},
                                                poll_s=0.02, out_dir=str(tmp_path)))
        await asyncio.sleep(0.4)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

    run(go())
    jobs = [j.id for j in eng.list() if j.request.prompt == "slow"]
    t0 = time.monotonic()
    while any(eng.get(j).state.value != "cancelled" for j in jobs) and time.monotonic() - t0 < 10:
        time.sleep(0.02)
    assert [eng.get(j).state.value for j in jobs] == ["cancelled"]


def test_short_memory_frees_host_models_and_release(live, tmp_path):
    c, eng = live
    freed = []
    eng.probe.set(available=8 * GB + GB // 4)      # reserve is 8 GB: a quarter GB left, dummy needs ~1 GB
    try:
        assert c.memory("dummy-image")["model"]["fits"] is False
        res = run(runner.generate(c, {"model": "dummy-image", "prompt": "x", "params": FAST},
                                  hooks=runner.Hooks(free_host_models=lambda: (freed.append(1),
                                                                               eng.probe.set(available=48 * GB))),
                                  poll_s=0.01, out_dir=str(tmp_path), keep_loaded=False))
    finally:
        eng.probe.set(available=48 * GB)
    assert freed == [1] and res.freed_host_models is True and res.memory["fits"] is False
    assert c.memory()["components"] == []        # keep_loaded=False released the pool
    # free_host_models=False never calls the hook
    eng.probe.set(available=8 * GB + GB // 4)
    try:
        freed.clear()
        with pytest.raises(Exception):
            run(runner.generate(c, {"model": "dummy-image", "prompt": "x", "params": FAST},
                                hooks=runner.Hooks(free_host_models=lambda: freed.append(1)),
                                free_host_models=False, poll_s=0.01, out_dir=str(tmp_path), timeout_s=1.0))
        assert freed == []
    finally:
        eng.probe.set(available=48 * GB)


def test_failed_job_raises(live, tmp_path):
    c, _ = live
    with pytest.raises(ShutterError) as e:
        run(runner.generate(c, {"model": "dummy-image", "prompt": "x", "params": {"fail": "boom", "steps": 1}},
                            poll_s=0.01, out_dir=str(tmp_path)))
    assert "failed" in str(e.value) and "boom" in str(e.value)


# ---------------------------------------------------------------- nodes.py

def test_nodes_import_without_comfyui():
    import shutter_comfy.nodes as nodes
    assert nodes.COMFY_AVAILABLE is False and not hasattr(nodes, "comfy_entrypoint")
    r = nodes.build_request("dummy", "p", "neg", 64, 0, 9, 0, 4, 3.5, -1, "low_memory", 0.5,
                            '{"framing": "small"}')
    assert r == {"model": "dummy", "prompt": "p", "negative_prompt": "neg", "width": 64, "frames": 9, "steps": 4,
                 "guidance": 3.5, "params": {"framing": "small", "mode": "low_memory", "efficiency": 0.5}}
    assert nodes.build_request("dummy", "p", seed=0, model_id=" other ")["model"] == "other"
    assert nodes.build_request("dummy", "p", seed=0)["seed"] == 0
    for bad in ('[1]', '{x'):
        with pytest.raises(ValueError):
            nodes.build_request("dummy", "p", params_json=bad)
    with pytest.raises(ValueError):
        nodes.build_request(nodes.UNREACHABLE, "p")
    assert nodes.model_options("http://127.0.0.1:1") == [nodes.UNREACHABLE]
    # the pack's top-level __init__ registers nothing outside ComfyUI
    spec = importlib.util.spec_from_file_location("comfyui_shutter_pack", PACK / "__init__.py",
                                                  submodule_search_locations=[str(PACK)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_shutter_pack"] = mod
    try:
        spec.loader.exec_module(mod)
        assert mod.COMFY_AVAILABLE is False and not hasattr(mod, "comfy_entrypoint")
    finally:
        sys.modules.pop("comfyui_shutter_pack", None)


class _Stub:
    """Records constructor arguments; stands in for every comfy_api io class."""

    def __init__(self, *a, **k):
        self.a, self.k = a, k


def _fake_comfy(interrupt_after=None):
    """Minimal stand-ins for comfy_api.latest, comfy.utils, comfy.model_management (NOT ComfyUI)."""
    calls = {"progress": [], "unload": 0, "empty": 0}
    io = types.SimpleNamespace()

    class T:
        Input = _Stub
        Output = _Stub

    for n in ("Combo", "String", "Int", "Float", "Boolean", "Image", "Mask", "Audio", "Video"):
        setattr(io, n, T)
    io.Schema = _Stub
    io.Hidden = types.SimpleNamespace(unique_id="UNIQUE_ID")

    class ComfyNode:
        hidden = types.SimpleNamespace(unique_id="7")

    io.ComfyNode = ComfyNode

    class NodeOutput:
        def __init__(self, *v):
            self.values = v

    io.NodeOutput = NodeOutput

    class VideoFromFile:
        def __init__(self, path):
            self.path = path

    latest = types.ModuleType("comfy_api.latest")
    latest.io, latest.ComfyExtension = io, object
    latest.InputImpl = types.SimpleNamespace(VideoFromFile=VideoFromFile)

    class Bar:
        def __init__(self, total, node_id=None):
            calls["bar"] = (total, node_id)

        def update_absolute(self, v, total=None, preview=None):
            calls["progress"].append(v)

    class InterruptProcessingException(Exception):
        pass

    t0 = [None]

    def processing_interrupted():
        if interrupt_after is None:
            return False
        t0[0] = t0[0] or time.monotonic()
        return time.monotonic() - t0[0] > interrupt_after

    def throw():
        if processing_interrupted():
            raise InterruptProcessingException()

    mm = types.ModuleType("comfy.model_management")
    mm.processing_interrupted = processing_interrupted
    mm.throw_exception_if_processing_interrupted = throw
    mm.unload_all_models = lambda: calls.__setitem__("unload", calls["unload"] + 1)
    mm.soft_empty_cache = lambda: calls.__setitem__("empty", calls["empty"] + 1)
    mm.InterruptProcessingException = InterruptProcessingException
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = Bar
    comfy = types.ModuleType("comfy")
    comfy.utils, comfy.model_management = utils, mm
    api = types.ModuleType("comfy_api")
    api.latest = latest
    mods = {"comfy_api": api, "comfy_api.latest": latest, "comfy": comfy, "comfy.utils": utils,
            "comfy.model_management": mm}
    return mods, calls


@pytest.fixture
def stub_nodes(monkeypatch):
    def load(interrupt_after=None):
        mods, calls = _fake_comfy(interrupt_after)
        for k, v in mods.items():
            monkeypatch.setitem(sys.modules, k, v)
        monkeypatch.delitem(sys.modules, "shutter_comfy.nodes", raising=False)
        nodes = importlib.import_module("shutter_comfy.nodes")
        return nodes, calls, mods

    yield load
    sys.modules.pop("shutter_comfy.nodes", None)


def _execute(nodes, url, **over):
    kw = dict(model="dummy-image", prompt="fox", width=32, height=32, frames=0, fps=0.0, steps=0, guidance=0.0,
              seed=1, memory_mode="default", efficiency=0.0, free_comfy_models=True, keep_loaded=True,
              server_url=url, params_json='{"steps": 2, "step_s": 0}')
    kw.update(over)
    return asyncio.run(nodes.ShutterGenerate.execute(**kw))


def test_nodes_with_stub_comfy_api(live, stub_nodes, monkeypatch):
    c, eng = live
    monkeypatch.setenv("SHUTTER_URL", c.url)
    nodes, calls, _ = stub_nodes()
    assert nodes.COMFY_AVAILABLE and nodes.HAS_VIDEO and nodes.HAS_AUDIO
    schema = nodes.ShutterGenerate.define_schema()
    names = [i.a[0] for i in schema.k["inputs"]]
    assert names[:3] == ["model", "prompt", "negative_prompt"] and {"image", "mask", "audio"} <= set(names)
    assert "dummy-image" in schema.k["inputs"][0].k["options"]
    assert len(schema.k["outputs"]) == 3 and schema.k["node_id"] == "ShutterGenerate"
    ext = asyncio.run(nodes.comfy_entrypoint())
    assert [n.__name__ for n in asyncio.run(ext.get_node_list())] == ["ShutterGenerate", "ShutterUnload",
                                                                       "ShutterMemory"]
    out = _execute(nodes, c.url)
    img, video, audio = out.values
    assert np.asarray(img).shape == (1, 32, 32, 3) and video is None and audio is None
    assert calls["bar"] == (nodes.PROGRESS_TOTAL, "7") and calls["progress"][-1] == nodes.PROGRESS_TOTAL
    # short memory: the node unloads ComfyUI's models in process
    eng.probe.set(available=8 * GB + GB // 4)
    try:
        monkeypatch.setattr(sys.modules["comfy.model_management"], "soft_empty_cache",
                            lambda: (calls.__setitem__("empty", calls["empty"] + 1),
                                     eng.probe.set(available=48 * GB)))
        _execute(nodes, c.url)
    finally:
        eng.probe.set(available=48 * GB)
    assert calls["unload"] == 1 and calls["empty"] == 1
    assert "released" in asyncio.run(nodes.ShutterUnload.execute(c.url)).values[0]
    assert "free_for_models" in asyncio.run(nodes.ShutterMemory.execute(c.url, "dummy")).values[0]


@needs_ffmpeg
def test_nodes_video_output_with_stub(live, stub_nodes):
    c, _ = live
    nodes, _, _ = stub_nodes()
    out = _execute(nodes, c.url, model="dummy", width=32, height=16, frames=4, fps=4.0,
                   params_json='{"steps": 1, "step_s": 0, "media": true}')
    img, video, audio = out.values
    assert np.asarray(img).shape == (4, 16, 32, 3) and video.path.endswith(".mp4") and audio is None


def test_nodes_interrupt_with_stub(live, stub_nodes):
    c, eng = live
    nodes, _, mods = stub_nodes(interrupt_after=0.3)
    with pytest.raises(mods["comfy.model_management"].InterruptProcessingException):
        _execute(nodes, c.url, model="dummy", params_json='{"steps": 100, "step_s": 0.05}')
    (jid,) = [j.id for j in eng.list()]
    t0 = time.monotonic()
    while eng.get(jid).state.value != "cancelled" and time.monotonic() - t0 < 10:
        time.sleep(0.02)
    assert eng.get(jid).state.value == "cancelled"
