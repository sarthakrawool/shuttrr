"""Fish Drama 3 adapter against a fake TensorFold package (no model, no GPU)."""

from __future__ import annotations

import sys
import textwrap
import wave
from pathlib import Path

import numpy as np
import pytest

from shutter.adapters.drama3 import Drama3Adapter, build_bytes, take_path, take_seeds
from shutter.adapters.base import RunContext
from shutter.core import modes as M
from shutter.types import GB, Cancelled, Job, JobRequest

FAKE = {
    "generate.py": """
        import numpy as np
        from dataclasses import dataclass, field
        LOADS = []
        CALLS = []
        @dataclass
        class Result:
            frames: np.ndarray
            decode_s: float = 0.5
            prefill_s: float = 0.1
            finish: str = "stop"
            stats: dict = field(default_factory=dict)
        class Drama3Generator:
            def __init__(self, model_dir):
                from tensorfold.families.drama3.codec import Codec
                import os
                LOADS.append((str(model_dir), {k: v for k, v in os.environ.items() if k.startswith("TF_D3_")}))
                self.model_dir = model_dir
                self.codec = Codec(model_dir)
            def synthesize(self, text, *, references=None, sampling=None, seed=0, max_chars=300, max_frames=None,
                           log=print, verify=False, verify_retries=2, stop=False):
                CALLS.append({"text": text, "seed": seed, "refs": references, "sampling": sampling, "stop": stop})
                log(f"[drama3] chunk 1/1 seed {seed}")
                n = 22
                audio = np.full(n * 2048, seed / 100.0, dtype=np.float32)
                return audio, [Result(frames=np.full((n, 11), seed, dtype=np.int64))]
    """,
    "prompt.py": """
        from dataclasses import dataclass
        import numpy as np
        @dataclass
        class Reference:
            transcript: str
            codes: np.ndarray
    """,
    "sampling.py": """
        from dataclasses import dataclass
        @dataclass(frozen=True)
        class Sampling:
            temperature: float = 1.0
            top_k: int = 30
            top_p: float = 0.9
            greedy: bool = False
    """,
    "codec.py": """
        import wave
        import numpy as np
        ENCODES = []
        class Codec:
            def __init__(self, path):
                self.encoder_freed = False
            def encode(self, audio):
                if self.encoder_freed:
                    raise RuntimeError("encoder was freed")
                ENCODES.append(len(audio))
                return np.arange(40, dtype=np.int32).reshape(10, 4)
            def free_encoder(self):
                self.encoder_freed = True
        def load_audio(path):
            return np.zeros(1000, dtype=np.float32)
        def save_wav(path, audio, sample_rate=44100):
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
                w.writeframes((np.clip(np.asarray(audio), -1, 1) * 32767).astype("<i2").tobytes())
    """,
}


@pytest.fixture
def fake_tf(tmp_path):
    root = tmp_path / "src" / "tensorfold" / "families" / "drama3"
    root.mkdir(parents=True)
    for d in (root, root.parent, root.parent.parent):
        (d / "__init__.py").write_text("")
    for name, body in FAKE.items():
        (root / name).write_text(textwrap.dedent(body))
    for m in [m for m in sys.modules if m == "tensorfold" or m.startswith("tensorfold.")]:
        del sys.modules[m]
    yield tmp_path / "src"
    for m in [m for m in sys.modules if m == "tensorfold" or m.startswith("tensorfold.")]:
        del sys.modules[m]
    if str(tmp_path / "src") in sys.path:
        sys.path.remove(str(tmp_path / "src"))


class Pool:
    def __init__(self):
        self.items, self.loads = {}, []

    def get(self, key, loader, size_bytes, *, keep=True, **kw):
        if key not in self.items:
            self.loads.append((key, size_bytes))
            obj = loader()
            if not keep:
                return obj
            self.items[key] = obj
        return self.items[key]

    def release(self, key):
        self.items.pop(key, None)

    def release_all(self, prefix=""):
        self.items.clear()


class Cache:
    def __init__(self):
        self.store = {}

    def get_or_compute(self, namespace, key_parts, compute):
        k = (namespace, tuple(map(str, key_parts)))
        if k not in self.store:
            self.store[k] = compute()
        return self.store[k]


def ctx_for(tmp_path, pool=None, cache=None, available=None, cancel_at=None):
    reports = []

    def check():
        if cancel_at is not None and len(reports) >= cancel_at:
            raise Cancelled("cancelled")

    c = RunContext(models=pool or Pool(), cache=cache or Cache(), work_dir=tmp_path,
                   report=lambda f, s: reports.append((f, s)), check_cancelled=check,
                   memory_sample=lambda: {"available": available} if available else {})
    c.reports = reports
    return c


def build(tmp_path, name, gb_tenths):
    d = tmp_path / name
    d.mkdir()
    with open(d / "backbone-00.safetensors", "wb") as f:
        f.truncate(int(gb_tenths * GB / 10))
    return d


def adapter(tmp_path, fake_tf, **options):
    main = build(tmp_path, "drama-3-mlx", 432)
    return Drama3Adapter("fish-drama3", main, {"tensorfold_src": str(fake_tf), **options})


def job(tmp_path, **kw):
    params = kw.pop("params", {})
    r = JobRequest(model="fish-drama3", prompt=kw.pop("prompt", "Hello there."), params=params,
                   output=str(tmp_path / "out" / "line.wav"), **kw)
    return Job(id="j1", request=r)


def test_registered_as_audio_family(tmp_path):
    from shutter.core.config import Config
    assert Drama3Adapter.kind == "audio" and Drama3Adapter.extension == "wav"
    from shutter.core import registry as R
    assert "shutter.adapters.drama3:Drama3Adapter" in open(R.__file__).read()


def test_seeds_and_take_paths():
    assert take_seeds(JobRequest(model="m", seed=5)) == [5]
    assert take_seeds(JobRequest(model="m", seed=5, params={"takes": 3})) == [5, 6, 7]
    assert take_seeds(JobRequest(model="m", params={"seeds": [11, 5]})) == [11, 5]
    with pytest.raises(ValueError):
        take_seeds(JobRequest(model="m", params={"seeds": [1, 1]}))
    out = Path("/x/line.wav")
    assert take_path(out, [5, 11], 0) == out
    assert take_path(out, [5, 11], 1) == Path("/x/line.s11.wav")


def test_estimate_follows_the_build_size(tmp_path, fake_tf):
    a = adapter(tmp_path, fake_tf)
    assert build_bytes(a.model_dir) == int(43.2 * GB)
    e = a.estimate(JobRequest(model="fish-drama3", prompt="hi"))
    # files - unused parts + 1 GB cache + 2 GB of working memory: the measured 8-bit peak was 43-44.4 GB
    assert 44 * GB < e.peak < 45 * GB
    spike = Drama3Adapter("m", a.model_dir, {"lowmem": False}).estimate(JobRequest(model="m", prompt="hi"))
    assert spike.peak > e.peak + 20 * GB
    with_verify = a.estimate(JobRequest(model="fish-drama3", prompt="hi", params={"verify": True}))
    assert with_verify.peak - e.peak == int(1.6 * GB)


def test_low_memory_steps_pick_smaller_builds(tmp_path, fake_tf):
    small = build(tmp_path, "drama-3-mlx-gptq5g64", 290)
    smallest = build(tmp_path, "drama-3-mlx-gptq4g32", 250)
    a = adapter(tmp_path, fake_tf, variants={"staged": str(small), "stream_tight": str(smallest)})
    b = a.builds()
    assert [b[r].name for r in M.RUNG_IDS] == ["drama-3-mlx", "drama-3-mlx", "drama-3-mlx-gptq5g64",
                                              "drama-3-mlx-gptq5g64", "drama-3-mlx-gptq4g32"]
    assert a.is_lossless("split") and not a.is_lossless("staged")
    r = JobRequest(model="fish-drama3", prompt="hi", params={"mode": "low_memory", "efficiency": 1.0})
    assert a.choose(r)["build"] == smallest
    # Fast on a 36 GB machine: the full build does not fit, so it steps down to the largest that does
    c = ctx_for(tmp_path, available=34 * GB)
    pick = a.choose(JobRequest(model="fish-drama3", prompt="hi"), c)
    assert pick["build"] == small and pick["stepped_down_from"] == "resident"
    assert pick["estimates"][pick["rung"]].peak <= 34 * GB
    with pytest.raises(ValueError):
        Drama3Adapter("m", tmp_path, {"variants": {"nope": "/x"}}).builds()


def test_run_keeps_the_model_warm_and_caches_the_reference(tmp_path, fake_tf):
    a = adapter(tmp_path, fake_tf)
    pool, cache = Pool(), Cache()
    clip = tmp_path / "me.wav"
    clip.write_bytes(b"voice")
    params = {"ref_text": "this is me", "takes": 2}
    out = a.run(job(tmp_path, audio=str(clip), seed=5, params=params), ctx_for(tmp_path, pool, cache))
    gen_mod = sys.modules["tensorfold.families.drama3.generate"]
    codec_mod = sys.modules["tensorfold.families.drama3.codec"]
    assert len(gen_mod.LOADS) == 1
    assert gen_mod.LOADS[0][1] == {"TF_D3_LOWMEM": "1", "TF_D3_DROP_LMHEAD": "1", "TF_D3_FREE_ENCODER": "1",
                                   "TF_D3_CACHE_GB": "1", "TF_D3_CODEC_WINDOW": "512"}
    assert [c["seed"] for c in gen_mod.CALLS] == [5, 6]
    assert gen_mod.CALLS[0]["sampling"].temperature == 0.75
    assert gen_mod.CALLS[0]["refs"][0].transcript == "this is me"
    assert out["reference"] == "encoded" and out["reference_frames"] == 4 and not out["warm"]
    assert out["outputs"] == [str(tmp_path / "out" / "line.wav"), str(tmp_path / "out" / "line.s6.wav")]
    assert out["lossless"] and out["build"] == "drama-3-mlx" and out["kind"] == "audio" and not out["batched"]
    with wave.open(out["outputs"][1]) as w:
        assert w.getframerate() == 44100 and w.getnframes() == 22 * 2048
    assert out["takes"][0]["frames"] == 22 and out["audio_s"] == pytest.approx(2 * 22 * 2048 / 44100, abs=1e-3)

    # second job: no reload, no second encode although the encoder is gone
    out2 = a.run(job(tmp_path, audio=str(clip), seed=9, params={"ref_text": "this is me"}),
                 ctx_for(tmp_path, pool, cache))
    assert len(gen_mod.LOADS) == 1 and len(codec_mod.ENCODES) == 1
    assert out2["warm"] and out2["reference"] == "cached"

    # a new clip after the encoder was freed: encoded with a second codec, the warm model stays
    other = tmp_path / "other.wav"
    other.write_bytes(b"another voice")
    out3 = a.run(job(tmp_path, audio=str(other), params={"ref_text": "someone"}), ctx_for(tmp_path, pool, cache))
    assert out3["reference"] == "encoded" and len(codec_mod.ENCODES) == 2 and len(gen_mod.LOADS) == 1


def test_batched_takes_are_opt_in(tmp_path, fake_tf):
    a = adapter(tmp_path, fake_tf)
    a.tensorfold()
    gen_mod = sys.modules["tensorfold.families.drama3.generate"]
    seen = []

    def synthesize_batch(self, text, *, references, sampling, seeds, max_chars=300, max_frames=None, verify=False,
                         verify_retries=2, stop=True, log=print):
        seen.append(list(seeds))
        return [self.synthesize(text, references=references, sampling=sampling, seed=s, log=log) for s in seeds]

    gen_mod.Drama3Generator.synthesize_batch = synthesize_batch
    out = a.run(job(tmp_path, params={"seeds": [1, 2, 3]}), ctx_for(tmp_path))
    assert not out["batched"] and seen == []
    out = a.run(job(tmp_path, params={"seeds": [1, 2, 3], "batch": True}), ctx_for(tmp_path))
    assert out["batched"] and seen == [[1, 2, 3]] and len(out["outputs"]) == 3


def test_reference_needs_its_text_and_prompt_must_not_be_empty(tmp_path, fake_tf):
    a = adapter(tmp_path, fake_tf)
    clip = tmp_path / "me.wav"
    clip.write_bytes(b"voice")
    with pytest.raises(ValueError, match="ref_text"):
        a.run(job(tmp_path, audio=str(clip)), ctx_for(tmp_path))
    with pytest.raises(ValueError, match="empty"):
        a.run(job(tmp_path, prompt="  "), ctx_for(tmp_path))


def test_cancel_between_takes(tmp_path, fake_tf):
    a = adapter(tmp_path, fake_tf)
    with pytest.raises(Cancelled):
        a.run(job(tmp_path, params={"takes": 4}), ctx_for(tmp_path, cancel_at=4))


def test_missing_tensorfold_is_a_clear_error(tmp_path):
    for m in [m for m in sys.modules if m == "tensorfold" or m.startswith("tensorfold.")]:
        del sys.modules[m]
    a = Drama3Adapter("fish-drama3", tmp_path, {"tensorfold_src": str(tmp_path / "nowhere")})
    with pytest.raises(RuntimeError, match="tensorfold_src"):
        a.tensorfold()


def test_vendored_runtime_is_the_default(tmp_path, monkeypatch):
    """No tensorfold_src: the adapter imports the vendored shutter.families.drama3 (needs mlx and tokenizers only)."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("tokenizers")
    for k in ("TF_D3_LOWMEM", "TF_D3_DROP_LMHEAD", "TF_D3_FREE_ENCODER", "TF_D3_CACHE_GB", "TF_D3_CODEC_WINDOW"):
        monkeypatch.delenv(k, raising=False)
    a = Drama3Adapter("fish-drama3", tmp_path, {})
    tf = a.tensorfold()
    assert tf["generate"].__name__ == "shutter.families.drama3.generate"


# ---- SSD streaming steps (48 GB Macs) ------------------------------------------------------------------------------
from shutter.adapters import drama3 as D3  # noqa: E402


def test_stream_peak_follows_the_measured_points():
    assert D3.stream_mlx_peak_gb(18) == 28.2 and D3.stream_mlx_peak_gb(24) == 34.6 and D3.stream_mlx_peak_gb(30) == 40.5
    assert 28.2 < D3.stream_mlx_peak_gb(20) < 34.6
    assert D3.stream_mlx_peak_gb(12) < 28.2 < D3.stream_mlx_peak_gb(36) - 10


def test_stream_steps_stream_the_main_build(tmp_path, fake_tf):
    a = adapter(tmp_path, fake_tf, stream_external=True)
    assert [a.stream_gb(r) for r in M.RUNG_IDS] == [0, 0, 0, 24, 18]
    assert all(p == a.model_dir for p in a.builds().values())
    assert all(a.is_lossless(r) for r in M.RUNG_IDS)            # same files, same audio
    e = a.rung_estimates(JobRequest(model="fish-drama3", prompt="hi"))
    assert e["stream_wide"].peak == e["resident"].peak - int((42.7 - 34.6) * GB)
    assert e["stream_tight"].peak == e["resident"].peak - int((42.7 - 28.2) * GB)
    # an external TensorFold checkout (may lack the expert cache): no streaming unless asked
    assert Drama3Adapter("m", a.model_dir, {"tensorfold_src": str(fake_tf)}).stream_gb("stream_tight") == 0
    # the person can change the cache, or switch a step off
    b = Drama3Adapter("fish-drama3", a.model_dir, {"tensorfold_src": str(fake_tf), "stream_external": True,
                                                   "stream_cache_gb": {"stream_wide": 20, "stream_tight": 0}})
    assert b.stream_gb("stream_wide") == 20 and b.stream_gb("stream_tight") == 0
    assert Drama3Adapter("m", a.model_dir, {"stream_cache_gb": {"stream_tight": 2}}).stream_gb("stream_tight") == 8


def test_vendored_runtime_streams_by_default(tmp_path):
    a = Drama3Adapter("fish-drama3", tmp_path, {})
    assert a.stream_gb("stream_wide") == 24 and a.stream_gb("stream_tight") == 18 and a.stream_gb("resident") == 0
    assert Drama3Adapter("m", tmp_path, {"lowmem": False}).stream_gb("stream_tight") == 0


def test_fast_on_a_48_gb_mac_steps_down_to_streaming(tmp_path, fake_tf):
    """~38 GB free (a 48 GB Mac with apps open): the whole model does not fit, the 24 GB cache step does."""
    a = adapter(tmp_path, fake_tf, stream_external=True)
    pick = a.choose(JobRequest(model="fish-drama3", prompt="hi"), ctx_for(tmp_path, available=38 * GB))
    assert pick["rung"] == "stream_wide" and pick["stream_gb"] == 24 and pick["build"] == a.model_dir
    pick = a.choose(JobRequest(model="fish-drama3", prompt="hi"), ctx_for(tmp_path, available=32 * GB))
    assert pick["rung"] == "stream_tight" and pick["stream_gb"] == 18
    pick = a.choose(JobRequest(model="fish-drama3", prompt="hi"), ctx_for(tmp_path, available=60 * GB))
    assert pick["rung"] == "resident" and pick["stream_gb"] == 0
    assert a.pool_key(a.model_dir, 24).endswith("/stream24") and a.pool_key(a.model_dir) .endswith("drama-3-mlx")


def test_estimate_time_uses_this_macs_memory(tmp_path, fake_tf, monkeypatch):
    """The engine admits jobs on estimate(): without a run context it reads this Mac's memory, so a 48 GB Mac
    gets the streaming step there too instead of waiting for 44 GB that never comes."""
    from shutter.core import memory as mem

    class Probe:
        def sample(self):
            class S:
                def to_dict(self):
                    return {"available": 36 * GB, "total": 48 * GB}
            return S()

    monkeypatch.setattr(mem, "default_probe", lambda: Probe())
    a = adapter(tmp_path, fake_tf, stream_external=True)
    e = a.estimate(JobRequest(model="fish-drama3", prompt="hi"))
    assert e.peak <= 36 * GB and "SSD" in e.note
    assert "/stream" in a.pool_keys(JobRequest(model="fish-drama3", prompt="hi"))[0]


def test_streamed_run_loads_with_the_expert_cache_set_only_for_that_load(tmp_path, fake_tf, monkeypatch):
    monkeypatch.delenv("TF_D3_EXPERT_CACHE_GB", raising=False)
    a = adapter(tmp_path, fake_tf, stream_external=True)
    pool = Pool()
    out = a.run(job(tmp_path, params={"mode": "low_memory", "efficiency": 1.0}), ctx_for(tmp_path, pool))
    import tensorfold.families.drama3.generate as G
    assert G.LOADS[-1][1].get("TF_D3_EXPERT_CACHE_GB") == "18"
    assert "TF_D3_EXPERT_CACHE_GB" not in __import__("os").environ
    assert out["rung"] == "stream_tight" and out["stream_gb"] == 18 and out["lossless"]
    assert pool.loads[0][0].endswith("/stream18")
