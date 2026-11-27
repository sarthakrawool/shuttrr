"""MiniMax-H3-Image family: a tiny random model in the Shuttrr build layout runs through the real pipeline.

Checks: every memory step gives the same image bit for bit (incl. blocks stored with different quantization, as the
release keeps a few matrices at 16 bits), the exact GGUF -> MLX mappings, the PyTorch-compatible noise, the PDD head
weights, the adapter's estimates / refusals / run, the catalog entry and the registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("tokenizers")

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from shutter.families.h3image import model as Mo  # noqa: E402
from shutter.families.h3image.convert import ShardWriter, q4_k_to_mlx, q8_0_to_mlx  # noqa: E402
from shutter.families.h3image.pipeline import H3ImagePipeline, Settings, torch_randn  # noqa: E402

GB = 1024**3
DIT = dict(hidden_size=64, num_layers=3, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=32,
           ffn_hidden_size=128, text_dim=64, adaln_curve_grid=17, rope_inv_freq_len=4)
TE = dict(vocab_size=200, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
          intermediate_size=128, head_dim=32, rope_theta=5e6)


def _randomize(module, seed: int, dtype=mx.bfloat16) -> None:
    flat = []
    for i, (k, v) in enumerate(tree_flatten(module.parameters())):
        r = mx.random.normal(v.shape, key=mx.random.key(seed * 100003 + i)) * (0.3 if "norm" in k else 0.08)
        if ("norm" in k and k.endswith("weight")) or k.endswith("scale1") or k.endswith("scale2"):
            r = r + 1.0
        flat.append((k, r.astype(dtype)))
    module.update(tree_unflatten(flat))


def _write(folder: Path, params: dict) -> None:
    w = ShardWriter(folder)
    for k, v in params.items():
        w.add(k, v)
    w.close()


def _tokenizer(folder: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    words = ["<unk>", " "] + [chr(c) for c in range(33, 127)] + [f"w{i}" for i in range(80)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(pattern=" ", behavior="removed")
    folder.mkdir(parents=True, exist_ok=True)
    tok.save(str(folder / "tokenizer.json"))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("h3image-tiny")
    dit = Mo.MiniMaxH3ImageDiT(**DIT)
    _randomize(dit, 1)
    # blocks 0 and 2 at 8-bit, block 1 left at 16 bits (the release keeps some matrices in BF16)
    nn.quantize(dit, group_size=32, bits=8,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and not p.startswith("blocks.1.")
                and m.weight.shape[1] % 32 == 0 and "adaln" not in p and "video_out" not in p)
    _write(d / "transformer", dict(tree_flatten(dit.parameters())))
    from shutter.families.zimage.model import EncoderLayer
    te = {"embed_tokens.weight": (mx.random.normal((TE["vocab_size"], 64), key=mx.random.key(7)) * 0.5)
          .astype(mx.bfloat16)}
    for i in range(TE["num_hidden_layers"]):
        layer = EncoderLayer(**TE)
        _randomize(layer, 10 + i)
        nn.quantize(layer, group_size=32, bits=4)
        te.update({f"layers.{i}.{k}": v for k, v in tree_flatten(layer.parameters())})
    _write(d / "text_encoder", te)
    vae = Mo.VAE(layers=1, dim=64)
    _randomize(vae, 3, mx.float16)
    _write(d / "vae", dict(tree_flatten(vae.parameters())))
    _tokenizer(d / "tokenizer")
    (d / "h3image.json").write_text(json.dumps({"format": "shuttrr-h3image-1", "dit_config": DIT, "te_config": TE,
                                                "vae_layers": 1, "vae_dim": 64}))
    return d


class Pool:
    def __init__(self):
        self.d = {}

    def get(self, key, loader, size, keep=True, **kw):
        if key not in self.d:
            self.d[key] = loader()
        return self.d[key]

    def release(self, key):
        self.d.pop(key, None)

    def __contains__(self, key):
        return key in self.d


def _run(d: Path, rung: str, pool=None, **kw):
    s = Settings(prompt="w1 w2 w3 hello", width=256, height=256, seed=5, **kw)
    return H3ImagePipeline(d, "tiny", pool=pool).generate(s, rung)


def test_every_memory_step_gives_the_same_image(tiny):
    ref = _run(tiny, "resident")
    assert ref[0].shape == (256, 256, 3) and ref[0].dtype == np.uint8 and ref[0].std() > 0
    for rung in ("split", "staged", "stream_wide", "stream_tight"):
        out = _run(tiny, rung)
        assert np.array_equal(out[0], ref[0]), rung


def test_warm_pool_and_shell_are_reused(tiny):
    pool = Pool()
    a = _run(tiny, "resident", pool)
    assert "h3image/tiny/dit" in pool
    b = _run(tiny, "stream_tight", pool)        # the warm whole DiT is used, same image
    assert np.array_equal(a[0], b[0])
    pool2 = Pool()
    H3ImagePipeline(tiny, "tiny", pool=pool2).load_shell()
    c = _run(tiny, "stream_tight", pool2)
    assert np.array_equal(a[0], c[0]) and "h3image/tiny/dit" not in pool2


def test_count_uses_seed_plus_i(tiny):
    two = _run(tiny, "staged", count=2)
    one = H3ImagePipeline(tiny, "tiny").generate(Settings(prompt="w1 w2 w3 hello", width=256, height=256, seed=6),
                                                 "staged")
    assert np.array_equal(two[1], one[0]) and not np.array_equal(two[0], two[1])


def test_torch_randn_matches_pytorch():
    # torch.manual_seed(42); torch.randn(16)[:4] and torch.randn(17) tail behaviour
    assert np.allclose(torch_randn(42, 16)[:4], [1.9269153, 1.4872841, 0.9007172, -2.1055212], atol=1e-6)
    x = torch_randn(42, 40)
    assert x.shape == (40,) and np.allclose(x[:16], torch_randn(42, 16))


def test_pdd_head_weights():
    g = Mo.PDD_SIGMAS
    for k in range(6):
        w = Mo.pdd_head_weights(g[k], g[k + 1])
        assert w[0] == 1.0
        assert sum(1 for x in w[1:] if x) == (1 if k > 0 else 0)
        if k:
            assert w[k] == pytest.approx(1.0)


def test_q8_0_maps_exactly():
    r = np.random.default_rng(0)
    q = r.integers(-128, 128, (8, 64)).astype(np.int8)
    d = (r.random((8, 2)) * 0.01).astype(np.float16)
    wq, s, b = q8_0_to_mlx(d, q)
    deq = np.array(mx.dequantize(wq, s.astype(mx.float32), b.astype(mx.float32), group_size=32, bits=8))
    ref = (q.astype(np.float32).reshape(8, 2, 32) * np.array(s.astype(mx.float32))[..., None]).reshape(8, 64)
    assert np.array_equal(deq, ref)


def test_q4_k_maps_exactly():
    from shutter.families.h3image.gguf import GGML_Q4_K, GGUFTensor, decode, q4_k_parts
    r = np.random.default_rng(1)
    raw = r.integers(0, 256, 144 * 4, dtype=np.uint8)
    raw.reshape(4, 144)[:, 0:4] = np.frombuffer(np.array([0.01, 0.005], np.float16).tobytes(), np.uint8)
    wq, s, b = q4_k_to_mlx(raw, (2, 512))
    sc, mn, q = q4_k_parts(raw)
    s32, b32 = s.astype(mx.float32), b.astype(mx.float32)
    deq = np.array(mx.dequantize(wq, s32, b32, group_size=32, bits=4))
    exp = q.reshape(2, 16, 32).astype(np.float32) * np.array(s32)[..., None] + np.array(b32)[..., None]
    assert np.array_equal(deq, exp.reshape(2, 512))
    # and it is the Q4_K value up to the scales' bfloat16 rounding
    ref = decode(GGUFTensor("x", (2, 512), GGML_Q4_K, 0, raw.size), raw)
    assert np.allclose(deq, ref, rtol=0.01, atol=0.01 * np.abs(ref).max())


def test_gguf_stream_reads_a_written_file(tmp_path):
    import io
    import struct
    from shutter.families.h3image.gguf import GGUFFile, stream_tensors

    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    hdr = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + s("general.architecture") + struct.pack("<I", 8) + s("hyvid")
    hdr += s("w") + struct.pack("<I", 2) + struct.pack("<QQ", 4, 3) + struct.pack("<IQ", 0, 0)
    pad = (32 - len(hdr) % 32) % 32
    blob = hdr + b"\0" * pad + a.tobytes()
    (tmp_path / "t.gguf").write_bytes(blob)
    g = GGUFFile(tmp_path / "t.gguf")
    assert g.tensors["w"].shape == (3, 4) and np.array_equal(g.array("w"), a)
    got = list(stream_tensors(io.BytesIO(blob)))
    assert got[0][0].name == "w" and np.array_equal(got[0][1].view(np.float32).reshape(3, 4), a)


def test_adapter_estimates_and_refusals(tiny, tmp_path):
    from shutter.adapters.h3image import H3ImageAdapter
    from shutter.core import modes as M
    from shutter.types import JobRequest
    a = H3ImageAdapter("tiny", tiny, {})
    assert a.can_load() == (True, "")
    req = JobRequest(model="tiny", prompt="x", width=1024, height=1024, params={"budget_bytes": 100 * GB})
    est = a.rung_estimates(req)
    peaks = [est[r].peak for r in M.RUNG_IDS]
    assert peaks == sorted(peaks, reverse=True)
    ok, why = H3ImageAdapter("x", tmp_path, {}).can_load()
    assert not ok and "not a MiniMax-H3-Image build" in why
    with pytest.raises(ValueError, match="count"):
        a.settings_for(JobRequest(model="tiny", prompt="x", params={"count": 9}))
    with pytest.raises(ValueError, match="LoRA"):
        a.settings_for(JobRequest(model="tiny", prompt="x", params={"loras": ["a.safetensors"]}))
    s = a.settings_for(JobRequest(model="tiny", prompt="a dog", width=1000, height=777, steps=9,
                                  params={"style": "3d-animated"}))
    assert (s.width, s.height) == (1024, 800) and s.prompt.startswith("3D animated film still: a dog")
    assert any("6 steps" in n for n in s.notes)


def test_adapter_default_estimates_without_a_folder():
    from shutter.adapters.h3image import H3ImageAdapter
    from shutter.types import JobRequest
    a = H3ImageAdapter("minimax-h3-image", Path("/nonexistent"), {})
    est = a.rung_estimates(JobRequest(model="minimax-h3-image", prompt="x", width=1024, height=1024))
    assert 21 * GB < est["resident"].peak < 30 * GB          # the whole 8-bit DiT
    assert est["stream_tight"].peak < 4 * GB                 # two blocks + the shell


def test_small_mac_fast_mode_uses_the_lowest_step(tiny):
    from shutter.adapters.h3image import H3ImageAdapter
    from shutter.types import JobRequest
    a = H3ImageAdapter("tiny", tiny, {})
    req = JobRequest(model="tiny", prompt="x", width=1024, height=1024,
                     params={"mode": "fast", "total_bytes": 16 * GB, "budget_bytes": 40 * GB})
    pick = a.choose_rung(req)
    assert pick["rung"] == "stream_tight" and pick["why"] == "small Mac"


def test_adapter_runs_and_writes_pngs(tiny, tmp_path):
    from shutter.adapters.base import RunContext
    from shutter.adapters.h3image import H3ImageAdapter
    from shutter.types import Job, JobRequest
    a = H3ImageAdapter("tiny", tiny, {})
    req = JobRequest(model="tiny", prompt="w1", width=256, height=256, seed=3,
                     output=str(tmp_path / "out.png"), params={"count": 2, "budget_bytes": 100 * GB})
    ctx = RunContext(models=Pool(), cache=None, work_dir=tmp_path, report=lambda p, s: None,
                     check_cancelled=lambda: None, memory_sample=lambda: {})
    res = a.run(Job(request=req), ctx)
    assert res["outputs"] == [str(tmp_path / "out.png"), str(tmp_path / "out.s4.png")]
    assert all(Path(p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in res["outputs"])
    assert res["kind"] == "image" and res["steps"] == 6 and res["variant"] == "minimax-h3-image"


def test_registry_and_request_options(tmp_path):
    from shutter.core import request_options as R
    from shutter.core.config import Config
    from shutter.core.registry import BUILTIN_FAMILIES
    assert BUILTIN_FAMILIES["h3-image"] == "shutter.adapters.h3image:H3ImageAdapter"
    cfg = Config.load(str(tmp_path / "config.json"))
    cfg.models = {"minimax-h3-image": {"adapter": "h3-image", "dir": str(tmp_path / "h3"), "options": {}}}
    o = R.request_options(cfg, "minimax-h3-image", total_bytes=16 * GB)
    assert o["kind"] == "image" and o["lora"] is False and o["fixed_steps"] == 6 and o["defaults"]["steps"] == 6


def test_vae_tiles_cover_the_image_with_overlap():
    V = Mo.VAE
    assert V.tile_starts(10) == ([0], 10)
    for n in (17, 32, 40, 64, 128):
        st, t = V.tile_starts(n)
        assert st[0] == 0 and st[-1] + t == n and t == 16
        assert all(b - a <= t - V.MIN_OVERLAP for a, b in zip(st, st[1:]))


def test_catalog_entry_pinned_with_licence_gate():
    from shutter.core import catalog as C
    e = C.CATALOG["minimax-h3-image-8bit"]
    assert e["adapter"] == "h3-image" and e["kind"] == "image" and e["beta"] and e["revision"]
    assert "MiniMax-H3-Image" in e["name"]
    assert all(f["sha256"] for f in e["files"] if f["path"].endswith(".safetensors"))
    assert any("United States" in p for p in e["licence"]["points"])
    assert any("MiniMax-H3-Image" in p for p in e["licence"]["points"])
    fit = C.classify_fit("minimax-h3-image-8bit", 16 * GB, baseline=3 * GB)
    assert fit["level"] == "recommended" and fit["measured"]
    assert C.default_request_any("minimax-h3-image-8bit", 16 * GB)[:2] == (640, 512)
