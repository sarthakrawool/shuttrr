"""Streamed DiT (rungs stream_wide / stream_tight) vs the resident DiT, with the REAL LongCat model classes at a tiny
synthetic size, quantized to 8 bits like the release (nn.quantize, group 64, release skip list), saved as shards +
index the way load_dit reads them, with synthetic DMD/OPSD LoRAs attached through runtime_lora. The streamed
denoise must equal the resident one exactly (max abs diff 0), compiled and eager, split and whole-block, with
and without LoRA, with the DMD -> OPSD swap between steps, through the KV-cache continuation path, and across a
staged release/reload. Skipped when the patched tree is not on this machine (LONGCAT_CODE_DIR overrides)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten  # noqa: E402

import shutter.adapters.longcat as L  # noqa: E402

REAL_CODE = Path(os.environ.get("LONGCAT_CODE_DIR", "/home/claude/ref/shutter-ref/longcat-mlx-code"))

TINY = {
    "in_channels": 4, "out_channels": 4, "hidden_size": 64, "depth": 4, "num_heads": 4, "caption_channels": 32,
    "mlp_ratio": 4, "adaln_tembed_dim": 32, "frequency_embedding_size": 32, "patch_size": [1, 2, 2],
    "text_tokens_zero_pad": True, "audio_window": 5, "audio_block": 5, "audio_channel": 32,
    "intermediate_dim": 32, "output_dim": 32, "context_tokens": 8, "vae_scale": 4, "audio_prenorm": False,
    "class_range": 24, "class_interval": 4,
}
SKIP = ["final_layer.linear", "t_embedder.", "y_embedder.", "adaLN_modulation.", "audio_adaLN_modulation."]
LORA_TARGETS = {"attn.qkv": 3, "attn.proj": 1, "cross_attn.q_linear": 1, "cross_attn.kv_linear": 2,
                "ffn.w1": 1, "ffn.w2": 1, "ffn.w3": 1}
RANK = 4


@pytest.fixture(scope="module")
def tree():
    if not REAL_CODE.is_dir():
        pytest.skip("patched LongCat tree not available")
    mod = L.import_generation(REAL_CODE)
    import longcat_video_avatar.fast as fast  # noqa: F401
    yield mod
    for name in [n for n in sys.modules if n.startswith("longcat_video_avatar")]:
        del sys.modules[name]
    if str(REAL_CODE.resolve()) in sys.path:
        sys.path.remove(str(REAL_CODE.resolve()))


def _rng_arr(shape, seed, scale=1.0, dtype=None):
    a = mx.array((np.random.default_rng(seed).standard_normal(shape) * scale).astype(np.float32))
    return a.astype(dtype) if dtype is not None else a


def _enc(path: str) -> str:
    return "lora___lorahyphen___" + path.replace(".", "___lorahyphen___")


@pytest.fixture(scope="module")
def weights(tree, tmp_path_factory):
    """variant folder: dit/ (2 shards + index + config with quantization) and lora/ (dmd + opsd)."""
    from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel

    root = tmp_path_factory.mktemp("lc-variant")
    mx.random.seed(3)
    dit = LongCatVideoAvatarTransformer3DModel.from_config(TINY)
    # non-trivial affine norms and biases so every tensor matters
    params = tree_flatten(dit.parameters())
    dit.update(__import__("mlx.utils", fromlist=["tree_unflatten"]).tree_unflatten(
        [(k, v + _rng_arr(v.shape, 100 + i, 0.05)) for i, (k, v) in enumerate(params)]))
    dit.set_dtype(mx.bfloat16)
    mods = []

    def pred(path, m):
        ok = isinstance(m, nn.Linear) and not any(s in path for s in SKIP) and m.weight.shape[1] % 64 == 0
        if ok:
            mods.append(path)
        return ok

    nn.quantize(dit, group_size=64, bits=8, class_predicate=pred)
    mx.eval(dit.parameters())
    flat = dict(tree_flatten(dit.parameters()))
    names = sorted(flat)
    (root / "dit").mkdir()
    half = len(names) // 2
    wm = {}
    for shard, part in (("model-00001-of-00002.safetensors", names[:half]),
                        ("model-00002-of-00002.safetensors", names[half:])):
        mx.save_safetensors(str(root / "dit" / shard), {k: flat[k] for k in part}, metadata={"format": "mlx"})
        wm.update({k: shard for k in part})
    (root / "dit" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wm}))
    cfg = dict(TINY, quantization={"method": "mlx.nn.quantize", "bits": 8, "group_size": 64, "skip_patterns": SKIP,
                                   "modules": sorted(set(mods))})
    (root / "dit" / "config.json").write_text(json.dumps(cfg))
    assert any(m.startswith("blocks.0.attn.qkv") for m in mods) and any("ffn.w2" in m for m in mods)

    (root / "lora").mkdir()
    loras = {}
    for li, (key, fname) in enumerate((("dmd", "dmd_loraA.safetensors"), ("opsd", "opsd_loraB.safetensors"))):
        sd = {}
        for b in range(TINY["depth"]):
            for t, (mname, nups) in enumerate(LORA_TARGETS.items()):
                path = f"blocks.{b}.{mname}"
                lin = dit.blocks[b]
                for part in mname.split("."):
                    lin = getattr(lin, part)
                out_dim = (lin.weight.shape[0] if not isinstance(lin, nn.QuantizedLinear)
                           else lin.scales.shape[0])
                in_dim = (lin.scales.shape[1] * lin.group_size if isinstance(lin, nn.QuantizedLinear)
                          else lin.weight.shape[1])
                seed = 1000 * li + 50 * b + t
                sd[f"{_enc(path)}.lora_down.weight"] = _rng_arr((nups * RANK, in_dim), seed, 0.3, mx.bfloat16)
                if nups == 1:
                    sd[f"{_enc(path)}.lora_up.weight"] = _rng_arr((out_dim, RANK), seed + 7, 0.3, mx.bfloat16)
                else:
                    for j in range(nups):
                        sd[f"{_enc(path)}.lora_up.blocks.{j}.weight"] = _rng_arr((out_dim // nups, RANK),
                                                                                 seed + 11 + j, 0.3, mx.bfloat16)
                if t % 2 == 0:   # some modules carry alpha_scale (varying per block), others use the 0.5 default
                    sd[f"{_enc(path)}.alpha_scale"] = mx.array(0.25 + 0.125 * b + 0.0625 * li, dtype=mx.bfloat16)
        mx.save_safetensors(str(root / "lora" / fname), sd)
        loras[key] = root / "lora" / fname
    return root, loras


def _inputs(T=3, H=4, W=4, cont=False):
    lat = _rng_arr((1, 4, T, H, W), 21)
    n_audio = 1 + 4 * (T - 2) if cont else 1 + 4 * (T - 1)
    audio = _rng_arr((1, n_audio, 5, 5, 32), 22)
    text = _rng_arr((1, 1, 6, 32), 23)
    mask = mx.array([[1, 1, 1, 1, 0, 0]])
    return lat, audio, text, mask


def _bare(tree, cls, root, loras, compile_, stream=None):
    g = object.__new__(cls)
    g.paths = tree.ModelPaths(components=root, dit=root / "dit", loras=dict(loras))
    g.compile = compile_
    g.dit = None
    g.lora_keys = []
    g._anchor = None
    if stream is not None:
        g._shutter_stream_cfg = stream
    return g


@pytest.fixture
def switches(monkeypatch, tree):
    fast = sys.modules["longcat_video_avatar.fast"]

    def set_(split: bool, ffn: int):
        monkeypatch.setenv("LC_BLOCK_SPLIT", "1" if split else "0")
        monkeypatch.setenv("LC_LORA_SWAP", "1")
        monkeypatch.setattr(fast, "FFN_CHUNKS", ffn)
    return set_


def _run(gen, tree, steps=2, cont=False, segments=1, release_between=False):
    lat, audio, text, mask = _inputs(T=6 if cont else 5, cont=cont)
    st = tree.Settings(steps=steps, mix_steps=1)
    outs = []
    for seg in range(segments):
        gen._load_dit()
        if cont:
            out = gen._denoise(lat, 2, text, mask, None, None, audio, st, num_ref_latents=1, log=lambda *_: None)
        else:
            out = gen._denoise(lat, 1, text, mask, None, None, audio, st, log=lambda *_: None)
        mx.eval(out)
        outs.append(np.array(out))
        if release_between:
            gen._shutter_release_dit()
    return outs


CASES = [
    # (compile, split, ffn chunks, loras?, window)
    (True, False, 1, True, 2),
    (True, False, 1, True, 8),
    (True, True, 4, True, 2),
    (True, True, 1, False, 8),
    (True, False, 1, False, 2),
    (False, False, 1, True, 2),
    (False, True, 4, True, 8),
]


@pytest.mark.parametrize("compile_,split,ffn,with_lora,window", CASES)
def test_streamed_equals_resident(tree, weights, switches, compile_, split, ffn, with_lora, window):
    root, loras = weights
    loras = loras if with_lora else {}
    switches(split, ffn)
    ref_gen = _bare(tree, tree.AvatarGenerator, root, loras, compile_)          # the production class
    ref = _run(ref_gen, tree)
    cls = L.make_generator_class(tree)
    sgen = _bare(tree, cls, root, loras, compile_, stream={"window": window, "nocache": False})
    got = _run(sgen, tree)
    sd = sgen.dit.__dict__["_shutter_stream"]
    assert np.abs(got[0] - ref[0]).max() == 0.0, "streamed DiT differs from the resident DiT"
    assert np.array_equal(got[0], ref[0])
    st = sd.stats()
    assert st["resident_groups_peak"] <= window
    assert st["bytes_read"] > 0
    # no block weights outlive their turn
    shell_sizes = [v.size for k, v in tree_flatten(sd.shell.parameters())]
    assert max(shell_sizes) == 0
    if with_lora:
        assert [s["to"] for s in st["lora_swaps"]][-1] == ["opsd"]       # DMD -> OPSD between the two steps
        assert sgen.lora_swap is False
    sd.close()


def test_streamed_kv_cache_continuation(tree, weights, switches):
    """Continuation segment: build_cache + forward_noise walk the streamed blocks (the cache is rebuilt at the
    LoRA phase change) and stay identical to the resident run."""
    root, loras = weights
    switches(False, 1)
    ref = _run(_bare(tree, tree.AvatarGenerator, root, loras, True), tree, cont=True)
    cls = L.make_generator_class(tree)
    got = _run(_bare(tree, cls, root, loras, True, stream={"window": 2}), tree, cont=True)
    assert np.array_equal(got[0], ref[0])


def test_staged_release_and_reload_stays_identical(tree, weights, switches):
    """Two segments with the DiT released in between (staged rung): reload gives the same result as a
    resident DiT that was never released, and the release really drops the DiT."""
    root, loras = weights
    switches(True, 4)
    ref = _run(_bare(tree, tree.AvatarGenerator, root, loras, True), tree, segments=2)
    cls = L.make_generator_class(tree)
    g = _bare(tree, cls, root, loras, True)
    got = _run(g, tree, segments=2, release_between=True)
    assert g.dit is None and g._shutter_releases == 2 and g._shutter_reloads == 1
    assert np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1])


def test_decode_hooks_release_and_latents(tree, weights, switches, tmp_path):
    root, loras = weights
    switches(True, 4)
    cls = L.make_generator_class(tree)
    g = _bare(tree, cls, root, loras, True)
    g._load_dit()
    calls = []

    class FakeMod:
        @staticmethod
        def vae_decode(vae, z, tile=0, overlap=12):
            calls.append((g.dit is None, tile))
            return np.zeros((1, 2, 2, 3), np.uint8)

    meter = L.StageMeter(mx)
    meter.start("prep")
    z = _rng_arr((1, 4, 3, 4, 4), 5)
    with L.decode_hooks(FakeMod, g, release="last", n_segments=2, save_dir=tmp_path / "lat", meter=meter) as info:
        FakeMod.vae_decode(None, z, 64)        # segment 1: kept ("last" policy)
        FakeMod.vae_decode(None, z * 2, 64)    # segment 2: released first
    assert calls == [(False, 64), (True, 64)]
    assert info["decodes"] == 2 and len(info["latents"]) == 2
    saved = np.load(tmp_path / "lat" / "latents_denorm.npy")
    assert saved.dtype == np.float32 and np.array_equal(saved, np.array(z * 2))     # overwritten per segment
    assert np.array_equal(np.load(tmp_path / "lat" / "latents_denorm_seg1.npy"), np.array(z))
    meter.close()
    stages = [r["stage"] for r in meter.records]
    assert stages == ["prep", "decode", "encode", "release", "decode", "finish"]
    assert FakeMod.vae_decode.__name__ == "vae_decode" and not hasattr(FakeMod.vae_decode, "__wrapped__")


def test_stream_sizes_header_only(weights):
    from shutter.adapters.longcat_stream import dit_stream_sizes

    root, _ = weights
    s = dit_stream_sizes(root / "dit")
    assert s["blocks"] == TINY["depth"] and s["block_max"] > 0 and s["nonblock"] > 0
    assert s["total"] == sum(f.stat().st_size for f in (root / "dit").glob("*.safetensors")) - sum(
        8 + int.from_bytes(open(f, "rb").read(8), "little") for f in (root / "dit").glob("*.safetensors"))


def test_equality_is_not_vacuous(tree, weights, switches, monkeypatch):
    """The comparison above is sensitive: feeding block (i+1)'s weights at block i, or dropping the LoRAs,
    changes the output."""
    from shutter.core import stream as S

    root, loras = weights
    switches(False, 1)
    ref = _run(_bare(tree, tree.AvatarGenerator, root, loras, True), tree)[0]
    no_lora = _run(_bare(tree, tree.AvatarGenerator, root, {}, True), tree)[0]
    assert np.abs(ref - no_lora).max() > 0
    orig = S.BlockStreamer.block

    def shifted(self, i):
        return orig(self, (i + 1) % len(self))

    monkeypatch.setattr(S.BlockStreamer, "block", shifted)
    cls = L.make_generator_class(tree)
    bad = _run(_bare(tree, cls, root, loras, True, stream={"window": 2}), tree)[0]
    assert np.abs(bad - ref).max() > 0
