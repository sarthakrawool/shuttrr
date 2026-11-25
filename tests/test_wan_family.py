"""Wan family: recognition, refusals, conversion, and streamed == resident (bit-identical), on tiny random models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tests import wan_fixtures as F  # noqa: E402
from shutter.families import inspect as inspect_ckpt  # noqa: E402
from shutter.families.wan import keys as K  # noqa: E402
from shutter.families.wan.convert import ConvertError, convert  # noqa: E402
from shutter.families.wan import loader as L  # noqa: E402

# CPU by default (bit-identical checks); SHUTTER_TEST_GPU=1 runs the same tests on Metal
if not __import__("os").environ.get("SHUTTER_TEST_GPU"):
    mx.set_default_device(mx.cpu)


def _repo(tmp: Path, layout="official", dit_extra=None, vae_z=16, cfg=F.TINY) -> Path:
    w = F.official_dit_weights(cfg, extra=dit_extra)
    if layout == "comfy":
        w = F.to_comfy(w)
    elif layout == "diffusers":
        w = F.to_diffusers(w)
    root = tmp / f"repo-{layout}"
    if layout == "diffusers":
        F.save_st(root / "transformer" / "diffusion_pytorch_model.safetensors", w)
        F.save_st(root / "text_encoder" / "model.safetensors", F.t5_weights())
        F.write_tokenizer_stub(root / "tokenizer")
        F.save_st(root / "vae" / "wan_vae.safetensors", {"conv2.weight": np.zeros((vae_z, vae_z, 1, 1, 1), np.float32)})
    else:
        F.save_st(root / "diffusion_pytorch_model.safetensors", w)
        F.save_st(root / "text_encoder" / "model.safetensors", F.t5_weights())
        F.write_tokenizer_stub(root / "google" / "umt5-xxl")
        F.save_st(root / "Wan2.1_VAE.safetensors", {"conv2.weight": np.zeros((vae_z, vae_z, 1, 1, 1), np.float32),
                                                   "decoder.conv1.weight": np.ones((8, vae_z, 3, 3, 3), np.float32)})
    return root


# ----------------------------------------------------------------------------- recognition


@pytest.mark.parametrize("layout", ["official", "comfy", "diffusers"])
def test_recognise_layouts(tmp_path, layout):
    root = _repo(tmp_path, layout)
    ck = inspect_ckpt(root)
    assert ck is not None and ck.family == "wan", ck
    assert ck.ok, ck.problems
    assert ck.config["dim"] == 256 and ck.config["num_layers"] == 3 and ck.config["num_heads"] == 2
    assert ck.task == "t2v" and ck.variant == "wan-custom"          # tiny size has no preset
    assert {"dit", "text_encoder", "vae", "tokenizer"} <= set(ck.parts)


def test_known_sizes_classified():
    from shutter.families.wan.spec import WanShape, classify
    assert classify(WanShape(1536, 16, 16, 8960, 30)) == "wan2.1-t2v-1.3b"
    assert classify(WanShape(3072, 48, 48, 14336, 30)) == "wan2.2-ti2v-5b"
    assert classify(WanShape(5120, 16, 16, 13824, 40)) == "wan2.1-t2v-14b"
    assert classify(WanShape(5120, 16, 16, 13824, 40, dual=True)) == "wan2.2-t2v-a14b"
    assert classify(WanShape(5120, 36, 16, 13824, 40, dual=True)) == "wan2.2-i2v-a14b"


@pytest.mark.parametrize("extra,phrase", [
    ({"img_emb.proj.1.weight": np.zeros((256, 1280), np.float32)}, "CLIP image branch"),
    ({"vace_blocks.0.before_proj.weight": np.zeros((4, 4), np.float32)}, "VACE"),
    ({"audio_proj.weight": np.zeros((4, 4), np.float32)}, "LongCat"),
])
def test_refuses_unsupported_variants(tmp_path, extra, phrase):
    ck = inspect_ckpt(_repo(tmp_path, dit_extra=extra))
    assert not ck.ok and any(phrase in p for p in ck.problems), ck.problems


def test_refuses_wrong_vae(tmp_path):
    ck = inspect_ckpt(_repo(tmp_path, vae_z=48))
    assert any("needs the Wan 2.1 VAE" in p for p in ck.problems), ck.problems


def test_bare_finetune_file_lists_missing_parts(tmp_path):
    f = tmp_path / "my_finetune.safetensors"
    F.save_st(f, F.to_comfy(F.official_dit_weights()))
    ck = inspect_ckpt(f)
    assert ck.ok and set(ck.missing) == {"text_encoder", "vae", "tokenizer"}
    assert any("video model only" in n for n in ck.notes)


def test_gguf_refused(tmp_path):
    f = tmp_path / "wan2.1-t2v-14b-Q4_K_M.gguf"
    f.write_bytes(b"GGUF....")
    ck = inspect_ckpt(f)
    assert ck is not None and not ck.ok and "GGUF" in ck.problems[0]


def test_not_wan_returns_none(tmp_path):
    f = tmp_path / "other.safetensors"
    F.save_st(f, {"model.layers.0.weight": np.zeros((4, 4), np.float32)})
    assert inspect_ckpt(f) is None


@pytest.mark.parametrize("naming", ["official", "kohya"])
def test_lora_recognised(tmp_path, naming):
    f = tmp_path / "style.safetensors"
    F.save_st(f, F.lora_weights(naming=naming))
    ck = inspect_ckpt(f)
    assert ck is not None and ck.variant == "lora" and ck.config["dims"] == [256]


def test_diffusers_mapping_roundtrip():
    w = F.official_dit_weights()
    d = F.to_diffusers(w)
    back = {K.diffusers_to_official(k): v for k, v in d.items()}
    assert set(back) == set(w)


def test_hf_t5_mapping():
    assert K.hf_t5_to_wan("encoder.block.3.layer.1.DenseReluDense.wi_0.weight") == "blocks.3.ffn.gate.0.weight"
    assert K.t5_to_mlx("encoder.block.3.layer.1.DenseReluDense.wi_0.weight", "hf") == "blocks.3.ffn.gate_proj.weight"
    assert K.hf_t5_to_wan("shared.weight") == "token_embedding.weight"
    assert K.hf_t5_to_wan("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight") == \
        "blocks.0.pos_embedding.embedding.weight"


# ----------------------------------------------------------------------------- conversion + run


def _inputs(cfg, seed=0):
    r = np.random.default_rng(seed)
    x = mx.array(r.standard_normal((cfg["in_dim"], 2, 4, 4)).astype(np.float32))
    ctx = mx.array(r.standard_normal((7, 4096)).astype(np.float32))
    return x, ctx


def _forward(model, cfg, x, ctx, cached_kv: bool):
    from shutter.families.wan.model.dit import WanModel  # noqa: F401
    seq_len = 2 * 2 * 2
    emb = model.embed_text([ctx, ctx])
    kv = model.prepare_cross_kv(emb) if cached_kv else None
    out = model([x, x], t=mx.array([700.0, 700.0]), context=emb, seq_len=seq_len, cross_kv_caches=kv)
    mx.eval(out)
    return np.array(out[0])


@pytest.mark.parametrize("bits", [8, None])
@pytest.mark.parametrize("layout", ["official", "diffusers"])
def test_convert_and_streamed_equals_resident(tmp_path, bits, layout):
    root = _repo(tmp_path, layout)
    out = tmp_path / "build"
    man = convert(root, out, bits=bits, report=lambda s: None)
    assert (out / "shuttrr.json").exists() and man["parts"]["dit"]["n_blocks"] == 3
    ck = inspect_ckpt(out)
    assert ck.ok and not ck.needs_conversion, ck.problems
    cfg = man["config"]
    part = man["parts"]["dit"]
    x, ctx = _inputs(cfg)
    resident = L.load_resident(out, cfg, part)
    a = _forward(resident, cfg, x, ctx, cached_kv=True)
    for window in (1, 2, 8):
        streamed = L.load_streamed(out, cfg, part, window=window)
        b = _forward(streamed, cfg, x, ctx, cached_kv=False)
        streamed.blocks.close()
        assert np.array_equal(a, b), f"window {window}: max diff {np.abs(a - b).max()}"


def test_converted_matches_reference_weights(tmp_path):
    """The bf16 build holds the official weights (as bf16), mapped to the right modules."""
    root = _repo(tmp_path)
    out = tmp_path / "b"
    man = convert(root, out, bits=None, report=lambda s: None)
    m = L.load_resident(out, man["config"], man["parts"]["dit"])
    w = F.official_dit_weights()
    q = np.array(m.blocks[1].self_attn.q.weight.astype(mx.float32))
    ref = np.array(mx.array(w["blocks.1.self_attn.q.weight"]).astype(mx.bfloat16).astype(mx.float32))
    assert np.array_equal(q, ref)
    pe = np.array(m.patch_embedding_proj.weight.astype(mx.float32))
    assert pe.shape == (256, 64)
    assert m.blocks[0].modulation.dtype == mx.float32


def test_convert_refuses_with_reasons(tmp_path):
    root = _repo(tmp_path, dit_extra={"img_emb.proj.1.weight": np.zeros((256, 1280), np.float32)})
    with pytest.raises(ConvertError, match="CLIP"):
        convert(root, tmp_path / "x", report=lambda s: None)


def test_bare_file_converts_with_parts_from_elsewhere(tmp_path):
    donor = _repo(tmp_path)
    f = tmp_path / "ft.safetensors"
    F.save_st(f, F.to_comfy(F.official_dit_weights(seed=5)))
    with pytest.raises(ConvertError, match="Missing"):
        convert(f, tmp_path / "x", report=lambda s: None)
    man = convert(f, tmp_path / "y", text_encoder=donor / "text_encoder", vae=donor / "Wan2.1_VAE.safetensors",
                  tokenizer=donor / "google" / "umt5-xxl", report=lambda s: None)
    assert man["parts"]["dit"]["n_blocks"] == 3


@pytest.mark.parametrize("naming", ["official", "kohya"])
def test_runtime_lora_streamed_equals_resident(tmp_path, naming):
    root = _repo(tmp_path)
    out = tmp_path / "b"
    man = convert(root, out, bits=8, report=lambda s: None)
    lf = tmp_path / "l.safetensors"
    F.save_st(lf, F.lora_weights(naming=naming))
    lora = L.read_lora(lf, strength=0.8)
    assert set(lora) == {f"blocks.{i}.{m}" for i in range(3) for m in ("self_attn.q", "cross_attn.v", "ffn.fc1")}
    cfg, part = man["config"], man["parts"]["dit"]
    x, ctx = _inputs(cfg)
    plain = _forward(L.load_resident(out, cfg, part), cfg, x, ctx, True)
    res = L.load_resident(out, cfg, part)
    L.apply_loras_resident(res, lora)
    a = _forward(res, cfg, x, ctx, True)
    assert not np.array_equal(a, plain)                       # the LoRA does something
    st = L.load_streamed(out, cfg, part, window=2, loras=lora)
    b = _forward(st, cfg, x, ctx, False)
    st.blocks.close()
    assert np.array_equal(a, b)
    L.apply_loras_resident(res, {})                           # switching LoRAs off restores the base model
    assert np.array_equal(_forward(res, cfg, x, ctx, True), plain)


def test_merged_lora_build(tmp_path):
    root = _repo(tmp_path)
    lf = tmp_path / "l.safetensors"
    F.save_st(lf, F.lora_weights())
    man = convert(root, tmp_path / "m", bits=None, loras=[(str(lf), 1.0)], report=lambda s: None)
    assert man["loras_merged"][0]["file"] == "l.safetensors"
    m = L.load_resident(tmp_path / "m", man["config"], man["parts"]["dit"])
    w = F.official_dit_weights()
    lw = F.lora_weights()
    base = "diffusion_model.blocks.0.self_attn.q"
    ref = w["blocks.0.self_attn.q.weight"] + (2 / 4) * (lw[base + ".lora_up.weight"] @ lw[base + ".lora_down.weight"])
    got = np.array(m.blocks[0].self_attn.q.weight.astype(mx.float32))
    assert np.allclose(got, ref, atol=1e-2)


# ----------------------------------------------------------------------------- prompt encode


def test_t5_block_by_block_equals_whole():
    from shutter.core.text_encode import encode_ids
    from shutter.families.wan.model.text_encoder import T5Encoder
    w = {K.t5_to_mlx(k, "wan"): mx.array(v) for k, v in F.t5_weights().items()}
    kw = dict(vocab_size=300, dim=64, dim_attn=64, dim_ffn=128, num_heads=4, num_layers=2, num_buckets=32,
              shared_pos=False)
    ids = mx.array([[5, 17, 42, 9, 0, 0]])
    mask = mx.array([[1, 1, 1, 1, 0, 0]])
    whole = T5Encoder(**kw)
    whole.load_weights(list(w.items()))
    with mx.stream(mx.cpu):
        ref = whole(ids, mask=mask)
        mx.eval(ref)
    got = encode_ids(T5Encoder(**kw), dict(w), ids, mask, mx)
    assert np.array_equal(np.array(ref), np.array(got))
    # float32 upcast path (what Wan uses) equals a float32 whole-model run
    w16 = {k: v.astype(mx.bfloat16) for k, v in w.items()}
    whole32 = T5Encoder(**kw)
    whole32.load_weights([(k, v.astype(mx.float32)) for k, v in w16.items()])
    with mx.stream(mx.cpu):                               # encode_ids runs on the CPU stream by design
        ref32 = whole32(ids, mask=mask)
        mx.eval(ref32)
    got32 = encode_ids(T5Encoder(**kw), dict(w16), ids, mask, mx, upcast=mx.float32)
    assert np.array_equal(np.array(ref32), np.array(got32))


# ----------------------------------------------------------------------------- memory


def test_memory_steps_order_for_a14b():
    from shutter.core import parts as P
    from shutter.families.wan.spec import WanShape, part_sizes, stage_needs
    cfg = WanShape(5120, 16, 16, 13824, 40, dual=True).to_config("wan2.2-t2v-a14b")
    ps = part_sizes(cfg)
    needs = stage_needs(cfg, 480, 832, 81)
    peaks = {r: P.estimate(ps, needs, r)["peak"] for r in ("resident", "split", "staged", "stream_wide", "stream_tight")}
    assert peaks["resident"] > peaks["split"] >= peaks["staged"] > peaks["stream_wide"] > peaks["stream_tight"]
    # one 8-bit expert of 14B is about 15 GB
    assert 13 * P.GB < ps[0].total < 17 * P.GB
