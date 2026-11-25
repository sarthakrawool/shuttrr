import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from shutter.kernels import flops as F

ROOT = Path(__file__).resolve().parent.parent


def test_token_counts_match_the_model():
    c = F.Config()
    assert c.latent_frames == 24
    assert c.grid == (24, 44, 82)
    assert c.tokens == 86_592
    assert c.cond_tokens == 44 * 82 and c.noise_tokens == 86_592 - 3_608
    c480 = F.Config(height=480, width=832)
    assert c480.grid == (24, 30, 52) and c480.tokens == 37_440


def test_ffn_width_and_head_dim():
    assert F.swiglu_hidden(4096, 4) == 11008
    assert F.Config().head_dim == 128


def test_attention_pairs_and_macs():
    c = F.Config()
    n, nn_, nc = c.tokens, c.noise_tokens, c.cond_tokens
    assert c.attn_pairs == nn_ * n + nc * nc
    ops = {o.name: o for o in F.block_ops(c)}
    assert ops["attn.sdpa.qk"].macs == c.heads * c.attn_pairs * c.head_dim
    assert ops["attn.sdpa.av"].macs == ops["attn.sdpa.qk"].macs
    # without the cond split it is the plain N^2 * C
    c0 = replace(c, num_cond_latents=0)
    ops0 = {o.name: o for o in F.block_ops(c0)}
    assert ops0["attn.sdpa.qk"].macs == n * n * c.hidden


def test_linear_macs_and_bytes():
    c = F.Config()
    ops = {o.name: o for o in F.block_ops(c)}
    N, C = c.tokens, c.hidden
    assert ops["attn.qkv"].macs == N * C * 3 * C
    assert ops["ffn.w1"].macs == N * C * 11008 and ops["ffn.w2"].macs == N * 11008 * C
    assert ops["cross_attn.q_linear"].macs == c.noise_tokens * C * C
    assert ops["audio_cross_attn.kv_linear"].macs == 23 * 32 * 768 * 2 * C
    # bytes: x in bf16, 8-bit weights + bf16 scale/bias per 64, y in bf16, bias
    wb = 1 + 4 / 64
    assert c.weight_bytes == pytest.approx(wb)
    assert ops["attn.proj"].bytes == pytest.approx(N * C * 2 + C * C * wb + N * C * 2 + C * 2)


def test_step_is_depth_times_block_plus_extra():
    c = F.Config()
    sc = F.step_counts(c)
    blk = sum(o.flops for o in F.block_ops(c))
    extra = sum(o.flops for o in F.step_extra_ops(c))
    assert sc["per_block_total"]["flops"] == pytest.approx(blk)
    assert sc["per_step_total"]["flops"] == pytest.approx(48 * blk + extra)
    assert sum(sc["share"].values()) == pytest.approx(1.0)
    # self-attention dominates at 86.6K tokens: QK + AV ~ 70% of FLOPs
    assert 0.65 < sc["share"]["self_attn_qk"] + sc["share"]["self_attn_av"] < 0.75


def test_headline_numbers():
    sc = F.step_counts(F.Config())
    assert sc["per_step_total"]["flops"] == pytest.approx(8.07e15, rel=0.01)
    sc480 = F.step_counts(F.Config(height=480, width=832))
    assert sc480["per_step_total"]["flops"] == pytest.approx(2.09e15, rel=0.01)


def test_ceiling_arithmetic():
    c = F.Config()
    tot = F.step_counts(c)["per_step_total"]["flops"]
    r = F.ceiling(164.0, 100e12, c)
    assert r["achieved_flops"] == pytest.approx(tot / 164.0)
    assert r["utilisation"] == pytest.approx(tot / 164.0 / 100e12)
    assert r["best_case_s"] == pytest.approx(tot / 100e12)
    assert r["max_speedup"] == pytest.approx(164.0 / r["best_case_s"])
    rb = F.ceiling(164.0, 100e12, c, bandwidth=500e9)
    assert rb["roofline_s"] >= r["best_case_s"]


def test_attention_traffic():
    c = F.Config()
    a = F.attention_traffic(c, score_bytes=4, bq=64)
    assert a["entries_per_head_full"] == 86_592 ** 2
    assert a["entries_per_head_full"] == pytest.approx(7.5e9, rel=0.01)
    qkvo = 32 * 4 * c.tokens * 128 * 2
    assert a["fused_compulsory_bytes"] == qkvo
    assert a["naive_bytes"] == 32 * c.attn_pairs * (2 * 4 + 2 * 2) + qkvo
    streamed = 32 * (math.ceil(c.noise_tokens / 64) * 2 * c.tokens * 128 * 2
                     + math.ceil(c.cond_tokens / 64) * 2 * c.cond_tokens * 128 * 2) + 32 * 2 * c.tokens * 128 * 2
    assert a["fused_streamed_bytes"] == streamed
    # intensity of a streamed tile ~ bq FLOP/byte
    assert a["intensity_streamed"] == pytest.approx(64, rel=0.02)


def test_lora_param_count_matches_the_checkpoint_size():
    # 7 targets at rank 128: 13.14 M params/block; x 48 blocks x 4 B = 2.52 GB (the dmd_lora.safetensors size)
    c = F.Config()
    C, Fh, r = 4096, 11008, 128
    per = {"attn.qkv": r * (3 * C + 3 * C), "attn.proj": r * 2 * C, "cross_attn.q_linear": r * 2 * C,
           "cross_attn.kv_linear": r * (2 * C + 2 * C), "ffn.w1": r * (C + Fh), "ffn.w2": r * (Fh + C),
           "ffn.w3": r * (C + Fh)}
    assert set(per) == set(c.lora_targets)
    total = sum(per.values())
    assert total * 48 * 4 == pytest.approx(2.523e9, rel=0.01)
    ops = {o.name: o for o in F.block_ops(c)}
    assert ops["lora.attn.qkv"].macs == c.tokens * C * 3 * r + c.tokens * r * 3 * C


def test_upper_bound():
    assert F.upper_bound(0.6, 1.0) == pytest.approx(1.0)
    assert F.upper_bound(0.6, 0.5) == pytest.approx(1 / (0.4 + 0.3))
    assert F.upper_bound(1.0, 0.25) == pytest.approx(4.0)


def test_measured_rates():
    rows = {r["part"]: r for r in F.measured_class_rates()}
    sd = rows["self-attn SDPA, per head (default)"]
    assert sd["achieved_flops"] == pytest.approx(sd["flops"] / 2.084)
    assert 50e12 < sd["achieved_flops"] < 60e12


def test_script_runs():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "flops.py")], capture_output=True, text=True,
                         check=True).stdout
    assert "86,592 tokens" in out and "37,440 tokens" in out and "ASSUMED" in out
