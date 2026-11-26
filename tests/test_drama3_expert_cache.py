"""The vendored expert cache: byte-exact expert reads from safetensors and the LRU budget (no model needed)."""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")

from shutter.families.drama3 import expert_cache as EC  # noqa: E402


def test_enabled_reads_the_env(monkeypatch):
    monkeypatch.delenv("TF_D3_EXPERT_CACHE_GB", raising=False)
    assert not EC.enabled()
    monkeypatch.setenv("TF_D3_EXPERT_CACHE_GB", "18")
    assert EC.enabled()


def test_expert_reads_are_byte_exact(tmp_path):
    w = mx.arange(4 * 3 * 5, dtype=mx.uint32).reshape(4, 3, 5)
    s = (mx.arange(4 * 3 * 2, dtype=mx.float32).reshape(4, 3, 2) / 7).astype(mx.bfloat16)
    mx.save_safetensors(str(tmp_path / "a.safetensors"), {"x.weight": w, "x.scales": s})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"x.weight": "a.safetensors", "x.scales": "a.safetensors"}}))
    f = EC._Files(tmp_path)
    for e in range(4):
        assert mx.array_equal(f.expert("x.weight", e), w[e]).item()
        got = f.expert("x.scales", e)
        assert got.dtype == mx.bfloat16 and mx.array_equal(got, s[e]).item()


def test_cache_keeps_the_budget_and_never_drops_what_this_step_uses():
    c = EC._Cache(budget_bytes=3 * 400)
    t = lambda: (mx.zeros((100,), dtype=mx.float32),)        # 400 bytes per expert
    for e in range(5):
        c.get((0, e), t)
    c.trim({(0, 3), (0, 4)})
    assert c.used <= c.budget and (0, 3) in c.items and (0, 4) in c.items
    c.get((0, 4), t)
    assert c.hits == 1 and c.misses == 5
