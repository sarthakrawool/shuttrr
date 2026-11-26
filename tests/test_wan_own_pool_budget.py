"""A Wan model already in the pool (loaded on purpose, or warm from the last request) is counted in the rung
estimates; its bytes must not also be missing from the budget, or a loaded 5B refuses its own request."""

from __future__ import annotations

import types
from pathlib import Path

from shutter.adapters.wan import WanAdapter
from shutter.types import MemoryEstimate
from shutter.types import JobRequest

GB = 1024**3


def _adapter(monkeypatch, entries):
    ad = WanAdapter("wan2.2-ti2v-5b", Path("/nonexistent"), {})
    ests = {r: MemoryEstimate(peak=int(g * GB)) for r, g in
            (("resident", 13.0), ("split", 13.0), ("staged", 6.9), ("stream_wide", 6.9), ("stream_tight", 6.3))}
    monkeypatch.setattr(ad, "rung_estimates", lambda req, s=None: ests)
    monkeypatch.setattr(ad, "budget_for", lambda req, ctx=None: 5 * GB)
    monkeypatch.setattr(ad, "mode_defaults", lambda ctx=None: {"mode": "fast", "efficiency": 0.5})
    pool = types.SimpleNamespace(stats=lambda: {"entries": entries})
    return ad, types.SimpleNamespace(models=pool, settings={})


def test_own_loaded_dit_counts_toward_the_budget(monkeypatch):
    ad, ctx = _adapter(monkeypatch, [{"key": "wan/wan2.2-ti2v-5b/dit/resident", "size_bytes": 5 * GB},
                                     {"key": "wan/other-model/dit/resident", "size_bytes": 3 * GB}])
    pick = ad.choose_rung(JobRequest("wan2.2-ti2v-5b", prompt="x"), ctx)
    assert pick["budget"] == 10 * GB and pick["rung"] == "staged" and pick["fits"]


def test_nothing_loaded_keeps_the_budget(monkeypatch):
    ad, ctx = _adapter(monkeypatch, [])
    pick = ad.choose_rung(JobRequest("wan2.2-ti2v-5b", prompt="x"), ctx)
    assert pick["budget"] == 5 * GB and not pick["fits"]


def test_explicit_budget_is_not_changed(monkeypatch):
    ad, ctx = _adapter(monkeypatch, [{"key": "wan/wan2.2-ti2v-5b/dit/resident", "size_bytes": 5 * GB}])
    pick = ad.choose_rung(JobRequest("wan2.2-ti2v-5b", prompt="x", params={"budget_bytes": 5 * GB}), ctx)
    assert pick["budget"] == 5 * GB
