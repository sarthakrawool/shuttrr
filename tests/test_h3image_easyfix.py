"""Texture fix for MiniMax-H3-Image text to image (shutter/families/h3image/easyfix.py)."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from shutter.families.h3image import easyfix as EF  # noqa: E402
from shutter.families.h3image.model import PDD_SIGMAS  # noqa: E402


def test_table_blocks_and_unit_directions():
    for step in (0, 1):
        t = EF.load_table(step)
        assert sorted(t) == list(range(25, 45))
        for u, _, _ in t.values():
            assert abs(float(mx.sqrt((u * u).sum())) - 1.0) < 1e-4


def test_plans():
    assert EF.parse_plan("RNmid") == "SSRRRS"
    assert EF.parse_plan("lcm") == "RRRRRS"          # the last step always runs the sampler
    assert EF.parse_plan("off") == "SSSSSS"
    assert EF.parse_plan("sr") == "SRSSSS"     # short plans are padded with S
    with pytest.raises(ValueError):
        EF.parse_plan("SSXRRS")


def test_from_params():
    opt = lambda k, d: d  # noqa: E731
    assert EF.from_params({"texture_fix": False}, opt) is None
    assert EF.from_params({"texture_fix": "off"}, opt) is None
    f = EF.from_params({}, opt)
    assert f.describe() == {"steer": 0.7, "steer_steps": [0], "plan": "SSRRRS", "noise": 0.95,
                            "sampler": "euler_ancestral"}
    with pytest.raises(ValueError):
        EF.from_params({"texture_sampler": "dpm"}, opt)
    with pytest.raises(ValueError):
        EF.from_params({"texture_steer_steps": [2]}, opt)


def _exact(x0, n, s):
    """latents on the flow path at sigma s and the exact velocity there."""
    return x0 * (1 - s) + n * s, n - x0


def test_steps_stay_on_the_flow_path():
    """With the exact velocity every step type gives x0 * (1 - s1) plus noise of variance s1^2."""
    rng = np.random.default_rng(0)
    x0 = mx.array(rng.standard_normal(200_000).astype(np.float32))
    n = mx.array(rng.standard_normal(200_000).astype(np.float32))
    fresh = lambda: mx.array(rng.standard_normal(200_000).astype(np.float32))  # noqa: E731
    fix = EF.TextureFix(noise=1.0)
    for sampler in ("euler", "euler_ancestral"):
        fix.sampler = sampler
        for letter in "SR":
            s, s1 = PDD_SIGMAS[1], PDD_SIGMAS[2]
            lat, v = _exact(x0, n, s)
            out = EF.step(lat, v, s, s1, letter, fix, fresh)
            rest = np.array(out - x0 * (1 - s1))
            assert abs(rest.std() - s1) < 0.01
    lat, v = _exact(x0, n, PDD_SIGMAS[5])
    assert np.allclose(np.array(EF.step(lat, v, PDD_SIGMAS[5], 0.0, "S", fix, fresh)), np.array(x0), atol=1e-5)


def test_steering_moves_projection_to_target():
    fix = EF.TextureFix(steer=1.0)
    stats = []
    hook = EF.steering_hook(fix, 0, stats)
    assert EF.steering_hook(fix, 3) is None
    assert EF.steering_hook(EF.TextureFix.off(), 0) is None
    u, c_target, _ = EF.load_table(0)[30]
    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((1, 40 + 64, 5376)).astype(np.float32))
    y = hook(30, x, 64)
    assert np.array_equal(np.array(y[:, :40]), np.array(x[:, :40]))       # text rows untouched
    assert abs(float(EF.trimmed_mean(y[0, 40:]) @ u) - c_target) < 1.0
    assert np.array_equal(np.array(hook(10, x, 64)), np.array(x))         # blocks outside 25-44 untouched
