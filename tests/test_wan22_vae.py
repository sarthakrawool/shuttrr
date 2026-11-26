"""Wan 2.2 VAE decoder (48 channels): the frame-by-frame streamed decode against the all-at-once decode, and the
banded conv. Tiny random decoders on the CPU (SHUTTER_TEST_GPU=1: Metal)."""

from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from shutter.families.wan.model import vae22 as V  # noqa: E402

if not os.environ.get("SHUTTER_TEST_GPU"):
    mx.set_default_device(mx.cpu)


def _decoder(seed=0, dec_dim=16):
    mx.random.seed(seed)
    vae = V.Wan22VAEDecoder(z_dim=48, dec_dim=dec_dim)
    vae.load_weights([(k, mx.random.normal(v.shape) * (0.5 / max(1, (v.size // v.shape[0]) ** 0.5))
                       + (1.0 if k.endswith("gamma") else 0.0)) for k, v in tree_flatten(vae.parameters())])
    return vae


@pytest.mark.parametrize("T", [2, 3])
def test_stream_matches_all_at_once(T):
    vae = _decoder()
    z = mx.random.normal((1, T, 2, 3, 48))
    a = vae(z)
    b = V.decode_stream(_decoder(), z)        # a fresh copy: the stream decoder re-lays out conv kernels
    mx.eval(a, b)
    assert b.shape == a.shape == (1, 1 + 4 * (T - 1), 32, 48, 3)
    # same windows and ops per frame; only the batching of 1x1 / upsample convs differs (float rounding)
    assert mx.max(mx.abs(a - b)).item() < 1e-4


def test_one_latent_frame_is_one_frame():
    # the reference decodes 1 latent frame into 1 frame (the vendored all-at-once path made 3 when T == 1)
    assert V.decode_stream(_decoder(), mx.random.normal((1, 1, 2, 2, 48))).shape == (1, 1, 32, 32, 3)


def test_stream_sink_and_determinism():
    z = mx.random.normal((1, 3, 2, 2, 48))
    got = []
    V.decode_stream(_decoder(), z, got.append)
    assert len(got) == 9 and all(f.shape == (1, 1, 32, 32, 3) for f in got)
    again = V.decode_stream(_decoder(), z)
    assert mx.array_equal(mx.concatenate(got, axis=1), again).item()


def test_kernels_marker_and_reload():
    vae = _decoder()
    conv = vae.decoder.conv1
    w5 = conv.weight
    k1 = conv._kernels()
    assert conv.weight.size == 0 and len(k1) == 3                 # 5D weight not held twice
    assert mx.array_equal(k1[1], w5[:, 1]).item()
    conv.weight = w5 * 2                                          # new weights -> new kernels
    assert mx.array_equal(conv._kernels()[1], w5[:, 1] * 2).item()


def test_banded_conv(monkeypatch):
    mx.random.seed(3)
    x = mx.random.normal((1, 42, 20, 8))
    w = mx.random.normal((8, 3, 3, 8))
    full = mx.conv_general(x, w)
    monkeypatch.setattr(V, "BAND_BYTES", 1)                       # force bands
    monkeypatch.setattr(V, "BAND_ROWS_MIN", 7)
    banded = V.conv2d_banded(x, w)
    mx.eval(full, banded)
    assert banded.shape == full.shape
    assert mx.max(mx.abs(full - banded)).item() < 1e-4
    # strided (encoder downsample) too
    xs = mx.random.normal((1, 41, 21, 8))
    assert mx.max(mx.abs(V.conv2d_banded(xs, w, (2, 2)) - mx.conv_general(xs, w, stride=(2, 2)))).item() < 1e-4


def test_fused_norm_silu_is_exact():
    import mlx.nn as nn
    mx.random.seed(5)
    n = V.RMS_norm(32)
    n.gamma = mx.random.normal((32,)) * 0.1 + 1
    x = mx.random.normal((1, 1, 6, 7, 32))
    assert mx.array_equal(V._norm_silu(x, n.gamma, n.scale), nn.silu(n(x))).item()


def test_single_frame_conv_matches_chunk_conv():
    """_cconv1 (padded history, fused sum) == CausalConv3d on the concatenated chunk, frame by frame."""
    vae = _decoder(seed=7)
    conv = vae.decoder.upsamples[0].upsamples[0].residual.layer_2
    xs = mx.random.normal((1, 4, 3, 5, conv._kernels()[0].shape[-1]))
    whole = conv(xs)                                   # all frames at once, zero history
    cache, out = [None], []
    for t in range(4):
        out.append(V._cconv1(conv, xs[:, t:t + 1], cache, [0]))
    assert mx.array_equal(mx.concatenate(out, axis=1), whole).item()
