"""MiniMax-H3-Image reference pictures: layout, tiling, rotary and request plumbing (no model weights needed)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from shutter.families.h3image import refs as R  # noqa: E402
from shutter.families.h3image.model import MiniMaxH3ImageDiT, spatial_axis  # noqa: E402
from shutter.families.h3image.pipeline import TorchRandn, torch_randn  # noqa: E402
from shutter.families.zimage.model import te_rotary  # noqa: E402


def test_bound_short_edge_matches_release():
    assert R.bound_short_edge(1024, 768, 1024, 2048) == (1376, 1024)      # short edge raised to 1024
    assert R.bound_short_edge(1500, 1200, 1024, 2048) == (1500, 1200)     # inside the bounds: unchanged
    assert R.bound_short_edge(4000, 3000, 1024, 1024) == (1376, 1024)     # vision copy: short edge 1024


def test_split_tiles_cover_exactly_with_latent_aligned_overlaps():
    for length in (256, 512, 1024, 1008, 1536, 2048, 1376):
        starts, overlaps = R.split_tiles(length)
        assert starts[0] == 0
        assert starts[-1] + min(R.TILE, length) == length
        assert all(o >= R.TILE_OVERLAP and o % R.RATIO == 0 for o in overlaps)
    assert R.split_tiles(1024) == ([0, 192, 384, 576, 768], [64, 64, 64, 64])
    assert R.split_tiles(200) == ([0], [])


def test_mrope_text_only_is_plain_rotary():
    ids = list(range(37))
    cos, sin = R.mrope_cos_sin(R.mrope_positions(ids, []), 128, 5e6, mx.float32)
    c2, s2 = te_rotary(128, 5e6, 37, mx.float32)
    assert np.allclose(np.array(cos), np.array(c2), atol=2e-6)
    assert np.allclose(np.array(sin), np.array(s2), atol=2e-6)


def test_mrope_positions_image_block():
    # text, text, <start>, 2x3 image block, <end>, text
    ids = [0, 0, 1] + [9] * 6 + [2, 0]
    pos = R.mrope_positions(ids, [(3, 2, 3)])
    assert pos[:, :3].tolist() == [[0, 1, 2]] * 3
    assert pos[0, 3:9].tolist() == [3] * 6
    assert pos[1, 3:9].tolist() == [3, 3, 3, 4, 4, 4]
    assert pos[2, 3:9].tolist() == [3, 4, 5, 3, 4, 5]
    assert pos[:, 9].tolist() == [6, 6, 6] and pos[:, 10].tolist() == [7, 7, 7]


def test_layout_references_take_one_time_step_each():
    pos, n = MiniMaxH3ImageDiT.layout(5, 8, 8, refs=((4, 6), (8, 8)))
    p = np.array(pos)
    assert n == 16
    assert p.shape == (5 + 6 + 16 + 16, 3)
    assert p[:5, 0].tolist() == [0, 1, 2, 3, 4]
    assert set(p[5:11, 0].tolist()) == {5.0} and set(p[11:27, 0].tolist()) == {6.0}
    assert set(p[27:, 0].tolist()) == {7.0}
    # each reference on its own aspect-normalized grid
    import math
    sq = math.sqrt(4 * 6)
    assert np.allclose(sorted(set(p[5:11, 1].tolist())), spatial_axis(4, sq))
    # no references: the 0.3.x layout
    pos0, n0 = MiniMaxH3ImageDiT.layout(5, 8, 8)
    assert np.array(pos0).shape == (21, 3) and set(np.array(pos0)[5:, 0].tolist()) == {5.0}


def test_torch_randn_stream_continues_like_one_draw():
    g = TorchRandn(7)
    a, b = g.randn(32), g.randn(48)
    assert np.allclose(np.concatenate([a, b]), torch_randn(7, 80))


def test_reference_latents_crop_and_normalize():
    m = np.zeros((48, 5, 7), np.float32)
    z = R.reference_latents(m, np.zeros(24 * 5 * 7, np.float32))
    assert z.shape == (24, 4, 6)


def test_vision_patches_shape_and_order():
    from PIL import Image
    img = Image.new("RGB", (64, 32), (255, 0, 0))
    p, grid = R.vision_patches(img)
    assert grid == (1, 2, 4) and p.shape == (8, 3 * 2 * 16 * 16)
    # channel-major (C, T, P, P): red is +1, green / blue -1
    assert np.allclose(p[0, :512], 1.0) and np.allclose(p[0, 512:], -1.0)


def test_reference_paths(tmp_path):
    from shutter.adapters.h3image import MAX_REFERENCES, reference_paths
    from shutter.types import JobRequest
    files = []
    for i in range(4):
        f = tmp_path / f"p{i}.png"
        f.write_bytes(b"x")
        files.append(str(f))
    req = JobRequest(model="m", inputs={"image_2": files[1], "image": files[0]}, params={"images": [files[2]]})
    assert reference_paths(req) == [files[0], files[1], files[2]]
    with pytest.raises(ValueError):
        reference_paths(JobRequest(model="m", params={"images": files[:MAX_REFERENCES + 1]}))
    with pytest.raises(ValueError):
        reference_paths(JobRequest(model="m", params={"images": [str(tmp_path / "missing.png")]}))


def test_velocity_shapes_with_references():
    """A tiny DiT runs a step with a reference and keeps the generated rows' shape."""
    cfg = dict(hidden_size=64, num_layers=1, token_refiner_num_layers=1, num_attention_heads=1, attention_head_dim=128,
               ffn_hidden_size=64, text_dim=32, adaln_curve_grid=9, rope_inv_freq_len=16)
    dit = MiniMaxH3ImageDiT(**cfg)
    mx.eval(dit.parameters())
    text = dit.refine_text(mx.zeros((1, 3, 32)))
    lat = mx.zeros((24, 4, 4))
    v0 = dit.velocity(lat, text, 1.0, 0.9675236940383911)
    v1 = dit.velocity(lat, text, 1.0, 0.9675236940383911, refs=[mx.zeros((24, 2, 6))], text_tags=[1, 0, 1])
    assert v0.shape == v1.shape == (24, 4, 4)
