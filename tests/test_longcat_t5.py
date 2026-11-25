"""Block-by-block umT5 encode equals the whole-model encode (real umT5 code from the reference tree, tiny size)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
REAL_CODE = Path("/home/claude/ref/shutter-ref/longcat-mlx-code")
pytestmark = pytest.mark.skipif(not (REAL_CODE / "longcat_video_avatar" / "models" / "umt5.py").exists(),
                                reason="reference LongCat tree not present")

CFG = {"vocab_size": 300, "d_model": 64, "d_kv": 8, "num_heads": 8, "d_ff": 128, "num_layers": 5,
       "relative_attention_num_buckets": 32}


@pytest.fixture
def umt5_cls():
    sys.path.insert(0, str(REAL_CODE))
    try:
        from longcat_video_avatar.models.umt5 import UMT5EncoderModel
        yield UMT5EncoderModel
    finally:
        sys.path.remove(str(REAL_CODE))
        for name in [n for n in sys.modules if n == "longcat_video_avatar" or n.startswith("longcat_video_avatar.")]:
            del sys.modules[name]     # other tests import a fake tree under the same name


def test_streamed_encode_is_bit_identical(tmp_path, umt5_cls):
    from mlx.utils import tree_flatten

    from shutter.adapters import longcat_t5 as T

    mx.random.seed(3)
    ref = umt5_cls.from_config(CFG)
    ref.set_dtype(mx.bfloat16)
    flat = dict(tree_flatten(ref.parameters()))
    mx.eval(flat)
    names = sorted(flat)
    folder = tmp_path / "text_encoder"
    folder.mkdir()
    half = len(names) // 2
    wm = {}
    for shard, part in (("model-00001.safetensors", names[:half]), ("model-00002.safetensors", names[half:])):
        mx.save_safetensors(str(folder / shard), {k: flat[k] for k in part})
        wm.update({k: shard for k in part})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))

    ids = mx.array(np.random.default_rng(0).integers(0, 300, size=(1, 40)))
    mask = mx.array(np.concatenate([np.ones((1, 30), np.int64), np.zeros((1, 10), np.int64)], axis=1))
    with mx.stream(mx.cpu):
        want = ref(ids, mask=mask)
        mx.eval(want)

    seen = []
    fresh = umt5_cls.from_config(CFG)
    weights = T.lazy_weights(folder, "model.safetensors.index.json", mx)
    got = T.encode_ids(fresh, weights, ids, mask, mx, on_block=lambda i, n: seen.append((i, n)))
    assert got.dtype == want.dtype and got.shape == want.shape
    assert np.array_equal(np.array(got.astype(mx.float32)), np.array(want.astype(mx.float32)))
    assert seen == [(i, 5) for i in range(1, 6)]
    assert all(b is None for b in fresh.blocks) and not [k for k in weights if k.startswith("blocks.")]


def test_install_swaps_and_restores():
    from shutter.adapters import longcat_t5 as T

    class Mod:
        @staticmethod
        def encode_prompt(components, prompt, cache_dir=None):
            return "original"

    m = Mod()
    assert T.install(m, True) is False          # a tree without _prompt_clean keeps its own encode
    assert m.encode_prompt("c", "p") == "original"
