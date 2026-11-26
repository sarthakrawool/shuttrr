"""scripts/pth_to_safetensors.py: tensors out of a .pth zip without PyTorch and without running pickle code."""
import collections
import importlib.util
import io
import json
import pickle
import struct
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pth_to_safetensors", REPO / "scripts" / "pth_to_safetensors.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def _fake_pth(path: Path, arrays: dict, extra=None):
    """Write a .pth zip laid out as torch.save does, with stand-in torch names (PyTorch is not needed).
    arrays: {name: (ndarray, "FloatStorage" | "BFloat16Storage")}; bf16 arrays are given as raw uint16 bits."""
    torch_mod = types.ModuleType("torch")
    utils = types.ModuleType("torch._utils")

    def _rebuild_tensor_v2(*a):  # only its name is pickled
        raise AssertionError

    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    for n in ("FloatStorage", "BFloat16Storage"):
        c = type(n, (), {})
        c.__module__ = "torch"
        setattr(torch_mod, n, c)

    class T:
        def __init__(self, key, st, shape):
            self.key, self.st, self.shape = key, st, tuple(shape)

        def __reduce__(self):
            stride = tuple(int(np.prod(self.shape[i + 1:])) for i in range(len(self.shape)))
            return (_rebuild_tensor_v2, (("storage", self.st, self.key, "cpu", 0), 0, self.shape, stride, False,
                                         collections.OrderedDict()))

    class Pk(pickle.Pickler):
        def persistent_id(self, obj):
            return obj if isinstance(obj, tuple) and obj and obj[0] == "storage" else None

    saved = {k: sys.modules.get(k) for k in ("torch", "torch._utils")}
    sys.modules["torch"], sys.modules["torch._utils"] = torch_mod, utils
    try:
        sd, blobs = collections.OrderedDict(), {}
        for i, (name, (arr, st)) in enumerate(arrays.items()):
            sd[name] = T(str(i), getattr(torch_mod, st), arr.shape)
            blobs[str(i)] = arr.tobytes()
        if extra is not None:
            sd["extra"] = extra
        buf = io.BytesIO()
        Pk(buf, protocol=2).dump(sd)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        for k, b in blobs.items():
            z.writestr(f"archive/data/{k}", b)


def _header(path: Path) -> dict:
    raw = path.read_bytes()
    return json.loads(raw[8:8 + struct.unpack("<Q", raw[:8])[0]])


def test_copies_f32_and_bf16_bit_for_bit(tmp_path):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    bf = (np.arange(6, dtype=np.uint16) + 16256).reshape(2, 3)          # raw bf16 bits
    _fake_pth(tmp_path / "m.pth", {"w": (a, "FloatStorage"), "b": (bf, "BFloat16Storage")})
    out = tmp_path / "m.safetensors"
    assert P.convert(tmp_path / "m.pth", out) == 2
    h = _header(out)
    assert h["w"]["dtype"] == "F32" and h["w"]["shape"] == [3, 4]
    assert h["b"]["dtype"] == "BF16" and h["b"]["shape"] == [2, 3]
    raw = out.read_bytes()
    base = 8 + struct.unpack("<Q", raw[:8])[0]
    s, e = h["w"]["data_offsets"]
    np.testing.assert_array_equal(np.frombuffer(raw[base + s:base + e], np.float32).reshape(3, 4), a)
    s, e = h["b"]["data_offsets"]
    np.testing.assert_array_equal(np.frombuffer(raw[base + s:base + e], np.uint16).reshape(2, 3), bf)


def test_refuses_any_other_global(tmp_path):
    _fake_pth(tmp_path / "bad.pth", {}, extra=print)       # a pickle that would call a builtin
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        P.read_pth(tmp_path / "bad.pth")
