#!/usr/bin/env python3
"""Copy the tensors of a PyTorch .pth checkpoint into a .safetensors file without PyTorch and without running the
pickle's code.

The pickle in a .pth zip is read with an unpickler that only knows the handful of names a plain state dict uses
(OrderedDict, torch._utils._rebuild_tensor_v2, the storage classes); anything else stops the read. Tensor bytes are
copied straight from the zip members (which torch stores uncompressed), one tensor at a time, so memory stays small
even for an 11 GB file. dtypes are kept (bf16 stays bf16).

    python scripts/pth_to_safetensors.py models_t5_umt5-xxl-enc-bf16.pth text_encoder.safetensors
"""

from __future__ import annotations

import collections
import json
import pickle
import struct
import sys
import zipfile
from pathlib import Path

# storage class name -> (safetensors dtype, bytes per element)
DTYPES = {
    "FloatStorage": ("F32", 4), "HalfStorage": ("F16", 2), "BFloat16Storage": ("BF16", 2),
    "DoubleStorage": ("F64", 8), "LongStorage": ("I64", 8), "IntStorage": ("I32", 4),
    "ShortStorage": ("I16", 2), "CharStorage": ("I8", 1), "ByteStorage": ("U8", 1), "BoolStorage": ("BOOL", 1),
}


class _StorageType:
    def __init__(self, name: str):
        self.name = name


class _Storage:
    def __init__(self, kind: str, key: str):
        self.kind, self.key = kind, key


class _Tensor:
    def __init__(self, storage: _Storage, offset: int, size: tuple, stride: tuple):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild_tensor_v2(storage, offset, size, stride, *_args):
    return _Tensor(storage, offset, size, stride)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_v2
        if module == "torch" and name in DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        st = pid[1]
        name = st.name if isinstance(st, _StorageType) else str(st)
        if name not in DTYPES:
            raise pickle.UnpicklingError(f"unsupported storage {name}")
        return _Storage(name, str(pid[2]))


def _contiguous(t: _Tensor) -> bool:
    exp = 1
    for s, st in zip(reversed(t.size), reversed(t.stride)):
        if s != 1 and st != exp:
            return False
        exp *= s
    return True


def read_pth(path: Path):
    """(zipfile, prefix, {name: _Tensor}) for a .pth zip."""
    z = zipfile.ZipFile(path)
    pkl = next(n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl")
    prefix = pkl[: -len("data.pkl")]
    with z.open(pkl) as f:
        sd = SafeUnpickler(f).load()
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    tensors = {k: v for k, v in sd.items() if isinstance(v, _Tensor)}
    if len(tensors) != len(sd):
        raise SystemExit(f"{path}: {len(sd) - len(tensors)} entries are not tensors")
    return z, prefix, tensors


def convert(src: Path, out: Path) -> int:
    z, prefix, tensors = read_pth(src)
    header, plan, pos = {}, [], 0
    for name, t in tensors.items():
        if not _contiguous(t):
            raise SystemExit(f"{name}: non-contiguous tensor, not supported")
        dt, nb = DTYPES[t.storage.kind]
        n = 1
        for s in t.size:
            n *= s
        header[name] = {"dtype": dt, "shape": list(t.size), "data_offsets": [pos, pos + n * nb]}
        plan.append((t, n * nb, nb))
        pos += n * nb
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out, "wb") as o:
        o.write(struct.pack("<Q", len(hb)))
        o.write(hb)
        for t, size, nb in plan:
            info = z.getinfo(f"{prefix}data/{t.storage.key}")
            if info.compress_type != zipfile.ZIP_STORED:
                raise SystemExit("compressed .pth members are not supported")
            with z.open(info) as f:
                f.seek(t.offset * nb)
                left = size
                while left:
                    chunk = f.read(min(left, 64 << 20))
                    if not chunk:
                        raise SystemExit(f"short read in storage {t.storage.key}")
                    o.write(chunk)
                    left -= len(chunk)
    return len(tensors)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    n = convert(Path(argv[0]), Path(argv[1]))
    print(f"wrote {n} tensors to {argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
