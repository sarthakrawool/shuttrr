"""A small GGUF reader: the header, and tensor data for the types the H3-Image DiT file uses (Q8_0, BF16, F16, F32).

Reads headers only until asked for a tensor; tensor data is memory-mapped, so converting one tensor at a time never
holds the 22 GB file in memory.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30
TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
              12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 24: "I8", 30: "BF16"}
#: (elements per block, bytes per block)
BLOCK = {GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_BF16: (1, 2), GGML_Q8_0: (32, 34)}

_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


@dataclass
class GGUFTensor:
    name: str
    shape: tuple[int, ...]       # numpy order (slowest first): ggml ne reversed, e.g. (out, in) for a weight
    type: int
    offset: int                  # absolute byte offset in the file
    nbytes: int

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type, str(self.type))


class GGUFFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta: dict[str, Any] = {}
        self.tensors: dict[str, GGUFTensor] = {}
        with open(self.path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"{self.path} is not a GGUF file")
            version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
            if version < 2:
                raise ValueError(f"GGUF version {version} is not supported")

            def rd(fmt: str):
                return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

            def rstr() -> str:
                return f.read(rd("Q")).decode("utf-8", "replace")

            def rval(t: int):
                if t == 8:
                    return rstr()
                if t == 9:
                    et, n = rd("I"), rd("Q")
                    return [rval(et) for _ in range(n)]
                return rd(_SCALARS[t])

            for _ in range(n_kv):
                k = rstr()
                self.meta[k] = rval(rd("I"))
            infos = []
            for _ in range(n_tensors):
                name = rstr()
                nd = rd("I")
                ne = [rd("Q") for _ in range(nd)]
                t, off = rd("I"), rd("Q")
                infos.append((name, ne, t, off))
            align = int(self.meta.get("general.alignment", 32))
            data_start = (f.tell() + align - 1) // align * align
        for name, ne, t, off in infos:
            n = int(np.prod(ne)) if ne else 1
            per, size = BLOCK_ALL.get(t, (0, 0))
            nbytes = n // per * size if per else 0
            self.tensors[name] = GGUFTensor(name, tuple(reversed(ne)), t, data_start + off, nbytes)
        self._fd = None

    def raw(self, name: str) -> np.ndarray:
        """The tensor's bytes, read now with F_NOCACHE (a 22 GB file read once would otherwise fill macOS's file
        cache and push other apps' memory out)."""
        import os
        from shutter.core.stream import _apply_nocache, _pread_into
        t = self.tensors[name]
        if t.type not in BLOCK:
            raise ValueError(f"{name}: GGUF type {t.type_name} is not supported by this reader")
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY)
            _apply_nocache(self._fd, True)
        buf = np.empty(t.nbytes, dtype=np.uint8)
        _pread_into(self._fd, memoryview(buf), t.offset, name)
        return buf

    def array(self, name: str) -> np.ndarray:
        """F32 / F16 / BF16 tensors as numpy (BF16 comes back as float32, exactly)."""
        t = self.tensors[name]
        b = self.raw(name)
        if t.type == GGML_F32:
            return b.view(np.float32).reshape(t.shape)
        if t.type == GGML_F16:
            return b.view(np.float16).reshape(t.shape)
        if t.type == GGML_BF16:
            return (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(t.shape)
        if t.type == GGML_Q8_0:
            d, q = self.q8_0(name)
            return (q.astype(np.float32).reshape(*d.shape, 32) * d.astype(np.float32)[..., None]).reshape(t.shape)
        raise ValueError(f"{name}: unsupported type {t.type_name}")

    def q8_0(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """A Q8_0 matrix as (scales float16 [rows, cols/32], values int8 [rows, cols])."""
        t = self.tensors[name]
        if t.type != GGML_Q8_0:
            raise ValueError(f"{name} is {t.type_name}, not Q8_0")
        rows, cols = (int(np.prod(t.shape[:-1])), t.shape[-1])
        blocks = self.raw(name).reshape(rows, cols // 32, 34)
        d = np.ascontiguousarray(blocks[:, :, :2]).view(np.float16).reshape(rows, cols // 32)
        q = np.ascontiguousarray(blocks[:, :, 2:]).view(np.int8).reshape(rows, cols)
        return d, q

    def close(self) -> None:
        import os
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


# ---------------------------------------------------------------------------------------------------------------------
# reading a GGUF as a stream (e.g. `ssh host cat file.gguf | python -m ... convert --dit -`): header first, then every
# tensor in file order, without the file ever being on this disk
# ---------------------------------------------------------------------------------------------------------------------

class _StreamReader:
    def __init__(self, f):
        self.f = f
        self.pos = 0

    def read(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            b = self.f.read(n - len(out))
            if not b:
                raise EOFError(f"stream ended at byte {self.pos + len(out)} (wanted {n} more)")
            out += b
        self.pos += n
        return bytes(out)

    def skip_to(self, offset: int) -> None:
        if offset < self.pos:
            raise ValueError(f"stream is past offset {offset} (at {self.pos})")
        while self.pos < offset:
            self.read(min(1 << 20, offset - self.pos))


def stream_tensors(f):
    """Yield (GGUFTensor, raw bytes as uint8 array) for every tensor of a GGUF read from file object `f`, in file
    order. Header parsing matches GGUFFile."""
    r = _StreamReader(f)

    def rd(fmt: str):
        return struct.unpack("<" + fmt, r.read(struct.calcsize("<" + fmt)))[0]

    def rstr() -> str:
        return r.read(rd("Q")).decode("utf-8", "replace")

    def rval(t: int):
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("I"), rd("Q")
            return [rval(et) for _ in range(n)]
        return rd(_SCALARS[t])

    if r.read(4) != b"GGUF":
        raise ValueError("not a GGUF stream")
    version, n_tensors, n_kv = struct.unpack("<IQQ", r.read(20))
    meta = {}
    for _ in range(n_kv):
        k = rstr()
        meta[k] = rval(rd("I"))
    infos = []
    for _ in range(n_tensors):
        name = rstr()
        nd = rd("I")
        ne = [rd("Q") for _ in range(nd)]
        t, off = rd("I"), rd("Q")
        infos.append((name, ne, t, off))
    align = int(meta.get("general.alignment", 32))
    data_start = (r.pos + align - 1) // align * align
    tensors = []
    for name, ne, t, off in infos:
        n = int(np.prod(ne)) if ne else 1
        per, size = BLOCK_ALL.get(t, (0, 0))
        if not per:
            raise ValueError(f"{name}: GGUF type {TYPE_NAMES.get(t, t)} is not supported")
        tensors.append(GGUFTensor(name, tuple(reversed(ne)), t, data_start + off, n // per * size))
    for tt in sorted(tensors, key=lambda x: x.offset):
        r.skip_to(tt.offset)
        yield tt, np.frombuffer(r.read(tt.nbytes), dtype=np.uint8)


GGML_Q4_K, GGML_Q6_K = 12, 14
BLOCK_ALL = {**BLOCK, GGML_Q4_K: (256, 144), GGML_Q6_K: (256, 210)}


def decode(t: GGUFTensor, raw: np.ndarray) -> np.ndarray:
    """Raw bytes of an F32 / F16 / BF16 / Q8_0 / Q4_K / Q6_K tensor -> float32 numpy array of t.shape."""
    if t.type == GGML_F32:
        return raw.view(np.float32).reshape(t.shape).copy()
    if t.type == GGML_F16:
        return raw.view(np.float16).astype(np.float32).reshape(t.shape)
    if t.type == GGML_BF16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(t.shape)
    if t.type == GGML_Q8_0:
        b = raw.reshape(-1, 34)
        d = np.ascontiguousarray(b[:, :2]).view(np.float16).astype(np.float32)
        q = np.ascontiguousarray(b[:, 2:]).view(np.int8).astype(np.float32)
        return (q * d).reshape(t.shape)
    if t.type == GGML_Q4_K:
        sc, mn, q = q4_k_parts(raw)
        return (q.astype(np.float32) * sc[..., None] - mn[..., None]).reshape(t.shape)
    if t.type == GGML_Q6_K:
        return q6_k_dequant(raw).reshape(t.shape)
    raise ValueError(f"{t.name}: unsupported type {t.type_name}")


def q8_0_parts(raw: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = int(np.prod(shape[:-1])), shape[-1]
    b = raw.reshape(rows, cols // 32, 34)
    d = np.ascontiguousarray(b[:, :, :2]).view(np.float16).reshape(rows, cols // 32)
    q = np.ascontiguousarray(b[:, :, 2:]).view(np.int8).reshape(rows, cols)
    return d, q


def q4_k_parts(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q4_K super-blocks -> per 32-value group (scale = d * sc, min = dmin * m) as float32 [nb, 8] and the 4-bit
    values uint8 [nb, 8, 32] in element order: w = scale * q - min (ggml dequantize_row_q4_K)."""
    b = raw.reshape(-1, 144)
    d = np.ascontiguousarray(b[:, 0:2]).view(np.float16).astype(np.float32)          # [nb, 1]
    dmin = np.ascontiguousarray(b[:, 2:4]).view(np.float16).astype(np.float32)
    s = b[:, 4:16].astype(np.uint16)                                                  # 12 scale bytes
    qs = b[:, 16:144]                                                                 # 128 bytes
    sc = np.empty((b.shape[0], 8), np.uint16)
    m = np.empty((b.shape[0], 8), np.uint16)
    sc[:, :4] = s[:, 0:4] & 63
    m[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    m[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    chunks = qs.reshape(-1, 4, 32)
    q = np.empty((b.shape[0], 8, 32), np.uint8)
    q[:, 0::2] = chunks & 0xF
    q[:, 1::2] = chunks >> 4
    return d * sc.astype(np.float32), dmin * m.astype(np.float32), q


def q6_k_dequant(raw: np.ndarray) -> np.ndarray:
    """ggml dequantize_row_q6_K, vectorized: float32 [nb * 256]."""
    b = raw.reshape(-1, 210)
    ql = b[:, 0:128].astype(np.int16)
    qh = b[:, 128:192].astype(np.int16)
    scales = np.ascontiguousarray(b[:, 192:208]).view(np.int8).astype(np.float32)    # [nb, 16]
    d = np.ascontiguousarray(b[:, 208:210]).view(np.float16).astype(np.float32)      # [nb, 1]
    out = np.empty((b.shape[0], 256), np.float32)
    for half in range(2):
        L = ql[:, half * 64:(half + 1) * 64]
        H = qh[:, half * 32:(half + 1) * 32]
        sc = scales[:, half * 8:(half + 1) * 8]
        l = np.arange(32)
        is_ = l // 16
        q1 = ((L[:, :32] & 0xF) | (((H >> 0) & 3) << 4)) - 32
        q2 = ((L[:, 32:] & 0xF) | (((H >> 2) & 3) << 4)) - 32
        q3 = ((L[:, :32] >> 4) | (((H >> 4) & 3) << 4)) - 32
        q4 = ((L[:, 32:] >> 4) | (((H >> 6) & 3) << 4)) - 32
        base = half * 128
        out[:, base + 0:base + 32] = d * sc[:, is_ + 0] * q1
        out[:, base + 32:base + 64] = d * sc[:, is_ + 2] * q2
        out[:, base + 64:base + 96] = d * sc[:, is_ + 4] * q3
        out[:, base + 96:base + 128] = d * sc[:, is_ + 6] * q4
    return out.reshape(-1)
