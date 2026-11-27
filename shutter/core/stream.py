"""SSD weight streaming for large transformer stacks.

A video denoise step walks the DiT blocks strictly in order and spends seconds of GPU
time per block, while an NVMe read of one block's weights takes ~0.1 s. So instead of
keeping every block resident we keep a small sliding window of blocks in memory and read
the next block on a background thread while the current one computes.

Pieces:

* :class:`SafetensorsIndex` parses ``.safetensors`` headers (no tensor data is read) and
  reads single tensors with ``os.pread`` (which releases the GIL).
* :class:`BlockStreamer` keeps groups ``i+1 .. i+readahead`` prefetched, at most
  ``window`` groups resident, and records how long the compute thread was blocked on I/O.
* :func:`group_names_by_block` splits tensor names into per-block groups.
* :class:`StreamedStack` runs a stack of identical blocks through one or two reusable
  module "shells" whose parameters are swapped per block. Output is bit-identical to the
  fully loaded stack because the very same bytes and the very same ops are used.

Thread safety: the reader thread only produces numpy buffers (pure ``os.pread`` into
preallocated memory). Every ``mx.array`` is created on the caller's thread, inside
``BlockStreamer.block()``. MLX streams/devices are per thread on some builds, so creating
arrays off the compute thread is avoided on purpose. The cost is one host memcpy per
tensor on the caller thread (recorded as ``convert_seconds``).
"""

from __future__ import annotations

import json
import os
import re
import struct
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TensorEntry",
    "SafetensorsIndex",
    "BlockStreamer",
    "SlotStreamer",
    "StreamError",
    "group_names_by_block",
    "default_strip_prefix",
    "StreamedStack",
    "to_mlx_array",
]

# safetensors dtype -> (numpy dtype used for the raw bytes, itemsize)
_NP_DTYPES: Dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),  # numpy has no bfloat16: keep the bits as uint16
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "U16": np.dtype("<u2"),
    "U32": np.dtype("<u4"),
    "U64": np.dtype("<u8"),
    "BOOL": np.dtype("?"),
}

_F_NOCACHE = 48  # macOS fcntl.h; only used on darwin

_MAX_HEADER = 100 * 1024 * 1024


class StreamError(RuntimeError):
    """Raised on the caller thread when the background reader failed."""


@dataclass(frozen=True)
class TensorEntry:
    file: str
    offset: int  # absolute byte offset in ``file``
    nbytes: int
    dtype: str  # safetensors dtype string, e.g. "BF16"
    shape: Tuple[int, ...]


# ---------------------------------------------------------------------------
# safetensors index
# ---------------------------------------------------------------------------


class SafetensorsIndex:
    """Header-only index over one or many ``.safetensors`` files."""

    def __init__(self, paths, nocache: bool = False):
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        self.paths: List[str] = [os.fspath(p) for p in paths]
        if not self.paths:
            raise ValueError("SafetensorsIndex needs at least one file")
        self.entries: Dict[str, TensorEntry] = {}
        self.metadata: Dict[str, str] = {}
        self._fds: Dict[str, int] = {}
        self._fd_lock = threading.Lock()
        self._nocache = bool(nocache)
        self._closed = False
        for path in self.paths:
            self._parse(path)

    def _parse(self, path: str) -> None:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f"{path}: not a safetensors file (too short)")
            (hlen,) = struct.unpack("<Q", raw)
            if hlen > _MAX_HEADER or 8 + hlen > size:
                raise ValueError(f"{path}: invalid safetensors header length {hlen}")
            header = json.loads(f.read(hlen))
        base = 8 + hlen
        for name, info in header.items():
            if name == "__metadata__":
                self.metadata.update(info or {})
                continue
            dtype = info["dtype"]
            if dtype not in _NP_DTYPES:
                raise ValueError(f"{path}: unsupported dtype {dtype} for {name}")
            start, end = info["data_offsets"]
            shape = tuple(int(s) for s in info["shape"])
            nbytes = end - start
            expect = _NP_DTYPES[dtype].itemsize * int(np.prod(shape, dtype=np.int64))
            if nbytes != expect:
                raise ValueError(f"{path}: {name} has {nbytes} bytes, expected {expect}")
            if name in self.entries:
                raise ValueError(f"duplicate tensor {name} in {path}")
            # Data beyond EOF (truncated file) is detected at read time, not here.
            self.entries[name] = TensorEntry(path, base + start, nbytes, dtype, shape)

    # -- queries -----------------------------------------------------------
    def names(self) -> List[str]:
        return list(self.entries)

    def nbytes(self, names: Optional[Iterable[str]] = None) -> int:
        if names is None:
            names = self.entries
        return sum(self.entries[n].nbytes for n in names)

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    # -- I/O ---------------------------------------------------------------
    def set_nocache(self, on: bool = True) -> None:
        """Ask macOS not to keep read data in the page cache (F_NOCACHE)."""
        self._nocache = bool(on)
        with self._fd_lock:
            for fd in self._fds.values():
                _apply_nocache(fd, self._nocache)

    def _fd(self, path: str) -> int:
        with self._fd_lock:
            if self._closed:
                raise ValueError("SafetensorsIndex is closed")
            fd = self._fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
                if self._nocache:
                    _apply_nocache(fd, True)
                self._fds[path] = fd
            return fd

    def read(self, name: str) -> np.ndarray:
        """Read one tensor into a fresh numpy array (BF16 comes back as uint16 bits)."""
        e = self.entries[name]
        arr = np.empty(e.shape, dtype=_NP_DTYPES[e.dtype])
        if e.nbytes:
            _pread_into(self._fd(e.file), memoryview(arr.reshape(-1).view(np.uint8)), e.offset, name)
        return arr

    def to_mlx(self, name: str):
        """Read one tensor as an ``mx.array`` with the stored dtype (bit-exact)."""
        return to_mlx_array(self.read(name), self.entries[name].dtype)

    def close(self) -> None:
        with self._fd_lock:
            self._closed = True
            fds, self._fds = list(self._fds.values()), {}
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):  # best effort, no leaked fds
        try:
            self.close()
        except Exception:
            pass


def _apply_nocache(fd: int, on: bool) -> None:
    if sys.platform != "darwin":
        return
    try:
        import fcntl

        fcntl.fcntl(fd, _F_NOCACHE, 1 if on else 0)
    except OSError:
        pass


def _pread_into(fd: int, buf: memoryview, offset: int, what: str = "") -> None:
    """Fill ``buf`` from ``fd`` at ``offset``; loops over short reads."""
    total = len(buf)
    done = 0
    has_preadv = hasattr(os, "preadv")
    while done < total:
        # keep single calls below 1 GiB (macOS caps a read at INT_MAX)
        chunk = min(total - done, 1 << 30)
        if has_preadv:
            n = os.preadv(fd, [buf[done : done + chunk]], offset + done)
        else:
            data = os.pread(fd, chunk, offset + done)
            n = len(data)
            buf[done : done + n] = data
        if n == 0:
            raise EOFError(f"unexpected end of file reading {what!r} ({done}/{total} bytes)")
        done += n


def to_mlx_array(arr: np.ndarray, dtype: str):
    """numpy buffer -> mx.array with the safetensors dtype, without changing any bit.

    BF16 is held as uint16 in numpy; ``mx.array(uint16).view(mx.bfloat16)`` is a pure
    bit reinterpretation (same itemsize), verified bit-exact in tests.
    """
    import mlx.core as mx

    if dtype == "BF16":
        return mx.array(arr).view(mx.bfloat16)
    return mx.array(arr)


# ---------------------------------------------------------------------------
# block streamer
# ---------------------------------------------------------------------------


class BlockStreamer:
    """Sliding window of per-block weight groups, prefetched on a daemon thread.

    ``window`` counts every group the streamer holds: the one(s) handed to the caller,
    those already prefetched, and one being read. With ``window=1`` nothing can be
    prefetched while a block is in use (reads then start when the block is released).
    """

    def __init__(
        self,
        index: SafetensorsIndex,
        groups: Sequence[Sequence[str]],
        window: int = 2,
        readahead: int = 1,
        cyclic: bool = True,
        to_device: bool = True,
        nocache: bool = False,
        read_fn: Optional[Callable[[str], np.ndarray]] = None,
        on_resident: Optional[Callable[[int, int], None]] = None,
        join_timeout: float = 30.0,
    ):
        if window < 1:
            raise ValueError("window must be >= 1")
        if readahead < 0:
            raise ValueError("readahead must be >= 0")
        self.index = index
        self.groups: List[List[str]] = [list(g) for g in groups]
        for g in self.groups:
            for n in g:
                if n not in index.entries:
                    raise KeyError(f"tensor {n!r} not in index")
        self.window = int(window)
        self.readahead = int(readahead)
        self.cyclic = bool(cyclic)
        self.to_device = bool(to_device)
        self._read_fn = read_fn or index.read
        self._on_resident = on_resident
        self._join_timeout = join_timeout
        self._group_bytes = [index.nbytes(g) for g in self.groups]
        if nocache:
            index.set_nocache(True)

        self._cond = threading.Condition()
        self._cache: Dict[int, Dict[str, np.ndarray]] = self._new_cache()  # prefetched, ready
        self._inflight: Optional[int] = None  # group the reader is reading now
        self._in_use: Dict[int, int] = {}  # groups handed out (or loading sync)
        self._cursor = -1  # last requested group
        self._error: Optional[BaseException] = None
        self._closed = False

        self._bytes_read = 0
        self._read_seconds = 0.0
        self._wait_seconds = 0.0
        self._convert_seconds = 0.0
        self._hits = 0
        self._misses = 0
        self._resident_bytes_peak = 0
        self._resident_groups_peak = 0

        self._thread: Optional[threading.Thread] = None
        if self.readahead > 0 and self.groups:
            # claim group 0 before the thread starts, so an immediate block(0) waits for
            # this read instead of racing it with a duplicate synchronous read
            self._inflight = 0
            self._thread = threading.Thread(target=self._reader, name="shutter-stream", daemon=True)
            self._thread.start()

    def __len__(self) -> int:
        return len(self.groups)

    def _new_cache(self) -> dict:
        return {}

    def _discard(self, data) -> None:
        """A group read after close(): nothing to give back for plain host buffers."""

    # -- residency bookkeeping (call with self._cond held) -------------------
    def _resident_ids(self) -> set:
        ids = set(self._cache) | set(self._in_use)
        if self._inflight is not None:
            ids.add(self._inflight)
        return ids

    def _note_resident(self) -> None:
        ids = self._resident_ids()
        nbytes = sum(self._group_bytes[i] for i in ids)
        self._resident_bytes_peak = max(self._resident_bytes_peak, nbytes)
        self._resident_groups_peak = max(self._resident_groups_peak, len(ids))
        if self._on_resident is not None:
            self._on_resident(len(ids), nbytes)

    def _desired(self) -> List[int]:
        n = len(self.groups)
        out = []
        for k in range(1, self.readahead + 1):
            j = self._cursor + k
            if j >= n:
                if not self.cyclic:
                    break
                j %= n
            if j not in out:
                out.append(j)
        return out

    # -- reader thread -------------------------------------------------------
    def _read_group(self, i: int) -> Dict[str, np.ndarray]:
        t0 = time.perf_counter()
        out = {name: self._read_fn(name) for name in self.groups[i]}
        dt = time.perf_counter() - t0
        with self._cond:
            self._read_seconds += dt
            self._bytes_read += self._group_bytes[i]
        return out

    def _reader(self) -> None:
        first: Optional[int] = self._inflight
        try:
            while True:
                with self._cond:
                    target, first = first, None
                    while target is None:
                        if self._closed:
                            return
                        desired = self._desired()
                        # drop prefetched groups nobody will want next
                        # (the cursor group is kept: the caller may be waking up to take it)
                        for j in list(self._cache):
                            if j not in desired and j != self._cursor:
                                del self._cache[j]
                        busy = self._resident_ids()
                        if len(busy) < self.window:
                            for j in desired:
                                if j not in busy:
                                    target = j
                                    break
                        if target is None:
                            self._cond.wait()
                    self._inflight = target
                    self._note_resident()
                data = self._read_group(target)  # GIL released inside pread
                with self._cond:
                    self._inflight = None
                    if not self._closed:
                        self._cache[target] = data
                        self._note_resident()
                    else:
                        self._discard(data)
                    self._cond.notify_all()
        except BaseException as exc:  # surface on the caller
            with self._cond:
                self._error = exc
                self._inflight = None
                self._cond.notify_all()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise StreamError(f"background weight reader failed: {self._error!r}") from self._error

    # -- caller API ----------------------------------------------------------
    def _acquire(self, i: int) -> Dict[str, np.ndarray]:
        t_wait0 = time.perf_counter()
        sync = False
        with self._cond:
            if self._closed:
                raise RuntimeError("BlockStreamer is closed")
            self._raise_if_failed()
            if i in self._in_use:
                raise RuntimeError(f"block {i} is already in use")
            self._cursor = i
            self._cond.notify_all()
            while True:
                self._raise_if_failed()
                if self._closed:
                    raise RuntimeError("BlockStreamer is closed")
                if i in self._cache:
                    data = self._cache.pop(i)
                    self._in_use[i] = 1
                    self._hits += 1
                    break
                if self._inflight == i:
                    self._cond.wait()  # being read right now: wait for it
                    continue
                # miss: make room, then read synchronously on this thread
                busy = self._resident_ids()
                if len(busy) + 1 > self.window:
                    evictable = [j for j in self._cache if j not in self._in_use]
                    if evictable:
                        del self._cache[evictable[0]]
                        continue
                    if self._inflight is not None:
                        self._cond.wait()
                        continue
                    raise RuntimeError(
                        f"window={self.window} too small: {len(self._in_use)} block(s) still in use"
                    )
                self._in_use[i] = 1
                self._misses += 1
                self._note_resident()
                sync = True
                break
        if sync:
            try:
                data = self._read_group(i)
            except BaseException:
                with self._cond:
                    self._in_use.pop(i, None)
                    self._cond.notify_all()
                raise
        with self._cond:
            self._wait_seconds += time.perf_counter() - t_wait0
        return data

    def _release(self, i: int) -> None:
        with self._cond:
            self._in_use.pop(i, None)
            self._note_resident()
            self._cond.notify_all()

    @contextmanager
    def block(self, i: int) -> Iterator[Dict[str, object]]:
        """Yield ``{name: mx.array}`` (or numpy when ``to_device=False``) for group ``i``."""
        if not 0 <= i < len(self.groups):
            raise IndexError(f"block {i} out of range (0..{len(self.groups) - 1})")
        raw = self._acquire(i)
        tensors: Dict[str, object]
        try:
            if self.to_device:
                t0 = time.perf_counter()
                entries = self.index.entries
                tensors = {n: to_mlx_array(a, entries[n].dtype) for n, a in raw.items()}
                self._convert_seconds += time.perf_counter() - t0
                raw = None  # free host buffers now
            else:
                tensors = raw
        except BaseException:
            self._release(i)
            raise
        try:
            yield tensors
        finally:
            tensors = None
            raw = None
            self._release(i)

    def stats(self) -> Dict[str, float]:
        with self._cond:
            return {
                "bytes_read": self._bytes_read,
                "read_seconds": self._read_seconds,
                "wait_seconds": self._wait_seconds,
                "convert_seconds": self._convert_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "resident_bytes_peak": self._resident_bytes_peak,
                "resident_groups_peak": self._resident_groups_peak,
            }

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cache.clear()
            self._cond.notify_all()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(self._join_timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _SlotCache(dict):
    """``BlockStreamer._cache`` for :class:`SlotStreamer`: dropping a prefetched group (``del`` / ``clear``) gives its
    slot back; ``pop`` (the caller taking the group) keeps it."""

    def __init__(self, owner: "SlotStreamer"):
        super().__init__()
        self._owner = owner

    def __delitem__(self, key):
        slot = super().pop(key)
        self._owner._free_slot(slot)

    def clear(self):
        for slot in list(self.values()):
            self._owner._free_slot(slot)
        super().clear()


_MX_DTYPES = {"F32": "float32", "F16": "float16", "BF16": "bfloat16", "I64": "int64", "I32": "int32",
              "I16": "int16", "I8": "int8", "U8": "uint8", "U16": "uint16", "U32": "uint32", "U64": "uint64",
              "BOOL": "bool_"}


class SlotStreamer(BlockStreamer):
    """BlockStreamer that reads every block straight into memory MLX already owns.

    ``window`` reusable slots are allocated once, on the caller's thread: one flat uint8 MLX buffer each, the size of
    the largest block (tensors packed at 256-byte boundaries). The reader thread fills a free slot with ``preadv`` (no
    MLX call off the caller's thread, no numpy copy, no page faults on fresh memory each block); ``block(i)`` hands out
    zero-copy views (``slab[off:off + n].view(dtype).reshape(shape)``). Same bytes, same ops: results are
    bit-identical to :class:`BlockStreamer`.

    The views are only valid inside ``with streamer.block(i)``: the slot is refilled with another block afterwards, so
    the caller must finish (``mx.eval``) the work that reads them before leaving the block and keep no reference.
    """

    _ALIGN = 256

    def __init__(self, index: SafetensorsIndex, groups: Sequence[Sequence[str]], window: int = 2,
                 readahead: int = 1, cyclic: bool = True, nocache: bool = False, **kw):
        import mlx.core as mx

        if kw.get("to_device") is False or kw.get("read_fn") is not None:
            raise ValueError("SlotStreamer always reads straight into MLX buffers")
        groups = [list(g) for g in groups]
        a = self._ALIGN
        self._layout: List[List[Tuple[str, int, int]]] = []      # per group: (name, offset in slab, nbytes)
        size = 1
        for g in groups:
            off, lay = 0, []
            for n in g:
                nb = index.entries[n].nbytes
                lay.append((n, off, nb))
                off += -(-nb // a) * a
            self._layout.append(lay)
            size = max(size, off)
        self._slot_lock = threading.Lock()
        self._bufs: List[object] = []
        self._host: List[np.ndarray] = []
        for _ in range(max(1, int(window))):
            slab = mx.zeros((size,), dtype=mx.uint8)
            mx.eval(slab)
            self._bufs.append(slab)
            self._host.append(np.array(slab, copy=False))
        self._free: List[int] = list(range(len(self._bufs)))
        self._slot_of: Dict[int, int] = {}
        self._slot_bytes = size
        super().__init__(index, groups, window=window, readahead=readahead, cyclic=cyclic, to_device=True,
                         nocache=nocache, **kw)

    def _new_cache(self) -> dict:
        return _SlotCache(self)

    def _discard(self, slot) -> None:
        self._free_slot(slot)

    # -- slots ---------------------------------------------------------------
    def _take_slot(self) -> int:
        with self._slot_lock:
            if not self._free:
                raise RuntimeError("SlotStreamer: no free slot (window accounting broken)")
            return self._free.pop()

    def _free_slot(self, slot: int) -> None:
        with self._slot_lock:
            if slot not in self._free:
                self._free.append(slot)

    @property
    def slot_bytes(self) -> int:
        """Bytes one slot holds (all slots: ``slot_bytes * window``)."""
        return self._slot_bytes

    # -- reading into a slot (reader thread, or the caller on a miss) ---------
    def _read_group(self, i: int):
        slot = self._take_slot()
        try:
            t0 = time.perf_counter()
            host = self._host[slot]
            for name, off, nb in self._layout[i]:
                if nb:
                    e = self.index.entries[name]
                    _pread_into(self.index._fd(e.file), memoryview(host[off: off + nb]), e.offset, name)
            dt = time.perf_counter() - t0
        except BaseException:
            self._free_slot(slot)
            raise
        with self._cond:
            self._read_seconds += dt
            self._bytes_read += self._group_bytes[i]
        return slot

    def _release(self, i: int) -> None:
        with self._cond:
            self._in_use.pop(i, None)
            slot = self._slot_of.pop(i, None)
            if slot is not None:
                self._free_slot(slot)
            self._note_resident()
            self._cond.notify_all()

    @contextmanager
    def block(self, i: int) -> Iterator[Dict[str, object]]:
        """Yield ``{name: mx.array}`` views of group ``i`` in its slot (valid only inside the ``with``)."""
        import mlx.core as mx

        if not 0 <= i < len(self.groups):
            raise IndexError(f"block {i} out of range (0..{len(self.groups) - 1})")
        slot = self._acquire(i)
        with self._cond:
            self._slot_of[i] = slot
        try:
            t0 = time.perf_counter()
            slab = self._bufs[slot]
            tensors: Dict[str, object] = {}
            for name, off, nb in self._layout[i]:
                e = self.index.entries[name]
                tensors[name] = slab[off: off + nb].view(getattr(mx, _MX_DTYPES[e.dtype])).reshape(e.shape)
            self._convert_seconds += time.perf_counter() - t0
        except BaseException:
            self._release(i)
            raise
        try:
            yield tensors
        finally:
            tensors = None
            self._release(i)

    def close(self) -> None:
        super().close()
        self._bufs = []
        self._host = []


# ---------------------------------------------------------------------------
# grouping helpers
# ---------------------------------------------------------------------------


def group_names_by_block(
    names: Iterable[str], pattern: str = r"(^|\.)blocks\.(\d+)\."
) -> Tuple[List[List[str]], List[str]]:
    """Split names into per-block groups (ordered by block number) and the rest.

    The block number is the last capture group of ``pattern``.
    """
    rx = re.compile(pattern)
    by_block: Dict[int, List[str]] = {}
    rest: List[str] = []
    for n in names:
        m = rx.search(n)
        if m is None:
            rest.append(n)
        else:
            by_block.setdefault(int(m.group(m.lastindex)), []).append(n)
    return [by_block[k] for k in sorted(by_block)], rest


_STRIP_RX = re.compile(r"^(.*?\.)?blocks\.\d+\.")


def default_strip_prefix(name: str) -> str:
    """``"model.blocks.7.attn.q.weight"`` -> ``"attn.q.weight"``."""
    return _STRIP_RX.sub("", name, count=1)


# ---------------------------------------------------------------------------
# streamed stack
# ---------------------------------------------------------------------------


class StreamedStack:
    """Run ``len(streamer)`` structurally identical blocks through reusable shells.

    Each shell is an ``nn.Module`` with the block's structure (already quantized with
    ``nn.quantize`` if the stored blocks are quantized). For block ``i`` the shell
    ``shells[i % len(shells)]`` gets block ``i``'s tensors, ``apply(shell, x, i)`` runs
    (default: ``shell(x)``), and the result is evaluated with ``mx.eval`` before moving
    on, so the block's weights can be freed. With ``release_weights=True`` (default) the
    shell's arrays are replaced with empty placeholders after each block, so no block's
    weights outlive its ``streamer.block()`` context.
    """

    def __init__(
        self,
        shells,
        streamer: BlockStreamer,
        strip_prefix_fn: Optional[Callable[[str], str]] = None,
        release_weights: bool = True,
    ):
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        self._mx = mx
        self._tree_unflatten = tree_unflatten
        if not isinstance(shells, (list, tuple)):
            shells = [shells]
        if not shells:
            raise ValueError("need at least one shell")
        self.shells = list(shells)
        self.streamer = streamer
        self.strip = strip_prefix_fn or default_strip_prefix
        self.release_weights = release_weights
        # expected parameter names + shapes, captured from the shells' initial params
        self._expected: List[Dict[str, Tuple[int, ...]]] = []
        for s in self.shells:
            self._expected.append({k: tuple(v.shape) for k, v in tree_flatten(s.parameters())})
        self._placeholder = mx.zeros((0,), dtype=mx.float32)
        if release_weights:
            for si in range(len(self.shells)):
                self._clear(si)

    def _clear(self, si: int) -> None:
        names = self._expected[si]
        self.shells[si].update(self._tree_unflatten([(k, self._placeholder) for k in names]), strict=False)

    def _load(self, shell, si: int, tensors: Dict[str, object], block: int) -> None:
        expected = self._expected[si]
        pairs = []
        for name, arr in tensors.items():
            local = self.strip(name)
            if local not in expected:
                raise ValueError(f"block {block}: unexpected tensor {name!r} (-> {local!r})")
            if tuple(arr.shape) != expected[local]:
                raise ValueError(
                    f"block {block}: {name!r} has shape {tuple(arr.shape)}, shell expects {expected[local]}"
                )
            pairs.append((local, arr))
        missing = set(expected) - {k for k, _ in pairs}
        if missing:
            raise ValueError(f"block {block}: missing tensors {sorted(missing)}")
        shell.update(self._tree_unflatten(pairs), strict=True)

    def run(self, x, apply: Optional[Callable] = None):
        mx = self._mx
        apply = apply or (lambda shell, h, i: shell(h))
        for i in range(len(self.streamer)):
            si = i % len(self.shells)
            shell = self.shells[si]
            with self.streamer.block(i) as tensors:
                self._load(shell, si, tensors, i)
                x = apply(shell, x, i)
                mx.eval(x)
                if self.release_weights:
                    self._clear(si)
                tensors = None
        return x

    __call__ = run
