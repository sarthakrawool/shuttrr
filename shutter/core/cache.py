"""Content-addressed array cache on disk (prompt embeddings, audio features, ...).

A value is a dict of arrays (numpy or MLX) plus optional JSON-able metadata under
"__meta__". It is stored as an uncompressed .npz at `root/<namespace>/<key[:2]>/<key>.npz`,
where key = sha256 of the canonical JSON of [namespace, key_parts]. `pathlib.Path` items in
key_parts are replaced by the sha256 of the file's content, so the same audio clip hits the
cache under any file name.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

__all__ = ["ContentCache", "file_digest", "cache_key", "GIB"]

GIB = 1024**3
META_KEY = "__meta__"
_HEADER = "__shutter_header__"   # JSON string array inside the npz: dtypes + meta
_SUFFIX = ".npz"

# -- file digests --------------------------------------------------------------------------

_digest_lock = threading.Lock()
_digest_memo: dict[tuple[str, int, int], str] = {}


def file_digest(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """sha256 hex digest of a file's content, memoised by (path, size, mtime_ns)."""
    p = os.path.abspath(os.fspath(path))
    st = os.stat(p)
    memo_key = (p, st.st_size, st.st_mtime_ns)
    with _digest_lock:
        hit = _digest_memo.get(memo_key)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    digest = h.hexdigest()
    with _digest_lock:
        if len(_digest_memo) > 10000:
            _digest_memo.clear()
        _digest_memo[memo_key] = digest
    return digest


def _canon(obj: Any) -> Any:
    if isinstance(obj, Path):
        return {"file": file_digest(obj)}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"cache key dict keys must be str, got {type(k).__name__}")
            out[k] = _canon(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise TypeError("cache key floats must be finite")
        return obj
    if isinstance(obj, np.generic):
        return _canon(obj.item())
    raise TypeError(f"cache key part of type {type(obj).__name__} is not JSON-serialisable")


def cache_key(namespace: str, key_parts: list[Any]) -> str:
    """sha256 hex of the canonical JSON of [namespace, key_parts]."""
    payload = json.dumps([namespace, _canon(list(key_parts))], sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -- MLX helpers (optional dependency) -----------------------------------------------------

def _mx():
    try:
        import mlx.core as mx  # type: ignore
        return mx
    except Exception:
        return None


def _is_mlx_array(v: Any) -> bool:
    t = type(v)
    return t.__name__ == "array" and t.__module__.startswith("mlx")


def _to_numpy(name: str, v: Any) -> tuple[np.ndarray, Optional[str]]:
    """Convert one value to numpy; returns (array, original dtype name if it must be
    restored on load)."""
    if _is_mlx_array(v):
        mx = _mx()
        if mx is not None and v.dtype == mx.bfloat16:
            return np.array(v.astype(mx.float32)), "bfloat16"
        return np.array(v), None
    arr = np.asarray(v)
    if arr.dtype == object:
        raise TypeError(f"cache value {name!r} is not a numeric/string array")
    return arr, None


def _to_output(arr: np.ndarray, orig_dtype: Optional[str], as_mlx: bool) -> Any:
    if not as_mlx:
        return arr
    mx = _mx()
    if mx is None:
        raise RuntimeError("as_mlx=True but mlx is not importable")
    if arr.dtype.kind in "US":  # strings stay numpy
        return arr
    out = mx.array(arr)
    if orig_dtype is not None:
        out = out.astype(getattr(mx, orig_dtype))
    return out


# -- the cache -----------------------------------------------------------------------------

class ContentCache:
    """Implements `adapters.base.CacheLike`. Thread-safe; safe against concurrent writers
    (each write goes to its own temp file and is published with os.replace)."""

    def __init__(self, root: str | os.PathLike, max_bytes: int = 20 * GIB, as_mlx: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.as_mlx = as_mlx
        self._lock = threading.RLock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._bytes = self._scan_bytes()

    # -- paths ---------------------------------------------------------------------------

    @staticmethod
    def _check_namespace(namespace: str) -> None:
        if (not namespace or not isinstance(namespace, str) or "/" in namespace
                or "\\" in namespace or namespace in (".", "..") or namespace.startswith(".")):
            raise ValueError(f"invalid cache namespace: {namespace!r}")

    def path_for(self, namespace: str, key: str) -> Path:
        self._check_namespace(namespace)
        return self.root / namespace / key[:2] / f"{key}{_SUFFIX}"

    def _files(self, base: Optional[Path] = None) -> list[Path]:
        base = base or self.root
        if not base.exists():
            return []
        return [p for p in base.rglob(f"*{_SUFFIX}") if p.is_file()]

    def _scan_bytes(self) -> int:
        total = 0
        for p in self._files():
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    # -- encode / decode -----------------------------------------------------------------

    @staticmethod
    def _encode(value: dict[str, Any]) -> dict[str, np.ndarray]:
        if not isinstance(value, dict):
            raise TypeError("compute() must return a dict of arrays")
        arrays: dict[str, np.ndarray] = {}
        dtypes: dict[str, str] = {}
        meta = value.get(META_KEY)
        for name, v in value.items():
            if name == META_KEY:
                continue
            if not isinstance(name, str) or name == _HEADER:
                raise ValueError(f"invalid array name: {name!r}")
            arr, orig = _to_numpy(name, v)
            arrays[name] = arr
            if orig:
                dtypes[name] = orig
        header = json.dumps({"v": 1, "dtypes": dtypes, "meta": meta}, allow_nan=False)
        arrays[_HEADER] = np.array(header)
        return arrays

    def _decode(self, data: dict[str, np.ndarray], as_mlx: bool) -> dict[str, Any]:
        header = json.loads(str(data.pop(_HEADER)[()]))
        dtypes = header.get("dtypes", {})
        out: dict[str, Any] = {
            name: _to_output(arr, dtypes.get(name), as_mlx) for name, arr in data.items()
        }
        if header.get("meta") is not None:
            out[META_KEY] = header["meta"]
        return out

    def _read(self, path: Path) -> Optional[dict[str, np.ndarray]]:
        """Load a cache file fully into memory. Returns None if missing; removes and
        returns None if corrupt."""
        try:
            with np.load(path, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            if _HEADER not in data:
                raise ValueError("missing header")
            json.loads(str(data[_HEADER][()]))
            return data
        except FileNotFoundError:
            return None
        except Exception:
            self._remove(path)
            return None

    def _remove(self, path: Path) -> None:
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            return
        with self._lock:
            self._bytes = max(0, self._bytes - size)

    def _write(self, path: Path, arrays: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.stem[:16]}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **arrays)
                f.flush()
                os.fsync(f.fileno())
            try:
                old = path.stat().st_size
            except OSError:
                old = 0
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        new = path.stat().st_size
        with self._lock:
            self._bytes += new - old

    # -- public API ----------------------------------------------------------------------

    def get(self, namespace: str, key_parts: list[Any], *, as_mlx: Optional[bool] = None
            ) -> Optional[dict[str, Any]]:
        """Cached value or None (does not count as a miss)."""
        as_mlx = self.as_mlx if as_mlx is None else as_mlx
        path = self.path_for(namespace, cache_key(namespace, key_parts))
        data = self._read(path)
        if data is None:
            return None
        self._touch(path)
        with self._lock:
            self._hits += 1
        return self._decode(data, as_mlx)

    def put(self, namespace: str, key_parts: list[Any], value: dict[str, Any]) -> Path:
        path = self.path_for(namespace, cache_key(namespace, key_parts))
        self._write(path, self._encode(value))
        if self._bytes > self.max_bytes:
            self.evict()
        return path

    def get_or_compute(self, namespace: str, key_parts: list[Any],
                       compute: Callable[[], dict[str, Any]], *, as_mlx: Optional[bool] = None
                       ) -> dict[str, Any]:
        """Return the cached value for (namespace, key_parts) or compute, store and return
        it. Arrays come back as numpy (or mx.array with as_mlx=True)."""
        as_mlx = self.as_mlx if as_mlx is None else as_mlx
        key = cache_key(namespace, key_parts)
        path = self.path_for(namespace, key)

        data = self._read(path)
        if data is None:
            with self._lock:
                klock = self._key_locks.setdefault(key, threading.Lock())
            with klock:
                data = self._read(path)  # another thread may have just written it
                if data is None:
                    with self._lock:
                        self._misses += 1
                    arrays = self._encode(compute())
                    self._write(path, arrays)
                    if self._bytes > self.max_bytes:
                        self.evict(keep=path)
                    return self._decode(dict(arrays), as_mlx)
        self._touch(path)
        with self._lock:
            self._hits += 1
        return self._decode(data, as_mlx)

    def _touch(self, path: Path) -> None:
        try:
            os.utime(path, None)
        except OSError:
            pass

    def evict(self, keep: Optional[Path] = None) -> list[Path]:
        """Remove least-recently-used files (by mtime; hits touch the file) until the
        cache is at or below max_bytes. `keep` is never removed."""
        removed: list[Path] = []
        with self._lock:
            entries = []
            total = 0
            for p in self._files():
                try:
                    st = p.stat()
                except OSError:
                    continue
                total += st.st_size
                entries.append((st.st_mtime_ns, str(p), p, st.st_size))
            self._bytes = total
            entries.sort()
            for _, _, p, size in entries:
                if self._bytes <= self.max_bytes:
                    break
                if keep is not None and p == keep:
                    continue
                try:
                    p.unlink()
                except OSError:
                    continue
                self._bytes -= size
                self._evictions += 1
                removed.append(p)
        return removed

    def stats(self) -> dict[str, Any]:
        with self._lock:
            files = self._files()
            namespaces: dict[str, dict[str, int]] = {}
            total = 0
            for p in files:
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                total += size
                ns = p.relative_to(self.root).parts[0]
                d = namespaces.setdefault(ns, {"entries": 0, "bytes": 0})
                d["entries"] += 1
                d["bytes"] += size
            self._bytes = total
            lookups = self._hits + self._misses
            return {
                "root": str(self.root),
                "entries": sum(d["entries"] for d in namespaces.values()),
                "bytes": total,
                "max_bytes": self.max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / lookups) if lookups else 0.0,
                "evictions": self._evictions,
                "namespaces": namespaces,
            }

    def clear(self, namespace: Optional[str] = None) -> None:
        with self._lock:
            if namespace is None:
                for child in self.root.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        try:
                            child.unlink()
                        except OSError:
                            pass
            else:
                self._check_namespace(namespace)
                shutil.rmtree(self.root / namespace, ignore_errors=True)
            self._bytes = self._scan_bytes()
