import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from shutter.core.cache import ContentCache, cache_key, file_digest


@pytest.fixture
def cache(tmp_path):
    return ContentCache(tmp_path / "cache")


def counting(value):
    calls = []

    def compute():
        calls.append(1)
        return value

    return compute, calls


def test_miss_then_hit(cache):
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    compute, calls = counting({"emb": arr, "__meta__": {"tokens": 7}})
    out1 = cache.get_or_compute("prompt", ["hello", 1], compute)
    out2 = cache.get_or_compute("prompt", ["hello", 1], compute)
    assert calls == [1]
    for out in (out1, out2):
        assert isinstance(out["emb"], np.ndarray)
        np.testing.assert_array_equal(out["emb"], arr)
        assert out["emb"].dtype == np.float32
        assert out["__meta__"] == {"tokens": 7}
    s = cache.stats()
    assert s["hits"] == 1 and s["misses"] == 1 and s["entries"] == 1
    assert s["namespaces"]["prompt"]["entries"] == 1


def test_different_keys_and_namespaces(cache):
    c1, n1 = counting({"x": np.zeros(2)})
    cache.get_or_compute("a", ["k"], c1)
    cache.get_or_compute("a", ["k2"], c1)
    cache.get_or_compute("b", ["k"], c1)
    assert len(n1) == 3


def test_key_canonical_dict_order():
    assert cache_key("n", [{"a": 1, "b": 2}]) == cache_key("n", [{"b": 2, "a": 1}])
    assert cache_key("n", [(1, 2)]) == cache_key("n", [[1, 2]])
    assert cache_key("n", ["x"]) != cache_key("m", ["x"])


def test_key_rejects_non_json(cache):
    with pytest.raises(TypeError):
        cache_key("n", [object()])
    with pytest.raises(TypeError):
        cache.get_or_compute("n", [{1, 2}], lambda: {"x": np.zeros(1)})


def test_file_content_keys(tmp_path, cache):
    a = tmp_path / "voice.wav"
    b = tmp_path / "renamed_copy.wav"
    a.write_bytes(b"RIFF" + bytes(range(256)) * 10)
    b.write_bytes(a.read_bytes())
    compute, calls = counting({"feat": np.ones(4)})
    cache.get_or_compute("audio", [a, 16000], compute)
    cache.get_or_compute("audio", [b, 16000], compute)
    assert calls == [1]
    # path string (not Path) is just a string: different key
    cache.get_or_compute("audio", [str(a), 16000], compute)
    assert calls == [1, 1]
    # content change -> new key
    b.write_bytes(b"different")
    cache.get_or_compute("audio", [b, 16000], compute)
    assert len(calls) == 3


def test_file_digest(tmp_path):
    p = tmp_path / "f.bin"
    data = os.urandom(3 * (1 << 20) + 17)
    p.write_bytes(data)
    import hashlib

    assert file_digest(p) == hashlib.sha256(data).hexdigest()
    assert file_digest(p, chunk=1000) == hashlib.sha256(data).hexdigest()
    p.write_bytes(b"new content")
    os.utime(p, ns=(1, 1))
    assert file_digest(p) == hashlib.sha256(b"new content").hexdigest()


def test_file_digest_memoised(tmp_path, monkeypatch):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    first = file_digest(p)
    import builtins

    real_open = builtins.open
    opened = []

    def spy(*a, **k):
        opened.append(a[0])
        return real_open(*a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    assert file_digest(p) == first
    assert str(p) not in [str(x) for x in opened]


def test_layout_on_disk(cache):
    cache.get_or_compute("prompt", ["x"], lambda: {"a": np.zeros(1)})
    key = cache_key("prompt", ["x"])
    path = cache.root / "prompt" / key[:2] / f"{key}.npz"
    assert path.is_file()
    assert cache.path_for("prompt", key) == path
    # no temp files left behind
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_invalid_namespace(cache):
    for bad in ("", "../x", "a/b", ".hidden"):
        with pytest.raises(ValueError):
            cache.get_or_compute(bad, ["k"], lambda: {"a": np.zeros(1)})


def test_persists_across_instances(tmp_path):
    c1 = ContentCache(tmp_path / "c")
    c1.get_or_compute("n", ["k"], lambda: {"a": np.arange(3)})
    c2 = ContentCache(tmp_path / "c")
    compute, calls = counting({"a": np.zeros(3)})
    out = c2.get_or_compute("n", ["k"], compute)
    assert calls == []
    np.testing.assert_array_equal(out["a"], np.arange(3))
    assert c2.stats()["bytes"] > 0


def test_various_dtypes_and_scalars(cache):
    val = {
        "i8": np.arange(5, dtype=np.int8),
        "f16": np.ones((2, 2), dtype=np.float16),
        "b": np.array([True, False]),
        "scalar": 3.5,
        "s": np.array(["a", "bc"]),
    }
    cache.get_or_compute("n", ["k"], lambda: val)
    out = cache.get_or_compute("n", ["k"], lambda: pytest.fail("should hit"))
    assert out["i8"].dtype == np.int8
    assert out["f16"].dtype == np.float16
    assert out["b"].dtype == bool
    assert float(out["scalar"]) == 3.5
    assert list(out["s"]) == ["a", "bc"]
    assert "__meta__" not in out


def test_object_values_rejected(cache):
    with pytest.raises(TypeError):
        cache.get_or_compute("n", ["k"], lambda: {"a": np.array([{"x": 1}], dtype=object)})
    with pytest.raises(TypeError):
        cache.get_or_compute("n", ["k2"], lambda: [1, 2])


def test_compute_error_stores_nothing(cache):
    def boom():
        raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError):
        cache.get_or_compute("n", ["k"], boom)
    assert cache.stats()["entries"] == 0
    assert list(cache.root.rglob("*.tmp")) == []


def test_corrupt_file_is_miss_and_removed(cache):
    cache.get_or_compute("n", ["k"], lambda: {"a": np.arange(4)})
    path = cache.path_for("n", cache_key("n", ["k"]))
    path.write_bytes(b"this is not a zip file")
    compute, calls = counting({"a": np.arange(4) * 2})
    out = cache.get_or_compute("n", ["k"], compute)
    assert calls == [1]
    np.testing.assert_array_equal(out["a"], np.arange(4) * 2)
    out = cache.get_or_compute("n", ["k"], compute)
    assert calls == [1]


def test_truncated_file_is_miss(cache):
    cache.get_or_compute("n", ["k"], lambda: {"a": np.arange(1000)})
    path = cache.path_for("n", cache_key("n", ["k"]))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    assert cache.get("n", ["k"]) is None
    assert not path.exists()


def test_get_and_put(cache):
    assert cache.get("n", ["k"]) is None
    cache.put("n", ["k"], {"a": np.ones(2), "__meta__": {"v": 1}})
    out = cache.get("n", ["k"])
    np.testing.assert_array_equal(out["a"], np.ones(2))
    assert out["__meta__"] == {"v": 1}


def test_atomic_write_failure_leaves_no_file(cache, monkeypatch):
    import shutter.core.cache as cache_mod

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(cache_mod.os, "replace", failing_replace)
    with pytest.raises(OSError):
        cache.get_or_compute("n", ["k"], lambda: {"a": np.zeros(10)})
    monkeypatch.undo()
    assert list(cache.root.rglob("*.npz")) == []
    assert list(cache.root.rglob("*.tmp")) == []


def test_concurrent_same_key_single_compute(cache):
    calls = []
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    def compute():
        with lock:
            calls.append(1)
        return {"a": np.arange(100000, dtype=np.float32)}

    results = []

    def worker():
        barrier.wait()
        results.append(cache.get_or_compute("n", ["same"], compute))

    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1
    for r in results:
        np.testing.assert_array_equal(r["a"], np.arange(100000, dtype=np.float32))


def test_two_caches_same_root_concurrent_writes(tmp_path):
    """Two instances (think two processes) writing the same key must not corrupt it."""
    root = tmp_path / "shared"
    caches = [ContentCache(root) for _ in range(2)]
    barrier = threading.Barrier(6)
    errors = []

    def worker(i):
        try:
            barrier.wait()
            for _ in range(10):
                caches[i % 2].put("n", ["k"], {"a": np.full(50000, 7, dtype=np.int32)})
                out = caches[(i + 1) % 2].get("n", ["k"])
                if out is not None:
                    assert (out["a"] == 7).all()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errors == []
    out = ContentCache(root).get("n", ["k"])
    assert (out["a"] == 7).all()
    assert list(root.rglob("*.tmp")) == []


def _set_mtime(cache, ns, parts, t):
    p = cache.path_for(ns, cache_key(ns, parts))
    os.utime(p, (t, t))
    return p


def test_lru_eviction(tmp_path):
    c = ContentCache(tmp_path / "c", max_bytes=10**9)
    for i in range(4):
        c.put("n", [i], {"a": np.zeros(10000, dtype=np.float64)})  # ~80 KB each
    size = c.path_for("n", cache_key("n", [0])).stat().st_size
    for i in range(4):
        _set_mtime(c, "n", [i], 1_000_000 + i)
    # hit on 0 touches it -> most recent
    assert c.get("n", [0]) is not None
    c.max_bytes = int(size * 2.5)
    removed = c.evict()
    assert len(removed) == 2
    remaining = {i for i in range(4) if c.path_for("n", cache_key("n", [i])).exists()}
    assert remaining == {0, 3}
    assert c.stats()["bytes"] <= c.max_bytes
    assert c.stats()["evictions"] == 2


def test_eviction_on_insert_keeps_new_entry(tmp_path):
    c = ContentCache(tmp_path / "c", max_bytes=10**9)
    c.put("n", ["old"], {"a": np.zeros(10000)})
    size = c.path_for("n", cache_key("n", ["old"])).stat().st_size
    _set_mtime(c, "n", ["old"], 1_000_000)
    c.max_bytes = int(size * 1.5)
    c.get_or_compute("n", ["new"], lambda: {"a": np.ones(10000)})
    assert not c.path_for("n", cache_key("n", ["old"])).exists()
    assert c.path_for("n", cache_key("n", ["new"])).exists()


def test_clear(cache):
    cache.put("a", ["k"], {"x": np.zeros(1)})
    cache.put("b", ["k"], {"x": np.zeros(1)})
    cache.clear("a")
    assert cache.get("a", ["k"]) is None
    assert cache.get("b", ["k"]) is not None
    cache.clear()
    assert cache.get("b", ["k"]) is None
    assert cache.stats()["bytes"] == 0
    assert cache.root.exists()


def test_meta_must_be_json(cache):
    with pytest.raises((TypeError, ValueError)):
        cache.get_or_compute("n", ["k"], lambda: {"a": np.zeros(1), "__meta__": {"x": object()}})


def test_npz_loads_without_pickle(cache):
    cache.put("n", ["k"], {"a": np.zeros(3), "__meta__": {"m": [1, 2]}})
    path = cache.path_for("n", cache_key("n", ["k"]))
    with np.load(path, allow_pickle=False) as z:
        assert "a" in z.files


# -- MLX -----------------------------------------------------------------------------------

def test_mlx_roundtrip(tmp_path):
    mx = pytest.importorskip("mlx.core")
    c = ContentCache(tmp_path / "c")
    a = mx.arange(6, dtype=mx.float32).reshape(2, 3)
    out = c.get_or_compute("n", ["k"], lambda: {"a": a, "i": mx.array([1, 2], dtype=mx.int32)})
    assert isinstance(out["a"], np.ndarray)
    np.testing.assert_array_equal(out["a"], np.arange(6, dtype=np.float32).reshape(2, 3))
    hit = c.get_or_compute("n", ["k"], lambda: pytest.fail("hit"), as_mlx=True)
    assert isinstance(hit["a"], mx.array)
    assert hit["a"].dtype == mx.float32 and hit["i"].dtype == mx.int32
    assert mx.array_equal(hit["a"], a).item()


def test_mlx_bfloat16_restored(tmp_path):
    mx = pytest.importorskip("mlx.core")
    c = ContentCache(tmp_path / "c", as_mlx=True)
    a = (mx.arange(8, dtype=mx.float32) / 3).astype(mx.bfloat16)
    first = c.get_or_compute("n", ["bf"], lambda: {"a": a})
    assert first["a"].dtype == mx.bfloat16
    hit = c.get_or_compute("n", ["bf"], lambda: pytest.fail("hit"))
    assert hit["a"].dtype == mx.bfloat16
    assert mx.array_equal(hit["a"], a).item()  # bf16 -> f32 -> bf16 is exact
    as_np = c.get("n", ["bf"], as_mlx=False)
    assert as_np["a"].dtype == np.float32
    np.testing.assert_array_equal(as_np["a"], np.array(a.astype(mx.float32)))


def test_mlx_and_numpy_keys_independent_of_storage(tmp_path):
    pytest.importorskip("mlx.core")
    c = ContentCache(tmp_path / "c", as_mlx=True)
    out = c.get_or_compute("n", ["k"], lambda: {"a": np.ones(3, dtype=np.float32), "__meta__": {"z": 1}})
    import mlx.core as mx

    assert isinstance(out["a"], mx.array)
    assert out["__meta__"] == {"z": 1}
