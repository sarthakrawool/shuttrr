"""Tests for shutter.core.stream (SSD weight streaming), on MLX CPU with tiny models."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten  # noqa: E402

from shutter.core.stream import (  # noqa: E402
    BlockStreamer,
    SafetensorsIndex,
    StreamError,
    StreamedStack,
    default_strip_prefix,
    group_names_by_block,
)

D, H, NBLOCKS = 64, 128, 6


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, H)
        self.fc2 = nn.Linear(H, D)

    def __call__(self, x):
        return x + self.fc2(nn.gelu(self.fc1(self.ln(x))))


class Model(nn.Module):
    def __init__(self, n=NBLOCKS):
        super().__init__()
        self.embed = nn.Linear(D, D)
        self.blocks = [Block() for _ in range(n)]

    def __call__(self, x):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        return x


def _make_model(dtype, quantized, seed=0):
    mx.random.seed(seed)
    m = Model()
    # non-trivial LayerNorm params so they matter
    for b in m.blocks:
        b.ln.weight = mx.random.normal((D,)) * 0.1 + 1.0
        b.ln.bias = mx.random.normal((D,)) * 0.1
    m.set_dtype(dtype)
    if quantized:
        nn.quantize(m, group_size=64, bits=8)
    mx.eval(m.parameters())
    return m


def _make_shell(dtype, quantized):
    s = Block()
    s.set_dtype(dtype)
    if quantized:
        # wrap so nn.quantize replaces the Linear children of the block
        nn.quantize(s, group_size=64, bits=8)
    return s


def _save(model, tmp_path, shards=1):
    flat = dict(tree_flatten(model.parameters()))
    names = sorted(flat)
    paths = []
    if shards == 1:
        p = str(tmp_path / "model.safetensors")
        mx.save_safetensors(p, flat)
        return [p]
    per = (len(names) + shards - 1) // shards
    for k in range(shards):
        part = {n: flat[n] for n in names[k * per : (k + 1) * per]}
        p = str(tmp_path / f"model-{k:05d}-of-{shards:05d}.safetensors")
        mx.save_safetensors(p, part)
        paths.append(p)
    return paths


def _input(dtype):
    mx.random.seed(123)
    return mx.random.normal((2, 5, D)).astype(dtype)


def _embed_from_index(index, rest, dtype, quantized):
    e = nn.Linear(D, D)
    e.set_dtype(dtype)
    if quantized:
        holder = nn.Sequential(e)
        nn.quantize(holder, group_size=64, bits=8)
        e = holder.layers[0]
    e.load_weights([(n.split(".", 1)[1], index.to_mlx(n)) for n in rest], strict=True)
    return e


# ---------------------------------------------------------------------------
# safetensors index
# ---------------------------------------------------------------------------


def test_index_parses_headers_and_reads(tmp_path):
    a = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    b = mx.array([1, -2, 3], dtype=mx.int32)
    c = mx.array([True, False, True])
    d = mx.array([7, 8], dtype=mx.uint8)
    p = str(tmp_path / "t.safetensors")
    mx.save_safetensors(p, {"a": a, "b": b, "c": c, "d": d}, metadata={"k": "v"})
    idx = SafetensorsIndex(p)
    assert set(idx.names()) == {"a", "b", "c", "d"}
    e = idx.entries["a"]
    assert e.dtype == "F32" and e.shape == (3, 4) and e.nbytes == 48 and e.file == p
    assert idx.nbytes(["a", "b"]) == 48 + 12
    assert idx.metadata == {"k": "v"}
    np.testing.assert_array_equal(idx.read("a"), np.array(a))
    for n, ref in [("a", a), ("b", b), ("c", c), ("d", d)]:
        got = idx.to_mlx(n)
        assert got.dtype == ref.dtype
        assert mx.array_equal(got, ref).item()
    idx.close()


def test_bfloat16_bit_exact_roundtrip(tmp_path):
    # every interesting bf16 pattern: zeros, denormals, inf, NaNs (incl. payloads), random
    special = np.array(
        [0x0000, 0x8000, 0x0001, 0x807F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81, 0x3F80, 0xC000],
        dtype=np.uint16,
    )
    rnd = np.random.default_rng(0).integers(0, 1 << 16, size=4096 - 11, dtype=np.uint16)
    bits = np.concatenate([special, rnd])
    t = mx.array(bits).view(mx.bfloat16).reshape(-1, 64)
    w = (mx.random.normal((33, 17))).astype(mx.bfloat16)
    p = str(tmp_path / "bf16.safetensors")
    mx.save_safetensors(p, {"t": t, "w": w})
    idx = SafetensorsIndex([p])
    assert idx.entries["t"].dtype == "BF16"
    for name, ref in [("t", t), ("w", w)]:
        got = idx.to_mlx(name)
        assert got.dtype == mx.bfloat16 and got.shape == ref.shape
        assert np.array_equal(np.array(got.view(mx.uint16)), np.array(ref.view(mx.uint16)))
        # also identical to MLX's own loader
        ref2 = mx.load(p)[name]
        assert np.array_equal(np.array(got.view(mx.uint16)), np.array(ref2.view(mx.uint16)))
    assert np.array_equal(idx.read("t").reshape(-1), bits)
    idx.close()


def test_index_multiple_shards_and_duplicates(tmp_path):
    p1, p2 = str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")
    mx.save_safetensors(p1, {"x": mx.ones((4,))})
    mx.save_safetensors(p2, {"y": mx.zeros((2, 2), dtype=mx.float16)})
    idx = SafetensorsIndex([p1, p2])
    assert idx.entries["x"].file == p1 and idx.entries["y"].file == p2
    assert idx.to_mlx("y").dtype == mx.float16
    idx.close()
    with pytest.raises(ValueError):
        SafetensorsIndex([p1, p1])


def test_index_rejects_garbage(tmp_path):
    p = str(tmp_path / "bad.safetensors")
    with open(p, "wb") as f:
        f.write((10**12).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError):
        SafetensorsIndex(p)


def test_close_releases_fds(tmp_path):
    p = str(tmp_path / "t.safetensors")
    mx.save_safetensors(p, {"a": mx.ones((8,))})
    fd_dir = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
    before = len(os.listdir(fd_dir))
    for _ in range(5):
        idx = SafetensorsIndex(p)
        idx.read("a")
        with BlockStreamer(idx, [["a"]], window=1) as s:
            with s.block(0):
                pass
        idx.close()
    assert len(os.listdir(fd_dir)) <= before
    with pytest.raises(ValueError):
        idx.read("a")


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


def test_group_names_by_block():
    names = [
        "embed.weight",
        "blocks.10.fc1.weight",
        "blocks.2.fc1.weight",
        "blocks.2.ln.bias",
        "model.blocks.0.attn.q.weight",
        "final_norm.weight",
        "subblocks.3.x",  # must not match ("blocks" not at a dotted boundary)
    ]
    groups, rest = group_names_by_block(names)
    assert groups == [["model.blocks.0.attn.q.weight"], ["blocks.2.fc1.weight", "blocks.2.ln.bias"], ["blocks.10.fc1.weight"]]
    assert rest == ["embed.weight", "final_norm.weight", "subblocks.3.x"]
    assert default_strip_prefix("model.blocks.7.attn.q.weight") == "attn.q.weight"
    assert default_strip_prefix("blocks.12.fc1.scales") == "fc1.scales"
    g2, r2 = group_names_by_block(["layers.1.a", "layers.0.b", "c"], pattern=r"(^|\.)layers\.(\d+)\.")
    assert g2 == [["layers.0.b"], ["layers.1.a"]] and r2 == ["c"]


# ---------------------------------------------------------------------------
# streamed stack == fully loaded model, bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("quantized", [False, True], ids=["plain", "q8"])
@pytest.mark.parametrize("window", [1, 2])
@pytest.mark.parametrize("shards", [1, 2])
def test_streamed_matches_reference(tmp_path, dtype, quantized, window, shards):
    model = _make_model(dtype, quantized)
    x = _input(dtype)
    ref = model(x)
    mx.eval(ref)

    paths = _save(model, tmp_path, shards)
    idx = SafetensorsIndex(paths)
    groups, rest = group_names_by_block(idx.names())
    assert len(groups) == NBLOCKS
    if quantized:
        assert any(n.endswith(".scales") for n in groups[0])

    peak_groups = []
    streamer = BlockStreamer(
        idx, groups, window=window, readahead=1, cyclic=True,
        on_resident=lambda g, b: peak_groups.append(g),
    )
    shells = [_make_shell(dtype, quantized) for _ in range(window)]
    stack = StreamedStack(shells, streamer)
    embed = _embed_from_index(idx, rest, dtype, quantized)
    with streamer:
        for _ in range(2):  # two denoise steps: second pass relies on cyclic prefetch
            out = stack.run(embed(x))
            assert out.dtype == ref.dtype
            assert mx.array_equal(out, ref).item()
        st = streamer.stats()
    assert max(peak_groups) <= window
    assert st["resident_groups_peak"] <= window
    assert st["hits"] + st["misses"] == 2 * NBLOCKS
    idx.close()


def test_two_shells_and_custom_apply(tmp_path):
    model = _make_model(mx.float32, False)
    x = _input(mx.float32)
    ref = x
    for i, b in enumerate(model.blocks):
        ref = b(ref) * (1.0 + 0.01 * i)
    mx.eval(ref)
    idx = SafetensorsIndex(_save(model, tmp_path))
    groups, _ = group_names_by_block(idx.names())
    seen = []

    def apply(shell, h, i):
        seen.append(i)
        return shell(h) * (1.0 + 0.01 * i)

    with BlockStreamer(idx, groups, window=3, readahead=2) as s:
        out = StreamedStack([_make_shell(mx.float32, False), _make_shell(mx.float32, False)], s).run(x, apply)
    assert seen == list(range(NBLOCKS))
    assert mx.array_equal(out, ref).item()
    idx.close()


def test_stack_rejects_mismatched_block(tmp_path):
    model = _make_model(mx.float32, False)
    idx = SafetensorsIndex(_save(model, tmp_path))
    groups, _ = group_names_by_block(idx.names())
    with BlockStreamer(idx, groups, window=2) as s:
        wrong = nn.Sequential(nn.Linear(D, D))  # different structure
        with pytest.raises(ValueError):
            StreamedStack([wrong], s).run(_input(mx.float32))
    idx.close()


def test_release_weights_frees_shell(tmp_path):
    model = _make_model(mx.float32, False)
    idx = SafetensorsIndex(_save(model, tmp_path))
    groups, _ = group_names_by_block(idx.names())
    shell = _make_shell(mx.float32, False)
    with BlockStreamer(idx, groups, window=2) as s:
        StreamedStack([shell], s).run(_input(mx.float32))
    assert all(v.size == 0 for _, v in tree_flatten(shell.parameters()))
    idx.close()


# ---------------------------------------------------------------------------
# streamer behaviour
# ---------------------------------------------------------------------------


def _tiny_store(tmp_path, n=NBLOCKS, per=3, size=1000, dtype=mx.float32):
    mx.random.seed(1)
    flat = {}
    for i in range(n):
        for j in range(per):
            flat[f"blocks.{i}.t{j}"] = mx.random.normal((size,)).astype(dtype)
    p = str(tmp_path / "tiny.safetensors")
    mx.save_safetensors(p, flat)
    idx = SafetensorsIndex(p)
    groups, _ = group_names_by_block(idx.names())
    return idx, groups, flat


def test_stats_bytes_and_hits(tmp_path):
    idx, groups, _ = _tiny_store(tmp_path)
    total = sum(idx.nbytes(g) for g in groups)
    passes = 3
    with BlockStreamer(idx, groups, window=2, readahead=1, cyclic=False) as s:
        for _ in range(passes):
            for i in range(len(groups)):
                with s.block(i) as t:
                    time.sleep(0.005)  # "compute", lets the reader get ahead
                    assert set(t) == set(groups[i])
        st = s.stats()
    # cyclic=False: group 0 is never prefetched after a pass, so the count is exact
    assert st["bytes_read"] == total * passes
    assert st["hits"] > 0
    assert st["hits"] + st["misses"] == passes * len(groups)
    assert st["resident_bytes_peak"] <= 2 * max(idx.nbytes(g) for g in groups)
    assert st["read_seconds"] > 0 and st["wait_seconds"] >= 0

    # cyclic: identical, except the reader may already have prefetched group 0 again
    with BlockStreamer(idx, groups, window=2, readahead=1, cyclic=True) as s:
        for _ in range(passes):
            for i in range(len(groups)):
                with s.block(i):
                    time.sleep(0.005)
        time.sleep(0.05)
        st = s.stats()
    assert st["bytes_read"] in (total * passes, total * passes + idx.nbytes(groups[0]))
    assert st["hits"] >= passes * len(groups) - 1
    idx.close()


def test_no_readahead_is_all_misses(tmp_path):
    idx, groups, _ = _tiny_store(tmp_path)
    with BlockStreamer(idx, groups, window=1, readahead=0) as s:
        assert not s.alive
        for i in range(len(groups)):
            with s.block(i):
                pass
        st = s.stats()
    assert st["misses"] == len(groups) and st["hits"] == 0
    idx.close()


def test_out_of_order_access(tmp_path):
    idx, groups, flat = _tiny_store(tmp_path)
    order = [3, 0, 5, 1, 1, 4, 2, 0, 5]
    with BlockStreamer(idx, groups, window=2, readahead=2) as s:
        for i in order:
            with s.block(i) as t:
                for n, a in t.items():
                    assert mx.array_equal(a, flat[n]).item()
            time.sleep(0.002)
        st = s.stats()
        assert st["resident_groups_peak"] <= 2
        assert st["hits"] + st["misses"] == len(order)
        with pytest.raises(IndexError):
            with s.block(len(groups)):
                pass
    idx.close()


def test_to_device_false_yields_numpy(tmp_path):
    idx, groups, flat = _tiny_store(tmp_path, dtype=mx.bfloat16)
    with BlockStreamer(idx, groups, to_device=False) as s:
        with s.block(0) as t:
            for n, a in t.items():
                assert isinstance(a, np.ndarray) and a.dtype == np.uint16
                assert np.array_equal(a, np.array(flat[n].view(mx.uint16)))
    idx.close()


def test_window_never_exceeded_under_random_access(tmp_path):
    idx, groups, _ = _tiny_store(tmp_path, n=10)
    rng = np.random.default_rng(0)
    for window, readahead in [(1, 1), (2, 1), (2, 3), (3, 2), (4, 4)]:
        counts = []
        lock = threading.Lock()

        def hook(g, b):
            with lock:
                counts.append(g)

        with BlockStreamer(idx, groups, window=window, readahead=readahead, on_resident=hook) as s:
            seq = list(range(10)) * 2 + list(rng.integers(0, 10, size=15))
            for i in seq:
                with s.block(int(i)):
                    time.sleep(0.001)
        assert counts and max(counts) <= window, (window, readahead, max(counts))
    idx.close()


def test_reader_error_surfaces_on_caller(tmp_path):
    idx, groups, _ = _tiny_store(tmp_path)
    path = idx.entries[groups[0][0]].file
    idx.close()
    # chop off the data of the last block: header still parses, the read fails
    last = SafetensorsIndex(path)
    cut = min(last.entries[n].offset for n in groups[-1]) + 10
    last.close()
    with open(path, "r+b") as f:
        f.truncate(cut)
    idx = SafetensorsIndex(path)
    s = BlockStreamer(idx, groups, window=2, readahead=1)
    with pytest.raises(StreamError) as ei:
        for i in range(len(groups)):
            with s.block(i):
                time.sleep(0.01)
    assert isinstance(ei.value.__cause__, EOFError)
    # stays failed, close does not hang
    with pytest.raises(StreamError):
        with s.block(0):
            pass
    t0 = time.perf_counter()
    s.close()
    assert time.perf_counter() - t0 < 2
    assert not s.alive
    idx.close()


def test_sync_read_error_is_raised_and_releases(tmp_path):
    idx, groups, _ = _tiny_store(tmp_path)
    calls = {"n": 0}

    def flaky(name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk hiccup")
        return idx.read(name)

    with BlockStreamer(idx, groups, window=1, readahead=0, read_fn=flaky) as s:
        with pytest.raises(OSError):
            with s.block(0):
                pass
        with s.block(0) as t:  # recovered, slot was released
            assert set(t) == set(groups[0])
    idx.close()


def test_close_while_read_pending(tmp_path):
    idx, groups, _ = _tiny_store(tmp_path)
    started = threading.Event()

    def slow(name):
        started.set()
        time.sleep(0.3)
        return idx.read(name)

    s = BlockStreamer(idx, groups, window=2, readahead=1, read_fn=slow)
    assert started.wait(2)
    t0 = time.perf_counter()
    s.close()
    assert time.perf_counter() - t0 < 3
    assert not s.alive
    with pytest.raises(RuntimeError):
        with s.block(0):
            pass
    s.close()  # idempotent
    idx.close()


def test_close_wakes_waiting_caller(tmp_path):
    idx, groups, _ = _tiny_store(tmp_path)
    gate = threading.Event()

    def blocked(name):
        gate.wait(5)
        return idx.read(name)

    s = BlockStreamer(idx, groups, window=2, readahead=1, read_fn=blocked)
    err = []

    def caller():
        try:
            with s.block(0):
                pass
        except Exception as e:  # noqa: BLE001
            err.append(e)

    th = threading.Thread(target=caller)
    th.start()
    time.sleep(0.1)
    closer = threading.Thread(target=s.close)
    closer.start()
    time.sleep(0.05)
    gate.set()
    th.join(5)
    closer.join(5)
    assert not th.is_alive() and not closer.is_alive()
    # caller either got block 0 (read finished before close) or a clean "closed" error
    assert not err or isinstance(err[0], RuntimeError)
    idx.close()


def test_slow_disk_reads_overlap_compute(tmp_path):
    idx, groups, _ = _tiny_store(tmp_path, n=8, per=2)
    read_s, compute_s = 0.04, 0.10  # per group: read < compute

    def slow(name):
        time.sleep(read_s / 2)  # 2 tensors per group
        return idx.read(name)

    waits = []
    with BlockStreamer(idx, groups, window=2, readahead=1, read_fn=slow) as s:
        for _ in range(2):
            for i in range(len(groups)):
                w0 = s.stats()["wait_seconds"]
                with s.block(i):
                    waits.append(s.stats()["wait_seconds"] - w0)
                    time.sleep(compute_s)
        st = s.stats()
    # first block may wait for its read; after that reads hide behind compute
    assert waits[0] <= read_s * 3
    assert sum(waits[1:]) < 0.03, waits
    assert st["read_seconds"] >= read_s * (2 * len(groups) - 1) * 0.9
    assert st["misses"] == 0
    idx.close()


def test_unknown_tensor_and_bad_args(tmp_path):
    idx, groups, _ = _tiny_store(tmp_path)
    with pytest.raises(KeyError):
        BlockStreamer(idx, [["nope"]])
    with pytest.raises(ValueError):
        BlockStreamer(idx, groups, window=0)
    idx.close()


def test_nocache_flag_is_harmless(tmp_path):
    idx, groups, flat = _tiny_store(tmp_path)
    with BlockStreamer(idx, groups, nocache=True) as s:
        with s.block(2) as t:
            n = groups[2][0]
            assert mx.array_equal(t[n], flat[n]).item()
    idx.close()
