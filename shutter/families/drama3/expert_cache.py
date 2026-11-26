"""Drama 3 exact low-memory mode: routed experts stream from the SSD through a RAM cache (Shuttrr's SSD streaming steps).

    TF_D3_EXPERT_CACHE_GB=20  (with TF_D3_LOWMEM=1)   ->  at most ~20 GB of experts resident

The 8-bit expert weights are never loaded whole. Each MoE layer keeps a reader over the model's safetensors files;
when the router picks experts, the ones not in RAM are read from disk (exact bytes, gate|up concatenated as the
fused model does) into a least-recently-used cache shared by all layers, and the step runs mx.gather_qmm over
just the picked experts. Same weights, same kernels; the only change is that the non-expert parts of a backbone
layer compile around the router (the router's choice must reach the host), which may move rounding in the last bit.
Stats: expert_cache.stats() -> hits, misses, bytes read, read seconds, resident GB.
"""

from __future__ import annotations

import json
import os
import struct
import time
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_DT = {"U32": (np.uint32, None), "BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, None),
       "F32": (np.float32, None), "U8": (np.uint8, None)}


def enabled() -> bool:
    return float(os.environ.get("TF_D3_EXPERT_CACHE_GB", "0") or 0) > 0


class _Files:
    """Byte-exact reads of one expert's slice of stacked [E, ...] tensors in safetensors files."""

    def __init__(self, model_dir: Path):
        self.dir = Path(model_dir)
        self.index = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers: dict = {}
        self.fds: dict = {}

    def meta(self, name: str):
        f = self.index[name]
        if f not in self.headers:
            fd = os.open(str(self.dir / f), os.O_RDONLY)
            try:                       # macOS F_NOCACHE: real SSD reads, no second copy in the file cache
                import fcntl
                fcntl.fcntl(fd, 48, 1)
            except Exception:          # noqa: BLE001
                pass
            n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            self.headers[f] = (json.loads(os.pread(fd, n, 8)), 8 + n)
            self.fds[f] = fd
        h, base = self.headers[f]
        t = h[name]
        return self.fds[f], base + t["data_offsets"][0], t["dtype"], t["shape"]

    def expert(self, name: str, e: int) -> mx.array:
        fd, off, dtype, shape = self.meta(name)
        np_dt, mx_view = _DT[dtype]
        per = int(np.prod(shape[1:])) * np.dtype(np_dt).itemsize
        buf = np.frombuffer(os.pread(fd, per, off + e * per), dtype=np_dt).reshape(shape[1:])
        a = mx.array(buf)
        return a.view(mx_view) if mx_view is not None else a


class _Cache:
    def __init__(self, budget_bytes: int):
        self.budget = budget_bytes
        self.items: OrderedDict = OrderedDict()       # (layer, expert) -> (tensors, nbytes)
        self.used = 0
        self.hits = self.misses = self.bytes_read = 0
        self.read_s = 0.0

    def get(self, key, load):
        item = self.items.get(key)
        if item is not None:
            self.items.move_to_end(key)
            self.hits += 1
            return item[0]
        t = time.perf_counter()
        tensors = load()
        mx.eval(*tensors)
        nbytes = sum(x.nbytes for x in tensors)
        self.read_s += time.perf_counter() - t
        self.misses += 1
        self.bytes_read += nbytes
        self.items[key] = (tensors, nbytes)
        self.used += nbytes
        return tensors

    def trim(self, keep: set):
        """Evict least-recently-used experts (never the ones this step uses) until within budget."""
        for key in list(self.items):
            if self.used <= self.budget:
                break
            if key in keep:
                continue
            self.used -= self.items.pop(key)[1]


CACHE: _Cache | None = None


def stats() -> dict:
    c = CACHE
    if c is None:
        return {}
    total = c.hits + c.misses
    return {"hits": c.hits, "misses": c.misses, "hit_rate": round(c.hits / max(1, total), 4),
            "gb_read": round(c.bytes_read / 1e9, 2), "read_s": round(c.read_s, 2),
            "resident_gb": round(c.used / 1e9, 2), "budget_gb": round(c.budget / 1e9, 2)}


class StreamedSwitchGLU(nn.Module):
    """FusedSwitchGLU with experts read on demand (gate|up fused per expert)."""

    def __init__(self, layer: int, prefix: str, files: _Files, group_size: int, bits: int, mode: str):
        super().__init__()
        self._layer, self._prefix, self._files = layer, prefix, files
        self._q = dict(group_size=group_size, bits=bits, mode=mode)

    def _load(self, e: int):
        f, p = self._files, self._prefix
        gw, gs, gb = (f.expert(f"{p}.gate_proj.{k}", e) for k in ("weight", "scales", "biases"))
        uw, us, ub = (f.expert(f"{p}.up_proj.{k}", e) for k in ("weight", "scales", "biases"))
        dw, ds, db = (f.expert(f"{p}.down_proj.{k}", e) for k in ("weight", "scales", "biases"))
        return (mx.concatenate([gw, uw], axis=0), mx.concatenate([gs, us], axis=0),
                mx.concatenate([gb, ub], axis=0), dw, ds, db)

    def __call__(self, x, indices):
        from shutter.families.drama3.mlx_lm_models.switch_layers import _gather_sort, _scatter_unsort

        host = np.array(indices)                                   # the router's choice, on the host
        uniq = sorted(set(int(v) for v in host.reshape(-1)))
        experts = [CACHE.get((self._layer, e), lambda e=e: self._load(e)) for e in uniq]
        CACHE.trim({(self._layer, e) for e in uniq})
        remap = np.zeros(max(uniq) + 1, dtype=np.uint32)
        remap[uniq] = np.arange(len(uniq), dtype=np.uint32)
        idx = mx.array(remap[host])
        stacked = [mx.stack([t[k] for t in experts]) for k in range(6)]
        x = mx.expand_dims(x, (-2, -3))
        do_sort = idx.size >= 64
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, idx)
        q = self._q
        gu = mx.gather_qmm(x, stacked[0], stacked[1], stacked[2], rhs_indices=idx, transpose=True,
                           sorted_indices=do_sort, **q)
        gate, up = mx.split(gu, 2, axis=-1)
        y = nn.silu(gate) * up
        y = mx.gather_qmm(y, stacked[3], stacked[4], stacked[5], rhs_indices=idx, transpose=True,
                          sorted_indices=do_sort, **q)
        if do_sort:
            y = _scatter_unsort(y, inv_order, host.shape)
        return y.squeeze(-2)


def attach(layer, i: int, model_dir) -> None:
    """Replace layer i's routed experts with a streamed reader before any expert weight is evaluated."""
    global CACHE
    if CACHE is None:
        CACHE = _Cache(int(float(os.environ["TF_D3_EXPERT_CACHE_GB"]) * (1 << 30)))
    files = getattr(attach, "_files", None)
    if files is None or files.dir != Path(model_dir):
        files = attach._files = _Files(Path(model_dir))
    sw = layer.mlp.switch_mlp
    g = sw.gate_proj
    layer.mlp.switch_mlp = StreamedSwitchGLU(i, f"backbone.model.layers.{i}.mlp.switch_mlp", files,
                                             int(g.group_size), int(g.bits), getattr(g, "mode", "affine"))


# -- the decode step: compiled around the router --------------------------------------------------------------------
def make_step(model):
    from shutter.families.drama3 import fast

    class StreamStep(fast.FastStep):
        @staticmethod
        def _linear_layer(layer):
            attn, mlp = layer.linear_attn, layer.mlp

            def pre(x, conv_state, state):
                qkv, z, b, a = attn.split_proj(layer.input_layernorm(x))
                out, conv_input, state = attn.mix(qkv, z, b, a, conv_state, state)
                h = x + out
                return h, layer.post_attention_layernorm(h), conv_input[:, 1:, :], state

            pre_c = mx.compile(pre)

            def run(x, conv_state, state):
                h, normed, conv, state = pre_c(x, conv_state, state)
                return h + mlp(normed), conv, state

            return run

        @staticmethod
        def _attention_layer(layer):
            attn, mlp = layer.self_attn, layer.mlp

            def pre(x, offset):
                return attn.project(layer.input_layernorm(x), offset)

            def mid(x, out, gate):
                h = x + attn.finish(out, gate)
                return h, layer.post_attention_layernorm(h)

            mid_c = mx.compile(mid)

            def post(x, out, gate):
                h, normed = mid_c(x, out, gate)
                return h + mlp(normed)

            return pre, post

        def __init__(self, model):
            # FastStep.__init__ compiles what _linear_layer / _attention_layer return; ours compile inside instead
            orig = mx.compile
            mx.compile = lambda f, *a, **k: f if getattr(f, "_stream_eager", False) else orig(f, *a, **k)
            try:
                super().__init__(model)
            finally:
                mx.compile = orig

    # mark the eager wrappers so FastStep.__init__ does not compile them (they sync on the router)
    lin, att = StreamStep._linear_layer, StreamStep._attention_layer

    def _lin(layer):
        f = lin(layer)
        f._stream_eager = True
        return f

    def _att(layer):
        pre, post = att(layer)
        post._stream_eager = True
        return pre, post

    StreamStep._linear_layer = staticmethod(_lin)
    StreamStep._attention_layer = staticmethod(_att)
    return StreamStep(model)
