"""Load a Wan DiT from a Shuttrr build: whole (resident) or block by block from the SSD (streamed), with LoRAs.

Both paths build the very same modules (mlx-video's WanModel / WanAttentionBlock, quantized at the same paths) and
feed them the very same bytes. The streamed path keeps one block "shell" and swaps each block's tensors into it just
before it runs (shutter.core.stream.BlockStreamer reads ahead on a background thread). Cross-attention K/V are
computed inside each block instead of being cached up front; mlx-video's `prepare_kv` runs exactly the same
operations, so the output is bit-identical (tests/test_wan_family.py checks this).

LoRAs: `LoRALinear` wraps a Linear or QuantizedLinear and adds ``scale * (x @ A.T) @ B.T``. The base weights are
never touched, so one 8-bit build serves any number of LoRAs and a LoRA can change between requests without a reload.
A LoRA's tensors are small (tens to hundreds of MB) and stay resident; when blocks stream, block i's LoRA tensors
are put on the shell together with block i's weights.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from shutter.families.wan import keys as K


def _mx():
    import mlx.core as mx
    import mlx.nn as nn
    return mx, nn


def model_config(cfg: dict[str, Any], num_layers: Optional[int] = None):
    from shutter.families.wan.model.config import WanModelConfig
    fields = {f.name for f in dataclasses.fields(WanModelConfig)}
    c = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in fields}
    if num_layers is not None:
        c["num_layers"] = num_layers
    return WanModelConfig(**c)


def quantize_module(module, quant: Optional[dict[str, Any]], prefix: str = "") -> None:
    """nn.quantize at exactly the paths the converter quantized (keys.is_quantized_linear)."""
    if not quant:
        return
    _, nn = _mx()
    nn.quantize(module, group_size=int(quant.get("group_size", 64)), bits=int(quant.get("bits", 8)),
                class_predicate=lambda path, m: hasattr(m, "to_quantized") and K.is_quantized_linear(prefix + path))


#: arrays the model computes itself (RoPE table): never in a checkpoint
COMPUTED = {"freqs"}


def load_checked(module, items: list) -> None:
    """load_weights, strict except for computed buffers: every tensor must land on a parameter, and every parameter
    must be given."""
    from mlx.utils import tree_flatten
    have = {k for k, _ in tree_flatten(module.parameters())} - COMPUTED
    given = {k for k, _ in items}
    extra, missing = given - have, have - given
    if extra or missing:
        raise ValueError(f"weights do not match the model: {len(extra)} unexpected (e.g. {sorted(extra)[:3]}), "
                         f"{len(missing)} missing (e.g. {sorted(missing)[:3]})")
    module.load_weights(items, strict=False)


def _part_files(build: Path, part: dict[str, Any]) -> list[Path]:
    return [Path(build) / f for f in part["files"]]


def load_resident(build: Path, cfg: dict[str, Any], part: dict[str, Any]):
    """The whole DiT in memory."""
    mx, _ = _mx()
    from shutter.families.wan.model.dit import WanModel
    model = WanModel(model_config(cfg))
    quantize_module(model, part.get("quant"))
    weights = []
    for f in _part_files(build, part):
        weights.extend(mx.load(str(f)).items())
    load_checked(model, weights)
    mx.eval(model.parameters())
    return model


# ----------------------------------------------------------------------------- LoRA


class LoRALinear:
    """Placeholder so the class exists before mlx is imported; replaced by `_lora_cls()`."""


_LORA_CLS = None


def _lora_cls():
    global _LORA_CLS
    if _LORA_CLS is not None:
        return _LORA_CLS
    mx, nn = _mx()

    class _LoRALinear(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear
            # [(A [r, in], B [out, r], scale)]: a plain attribute, so it is not part of the parameter tree
            object.__setattr__(self, "_lora_list", [])

        def __call__(self, x):
            y = self.linear(x)
            for a, b, s in object.__getattribute__(self, "_lora_list"):
                y = y + (s * ((x.astype(a.dtype) @ a.T) @ b.T)).astype(y.dtype)
            return y

        def set_adapters(self, items):
            object.__setattr__(self, "_lora_list", list(items))

    _LORA_CLS = _LoRALinear
    return _LORA_CLS


def read_lora(path: Path, strength: float = 1.0) -> dict[str, list]:
    """{mlx module path (blocks.i....): [(A, B, scale)]} from a LoRA file in any common naming."""
    mx, _ = _mx()
    raw = mx.load(str(path))
    by_mod: dict[str, dict[str, Any]] = {}
    for name, arr in raw.items():
        t = K.lora_target(name)
        if t is None:
            continue
        by_mod.setdefault(t[0], {})[t[1]] = arr
    out: dict[str, list] = {}
    for mod, d in by_mod.items():
        if "down" not in d or "up" not in d or not mod.startswith("blocks."):
            continue
        a, b = d["down"].astype(mx.bfloat16), d["up"].astype(mx.bfloat16)
        rank = a.shape[0]
        alpha = float(d["alpha"].item()) if "alpha" in d else float(rank)
        out.setdefault(K.lora_mlx_module(mod), []).append((a, b, strength * alpha / rank))
    return out


def merge_loras(*loras: dict[str, list]) -> dict[str, list]:
    out: dict[str, list] = {}
    for d in loras:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


def _get(module, path: str):
    obj = module
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _set(module, path: str, value) -> None:
    head, _, last = path.rpartition(".")
    parent = _get(module, head) if head else module
    setattr(parent, last, value)


LORA_TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.fc1", "ffn.fc2")


def wrap_block_for_lora(block) -> None:
    cls = _lora_cls()
    for t in LORA_TARGETS:
        lin = _get(block, t)
        if not isinstance(lin, cls):
            _set(block, t, cls(lin))


def set_block_lora(block, block_index: int, loras: dict[str, list]) -> None:
    """Point a (wrapped) block's LoRA slots at block `block_index`'s adapters (empty list: none)."""
    cls = _lora_cls()
    for t in LORA_TARGETS:
        w = _get(block, t)
        if isinstance(w, cls):
            w.set_adapters(loras.get(f"blocks.{block_index}.{t}", []))


def apply_loras_resident(model, loras: dict[str, list]) -> None:
    for i, block in enumerate(model.blocks):
        if loras:
            wrap_block_for_lora(block)
        set_block_lora(block, i, loras)


# ----------------------------------------------------------------------------- streaming


class StreamedBlocks:
    """Stands in for ``WanModel.blocks``: iterating yields one callable per block that loads block i into the
    shell, runs it, evaluates the output and drops the weights."""

    def __init__(self, shell, streamer, n: int, loras: Optional[dict[str, list]] = None,
                 on_block: Optional[Callable[[int, int], None]] = None):
        from shutter.core.stream import StreamedStack
        self.shell = shell
        self.streamer = streamer
        self.n = n
        self.loras = loras or {}
        self.on_block = on_block
        # LoRA slots are always there (empty when no LoRA is active), so LoRAs can change between requests
        # without rebuilding the shell; an empty slot adds nothing (same output, bit for bit)
        wrap_block_for_lora(shell)
        # StreamedStack does the name/shape checks, the load and the release
        self._stack = StreamedStack(shell, streamer, strip_prefix_fn=_block_strip(True))

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[Callable]:
        mx, _ = _mx()
        stack = self._stack

        def bound(i: int):
            def run(x, **kw):
                with self.streamer.block(i) as tensors:
                    stack._load(self.shell, 0, tensors, i)
                    set_block_lora(self.shell, i, self.loras)
                    out = self.shell(x, **kw)
                    mx.eval(out)
                    stack._clear(0)
                if self.on_block:
                    self.on_block(i + 1, self.n)
                return out
            return run

        for i in range(self.n):
            yield bound(i)

    def close(self) -> None:
        self.streamer.close()


def _block_strip(wrapped: bool) -> Callable[[str], str]:
    """Block tensor name -> name inside the shell. LoRA wrappers move a base layer one level down
    (``self_attn.q.weight`` -> ``self_attn.q.linear.weight``)."""
    from shutter.core.stream import default_strip_prefix

    def strip(name: str) -> str:
        local = default_strip_prefix(name)
        if wrapped:
            mod, _, leaf = local.rpartition(".")
            if mod in LORA_TARGETS:
                return f"{mod}.linear.{leaf}"
        return local
    return strip


def load_streamed(build: Path, cfg: dict[str, Any], part: dict[str, Any], window: int,
                  loras: Optional[dict[str, list]] = None, nocache: bool = False,
                  on_block: Optional[Callable[[int, int], None]] = None):
    """A WanModel whose blocks are read from the SSD while it runs, `window` blocks in memory at a time."""
    mx, _ = _mx()
    from shutter.core.stream import BlockStreamer, SafetensorsIndex, group_names_by_block
    from shutter.families.wan.model.dit import WanModel
    from shutter.families.wan.model.transformer import WanAttentionBlock

    files = [str(f) for f in _part_files(build, part)]
    index = SafetensorsIndex(files)
    groups, rest = group_names_by_block(index.names(), pattern=r"^blocks\.(\d+)\.")
    n = len(groups)
    trunk = WanModel(model_config(cfg, num_layers=0))
    quantize_module(trunk, part.get("quant"))
    load_checked(trunk, [(k, index.to_mlx(k)) for k in rest])
    mx.eval(trunk.parameters())
    shell = WanAttentionBlock(dim=cfg["dim"], ffn_dim=cfg["ffn_dim"], num_heads=cfg["num_heads"],
                              window_size=(-1, -1), qk_norm=cfg.get("qk_norm", True),
                              cross_attn_norm=cfg.get("cross_attn_norm", True), eps=1e-6)
    quantize_module(shell, part.get("quant"), prefix="blocks.0.")
    w = max(1, min(int(window), n))
    streamer = BlockStreamer(index, groups, window=w, readahead=max(0, w - 1), cyclic=True, nocache=nocache)
    blocks = StreamedBlocks(shell, streamer, n, loras=loras, on_block=on_block)
    object.__setattr__(trunk, "_streamed_blocks", blocks)
    # WanModel.__call__ iterates self.blocks: point it at the streamed sequence
    if "blocks" in trunk:
        del trunk["blocks"]
    object.__setattr__(trunk, "blocks", blocks)
    return trunk


def read_manifest(build: Path) -> dict[str, Any]:
    return json.loads((Path(build) / "shuttrr.json").read_text())
