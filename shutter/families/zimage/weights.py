"""Reading an mflux-saved Z-Image build: one folder per component (transformer/, text_encoder/, vae/), each with
numbered .safetensors shards and a model.safetensors.index.json. Tensor names are already the MLX module names, and
quantized layers are stored as (weight, scales, biases) at the bits and group size they were saved with.

Nothing here reads tensor data up front: `mx.load` maps the shards lazily, so a part's bytes are read only when that
part runs. `load_into` builds the quantized structure from the stored shapes (as mflux's WeightApplier does) and then
sets the weights.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

GB = 1024**3


class Component:
    """One component folder: tensor name -> shard, and lazily opened shards."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        idx = self.folder / "model.safetensors.index.json"
        if idx.exists():
            data = json.loads(idx.read_text())
            self.weight_map: dict[str, str] = dict(data["weight_map"])
            self.metadata: dict[str, Any] = dict(data.get("metadata") or {})
        else:
            # no index: every shard's header
            from shutter.core.stream import SafetensorsIndex
            self.weight_map = {}
            self.metadata = {}
            for f in sorted(self.folder.glob("*.safetensors")):
                with SafetensorsIndex([str(f)]) as ix:
                    for n in ix.names():
                        self.weight_map[n] = f.name
        if not self.weight_map:
            raise FileNotFoundError(f"no weights in {self.folder}")

    @property
    def files(self) -> list[Path]:
        return [self.folder / f for f in sorted(set(self.weight_map.values()))]

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.files if p.exists())

    def names(self, prefix: str = "") -> list[str]:
        return [n for n in self.weight_map if n.startswith(prefix)]

    def load(self, names, nocache: bool = False) -> dict[str, Any]:
        """{name: mx.array} for `names`, read lazily (a fresh map of the shards each call, so nothing outlives the
        caller's references). nocache=True reads the bytes now with F_NOCACHE instead, so they do not pass through
        macOS's file cache: reading the 4 GB prompt encoder in about a second through the cache made macOS push
        other apps' memory to swap on a busy 16 GB Mac (0.3.0 requests stopped by the swap watchdog at 2 s)."""
        import mlx.core as mx
        if nocache:
            from shutter.core.stream import SafetensorsIndex, to_mlx_array
            with SafetensorsIndex([str(p) for p in self.files], nocache=True) as ix:
                return {n: to_mlx_array(ix.read(n), ix.entries[n].dtype) for n in names}
        by_file: dict[str, list[str]] = {}
        for n in names:
            by_file.setdefault(self.weight_map[n], []).append(n)
        out: dict[str, Any] = {}
        for f, ns in by_file.items():
            arrays = mx.load(str(self.folder / f))
            for n in ns:
                out[n] = arrays[n]
            del arrays
        return out

    def tensor_index(self):
        """A shutter.core.stream.SafetensorsIndex over this component's shards (header only)."""
        from shutter.core.stream import SafetensorsIndex
        return SafetensorsIndex([str(p) for p in self.files])


def strip(weights: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}


def stored_quant(weights: dict[str, Any], path: str, module) -> Any:
    """nn.quantize class predicate result for `module` at `path`: the bits / group size it was stored with, or False
    when it was stored unquantized (mflux WeightApplier._stored_layer_predicate)."""
    if not hasattr(module, "to_quantized"):
        return False
    scales = weights.get(f"{path}.scales")
    if scales is None:
        return False
    stored = weights.get(f"{path}.weight")
    mshape = getattr(getattr(module, "weight", None), "shape", None)
    if stored is None or not mshape:
        return True
    in_dims = mshape[-1]
    bits = stored.shape[-1] * 32 / in_dims
    group = in_dims / scales.shape[-1]
    if bits in (2, 3, 4, 5, 6, 8) and group in (32, 64, 128):
        return {"bits": int(bits), "group_size": int(group)}
    raise ValueError(f"{path}: cannot read the stored quantization (bits {bits}, group size {group})")


def load_into(module, weights: dict[str, Any], *, strict_missing: bool = True) -> None:
    """Quantize `module` the way `weights` were stored, then set them. Names in `weights` not in the module are an
    error; module parameters missing from `weights` are an error when `strict_missing`."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    nn.quantize(module, class_predicate=lambda path, m: stored_quant(weights, path, m))
    params = dict(tree_flatten(module.parameters()))
    extra = sorted(set(weights) - set(params))
    if extra:
        raise ValueError(f"unexpected tensors (not in the model): {extra[:5]}{' ...' if len(extra) > 5 else ''}")
    if strict_missing:
        missing = sorted(set(params) - set(weights))
        if missing:
            raise ValueError(f"missing tensors: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    for k, v in weights.items():
        if tuple(params[k].shape) != tuple(v.shape):
            raise ValueError(f"{k}: stored shape {tuple(v.shape)}, model expects {tuple(params[k].shape)}")
    module.update(tree_unflatten(list(weights.items())), strict=False)


_LAYER = re.compile(r"^layers\.(\d+)\.")


def layer_of(name: str) -> Optional[int]:
    m = _LAYER.match(name)
    return int(m.group(1)) if m else None


def part_bytes(comp: Component, keep: Callable[[str], bool]) -> int:
    """Stored bytes of the tensors whose name passes `keep` (from the shard headers)."""
    with comp.tensor_index() as ix:
        return sum(ix.entries[n].nbytes for n in comp.weight_map if keep(n))
