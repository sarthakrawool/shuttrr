"""Prompt encoders run one block at a time (shared by every family that uses a T5-style text encoder).

A T5 / umT5 encoder runs once per prompt, but loading it whole costs 11 GB (umT5-XXL in bf16) or 23 GB (fp32), which
alone pushes a 16 or 24 GB Mac into swap. `encode_ids` runs the same forward with the same weights and the same
operations while holding only the token embedding (read once, then dropped) and one block at a time. Blocks are read
lazily from safetensors (`lazy_weights`), so nothing else is ever in memory.

`upcast` lets a family run each block in another dtype (Wan runs its umT5 in float32, as mlx-video and the reference
do); only the current block is upcast, so the cost is one block, not the whole encoder.

An encoder folder may also be stored in 8 bits (`quantize_folder`, MLX's own layout: `<name>.weight` packed as
uint32 next to `<name>.scales` and `<name>.biases`, the bits and group size in the index's "quantization"
metadata). `lazy_weights` turns such tensors back into plain weights lazily, so each one is dequantized only when its
block is evaluated: the encode still holds one block at a time, and the forward is the same code.

Used by LongCat (shutter/adapters/longcat_t5.py) and the Wan family (shutter/families/wan/).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = ["encode_ids", "lazy_weights", "quantize_folder", "folder_quantization", "QuantizedRows", "QUANT_MIN_SIZE"]

INDEX = "model.safetensors.index.json"
QUANT_MIN_SIZE = 4096 * 64   # 2D tensors smaller than this (position biases) stay as they are


def encode_ids(umt5: Any, weights: dict[str, Any], ids: Any, mask: Any, mx: Any,
               on_block: Optional[Callable[[int, int], None]] = None, *, device: Any = None,
               upcast: Any = None) -> Any:
    """`umt5(ids, mask=mask)` with lazily loaded `weights` ({name: array}), evaluating and dropping one block at a
    time. `umt5` is a fresh encoder (token_embedding, blocks[i] with their own position bias, norm); it is emptied
    by this call. `device` defaults to the CPU (as LongCat's own encode does); `upcast` (an MLX dtype) casts the
    embedding rows and each block's weights to that dtype just before use."""
    if getattr(umt5, "pos_embedding", None) is not None:
        raise NotImplementedError("shared position bias (plain T5) is not handled; umT5 has one per block")

    def take(prefix: str) -> list[tuple[str, Any]]:
        names = [k for k in weights if k.startswith(prefix)]
        rows = [(k, weights.pop(k)) for k in names]
        if upcast is not None:
            rows = [(k, v.astype(upcast)) for k, v in rows]
        return rows

    with mx.stream(device if device is not None else mx.cpu):
        table = weights.pop("token_embedding.weight")
        x = table[ids]
        if upcast is not None:
            x = x.astype(upcast)
        mx.eval(x)
        del table
        mx.clear_cache()
        n = len(umt5.blocks)
        for i in range(n):
            rows = take(f"blocks.{i}.")
            if not rows:
                raise ValueError(f"text encoder weights have no blocks.{i}.*")
            umt5.load_weights(rows, strict=False)
            block = umt5.blocks[i]
            mx.eval(block.parameters())
            x = block(x, mask=mask, pos_bias=None)
            mx.eval(x)
            umt5.blocks[i] = None
            del block, rows
            mx.clear_cache()
            if on_block is not None:
                on_block(i + 1, n)
        umt5.load_weights(take("norm."), strict=False)
        x = umt5.norm(x)
        mx.eval(x)
    return x


def folder_quantization(folder: Path, index_name: Optional[str] = INDEX) -> Optional[dict[str, int]]:
    """{"bits", "group_size"} when the folder holds an 8-bit (or other) quantized encoder, else None."""
    folder = Path(folder)
    if index_name and (folder / index_name).exists():
        q = (json.loads((folder / index_name).read_text()).get("metadata") or {}).get("quantization")
        if isinstance(q, str):
            q = json.loads(q)
        if q:
            return {"bits": int(q["bits"]), "group_size": int(q["group_size"])}
    return None


class QuantizedRows:
    """A quantized embedding table that dequantizes only the rows it is indexed with (`table[ids]`), so the prompt
    stage never builds the whole bf16 table (2.1 GB for umT5-XXL). Same values as dequantizing it whole."""

    def __init__(self, q: Any, scales: Any, biases: Any, bits: int, group_size: int, mx: Any):
        self.q, self.scales, self.biases, self.bits, self.group_size, self._mx = q, scales, biases, bits, group_size, mx
        self.shape = (q.shape[0], q.shape[1] * 32 // bits)
        self.dtype = scales.dtype

    def __getitem__(self, ids: Any) -> Any:
        mx = self._mx
        return mx.dequantize(self.q[ids], self.scales[ids], self.biases[ids], group_size=self.group_size,
                             bits=self.bits).astype(self.dtype)

    def dequantize(self) -> Any:
        return self[self._mx.arange(self.q.shape[0])]


def _dequantize_lazy(w: dict[str, Any], q: Optional[dict[str, int]], mx: Any) -> dict[str, Any]:
    """Replace each (<name>.weight uint32, <name>.scales, <name>.biases) triple by one lazy bf16 <name>.weight
    (the token embedding by a QuantizedRows, read row by row)."""
    scaled = [k[:-len(".scales")] for k in w if k.endswith(".scales")]
    if not scaled:
        return w
    if q is None:
        q = {"bits": 8, "group_size": 64}
    out = dict(w)
    for base in scaled:
        wk = base + ".weight"
        if wk not in out or base + ".biases" not in out or out[wk].dtype != mx.uint32:
            continue
        s, b = out.pop(base + ".scales"), out.pop(base + ".biases")
        if base == "token_embedding":
            out[wk] = QuantizedRows(out[wk], s, b, q["bits"], q["group_size"], mx)
            continue
        out[wk] = mx.dequantize(out[wk], s, b, group_size=q["group_size"], bits=q["bits"]).astype(s.dtype)
    return out


def lazy_weights(folder: Path, index_name: Optional[str] = None, mx: Any = None) -> dict[str, Any]:
    """Every tensor of a safetensors folder, not yet read (MLX reads a tensor when it is evaluated). With
    `index_name` the shards listed in that index; without it every *.safetensors file in the folder."""
    if mx is None:
        import mlx.core as mx
    folder = Path(folder)
    if index_name and (folder / index_name).exists():
        idx = json.loads((folder / index_name).read_text())
        shards = sorted(set(idx["weight_map"].values()))
    else:
        shards = sorted(p.name for p in folder.glob("*.safetensors"))
    out: dict[str, Any] = {}
    for shard in shards:
        out.update(mx.load(str(folder / shard)))
    return _dequantize_lazy(out, folder_quantization(folder, index_name), mx)


def quantize_folder(src: Path, dst: Path, *, bits: int = 8, group_size: int = 64, mx: Any = None,
                    min_size: int = QUANT_MIN_SIZE, report: Callable[[str], None] = print) -> dict[str, Any]:
    """Write an encoder folder (`src`, bf16 shards + model.safetensors.index.json) to `dst` with every large 2D weight
    quantized (mx.quantize, `bits`, `group_size`; scales and biases in bf16). Same shard names and the same tensors
    otherwise. Deterministic: the same input and MLX version give the same bytes. Returns {"files", "total_bytes"}."""
    if mx is None:
        import mlx.core as mx
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    idx = json.loads((src / INDEX).read_text())
    meta = {"bits": bits, "group_size": group_size}
    weight_map: dict[str, str] = {}
    shards = sorted(set(idx["weight_map"].values()), key=lambda n: (n != "rest.safetensors", n))
    for shard in shards:
        w = mx.load(str(src / shard))
        out: dict[str, Any] = {}
        for k in sorted(w):
            v = w[k]
            if k.endswith(".weight") and v.ndim == 2 and v.shape[-1] % group_size == 0 and v.size >= min_size:
                base = k[:-len(".weight")]
                q, s, b = mx.quantize(v.astype(mx.bfloat16), group_size=group_size, bits=bits)
                out[k], out[base + ".scales"], out[base + ".biases"] = q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)
            else:
                out[k] = v
        mx.eval(out)
        mx.save_safetensors(str(dst / shard), out, metadata={"format": "mlx", "quantization": json.dumps(meta)})
        weight_map.update({k: shard for k in out})
        report(f"  {shard}: {(dst / shard).stat().st_size / 1e9:.2f} GB")
        del w, out
        mx.clear_cache()
    (dst / INDEX).write_text(json.dumps({"metadata": {"quantization": meta}, "weight_map": weight_map}, indent=1,
                                        sort_keys=True))
    return {"files": shards, "total_bytes": sum((dst / f).stat().st_size for f in shards), "quant": meta}
