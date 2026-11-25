"""Prompt encoders run one block at a time (shared by every family that uses a T5-style text encoder).

A T5 / umT5 encoder runs once per prompt, but loading it whole costs 11 GB (umT5-XXL in bf16) or 23 GB (fp32), which
alone pushes a 16 or 24 GB Mac into swap. `encode_ids` runs the same forward with the same weights and the same
operations while holding only the token embedding (read once, then dropped) and one block at a time. Blocks are read
lazily from safetensors (`lazy_weights`), so nothing else is ever in memory.

`upcast` lets a family run each block in another dtype (Wan runs its umT5 in float32, as mlx-video and the reference
do); only the current block is upcast, so the cost is one block, not the whole encoder.

Used by LongCat (shutter/adapters/longcat_t5.py) and the Wan family (shutter/families/wan/).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = ["encode_ids", "lazy_weights"]


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
    return out
