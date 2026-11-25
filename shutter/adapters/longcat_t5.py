"""umT5 prompt encode, one block at a time (for Macs with less memory than the 11 GB text encoder).

LongCat's `encode_prompt` reads the whole umT5 encoder (11.4 GB) into memory for one forward pass on the CPU and
then frees it. On a 16 or 24 GB Mac that alone pushes the machine into swap. This version runs the same forward,
on the CPU, with the same weights and the same operations, but holds only the token embedding (read once, then
dropped) and one of the 24 blocks at a time: about 2.5 GB instead of 11.4 GB. The result goes to the same cache
file as the original, so the encode happens once per prompt.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional


# The block-by-block encode itself is family-neutral and lives in shutter/core/text_encode.py.
from shutter.core.text_encode import encode_ids, lazy_weights  # noqa: E402,F401  (re-exported)


def make_encode_prompt(gen_mod: Any, on_block: Optional[Callable[[int, int], None]] = None) -> Callable:
    """A drop-in for `gen_mod.encode_prompt` (same signature, same cache file, same result)."""
    import mlx.core as mx

    from longcat_video_avatar.models.umt5 import UMT5EncoderModel

    clean = gen_mod._prompt_clean

    def encode_prompt(components: Path, prompt: str, cache_dir: Optional[Path] = None):
        components = Path(components)
        key = hashlib.sha1(clean(prompt).encode()).hexdigest()[:16]
        f = Path(cache_dir) / f"t5-{key}.safetensors" if cache_dir else None
        if f is not None and f.exists():
            d = mx.load(str(f))
            return d["emb"], d["mask"]
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(components / "tokenizer"))
        enc = tok([clean(prompt)], padding="max_length", max_length=512, truncation=True,
                  add_special_tokens=True, return_attention_mask=True, return_tensors="np")
        cfg = json.loads((components / "text_encoder" / "config.json").read_text())
        umt5 = UMT5EncoderModel.from_config(cfg)
        weights = lazy_weights(components / "text_encoder", "model.safetensors.index.json", mx)
        ids, mask = mx.array(enc.input_ids), mx.array(enc.attention_mask)
        started = time.time()
        emb = encode_ids(umt5, weights, ids, mask, mx, on_block)
        with mx.stream(mx.cpu):
            emb = emb.astype(mx.bfloat16)[:, None]
            mx.eval(emb)
        print(f"umT5 prompt encode (block by block, CPU) {time.time() - started:.1f}s, cached as {f}", flush=True)
        if f is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(f), {"emb": emb, "mask": mask})
        del umt5, weights
        mx.clear_cache()
        return emb, mask

    encode_prompt.shutter_streamed = True
    return encode_prompt


def install(gen_mod: Any, enabled: bool, on_block: Optional[Callable[[int, int], None]] = None) -> bool:
    """Swap the tree's `encode_prompt` for the block-by-block one (or put the original back). Returns whether the
    streamed encode is now in place."""
    if not hasattr(gen_mod, "encode_prompt"):
        return False
    original = getattr(gen_mod, "_shutter_encode_prompt_original", None)
    if original is None:
        original = gen_mod.encode_prompt
        gen_mod._shutter_encode_prompt_original = original
    if not enabled or not hasattr(gen_mod, "_prompt_clean"):
        gen_mod.encode_prompt = original
        return False
    gen_mod.encode_prompt = make_encode_prompt(gen_mod, on_block)
    return True
