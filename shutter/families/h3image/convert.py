r"""Build Shuttrr's MiniMax-H3-Image folder from local files (no download):

    python -m shutter.families.h3image.convert \
        --dit  minimax_h3_image_pruned-Q8_0.gguf \          # ComfyUI / leejet GGUF of the pruned DiT
        --te   MiniMax-H3/FL2VA/text_encoder \              # the bf16 Qwen3-VL-32B of MiniMax-H3 (HF safetensors)
        --vae  minimax_h3_image_vae_fp16.safetensors \
        --tokenizer MiniMax-H3/FL2VA/tokenizer \
        --out  ~/.shutter/models/MiniMax-H3-Image-Shuttrr

Output (every component is a folder of .safetensors shards + model.safetensors.index.json, names = MLX module names):

    transformer/   the DiT. --dit-bits 8 (default): every Q8_0 matrix maps onto MLX 8-bit group 32 exactly
                   (value q in [-128, 127] times scale d -> stored q + 128, scale d, bias -128 d), with the scale
                   rounded to bfloat16 so the matmuls run in bfloat16; matrices the GGUF keeps in BF16 stay BF16.
                   --dit-bits 4: re-quantized to 4-bit group 64 (half the size, needs a quality check).
    text_encoder/  layers 0-49 of the text part (the ones H3 reads), 4-bit group 64 with v_proj / down_proj at 6
                   bits (about what Q4_K_M keeps), or --te-bits 8; the token embedding stays bfloat16 (only the
                   prompt's rows are ever read). Layers 50-63, the final norm, lm_head and the vision tower are left
                   out (text to image only).
    vae/           post_quant_conv + the ViT decoder, float16 as released.
    tokenizer/     tokenizer.json (+ config files)
    h3image.json   what was built from what
    NOTICE         the notice the MiniMax-H3-Image licence asks for (§III.4); these files are modified (converted)

Runs on the CPU, one tensor at a time (peak a few GB), so it can run beside other work.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

SHARD_BYTES = 128 * 1024**2
TE_LAYERS = 50
NOTICE = ("MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community License Agreement, Copyright © 2026 "
          "MiniMax. All Rights Reserved.\n\n"
          "The weight files in this folder were modified by Shuttrr: converted from the ComfyUI / GGUF release and "
          "the MiniMax-H3 text encoder to MLX's format (re-quantized), and cut to the parts text to image uses.\n"
          "Licence: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE\n")


class ShardWriter:
    """Writes {name: mx.array} into numbered shards of about SHARD_BYTES and an index."""

    def __init__(self, folder: Path, prefix: str = "model"):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.pending: dict[str, Any] = {}
        self.pending_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.n = 0
        self.total = 0

    def add(self, name: str, arr) -> None:
        import mlx.core as mx
        mx.eval(arr)
        self.pending[name] = arr
        self.pending_bytes += arr.nbytes
        self.total += arr.nbytes
        if self.pending_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        import mlx.core as mx
        if not self.pending:
            return
        self.n += 1
        fname = f"{self.prefix}-{self.n:05d}.safetensors"
        write_safetensors(self.folder / fname, self.pending, {"format": "mlx"})
        for k in self.pending:
            self.weight_map[k] = fname
        self.pending, self.pending_bytes = {}, 0

    def close(self, metadata: Optional[dict] = None) -> int:
        self.flush()
        idx = {"metadata": {"total_size": self.total, **(metadata or {})}, "weight_map": dict(sorted(self.weight_map.items()))}
        (self.folder / "model.safetensors.index.json").write_text(json.dumps(idx, indent=1))
        return self.total


_ST_DTYPES = {"uint32": "U32", "bfloat16": "BF16", "float16": "F16", "float32": "F32", "int32": "I32",
              "uint8": "U8", "int8": "I8"}


def write_safetensors(path: Path, tensors: dict[str, Any], metadata: Optional[dict] = None) -> None:
    """mx.save_safetensors, but written with F_NOCACHE: 40 GB of converted weights written through macOS's file
    cache would push other apps' memory to swap on a 16 GB Mac."""
    import os
    import struct
    import mlx.core as mx
    from shutter.core.stream import _apply_nocache
    header: dict[str, Any] = {"__metadata__": dict(metadata or {})}
    off = 0
    blobs = []
    for k, a in tensors.items():
        dt = str(a.dtype).split(".")[-1]
        if dt == "bfloat16":
            buf = np.array(a.view(mx.uint16))
        else:
            buf = np.array(a)
        b = np.ascontiguousarray(buf).view(np.uint8).reshape(-1)
        header[k] = {"dtype": _ST_DTYPES[dt], "shape": list(a.shape), "data_offsets": [off, off + b.size]}
        off += b.size
        blobs.append(b)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        _apply_nocache(fd, True)
        os.write(fd, struct.pack("<Q", len(hj)) + hj)
        for b in blobs:
            mv = memoryview(b)
            while len(mv):
                n = os.write(fd, mv[: 1 << 28])
                mv = mv[n:]
    finally:
        os.close(fd)


def write_safetensors_chunks(path: Path, name: str, dtype: str, shape, nbytes: int, chunks) -> None:
    """One tensor written from an iterator of byte chunks (F_NOCACHE), never whole in memory."""
    import os
    import struct
    from shutter.core.stream import _apply_nocache
    header = {"__metadata__": {"format": "mlx"}, name: {"dtype": dtype, "shape": list(shape),
                                                      "data_offsets": [0, int(nbytes)]}}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        _apply_nocache(fd, True)
        os.write(fd, struct.pack("<Q", len(hj)) + hj)
        done = 0
        for c in chunks:
            mv = memoryview(c)
            while len(mv):
                k = os.write(fd, mv)
                mv = mv[k:]
            done += len(c)
        if done != nbytes:
            raise EOFError(f"{name}: wrote {done} of {nbytes} bytes")
    finally:
        os.close(fd)


def q8_0_to_mlx(d: np.ndarray, q: np.ndarray):
    """Q8_0 (float16 scales [r, c/32], int8 [r, c]) -> MLX 8-bit group 32 (packed uint32, scales, biases) in
    bfloat16. Dequantizes to d_bf16 * q exactly."""
    import mlx.core as mx
    packed = (q.astype(np.int16) + 128).astype(np.uint8)
    packed = np.ascontiguousarray(packed).view(np.uint32)          # little endian: element j at bits 8 * (j % 4)
    scales = mx.array(d.astype(np.float32)).astype(mx.bfloat16)
    biases = scales * -128.0
    return mx.array(packed), scales, biases


def _source(path):
    """A GGUF path, or "-" for stdin (e.g. `ssh host cat file.gguf | python -m ...convert --dit -`)."""
    import sys
    return sys.stdin.buffer if str(path) == "-" else open(path, "rb", buffering=1 << 22)


def convert_dit(src, out: Path, bits: int = 8, log=print) -> dict[str, Any]:
    """The DiT GGUF (read once, in file order) -> transformer/."""
    import mlx.core as mx
    from shutter.families.h3image import gguf as G
    w = ShardWriter(out)
    kept_bf16 = 0
    i = 0
    with _source(src) as f:
        for t, raw in G.stream_tensors(f):
            i += 1
            tgt = _dit_name(t.name)
            if tgt is None:
                continue
            if t.type == G.GGML_Q8_0:
                d, q = G.q8_0_parts(raw, t.shape)
                if bits == 8:
                    wq, s_, b = q8_0_to_mlx(d, q)
                else:
                    full = mx.array(G.decode(t, raw)).astype(mx.bfloat16)
                    wq, s_, b = mx.quantize(full, group_size=64, bits=bits)
                    s_, b = s_.astype(mx.bfloat16), b.astype(mx.bfloat16)
                base = tgt[: -len(".weight")]
                w.add(f"{base}.weight", wq)
                w.add(f"{base}.scales", s_)
                w.add(f"{base}.biases", b)
            elif t.type == G.GGML_BF16 and len(t.shape) == 2 and min(t.shape) > 64:
                arr = mx.array(G.decode(t, raw)).astype(mx.bfloat16)      # a matrix the release kept at 16 bits
                if bits != 8:
                    wq, s_, b = mx.quantize(arr, group_size=64, bits=8)
                    base = tgt[: -len(".weight")]
                    w.add(f"{base}.weight", wq)
                    w.add(f"{base}.scales", s_.astype(mx.bfloat16))
                    w.add(f"{base}.biases", b.astype(mx.bfloat16))
                else:
                    w.add(tgt, arr)
                kept_bf16 += 1
            elif t.type == G.GGML_BF16 and len(t.shape) == 1 and "norm" in t.name:
                w.add(tgt, mx.array(G.decode(t, raw)).astype(mx.bfloat16))   # norm weights: bf16, as released
            elif t.type in (G.GGML_F32, G.GGML_F16, G.GGML_BF16):
                w.add(tgt, mx.array(G.decode(t, raw)))                      # small tensors: float32 (exact)
            else:
                raise ValueError(f"{t.name}: unexpected GGUF type {t.type_name}")
            if i % 25 == 0:
                log(f"  dit {i} tensors ({t.name}), {w.total / 1e9:.1f} GB written")
    total = w.close({"bits": bits, "group_size": 32 if bits == 8 else 64})
    return {"bytes": total, "bf16_matrices": kept_bf16, "bits": bits}


def q4_k_to_mlx(raw: np.ndarray, shape: tuple[int, ...]):
    """Q4_K -> MLX 4-bit group 32, exactly (up to the scale rounded to bfloat16): each 32-value group of a Q4_K
    super-block is w = (d * sc) * q - dmin * m with q in [0, 15], i.e. MLX affine with scale d * sc, bias -dmin * m."""
    import mlx.core as mx
    from shutter.families.h3image import gguf as G
    sc, mn, q = G.q4_k_parts(raw)                    # [nb, 8], [nb, 8], [nb, 8, 32]
    rows, cols = int(np.prod(shape[:-1])), shape[-1]
    q = q.reshape(rows, cols)
    packed = np.zeros((rows, cols // 8), np.uint32)
    for j in range(8):
        packed |= q[:, j::8].astype(np.uint32) << (4 * j)
    scales = mx.array(sc.reshape(rows, cols // 32)).astype(mx.bfloat16)
    biases = mx.array(-mn.reshape(rows, cols // 32)).astype(mx.bfloat16)
    return mx.array(packed), scales, biases


def convert_te_gguf(src, out: Path, log=print) -> dict[str, Any]:
    """The MiniMax-H3 text encoder GGUF (Q4_K_M, already cut to 50 layers) -> text_encoder/. Q4_K maps exactly onto
    MLX 4-bit group 32; Q6_K (v_proj, down_proj) is re-quantized to 8-bit group 32 (no visible loss); BF16 stays.
    The vision tower is left out (text to image only)."""
    import mlx.core as mx
    from shutter.families.h3image import gguf as G
    w = ShardWriter(out)
    n = 0
    big = lambda t: t.type == G.GGML_BF16 and t.nbytes > (256 << 20)      # noqa: E731  (the token embedding)
    with _source(src) as f:
        for t, raw in G.stream_tensors(f, chunked=big):
            if not t.name.startswith("model."):
                continue                            # visual.*
            if big(t):
                name = t.name[len("model."):]
                fname = "embed-00001.safetensors"
                write_safetensors_chunks(out / fname, name, "BF16", t.shape, t.nbytes, raw)
                w.weight_map[name] = fname
                w.total += t.nbytes
                n += 1
                continue
            name = t.name[len("model."):]
            if name.startswith("layers.") and int(name.split(".")[1]) >= TE_LAYERS:
                continue
            if name == "norm.weight":
                continue                            # H3 reads the state before the final norm
            base = name[: -len(".weight")] if name.endswith(".weight") else name
            if t.type == G.GGML_Q4_K:
                wq, s_, b = q4_k_to_mlx(raw, t.shape)
                w.add(f"{base}.weight", wq); w.add(f"{base}.scales", s_); w.add(f"{base}.biases", b)
            elif t.type == G.GGML_Q6_K:
                wq, s_, b = mx.quantize(mx.array(G.decode(t, raw)), group_size=32, bits=8)
                w.add(f"{base}.weight", wq)
                w.add(f"{base}.scales", s_.astype(mx.bfloat16)); w.add(f"{base}.biases", b.astype(mx.bfloat16))
            elif t.type == G.GGML_Q8_0:
                d, q = G.q8_0_parts(raw, t.shape)
                wq, s_, b = q8_0_to_mlx(d, q)
                w.add(f"{base}.weight", wq); w.add(f"{base}.scales", s_); w.add(f"{base}.biases", b)
            elif t.type in (G.GGML_BF16, G.GGML_F16, G.GGML_F32):
                w.add(name, mx.array(G.decode(t, raw)).astype(mx.bfloat16))
            else:
                raise ValueError(f"{t.name}: unexpected GGUF type {t.type_name}")
            n += 1
            if n % 45 == 0:
                log(f"  te {n} tensors ({name}), {w.total / 1e9:.1f} GB written")
    total = w.close({"source": "gguf", "layers": TE_LAYERS})
    return {"bytes": total, "layers": TE_LAYERS, "source": "gguf q4_k_m"}


def _dit_order(name: str):
    import re
    m = re.match(r"^blocks\.(\d+)\.", name)
    return (1, int(m.group(1)), name) if m else (0 if not name.startswith("final_layer") else 2, 0, name)


def _dit_name(name: str) -> Optional[str]:
    if name == "rope.inv_freq":
        return "rope_inv_freq"
    if name.startswith(("audio_", "final_layer.audio_out")):
        return None
    return name


def convert_te(src: Path, out: Path, bits: int = 4, log=print) -> dict[str, Any]:
    """HF Qwen3-VL safetensors -> layers 0..TE_LAYERS-1 + embedding, MLX quantized."""
    import mlx.core as mx
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model."
    want: dict[str, str] = {}
    for k in idx:
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        if rest.startswith("layers."):
            if int(rest.split(".")[1]) >= TE_LAYERS:
                continue
        elif rest != "embed_tokens.weight":
            continue          # the final norm is never used
        want[k] = rest
    by_file: dict[str, list[str]] = {}
    for k in want:
        by_file.setdefault(idx[k], []).append(k)
    from shutter.core.stream import SafetensorsIndex
    w = ShardWriter(out)
    hi = ("self_attn.v_proj", "mlp.down_proj")
    done = 0
    for f in sorted(by_file):
        ix = SafetensorsIndex([str(src / f)], nocache=True)      # read once: keep it out of the file cache
        for k in sorted(by_file[f], key=_te_order):
            name = want[k]
            a = ix.to_mlx(k)
            if name.endswith(".weight") and a.ndim == 2 and name != "embed_tokens.weight":
                b_ = 6 if (bits == 4 and any(h in name for h in hi)) else bits
                wq, s, b = mx.quantize(a.astype(mx.bfloat16), group_size=64, bits=b_)
                base = name[: -len(".weight")]
                w.add(f"{base}.weight", wq)
                w.add(f"{base}.scales", s.astype(mx.bfloat16))
                w.add(f"{base}.biases", b.astype(mx.bfloat16))
            else:
                w.add(name, a.astype(mx.bfloat16))
            done += 1
        ix.close()
        log(f"  te {done}/{len(want)} ({f})")
    total = w.close({"bits": bits, "group_size": 64, "layers": TE_LAYERS})
    return {"bytes": total, "bits": bits, "layers": TE_LAYERS}


def _te_order(k: str):
    import re
    m = re.search(r"layers\.(\d+)\.", k)
    return (int(m.group(1)) if m else -1, k)


def convert_vae(src: Path, out: Path, log=print) -> dict[str, Any]:
    import mlx.core as mx
    arrays = mx.load(str(src))
    w = ShardWriter(out)
    for k in sorted(arrays):
        if k.startswith("decoder."):
            a = arrays[k]
            if k in ("decoder.register_tokens", "decoder.mask_token"):
                a = a.reshape(1, -1, a.shape[-1])
            w.add(k, a)
        elif k.startswith("post_quant_conv."):
            a = arrays[k]
            w.add(k, a.reshape(a.shape[0], -1) if a.ndim > 1 else a)
    total = w.close({"dtype": "float16"})
    log(f"  vae {total / 1e9:.2f} GB")
    return {"bytes": total}


def build(dit: Path, te: Path, vae: Path, tokenizer: Path, out: Path, dit_bits: int = 8, te_bits: int = 4,
          parts: Iterable[str] = ("transformer", "text_encoder", "vae", "tokenizer"), log=print) -> dict[str, Any]:
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    info: dict[str, Any] = {"format": "shuttrr-h3image-1", "family": "h3-image", "variant": "minimax-h3-image",
                            "sources": {}}
    parts = set(parts)
    info["sources"] = {k: str(v) for k, v in (("dit", dit), ("te", te), ("vae", vae)) if v}
    if "transformer" in parts:
        log(f"DiT from {dit}")
        info["transformer"] = convert_dit(dit, out / "transformer", dit_bits, log)
    if "text_encoder" in parts:
        log(f"text encoder from {te}")
        if str(te) == "-" or str(te).endswith(".gguf"):
            info["text_encoder"] = convert_te_gguf(te, out / "text_encoder", log)
        else:
            info["text_encoder"] = convert_te(Path(te), out / "text_encoder", te_bits, log)
    if "vae" in parts and vae:
        info["vae"] = convert_vae(Path(vae), out / "vae", log)
    if "tokenizer" in parts and tokenizer:
        (out / "tokenizer").mkdir(exist_ok=True)
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            if (Path(tokenizer) / f).exists():
                shutil.copy2(Path(tokenizer) / f, out / "tokenizer" / f)
    old = json.loads((out / "h3image.json").read_text()) if (out / "h3image.json").exists() else {}
    old.update(info)
    old["seconds"] = round(time.time() - t0, 1)
    (out / "h3image.json").write_text(json.dumps(old, indent=1))
    (out / "NOTICE").write_text(NOTICE)
    log(f"done in {time.time() - t0:.0f} s -> {out}")
    return old


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", default="", help="DiT GGUF path, or - for stdin")
    ap.add_argument("--te", default="", help="text encoder: HF folder, GGUF path, or - for a GGUF on stdin")
    ap.add_argument("--vae", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dit-bits", type=int, default=8, choices=(4, 8))
    ap.add_argument("--te-bits", type=int, default=4, choices=(4, 8))
    ap.add_argument("--parts", default="transformer,text_encoder,vae,tokenizer")
    a = ap.parse_args(argv)
    import mlx.core as mx
    mx.set_default_device(mx.cpu)            # conversion never touches the GPU
    build(a.dit, a.te, a.vae, a.tokenizer, Path(a.out), a.dit_bits, a.te_bits,
          [p.strip() for p in a.parts.split(",") if p.strip()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
