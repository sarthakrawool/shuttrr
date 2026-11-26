"""Convert any supported Wan checkpoint into Shuttrr's 8-bit low-memory build.

    shutter convert <folder | file | hf-repo-folder> <out> [--bits 8] [--lora file[:strength]]...
                    [--text-encoder DIR] [--vae FILE] [--tokenizer DIR]

The build ("shuttrr" layout) is what the Wan adapter runs:

    shuttrr.json            family, variant, config, parts, sizes (read by `inspect` and the adapter)
    dit/ | dit_high/ + dit_low/
        rest.safetensors    embeddings, head (bf16, modulation in float32)
        blocks-NN-MM.safetensors   attention and FFN linears in 8-bit (group 64), the rest bf16
    text_encoder/           umT5-XXL in bf16, MLX names (run one block at a time, in float32)
    tokenizer/              copied
    vae/model.safetensors   float32, channels-last convs (+ encoder for image-to-video)

The converter itself is low-memory: it reads one tensor at a time from the source (safetensors via os.pread, no
whole-file load) and writes a shard every few blocks, so a 14B model converts on a 16 GB Mac. LoRAs given with
--lora are merged into the weights before quantization (a merged build; LoRAs can also be applied at run time
without converting).

8-bit with group size 64 is the same scheme as LongCat's low-memory build: the DiT's attention and FFN linears
(about 97% of its weights) are quantized; embeddings, norms, the head and the modulation stay in bf16/fp32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from shutter.families.wan import keys as K
from shutter.families.wan.detect import recognise
from shutter.families.base import Checkpoint

GB = 1024**3
BLOCKS_PER_SHARD = 4
FP32_KEEP = (".modulation",)          # kept in float32, as the reference keeps them


class ConvertError(RuntimeError):
    pass


def _mx():
    import mlx.core as mx
    return mx


def _np_to_mx(arr, dtype_name: str):
    """numpy buffer from SafetensorsIndex.read -> mx.array of the stored dtype."""
    from shutter.core.stream import to_mlx_array
    return to_mlx_array(arr, dtype_name)


class Source:
    """Read tensors one at a time from a list of safetensors files."""

    def __init__(self, files: list[str]):
        from shutter.core.stream import SafetensorsIndex
        self.index = SafetensorsIndex(files)

    def names(self) -> list[str]:
        return self.index.names()

    def get(self, name: str):
        e = self.index.entries[name]
        return _np_to_mx(self.index.read(name), e.dtype)

    def close(self):
        self.index.close()


class TorchSource:
    """A .pth file (needs PyTorch). Loaded whole on the CPU: these are the text encoder and VAE, not the DiT."""

    def __init__(self, path: str):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ConvertError(f"{Path(path).name} is a PyTorch file and PyTorch is not installed. Install it "
                               "(pip install torch) or give a safetensors copy (--text-encoder / --vae).") from None
        import torch
        self.sd = torch.load(path, map_location="cpu", weights_only=True)

    def names(self) -> list[str]:
        return list(self.sd.keys())

    def get(self, name: str):
        mx = _mx()
        t = self.sd[name]
        if str(t.dtype) == "torch.bfloat16":
            return mx.array(t.float().numpy()).astype(mx.bfloat16)
        return mx.array(t.numpy())

    def close(self):
        self.sd = None


def _open(files: list[str]):
    if len(files) == 1 and files[0].endswith(".pth"):
        return TorchSource(files[0])
    return Source(files)


# ----------------------------------------------------------------------------- DiT


def _quantize(w, bits: int, group: int):
    mx = _mx()
    wq, scales, biases = mx.quantize(w.astype(mx.bfloat16), group_size=group, bits=bits)
    return wq, scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)


def convert_dit(files: list[str], layout: str, out_dir: Path, *, bits: Optional[int], group: int,
                loras: Optional[dict[str, list]] = None, report: Callable[[str], None] = print) -> dict[str, Any]:
    mx = _mx()
    src = Source(files)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_block: dict[int, list[tuple[str, str]]] = {}
    rest: list[tuple[str, str]] = []
    for name in src.names():
        mlx_name, transform = K.dit_to_mlx(name, layout)
        if mlx_name is None:
            continue
        parts = mlx_name.split(".")
        if parts[0] == "blocks":
            by_block.setdefault(int(parts[1]), []).append((name, mlx_name))
        else:
            rest.append((name, mlx_name))
    n_blocks = len(by_block)

    def convert_one(src_name: str, mlx_name: str) -> dict[str, Any]:
        _, transform = K.dit_to_mlx(src_name, layout)
        w = src.get(src_name)
        if transform == "flatten":
            w = w.reshape(w.shape[0], -1)
        module, _, leaf = mlx_name.rpartition(".")
        if loras and leaf == "weight" and module in loras:
            w = w.astype(mx.float32)
            for a, b, s in loras[module]:
                w = w + s * (b.astype(mx.float32) @ a.astype(mx.float32))
        if mlx_name.endswith(FP32_KEEP):
            return {mlx_name: w.astype(mx.float32)}
        if bits and leaf == "weight" and K.is_quantized_linear(module):
            wq, sc, bi = _quantize(w, bits, group)
            return {mlx_name: wq, f"{module}.scales": sc, f"{module}.biases": bi}
        return {mlx_name: w.astype(mx.bfloat16)}

    files_out, sizes = [], {}
    shard: dict[str, Any] = {}
    for src_name, mlx_name in rest:
        shard.update(convert_one(src_name, mlx_name))
    mx.eval(shard)
    rest_bytes = sum(v.nbytes for v in shard.values())
    mx.save_safetensors(str(out_dir / "rest.safetensors"), shard)
    files_out.append("rest.safetensors")
    shard = {}
    block_bytes = 0
    t0 = time.time()
    ids = sorted(by_block)
    for start in range(0, len(ids), BLOCKS_PER_SHARD):
        group_ids = ids[start:start + BLOCKS_PER_SHARD]
        for i in group_ids:
            blk = {}
            for src_name, mlx_name in by_block[i]:
                blk.update(convert_one(src_name, mlx_name))
            mx.eval(blk)
            block_bytes = max(block_bytes, sum(v.nbytes for v in blk.values()))
            shard.update(blk)
        fn = f"blocks-{group_ids[0]:02d}-{group_ids[-1]:02d}.safetensors"
        mx.save_safetensors(str(out_dir / fn), shard)
        files_out.append(fn)
        shard = {}
        mx.clear_cache()
        report(f"  blocks {group_ids[0]}-{group_ids[-1]} of {n_blocks} written ({time.time() - t0:.0f} s)")
    src.close()
    total = sum((out_dir / f).stat().st_size for f in files_out)
    return {"role": "denoiser", "files": [f"{out_dir.name}/{f}" for f in files_out], "n_blocks": n_blocks,
            "block_bytes": block_bytes, "rest_bytes": rest_bytes, "total_bytes": total,
            "quant": {"bits": bits, "group_size": group} if bits else None, "dtype": "U32" if bits else "BF16"}


# ----------------------------------------------------------------------------- text encoder


def _copy_quantized_t5(files: list[str], out_dir: Path, report: Callable[[str], None]) -> dict[str, Any]:
    """An already 8-bit Shuttrr text encoder (text_encoder_q8/): copied as it is (hard links when on the same disk)."""
    from shutter.core.text_encode import INDEX, folder_quantization
    src_dir = Path(files[0]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(Path(f).name for f in files)
    for n in names + [INDEX]:
        dst = out_dir / n
        if dst.exists():
            dst.unlink()
        try:
            dst.hardlink_to(src_dir / n)
        except OSError:
            shutil.copyfile(src_dir / n, dst)
    q = folder_quantization(out_dir) or {"bits": 8, "group_size": 64}
    report(f"  text encoder: copied the {q['bits']}-bit copy")
    names.sort(key=lambda n: (n != "rest.safetensors", n))
    return {"role": "text_encoder", "files": [f"{out_dir.name}/{f}" for f in names], "n_blocks": 24,
            "total_bytes": sum((out_dir / f).stat().st_size for f in names), "dtype": "U32",
            "quant": q}


def convert_t5(files: list[str], layout: str, out_dir: Path, report: Callable[[str], None] = print) -> dict[str, Any]:
    if layout == "mlx-q":
        return _copy_quantized_t5(files, out_dir.with_name(out_dir.name + "_q8"), report)
    mx = _mx()
    src = _open(files)
    names = src.names()
    lay = "hf" if layout == "hf-t5" else ("wan" if layout in ("official", "pth") else K.t5_layout(names))
    if not lay:
        lay = K.t5_layout(names)
    if not lay:
        raise ConvertError("The text encoder's tensor names are neither Wan's umT5 nor Hugging Face T5.")
    out_dir.mkdir(parents=True, exist_ok=True)
    mapped = [(n, K.t5_to_mlx(n, lay)) for n in names]
    mapped = [(a, b) for a, b in mapped if b]
    if not any(b == "token_embedding.weight" for _, b in mapped):
        raise ConvertError("The text encoder has no token embedding.")
    by_block: dict[int, list] = {}
    other = []
    for a, b in mapped:
        p = b.split(".")
        (by_block.setdefault(int(p[1]), []) if p[0] == "blocks" else other).append((a, b))
    files_out = []
    shard = {b: src.get(a).astype(mx.bfloat16) for a, b in other}
    mx.save_safetensors(str(out_dir / "rest.safetensors"), shard)
    files_out.append("rest.safetensors")
    weight_map = {k: "rest.safetensors" for k in shard}
    ids = sorted(by_block)
    for start in range(0, len(ids), 6):
        g = ids[start:start + 6]
        shard = {}
        for i in g:
            for a, b in by_block[i]:
                shard[b] = src.get(a).astype(mx.bfloat16)
        fn = f"blocks-{g[0]:02d}-{g[-1]:02d}.safetensors"
        mx.save_safetensors(str(out_dir / fn), shard)
        files_out.append(fn)
        weight_map.update({k: fn for k in shard})
        mx.clear_cache()
    (out_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}, indent=1))
    src.close()
    report(f"  text encoder: {len(ids)} blocks")
    return {"role": "text_encoder", "files": [f"{out_dir.name}/{f}" for f in files_out], "n_blocks": len(ids),
            "total_bytes": sum((out_dir / f).stat().st_size for f in files_out), "dtype": "BF16"}


# ----------------------------------------------------------------------------- VAE


def convert_vae(files: list[str], kind: str, out_dir: Path, include_encoder: bool) -> dict[str, Any]:
    mx = _mx()
    src = _open(files)
    raw = {n: src.get(n) for n in src.names()}
    src.close()
    if kind == "wan2.2":
        from shutter.families.wan.model.vae22 import sanitize_wan22_vae_weights
        w = sanitize_wan22_vae_weights(raw, include_encoder=include_encoder)
    else:
        w = {}
        for k, v in raw.items():
            if not include_encoder and (k.startswith("encoder.") or k.startswith("conv1.")):
                continue
            if k.endswith("weight") and v.ndim == 5:
                v = mx.transpose(v, (0, 2, 3, 4, 1))
            elif k.endswith("weight") and v.ndim == 4:
                v = mx.transpose(v, (0, 2, 3, 1))
            w[k] = v
    w = {k: v.astype(mx.float32) for k, v in w.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "model.safetensors"), w)
    return {"role": "decoder", "files": [f"{out_dir.name}/model.safetensors"], "kind": kind,
            "encoder": include_encoder, "total_bytes": (out_dir / "model.safetensors").stat().st_size,
            "dtype": "F32"}


# ----------------------------------------------------------------------------- driver


def _fingerprint(files: list[str]) -> str:
    h = hashlib.sha1()
    for f in sorted(files):
        p = Path(f)
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
        with open(p, "rb") as fh:
            h.update(fh.read(1 << 16))
    return h.hexdigest()[:16]


def plan(source: Path, *, text_encoder: Optional[Path] = None, vae: Optional[Path] = None,
         tokenizer: Optional[Path] = None) -> Checkpoint:
    """Inspect `source` and fill missing parts from the given folders/files. Raises ConvertError with the reasons
    when it cannot be converted."""
    ck = recognise(Path(source))
    if ck is None:
        raise ConvertError(f"{source} is not a Wan checkpoint Shuttrr recognises.")
    if ck.variant == "lora":
        raise ConvertError("This is a LoRA. Give it with --lora when converting a base model, or add it to a "
                           "request at run time.")
    from shutter.families.wan.detect import _find_t5, _find_tokenizer, _find_vae
    for name, given, finder in (("text_encoder", text_encoder, _find_t5), ("vae", vae, _find_vae),
                                ("tokenizer", tokenizer, _find_tokenizer)):
        if given is None:
            continue
        given = Path(given).expanduser()
        part = None
        if name == "vae" and given.is_file():
            from shutter.families.wan.detect import read_header
            if given.suffix == ".pth":
                kind = "wan2.2" if "2.2" in given.name else "wan2.1"
                from shutter.families.base import Part
                part = Part("vae", "decoder", [str(given)], f"pth:{kind}", given.stat().st_size)
            else:
                h = read_header(given)
                kind = K.vae_kind(h, {k: v["shape"] for k, v in h.items()})
                from shutter.families.base import Part
                part = Part("vae", "decoder", [str(given)], kind, given.stat().st_size) if kind else None
        else:
            part = finder(given if given.is_dir() else given.parent)
        if part is None:
            raise ConvertError(f"No {name.replace('_', ' ')} found in {given}.")
        ck.parts[name] = part
        if name in ck.missing:
            ck.missing.remove(name)
    if ck.missing:
        ck.problems.append("Missing: " + ", ".join(m.replace("_", " ") for m in ck.missing)
                           + ". Name them with --text-encoder / --vae / --tokenizer (an installed LongCat or Wan "
                             "model has all three).")
    # re-check the VAE kind against the model now that it may have been supplied
    want = "wan2.2" if ck.config.get("vae_z_dim") == 48 else "wan2.1"
    v = ck.parts.get("vae")
    if v is not None and v.layout.replace("pth:", "") not in (want, "shuttrr"):
        msg = f"The VAE given is {v.layout.replace('pth:', '')}, but this model needs the {want} VAE."
        if msg not in ck.problems:
            ck.problems = [p for p in ck.problems if "VAE" not in p] + [msg]
    if ck.problems:
        raise ConvertError("Cannot convert " + str(source) + ":\n- " + "\n- ".join(ck.problems))
    return ck


def convert(source: Path, out: Path, *, bits: Optional[int] = 8, group: int = 64,
            loras: Optional[list[tuple[str, float]]] = None, text_encoder: Optional[Path] = None,
            vae: Optional[Path] = None, tokenizer: Optional[Path] = None,
            report: Callable[[str], None] = print) -> dict[str, Any]:
    ck = plan(source, text_encoder=text_encoder, vae=vae, tokenizer=tokenizer)
    if not ck.needs_conversion:
        raise ConvertError(f"{source} is already a Shuttrr build.")
    out = Path(out).expanduser()
    if (out / "shuttrr.json").exists():
        raise ConvertError(f"{out} already holds a build; choose an empty folder.")
    out.mkdir(parents=True, exist_ok=True)
    from shutter.families.wan.loader import merge_loras, read_lora
    merged = merge_loras(*[read_lora(Path(p), s) for p, s in (loras or [])]) if loras else None
    started = time.time()
    parts: dict[str, Any] = {}
    for name in ("dit", "dit_high", "dit_low"):
        p = ck.parts.get(name)
        if p is None:
            continue
        report(f"Converting {name} ({p.n_blocks} blocks, {p.total_bytes / GB:.1f} GB, {p.layout} names)")
        parts[name] = convert_dit(p.files, p.layout, out / name, bits=bits, group=group, loras=merged,
                                  report=report)
    te = ck.parts["text_encoder"]
    report(f"Converting the text encoder ({te.layout})")
    parts["text_encoder"] = convert_t5(te.files, te.layout, out / "text_encoder", report=report)
    tok = Path(ck.parts["tokenizer"].files[0])
    shutil.copytree(tok, out / "tokenizer", dirs_exist_ok=True)
    parts["tokenizer"] = {"role": "tokenizer", "files": ["tokenizer"]}
    v = ck.parts["vae"]
    kind = v.layout.replace("pth:", "")
    report(f"Converting the VAE ({kind})")
    parts["vae"] = convert_vae(v.files, kind, out / "vae", include_encoder=ck.task in ("i2v", "ti2v"))
    manifest = {
        "family": "wan", "format": 1, "variant": ck.variant, "label": ck.label, "config": ck.config,
        "parts": parts, "source": {"path": str(source), "fingerprint": _fingerprint(
            [f for p in ck.parts.values() for f in p.files if Path(f).is_file()][:8])},
        "loras_merged": [{"file": Path(p).name, "strength": s} for p, s in (loras or [])],
        "converted": time.strftime("%Y-%m-%d"), "converter_seconds": round(time.time() - started, 1),
        "notes": ck.notes, "checked_on": [],
    }
    (out / "shuttrr.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (out / "NOTICE").write_text(
        "Converted by Shuttrr from the Wan model weights (Apache-2.0, Wan-AI / Alibaba). Model code: mlx-video\n"
        "(MIT, Prince Canuma). This is an unofficial 8-bit build; see the original model card for the license\n"
        "and acceptable-use terms.\n")
    report(f"Done in {time.time() - started:.0f} s: {out}")
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="shutter convert", description="Convert a Wan checkpoint into Shuttrr's "
                                 "8-bit low-memory build.")
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=8, choices=[0, 4, 8], help="0 keeps bf16 (no quantization)")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--lora", action="append", default=[], help="file[:strength], merged before quantizing")
    ap.add_argument("--text-encoder")
    ap.add_argument("--vae")
    ap.add_argument("--tokenizer")
    a = ap.parse_args(argv)
    from shutter.families.wan.sources import parse_lora
    loras = [parse_lora(spec) for spec in a.lora]
    try:
        convert(Path(a.source), Path(a.out), bits=a.bits or None, group=a.group_size, loras=loras,
                text_encoder=Path(a.text_encoder) if a.text_encoder else None,
                vae=Path(a.vae) if a.vae else None, tokenizer=Path(a.tokenizer) if a.tokenizer else None)
    except ConvertError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
