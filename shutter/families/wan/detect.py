"""Recognise a Wan checkpoint from headers alone, and say plainly what it cannot do.

Accepted inputs (a folder from Hugging Face or anywhere on disk, or a single file):

* an official Wan-AI repo (``diffusion_pytorch_model*.safetensors`` at the top, or ``high_noise_model/`` +
  ``low_noise_model/`` for Wan 2.2 A14B, with ``models_t5_umt5-xxl-enc-bf16.pth`` and ``Wan2.x_VAE.pth``);
* a ``*-Diffusers`` repo (``transformer/`` [+ ``transformer_2/``], ``text_encoder/``, ``vae/``, ``tokenizer/``);
* a ComfyUI repack or a fine-tune: one DiT ``.safetensors`` file (official or ``model.diffusion_model.`` names),
  with the text encoder and VAE taken from elsewhere (``missing`` lists what to supply);
* a LoRA ``.safetensors`` (recognised, applied at conversion or at run time);
* a Shuttrr build (``shuttrr.json``), which runs as it is.

Only safetensors headers are read. ``.pth`` files are pickles: they are listed, and converting them needs PyTorch.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Optional

from shutter.families.base import Checkpoint, Part
from shutter.families.wan import keys as K
from shutter.families.wan.spec import VARIANTS, WanShape, classify

GB = 1024**3

_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "U32": 4, "I16": 2, "U16": 2,
                "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def read_header(path: Path) -> dict[str, dict[str, Any]]:
    """{name: {"dtype", "shape", "nbytes"}} of one .safetensors file (header only)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 100 * 1024 * 1024:
            raise ValueError(f"{path.name} is not a safetensors file (header size {n})")
        hdr = json.loads(f.read(n))
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out[k] = {"dtype": v["dtype"], "shape": tuple(v["shape"]), "nbytes": b - a, "file": str(path)}
    return out


def read_headers(files: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in files:
        out.update(read_header(f))
    return out


def _st_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.safetensors") if not p.name.startswith("."))


def _dominant_dtype(h: dict[str, dict[str, Any]]) -> str:
    acc: dict[str, int] = {}
    for v in h.values():
        acc[v["dtype"]] = acc.get(v["dtype"], 0) + v["nbytes"]
    return max(acc, key=acc.get) if acc else ""


def is_lora(h: dict[str, dict[str, Any]]) -> bool:
    return bool(h) and sum(1 for k in h if K.lora_target(k) and K.lora_target(k)[1] in ("down", "up")) >= len(h) // 2


# ----------------------------------------------------------------------------- DiT analysis


def dit_shape(h: dict[str, dict[str, Any]], layout: str) -> tuple[Optional[WanShape], list[str], dict[str, Any]]:
    """(shape, problems, info) from a DiT's tensor names/shapes, in any layout."""
    problems: list[str] = []
    off: dict[str, dict[str, Any]] = {}
    for name, v in h.items():
        if layout == "shuttrr":
            off[name] = v
            continue
        o = K.to_official(name, layout)
        if o is not None:
            off[o] = v
    info: dict[str, Any] = {"layout": layout}
    for rx, why in K.UNSUPPORTED_MARKERS:
        if any(rx.search(n) for n in off):
            problems.append(f"Not supported: {why}")
    fp8 = [n for n, v in off.items() if v["dtype"].startswith("F8")]
    if fp8:
        problems.append("Not supported yet: the weights are stored in 8-bit float (fp8). Use the bf16 or fp16 "
                        "release of this model; Shuttrr makes its own 8-bit build from it.")
    if any(n.endswith(".scale_weight") or n == "scaled_fp8" for n in off):
        problems.append("Not supported yet: this is a 'scaled fp8' repack. Use the bf16 or fp16 release.")
    if layout == "shuttrr":
        pe = off.get("patch_embedding_proj.weight")
        if pe is None:
            return None, problems + ["The Shuttrr build has no patch embedding; it is incomplete."], info
        dim, in_dim = pe["shape"][0], pe["shape"][1] // 4
        head = off.get("head.head.weight")
        fc1 = next((v for n, v in off.items() if n.startswith("blocks.0.ffn.fc1.")), None)
        ffn = fc1["shape"][0] if fc1 else 0
        text = off.get("text_embedding_0.weight", {"shape": (0, 4096)})["shape"][1]
    else:
        pe = off.get("patch_embedding.weight")
        if pe is None:
            return None, problems + ["No patch embedding found: this is not a complete Wan DiT."], info
        dim, in_dim = pe["shape"][0], pe["shape"][1]
        if len(pe["shape"]) == 5 and tuple(pe["shape"][2:]) != (1, 2, 2):
            problems.append(f"Unexpected patch size {tuple(pe['shape'][2:])}; Wan uses (1, 2, 2).")
        head = off.get("head.head.weight")
        fc1 = off.get("blocks.0.ffn.0.weight")
        ffn = fc1["shape"][0] if fc1 else 0
        text = off.get("text_embedding.0.weight", {"shape": (0, 4096)})["shape"][1]
    blocks = {int(n.split(".")[1]) for n in off if n.startswith("blocks.") and n.split(".")[1].isdigit()}
    n_layers = (max(blocks) + 1) if blocks else 0
    if blocks and len(blocks) != n_layers:
        problems.append(f"Blocks are missing (found {len(blocks)} of {n_layers}); the file looks truncated.")
    if head is None:
        problems.append("No output head found: this is not a complete Wan DiT.")
        return None, problems, info
    out_dim = head["shape"][0] // 4
    if dim % 128:
        problems.append(f"Width {dim} is not a multiple of 128, so this is not a Wan-architecture model.")
    shape = WanShape(dim=dim, in_dim=in_dim, out_dim=out_dim, ffn_dim=ffn, num_layers=n_layers, text_dim=text,
                     cross_attn_norm=any(".norm3." in n for n in off),
                     qk_norm=any(".self_attn.norm_q." in n for n in off))
    if text != 4096:
        problems.append(f"Text width {text}: Wan uses umT5-XXL (4096). This model expects another text encoder.")
    info["blocks"] = n_layers
    return shape, problems, info


def _block_bytes(h: dict[str, dict[str, Any]], layout: str) -> tuple[int, int]:
    """(largest block, rest) as stored."""
    per: dict[int, int] = {}
    rest = 0
    for name, v in h.items():
        o = name if layout == "shuttrr" else (K.to_official(name, layout) or name)
        parts = o.split(".")
        if parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
            per[int(parts[1])] = per.get(int(parts[1]), 0) + v["nbytes"]
        else:
            rest += v["nbytes"]
    return (max(per.values()) if per else 0), rest


def dit_part(name: str, files: list[Path], h: dict[str, dict[str, Any]], layout: str, n_blocks: int) -> Part:
    bb, rest = _block_bytes(h, layout)
    return Part(name=name, role="denoiser", files=[str(f) for f in files], layout=layout,
                total_bytes=sum(v["nbytes"] for v in h.values()), n_blocks=n_blocks, block_bytes=bb,
                rest_bytes=rest, dtype=_dominant_dtype(h))


# ----------------------------------------------------------------------------- folder discovery


def _find_t5(root: Path) -> Optional[Part]:
    for cand in (root / "text_encoder_q8", root / "text_encoder", root / "components" / "text_encoder",
                 root / "split_files" / "text_encoders", root):
        if not cand.is_dir():
            continue
        files = [f for f in _st_files(cand) if cand.name.startswith("text_encoder") or "umt5" in f.name.lower()
                 or "t5" in f.name.lower() or cand == root]
        # keep only files whose own tensors are umT5 (a folder may also hold the DiT)
        files = [f for f in files if K.t5_layout(read_header(f)) or
                 any(k.startswith(("blocks.", "encoder.block.")) and ".attn." in k or "SelfAttention" in k
                     for k in read_header(f)) and not K.dit_layout(read_header(f))]
        groups = [files]
        if cand.name.startswith("text_encoder"):
            # a sharded copy (a Shuttrr build): the embedding and the blocks sit in different files
            groups.append(_st_files(cand))
        for files in groups:
            if not files:
                continue
            h = read_headers(files)
            lay = K.t5_layout(h)
            if lay:
                # an 8-bit Shuttrr copy (text_encoder_q8/): MLX names, packed weights with scales and biases
                q8 = lay == "wan" and any(k.endswith(".scales") for k in h)
                return Part("text_encoder", "text_encoder", [str(f) for f in files],
                            "mlx-q" if q8 else ("hf-t5" if lay == "hf" else "official"),
                            sum(v["nbytes"] for v in h.values()), n_blocks=24, dtype=_dominant_dtype(h))
    pth = sorted(root.glob("models_t5_umt5*.pth"))
    if pth:
        return Part("text_encoder", "text_encoder", [str(pth[0])], "pth", pth[0].stat().st_size, n_blocks=24)
    return None


def _find_vae(root: Path) -> Optional[Part]:
    for cand in (root / "vae", root / "split_files" / "vae", root):
        if not cand.is_dir():
            continue
        for f in _st_files(cand):
            if cand.name != "vae" and "vae" not in f.name.lower():
                continue
            h = read_header(f)
            kind = K.vae_kind(h, {k: v["shape"] for k, v in h.items()})
            if kind:
                return Part("vae", "decoder", [str(f)], kind, sum(v["nbytes"] for v in h.values()),
                            dtype=_dominant_dtype(h))
    for nm, kind in (("Wan2.2_VAE.pth", "wan2.2"), ("Wan2.1_VAE.pth", "wan2.1")):
        if (root / nm).exists():
            return Part("vae", "decoder", [str(root / nm)], f"pth:{kind}", (root / nm).stat().st_size)
    return None


def _find_tokenizer(root: Path) -> Optional[Part]:
    for cand in (root / "tokenizer", root / "google" / "umt5-xxl", root / "components" / "tokenizer", root):
        if (cand / "tokenizer_config.json").exists() or (cand / "spiece.model").exists() \
                or (cand / "tokenizer.json").exists():
            return Part("tokenizer", "tokenizer", [str(cand)], "hf")
    return None


def _dit_candidates(root: Path) -> list[tuple[str, list[Path]]]:
    """[(part name, files)] for the DiT(s) in a folder."""
    if (root / "high_noise_model").is_dir() and (root / "low_noise_model").is_dir():
        return [("dit_high", _st_files(root / "high_noise_model")), ("dit_low", _st_files(root / "low_noise_model"))]
    if (root / "transformer").is_dir() and (root / "transformer_2").is_dir():
        # diffusers Wan 2.2 A14B: transformer = high-noise expert, transformer_2 = low-noise expert
        return [("dit_high", _st_files(root / "transformer")), ("dit_low", _st_files(root / "transformer_2"))]
    if (root / "transformer").is_dir():
        return [("dit", _st_files(root / "transformer"))]
    if (root / "dit").is_dir():
        return [("dit", _st_files(root / "dit"))]
    top = [f for f in _st_files(root) if f.name.startswith("diffusion_pytorch_model") or f.name.startswith("model")]
    if top:
        return [("dit", top)]
    dm = root / "split_files" / "diffusion_models"
    if dm.is_dir():
        files = _st_files(dm)
        if len(files) == 1:
            return [("dit", files)]
        if files:
            return [("ambiguous", files)]
    return []


# ----------------------------------------------------------------------------- entry point


def recognise(path: Path) -> Optional[Checkpoint]:
    path = Path(path).expanduser()
    if not path.exists():
        return None
    if path.is_file():
        return _recognise_file(path)
    return _recognise_folder(path)


def _recognise_file(path: Path) -> Optional[Checkpoint]:
    if path.suffix == ".gguf":
        ck = Checkpoint(path=str(path), family="wan")
        ck.problems.append("GGUF files are not supported. Shuttrr converts from the safetensors release of a model "
                           "into its own 8-bit build; point it at the .safetensors file or the Hugging Face repo.")
        return ck if "wan" in path.name.lower() else None
    if path.suffix != ".safetensors":
        return None
    h = read_header(path)
    if is_lora(h):
        return _lora_checkpoint(path, h)
    layout = K.dit_layout(h)
    if not layout:
        return None
    return _finish(path, [("dit", [path], h, layout)], path.parent, bare=True)


def _lora_checkpoint(path: Path, h: dict[str, dict[str, Any]]) -> Optional[Checkpoint]:
    mods = {}
    for k in h:
        t = K.lora_target(k)
        if t:
            mods.setdefault(t[0], set()).add(t[1])
    if not any(m.startswith("blocks.") for m in mods):
        return None
    ck = Checkpoint(path=str(path), family="wan", variant="lora", label="Wan LoRA", task="lora",
                    needs_conversion=False)
    # rank and width tell which Wan size it was trained for
    dims = set()
    for k, v in h.items():
        t = K.lora_target(k)
        if t and t[1] == "up" and t[0].endswith(".self_attn.q"):
            dims.add(v["shape"][0])
    ck.config = {"modules": len(mods), "dims": sorted(dims),
                 "blocks": len({m.split(".")[1] for m in mods if m.startswith("blocks.")})}
    size = {1536: "Wan 1.3B", 3072: "Wan 2.2 5B", 5120: "Wan 14B (2.1, or 2.2 A14B)"}
    if dims:
        ck.label = "Wan LoRA for " + " / ".join(size.get(d, f"width {d}") for d in sorted(dims))
    unknown = [m for m in mods if not m.startswith("blocks.")]
    if unknown:
        ck.notes.append(f"{len(unknown)} LoRA targets outside the blocks are ignored (e.g. {sorted(unknown)[0]}).")
    ck.parts["lora"] = Part("lora", "lora", [str(path)], "lora", path.stat().st_size, dtype=_dominant_dtype(h))
    return ck


def _recognise_folder(root: Path) -> Optional[Checkpoint]:
    manifest = root / "shuttrr.json"
    if manifest.exists():
        return _recognise_build(root, json.loads(manifest.read_text()))
    cands = _dit_candidates(root)
    if not cands:
        return None
    if cands[0][0] == "ambiguous":
        ck = Checkpoint(path=str(root), family="wan")
        names = ", ".join(f.name for f in cands[0][1][:6])
        ck.problems.append(f"This folder holds {len(cands[0][1])} different models ({names}). "
                           "Point Shuttrr at the one file you want.")
        return ck
    dits = []
    for name, files in cands:
        if not files:
            continue
        h = read_headers(files)
        layout = K.dit_layout(h)
        if not layout:
            if is_lora(h):
                return _lora_checkpoint(files[0], h)
            return None
        dits.append((name, files, h, layout))
    if not dits:
        return None
    return _finish(root, dits, root, bare=False)


def _finish(path: Path, dits: list, root: Path, bare: bool) -> Checkpoint:
    ck = Checkpoint(path=str(path), family="wan")
    shapes = []
    for name, files, h, layout in dits:
        shape, problems, info = dit_shape(h, layout)
        ck.problems.extend(p for p in problems if p not in ck.problems)
        if shape is None:
            return ck
        shapes.append(shape)
        ck.parts[name] = dit_part(name, files, h, layout, shape.num_layers)
    s0 = shapes[0]
    if len(shapes) == 2:
        if (shapes[1].dim, shapes[1].in_dim, shapes[1].num_layers) != (s0.dim, s0.in_dim, s0.num_layers):
            ck.problems.append("The two experts have different shapes, so they are not a Wan 2.2 A14B pair.")
        s0.dual = True
    variant = classify(s0)
    ck.variant = variant or "wan-custom"
    v = VARIANTS.get(variant, {})
    ck.label = v.get("label") or f"Wan-architecture model ({s0.num_layers} blocks, width {s0.dim})"
    ck.config = s0.to_config(variant)
    ck.task = ck.config["model_type"]
    if not variant:
        ck.notes.append("No preset for this size: Shuttrr uses Wan defaults (50 steps, guidance 5, shift 5). "
                        "Check the model card for its own settings.")
    if s0.dim == 5120 and not s0.dual and s0.in_dim == 36 and not ck.problems:
        ck.problems.append("This is one expert of Wan 2.2 image-to-video A14B. Shuttrr needs both experts "
                           "(high noise and low noise); point it at the folder that holds both.")
    if s0.dim == 5120 and not s0.dual and s0.in_dim == 16:
        ck.notes.append("If this is one expert of Wan 2.2 A14B rather than Wan 2.1 14B, point Shuttrr at the folder "
                        "with both experts; on its own it runs as a single-model Wan 2.1 14B.")
    # the other parts
    te, vae, tok = _find_t5(root), _find_vae(root), _find_tokenizer(root)
    for part, what in ((te, "text_encoder"), (vae, "vae"), (tok, "tokenizer")):
        if part is not None:
            ck.parts[what] = part
        else:
            ck.missing.append(what)
    want_vae = "wan2.2" if ck.config["vae_z_dim"] == 48 else "wan2.1"
    if vae is not None:
        got = vae.layout.replace("pth:", "")
        if got == "diffusers":
            ck.problems.append("The VAE is in diffusers format, which the converter does not read yet. Supply the "
                               f"official {want_vae.replace('wan', 'Wan')} VAE file (Wan2.x_VAE.pth or ComfyUI's "
                               "wan_2.1_vae / wan2.2_vae .safetensors).")
        elif got != want_vae:
            ck.problems.append(f"The VAE in this folder is {got.replace('wan', 'Wan ')}, but this model needs the "
                               f"{want_vae.replace('wan', 'Wan ')} VAE.")
    pth = [p for p in ck.parts.values() if p.layout.startswith("pth")]
    if pth:
        ck.notes.append("Some parts are PyTorch .pth files (" + ", ".join(p.name for p in pth) + "); converting them "
                        "needs PyTorch in Shuttrr's Python (pip install torch). A text encoder can also be reused "
                        "from an installed LongCat model.")
    if bare and ck.missing:
        ck.notes.append("This file is the video model only. The converter takes the text encoder, tokenizer and VAE "
                        "from an installed Wan or LongCat model, or from folders you name.")
    return ck


def _recognise_build(root: Path, man: dict[str, Any]) -> Checkpoint:
    ck = Checkpoint(path=str(root), family="wan", variant=man.get("variant", "wan-custom"),
                    label=man.get("label", "Wan"), task=man.get("config", {}).get("model_type", ""),
                    config=man.get("config", {}), needs_conversion=False)
    if man.get("family") != "wan":
        ck.problems.append(f"shuttrr.json says family {man.get('family')!r}, not 'wan'.")
    for name, p in man.get("parts", {}).items():
        files = [str(root / f) for f in p.get("files", [])]
        absent = [f for f in files if not Path(f).exists()]
        if absent:
            ck.problems.append(f"The build is incomplete: {Path(absent[0]).name} is missing from {name}.")
        ck.parts[name] = Part(name=name, role=p.get("role", ""), files=files, layout="shuttrr",
                              total_bytes=p.get("total_bytes", 0), n_blocks=p.get("n_blocks", 0),
                              block_bytes=p.get("block_bytes", 0), rest_bytes=p.get("rest_bytes", 0),
                              dtype=p.get("dtype", ""), quant=p.get("quant"))
    for need in ("text_encoder", "vae", "tokenizer"):
        if need not in ck.parts:
            ck.problems.append(f"The build has no {need.replace('_', ' ')}.")
    if not any(p.role == "denoiser" for p in ck.parts.values()):
        ck.problems.append("The build has no video model (dit).")
    ck.notes.extend(man.get("notes", []))
    return ck
