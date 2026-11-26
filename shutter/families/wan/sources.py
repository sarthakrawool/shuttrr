"""Where the parts of a Wan conversion come from, when the checkpoint does not bring them all.

A fine-tune is often only the video model (one .safetensors file), and the official Wan-AI repos ship the text
encoder and VAE as PyTorch `.pth` files, which need PyTorch to read. Both cases can borrow the missing parts from a
model that is already installed:

* a Shuttrr Wan build (``shuttrr.json``): its umT5 text encoder and tokenizer;
* a LongCat model: its umT5 text encoder (``text_encoder/`` or ``components/text_encoder/``) and tokenizer, and its
  VAE, but only when its tensors are recognised as the Wan 2.1 VAE (LongCat's own VAE format may differ).

A Wan 2.2 TI2V 5B needs the Wan 2.2 VAE, which nothing else installed has; that one comes from the source repo.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Iterable, Optional

from shutter.families.base import Checkpoint, Part

PARTS = ("text_encoder", "vae", "tokenizer")


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def parse_lora(spec: str) -> tuple[str, float]:
    """``file[:strength]`` -> (file, strength). The text after the last ':' is a strength only when it is a number,
    so ``a.safetensors:0.8``, ``a.safetensors`` and ``/odd:folder/a.safetensors`` all read correctly."""
    path, sep, tail = spec.rpartition(":")
    if sep and path:
        try:
            s = float(tail)
        except ValueError:
            s = None
        if s is not None and math.isfinite(s):
            return path, s
    return spec, 1.0


def want_vae(ck: Checkpoint) -> str:
    return "wan2.2" if (ck.config or {}).get("vae_z_dim") == 48 else "wan2.1"


def needed(ck: Checkpoint, have_torch: Optional[bool] = None) -> list[str]:
    """Parts that must come from somewhere else: missing ones, and .pth ones when PyTorch is not installed."""
    torch_ok = torch_available() if have_torch is None else have_torch
    out = []
    for name in PARTS:
        p = ck.parts.get(name)
        if name in ck.missing or p is None or (p.layout.startswith("pth") and not torch_ok):
            out.append(name)
    return out


def candidate_dirs(roots: Iterable[Path], exclude: Iterable[Path] = ()) -> list[Path]:
    """Installed model folders under each root (one level down, plus ``sources/*``), and the roots' own model
    folders, without the excluded ones."""
    skip = set()
    for e in exclude:
        try:
            skip.add(Path(e).expanduser().resolve())
        except OSError:
            pass
    seen, out = set(), []

    def add(d: Path) -> None:
        try:
            r = d.resolve()
        except OSError:
            return
        if r in seen or r in skip or not d.is_dir():
            return
        if any(r == s or s in r.parents for s in skip):
            return
        seen.add(r)
        out.append(d)

    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        if (root / "shuttrr.json").exists() or (root / "text_encoder").is_dir() \
                or (root / "text_encoder_q8").is_dir() \
                or (root / "components").is_dir():
            add(root)
            continue
        for d in sorted(root.iterdir()):
            if d.name == "sources" and d.is_dir():
                for s in sorted(d.iterdir()):
                    add(s)
            else:
                add(d)
    return out


def _find(name: str, folder: Path) -> Optional[Part]:
    from shutter.families.wan import detect as D
    finder = {"text_encoder": D._find_t5, "vae": D._find_vae, "tokenizer": D._find_tokenizer}[name]
    try:
        return finder(folder)
    except (OSError, ValueError):
        return None


def find_parts(ck: Checkpoint, dirs: Iterable[Path], names: Optional[Iterable[str]] = None,
               have_torch: Optional[bool] = None) -> dict[str, Path]:
    """{part name: folder to pass to the converter} for each needed part found in `dirs` (first match wins).
    A VAE is used only when it is the kind this model needs; .pth parts only when PyTorch is installed."""
    torch_ok = torch_available() if have_torch is None else have_torch
    todo = list(names) if names is not None else needed(ck, torch_ok)
    vae_kind = want_vae(ck)
    found: dict[str, Path] = {}
    for d in dirs:
        for name in todo:
            if name in found:
                continue
            p = _find(name, d)
            if p is None or (p.layout.startswith("pth") and not torch_ok):
                continue
            if name == "vae" and p.layout.replace("pth:", "") != vae_kind:
                continue
            found[name] = d
        if len(found) == len(todo):
            break
    return found
