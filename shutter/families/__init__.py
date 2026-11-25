"""Supported model families (see base.py). `inspect(path)` recognises a checkpoint in any supported family."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

from shutter.families.base import Checkpoint, Family, Part

__all__ = ["FAMILIES", "inspect", "family", "Checkpoint", "Part"]

#: family id -> "module:attribute" of its Family instance (imported lazily)
FAMILIES: dict[str, str] = {
    "wan": "shutter.families.wan.family:WAN",
}


def family(name: str) -> Family:
    mod, _, attr = FAMILIES[name].partition(":")
    return getattr(importlib.import_module(mod), attr)


def inspect(path) -> Optional[Checkpoint]:
    """The first family that recognises `path` (a folder or a file), or None."""
    for name in FAMILIES:
        ck = family(name).recognise(Path(path).expanduser())
        if ck is not None:
            return ck
    return None
