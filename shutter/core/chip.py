"""This Mac's chip: its name, its tier (base M-series, Pro, Max or Ultra) and its GPU core count.

Some catalog defaults depend on the chip as well as the memory: a base M-series chip with lots of memory can hold a
big job but runs it slowly, so it gets a shorter default clip than a Pro / Max / Ultra (catalog.default_request).
The GPU core count also scales measured times to this Mac (catalog.expected_time).
"""

from __future__ import annotations

import functools
import re
import subprocess
from typing import Any, Optional

TIERS = ("base", "pro", "max", "ultra")


def tier_of(name: str) -> str:
    """"Apple M5" -> "base", "Apple M4 Pro" -> "pro", "Apple M3 Max" -> "max", "Apple M2 Ultra" -> "ultra".
    Anything that is not an Apple M-series name -> "" (unknown)."""
    m = re.match(r"^\s*Apple\s+M\d+(?:\s+(Pro|Max|Ultra))?\s*$", name or "", re.IGNORECASE)
    if not m:
        return ""
    return (m.group(1) or "base").lower()


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _gpu_cores() -> Optional[int]:
    m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', _run(["ioreg", "-rd1", "-c", "IOAccelerator"]))
    if m:
        return int(m.group(1))
    m = re.search(r"Total Number of Cores:\s*(\d+)", _run(["system_profiler", "SPDisplaysDataType"]))
    return int(m.group(1)) if m else None


@functools.lru_cache(maxsize=1)
def chip_info() -> dict[str, Any]:
    """{"name": "Apple M5", "tier": "base", "gpu_cores": 10}; tier "" and gpu_cores None when unknown."""
    name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    return {"name": name, "tier": tier_of(name), "gpu_cores": _gpu_cores()}
