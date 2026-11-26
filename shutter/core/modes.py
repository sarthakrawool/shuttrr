"""Fast and Low-memory modes (docs/MODES.md).

A job runs in one of two modes:

* ``fast``: lossless speed-ups, weights resident and warm. Before a job the planner takes
  ``budget = available`` (what macOS can give now; no reserve since 0.2.3) and, if the fast plan's estimated peak does not fit, steps
  down the ladder below just far enough to fit (``fit_to_budget``). It never starts a job whose
  estimate exceeds the budget: when even the last rung does not fit, the job waits.
* ``low_memory`` with an efficiency slider 0..1: the slider picks the highest rung whose ``at``
  is <= the value (``rung_for``).

Every rung is bit-identical to the reference output unless its ``identical`` field says
otherwise (``True`` = identical to the production compiled path; ``"eager"`` would mean identical
only to the uncompiled path). Adapters map a rung id to their own switches and report a
``MemoryEstimate`` per rung; measured numbers (written by scripts/mac_test_chain.sh) live in
``<shutter home>/measured.json`` and are preferred over formulas:

    {model_id: {size: {rung_id: {"peak_bytes": int, "seconds": float, "identical": bool}}}}

``size`` is the generated size as ``"HxW"`` (the LongCat tree's ``--size`` convention).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

__all__ = ["LADDER", "RUNG_IDS", "MODES", "ModeChoice", "rung_for", "rung", "resolve_mode", "fit_to_budget",
           "ladder_with_measurements", "measured_path", "load_measured", "save_measured",
           "record_measurement", "measured_for", "size_key"]

MODES = ("fast", "low_memory")

LADDER: list[dict[str, Any]] = [
    {"at": 0.00, "id": "resident", "label": "Full speed",
     "detail": "Same as Fast: the whole model stays in memory.", "identical": True},
    {"at": 0.25, "id": "split", "label": "Slightly lower memory",
     "detail": "Each model block runs in 3 smaller pieces: about 2 GB less memory, about 2% slower.", "identical": True},
    {"at": 0.50, "id": "staged", "label": "Lower memory",
     "detail": "Also works through the frames in 4 chunks, and frees the main model before the final decode.", "identical": True},
    {"at": 0.75, "id": "stream_wide", "label": "Very low memory",
     "detail": "Reads the model from the SSD as it goes, keeping 8 blocks in memory at a time.", "identical": True},
    {"at": 1.00, "id": "stream_tight", "label": "Lowest memory",
     "detail": "Reads the model from the SSD as it goes, keeping only 2 blocks in memory: lowest memory, slowest.", "identical": True},
]
RUNG_IDS = [r["id"] for r in LADDER]
_BY_ID = {r["id"]: r for r in LADDER}


def rung(rung_id: str) -> dict[str, Any]:
    """The ladder entry for an id (a copy). Raises ValueError for an unknown id."""
    if rung_id not in _BY_ID:
        raise ValueError(f"unknown rung {rung_id!r} (known: {', '.join(RUNG_IDS)})")
    return dict(_BY_ID[rung_id])


def _clamp01(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"efficiency must be a number between 0 and 1, got {v!r}") from None
    if f != f:  # NaN
        raise ValueError("efficiency must be a number between 0 and 1, got NaN")
    return min(1.0, max(0.0, f))


def rung_for(efficiency: Any) -> dict[str, Any]:
    """The highest rung whose ``at`` is <= efficiency (clamped to 0..1)."""
    e = _clamp01(efficiency)
    best = LADDER[0]
    for r in LADDER:
        if r["at"] <= e + 1e-9:
            best = r
    return dict(best)


@dataclass
class ModeChoice:
    mode: str                  # "fast" | "low_memory"
    efficiency: float          # the slider value in force (0 in fast mode unless given)
    rung: str                  # the starting rung: "resident" in fast mode, rung_for(efficiency) otherwise
    explicit_rung: bool = False  # params["rung"] forced the rung (test chain, power users)
    step_down: bool = True     # fast mode may step further down to fit the budget

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_mode(params: Optional[Mapping[str, Any]], defaults: Optional[Mapping[str, Any]] = None) -> ModeChoice:
    """The mode for a job: ``params`` (``mode``, ``efficiency``, optional ``rung``) win over ``defaults``
    (``mode``/``default_mode``, ``efficiency``/``default_efficiency``; engine-wide). Missing everything = fast.

    * fast: start at ``resident``; may step down to fit (``step_down=True``).
    * low_memory: ``rung_for(efficiency)``; the slider is a floor on memory saving, so the planner may still
      step further down if the chosen rung does not fit, but never up.
    * ``params["rung"]`` (an id) pins the rung exactly (no stepping): used by the test chain.
    """
    p = dict(params or {})
    d = dict(defaults or {})
    mode = p.get("mode") or d.get("mode") or d.get("default_mode") or "fast"
    mode = str(mode).strip().lower().replace("-", "_")
    if mode in ("low", "lowmem", "low_mem", "memory"):
        mode = "low_memory"
    if mode not in MODES:
        raise ValueError(f"mode must be 'fast' or 'low_memory', got {mode!r}")
    eff_raw = p.get("efficiency", d.get("efficiency", d.get("default_efficiency", 0.0 if mode == "fast" else 0.5)))
    eff = _clamp01(0.0 if eff_raw is None else eff_raw)
    if p.get("rung"):
        rid = str(p["rung"])
        rung(rid)  # validate
        return ModeChoice(mode=mode, efficiency=eff, rung=rid, explicit_rung=True, step_down=False)
    if mode == "fast":
        return ModeChoice(mode="fast", efficiency=eff, rung="resident")
    return ModeChoice(mode="low_memory", efficiency=eff, rung=rung_for(eff)["id"])


def _peak_of(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, Mapping):
        x = v.get("peak_bytes", v.get("peak"))
        return None if x is None else int(x)
    x = getattr(v, "peak", None)
    return None if x is None else int(x)


def fit_to_budget(estimates_by_rung: Mapping[str, Any], budget_bytes: Optional[int], start_rung: str = "resident") -> str:
    """The first rung, from ``start_rung`` down the ladder (towards less memory), whose estimated peak is
    <= ``budget_bytes``. Estimates may be ints, MemoryEstimate-like objects (``.peak``) or dicts
    (``peak_bytes``/``peak``); rungs without an estimate are skipped. ``budget_bytes=None`` = no limit
    (returns ``start_rung``). When nothing fits, returns the last rung that has an estimate (the lowest
    memory one): the caller's guard then refuses to start it, so a job is never started over budget."""
    rung(start_rung)
    start = RUNG_IDS.index(start_rung)
    if budget_bytes is None:
        return start_rung
    last = None
    for rid in RUNG_IDS[start:]:
        peak = _peak_of(estimates_by_rung.get(rid))
        if peak is None:
            continue
        last = rid
        if peak <= int(budget_bytes):
            return rid
    return last or start_rung


# ----------------------------------------------------------------------------- measured.json


def measured_path(home: Optional[Union[str, os.PathLike]] = None) -> Path:
    """``<home>/measured.json``; home = argument, else $SHUTTER_HOME, else ~/.shutter."""
    if home is None:
        home = os.environ.get("SHUTTER_HOME") or "~/.shutter"
    return Path(os.path.expanduser(str(home))) / "measured.json"


def size_key(height: int, width: int) -> str:
    return f"{int(height)}x{int(width)}"


def load_measured(path: Optional[Union[str, os.PathLike]] = None) -> dict[str, Any]:
    """The measured table, ``{}`` when the file is absent or unreadable (estimates then use formulas)."""
    p = Path(path) if path is not None else measured_path()
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_measured(data: Mapping[str, Any], path: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Write atomically (temp file + rename) so a reader never sees half a file."""
    p = Path(path) if path is not None else measured_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".measured.", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def record_measurement(model_id: str, size: str, rung_id: str, *, peak_bytes: int, seconds: float,
                       identical: Optional[bool] = None, path: Optional[Union[str, os.PathLike]] = None,
                       extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Merge one measurement into measured.json and return the entry written."""
    rung(rung_id)
    data = load_measured(path)
    entry: dict[str, Any] = {"peak_bytes": int(peak_bytes), "seconds": round(float(seconds), 1),
                             "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if identical is not None:
        entry["identical"] = bool(identical)
    if extra:
        entry.update(dict(extra))
    data.setdefault(str(model_id), {}).setdefault(str(size), {})[rung_id] = entry
    save_measured(data, path)
    return entry


def measured_for(model_id: str, size: str, rung_id: str, data: Optional[Mapping[str, Any]] = None
                 ) -> Optional[dict[str, Any]]:
    data = load_measured() if data is None else data
    try:
        e = data[str(model_id)][str(size)][rung_id]
    except (KeyError, TypeError):
        return None
    return dict(e) if isinstance(e, Mapping) and "peak_bytes" in e else None


def ladder_with_measurements(model_id: Optional[str] = None, size: Optional[str] = None,
                             data: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    """LADDER entries plus ``peak_bytes``/``seconds`` where measured (the /v1/settings shape)."""
    data = load_measured() if data is None else data
    out = []
    for r in LADDER:
        e = dict(r)
        if model_id and size:
            m = measured_for(model_id, size, r["id"], data)
            if m:
                e["peak_bytes"] = m["peak_bytes"]
                if "seconds" in m:
                    e["seconds"] = m["seconds"]
        out.append(e)
    return out
