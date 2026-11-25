"""Reels add-on: the content-specific pieces of Shutter's first use (talking-avatar reels).

The engine core is a plain inference engine and knows nothing in this package. Adapters that support it
(today: `longcat-avatar`) read its options from `JobRequest.params`:

  params.framing      "full" | "small" | "auto": how large the subject is in the finished edit. Drives the
                      LongCat render rule: small (corner card, split screen, picture in picture) renders the
                      480p bucket full frame and upscales; full/auto renders the 704x1312 class.
  params.crop_box     generate only a full-height box of the frame and composite it on the static plate.
  params.room_lock    put the room (background) back at every denoise step; needs crop_box + person_box.
  params.upscale_to   "WxH" | "off": upscale to the edit size (`upscale.py`), with
  params.zoom_cancel  "auto" | float | "off": the model's slow camera push cancelled inside the same resample.

Older clients sent `framing` as a top-level request key; the core folds it into `params.framing` and never
emits it again.

Modules: `upscale` (ffmpeg / Lanczos upscale with zoom cancel), `planner` (shot placement of a clip's segment
chains over one or more Macs).
"""

from __future__ import annotations

import enum
from typing import Any

__all__ = ["Framing", "framing_of"]


class Framing(str, enum.Enum):
    """How large the subject is in the finished edit. Drives the LongCat render rule."""

    FULL = "full"    # subject fills the frame: 720p-class render
    SMALL = "small"  # corner card, split screen, picture in picture: 480p full frame + upscale
    AUTO = "auto"    # adapter decides (defaults to FULL)


def framing_of(request: Any) -> Framing:
    """The request's framing from `params.framing` (default auto). Raises ValueError for an unknown value."""
    v = (getattr(request, "params", None) or {}).get("framing")
    if v is None or v == "":
        return Framing.AUTO
    if isinstance(v, Framing):
        return v
    try:
        return Framing(str(v).strip().lower())
    except ValueError:
        raise ValueError(f"framing must be one of: full, small, auto (got {v!r})") from None
