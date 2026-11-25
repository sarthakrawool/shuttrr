"""The Wan family (Wan 2.1 / 2.2: 1.3B, 5B, 14B, A14B, their fine-tunes and LoRAs)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from shutter.core import parts as P
from shutter.families.base import Checkpoint, Family
from shutter.families.wan import detect
from shutter.families.wan.spec import part_sizes, stage_needs

GB = 1024**3
# smallest request we consider when deciding whether a Mac can run a model at all
MIN_SIZE = (320, 512, 17)


class WanFamily(Family):
    name = "wan"
    label = "Wan video"
    adapter = "wan"

    def recognise(self, path: Path) -> Optional[Checkpoint]:
        return detect.recognise(path)

    def check_memory(self, ckpt: Checkpoint, total_bytes: int, headroom: int = 4 * GB) -> list[str]:
        if not ckpt.config or not total_bytes:
            return []
        h, w, f = MIN_SIZE
        ps = part_sizes(ckpt.config, image_encoder=ckpt.task in ("i2v", "ti2v"))
        est = P.estimate(ps, stage_needs(ckpt.config, h, w, f, tiled_decode=True), "stream_tight")
        budget = total_bytes - headroom
        if est["peak"] > budget:
            return [f"Even at Lowest memory this model needs about {est['peak'] / GB:.0f} GB "
                    f"({est['worst']} stage), and this Mac has {total_bytes / GB:.0f} GB."]
        return []


WAN = WanFamily()
