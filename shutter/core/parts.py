"""Memory per step, worked out from a family's parts (family-neutral).

A run goes through stages: encode the prompt, (encode an image), denoise, decode. Each stage needs some parts in
memory plus that stage's working memory (activations). What the five memory steps change is which parts stay in
memory and how much of the denoiser's block stack is resident:

    resident      every denoiser part stays in memory between requests (warm); other parts load for their stage
    split         as resident, but a model with two denoisers (Wan 2.2 A14B high/low noise) holds one at a time
    staged        as split, and the denoiser is freed before the decode; nothing is kept warm
    stream_wide   the denoiser's blocks are read from the SSD, `wide` blocks in memory at a time
    stream_tight  the same with `tight` blocks

The prompt encoder always runs one block at a time (shutter.core.text_encode), so it never costs more than its
token embedding plus one block.

Numbers here are formulas, used until a measured peak exists (shutter.core.modes.measured_for wins over them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

GB = 1024**3
#: Python, MLX runtime, small buffers
BASE_BYTES = int(1.2 * GB)

STREAM_WINDOWS = {"stream_wide": 8, "stream_tight": 2}


@dataclass
class PartSize:
    name: str
    role: str                 # "denoiser" | "text_encoder" | "decoder" | "image_encoder"
    total: int                # bytes when fully loaded (in the dtype it runs in)
    n_blocks: int = 0
    block: int = 0            # one block, loaded
    embed: int = 0            # text encoder: token embedding table (the rest of `total` is blocks)

    @property
    def rest(self) -> int:
        return max(0, self.total - self.n_blocks * self.block)


@dataclass
class StageNeeds:
    """Working memory per stage (bytes), from the family's activation formulas."""

    encode: int = 0
    image: int = 0
    denoise: int = 0
    decode: int = 0
    carried: int = 0          # tensors that live across stages (latents, prompt embeddings)
    extras: dict[str, int] = field(default_factory=dict)


def _by_role(parts: Iterable[PartSize], role: str) -> list[PartSize]:
    return [p for p in parts if p.role == role]


def denoiser_bytes(dens: list[PartSize], rung: str, window: Optional[int] = None) -> int:
    """Memory the denoiser part(s) hold during the denoise stage at `rung`."""
    if not dens:
        return 0
    if rung in STREAM_WINDOWS or window:
        w = window or STREAM_WINDOWS[rung]
        # only one denoiser streams at a time; each keeps its non-block tensors resident while in use
        return max(p.rest + min(w, p.n_blocks) * p.block for p in dens)
    if rung == "resident" or len(dens) == 1:
        return sum(p.total for p in dens)
    return max(p.total for p in dens)      # split / staged: one expert at a time


def stage_peaks(parts: list[PartSize], needs: StageNeeds, rung: str, window: Optional[int] = None) -> dict[str, int]:
    """Estimated process peak per stage at `rung`."""
    dens = _by_role(parts, "denoiser")
    te = _by_role(parts, "text_encoder")
    dec = _by_role(parts, "decoder")
    img = _by_role(parts, "image_encoder")
    warm = sum(p.total for p in dens) if rung == "resident" else 0   # kept between requests
    d_now = denoiser_bytes(dens, rung, window)
    te_now = max((p.embed + p.block for p in te), default=0)
    out = {
        "encode": BASE_BYTES + warm + te_now + needs.encode,
        "denoise": BASE_BYTES + d_now + needs.denoise + needs.carried,
    }
    if img:
        out["image"] = BASE_BYTES + warm + sum(p.total for p in img) + needs.image + needs.carried
    # resident and split keep the denoiser while decoding (the next request reuses it); staged/stream free it
    keep_for_decode = d_now if rung in ("resident", "split") else 0
    out["decode"] = BASE_BYTES + keep_for_decode + sum(p.total for p in dec) + needs.decode + needs.carried
    return out


def estimate(parts: list[PartSize], needs: StageNeeds, rung: str, window: Optional[int] = None) -> dict[str, Any]:
    """{"peak": bytes, "stages": {stage: bytes}, "worst": stage, "steady": bytes kept warm after the request}."""
    st = stage_peaks(parts, needs, rung, window)
    worst = max(st, key=st.get)
    steady = sum(p.total for p in _by_role(parts, "denoiser")) if rung == "resident" else 0
    if rung == "split":
        steady = denoiser_bytes(_by_role(parts, "denoiser"), rung)
    return {"peak": st[worst], "stages": st, "worst": worst, "steady": steady}
