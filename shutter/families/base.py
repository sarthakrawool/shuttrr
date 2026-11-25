"""Model families: what a family declares, and what the engine does with it.

A family (Wan, later Flux / Qwen-Image) is a set of checkpoints that share one architecture. Shuttrr supports
families rather than single models, the way Splash and Magnitude support "any GGUF in a supported family": any
checkpoint whose tensors match a family's shapes runs, including fine-tunes and other people's builds, and anything
that does not match is refused with a reason before a single byte is loaded.

A family only declares its parts (`Part`): which files hold which component, which tensors form the repeated block
stack and how big each block is. Everything to do with memory is engine code that works on parts, not on a model:

* `shutter.core.parts.estimate` turns parts + an activation figure into a peak per memory step (resident, staged,
  SSD streaming), so the five steps of the memory slider work for a new family without new code;
* `shutter.core.stream` streams any part's block stack from the SSD;
* `shutter.core.text_encode` runs a T5-style prompt encoder one block at a time.

`inspect(path)` (in `shutter.families`) asks each family to recognise a folder or file and returns a `Checkpoint`:
the family, the variant (e.g. "Wan 2.2 TI2V 5B"), its parts, and `problems`, the reasons it cannot run. A checkpoint
with no problems can be converted to Shuttrr's 8-bit low-memory build and run.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

GB = 1024**3


@dataclass
class Part:
    """One component of a checkpoint, as found on disk."""

    name: str                         # "dit", "dit_high", "dit_low", "text_encoder", "vae", "tokenizer", "lora"
    role: str                         # "denoiser" | "text_encoder" | "decoder" | "tokenizer" | "lora"
    files: list[str] = field(default_factory=list)
    layout: str = ""                  # key naming: "official", "comfy", "diffusers", "hf-t5", "shuttrr", "pth"
    total_bytes: int = 0              # bytes as stored
    n_blocks: int = 0                 # repeated blocks (0: not a block stack)
    block_bytes: int = 0              # largest block, as stored
    rest_bytes: int = 0               # everything outside the blocks
    dtype: str = ""                   # dominant stored dtype ("BF16", "F16", "F32", "U32" for packed 8-bit, ...)
    quant: Optional[dict[str, Any]] = None   # {"bits": 8, "group_size": 64} for a Shuttrr build

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Checkpoint:
    """What `inspect` found. `problems` empty means it can run (after conversion when `needs_conversion`)."""

    path: str
    family: str = ""
    variant: str = ""                 # id, e.g. "wan2.2-ti2v-5b"
    label: str = ""                   # for people, e.g. "Wan 2.2 TI2V 5B"
    task: str = ""                    # "t2v" | "i2v" | "ti2v"
    config: dict[str, Any] = field(default_factory=dict)
    parts: dict[str, Part] = field(default_factory=dict)
    needs_conversion: bool = True     # False for a Shuttrr build
    problems: list[str] = field(default_factory=list)   # why it cannot run, in plain words
    notes: list[str] = field(default_factory=list)      # things worth knowing that do not block it
    missing: list[str] = field(default_factory=list)    # parts it needs from elsewhere (e.g. a bare fine-tune)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d


class Family(abc.ABC):
    """A supported architecture."""

    #: id used in config and the registry, e.g. "wan"
    name: str = ""
    #: for people
    label: str = ""
    #: adapter that runs checkpoints of this family (registry family id)
    adapter: str = ""

    @abc.abstractmethod
    def recognise(self, path: Path) -> Optional[Checkpoint]:
        """A Checkpoint if `path` (folder or file) looks like this family, even one it cannot run (then with
        `problems`); None if it is not this family at all. Must read headers only, never tensor data."""

    def check_memory(self, ckpt: Checkpoint, total_bytes: int, headroom: int) -> list[str]:
        """Reasons this Mac cannot run `ckpt` even at the lowest memory step (empty: it can)."""
        return []
