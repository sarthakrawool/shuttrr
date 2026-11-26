"""Cancel points inside long GPU work (Shuttrr 0.2.11).

The pipeline sets `CHECK` to the running request's check_cancelled() for the length of a run. The decode calls
`poll()` after every latent frame or tile, and the DiT, when a check is set, evaluates its hidden state every few
blocks and calls `poll()`, so Cancel stops within a fraction of a step instead of waiting for the whole step or the
whole decode. Evaluating at those points runs the same operations in the same order: the output is identical.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

_local = threading.local()

# DiT: check every this many blocks (30 blocks in Wan 2.1 1.3B / 2.2 5B: three checks per step)
BLOCK_EVERY = 10


def set_check(fn: Optional[Callable[[], None]]) -> None:
    _local.check = fn


def get_check() -> Optional[Callable[[], None]]:
    return getattr(_local, "check", None)


def poll() -> None:
    fn = get_check()
    if fn is not None:
        fn()
