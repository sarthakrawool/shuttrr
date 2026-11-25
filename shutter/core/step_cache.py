"""TeaCache-style step caching (Liu et al., "Timestep Embedding Tells: It's Time to Cache for Video Diffusion
Model", arXiv 2411.19108). Opt-in, NOT bit-identical. Model-agnostic: the caller supplies the indicator (the
timestep-modulated input of the first block) and the input/output of the block stack; arrays only need `-`, `+`,
`abs()`, `.mean()`/`astype` (MLX arrays and NumPy arrays both work).

Rule, per call "slot" (a slot is one DiT call position inside a step: with guidance a step makes several calls,
each with its own inputs, so each gets its own cache):

  rel_l1   = mean |ind_t - ind_{t-1}| / mean |ind_{t-1}|        (always updated, also on skipped steps)
  acc     += rescale(rel_l1)                                     (rescale = polynomial fitted per model; identity
                                                                  when no coefficients are given)
  skip when acc < threshold, a previous residual of the same shape exists, the step is not in the warm-up, not the
  last step (when `keep_last`), the cache was not invalidated, and fewer than `max_consecutive` steps were skipped
  in a row; otherwise compute and reset acc to 0.
  computed step: residual = stack_out - stack_in;  skipped step: stack_out = stack_in + residual.

threshold = 0 never skips (acc < 0 is never true), so the output is the uncached output exactly.

How much there is to skip depends on the schedule: TeaCache's published speed-ups are on 30-50 step samplers
(Open-Sora, Latte, Open-Sora-Plan; later ports to Wan/HunyuanVideo/FLUX). A distilled few-step schedule (LongCat
Avatar 1.6: 8 steps, large sigma jumps between steps) changes the modulated input a lot every step, so at sane
thresholds little or nothing is skipped; it mostly helps long-schedule image models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

__all__ = ["TeaCache", "SlotState", "rel_l1"]


def _f(x) -> float:
    """Python float of a 0-d array (MLX: forces evaluation; NumPy: plain)."""
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def rel_l1(cur, prev) -> float:
    """mean |cur - prev| / mean |prev|, computed in fp32 when the arrays support astype."""
    a, b = cur, prev
    if hasattr(a, "astype") and hasattr(a, "dtype"):
        try:
            import mlx.core as mx  # noqa: F401

            if type(a).__module__.startswith("mlx"):
                a, b = a.astype(mx.float32), b.astype(mx.float32)
        except Exception:
            pass
    den = _f(abs(b).mean())
    num = _f(abs(a - b).mean())
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


@dataclass
class SlotState:
    prev_indicator: Any = None
    residual: Any = None
    shape: Optional[tuple] = None
    acc: float = 0.0
    consecutive: int = 0
    computed: int = 0
    skipped: int = 0


@dataclass
class TeaCache:
    threshold: float
    coefficients: Optional[Sequence[float]] = None   # polynomial, highest power first (numpy.polyval order)
    warmup: int = 1                                  # first N steps of a run always computed
    keep_last: bool = True                           # last step always computed (needs num_steps)
    max_consecutive: Optional[int] = None            # cap on skips in a row (None = no cap)
    num_steps: Optional[int] = None
    slots: dict = field(default_factory=dict)
    step: int = -1
    _slot_counter: int = 0
    _calls_this_step: int = 0
    events: list = field(default_factory=list)       # (run, step, slot, "compute"|"skip", rel_l1, acc)
    runs: int = 0
    steps_without_calls: int = 0

    def __post_init__(self):
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")

    # ---- step bookkeeping: new_step(i) at the top of every denoise step; a step index that does not increase
    # (a new segment / sample) starts a new run with empty caches
    def new_run(self, num_steps: Optional[int] = None) -> None:
        self.slots = {}
        self.runs += 1
        if num_steps is not None:
            self.num_steps = num_steps

    def new_step(self, i: int) -> None:
        if self.step >= 0 and self._calls_this_step == 0:
            self.steps_without_calls += 1
        if self.step < 0 or i <= self.step:
            self.new_run()
        self.step = i
        self._slot_counter = 0
        self._calls_this_step = 0

    def finish(self) -> None:
        """Close the run (counts a last step that made no cached call, e.g. the KV-cache path)."""
        if self.step >= 0 and self._calls_this_step == 0:
            self.steps_without_calls += 1
        self.step = -1
        self._calls_this_step = 0

    def invalidate(self) -> None:
        """Force the next call of every slot to compute (e.g. the model's weights / LoRA changed)."""
        for st in self.slots.values():
            st.residual = None

    def _rescale(self, x: float) -> float:
        if not self.coefficients:
            return x
        y = 0.0
        for c in self.coefficients:
            y = y * x + float(c)
        return y

    # ---- per call
    def begin(self, indicator, shape: Optional[tuple] = None) -> tuple[int, bool]:
        """Register one call; returns (slot, skip). Always updates the slot's indicator and accumulator."""
        slot = self._slot_counter
        self._slot_counter += 1
        self._calls_this_step += 1
        st = self.slots.setdefault(slot, SlotState())
        r = None
        if st.prev_indicator is not None and (shape is None or st.shape == tuple(shape)):
            r = rel_l1(indicator, st.prev_indicator)
            st.acc += self._rescale(r)
        st.prev_indicator = indicator
        step = max(self.step, 0)
        last = self.keep_last and self.num_steps is not None and step >= self.num_steps - 1
        can = (r is not None and st.residual is not None and step >= self.warmup and not last
               and (self.max_consecutive is None or st.consecutive < self.max_consecutive)
               and (shape is None or st.shape == tuple(shape)))
        skip = bool(can and st.acc < self.threshold)
        if skip:
            st.skipped += 1
            st.consecutive += 1
        else:
            st.computed += 1
            st.consecutive = 0
            st.acc = 0.0
        self.events.append((self.runs, step, slot, "skip" if skip else "compute", r, st.acc))
        return slot, skip

    def store(self, slot: int, x_in, x_out, residual_fn: Optional[Callable] = None) -> None:
        """After a computed call: keep residual = x_out - x_in (residual_fn overrides the subtraction)."""
        st = self.slots[slot]
        st.residual = residual_fn(x_in, x_out) if residual_fn else x_out - x_in
        st.shape = tuple(getattr(x_in, "shape", ()) or ())

    def apply(self, slot: int, x_in, apply_fn: Optional[Callable] = None):
        """For a skipped call: x_in + residual (apply_fn overrides the addition)."""
        res = self.slots[slot].residual
        return apply_fn(x_in, res) if apply_fn else x_in + res

    def run(self, x_in, indicator, fn: Callable, shape: Optional[tuple] = None):
        """Convenience: one cached call of the block stack `fn(x_in)`."""
        slot, skip = self.begin(indicator, tuple(getattr(x_in, "shape", ())) if shape is None else shape)
        if skip:
            return self.apply(slot, x_in)
        out = fn(x_in)
        self.store(slot, x_in, out)
        return out

    # ---- report
    def stats(self) -> dict[str, Any]:
        computed = sum(1 for e in self.events if e[3] == "compute")
        skipped = sum(1 for e in self.events if e[3] == "skip")
        return {"threshold": self.threshold, "coefficients": list(self.coefficients) if self.coefficients else None,
                "warmup": self.warmup, "keep_last": self.keep_last, "max_consecutive": self.max_consecutive,
                "calls": computed + skipped, "computed": computed, "skipped": skipped,
                "skipped_steps": [[e[0], e[1], e[2]] for e in self.events if e[3] == "skip"],
                "rel_l1": [None if e[4] is None else round(e[4], 5) for e in self.events],
                "steps_without_cached_calls": self.steps_without_calls}
