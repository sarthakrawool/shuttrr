"""Time left for a running request, from what this run has measured so far (Shuttrr 0.2.11).

The rule is: never show an invented number. Before the first denoising step finishes there is nothing to measure,
so the estimate is None (the app shows "Estimating…"). After that:

    time left = steps still to run x seconds per step (measured on this run) + the decode still to come

Seconds per step: after two or more finished steps it is the step-to-step rate, which leaves out the first step's
model load; after only one it is that first step's time (a little high when the model loaded during it, and it
corrects itself at the next step). The decode comes from earlier finished runs of the same model, size, frames and
decode (their `timings.decode_s`), or from the catalog's measured run for this request; when neither exists the
estimate covers the steps only and says so (`decode_known` False). During the decode the time left is the decode
estimate minus the time spent decoding so far, never below zero.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from shutter.core.stats import parse_step


class EtaTracker:
    def __init__(self, decode_estimate: Optional[float] = None, clock: Optional[Callable[[], float]] = None):
        import time
        self.clock = clock or time.monotonic
        self.decode_estimate = decode_estimate
        self.reset()

    def reset(self) -> None:
        self.last_plain_t: Optional[float] = None     # last report without a step counter (denoise starts after it)
        self.steps: list[tuple[int, float]] = []      # (step done, when)
        self.total = 0
        self.decode_started: Optional[float] = None
        self.base = 0                                 # steps from earlier segments (counters that restart)

    def update(self, stage: str, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        st = parse_step(stage)
        if st is not None:
            step, steps = st
            if self.steps and step + self.base < self.steps[-1][0]:
                self.base = self.steps[-1][0]
            n = self.base + step
            if step >= 1 and (not self.steps or n > self.steps[-1][0]):
                self.steps.append((n, now))
            self.total = self.base + steps
            self.decode_started = None
            return
        if "decod" in (stage or "").lower():
            if self.decode_started is None:
                self.decode_started = now
            return
        if not self.steps:
            self.last_plain_t = now

    def seconds_per_step(self) -> Optional[float]:
        if len(self.steps) >= 2:
            (n0, t0), (n1, t1) = self.steps[0], self.steps[-1]
            return (t1 - t0) / max(1, n1 - n0)
        if len(self.steps) == 1 and self.last_plain_t is not None:
            n, t = self.steps[0]
            return (t - self.last_plain_t) / max(1, n)
        return None

    def estimate(self, now: Optional[float] = None) -> dict[str, Any]:
        """{"eta_s": seconds left or None, "decode_known": bool}."""
        now = self.clock() if now is None else now
        dec = self.decode_estimate
        if self.decode_started is not None:
            if dec is None:
                return {"eta_s": None, "decode_known": False}
            return {"eta_s": round(max(0.0, dec - (now - self.decode_started)), 1), "decode_known": True}
        sps = self.seconds_per_step()
        if sps is None or not self.steps:
            return {"eta_s": None, "decode_known": dec is not None}
        done_n, done_t = self.steps[-1]
        left_steps = max(0, self.total - done_n)
        # the step in progress has already been running since the last finished one
        in_step = min(now - done_t, sps) if left_steps else 0.0
        eta = left_steps * sps - in_step + (dec or 0.0)
        return {"eta_s": round(max(0.0, eta), 1), "decode_known": dec is not None}


def decode_estimate_from_history(jobs: list[Any], model: str, request: Any) -> Optional[float]:
    """Median decode seconds of up to 3 recent finished runs of the same model with the same size, frames and decode
    (tiled or not); None when there are none."""
    def key(r: Any) -> tuple:
        p = getattr(r, "params", None) or {}
        return (getattr(r, "width", None), getattr(r, "height", None), getattr(r, "frames", None),
                bool(p.get("vae_tile")) if isinstance(p, dict) else False)
    want = key(request)
    found: list[float] = []
    for j in jobs:
        if getattr(j, "request", None) is None or j.request.model != model or j.request.params.get("_action"):
            continue
        if getattr(j.state, "value", j.state) != "done" or key(j.request) != want:
            continue
        info = (j.result or {}).get("info") or {}
        d = (info.get("timings") or {}).get("decode_s")
        if isinstance(d, (int, float)) and d > 0:
            found.append(float(d))
        if len(found) >= 3:
            break
    if not found:
        return None
    found.sort()
    return found[len(found) // 2]
