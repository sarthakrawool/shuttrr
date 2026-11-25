"""Shot placement over one or more Macs (shot-parallel).

The unit a node accepts is a *chain*: the shots that continue from each other (for LongCat, the 93-frame segments
of one clip, each conditioned on the previous segment's last 13 frames). A chain runs in order on one node; different
chains are independent and can run on different nodes at the same time. A job is deterministic given its shots
(each shot carries its seed and params, see `types.Shot`), so a chain run on another Mac yields the same frames as
a local run of the same model build, and the outputs are simply concatenated in `join_order`.

How this maps onto exo later: an exo bridge advertises each Mac as a `Node` (memory, measured speed, which model
builds it holds); `plan_placement` decides which node gets which chains; each node runs its chains through its own
Shutter engine (same adapter, same memory guard) and returns files; the coordinator concatenates them. Expert- or
token-parallel execution of a single forward pass (splitting one DiT step across Macs) is a different layer: it
would live below the adapter, inside the model code, and is not what this planner does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from shutter.types import Shot

__all__ = ["Node", "Assignment", "plan_placement", "join_order", "NoNodeFits"]


class NoNodeFits(Exception):
    """No node has the model and enough memory for a chain."""


@dataclass(frozen=True)
class Node:
    id: str
    host: str = "local"
    memory_bytes: int = 0          # memory available for a job; 0 = unknown / no limit
    speed: float = 1.0             # relative throughput (2.0 = finishes a shot in half the time)
    models: tuple[str, ...] = ()   # model ids it can run; empty = any


@dataclass
class Assignment:
    node_id: str
    chain: int
    shots: list[Shot] = field(default_factory=list)
    est_seconds: float = 0.0       # on that node, speed applied
    start_s: float = 0.0           # expected start offset on that node (queue position)

    @property
    def finish_s(self) -> float:
        return self.start_s + self.est_seconds


def _shot_cost(shot: Shot, seconds_per_shot: Optional[float]) -> float:
    if "est_seconds" in shot.params:
        return float(shot.params["est_seconds"])
    if seconds_per_shot is not None:
        return float(seconds_per_shot)
    return max(shot.duration_s, 1e-3)   # proportional to content length when nothing better is known


def _fits(node: Node, model: Optional[str], need: int) -> bool:
    if model and node.models and model not in node.models:
        return False
    return not (need and node.memory_bytes and node.memory_bytes < need)


def plan_placement(shots: Sequence[Shot], nodes: Sequence[Node], seconds_per_shot: Optional[float] = None,
                   *, model: Optional[str] = None, memory_bytes: int = 0) -> list[Assignment]:
    """Group shots by chain (kept in index order), then place chains greedily, longest first, on the eligible node
    with the earliest expected finish time (queue end + chain cost / speed). Ties go to the faster node, then to
    node order. `model` / `memory_bytes` filter nodes; a shot's `params["memory_bytes"]` raises the requirement for
    its chain. Returns assignments ordered by node, then by start time."""
    if not nodes:
        raise NoNodeFits("no nodes")
    chains: dict[int, list[Shot]] = {}
    for s in shots:
        chains.setdefault(s.chain, []).append(s)
    for c in chains.values():
        c.sort(key=lambda s: s.index)
    work = []
    for cid, cs in chains.items():
        cost = sum(_shot_cost(s, seconds_per_shot) for s in cs)
        need = max([memory_bytes] + [int(s.params.get("memory_bytes", 0)) for s in cs])
        work.append((cost, cid, cs, need))
    work.sort(key=lambda w: (-w[0], w[1]))

    queue_end = {n.id: 0.0 for n in nodes}
    order = {n.id: i for i, n in enumerate(nodes)}
    out: list[Assignment] = []
    for cost, cid, cs, need in work:
        ok = [n for n in nodes if _fits(n, model, need)]
        if not ok:
            raise NoNodeFits(f"chain {cid}: no node has model {model!r} and {need} bytes free")
        best = min(ok, key=lambda n: (queue_end[n.id] + cost / max(n.speed, 1e-9), -n.speed, order[n.id]))
        est = cost / max(best.speed, 1e-9)
        out.append(Assignment(node_id=best.id, chain=cid, shots=list(cs), est_seconds=est,
                              start_s=queue_end[best.id]))
        queue_end[best.id] += est
    out.sort(key=lambda a: (order[a.node_id], a.start_s))
    return out


def join_order(assignments: Sequence[Assignment]) -> list[tuple[str, int, int]]:
    """Order in which to concatenate outputs: (node_id, chain, shot index) sorted by the shots' place in the
    edit (start_s, then chain, then index), independent of which node ran them."""
    items = [(s.start_s, a.chain, s.index, a.node_id) for a in assignments for s in a.shots]
    items.sort()
    return [(node, chain, idx) for _, chain, idx, node in items]
