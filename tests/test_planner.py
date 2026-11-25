from __future__ import annotations

import pytest

from shutter.addons.reels.planner import Assignment, Node, NoNodeFits, join_order, plan_placement
from shutter.types import GB, Shot


def shot(i, chain, start=None, dur=4.0, **params):
    return Shot(index=i, start_s=start if start is not None else i * dur, duration_s=dur, seed=i, prompt="p",
                chain=chain, params=params)


def test_chain_stays_together_and_in_order():
    shots = [shot(2, 0), shot(0, 0), shot(1, 0), shot(0, 1, start=100), shot(1, 1, start=104)]
    out = plan_placement(shots, [Node("a"), Node("b")], seconds_per_shot=10)
    by_chain = {a.chain: a for a in out}
    assert [s.index for s in by_chain[0].shots] == [0, 1, 2]
    assert [s.index for s in by_chain[1].shots] == [0, 1]
    assert by_chain[0].node_id != by_chain[1].node_id
    assert by_chain[0].est_seconds == 30 and by_chain[1].est_seconds == 20


def test_balances_by_expected_finish_with_speed():
    shots = [shot(0, c, start=c * 10.0) for c in range(6)]
    fast, slow = Node("fast", speed=2.0), Node("slow", speed=1.0)
    out = plan_placement(shots, [slow, fast], seconds_per_shot=60)
    counts = {n: sum(a.node_id == n for a in out) for n in ("fast", "slow")}
    assert counts == {"fast": 4, "slow": 2}
    finish = {n: max(a.finish_s for a in out if a.node_id == n) for n in counts}
    assert finish["fast"] == pytest.approx(120) and finish["slow"] == pytest.approx(120)
    for n in counts:   # queue positions on one node don't overlap
        mine = sorted((a for a in out if a.node_id == n), key=lambda a: a.start_s)
        for x, y in zip(mine, mine[1:]):
            assert y.start_s == pytest.approx(x.finish_s)


def test_longest_chain_first_and_cost_from_params():
    shots = [shot(0, 0, est_seconds=10), shot(0, 1, start=50, est_seconds=100), shot(1, 1, start=54, est_seconds=100)]
    out = plan_placement(shots, [Node("a"), Node("b")])
    long = next(a for a in out if a.chain == 1)
    assert long.est_seconds == 200 and long.start_s == 0
    assert next(a for a in out if a.chain == 0).node_id != long.node_id


def test_filters_by_model_and_memory():
    nodes = [Node("small", memory_bytes=32 * GB, models=("lc",)), Node("big", memory_bytes=128 * GB, models=("lc",)),
             Node("other", memory_bytes=256 * GB, models=("flux",))]
    shots = [shot(0, c, start=c) for c in range(3)]
    out = plan_placement(shots, nodes, 10, model="lc", memory_bytes=40 * GB)
    assert {a.node_id for a in out} == {"big"}
    out2 = plan_placement(shots + [shot(0, 9, memory_bytes=200 * GB)], [Node("any", memory_bytes=256 * GB)] + nodes,
                          10, model="lc")
    assert next(a for a in out2 if a.chain == 9).node_id == "any"     # empty models = any model
    with pytest.raises(NoNodeFits):
        plan_placement(shots, nodes, 10, model="lc", memory_bytes=300 * GB)
    with pytest.raises(NoNodeFits):
        plan_placement(shots, [], 10)


def test_unknown_memory_and_duration_default_cost():
    out = plan_placement([shot(0, 0, dur=3.0), shot(1, 0, dur=5.0)], [Node("a")])
    assert out[0].est_seconds == pytest.approx(8.0)


def test_join_order_follows_the_edit_not_the_nodes():
    shots = [shot(0, 0, start=0), shot(1, 0, start=4), shot(0, 1, start=8), shot(1, 1, start=12),
             shot(0, 2, start=16)]
    out = plan_placement(shots, [Node("a"), Node("b", speed=3.0)], seconds_per_shot=5)
    order = join_order(out)
    assert [(c, i) for _, c, i in order] == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    node_of = {a.chain: a.node_id for a in out}
    assert all(n == node_of[c] for n, c, _ in order)


def test_assignment_dataclass():
    a = Assignment("n", 0, [shot(0, 0)], 12.0, start_s=3.0)
    assert a.finish_s == 15.0


def test_planner_accepts_longcat_plan():
    from pathlib import Path

    from shutter.adapters.longcat import LongCatAvatarAdapter
    from shutter.types import JobRequest

    ad = LongCatAvatarAdapter("lc", Path("/nonexistent"))
    shots = ad.plan(JobRequest(model="lc", audio="x.wav", duration_s=20.0, framing="small"))
    out = plan_placement(shots, [Node("a"), Node("b")])
    assert len(out) == 1 and len(out[0].shots) == len(shots) and out[0].est_seconds > 0
    assert all(s.params["framing"] == "small" for s in shots)   # the reels option travels in the shot params
