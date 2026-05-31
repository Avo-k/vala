"""Per-move time management.

Translate a move budget (ms) into search knobs. Cost model: a quiet move is
MultiPV + the cheap trigger screen; a triggered move adds the level-synchronous
deep search whose wall-clock ≈ (engine jobs / pool size) × leaf_ms, with jobs
growing as ~top_replies^human_depth. So we scale the movetimes to the budget and
cap human_depth / top_replies so that even a triggered move fits.
"""
from __future__ import annotations

from dataclasses import dataclass


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class Plan:
    cand_ms: int
    leaf_ms: int
    vala_ms: int
    human_depth: int
    top_replies: int
    run_deep: bool   # False ⇒ too little time to bait safely; just play engine-best


def plan(budget_ms: int, base_depth: int, base_top_replies: int) -> Plan:
    """Search knobs for a per-move budget of `budget_ms` wall-clock milliseconds."""
    budget = max(50, budget_ms)
    cand_ms = _clamp(int(budget * 0.25), 80, 350)
    leaf_ms = _clamp(int(budget * 0.05), 25, 70)

    # Bullet-ish: no time for a tree dive — play the sound move fast.
    if budget < 400:
        return Plan(cand_ms=_clamp(budget, 50, 250), leaf_ms=25, vala_ms=25,
                    human_depth=base_depth, top_replies=base_top_replies, run_deep=False)

    if budget < 1500:                       # fast blitz
        depth = min(base_depth, 2)
        tr = min(base_top_replies, 3 if budget < 800 else 4)
    elif budget < 4000:                     # blitz / fast rapid
        depth = min(base_depth, 2)
        tr = base_top_replies
    else:                                   # rapid / classical
        depth = base_depth
        tr = base_top_replies

    return Plan(cand_ms=cand_ms, leaf_ms=leaf_ms, vala_ms=leaf_ms,
                human_depth=depth, top_replies=tr, run_deep=True)
