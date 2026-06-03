"""Per-move time management.

Translate a move budget (ms) into search knobs. A quiet move costs MultiPV + the
cheap trigger screen; a *triggered* move adds the level-synchronous deep search,
whose wall-clock ≈ (engine jobs / pool size) × leaf_ms, with jobs growing as
~Σ_levels top_replies^level. So the deep search must be *sized to the actual pool*
— on a small pool (the deployed VM runs Pool=2, shared with rorschach) the same
depth/breadth is many times slower, and a search that fits a 16-worker box will
flag on a 2-worker box.

We therefore estimate the deep-search cost for the real pool and pick the richest
(depth, top_replies) whose estimate fits a fraction of the budget; if even the
minimal deep search doesn't fit, we play the engine's best move (no bait this move)
rather than risk a time forfeit.
"""
from __future__ import annotations

from dataclasses import dataclass

# Cost anchor: a depth-2 / top_replies-4 deep search measured ~1.6s on the dev box
# at Pool=16 (CLAUDE.md). Cost scales ∝ (REF_POOL / pool) and ∝ jobs(depth, tr).
# We bias the constant HIGH (overestimate → degrade to fast-best sooner) because a
# slow move that still plays is fine, a flagged game is not.
_REF_POOL = 16
_ANCHOR_S = 2.0
_ANCHOR_JOBS = 20            # jobs(depth=2, tr=4) = 4 + 16
_DEEP_DEPTH3_MIN_MS = 8000   # only consider depth-3 when there's real time (rapid+)
_SAFETY = 0.5               # search must fit this fraction of the budget
# Screen-bait fallback (depth-1): one leaf-eval wave over ~k candidates × tr replies.
_SCREEN_K = 6               # MultiPV candidate cap
_SCREEN_FUDGE = 1.5         # overhead on the raw leaf-wave time


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _jobs(depth: int, tr: int) -> int:
    return sum(tr ** lvl for lvl in range(1, depth + 1))


def _deep_cost_s(depth: int, tr: int, pool_size: int) -> float:
    """Estimated wall-clock seconds of the deep search on a pool of `pool_size`."""
    return _ANCHOR_S * (_REF_POOL / max(1, pool_size)) * (_jobs(depth, tr) / _ANCHOR_JOBS)


def _screen_cost_s(tr: int, pool_size: int, leaf_ms: int, cand_ms: int) -> float:
    """Estimated wall-clock seconds of the depth-1 screen (one leaf-eval wave over
    ~_SCREEN_K candidates × tr replies) plus the MultiPV call."""
    leaf_jobs = _SCREEN_K * tr
    return cand_ms / 1000.0 + (leaf_jobs / max(1, pool_size)) * (leaf_ms / 1000.0) * _SCREEN_FUDGE


@dataclass(frozen=True)
class Plan:
    cand_ms: int
    leaf_ms: int
    vala_ms: int
    human_depth: int
    top_replies: int
    run_deep: bool    # full expectimax fits the budget on this pool
    run_screen: bool  # the cheap depth-1 screen-bait fits (always True when run_deep);
                      # False ⇒ too little time even to screen → play engine-best


def plan(budget_ms: int, base_depth: int, base_top_replies: int,
         pool_size: int = _REF_POOL) -> Plan:
    """Search knobs for a per-move budget of `budget_ms` wall-clock ms on a Patricia
    pool of `pool_size` workers."""
    budget = max(50, budget_ms)
    cand_ms = _clamp(int(budget * 0.25), 80, 350)
    leaf_ms = _clamp(int(budget * 0.05), 25, 70)

    # Bullet-ish: no time for a tree dive — play the sound move fast.
    if budget < 400:
        return Plan(cand_ms=_clamp(budget, 50, 250), leaf_ms=25, vala_ms=25,
                    human_depth=base_depth, top_replies=base_top_replies,
                    run_deep=False, run_screen=False)

    # Depth-3 only when there's genuinely time; otherwise cap at 2 (and it's
    # non-monotonic anyway — see FINDINGS).
    max_depth = base_depth if budget >= _DEEP_DEPTH3_MIN_MS else min(base_depth, 2)
    fit_s = budget / 1000.0 * _SAFETY

    # 1. Richest (depth, tr) — depth first, then breadth — whose deep search fits.
    for depth in range(max_depth, 1, -1):
        for tr in range(base_top_replies, 1, -1):
            if _deep_cost_s(depth, tr, pool_size) <= fit_s:
                return Plan(cand_ms=cand_ms, leaf_ms=leaf_ms, vala_ms=leaf_ms,
                            human_depth=depth, top_replies=tr,
                            run_deep=True, run_screen=True)

    # 2. Deep search doesn't fit (small pool / blitz) — can we still afford the
    #    depth-1 screen to bait cheaply? Pick the widest screen that fits.
    for tr in range(base_top_replies, 1, -1):
        if _screen_cost_s(tr, pool_size, leaf_ms, cand_ms) <= fit_s:
            return Plan(cand_ms=cand_ms, leaf_ms=leaf_ms, vala_ms=leaf_ms,
                        human_depth=1, top_replies=tr, run_deep=False, run_screen=True)

    # 3. No time even to screen — play the engine's best move.
    return Plan(cand_ms=cand_ms, leaf_ms=leaf_ms, vala_ms=leaf_ms,
                human_depth=min(base_depth, 2), top_replies=2,
                run_deep=False, run_screen=False)
