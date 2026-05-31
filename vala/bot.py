"""vala composition root: solid by default, occasional Maia-baited swindle.

The move pipeline, per the project's framing ("plays solid, but de temps en temps
sets a trap"):

  1. Patricia MultiPV — always (we play one of these, and it's the candidate set).
  2. Cheap trigger screen — a depth-1 pass measuring "blunder mass": the
     opponent-model probability on replies that hand us ≥ `blunder_cp` beyond a
     candidate's sound value. Most positions are quiet → screen is the whole
     cost, and we play the engine-best move.
  3. Only when some candidate's blunder mass clears `trigger_thresh` do we pay
     for the deeper level-synchronous expectimax (`ev_select`) and possibly
     deviate to a bait.

Profiles bundle the calibration dial. `risk_cp` is the "spice" knob the user
asked for: how much objective eval vala will gamble to set a trap (the
exploit↔safety / Restricted-Nash dial). Higher = trappier and more exploitable;
lower = closer to plain best-move play.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

from vala.engine import Candidate
from vala.pool import PatriciaPool
from vala.search import (
    MATE_CUTOFF, MoveEval, SelectResult, ev_select, reply_dists, _to_root_pov,
)

PROFILES: dict[str, dict] = {
    # risk_cp: eval (cp) vala will gamble at best opposing play (the spice dial)
    # margin_cp: EV uplift a bait must beat engine-best by (noise guard)
    # human_depth: opponent error-chain plies modeled (≥2 to see real swindles)
    # trigger_cp: depth-1 expected uplift (cp) needed before paying for the deep
    #   search. The screen sums P(reply)·max(0, our_cp(reply) − sound) over the
    #   opponent's likely replies — a cheap proxy for "deeper search will find
    #   something" that, unlike a hard one-move blunder test, also fires on the
    #   accumulation of small human imperfections that compound into a swindle.
    "solid":      dict(risk_cp=40,  margin_cp=25, human_depth=2, top_replies=4, trigger_cp=35),
    "balanced":   dict(risk_cp=90,  margin_cp=15, human_depth=2, top_replies=4, trigger_cp=22),
    "aggressive": dict(risk_cp=170, margin_cp=8,  human_depth=3, top_replies=5, trigger_cp=12),
}


@dataclass
class MoveInfo:
    profile: str
    best_cp: int
    chosen_cp: int
    eval_loss: int            # objective cp sacrificed vs engine-best (≥0)
    ev_best: float
    ev_chosen: float
    deviated: bool
    triggered: bool           # did the screen escalate to the deep search?
    trap_potential: float     # max over candidates of depth-1 expected uplift (cp)
    oracle: str               # "E" = Lichess explorer used, "M" = Maia
    n_engine_calls: int
    n_maia_calls: int
    bypass: str | None = None


def screen_trap_potential(
    board: chess.Board,
    feasible: list[Candidate],
    pool: PatriciaPool,
    maia,
    *,
    human_elo: int,
    opp_model,
    top_replies: int,
    leaf_ms: int,
) -> tuple[float, str, int, int]:
    """Depth-1 trap detector. For each feasible candidate, the expected upside
    from opponent imperfection: Σ_r P(r)·max(0, our_cp(r) − sound_cp). Summing
    every positive deviation (not just one-move blunders) makes this a proxy for
    "deeper search will find a swindle" — it fires both on immediate blunders and
    on the accumulation of small human errors that compound at depth ≥ 2.
    Returns (max_expected_uplift_cp, oracle, n_engine, n_maia).
    """
    root_white = board.turn
    boards, cs = [], []
    for c in feasible:
        b = board.copy()
        b.push(c.move)
        if not b.is_game_over():
            boards.append(b)
            cs.append(c)
    if not boards:
        return 0.0, "M", 0, 0

    dists, n_maia = reply_dists(boards, maia, human_elo, opp_model=opp_model, allow_explorer=True)
    used_explorer = opp_model is not None and n_maia < len(boards)

    spec: list[tuple[int, float, chess.Board]] = []  # (candidate index, prob, board after reply)
    for ci, (b, dist) in enumerate(zip(boards, dists)):
        for uci, p in sorted(dist.items(), key=lambda kv: -kv[1])[:top_replies]:
            bb = b.copy()
            bb.push(chess.Move.from_uci(uci))
            spec.append((ci, p, bb))

    vals = pool.map_eval([bb for _, _, bb in spec], leaf_ms)
    uplift = [0.0] * len(cs)
    for (ci, p, bb), v in zip(spec, vals):
        our_cp = _to_root_pov(v, bb.turn, root_white)
        uplift[ci] += p * max(0.0, our_cp - cs[ci].cp)

    oracle = "E" if used_explorer else "M"
    return (max(uplift) if uplift else 0.0), oracle, len(spec), n_maia


def vala_move(
    board: chess.Board,
    pool: PatriciaPool,
    maia,
    *,
    profile: str = "balanced",
    human_elo: int = 1500,
    opp_model=None,
    k: int = 6,
    cand_ms: int = 200,
    leaf_ms: int = 50,
    vala_ms: int = 50,
    human_depth: int | None = None,
    top_replies: int | None = None,
) -> tuple[chess.Move, MoveInfo]:
    """Return vala's move plus diagnostics.

    `human_depth` / `top_replies` override the profile (used by time management).
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {list(PROFILES)}")
    kn = dict(PROFILES[profile])
    if human_depth is not None:
        kn["human_depth"] = human_depth
    if top_replies is not None:
        kn["top_replies"] = top_replies

    cands = pool.multipv(board, k=k, time_ms=cand_ms)
    n_engine = 1
    n_maia = 0
    best = cands[0]

    def info(chosen_eval: MoveEval, best_eval: MoveEval, *, deviated: bool,
             triggered: bool, potential: float, oracle: str, bypass: str | None) -> MoveInfo:
        return MoveInfo(
            profile=profile, best_cp=best.cp, chosen_cp=chosen_eval.cand.cp,
            eval_loss=best.cp - chosen_eval.cand.cp,
            ev_best=best_eval.ev, ev_chosen=chosen_eval.ev,
            deviated=deviated, triggered=triggered, trap_potential=potential, oracle=oracle,
            n_engine_calls=n_engine, n_maia_calls=n_maia, bypass=bypass,
        )

    # Mate / no real alternative → just play the engine's best, no trap machinery.
    best_me = MoveEval(cand=best, ev=float(best.cp), sound_cp=best.cp)
    if abs(best.cp) >= MATE_CUTOFF:
        return best.move, info(best_me, best_me, deviated=False, triggered=False,
                               potential=0.0, oracle="-", bypass="mate")
    feasible = [c for c in cands if best.cp - c.cp <= kn["risk_cp"]]
    if len(feasible) == 1:
        return best.move, info(best_me, best_me, deviated=False, triggered=False,
                               potential=0.0, oracle="-", bypass="no-alt")

    # Cheap trigger screen.
    potential, oracle, se, sm = screen_trap_potential(
        board, feasible, pool, maia, human_elo=human_elo, opp_model=opp_model,
        top_replies=kn["top_replies"], leaf_ms=leaf_ms,
    )
    n_engine += se
    n_maia += sm
    if potential < kn["trigger_cp"]:
        return best.move, info(best_me, best_me, deviated=False, triggered=False,
                               potential=potential, oracle=oracle, bypass=None)

    # Trap plausible → pay for the deep expectimax.
    res: SelectResult = ev_select(
        board, pool, maia, cands=cands, human_elo=human_elo,
        human_depth=kn["human_depth"], top_replies=kn["top_replies"],
        leaf_ms=leaf_ms, vala_ms=vala_ms,
        risk_cp=kn["risk_cp"], margin_cp=kn["margin_cp"], opp_model=opp_model,
    )
    n_engine += res.n_engine_calls
    n_maia += res.n_maia_calls
    return res.chosen, info(res.chosen_eval, res.best_eval, deviated=res.deviated,
                            triggered=True, potential=potential, oracle=oracle, bypass=res.bypass)
