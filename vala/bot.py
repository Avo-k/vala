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

from vala.allie import DEFAULT_TIME_CONTROL
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
    mode: str | None = None   # which selector ran: "deep" | "screen" | None (fast/no machinery)
    # Allie v1 diagnostics (step 1: measured, never used in the decision). Computed
    # on the position the *human* will move in (after vala's chosen move). See
    # vala/allie.py and memory `roadmap-opponent-model`.
    allie_think_time: float | None = None   # predicted human seconds on the reply
    allie_value: float | None = None        # Allie value there, White-POV [-1, 1]
    allie_reply_top: str | None = None      # Allie's most likely human reply (uci)
    allie_reply_p: float | None = None       # ...and its probability


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
) -> tuple[list[float], list[Candidate], str, int, int]:
    """Depth-1 trap detector. For each feasible candidate, the expected upside
    from opponent imperfection: Σ_r P(r)·max(0, our_cp(r) − sound_cp). Summing
    every positive deviation (not just one-move blunders) makes this a proxy for
    "deeper search will find a swindle" — it fires both on immediate blunders and
    on the accumulation of small human errors that compound at depth ≥ 2.

    Returns (uplifts, scored, oracle, n_engine, n_maia), where uplifts[i] is the
    expected uplift (cp) of scored[i] (the feasible candidates that don't end the
    game). The max is the trigger signal; the full list lets a blitz fast path bait
    on the screen alone (see `decide_screen_bait`) when there's no time to go deep.
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
        return [], [], "M", 0, 0

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
    return uplift, cs, oracle, len(spec), n_maia


def decide_screen_bait(
    scored: list[Candidate], uplifts: list[float], best_move: chess.Move, margin_cp: int,
) -> tuple[Candidate, bool]:
    """Depth-1 bait decision from the screen alone (the blitz/small-pool fast path).

    Proxy EV of a candidate = its sound cp + the screen's expected uplift from
    opponent error. Deviate to the highest-proxy-EV move only if it beats engine-
    best's proxy EV by `margin_cp` (mirrors `search.decide`, but with the cheap
    depth-1 uplift instead of the deep expectimax). Pure → unit-testable. `scored`
    is already risk_cp-filtered by the caller.
    """
    proxies = [c.cp + u for c, u in zip(scored, uplifts)]
    bi = next((i for i, c in enumerate(scored) if c.move == best_move),
              max(range(len(scored)), key=lambda i: scored[i].cp))
    ti = max(range(len(scored)), key=lambda i: proxies[i])
    if scored[ti].move != best_move and proxies[ti] >= proxies[bi] + margin_cp:
        return scored[ti], True
    return scored[bi], False


def vala_move(
    board: chess.Board,
    pool: PatriciaPool,
    maia,
    *,
    profile: str = "balanced",
    human_elo: int = 1500,
    opp_model=None,
    allie=None,
    time_control: str | None = None,
    allow_deep: bool = True,
    k: int = 6,
    cand_ms: int = 200,
    leaf_ms: int = 50,
    vala_ms: int = 50,
    human_depth: int | None = None,
    top_replies: int | None = None,
) -> tuple[chess.Move, MoveInfo]:
    """Return vala's move plus diagnostics.

    `human_depth` / `top_replies` override the profile (used by time management).
    `allow_deep=False` (small pool / blitz: no time for the expectimax) keeps the
    cheap trigger screen but baits from it directly via `decide_screen_bait` instead
    of escalating to `ev_select` — so vala can still spring traps on a 2-worker pool.
    `allie`, when given (an `AlliePredictor`), only *annotates* the returned
    MoveInfo with Allie's think-time/value/top-reply for the resulting human-to-move
    position — it never affects which move vala plays (roadmap step 1: measure first).
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
             triggered: bool, potential: float, oracle: str, bypass: str | None,
             mode: str | None = None) -> MoveInfo:
        return MoveInfo(
            profile=profile, best_cp=best.cp, chosen_cp=chosen_eval.cand.cp,
            eval_loss=best.cp - chosen_eval.cand.cp,
            ev_best=best_eval.ev, ev_chosen=chosen_eval.ev,
            deviated=deviated, triggered=triggered, trap_potential=potential, oracle=oracle,
            n_engine_calls=n_engine, n_maia_calls=n_maia, bypass=bypass, mode=mode,
        )

    def finish(move: chess.Move, minfo: MoveInfo) -> tuple[chess.Move, MoveInfo]:
        """Attach Allie diagnostics for the resulting human-to-move position.
        Pure annotation — does not change `move`."""
        if allie is not None:
            after = board.copy()
            after.push(move)
            if not after.is_game_over():
                o = allie.predict(
                    after, elo=human_elo,
                    time_control=time_control or DEFAULT_TIME_CONTROL,
                )
                minfo.allie_think_time = o.think_time
                minfo.allie_value = o.value
                if o.move_probs:
                    top_uci, top_p = max(o.move_probs.items(), key=lambda kv: kv[1])
                    minfo.allie_reply_top = top_uci
                    minfo.allie_reply_p = top_p
        return move, minfo

    # Mate / no real alternative → just play the engine's best, no trap machinery.
    best_me = MoveEval(cand=best, ev=float(best.cp), sound_cp=best.cp)
    if abs(best.cp) >= MATE_CUTOFF:
        return finish(best.move, info(best_me, best_me, deviated=False, triggered=False,
                                      potential=0.0, oracle="-", bypass="mate"))
    feasible = [c for c in cands if best.cp - c.cp <= kn["risk_cp"]]
    if len(feasible) == 1:
        return finish(best.move, info(best_me, best_me, deviated=False, triggered=False,
                                      potential=0.0, oracle="-", bypass="no-alt"))

    # Cheap trigger screen.
    uplifts, scored, oracle, se, sm = screen_trap_potential(
        board, feasible, pool, maia, human_elo=human_elo, opp_model=opp_model,
        top_replies=kn["top_replies"], leaf_ms=leaf_ms,
    )
    n_engine += se
    n_maia += sm
    potential = max(uplifts) if uplifts else 0.0
    if potential < kn["trigger_cp"]:
        return finish(best.move, info(best_me, best_me, deviated=False, triggered=False,
                                      potential=potential, oracle=oracle, bypass=None))

    # Trap plausible. With time, confirm with the deep expectimax; otherwise (small
    # pool / blitz) bait from the depth-1 screen directly so vala still springs traps.
    if not allow_deep:
        chosen_c, deviated = decide_screen_bait(scored, uplifts, best.move, kn["margin_cp"])
        upl = {c.move: u for c, u in zip(scored, uplifts)}
        best_me = MoveEval(cand=best, ev=float(best.cp) + upl.get(best.move, 0.0), sound_cp=best.cp)
        chosen_me = MoveEval(cand=chosen_c, ev=float(chosen_c.cp) + upl.get(chosen_c.move, 0.0),
                             sound_cp=chosen_c.cp)
        return finish(chosen_c.move, info(chosen_me, best_me, deviated=deviated, triggered=True,
                                          potential=potential, oracle=oracle, bypass=None, mode="screen"))

    res: SelectResult = ev_select(
        board, pool, maia, cands=cands, human_elo=human_elo,
        human_depth=kn["human_depth"], top_replies=kn["top_replies"],
        leaf_ms=leaf_ms, vala_ms=vala_ms,
        risk_cp=kn["risk_cp"], margin_cp=kn["margin_cp"], opp_model=opp_model,
    )
    n_engine += res.n_engine_calls
    n_maia += res.n_maia_calls
    return finish(res.chosen, info(res.chosen_eval, res.best_eval, deviated=res.deviated,
                                   triggered=True, potential=potential, oracle=oracle,
                                   bypass=res.bypass, mode="deep"))
