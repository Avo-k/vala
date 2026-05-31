"""vala's selector: level-synchronous parallel expectimax / best-response.

vala maximizes expected value against a *fallible* opponent. At human-to-move
nodes the reply is the opponent model's distribution P(r | s, elo) (real Lichess
frequencies when available, else Maia); at vala-to-move nodes vala plays its own
best move. The tree is expanded for `human_depth` human plies, then leaves are
scored by the engine. Values are centipawns from the **root mover's** (vala's) POV.

    V(human node) = Σ_r P(r)·V(after r → vala-best)  +  (1 − covered)·tail
    V(leaf)       = engine_eval  (terminal: ±mate / 0)

Expansion is **level-synchronous**: each ply, all the opponent-model queries are
issued as one batched Maia forward and all the engine searches (tail evals,
vala best-moves, leaf evals) are dispatched to a `PatriciaPool` so a whole level
runs concurrently. This is what makes depth ≥ 2 affordable on CPU. Values are
backed up bottom-up afterward (pure arithmetic, no engine calls).

Why depth must be ≥ 2: a one-ply rollout assumes the human plays perfectly after
their first reply, which prices real (multi-move) swindles at ~0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from vala.engine import Candidate, MATE_SCORE
from vala.pool import PatriciaPool

MATE_CUTOFF = 9000
WIN_VALUE = float(MATE_SCORE)


# ---- value/POV helpers ------------------------------------------------------

def _to_root_pov(cp_stm: int | None, turn: bool, root_white: bool) -> float:
    """Convert a side-to-move cp score to the root mover's POV."""
    if cp_stm is None:
        return 0.0
    return float(cp_stm) if (turn == root_white) else float(-cp_stm)


def _terminal_root_value(board: chess.Board, root_white: bool) -> float:
    """Root-POV value of a finished game (no engine call)."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.0
    if outcome.winner is None:
        return 0.0
    return WIN_VALUE if (outcome.winner == root_white) else -WIN_VALUE


# ---- tree -------------------------------------------------------------------

@dataclass
class _Node:
    """A human-to-move node awaiting/holding its children."""
    board: chess.Board
    H: int
    tail_cp: float = 0.0
    covered: float = 0.0
    # (prob, reply_uci, kind 'leaf'|'node', payload value|_Node)
    children: list[tuple[float, str, str, object]] = field(default_factory=list)

    def value(self) -> float:
        acc = 0.0
        for prob, _uci, kind, payload in self.children:
            acc += prob * (payload if kind == "leaf" else payload.value())  # type: ignore[union-attr]
        return acc + (1.0 - self.covered) * self.tail_cp


# ---- public types -----------------------------------------------------------

@dataclass(frozen=True)
class ReplyEval:
    uci: str
    prob: float
    our_cp: float  # backed-up value after this reply (and vala's follow-up), root POV


@dataclass
class MoveEval:
    cand: Candidate
    ev: float
    sound_cp: int
    covered: float = 1.0
    replies: list[ReplyEval] = field(default_factory=list)

    @property
    def is_bait(self) -> bool:
        return self.ev - self.sound_cp > 1.0


@dataclass
class SelectResult:
    chosen: chess.Move
    chosen_eval: MoveEval
    best_eval: MoveEval
    evals: list[MoveEval]
    delta_risk: int
    deviated: bool
    bypass: str | None = None
    n_engine_calls: int = 0
    n_maia_calls: int = 0


def decide(
    best_eval: MoveEval, evals: list[MoveEval], margin_cp: int,
) -> tuple[MoveEval, bool]:
    """Pick the highest-EV move; deviate from engine-best only past `margin_cp`.

    Pure (no engine/Maia/board state) so the deviation policy is unit-testable.
    `evals` is sorted by EV descending; `best_eval` is the engine #1 move's eval.
    """
    top = evals[0]
    if top.cand.move != best_eval.cand.move and top.ev >= best_eval.ev + margin_cp:
        return top, True
    return best_eval, False


# ---- opponent model dispatch ------------------------------------------------

def reply_dists(
    boards: list[chess.Board], maia, human_elo: int, *,
    opp_model=None, allow_explorer: bool = False,
) -> tuple[list[dict[str, float]], int]:
    """Opponent reply distribution per board. Real Lichess frequencies (Explorer)
    at shallow nodes when allowed and available; Maia (one batched forward)
    otherwise. Returns (dists, n_maia_queried)."""
    dists: list[dict[str, float] | None] = [None] * len(boards)
    if opp_model is not None and allow_explorer:
        for i, b in enumerate(boards):
            try:
                d = opp_model.predict(b)
            except Exception:
                d = None
            if d:
                dists[i] = d

    need = [i for i, d in enumerate(dists) if d is None]
    n_maia = 0
    if need and maia is not None:
        out = maia.predict_batch(
            [boards[i] for i in need],
            [human_elo] * len(need), [human_elo] * len(need),
        )
        n_maia = len(need)
        for i, (probs, _win) in zip(need, out):
            dists[i] = probs
    return [d or {} for d in dists], n_maia


# ---- the search -------------------------------------------------------------

def ev_select(
    board: chess.Board,
    pool: PatriciaPool,
    maia,
    *,
    human_elo: int = 1500,
    k: int = 5,
    human_depth: int = 2,
    cand_ms: int = 200,
    leaf_ms: int = 50,
    vala_ms: int = 50,
    top_replies: int = 4,
    risk_cp: int = 100,
    margin_cp: int = 15,
    opp_model=None,
    cands: list[Candidate] | None = None,
) -> SelectResult:
    """Pick the move maximizing expected value against the modeled human.

    `cands` lets a caller (e.g. the trigger screen) pass already-computed MultiPV
    candidates so we don't re-search the root.
    """
    n_engine = 0
    n_maia = 0
    root_white = board.turn

    if cands is None:
        cands = pool.multipv(board, k=k, time_ms=cand_ms)
        n_engine += 1
    if not cands:
        raise ValueError("ev_select got no candidates")
    best = cands[0]

    if abs(best.cp) >= MATE_CUTOFF:
        me = MoveEval(cand=best, ev=float(best.cp), sound_cp=best.cp)
        return SelectResult(best.move, me, me, [me], 0, False, bypass="mate")

    feasible = [c for c in cands if best.cp - c.cp <= risk_cp]

    # Build root human-to-move nodes (one per feasible candidate).
    roots: list[tuple[Candidate, _Node, float | None]] = []
    open_nodes: list[_Node] = []
    for c in feasible:
        b = board.copy()
        b.push(c.move)
        if b.is_game_over():
            roots.append((c, _Node(board=b, H=0), _terminal_root_value(b, root_white)))
        else:
            node = _Node(board=b, H=human_depth)
            roots.append((c, node, None))
            open_nodes.append(node)

    first_level = True
    while open_nodes:
        # 1. opponent reply distributions (Explorer at the root level, Maia deeper)
        dists, nm = reply_dists(
            [n.board for n in open_nodes], maia, human_elo,
            opp_model=opp_model, allow_explorer=first_level,
        )
        n_maia += nm

        # 2. tail value (human plays soundly) — one parallel batch
        tails = pool.map_eval([n.board for n in open_nodes], leaf_ms)
        n_engine += len(open_nodes)
        for n, t in zip(open_nodes, tails):
            n.tail_cp = _to_root_pov(t, n.board.turn, root_white)

        # 3. expand replies → vala-to-move positions
        vala_specs: list[tuple[_Node, float, str, chess.Board]] = []
        term_leaves: list[tuple[_Node, float, str, chess.Board]] = []
        for n, dist in zip(open_nodes, dists):
            ranked = sorted(dist.items(), key=lambda kv: -kv[1])[:top_replies]
            n.covered = sum(p for _, p in ranked)
            for uci, p in ranked:
                b2 = n.board.copy()
                b2.push(chess.Move.from_uci(uci))
                (term_leaves if b2.is_game_over() else vala_specs).append((n, p, uci, b2))

        # 4. vala's best reply at each vala node — one parallel batch
        vmoves = pool.map_best_move([b2 for _, _, _, b2 in vala_specs], vala_ms)
        n_engine += len(vala_specs)

        # 5. assemble children; collect next-level nodes + leaf-eval jobs
        next_open: list[_Node] = []
        leaf_jobs: list[tuple[_Node, float, str, chess.Board]] = list(term_leaves)
        for (n, p, uci, b2), vm in zip(vala_specs, vmoves):
            if vm is None:
                leaf_jobs.append((n, p, uci, b2))
                continue
            b3 = b2.copy()
            b3.push(vm)
            if b3.is_game_over() or (n.H - 1) == 0:
                leaf_jobs.append((n, p, uci, b3))
            else:
                child = _Node(board=b3, H=n.H - 1)
                n.children.append((p, uci, "node", child))
                next_open.append(child)

        # 6. evaluate leaves: terminals analytically, the rest in one parallel batch
        eval_jobs = [(n, p, uci, b) for (n, p, uci, b) in leaf_jobs if not b.is_game_over()]
        term_jobs = [(n, p, uci, b) for (n, p, uci, b) in leaf_jobs if b.is_game_over()]
        if eval_jobs:
            vals = pool.map_eval([b for _, _, _, b in eval_jobs], leaf_ms)
            n_engine += len(eval_jobs)
            for (n, p, uci, b), v in zip(eval_jobs, vals):
                n.children.append((p, uci, "leaf", _to_root_pov(v, b.turn, root_white)))
        for (n, p, uci, b) in term_jobs:
            n.children.append((p, uci, "leaf", _terminal_root_value(b, root_white)))

        open_nodes = next_open
        first_level = False

    # Back up values and assemble MoveEvals.
    evals: list[MoveEval] = []
    for c, node, term_val in roots:
        if term_val is not None:
            ev = term_val
            replies: list[ReplyEval] = []
            covered = 1.0
        else:
            ev = node.value()
            covered = node.covered
            replies = [
                ReplyEval(uci=uci, prob=prob,
                          our_cp=(payload if kind == "leaf" else payload.value()))  # type: ignore[union-attr]
                for prob, uci, kind, payload in node.children
            ]
        evals.append(MoveEval(cand=c, ev=ev, sound_cp=c.cp, covered=covered, replies=replies))

    evals.sort(key=lambda me: -me.ev)
    best_eval = next(me for me in evals if me.cand.move == best.move)
    chosen_eval, deviated = decide(best_eval, evals, margin_cp)

    return SelectResult(
        chosen=chosen_eval.cand.move, chosen_eval=chosen_eval, best_eval=best_eval,
        evals=evals, delta_risk=risk_cp, deviated=deviated,
        n_engine_calls=n_engine, n_maia_calls=n_maia,
    )
