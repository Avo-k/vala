"""Tests for vala's EV selector and bot pipeline.

Fast layer: pure-logic deviation policy + mate bypass (no models).
Slow layer: real Patricia+Maia — the Blackburne trap regression and a quiet
position staying solid. Skipped when models/binary aren't available.
"""
from __future__ import annotations

import chess
import pytest

from vala.engine import Candidate, MATE_SCORE
from vala.search import MoveEval, decide, ev_select


def _me(uci: str, ev: float, sound_cp: int) -> MoveEval:
    return MoveEval(cand=Candidate(move=chess.Move.from_uci(uci), cp=sound_cp),
                    ev=ev, sound_cp=sound_cp)


def test_decide_stays_solid_without_margin():
    best = _me("e2e4", ev=20.0, sound_cp=20)
    bait = _me("d2d4", ev=28.0, sound_cp=-30)  # higher EV but only +8 over best
    chosen, deviated = decide(best, sorted([bait, best], key=lambda m: -m.ev), margin_cp=15)
    assert not deviated and chosen is best


def test_decide_deviates_past_margin():
    best = _me("e2e4", ev=20.0, sound_cp=20)
    bait = _me("d2d4", ev=40.0, sound_cp=-30)  # +20 over best, beats margin 15
    chosen, deviated = decide(best, sorted([bait, best], key=lambda m: -m.ev), margin_cp=15)
    assert deviated and chosen is bait


def test_decide_best_is_also_top_ev():
    best = _me("e2e4", ev=50.0, sound_cp=20)
    other = _me("d2d4", ev=10.0, sound_cp=10)
    chosen, deviated = decide(best, sorted([other, best], key=lambda m: -m.ev), margin_cp=15)
    assert not deviated and chosen is best


class _MatePool:
    """Minimal pool stub reporting a forced mate so ev_select bypasses search."""
    def multipv(self, board, k, time_ms):
        return [Candidate(move=next(iter(board.legal_moves)), cp=MATE_SCORE - 1)]


def test_mate_bypass_skips_model():
    res = ev_select(chess.Board(), _MatePool(), maia=None, human_depth=2)
    assert res.bypass == "mate"
    assert not res.deviated
    assert res.n_engine_calls == 0  # used the passed candidate; never searched


# ---- Allie diagnostics annotation (roadmap step 1): fast, no model needed -----

class _NoAltPool:
    """Two candidates with only the best feasible → vala_move takes the 'no-alt'
    fast path (no screen/Maia), so we can test the Allie annotation in isolation."""
    def multipv(self, board, k=6, time_ms=200):
        a, b = list(board.legal_moves)[:2]
        return [Candidate(move=a, cp=30), Candidate(move=b, cp=-300)]


class _FakeAllie:
    """Stands in for AlliePredictor — no torch/checkpoint."""
    def __init__(self):
        self.calls = 0

    def predict(self, board, elo=1500, time_control="300+0"):
        self.calls += 1
        from vala.allie import AllieOut
        return AllieOut(move_probs={"e7e5": 0.7, "c7c5": 0.3}, value=0.12, think_time=4.2)


def test_allie_annotates_without_changing_move():
    from vala.bot import vala_move
    board = chess.Board()
    expected = _NoAltPool().multipv(board)[0].move
    allie = _FakeAllie()
    move, mi = vala_move(board, _NoAltPool(), maia=None, allie=allie, human_elo=1500)
    assert move == expected                    # diagnostics never change the move
    assert allie.calls == 1
    assert mi.allie_think_time == 4.2
    assert mi.allie_value == 0.12
    assert mi.allie_reply_top == "e7e5" and mi.allie_reply_p == 0.7


def test_allie_none_leaves_annotation_empty():
    from vala.bot import vala_move
    move, mi = vala_move(chess.Board(), _NoAltPool(), maia=None, allie=None)
    assert mi.allie_think_time is None and mi.allie_reply_top is None


# ---- screen-bait (allow_deep=False, the blitz/small-pool fast path) -----------

def test_decide_screen_bait_stays_solid_without_trap():
    from vala.bot import decide_screen_bait
    best = Candidate(move=chess.Move.from_uci("e2e4"), cp=30)
    alt = Candidate(move=chess.Move.from_uci("d2d4"), cp=10)
    chosen, dev = decide_screen_bait([best, alt], [1.0, 2.0], best.move, margin_cp=15)
    assert not dev and chosen is best


def test_decide_screen_bait_deviates_on_strong_trap():
    from vala.bot import decide_screen_bait
    best = Candidate(move=chess.Move.from_uci("e2e4"), cp=30)
    alt = Candidate(move=chess.Move.from_uci("d2d4"), cp=10)
    # alt is 20cp worse but the screen sees +60 expected uplift → proxy 70 vs 31.
    chosen, dev = decide_screen_bait([best, alt], [1.0, 60.0], best.move, margin_cp=15)
    assert dev and chosen is alt


class _ScreenPool:
    """Mock pool: two candidates; leaf evals make the d2d4 line a strong trap."""
    def multipv(self, board, k=6, time_ms=200):
        return [Candidate(move=chess.Move.from_uci("e2e4"), cp=30),
                Candidate(move=chess.Move.from_uci("d2d4"), cp=10)]

    def map_eval(self, boards, ms):
        return [400 if "d2d4" in [m.uci() for m in b.move_stack] else 100 for b in boards]


class _OneReplyMaia:
    def predict_batch(self, boards, elos_self, elos_oppo):
        return [({next(iter(b.legal_moves)).uci(): 1.0}, 0.5) for b in boards]


def test_screen_bait_path_springs_trap_without_deep_search():
    from vala.bot import vala_move
    move, mi = vala_move(chess.Board(), _ScreenPool(), _OneReplyMaia(),
                         profile="balanced", allow_deep=False)
    assert mi.mode == "screen"           # used the depth-1 screen, not ev_select
    assert mi.deviated and mi.triggered
    assert move == chess.Move.from_uci("d2d4")


# ---- slow integration (real Patricia + Maia) -------------------------------

@pytest.fixture(scope="module")
def models():
    try:
        from vala.maia import make_predictor
        from vala.pool import PatriciaPool
        pool = PatriciaPool(n=8)
        maia = make_predictor("maia3-5m", device="cpu")
    except Exception as exc:
        pytest.skip(f"models unavailable: {exc}")
    yield pool, maia
    pool.close()


@pytest.mark.slow
def test_blackburne_trap_fires_at_depth2(models):
    pool, maia = models
    board = chess.Board()
    for m in "e4 e5 Nf3 Nc6 Bc4".split():
        board.push_san(m)
    res = ev_select(board, pool, maia, human_elo=1100, k=6, human_depth=2,
                    cand_ms=250, leaf_ms=40, vala_ms=40, top_replies=4,
                    risk_cp=120, margin_cp=15)
    assert res.deviated, "expected vala to deviate to the bait at depth 2"
    assert res.chosen.uci() == "c6d4", f"expected ...Nd4, got {res.chosen.uci()}"


@pytest.mark.slow
def test_bot_plays_solid_in_quiet_position(models):
    pool, maia = models
    from vala.bot import vala_move
    board = chess.Board()
    for m in "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3".split():
        board.push_san(m)
    move, info = vala_move(board, pool, maia, profile="balanced", human_elo=1500)
    assert not info.deviated, f"expected solid play, deviated to {move.uci()} ({info})"
