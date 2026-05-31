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
