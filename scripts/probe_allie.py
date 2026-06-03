"""Probe Allie v1 inside vala (roadmap step 1: diagnostics only).

Two checks:
  1. AlliePredictor directly — policy/value/think_time on a known bait position.
  2. vala_move with `allie=` — confirm it annotates MoveInfo without changing the
     move (here via a mocked Patricia pool so it runs without the engine/Maia).

Usage:
    ALLIE_MODEL_DIR=/path/to/allie/small python scripts/probe_allie.py
(small checkpoint: `huggingface-cli download --repo-type dataset \
 yimingzhang/allie-models --include 'ablations/small/*' --local-dir models`)
"""
from __future__ import annotations

import os

import chess

from vala.allie import AlliePredictor
from vala.bot import vala_move
from vala.engine import Candidate

CKPT = os.environ.get("ALLIE_MODEL_DIR", "models/ablations/small")


def blackburne_white_to_move() -> chess.Board:
    b = chess.Board()
    for u in "e2e4 e7e5 g1f3 b8c6 f1c4 c6d4".split():
        b.push_uci(u)
    return b  # White to move; the bait 3...Nd4 is set, will White grab e5??


class _FakePool:
    """Minimal Patricia stand-in: returns two candidates, only the best feasible
    (so vala_move takes the 'no-alt' fast path — no screen/Maia needed)."""

    def multipv(self, board, k=6, time_ms=200):
        legal = list(board.legal_moves)
        best, other = legal[0], legal[1]
        return [Candidate(move=best, cp=30), Candidate(move=other, cp=-200)]


def main() -> None:
    allie = AlliePredictor(CKPT, device="cpu")
    print(f"loaded {allie.name}\n")

    # 1. direct diagnostics
    b = blackburne_white_to_move()
    for elo in (1100, 1500, 1900):
        o = allie.predict(b, elo=elo)
        top = sorted(o.move_probs.items(), key=lambda kv: -kv[1])[:3]
        print(f"Elo {elo}: think={o.think_time:4.1f}s  value(W)={o.value:+.3f}  "
              f"top={[(m, round(p*100,1)) for m, p in top]}")
    print()

    # 2. vala_move annotation path (mocked pool; allie must not change the move)
    pool = _FakePool()
    board = blackburne_white_to_move()
    expected = pool.multipv(board)[0].move
    move, mi = vala_move(board, pool, maia=None, allie=allie, human_elo=1500, profile="balanced")
    assert move == expected, "allie diagnostics must not change the chosen move!"
    print(f"vala_move chose {move} (bypass={mi.bypass}) — move unchanged ✓")
    print(f"  annotated: think={mi.allie_think_time:.1f}s  value={mi.allie_value:+.3f}  "
          f"reply~{mi.allie_reply_top} ({mi.allie_reply_p*100:.0f}%)")
    # and with allie=None nothing is computed
    _, mi0 = vala_move(board, pool, maia=None, allie=None, human_elo=1500)
    assert mi0.allie_think_time is None
    print("  allie=None → no annotation ✓")


if __name__ == "__main__":
    main()
