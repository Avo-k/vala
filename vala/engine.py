"""Patricia UCI wrapper. Returns top-K candidates with side-to-move cp scores.

Vendored from rorschach (treat Patricia as a black-box UCI binary). Adds nothing
beyond what vala's search needs: MultiPV candidate generation and a single-PV
eval-after-move probe.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "bin" / "patricia"
MATE_SCORE = 10000


@dataclass(frozen=True)
class Candidate:
    move: chess.Move
    cp: int           # side-to-move perspective; mates encoded as ±MATE_SCORE - dist
    depth: int | None = None


class PatriciaEngine:
    """Thin wrapper over `chess.engine.SimpleEngine`. Use as a context manager."""

    def __init__(
        self,
        path: Path | str = DEFAULT_PATH,
        hash_mb: int = 64,
        threads: int = 1,
        uci_elo: int | None = None,
        skill: int | None = None,
    ) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(str(path))
        cfg: dict = {"Hash": hash_mb, "Threads": threads}
        # Optional handicap: cap playing strength so a deployed vala can be
        # matched against comparable humans (where baiting actually matters).
        if uci_elo is not None:
            cfg["UCI_LimitStrength"] = True
            cfg["UCI_Elo"] = int(uci_elo)
        if skill is not None:
            cfg["Skill_Level"] = int(skill)
        self._engine.configure(cfg)

    def multipv_search(
        self,
        board: chess.Board,
        k: int = 5,
        time_ms: int = 200,
    ) -> list[Candidate]:
        """Up to `k` distinct first-moves, best-first (highest cp), side-to-move POV.

        Patricia occasionally emits duplicate first-moves across PVs; dedupe by
        first move, keeping the best score seen for each.
        """
        infos = self._engine.analyse(
            board,
            limit=chess.engine.Limit(time=time_ms / 1000),
            multipv=k,
        )
        best_per_move: dict[str, Candidate] = {}
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            move = pv[0]
            score = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
            if score is None:
                continue
            cand = Candidate(move=move, cp=score, depth=info.get("depth"))
            existing = best_per_move.get(move.uci())
            if existing is None or cand.cp > existing.cp:
                best_per_move[move.uci()] = cand
        return sorted(best_per_move.values(), key=lambda c: -c.cp)

    def eval_position(self, board: chess.Board, time_ms: int) -> int | None:
        """Single-PV eval of `board`, cp from the side-to-move POV. None if no score."""
        info = self._engine.analyse(board, limit=chess.engine.Limit(time=time_ms / 1000))
        return info["score"].pov(board.turn).score(mate_score=MATE_SCORE)

    def eval_after_move(self, board: chess.Board, move: chess.Move, time_ms: int) -> int | None:
        """Push `move`, single-PV eval, cp from the MOVING player's POV."""
        moving_player = board.turn
        board.push(move)
        try:
            info = self._engine.analyse(board, limit=chess.engine.Limit(time=time_ms / 1000))
            return info["score"].pov(moving_player).score(mate_score=MATE_SCORE)
        finally:
            board.pop()

    def quit(self) -> None:
        self._engine.quit()

    def __enter__(self) -> "PatriciaEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.quit()
