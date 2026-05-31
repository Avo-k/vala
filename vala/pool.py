"""A pool of Patricia subprocesses for parallel evaluation on CPU.

Patricia is single-threaded and `chess.engine.SimpleEngine` is synchronous, so
the way to use 32 cores is N independent engine subprocesses driven by N worker
threads. vala's expectimax issues many independent leaf/best-move searches per
ply; the pool runs a whole level's worth concurrently — the lever that turns the
serial scaling wall (≈ top_replies^depth × 45 ms) into wall-clock ≈ that count
divided by the pool size.

Each job borrows an engine from a queue for its duration, so exactly `n` searches
run at once and engines are never shared between threads.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import chess

from vala.engine import Candidate, PatriciaEngine, DEFAULT_PATH


def default_pool_size() -> int:
    return max(1, min(16, (os.cpu_count() or 4) - 2))


class PatriciaPool:
    def __init__(
        self,
        n: int | None = None,
        path: Path | str = DEFAULT_PATH,
        hash_mb: int = 32,
        uci_elo: int | None = None,
        skill: int | None = None,
    ) -> None:
        self.n = n or default_pool_size()
        self._engines: Queue[PatriciaEngine] = Queue()
        self._all: list[PatriciaEngine] = []
        for _ in range(self.n):
            eng = PatriciaEngine(path, hash_mb=hash_mb, threads=1, uci_elo=uci_elo, skill=skill)
            self._all.append(eng)
            self._engines.put(eng)
        self._pool = ThreadPoolExecutor(max_workers=self.n)

    # -- single-shot (borrow one engine) -----------------------------------
    def multipv(self, board: chess.Board, k: int, time_ms: int) -> list[Candidate]:
        eng = self._engines.get()
        try:
            return eng.multipv_search(board, k=k, time_ms=time_ms)
        finally:
            self._engines.put(eng)

    # -- parallel maps (one job per board, distributed over the pool) ------
    def map_eval(self, boards: list[chess.Board], time_ms: int) -> list[int | None]:
        """Parallel single-PV eval of each board, cp from each board's side-to-move POV."""
        def job(b: chess.Board) -> int | None:
            eng = self._engines.get()
            try:
                return eng.eval_position(b, time_ms)
            finally:
                self._engines.put(eng)
        if not boards:
            return []
        return list(self._pool.map(job, boards))

    def map_best_move(self, boards: list[chess.Board], time_ms: int) -> list[chess.Move | None]:
        """Parallel MultiPV(k=1): the engine's best move on each board (None if none)."""
        def job(b: chess.Board) -> chess.Move | None:
            eng = self._engines.get()
            try:
                cands = eng.multipv_search(b, k=1, time_ms=time_ms)
                return cands[0].move if cands else None
            finally:
                self._engines.put(eng)
        if not boards:
            return []
        return list(self._pool.map(job, boards))

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for eng in self._all:
            try:
                eng.quit()
            except Exception:
                pass
        self._all.clear()

    def __enter__(self) -> "PatriciaPool":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
