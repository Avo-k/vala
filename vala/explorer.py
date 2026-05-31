"""Lichess Opening Explorer — real human reply frequencies as the opponent model.

This is the *modern-API* realization of Thomas Ahle's
`chess-openings-expectimax` (https://github.com/thomasahle/chess-openings-expectimax),
which ran expectimax over opening trees using empirical human-move frequencies
from a Lichess game dump. We credit that idea; here the same signal comes live
from the Opening Explorer endpoint (https://lichess.org/api#tag/Opening-Explorer),
filtered to the band nearest the opponent's actual rating.

`predict(board)` returns `{uci: prob}` from raw game counts when the position has
≥ `min_games` games, else `None` (caller falls back to Maia). Beautiful side
effect: a sound move never played in the DB is simply absent (P = 0) — real,
never invented, which is exactly the tail we don't want a model to hallucinate.

Adapted from the sibling rorschach project. Network-bound, so vala consults it
only at shallow (opening) nodes; the tree interior uses Maia.
"""
from __future__ import annotations

import os

import chess
import requests

URL = "https://explorer.lichess.ovh/{db}"

# Lichess Explorer rating buckets (the lower bound of each band).
_BUCKETS = (0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500)


def bands_for_elo(elo: int) -> tuple[int, ...]:
    """The two buckets straddling `elo` — the human reply mix nearest their level."""
    below = [b for b in _BUCKETS if b <= elo]
    lo = below[-1] if below else _BUCKETS[0]
    above = [b for b in _BUCKETS if b > lo]
    hi = above[0] if above else lo
    return (lo, hi) if hi != lo else (lo,)


class OpeningExplorer:
    def __init__(
        self,
        db: str = "lichess",
        speeds: tuple[str, ...] = ("blitz", "rapid"),
        ratings: tuple[int, ...] = (1400, 1600, 1800, 2000),
        min_games: int = 10,
        timeout: float = 5.0,
        token: str | None = None,
    ) -> None:
        self.db = db
        self.speeds = ",".join(speeds)
        self.ratings = ",".join(str(r) for r in ratings)
        self.min_games = min_games
        self.timeout = timeout
        self.token = token if token is not None else os.environ.get("LICHESS_TOKEN")
        self._cache: dict[str, dict[str, float] | None] = {}
        self._headers = {"User-Agent": "vala/0.1"}
        if self.token:
            self._headers["Authorization"] = f"Bearer {self.token}"
        self.n_hits = 0
        self.n_misses = 0

    @classmethod
    def for_elo(cls, elo: int, **kw) -> "OpeningExplorer":
        """Explorer filtered to the rating band straddling the opponent's Elo."""
        return cls(ratings=bands_for_elo(elo), **kw)

    def predict(self, board: chess.Board) -> dict[str, float] | None:
        """`{uci: prob}` from real games, or None if too few games at this position."""
        fen = board.fen()
        if fen in self._cache:
            return self._cache[fen]
        params = {
            "fen": fen, "variant": "standard",
            "speeds": self.speeds, "ratings": self.ratings, "moves": 30,
        }
        try:
            r = requests.get(URL.format(db=self.db), params=params,
                             headers=self._headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError):
            self._cache[fen] = None
            self.n_misses += 1
            return None

        total = data.get("white", 0) + data.get("draws", 0) + data.get("black", 0)
        if total < self.min_games:
            self._cache[fen] = None
            self.n_misses += 1
            return None
        probs = {
            m["uci"]: (m["white"] + m["draws"] + m["black"]) / total
            for m in data.get("moves", [])
        }
        self._cache[fen] = probs
        self.n_hits += 1
        return probs
