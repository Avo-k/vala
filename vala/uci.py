"""UCI shim for vala.

Speaks UCI on stdin/stdout so any host (lichess-bot, Cute Chess, Arena) can drive
vala. Resources (the Patricia pool + Maia + optional Lichess Explorer) load lazily
on the first `isready` so `uci` stays cheap. Per-move time management derives the
search depth and movetimes from the clock.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass

import chess
from dotenv import load_dotenv

from vala import timectl
from vala.bot import PROFILES, vala_move
from vala.pool import PatriciaPool, default_pool_size

MAIA_TYPES = ("maia3-5m", "maia3-23m", "maia3-79m")
DEFAULT_ELO = 1500
_ELO_MIN, _ELO_MAX = 600, 2600


@dataclass
class Options:
    profile: str = "balanced"
    time_ms: int = 0                 # 0 = derive from clock; else fixed ms/move
    maia_type: str = "maia3-5m"
    elo_override: int | None = None  # pin the modeled-human Elo; None = track opponent
    oppo_elo: int | None = None      # from UCI_Opponent
    pool_size: int = default_pool_size()
    use_explorer: bool = False       # real Lichess frequencies at the opening

    @property
    def human_elo(self) -> int:
        """The rating vala models the (human) opponent at."""
        if self.elo_override is not None:
            return self.elo_override
        return self.oppo_elo if self.oppo_elo is not None else DEFAULT_ELO


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    sys.stderr.write(f"[vala] {msg}\n")
    sys.stderr.flush()


def _movelog(record: dict) -> None:
    """Persist one per-move record as JSONL. Always to stderr with a `MOVELOG`
    prefix (captured by `docker logs` since silence_stderr=false), and also to the
    file named by env `VALA_MOVELOG` when set. This is how we measure the deployed
    bot's real behavior on humans (bait rate, eval sacrificed, trap_potential)."""
    line = json.dumps(record, separators=(",", ":"))
    sys.stderr.write(f"MOVELOG {line}\n")
    sys.stderr.flush()
    path = os.environ.get("VALA_MOVELOG")
    if path:
        try:
            with open(path, "a") as fh:
                fh.write(line + "\n")
        except Exception as exc:
            _log(f"movelog file write failed: {exc}")


def _parse_position(args: list[str]) -> chess.Board:
    if not args:
        return chess.Board()
    if args[0] == "startpos":
        board = chess.Board()
        moves_at = 1
    elif args[0] == "fen":
        if len(args) < 7:
            raise ValueError("bad position fen")
        board = chess.Board(" ".join(args[1:7]))
        moves_at = 7
    else:
        raise ValueError(f"bad position prefix {args[0]!r}")
    if moves_at < len(args):
        if args[moves_at] != "moves":
            raise ValueError(f"expected 'moves', got {args[moves_at]!r}")
        for uci in args[moves_at + 1:]:
            board.push(chess.Move.from_uci(uci))
    return board


def _parse_go(args: list[str], turn_white: bool) -> int:
    """Per-move budget (ms) from UCI go args."""
    movetime = wtime = btime = winc = binc = None
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "movetime":
            movetime = int(args[i + 1]); i += 2
        elif tok == "wtime":
            wtime = int(args[i + 1]); i += 2
        elif tok == "btime":
            btime = int(args[i + 1]); i += 2
        elif tok == "winc":
            winc = int(args[i + 1]); i += 2
        elif tok == "binc":
            binc = int(args[i + 1]); i += 2
        elif tok in ("infinite", "ponder"):
            i += 1
        elif tok in ("depth", "nodes", "mate", "movestogo"):
            i += 2
        else:
            i += 1
    if movetime is not None:
        return max(50, movetime)
    my_time = wtime if turn_white else btime
    my_inc = (winc if turn_white else binc) or 0
    if my_time is None:
        return 1500
    # ~4% of remaining + most of the increment, clamped. vala's deep search is
    # spiky, so we keep the per-move cap modest.
    return max(150, min(6000, int(my_time * 0.04 + my_inc * 0.8)))


def _emit_options() -> None:
    profiles = " ".join(f"var {p}" for p in PROFILES)
    _emit(f"option name Profile type combo default balanced {profiles}")
    _emit("option name TimeMs type spin default 0 min 0 max 20000")
    _emit("option name Elo type spin default 0 min 0 max 2600")
    maia_vars = " ".join(f"var {t}" for t in MAIA_TYPES)
    _emit(f"option name MaiaType type combo default maia3-5m {maia_vars}")
    _emit(f"option name Pool type spin default {default_pool_size()} min 1 max 64")
    _emit("option name UseExplorer type check default false")
    _emit("option name UCI_Opponent type string default")


def _parse_uci_opponent_elo(value: str) -> int | None:
    tokens = value.split()
    if len(tokens) < 2:
        return None
    raw = tokens[1]
    if raw.lower() == "none":
        return None
    try:
        elo = int(raw)
    except ValueError:
        return None
    return max(_ELO_MIN, min(_ELO_MAX, elo))


def _set_option(opts: Options, words: list[str]) -> None:
    if "name" not in words:
        return
    name = words[words.index("name") + 1]
    val = " ".join(words[words.index("value") + 1:]) if "value" in words else ""
    if name == "Profile" and val in PROFILES:
        opts.profile = val
    elif name == "TimeMs":
        opts.time_ms = int(val)
    elif name == "Elo":
        v = int(val)
        opts.elo_override = v if v >= _ELO_MIN else None
    elif name == "MaiaType" and val in MAIA_TYPES:
        opts.maia_type = val
    elif name == "Pool":
        opts.pool_size = max(1, int(val))
    elif name == "UseExplorer":
        opts.use_explorer = val.strip().lower() in ("true", "1", "yes")
    elif name == "UCI_Opponent":
        opts.oppo_elo = _parse_uci_opponent_elo(val)


class _Resources:
    def __init__(self) -> None:
        self.pool: PatriciaPool | None = None
        self.maia = None
        self.explorer = None
        self.game_idx = 0  # bumped on ucinewgame so move logs group into games

    def ensure(self, opts: Options) -> None:
        if self.pool is None or self.pool.n != opts.pool_size:
            if self.pool is not None:
                self.pool.close()
            _log(f"starting Patricia pool (n={opts.pool_size}) ...")
            self.pool = PatriciaPool(n=opts.pool_size)
        if self.maia is None or getattr(self.maia, "_name", None) != opts.maia_type:
            _log(f"loading Maia ({opts.maia_type}, cpu) ...")
            with contextlib.redirect_stdout(sys.stderr):
                from vala.maia import make_predictor
                self.maia = make_predictor(opts.maia_type, device="cpu")
        if opts.use_explorer and self.explorer is None:
            from vala.explorer import OpeningExplorer
            self.explorer = OpeningExplorer()
            if not self.explorer.token:
                _log("warning: LICHESS_TOKEN not set, explorer likely 401s")

    def shutdown(self) -> None:
        if self.pool is not None:
            try:
                self.pool.close()
            except Exception:
                pass
            self.pool = None


def main() -> None:
    load_dotenv()
    opts = Options()
    res = _Resources()
    board = chess.Board()

    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            words = line.split()
            cmd = words[0]

            if cmd == "uci":
                _emit("id name vala")
                _emit("id author Avo-k")
                _emit_options()
                _emit("uciok")
            elif cmd == "isready":
                res.ensure(opts)
                _emit("readyok")
            elif cmd == "setoption":
                _set_option(opts, words)
            elif cmd == "ucinewgame":
                board = chess.Board()
                res.game_idx += 1
            elif cmd == "position":
                try:
                    board = _parse_position(words[1:])
                except (ValueError, chess.InvalidMoveError) as exc:
                    _log(f"bad position: {exc}")
            elif cmd == "go":
                res.ensure(opts)
                budget = opts.time_ms if opts.time_ms > 0 else _parse_go(words[1:], board.turn == chess.WHITE)
                try:
                    _emit_move(board, res, opts, budget)
                except Exception as exc:  # never hang the host
                    _log(f"search error: {exc!r}")
                    legal = list(board.legal_moves)
                    _emit(f"bestmove {legal[0].uci() if legal else '0000'}")
            elif cmd in ("stop", "ponderhit"):
                pass
            elif cmd == "quit":
                break
    finally:
        res.shutdown()


def _emit_move(board: chess.Board, res: _Resources, opts: Options, budget: int) -> None:
    base = PROFILES[opts.profile]
    pl = timectl.plan(budget, base["human_depth"], base["top_replies"],
                      pool_size=res.pool.n)
    opp_model = res.explorer if opts.use_explorer else None

    ply = len(board.move_stack)
    if not pl.run_screen:
        # Too little time even for the depth-1 screen: just the engine's best move.
        move = res.pool.multipv(board, k=1, time_ms=pl.cand_ms)[0].move
        _emit(f"info string budget={budget} fast-best")
        _movelog(dict(t=round(time.time(), 3), game=res.game_idx, ply=ply,
                      profile=opts.profile, budget=budget, human_elo=opts.human_elo,
                      move=move.uci(), tag="fast-best", deviated=False, triggered=False,
                      mode="fast", fen=board.fen()))
        _emit(f"bestmove {move.uci()}")
        return

    # run_deep ⇒ full expectimax; run_screen-only ⇒ cheap depth-1 screen-bait.
    move, info = vala_move(
        board, res.pool, res.maia,
        profile=opts.profile, human_elo=opts.human_elo, opp_model=opp_model,
        allow_deep=pl.run_deep,
        cand_ms=pl.cand_ms, leaf_ms=pl.leaf_ms, vala_ms=pl.vala_ms,
        human_depth=pl.human_depth, top_replies=pl.top_replies,
    )
    tag = "BAIT" if info.deviated else ("trig" if info.triggered else "solid")
    if info.bypass:
        tag = info.bypass
    _emit(
        f"info depth {pl.human_depth} score cp {info.best_cp} "
        f"string budget={budget} {tag} loss={info.eval_loss} "
        f"EV={info.ev_best:.0f}->{info.ev_chosen:.0f} pot={info.trap_potential:.0f} "
        f"elo={opts.human_elo} oracle={info.oracle} "
        f"calls={info.n_engine_calls}e/{info.n_maia_calls}m"
    )
    _movelog(dict(
        t=round(time.time(), 3), game=res.game_idx, ply=ply,
        profile=opts.profile, budget=budget, human_elo=opts.human_elo,
        move=move.uci(), tag=tag, mode=info.mode,
        deviated=info.deviated, triggered=info.triggered,
        best_cp=info.best_cp, chosen_cp=info.chosen_cp, eval_loss=info.eval_loss,
        trap_potential=round(info.trap_potential, 1),
        ev_best=round(info.ev_best, 1), ev_chosen=round(info.ev_chosen, 1),
        oracle=info.oracle, bypass=info.bypass, fen=board.fen(),
    ))
    _emit(f"bestmove {move.uci()}")


if __name__ == "__main__":
    main()
