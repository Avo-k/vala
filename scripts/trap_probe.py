"""End-to-end probe of vala's bot pipeline (parallel Patricia + trigger).

Runs `vala_move` on curated positions at several human Elos and prints, per
position: whether the trigger screen fired, whether vala deviated to a bait, the
EV decomposition, engine/Maia call counts, and wall-clock — the headline being
that parallel Patricia brings depth-2 from ~11 s (serial) down to ~1 s on CPU.

Run:  .venv/bin/python scripts/trap_probe.py --profile balanced --maia maia3-5m
      .venv/bin/python scripts/trap_probe.py --pool 16 --elos 1100,1500,2000
"""
from __future__ import annotations

import argparse
import time

import chess

from vala.bot import PROFILES, vala_move
from vala.maia import make_predictor
from vala.pool import PatriciaPool, default_pool_size


def from_moves(s: str) -> chess.Board:
    b = chess.Board()
    for tok in s.split():
        b.push_san(tok)
    return b


def make_positions() -> list[tuple[str, chess.Board, str]]:
    return [
        ("startpos", chess.Board(), "no trap; expect solid, not even triggered"),
        ("blackburne-shilling", from_moves("e4 e5 Nf3 Nc6 Bc4"),
         "Black to move; ...Nd4 baits 4.Nxe5?? (multi-ply swindle)"),
        ("italian-quiet", from_moves("e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3"),
         "quiet Italian; expect solid"),
        ("scholars-defense", from_moves("e4 e5 Bc4 Nc6 Qh5"),
         "Black to move; Qh5 attack — any baitable lines?"),
        ("KQvKR", chess.Board("8/8/8/4k3/8/8/4r3/4K2Q w - - 0 1"),
         "Q vs R conversion"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maia", default="maia3-5m")
    ap.add_argument("--profile", default="balanced", choices=list(PROFILES))
    ap.add_argument("--elos", default="1100,1500,2000")
    ap.add_argument("--pool", type=int, default=default_pool_size())
    ap.add_argument("--cand-ms", type=int, default=200)
    ap.add_argument("--leaf-ms", type=int, default=50)
    args = ap.parse_args()

    elos = [int(x) for x in args.elos.split(",")]
    print(f"loading Patricia pool (n={args.pool}, cpu) + {args.maia} (cpu) ...")
    maia = make_predictor(args.maia, device="cpu")
    positions = make_positions()

    fired = {e: 0 for e in elos}
    dev = {e: 0 for e in elos}
    tsum = {e: 0.0 for e in elos}
    n = len(positions)

    with PatriciaPool(n=args.pool) as pool:
        for label, board, note in positions:
            print(f"\n{'='*82}\n{label}  ({'white' if board.turn else 'black'} to move) — {note}")
            for elo in elos:
                t0 = time.perf_counter()
                move, info = vala_move(board, pool, maia, profile=args.profile,
                                       human_elo=elo, cand_ms=args.cand_ms, leaf_ms=args.leaf_ms)
                dt = time.perf_counter() - t0
                tsum[elo] += dt
                if info.triggered:
                    fired[elo] += 1
                if info.deviated:
                    dev[elo] += 1
                tag = "BAIT" if info.deviated else ("trig" if info.triggered else "solid")
                if info.bypass:
                    tag = info.bypass
                print(f"  elo {elo}: [{tag:5s}] play={board.san(move):7s} "
                      f"loss={info.eval_loss:+4d} EV {info.ev_best:+.0f}->{info.ev_chosen:+.0f} "
                      f"pot={info.trap_potential:4.0f}cp oracle={info.oracle} "
                      f"calls={info.n_engine_calls}e/{info.n_maia_calls}m  {dt*1000:.0f}ms")

    print(f"\n{'='*82}\nSUMMARY  ({n} positions, profile={args.profile}, pool={args.pool})")
    for elo in elos:
        print(f"  elo {elo}: triggered {fired[elo]}/{n}  deviated {dev[elo]}/{n}  "
              f"mean {tsum[elo]/n*1000:.0f}ms/move")


if __name__ == "__main__":
    main()
