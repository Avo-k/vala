"""Self-play: vala vs a sampling-Maia "human".

The opponent plays moves *sampled* from Maia's distribution at `--opp-elo` (a
crude human stand-in). This is biased — vala's own opponent model is Maia, so
vs sampling-Maia vala has a near-perfect model and baiting is closer to an upper
bound than to real-world performance. Two ways the script pushes back on that:

  * It compares two vala conditions against the *same* opponent — a baiting
    profile vs `best` (engine-best, no baiting) — so the delta isolates what
    baiting buys, not how strong Patricia is.
  * `--vala-elo` (what vala assumes) can differ from `--opp-elo` (what the
    opponent actually is) to probe the "model is wrong" / spice regime.

Reports W/D/L, score, and implied Elo (maia_elo + 400·log10(s/(1−s))) per
condition, plus how often vala actually deviated to a bait.

Run:  .venv/bin/python scripts/play_vs_maia.py --games 12 --opp-elo 1200 --modes balanced,best
"""
from __future__ import annotations

import argparse
import math
import random
import time

import chess

from vala.bot import vala_move
from vala.maia import make_predictor
from vala.pool import PatriciaPool, default_pool_size


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r, or None if undefined (n<2 or a constant series)."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx * syy) ** 0.5


def implied_elo_diff(score: float) -> float:
    if score >= 0.999:
        return 800.0
    if score <= 0.001:
        return -800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def _vala_pov_cp(board, pool, vala_white, ms):
    """Realized eval from vala's POV (handles terminal positions)."""
    if board.is_game_over(claim_draw=True):
        o = board.outcome(claim_draw=True)
        if o is None or o.winner is None:
            return 0.0
        return 9000.0 if (o.winner == vala_white) else -9000.0
    cp = pool.map_eval([board], ms)[0]
    if cp is None:
        return 0.0
    return float(cp) if (board.turn == vala_white) else float(-cp)


def play_game(pool, maia, *, vala_white, mode, profile, vala_elo, opp_elo,
              temperature, rng, cand_ms, leaf_ms, max_plies, realized_ms=80,
              allie=None, allie_elo=None):
    board = chess.Board()
    plies = baits = 0
    bait_nets: list[float] = []   # realized eval gain (vala POV) per bait, vs pre-move best
    allie_recs: list[dict] = []   # per-bait Allie signal + realized outcome (when allie given)
    pending = None                # {best, ao} for a bait awaiting the opponent's reply
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        vala_to_move = (board.turn == chess.WHITE) == vala_white
        just_baited = None
        if vala_to_move:
            if mode == "best":
                move = pool.multipv(board, k=1, time_ms=cand_ms)[0].move
            else:
                move, info = vala_move(
                    board, pool, maia, profile=profile, human_elo=vala_elo,
                    cand_ms=cand_ms, leaf_ms=leaf_ms, vala_ms=leaf_ms,
                )
                if info.deviated:
                    baits += 1
                    just_baited = float(info.best_cp)  # vala POV, before the bait
        else:
            move = maia.sample(board, opp_elo, opp_elo, rng=rng, temperature=temperature)
            if move is None:
                break
        board.push(move)
        plies += 1
        # A bait was just played → board is now the human-to-move position. Snapshot
        # Allie's read of that position (think-time, value, full reply distribution).
        if just_baited is not None:
            ao = None
            if allie is not None and not board.is_game_over(claim_draw=True):
                ao = allie.predict(board, elo=allie_elo)
            pending = {"best": just_baited, "ao": ao}
        # After the opponent has replied to a bait, measure realized eval and bind it
        # to the Allie snapshot taken when the bait was set.
        elif pending is not None and not vala_to_move:
            realized = _vala_pov_cp(board, pool, vala_white, realized_ms)
            net = min(realized, 2000.0) - pending["best"]  # clamp mate-y leaves
            bait_nets.append(net)
            ao = pending["ao"]
            if ao is not None:
                actual = move.uci()
                top_uci, top_p = (max(ao.move_probs.items(), key=lambda kv: kv[1])
                                  if ao.move_probs else (None, 0.0))
                allie_recs.append(dict(
                    net=net,
                    think_time=ao.think_time,
                    value_vala=ao.value if vala_white else -ao.value,  # vala-POV
                    p_actual=ao.move_probs.get(actual, 0.0),  # Allie's P on the played reply
                    top_uci=top_uci, top_p=top_p, actual=actual,
                    matched=(top_uci == actual),
                ))
            pending = None

    res = board.result(claim_draw=True)
    if res == "1-0":
        s = 1.0 if vala_white else 0.0
    elif res == "0-1":
        s = 0.0 if vala_white else 1.0
    else:
        s = 0.5
    return s, plies, baits, bait_nets, allie_recs


def _allie_summary(recs: list[dict]) -> dict | None:
    """Aggregate Allie's per-bait signal vs realized outcome — the step-1 question:
    do Allie's think-time / value / reply-probability predict bait success?"""
    if not recs:
        return None
    nets = [r["net"] for r in recs]
    tts = [r["think_time"] for r in recs]
    vs = [r["value_vala"] for r in recs]
    pa = [r["p_actual"] for r in recs]
    sprung = [r for r in recs if r["net"] >= 80]
    flat = [r for r in recs if r["net"] < 80]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return dict(
        n=len(recs),
        match_rate=mean([1.0 if r["matched"] else 0.0 for r in recs]),  # Allie top == actual reply
        mean_p_actual=mean(pa),            # Allie's avg prob on the move actually played
        r_tt_net=_pearson(tts, nets),      # think-time vs realized net
        r_val_net=_pearson(vs, nets),      # predicted (vala-POV) value vs realized net
        r_pact_net=_pearson(pa, nets),     # P(actual reply) vs realized net
        tt_sprung=mean([r["think_time"] for r in sprung]),
        tt_flat=mean([r["think_time"] for r in flat]),
        pact_sprung=mean([r["p_actual"] for r in sprung]),
        pact_flat=mean([r["p_actual"] for r in flat]),
        n_sprung=len(sprung),
    )


def run_condition(label, pool, maia, *, profile, mode, n_games, opp_elo, vala_elo,
                  temperature, seed, cand_ms, leaf_ms, max_plies, allie=None, allie_elo=None):
    w = d = l = 0
    baits_total = plies_total = 0
    all_nets: list[float] = []
    all_recs: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n_games):
        rng = random.Random(seed + i)  # same opponent stream per game index across conditions
        s, plies, baits, nets, recs = play_game(
            pool, maia, vala_white=(i % 2 == 0), mode=mode, profile=profile,
            vala_elo=vala_elo, opp_elo=opp_elo, temperature=temperature, rng=rng,
            cand_ms=cand_ms, leaf_ms=leaf_ms, max_plies=max_plies,
            allie=allie, allie_elo=allie_elo,
        )
        w += int(s == 1.0); d += int(s == 0.5); l += int(s == 0.0)
        baits_total += baits; plies_total += plies; all_nets += nets; all_recs += recs
        print(f"  [{label}] game {i+1:2d}/{n_games} "
              f"({'W' if i%2==0 else 'B'}): {'win' if s==1 else 'draw' if s==0.5 else 'loss':4s} "
              f"({plies} plies, {baits} baits)")
    n = w + d + l
    score = (w + 0.5 * d) / n if n else 0.0
    implied = opp_elo + implied_elo_diff(score)
    dt = time.perf_counter() - t0
    # Realized-bait stats: did the opponent actually walk into the trap?
    sprang = sum(1 for x in all_nets if x >= 80)
    backfired = sum(1 for x in all_nets if x <= -50)
    mean_net = sum(all_nets) / len(all_nets) if all_nets else 0.0
    return dict(label=label, w=w, d=d, l=l, score=score, implied=implied,
                baits=baits_total, plies=plies_total, secs=dt,
                n_nets=len(all_nets), sprang=sprang, backfired=backfired, mean_net=mean_net,
                allie=_allie_summary(all_recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--modes", default="balanced,best",
                    help="comma list; a profile name baits, 'best' = engine-best baseline")
    ap.add_argument("--opp-elo", type=int, default=1200)
    ap.add_argument("--vala-elo", type=int, default=None, help="what vala assumes (default = opp-elo)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temps", default=None,
                    help="comma list of opponent temperatures to sweep (default: just --temperature)")
    ap.add_argument("--maia", default="maia3-5m")
    ap.add_argument("--pool", type=int, default=default_pool_size())
    ap.add_argument("--engine-elo", type=int, default=None,
                    help="handicap vala's Patricia to this UCI_Elo (match the opponent's level)")
    ap.add_argument("--cand-ms", type=int, default=120)
    ap.add_argument("--leaf-ms", type=int, default=35)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--allie", nargs="?", const="models/ablations/small", default=None,
                    help="enable Allie diagnostics on baits; optional checkpoint dir "
                         "(default models/ablations/small, or env ALLIE_MODEL_DIR)")
    args = ap.parse_args()

    vala_elo = args.vala_elo if args.vala_elo is not None else args.opp_elo
    modes = [m.strip() for m in args.modes.split(",")]
    temps = [float(x) for x in args.temps.split(",")] if args.temps else [args.temperature]
    print(f"loading Patricia pool (n={args.pool}) + {args.maia} (cpu) ...")
    print(f"opponent = sampling-Maia@{args.opp_elo} (T sweep={temps}); "
          f"vala models opponent@{vala_elo}; {args.games} games/condition")
    if args.engine_elo:
        print(f"vala's Patricia handicapped to UCI_Elo {args.engine_elo}")
    maia = make_predictor(args.maia, device="cpu")

    allie = None
    if args.allie is not None:
        import os
        from vala.allie import AlliePredictor
        ckpt = os.environ.get("ALLIE_MODEL_DIR", args.allie)
        print(f"loading Allie diagnostics from {ckpt} (cpu); modeling opponent@{args.opp_elo}")
        allie = AlliePredictor(ckpt, device="cpu")

    results = []
    with PatriciaPool(n=args.pool, uci_elo=args.engine_elo) as pool:
        for T in temps:
            for mode in modes:
                label = "best" if mode == "best" else mode
                print(f"\n=== T={T}  condition: {label} ===")
                r = run_condition(
                    label, pool, maia, profile=args.profile, mode=mode,
                    n_games=args.games, opp_elo=args.opp_elo, vala_elo=vala_elo,
                    temperature=T, seed=args.seed,
                    cand_ms=args.cand_ms, leaf_ms=args.leaf_ms, max_plies=args.max_plies,
                    allie=allie, allie_elo=args.opp_elo,
                )
                r["temp"] = T
                results.append(r)

    print(f"\n{'='*88}\nSUMMARY  (opp=Maia@{args.opp_elo}, vala-model@{vala_elo}, "
          f"engine_elo={args.engine_elo}, {args.games} games each)")
    print(f"{'T':>5s} {'cond':10s} {'W/D/L':>8s} {'score':>6s} {'plies':>6s} "
          f"{'baits':>6s} {'sprang':>11s} {'backfire':>9s} {'meanNet':>8s}")
    for r in results:
        wdl = f"{r['w']}/{r['d']}/{r['l']}"
        avg_plies = r['plies'] / max(1, r['w'] + r['d'] + r['l'])
        nn = r['n_nets']
        sprang = f"{r['sprang']}/{nn}" if nn else "-"
        bf = f"{r['backfired']}/{nn}" if nn else "-"
        print(f"{r['temp']:>5.2f} {r['label']:10s} {wdl:>8s} {r['score']:>6.2f} {avg_plies:>6.0f} "
              f"{r['baits']:>6d} {sprang:>11s} {bf:>9s} {r['mean_net']:>+8.0f}")
    print("\nRealized-bait read [per bait: eval gain (vala POV) vs the pre-move best]:")
    print("  sprang = opponent gave back ≥80cp (walked into it);  backfire = we lost ≥50cp;")
    print("  meanNet > 0 ⇒ baiting pays in realized eval against this opponent.")

    if any(r.get("allie") for r in results):
        print(f"\n{'='*88}\nALLIE SIGNAL vs bait outcome  (does Allie's read predict whether a bait works?)")
        print("  r_* = Pearson corr with realized net; match = Allie top-reply == opponent's actual reply;")
        print("  p_act = Allie's prob on the played reply.  Hypotheses: traps spring more when Allie")
        print("  predicts LOW think-time and/or HIGH prob on the (blundering) reply.")
        print(f"{'T':>5s} {'cond':10s} {'baits':>6s} {'match':>6s} {'p_act':>6s} "
              f"{'r(tt,net)':>10s} {'r(val,net)':>11s} {'r(pact,net)':>12s} "
              f"{'tt:spr/flat':>13s} {'pact:spr/flat':>14s}")

        def _f(x, w=6, p=2):
            return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) and x == x else f"{'-':>{w}}"

        for r in results:
            a = r.get("allie")
            if not a:
                continue
            print(f"{r['temp']:>5.2f} {r['label']:10s} {a['n']:>6d} "
                  f"{_f(a['match_rate'])} {_f(a['mean_p_actual'])} "
                  f"{_f(a['r_tt_net'],10):>10s} {_f(a['r_val_net'],11):>11s} {_f(a['r_pact_net'],12):>12s} "
                  f"{_f(a['tt_sprung'],5,1)}/{_f(a['tt_flat'],4,1).strip():<4s} "
                  f"{_f(a['pact_sprung'],6)}/{_f(a['pact_flat'],5).strip():<5s}")


if __name__ == "__main__":
    main()
