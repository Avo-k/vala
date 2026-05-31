# vala — first-setup feasibility findings (2026-05-31)

Hardware: RTX 4090 (24 GB), 32-core CPU, 49 GB RAM. Patricia 5.0 (vendored),
Maia-3 (5M/23M/79M) via `maia3` package. All numbers from `scripts/bench_maia.py`
and `scripts/trap_probe.py`.

## 1. Patricia 5.0 candidate generation
- MultiPV up to 255; `UCI_LimitStrength`, `Skill_Level` (1–21), `UCI_Elo`
  (500–3001), Syzygy. Single-threaded NNUE.
- At 200 ms / MultiPV 5 it reaches depth 13–14. Good enough as a candidate
  generator and leaf evaluator.

## 2. Maia-3 latency & throughput (the scaling crux)
Per-position cost, single (batch=1) and batched. **Key result: Maia is *not* the
bottleneck, and the GPU barely helps** — the models are tiny (5–79 M params), so
per-forward Python work (board tokenization + 8-ply history reconstruction +
host↔device transfer) dominates, not the matmuls. The 4090 is idle.

| model | params | CPU single | CPU batched | GPU single | GPU batched (peak) |
|---|---|---|---|---|---|
| maia3-5m  | 5.2 M  | 11.8 ms (85/s)  | 3.0 ms (335/s) | 10.2 ms (98/s) | 1.9 ms (**526/s**) |
| maia3-23m | 22.9 M | 12.2 ms (82/s)  | 5.4 ms (184/s) | 15.1 ms (66/s) | 2.2 ms (446/s) |
| maia3-79m | 78.9 M | 24.3 ms (41/s)  | 13.6 ms (74/s) | 22.6 ms (44/s) | 2.3 ms (**436/s**) |

Consequences:
- **GPU batched throughput is nearly model-size-independent** (~2 ms/pos for all
  three) — overhead-bound. So the accuracy-vs-speed tradeoff between 5M and 79M
  *vanishes* once you batch on GPU. Prefer the **most accurate** model.
- On CPU the 79M is ~2× the 5M single-shot but still only 24 ms — affordable.
- The real per-move bottleneck is **Patricia** (UCI subprocess, serial,
  ~40–50 ms/search), not Maia.

## 3. Proof of concept: does the swindle mechanism work?
`scripts/trap_probe.py` runs the depth-`H` expectimax selector. Headline result
on the **Blackburne Shilling** position (1.e4 e5 2.Nf3 Nc6 3.Bc4, Black to move;
the bait is the dubious ...Nd4, hoping for 4.Nxe5?? Qg5), human Elo 1100:

| `human_depth` | move chosen | EV(Nd4) | EV(Nf6, sound) | engine calls | wall-clock |
|---|---|---|---|---|---|
| 1 | Nf6 (solid) | −39.6 (worst) | −3.9 (best) | 54 | 2.7 s |
| **2** | **Nd4 (BAIT)** | **+46.1 (best)** | +2.2 | 246 | 11 s |
| 3 | Nf6 | (off top-6) | +32.5 (best) | 1014 | 46 s |

**The single most important finding:** depth-1 expectimax *cannot* see real
swindles. After ...Nd4 4.Nxe5 (which Maia rates ~31% at 1100!), Patricia
evaluates the position at only **−6 cp** — because it assumes White then finds
the saving 5.Nxf7!. The Blackburne trap only pays if the human errs *twice in a
row*. A depth-1 rollout assumes perfect play after the human's first reply and
therefore prices the trap at zero. **Depth-2 (two human plies) is the minimum
that captures multi-ply swindles**, and it flips Nd4 from the worst candidate
(−40) to the best (+46), making vala spring the trap.

At depth-3 the picture shifts again (the rollout sees the human can recover, or
finds longer error chains in other lines) — a reminder that deeper ≠ strictly
better here; depth interacts with how far the human's error streak realistically
extends. Worth a sweep.

## 4. The scaling wall
Engine calls grow as ≈ `(top_replies)^human_depth`:
54 → 246 → 1014 for H = 1 → 2 → 3 (top_replies=4). At ~45 ms/call **serial
Patricia**, that is 2.7 s → 11 s → 46 s per move. Levers:
1. **Parallelize Patricia** across the 32 cores (pool of ~16 instances) → ~16×.
2. **Cheaper leaves/internal nodes:** Maia value head for leaves (free, batched
   on GPU) or much shorter Patricia movetimes inside the tree.
3. **Selective deepening:** only expand human replies that are *plausible
   blunders* (high Maia P **and** large eval drop); prune branches where the
   human's likely move is also the engine's best (no trap there).
4. **Trigger the deep search only "de temps en temps":** a cheap depth-1 / entropy
   screen flags candidate trap positions; pay for depth-2/3 only there. This is
   exactly the user's framing — solid by default, occasional short tree dives.

## 5. Verdict (short)
The idea is **sound and demonstrated**: vala provably picks a real swindle when
one exists (depth-2) and stays solid otherwise. It has clear precedent (poker
miximax / Vexbot; Ahle's chess-openings-expectimax) and a clean formalism
(best-response / Restricted Nash). Feasibility hinges on the **scaling wall**
(serial Patricia × exponential depth), addressable by the levers above, and on
**Maia tail calibration** + **robustness when the opponent deviates from the
model** (the literature's documented failure mode). Use the largest Maia that
fits the latency budget (79M on GPU is free relative to 5M); keep `risk_cp` as
the safety dial.

## 6. Update — parallel Patricia + trigger architecture (same day)

Built the `PatriciaPool` (N=16 subprocesses), rewrote the search as
level-synchronous (one batched Maia forward + one parallel Patricia wave per
ply), and added a cheap trigger screen. Deploy target is **CPU**; all numbers
below are CPU, `maia3-5m`, pool=16, `balanced` profile.

**The scaling wall fell.** A triggered depth-2 move: ~**1.0–1.6 s** (≈270 engine
calls run in parallel waves) vs ~11 s serial earlier — roughly the pool-size
speedup. Quiet moves never enter the deep search: ~**0.3 s** (MultiPV + screen).

**The trigger works and is Elo-selective** (`screen_trap_potential` = depth-1
expected upside `Σ_r P(r)·max(0, our_cp(r) − sound)`):

| position | elo 1100 | elo 1500 | elo 2000 |
|---|---|---|---|
| Blackburne (...Nd4 bait) | **BAIT ...Nd4** | **BAIT ...f5** | solid Nf6 |
| scholar's defense | triggered→solid | **BAIT ...Qe7** | solid g6 |
| quiet Italian / startpos | solid | solid | solid |
| mean ms/move (5 pos) | ~690 | ~780 | ~324 |

Two design points confirmed: (a) the two-stage screen→search split works — a
position can *trigger* the deep search yet the search still *arbitrates* for the
solid move (scholar's @1100); (b) strong opponents (2000) get no baits — vala
plays straight, exactly the intended Elo behavior.

**Known limitation:** the trigger is depth-1. The expected-upside proxy catches
Blackburne-type swindles because their first ply already shows opponent
imperfection mass, but a swindle whose first ply looks perfectly sound could slip
past the screen even though `ev_select` would find it at depth-2. Deepen the
screen or lower `trigger_cp` if recall matters.

**Lichess Explorer wired** (`vala/explorer.py`, crediting Ahle): real
rating-filtered human frequencies as the opponent model at shallow/opening nodes,
Maia in the tree interior (network latency rules out per-node calls). Needs
`LICHESS_TOKEN`.

## 7. UCI + lichess-bot + self-play (next day)

Shipped the UCI shim (`vala/uci.py`, entry point `vala-uci`), per-move time
management (`timectl.py`: clock → movetimes + depth cap; bullet → fast-best),
lichess-bot config, and a self-play harness (`scripts/play_vs_maia.py`) pitting a
baiting profile against the `best` (engine-best) baseline vs a *sampling-Maia*
opponent, comparing both against the same opponent stream.

**Self-play result — the metric saturates, and it's instructive.**

| vala | opponent | baiting (balanced) | baseline (best) | Δscore |
|---|---|---|---|---|
| full Patricia | Maia@1200 (T=1.0) | 10/10 (100%) | 10/10 (100%) | +0.000 |
| Patricia @UCI_Elo 1500 | Maia@1500 (T=1.0) | 12/12 (100%) | 12/12 (100%) | +0.000 |

Both saturate at 100%. Two compounding causes:
1. **A *sampling*-Maia plays far below its nominal Elo.** Sampling the move
   distribution every ply accumulates imprecision, so Maia@1500 (T=1.0) behaves
   like a much weaker, erratic player and loses ~every game to anything sound.
2. **Handicapping vala via `UCI_Elo` doesn't help measure baiting** — it weakens
   the *same* Patricia that does the leaf evals, so trap detection degrades along
   with playing strength (the pool is shared). Even at UCI_Elo 1500 vala still
   won 100%.

Baiting demonstrably **fires** (53 baits across the 12 balanced games, ~4.4/game)
and **never hurts** (still 100%), but in this matchup we cannot show it *helps*.
Plies-to-win is no rescue: balanced mean 44.5 / median 47 vs best mean 48.0 /
median 39.5 — means lean baiting-faster, medians lean the other way; too noisy at
N=12 (best has huge variance: 17-ply wins alongside 86–90-ply grinds).

**Why it's hard, and the fix.** Baiting pays only against opponents that (a) err
sometimes yet (b) are strong enough that vala doesn't auto-win — a mid band that
sampling-Maia@T=1.0 overshoots (errs constantly, loses always). The clean
experiment needs to **decouple strength from trap-detection accuracy**: a
*strong* "analyst" Patricia for leaf evals (accurate "is this a blunder?") and a
*limited* "player" Patricia for candidate generation and the vala-node moves
inside the rollout (so vala plays at a matched rating). Plus a realistic opponent
— sampling-Maia at low temperature, or better, **real humans on Lichess**. The
honest read: the mechanism is correct and safe (never backfired here), but its
*value* must be measured in a strength-matched, realistic-blunder regime, which
this harness doesn't yet create.
