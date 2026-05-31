# vala

A chess engine that plays **solid moves**, but occasionally plays a deliberately
suboptimal **bait** when a short search predicts the human opponent is likely to
blunder in reply — so the bait has higher *expected value* than the objectively
best move. It composes a strong engine (Patricia 5) with a human-move model
(Maia-3) used as an **opponent model** inside a depth-limited expectimax.

Sister project to **[rorschach](https://github.com/Avo-k/rorschach)**, which
uses Maia as a *negative filter* to play sound-but-alien moves. vala inverts the
sign: it uses Maia as the opponent's *reply distribution* and bets on human
error. *(In Tolkien, the Valar command the Maiar; here vala wraps Maia.)*

## Core idea (miximax / best-response to an opponent model)

At each node where the human moves, their reply is the Maia distribution
`P(r | s, elo)`; at each node where vala moves, vala plays its own best move. The
tree is expanded for `human_depth` human plies, then scored with the engine.
Values are centipawns from vala's point of view:

```
EV(m) = Σ_r P_Maia(r | s', elo)·(−cp_opp(r))  +  (1 − covered)·sound_cp
```

vala branches over Patricia's MultiPV candidates within an objective-risk budget
of the best move and plays `argmax EV`, deviating from engine-best only when a
bait beats it by a noise margin.

- **`risk_cp`** — max objective eval sacrificed at best opposing play (downside
  cap). The safety dial: lower ⇒ closer to minimax ⇒ more robust, less swindle.
- **`margin_cp`** — EV uplift a bait must beat engine-best by before deviating.
- **`human_depth`** — human plies of rollout. **Must be ≥ 2**: real swindles need
  the human to err more than once, and a 1-ply rollout assumes perfect play after
  the first reply (see `docs/FINDINGS.md`).

## Pipeline (solid by default, occasional trap)

`vala/bot.py::vala_move` runs: **(1)** Patricia MultiPV → candidates; **(2)** a
cheap depth-1 *trigger screen* (expected opponent-imperfection upside) — most
positions are quiet, cost ~0.3 s, play engine-best; **(3)** only on a flagged
position, the deeper level-synchronous expectimax, which may deviate to a bait.

## Components

| Layer | Tool | Role |
|---|---|---|
| Strong engine | **Patricia 5** (UCI) | MultiPV candidates + cp leaf evals. Single-threaded NNUE, MIT. Vendored at `bin/patricia`. |
| Parallel engine | `vala/pool.py` | N Patricia subprocesses; runs a whole search level concurrently on CPU cores. |
| Opponent model | **Maia-3 / Chessformer** (PyTorch) | `P(reply | pos, elo)` over legal moves. Batched forward (`predict_batch`) for tree fan-out. |
| Opening oracle | **[Lichess Opening Explorer](https://lichess.org/api#tag/Opening-Explorer)** | *real* human reply frequencies, rating-filtered, at shallow nodes — the live-API realization of [Thomas Ahle's chess-openings-expectimax](https://github.com/thomasahle/chess-openings-expectimax) (credited). Maia in the tree interior. |
| Selector + bot | `vala/search.py`, `vala/bot.py` | level-sync EV / best-response + the trigger and calibration profiles. |

## Status — parallel + trigger architecture (2026-05-31)

**Working, on CPU (the deploy target):** vala springs the Blackburne Shilling
trap at `human_depth=2` (low-Elo opponents), plays solid in quiet positions and
vs high-Elo opponents, and the `PatriciaPool` brought a triggered depth-2 move
from ~11 s (serial) to **~1–1.6 s**; quiet moves are ~0.3 s. See
`docs/FINDINGS.md` for numbers and `docs/research-brief.md` for prior art (poker
miximax / Vexbot; Ahle) and the documented failure modes. 6 tests green.

**Not yet built:** UCI shim (`vala/uci.py`) + lichess-bot wiring, per-move time
management, self-play Elo harness, Maia tail-calibration check.

## Quick start

```bash
uv sync                          # maia3 from git (checkpoint downloads on first use)
uv run pytest -m "not slow"      # fast logic tests
uv run pytest -m slow            # real Patricia+Maia: Blackburne trap regression (~4s)

# end-to-end pipeline probe (CPU, pool of 16 Patricia)
uv run python scripts/trap_probe.py --profile balanced --maia maia3-5m --pool 16
# Maia latency/throughput
uv run python scripts/bench_maia.py --device cpu --model maia3-5m

# one move over UCI
printf 'uci\nsetoption name Elo value 1100\nisready\nposition startpos moves e2e4 e7e5 g1f3 b8c6 f1c4\ngo movetime 2000\nquit\n' | uv run vala-uci
```

## Lichess deployment

`vala-uci` is a normal UCI engine (lazy-loads the Patricia pool + Maia on first
`isready`). To run as a Lichess bot:

```bash
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot && pip install -r requirements.txt
cp ../vala/configs/lichess-bot.yml.example config.yml   # edit the BOT token + paths
python lichess-bot.py
```

The account must be a registered **BOT** account — vala deliberately sets traps,
which is fine for a BOT but an instant ban on a human login. Set `Pool` to the
VM's vCPU count. Per-move time management derives search depth/movetimes from the
clock; bullet budgets fall back to a fast sound move.

### UCI options

| Option | Type | Values | Default | Meaning |
|---|---|---|---|---|
| `Profile` | combo | `solid`/`balanced`/`aggressive` | `balanced` | the spice dial (`risk_cp`, depth) |
| `Elo` | spin | 0 = track opponent, or 600..2600 | 0 | rating vala models the human at |
| `MaiaType` | combo | `maia3-5m`/`-23m`/`-79m` | `maia3-5m` | opponent model size |
| `Pool` | spin | 1..64 | ~cores−2 | parallel Patricia subprocesses |
| `TimeMs` | spin | 0 = use clock, else ms | 0 | fixed per-move budget override |
| `UseExplorer` | check | true/false | false | real Lichess opening frequencies (needs `LICHESS_TOKEN`) |

Per-move `info ... string` line reports the tag (`solid`/`trig`/`BAIT`/`mate`),
eval loss, `EV` before→after, trigger potential, modeled Elo, oracle, call counts.

## Prior art

- **Opponent Modelling and Search in Poker** (Vexbot, miximax) — [Billings thesis](https://poker.cs.ualberta.ca/publications/billings.phd.pdf)
- **chess-openings-expectimax** — [Thomas Ahle](https://github.com/thomasahle/chess-openings-expectimax) (expectimax over human opening frequencies + Stockfish leaves)
- **Maia** — [KDD 2020](https://www.cs.toronto.edu/~ashton/pubs/maia-kdd2020.pdf) · [maia3](https://github.com/CSSLab/maia3)
- **Restricted Nash Response** — [Johanson et al., NIPS 2007](https://arxiv.org/pdf/1603.03491) (the exploit-vs-safety dial)
- **Safe Opponent Exploitation** — [Ganzfried & Sandholm](https://www.cs.cmu.edu/~sandholm/safeExploitation.teac15.pdf)
- **Patricia** — [github](https://github.com/Adam-Kulju/Patricia)
