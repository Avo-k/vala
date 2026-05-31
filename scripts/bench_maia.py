"""Maia-3 latency + batched-throughput benchmark.

The decisive feasibility measurement for vala: vala's expectimax evaluates the
opponent's reply distribution with Maia at every node where the opponent moves.
Cost is dominated by Maia forward passes. Maia-3 batches natively (the model
takes (B, 64, dim) token tensors + (B,) elo tensors), so the whole question of
"can the tree search scale" reduces to "how many positions/sec can we push
through Maia at an acceptable per-move budget".

Run (CPU, via rorschach's installed env):
    /home/avok/code/rorschach/.venv/bin/python scripts/bench_maia.py --device cpu --model maia3-5m

Run (GPU, via vala's CUDA env):
    .venv/bin/python scripts/bench_maia.py --device cuda --model maia3-5m --model maia3-23m --model maia3-79m
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import deque
from types import SimpleNamespace

import chess


def build_predictor(alias: str, device: str):
    """Load a Maia-3 model + the helpers needed for batched inference.

    Mirrors rorschach's Maia3Predictor.__init__ but exposes a batched forward.
    """
    import torch
    from maia3.dataset import (
        get_historical_tokens,
        get_legal_moves_mask,
        tokenize_board,
    )
    from maia3.model_registry import (
        apply_model_config,
        resolve_checkpoint_path,
        resolve_model_spec,
    )
    from maia3.models import MAIA3Model
    from maia3.utils import get_all_possible_moves, mirror_move

    spec = resolve_model_spec(alias)
    cfg = SimpleNamespace(
        device=device,
        use_amp=False,
        trust_checkpoint=False,
        history=8, use_padding=True, include_time_info=False,
        dim_emb=128, dim_vit=192, num_blocks=8, num_heads=6, mlp_ratio=2.0,
        dropout=0.0, head_hid_dim=192,
        use_gab=True, gab_gen_size=64, gab_per_square_dim=0,
        gab_intermediate_dim=64, use_rms_norm=True, omit_qkv_biases=True,
        activation="gelu",
        use_relative_bias=False, use_absolute_pe=False,
    )
    apply_model_config(cfg, spec)
    cfg.checkpoint_path = resolve_checkpoint_path(spec)

    model = MAIA3Model(cfg).to(device)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    renamed = {k.replace("smolgen", "gab"): v for k, v in state_dict.items()}
    model.load_state_dict(renamed, strict=False)
    model.eval()

    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}

    return SimpleNamespace(
        torch=torch, model=model, cfg=cfg,
        tokenize_board=tokenize_board,
        get_historical_tokens=get_historical_tokens,
        get_legal_moves_mask=get_legal_moves_mask,
        mirror_move=mirror_move,
        all_moves_dict=all_moves_dict,
        n_params=sum(p.numel() for p in model.parameters()),
    )


def build_history(P, board: chess.Board) -> deque:
    moves = list(board.move_stack)
    history: deque = deque(maxlen=P.cfg.history)
    if not moves:
        history.append(P.tokenize_board(board))
        return history
    replay = board.copy()
    for _ in range(len(moves)):
        replay.pop()
    history.append(P.tokenize_board(replay))
    for mv in moves:
        replay.push(mv)
        history.append(P.tokenize_board(replay))
    return history


def forward_batch(P, boards, elos_self, elos_oppo):
    """One batched forward over `boards`. Returns list of (move_probs, win_prob)."""
    torch = P.torch
    cfg = P.cfg
    with torch.no_grad():
        tok_list, mask_list = [], []
        for b in boards:
            hist = build_history(P, b)
            t = P.get_historical_tokens(hist, cfg, base=0.0, inc=0.0,
                                        clk_left_before=0.0, clk_ponder=0.0)
            tok_list.append(t)
            mask_list.append(P.get_legal_moves_mask(b, P.all_moves_dict))
        tokens = torch.stack(tok_list, dim=0).to(cfg.device)
        masks = torch.stack(mask_list, dim=0).to(cfg.device)
        self_e = torch.tensor(elos_self, dtype=torch.long, device=cfg.device)
        oppo_e = torch.tensor(elos_oppo, dtype=torch.long, device=cfg.device)

        logits_move, logits_value, _ = P.model(tokens, self_e, oppo_e)
        logits = logits_move.float().masked_fill(~masks, float("-inf"))
        probs = torch.softmax(logits, dim=-1).cpu()
        vals = torch.softmax(logits_value.float(), dim=-1).cpu()

        out = []
        for i, b in enumerate(boards):
            d = {}
            for move in b.legal_moves:
                uci = move.uci()
                key = uci if b.turn == chess.WHITE else P.mirror_move(uci)
                idx = P.all_moves_dict.get(key)
                if idx is not None:
                    d[uci] = float(probs[i, idx])
            out.append((d, float(vals[i, 2])))
        return out


def make_positions(n: int, rng: random.Random) -> list[chess.Board]:
    """Generate `n` varied mid-game positions with real move_stacks (so history
    reconstruction does real work, like in a live game)."""
    boards = []
    while len(boards) < n:
        b = chess.Board()
        plies = rng.randint(6, 30)
        ok = True
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                ok = False
                break
            b.push(rng.choice(moves))
        if ok and not b.is_game_over():
            boards.append(b)
    return boards


def bench_model(alias: str, device: str, batch_sizes, warmup: int, iters: int, rng):
    print(f"\n{'='*64}\n  {alias}  on  {device}\n{'='*64}")
    t0 = time.perf_counter()
    P = build_predictor(alias, device)
    load_s = time.perf_counter() - t0
    print(f"  params: {P.n_params/1e6:.1f}M   load: {load_s:.1f}s")

    pool = make_positions(max(batch_sizes) + 8, rng)
    sync = (lambda: P.torch.cuda.synchronize()) if device.startswith("cuda") else (lambda: None)

    results = {}
    for bs in batch_sizes:
        boards = pool[:bs]
        es = [rng.randint(1100, 2000) for _ in boards]
        # warmup
        for _ in range(warmup):
            forward_batch(P, boards, es, es)
        sync()
        times = []
        for _ in range(iters):
            t = time.perf_counter()
            forward_batch(P, boards, es, es)
            sync()
            times.append((time.perf_counter() - t) * 1000)
        med = statistics.median(times)
        per_pos = med / bs
        thru = 1000.0 / per_pos
        results[bs] = (med, per_pos, thru)
        print(f"  batch={bs:4d}  {med:8.2f} ms/batch   {per_pos:7.3f} ms/pos   {thru:9.0f} pos/s")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", action="append", dest="models", default=None)
    ap.add_argument("--batches", default="1,4,16,64,256")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    models = args.models or ["maia3-5m"]
    batch_sizes = [int(x) for x in args.batches.split(",")]
    rng = random.Random(args.seed)

    for alias in models:
        bench_model(alias, args.device, batch_sizes, args.warmup, args.iters, rng)


if __name__ == "__main__":
    main()
