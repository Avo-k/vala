"""Maia-3 human-move predictor — vala's opponent model.

Returns P(move | position, elos) over all legal moves, plus the value head's
win probability. Unlike rorschach's predictor, this one exposes a **batched**
forward (`predict_batch`): vala's expectimax evaluates many sibling positions
per move, and Maia batches them in a single forward — the key scaling lever,
especially on GPU.

    predict(board, elo_self, elo_oppo)        -> (move_probs: {uci: float}, win_prob)
    predict_batch(boards, elos_self, elos_oppo) -> list[(move_probs, win_prob)]
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import chess

ELO_MIN, ELO_MAX = 600, 2600


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Maia3Predictor:
    def __init__(self, alias: str = "maia3-5m", device: str = "auto") -> None:
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

        device = _pick_device(device)
        self._torch = torch
        self._tokenize_board = tokenize_board
        self._get_historical_tokens = get_historical_tokens
        self._get_legal_moves_mask = get_legal_moves_mask
        self._mirror_move = mirror_move

        spec = resolve_model_spec(alias)
        cfg = SimpleNamespace(
            device=device, use_amp=False, trust_checkpoint=False,
            history=8, use_padding=True, include_time_info=False,
            dim_emb=128, dim_vit=192, num_blocks=8, num_heads=6, mlp_ratio=2.0,
            dropout=0.0, head_hid_dim=192,
            use_gab=True, gab_gen_size=64, gab_per_square_dim=0,
            gab_intermediate_dim=64, use_rms_norm=True, omit_qkv_biases=True,
            activation="gelu", use_relative_bias=False, use_absolute_pe=False,
        )
        apply_model_config(cfg, spec)
        cfg.checkpoint_path = resolve_checkpoint_path(spec)
        self.cfg = cfg

        model = MAIA3Model(cfg).to(device)
        ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        renamed = {k.replace("smolgen", "gab"): v for k, v in state_dict.items()}
        model.load_state_dict(renamed, strict=False)
        model.eval()
        self._model = model

        self._all_moves_dict = {m: i for i, m in enumerate(get_all_possible_moves())}
        self._name = alias
        self.device = device

    def _build_history(self, board: chess.Board) -> deque:
        moves = list(board.move_stack)
        history: deque = deque(maxlen=self.cfg.history)
        if not moves:
            history.append(self._tokenize_board(board))
            return history
        replay = board.copy()
        for _ in range(len(moves)):
            replay.pop()
        history.append(self._tokenize_board(replay))
        for mv in moves:
            replay.push(mv)
            history.append(self._tokenize_board(replay))
        return history

    @staticmethod
    def _clamp(elo: int) -> int:
        return max(ELO_MIN, min(ELO_MAX, int(elo)))

    def predict_batch(
        self,
        boards: list[chess.Board],
        elos_self: list[int],
        elos_oppo: list[int],
    ) -> list[tuple[dict[str, float], float]]:
        """One batched forward over `boards`. Game-over boards return ({}, 0.5)."""
        torch = self._torch
        cfg = self.cfg

        live = [(i, b) for i, b in enumerate(boards) if not b.is_game_over()]
        out: list[tuple[dict[str, float], float]] = [({}, 0.5)] * len(boards)
        if not live:
            return out

        with torch.no_grad():
            tok_list, mask_list = [], []
            for _, b in live:
                hist = self._build_history(b)
                tok_list.append(self._get_historical_tokens(
                    hist, cfg, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0,
                ))
                mask_list.append(self._get_legal_moves_mask(b, self._all_moves_dict))
            tokens = torch.stack(tok_list, dim=0).to(cfg.device)
            masks = torch.stack(mask_list, dim=0).to(cfg.device)
            self_e = torch.tensor([self._clamp(elos_self[i]) for i, _ in live],
                                  dtype=torch.long, device=cfg.device)
            oppo_e = torch.tensor([self._clamp(elos_oppo[i]) for i, _ in live],
                                  dtype=torch.long, device=cfg.device)

            logits_move, logits_value, _ = self._model(tokens, self_e, oppo_e)
            logits = logits_move.float().masked_fill(~masks, float("-inf"))
            probs = torch.softmax(logits, dim=-1).cpu()
            vals = torch.softmax(logits_value.float(), dim=-1).cpu()

            for row, (orig_i, b) in enumerate(live):
                d: dict[str, float] = {}
                for move in b.legal_moves:
                    uci = move.uci()
                    key = uci if b.turn == chess.WHITE else self._mirror_move(uci)
                    idx = self._all_moves_dict.get(key)
                    if idx is not None:
                        d[uci] = float(probs[row, idx])
                out[orig_i] = (d, float(vals[row, 2]))
        return out

    def predict(
        self,
        board: chess.Board,
        elo_self: int = 1900,
        elo_oppo: int = 1900,
    ) -> tuple[dict[str, float], float]:
        return self.predict_batch([board], [elo_self], [elo_oppo])[0]

    def sample(
        self,
        board: chess.Board,
        elo_self: int = 1500,
        elo_oppo: int = 1500,
        *,
        rng=None,
        temperature: float = 1.0,
    ) -> chess.Move | None:
        """Sample a move from Maia's distribution — a stand-in for a human player.

        `temperature` > 1 flattens the distribution (more erratic), < 1 sharpens
        it toward the modal move. Returns None only when there are no legal moves.
        """
        import random as _random
        rng = rng or _random
        probs, _ = self.predict(board, elo_self, elo_oppo)
        if not probs:
            return None
        moves = list(probs.keys())
        weights = list(probs.values())
        if temperature != 1.0:
            inv = 1.0 / max(1e-6, temperature)
            weights = [w ** inv for w in weights]
        total = sum(weights)
        if total <= 0:
            return chess.Move.from_uci(rng.choice(moves))
        r = rng.random() * total
        acc = 0.0
        for uci, w in zip(moves, weights):
            acc += w
            if r <= acc:
                return chess.Move.from_uci(uci)
        return chess.Move.from_uci(moves[-1])


def make_predictor(name: str, device: str = "auto") -> "Maia3Predictor":
    if name.startswith("maia3"):
        return Maia3Predictor(alias=name, device=device)
    raise ValueError(f"unknown maia predictor {name!r}")
