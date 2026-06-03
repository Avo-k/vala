"""Allie v1 — a second, sequence-based human model for vala (diagnostics first).

Where Maia is position-only and predicts moves, Allie (a small gpt2 over the UCI
move sequence) emits three heads in one forward — **policy** (Elo-conditioned move
distribution, like Maia but history-aware), **value** (practical outcome at that
Elo, White-POV in [-1, 1] — NOT an engine eval), and **think_time** (predicted
human seconds on this position). See memory `allie-think-time` for the probe that
motivated this.

Step 1 of the roadmap uses these as *diagnostics only* (logged in `MoveInfo`, no
decision change) so we can measure whether Allie's value-gap / think-time / policy
correlate with successful baits before wiring any of them into the selector.

Interface mirrors `Maia3Predictor` closely enough to become a swappable opponent
model later, but returns the richer `AllieOut` (move_probs + value + think_time).

    predict_batch(boards, elos) -> list[AllieOut]      # one batched forward

Caveats baked in from the probe: Allie was trained on Lichess **blitz**; its
think-time is conditioned on position + Elo + time-control token only (NOT the live
clock), and time-controls outside its 24-token blitz set fall back to `<unk>`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import yaml

# Allie's blitz time-control vocabulary (seconds+increment). Anything else tokenizes
# to <unk> and degrades the time/policy heads, so default to a covered control.
DEFAULT_TIME_CONTROL = "300+0"
ELO_MIN, ELO_MAX = 600, 2600


@dataclass(frozen=True)
class AllieOut:
    move_probs: dict[str, float]  # P(move | pos, elo) over legal moves
    value: float                  # White-POV expected result in [-1, 1] (practical, at Elo)
    think_time: float             # predicted human think time (seconds)


class AlliePredictor:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cpu") -> None:
        import torch

        from vala._allie.data import (
            TIME_MEAN,
            TIME_STDEV,
            Game,
            UCITokenizer,
        )
        from vala._allie.model import initialize_model

        self._torch = torch
        self._Game = Game
        self._TIME_MEAN = TIME_MEAN
        self._TIME_STDEV = TIME_STDEV

        ckpt = Path(checkpoint_dir)
        cfg = yaml.safe_load((ckpt / "config.yaml").read_text())
        self._tok = UCITokenizer(**cfg["data_config"]["tokenizer_config"])

        model_cfg = dict(cfg["model_config"])
        model_cfg["use_pretrained"] = False  # weights come from the checkpoint
        model = initialize_model(self._tok, **model_cfg)
        state = torch.load(ckpt / "best.pt", map_location=device)["model"]
        model.load_state_dict(state, strict=False)
        model.eval()
        self._model = model.to(device)
        self.device = device
        self.name = f"allie:{ckpt.name}"

    @staticmethod
    def _clamp(elo: int) -> int:
        return max(ELO_MIN, min(ELO_MAX, int(elo)))

    def _game(self, board: chess.Board, elo: int, time_control: str):
        # Per-move times are masked out of the model input (they are training
        # labels only), so zeros here are inert — only their count must match.
        moves = [m.uci() for m in board.move_stack]
        e = self._clamp(elo)
        return self._Game(
            time_control=time_control,
            white_elo=e,
            black_elo=e,
            outcome=None,
            normal_termination=False,
            moves=moves,
            moves_seconds=[0] * len(moves),
        )

    def predict_batch(
        self,
        boards: list[chess.Board],
        elos_self: list[int],
        elos_oppo: list[int] | None = None,
        time_controls: list[str] | None = None,
    ) -> list[AllieOut]:
        """One batched forward over `boards`. Game-over boards return an empty
        distribution, neutral value 0.0, and 0.0 think time.

        `elos_oppo` / `time_controls` are accepted for interface symmetry; Allie
        conditions on a single Elo pair (we pass the human Elo for both colors,
        matching vala's neutral convention) and a per-board time-control token.
        """
        torch = self._torch
        tok = self._tok
        n = len(boards)
        tcs = time_controls or [DEFAULT_TIME_CONTROL] * n

        out: list[AllieOut] = [AllieOut({}, 0.0, 0.0)] * n
        live = [(i, b) for i, b in enumerate(boards) if not b.is_game_over()]
        if not live:
            return out

        games = [self._game(b, elos_self[i], tcs[i]) for i, b in live]
        batch = tok.pad_and_collate(games)

        with torch.inference_mode():
            res = self._model(**batch)
        logits = res["logits"].float()        # (B, T, V)
        time_logits = res["time_logits"].float()
        value_logits = res["value_logits"].float()
        attn = batch["attention_mask"]
        last = attn.sum(dim=1).long() - 1     # true last (non-pad) token per row

        for row, (orig_i, b) in enumerate(live):
            t = int(last[row].item())
            mv_logits = logits[row, t]
            legal = list(b.legal_moves)
            ids = torch.tensor([tok.token_to_id[m.uci()] for m in legal], dtype=torch.long)
            probs = torch.softmax(mv_logits[ids], dim=0)
            move_probs = {m.uci(): float(p) for m, p in zip(legal, probs)}
            value = float(value_logits[row, t].item())
            tt = float(time_logits[row, t].item()) * self._TIME_STDEV + self._TIME_MEAN
            out[orig_i] = AllieOut(move_probs=move_probs, value=value, think_time=tt)
        return out

    def predict(
        self, board: chess.Board, elo: int = 1500, time_control: str = DEFAULT_TIME_CONTROL,
    ) -> AllieOut:
        return self.predict_batch([board], [elo], time_controls=[time_control])[0]
