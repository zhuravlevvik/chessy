"""Deterministic, independently resumable samplers for personal RL."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from chessy.feedback.dataset import FeedbackDataset
from chessy.personal.dataset import PersonalDataset
from chessy.personal.sampler import PersonalBatchSampler
from chessy.replay.dataset import ReplayDataset
from chessy.replay.sampler import ReplaySampler


class FeedbackBatchSampler:
    """No-replacement per-game capped feedback epochs, kept separate from history."""
    def __init__(self, dataset: FeedbackDataset, batch_size: int, seed: int, max_positions_per_game: int) -> None:
        if batch_size <= 0 or max_positions_per_game <= 0: raise ValueError("invalid feedback sampler dimensions")
        self.dataset, self.batch_size, self.seed, self.max_positions_per_game = dataset, batch_size, seed, max_positions_per_game
        self.generator = torch.Generator(device="cpu"); self.generator.manual_seed(seed)
        self.epoch = 0; self.cursor = 0; self.pool: list[int] = []; self._make_epoch()

    def _make_epoch(self) -> None:
        selected: list[int] = []
        for game in sorted(self.dataset.indices_by_game):
            rows = self.dataset.indices_by_game[game]
            order = torch.randperm(len(rows), generator=self.generator).tolist()
            selected.extend(rows[index] for index in order[:self.max_positions_per_game])
        order = torch.randperm(len(selected), generator=self.generator).tolist()
        self.pool, self.cursor = [selected[index] for index in order], 0

    def next_batch(self) -> list[int]:
        if self.cursor >= len(self.pool): self.epoch += 1; self._make_epoch()
        result = self.pool[self.cursor:self.cursor + self.batch_size]; self.cursor += len(result)
        return result

    def state_dict(self) -> dict[str, object]:
        return {"format": "chessy-personal-rl-feedback-sampler-v1", "dataset_fingerprint": self.dataset.fingerprint, "batch_size": self.batch_size, "seed": self.seed, "max_positions_per_game": self.max_positions_per_game, "epoch": self.epoch, "cursor": self.cursor, "pool": self.pool, "generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        required = {"format", "dataset_fingerprint", "batch_size", "seed", "max_positions_per_game", "epoch", "cursor", "pool", "generator_state"}
        if set(state) != required or state["format"] != "chessy-personal-rl-feedback-sampler-v1": raise ValueError("invalid personal RL feedback sampler state")
        expected = {"dataset_fingerprint": self.dataset.fingerprint, "batch_size": self.batch_size, "seed": self.seed, "max_positions_per_game": self.max_positions_per_game}
        if any(state[key] != value for key, value in expected.items()) or not isinstance(state["generator_state"], torch.Tensor) or not isinstance(state["pool"], list): raise ValueError("incompatible personal RL feedback sampler")
        pool = [int(item) for item in state["pool"]]
        if len(pool) != len(set(pool)) or any(item < 0 or item >= len(self.dataset) for item in pool): raise ValueError("invalid feedback sampler pool")
        self.epoch, self.cursor, self.pool = int(state["epoch"]), int(state["cursor"]), pool
        if self.epoch < 0 or not 0 <= self.cursor <= len(pool): raise ValueError("invalid feedback sampler cursor")
        self.generator.set_state(state["generator_state"])


@dataclass
class PersonalRLSamplers:
    replay: ReplaySampler
    historical: PersonalBatchSampler
    feedback: FeedbackBatchSampler | None = None

    @classmethod
    def create(cls, *, replay: ReplayDataset, historical: PersonalDataset, feedback: FeedbackDataset | None, rl_batch_size: int, historical_batch_size: int, feedback_batch_size: int, seed: int, recent_fraction: float, recent_generations: int, sample_kind_weights: dict[str, float], historical_max_positions_per_game: int, feedback_max_positions_per_game: int) -> "PersonalRLSamplers":
        return cls(ReplaySampler(replay, rl_batch_size, seed ^ 0xA11CE, recent_fraction, recent_generations), PersonalBatchSampler(historical, historical_batch_size, seed ^ 0x5157, sample_kind_weights, historical_max_positions_per_game), None if feedback is None else FeedbackBatchSampler(feedback, feedback_batch_size, seed ^ 0xFEED, feedback_max_positions_per_game))

    def next(self) -> dict[str, list[int] | torch.Tensor]:
        historical = self.historical.next_batch()
        if historical is None:
            self.historical.next_epoch(); historical = self.historical.next_batch()
        assert historical is not None
        return {"rl": self.replay.next_batch(), "historical": historical, **({"feedback": self.feedback.next_batch()} if self.feedback else {})}

    def state_dict(self) -> dict[str, object]:
        return {"format": "chessy-personal-rl-samplers-v1", "rl": self.replay.state_dict(), "historical": self.historical.state_dict(), "feedback": None if self.feedback is None else self.feedback.state_dict()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if set(state) != {"format", "rl", "historical", "feedback"} or state["format"] != "chessy-personal-rl-samplers-v1" or not isinstance(state["rl"], dict) or not isinstance(state["historical"], dict): raise ValueError("invalid personal RL sampler state")
        if (state["feedback"] is None) != (self.feedback is None): raise ValueError("feedback sampler resume mismatch")
        self.replay.load_state_dict(state["rl"]); self.historical.load_state_dict(state["historical"])
        if self.feedback: self.feedback.load_state_dict(state["feedback"])  # type: ignore[arg-type]
