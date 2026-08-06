"""Deterministic mixed historical + online-feedback batch sampler."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from chessy.feedback.dataset import FeedbackDataset
from chessy.personal.dataset import PersonalDataset
from chessy.personal.sampler import PersonalBatchSampler


@dataclass
class MixedPersonalBatchSampler:
    historical: PersonalBatchSampler
    feedback: FeedbackDataset
    max_batch_fraction: float = .25
    max_positions_per_game: int = 16
    seed: int = 0
    def __post_init__(self) -> None:
        if not 0 < self.max_batch_fraction <= .5 or self.max_positions_per_game <= 0: raise ValueError("invalid mixed sampler configuration")
        if not len(self.feedback): raise ValueError("feedback dataset cannot be empty")
        self.generator = torch.Generator(device="cpu"); self.generator.manual_seed(self.seed ^ 0x5EEDFACE)
        self.pool: list[int] = []; self.cursor = 0; self.cycle = 0; self._make_cycle()
    @property
    def batch_size(self) -> int: return self.historical.batch_size
    def _make_cycle(self) -> None:
        selected: list[int] = []
        for game in sorted(self.feedback.indices_by_game):
            candidates = self.feedback.indices_by_game[game]; order = torch.randperm(len(candidates), generator=self.generator).tolist()
            selected.extend(candidates[item] for item in order[:self.max_positions_per_game])
        order = torch.randperm(len(selected), generator=self.generator).tolist(); self.pool = [selected[item] for item in order]; self.cursor = 0
    def _feedback_count(self, historical_count: int) -> int:
        if historical_count <= 0: return 0
        return math.floor(historical_count * self.max_batch_fraction)
    def _take_feedback(self, count: int) -> list[int]:
        result: list[int] = []
        if self.cursor < len(self.pool):
            take = min(count, len(self.pool) - self.cursor)
            result.extend(self.pool[self.cursor:self.cursor + take]); self.cursor += take
        return result
    def next_batch(self) -> dict[str, list[int]] | None:
        historical = self.historical.next_batch()
        if historical is None: return None
        # Historical is the fixed epoch driver.  Replace only as many rows as
        # permitted, including on the short final batch.
        feedback_count = min(self._feedback_count(len(historical)), len(historical) - 1 if len(historical) > 1 else 0)
        feedback = self._take_feedback(feedback_count) if feedback_count else []
        return {"historical": historical[:len(historical) - len(feedback)], "feedback": feedback}
    def next_epoch(self) -> None:
        self.historical.next_epoch(); self.cycle += 1; self._make_cycle()
    def state_dict(self) -> dict[str, object]:
        return {"format": "chessy-mixed-personal-sampler-v1", "historical_fingerprint": self.historical.dataset.fingerprint, "feedback_fingerprint": self.feedback.fingerprint, "historical_state": self.historical.state_dict(), "batch_size": self.batch_size, "max_batch_fraction": self.max_batch_fraction, "max_positions_per_game": self.max_positions_per_game, "seed": self.seed, "feedback_pool": self.pool, "feedback_cursor": self.cursor, "feedback_cycle": self.cycle, "generator_state": self.generator.get_state()}
    def load_state_dict(self, state: dict[str, object]) -> None:
        required = {"format", "historical_fingerprint", "feedback_fingerprint", "historical_state", "batch_size", "max_batch_fraction", "max_positions_per_game", "seed", "feedback_pool", "feedback_cursor", "feedback_cycle", "generator_state"}
        if set(state) != required or state.get("format") != "chessy-mixed-personal-sampler-v1": raise ValueError("invalid mixed sampler state")
        expected = {"historical_fingerprint": self.historical.dataset.fingerprint, "feedback_fingerprint": self.feedback.fingerprint, "batch_size": self.batch_size, "max_batch_fraction": self.max_batch_fraction, "max_positions_per_game": self.max_positions_per_game, "seed": self.seed}
        if any(state[key] != value for key, value in expected.items()): raise ValueError("mixed sampler state is incompatible")
        if not isinstance(state["historical_state"], dict) or not isinstance(state["feedback_pool"], list) or not isinstance(state["generator_state"], torch.Tensor): raise ValueError("invalid mixed sampler state payload")
        pool = [int(item) for item in state["feedback_pool"]]
        if not pool or len(set(pool)) != len(pool) or any(item < 0 or item >= len(self.feedback) for item in pool): raise ValueError("invalid mixed feedback pool")
        self.historical.load_state_dict(state["historical_state"]); self.pool = pool; self.cursor = int(state["feedback_cursor"]); self.cycle = int(state["feedback_cycle"])
        if not 0 <= self.cursor <= len(pool) or self.cycle < 0: raise ValueError("invalid mixed feedback cursor")
        self.generator.set_state(state["generator_state"])
