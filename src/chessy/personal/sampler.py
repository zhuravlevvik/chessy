"""Stateful weighted, game-capped sampling for one historical epoch."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from chessy.personal.dataset import PersonalDataset


@dataclass
class PersonalBatchSampler:
    dataset: PersonalDataset
    batch_size: int
    seed: int
    kind_weights: dict[str, float] = field(default_factory=lambda: {"good_move": .75, "full_game": 1.0})
    max_positions_per_game: int = 16
    drop_last: bool = False
    epoch: int = 0

    def __post_init__(self) -> None:
        if self.dataset.split != "train" or self.batch_size <= 0 or self.max_positions_per_game <= 0:
            raise ValueError("personal sampler requires train data and positive sizes")
        if self.kind_weights != {"good_move": self.kind_weights.get("good_move"), "full_game": self.kind_weights.get("full_game")} or any(not isinstance(value, (int, float)) or value <= 0 for value in self.kind_weights.values()):
            raise ValueError("invalid personal kind weights")
        self.generator = torch.Generator(device="cpu"); self.generator.manual_seed(self.seed)
        self.pool: list[int] = []; self.cursor = 0
        self._make_epoch()

    def _make_epoch(self) -> None:
        capped_by_kind: dict[str, list[int]] = {"good_move": [], "full_game": []}
        for game in sorted(self.dataset.indices_by_game):
            candidates = self.dataset.indices_by_game[game]
            order = torch.randperm(len(candidates), generator=self.generator).tolist()
            for position in order[: self.max_positions_per_game]:
                index = candidates[position]
                kind = self.dataset[index]["sample_kind"]
                capped_by_kind["good_move" if kind == 0 else "full_game"].append(index)

        # Each historical game belongs to exactly one sample kind, so weighting
        # only inside a game cannot affect composition. Apply relative thinning
        # after the per-game cap instead; this is deterministic, without
        # replacement, and retains a representative from every non-empty kind.
        selected: list[int] = []
        maximum = max(self.kind_weights.values())
        for name in ("good_move", "full_game"):
            candidates = capped_by_kind[name]
            if not candidates:
                continue
            keep = max(1, round(len(candidates) * self.kind_weights[name] / maximum))
            order = torch.randperm(len(candidates), generator=self.generator).tolist()
            selected.extend(candidates[position] for position in order[:keep])
        order = torch.randperm(len(selected), generator=self.generator).tolist()
        self.pool = [selected[index] for index in order]
        if self.drop_last:
            self.pool = self.pool[: len(self.pool) // self.batch_size * self.batch_size]
        self.cursor = 0

    def next_batch(self) -> list[int] | None:
        if self.cursor >= len(self.pool):
            return None
        result = self.pool[self.cursor:self.cursor + self.batch_size]
        self.cursor += len(result)
        return result

    def next_epoch(self) -> None:
        if self.cursor < len(self.pool):
            raise ValueError("cannot advance personal sampler before epoch is consumed")
        self.epoch += 1
        self._make_epoch()

    def state_dict(self) -> dict[str, object]:
        return {"format": "chessy-personal-sampler-v1", "dataset_fingerprint": self.dataset.fingerprint,
                "batch_size": self.batch_size, "seed": self.seed, "kind_weights": self.kind_weights,
                "max_positions_per_game": self.max_positions_per_game, "drop_last": self.drop_last,
                "epoch": self.epoch, "cursor": self.cursor, "pool": self.pool,
                "generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        required = {"format", "dataset_fingerprint", "batch_size", "seed", "kind_weights", "max_positions_per_game", "drop_last", "epoch", "cursor", "pool", "generator_state"}
        if set(state) != required or state.get("format") != "chessy-personal-sampler-v1":
            raise ValueError("invalid personal sampler state")
        for name in ("dataset_fingerprint", "batch_size", "seed", "kind_weights", "max_positions_per_game", "drop_last"):
            expected = self.dataset.fingerprint if name == "dataset_fingerprint" else getattr(self, name)
            if state[name] != expected:
                raise ValueError(f"personal sampler incompatible: {name}")
        if not isinstance(state["generator_state"], torch.Tensor) or not isinstance(state["pool"], list):
            raise ValueError("invalid personal sampler serialized values")
        pool = [int(index) for index in state["pool"]]
        if len(set(pool)) != len(pool) or any(index < 0 or index >= len(self.dataset) for index in pool):
            raise ValueError("invalid personal sampler pool")
        self.epoch, self.cursor, self.pool = int(state["epoch"]), int(state["cursor"]), pool
        if not 0 <= self.cursor <= len(self.pool):
            raise ValueError("invalid personal sampler cursor")
        self.generator.set_state(state["generator_state"])
