"""Validated configuration for the versioned PUCT search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Configuration contract for ``mcts-puct-v1``."""

    version: str = "mcts-puct-v1"
    simulations: int = 128
    c_puct: float = 1.5
    temperature: float = 0.0
    root_noise: bool = False
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_batch_size: int = 32
    max_batch_wait_ms: float = 2.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.version != "mcts-puct-v1":
            raise ValueError("version must be 'mcts-puct-v1'")
        if isinstance(self.simulations, bool) or not isinstance(self.simulations, int) or self.simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        if isinstance(self.c_puct, bool) or not isinstance(self.c_puct, (int, float)) or self.c_puct <= 0:
            raise ValueError("c_puct must be positive")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not isinstance(self.root_noise, bool):
            raise ValueError("root_noise must be boolean")
        if isinstance(self.dirichlet_alpha, bool) or not isinstance(self.dirichlet_alpha, (int, float)) or self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if (
            isinstance(self.dirichlet_epsilon, bool)
            or not isinstance(self.dirichlet_epsilon, (int, float))
            or not 0 <= self.dirichlet_epsilon <= 1
        ):
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if isinstance(self.max_batch_size, bool) or not isinstance(self.max_batch_size, int) or not 1 <= self.max_batch_size <= 32:
            raise ValueError("max_batch_size must be an integer in [1, 32]")
        if (
            isinstance(self.max_batch_wait_ms, bool)
            or not isinstance(self.max_batch_wait_ms, (int, float))
            or not 0 <= self.max_batch_wait_ms <= 100
        ):
            raise ValueError("max_batch_wait_ms must be in [0, 100]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "MCTSConfig":
        if not isinstance(values, Mapping):
            raise ValueError("MCTS configuration must be a mapping")
        expected = {field.name for field in fields(cls)}
        unknown = set(values) - expected
        missing = expected - set(values)
        if unknown:
            raise ValueError(f"unknown MCTS configuration fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing MCTS configuration fields: {sorted(missing)}")
        return cls(**dict(values))


_PROFILE_SIMULATIONS = {"fast": 32, "normal": 128, "deep": 512}


def profile_config(profile: str, *, simulations: int | None = None, seed: int = 0) -> MCTSConfig:
    """Return a deterministic human-play profile with root noise disabled."""
    if profile not in _PROFILE_SIMULATIONS:
        raise ValueError(f"unknown MCTS profile: {profile!r}")
    config = MCTSConfig(simulations=_PROFILE_SIMULATIONS[profile], seed=seed)
    return replace(config, simulations=simulations) if simulations is not None else config


def profile_names() -> tuple[str, ...]:
    return tuple(_PROFILE_SIMULATIONS)
