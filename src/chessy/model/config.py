"""Configuration contract for the first Chessy residual CNN."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from chessy.encoding import ACTION_PLANES, ACTION_SIZE, BOARD_PLANES


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Validated configuration for the versioned ``residual-cnn-v1`` model."""

    architecture: str = "residual-cnn-v1"
    input_planes: int = 119
    action_planes: int = 73
    board_size: int = 8
    channels: int = 96
    residual_blocks: int = 8
    group_norm_groups: int = 8
    value_channels: int = 32
    value_hidden: int = 128
    value_classes: int = 3

    def __post_init__(self) -> None:
        if self.architecture != "residual-cnn-v1":
            raise ValueError("architecture must be 'residual-cnn-v1'")

        numeric_fields = (
            "input_planes",
            "action_planes",
            "board_size",
            "channels",
            "residual_blocks",
            "group_norm_groups",
            "value_channels",
            "value_hidden",
            "value_classes",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.input_planes != BOARD_PLANES:
            raise ValueError(f"input_planes must match board119-v1 ({BOARD_PLANES})")
        if self.action_planes != ACTION_PLANES:
            raise ValueError(f"action_planes must match az73-v1 ({ACTION_PLANES})")
        if self.board_size != 8:
            raise ValueError("board_size must be 8 for residual-cnn-v1")
        if self.value_classes != 3:
            raise ValueError("value_classes must be 3 (loss/draw/win) for residual-cnn-v1")
        if self.action_planes * self.board_size * self.board_size != ACTION_SIZE:
            raise ValueError("action planes and board size are incompatible with az73-v1")
        if self.channels % self.group_norm_groups:
            raise ValueError("channels must be divisible by group_norm_groups")
        if self.value_channels % self.group_norm_groups:
            raise ValueError("value_channels must be divisible by group_norm_groups")

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible configuration."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Create a config while rejecting missing and unknown fields."""
        if not isinstance(values, Mapping):
            raise ValueError("model configuration must be a mapping")
        expected = {field.name for field in fields(cls)}
        actual = set(values)
        unknown = actual - expected
        missing = expected - actual
        if unknown:
            raise ValueError(f"unknown model configuration fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing model configuration fields: {sorted(missing)}")
        return cls(**dict(values))
