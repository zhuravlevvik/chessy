from __future__ import annotations

from dataclasses import replace

import pytest

from chessy.encoding import ACTION_PLANES, ACTION_SIZE, BOARD_PLANES
from chessy.model import ModelConfig


def test_default_config_matches_the_v1_contract() -> None:
    config = ModelConfig()
    assert config.architecture == "residual-cnn-v1"
    assert config.input_planes == BOARD_PLANES == 119
    assert config.action_planes == ACTION_PLANES == 73
    assert config.board_size == 8
    assert config.channels == 96
    assert config.residual_blocks == 8
    assert config.group_norm_groups == 8
    assert config.value_channels == 32
    assert config.value_hidden == 128
    assert config.value_classes == 3
    assert config.action_planes * config.board_size**2 == ACTION_SIZE


def test_config_dict_round_trip_is_strict() -> None:
    config = ModelConfig()
    assert ModelConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="unknown"):
        ModelConfig.from_dict({**config.to_dict(), "extra": 1})
    incomplete = config.to_dict()
    incomplete.pop("channels")
    with pytest.raises(ValueError, match="missing"):
        ModelConfig.from_dict(incomplete)


@pytest.mark.parametrize(
    "changes",
    [
        {"architecture": "other"},
        {"input_planes": 118},
        {"action_planes": 72},
        {"board_size": 7},
        {"value_classes": 2},
        {"channels": 95},
        {"value_channels": 31},
        {"residual_blocks": 0},
    ],
)
def test_config_rejects_incompatible_values(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(ModelConfig(), **changes)
