from __future__ import annotations

from dataclasses import replace

import chess
import pytest

from chessy.mcts import MCTSConfig, SearchNode, backup, profile_config, puct_score, select_child


def test_config_profiles_and_strict_round_trip() -> None:
    config = MCTSConfig()
    assert config.version == "mcts-puct-v1"
    assert config.c_puct == 1.5
    assert MCTSConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="unknown"):
        MCTSConfig.from_dict({**config.to_dict(), "surprise": True})
    for name, simulations in (("fast", 32), ("normal", 128), ("deep", 512)):
        profile = profile_config(name)
        assert profile.simulations == simulations
        assert profile.temperature == 0
        assert profile.root_noise is False


@pytest.mark.parametrize(
    "changes",
    [
        {"simulations": 0}, {"c_puct": 0}, {"temperature": -1},
        {"dirichlet_alpha": 0}, {"dirichlet_epsilon": 1.1},
        {"max_batch_size": 33}, {"max_batch_wait_ms": 101}, {"seed": True},
    ],
)
def test_config_rejects_invalid_values(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(MCTSConfig(), **changes)


def test_exact_puct_formula_and_tie_break() -> None:
    parent = SearchNode(prior=1, to_play=chess.WHITE, visit_count=9)
    child = SearchNode(prior=0.25, to_play=chess.BLACK, visit_count=2, value_sum=1.0)
    assert puct_score(parent, child, 1.5) == pytest.approx(-0.5 + 1.5 * 0.25 * 3 / 3)
    parent.children = {
        9: SearchNode(prior=0.5, to_play=chess.BLACK),
        3: SearchNode(prior=0.5, to_play=chess.BLACK),
    }
    assert select_child(parent, 1.5)[0] == 3


@pytest.mark.parametrize("length", (1, 2, 3))
def test_backup_alternates_sign_for_every_path_length(length: int) -> None:
    path = [SearchNode(prior=1, to_play=bool(index % 2)) for index in range(length)]
    backup(path, 0.75)
    for distance, node in enumerate(reversed(path)):
        assert node.visit_count == 1
        assert node.value_sum == pytest.approx(0.75 * (-1 if distance % 2 else 1))
