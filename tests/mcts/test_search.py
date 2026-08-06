from __future__ import annotations

import random
from collections.abc import Sequence

import chess
import numpy as np
import pytest

from chessy.chess import ChessEnvironment
from chessy.encoding import ACTION_SIZE, encode_move
from chessy.mcts import Evaluation, MCTS, MCTSConfig


class ScriptedEvaluator:
    def __init__(self, preferred_uci: str | None = None, value: float = 0.0) -> None:
        self.preferred_uci = preferred_uci
        self.value = value
        self.calls = 0

    def evaluate(self, history: Sequence[chess.Board]) -> Evaluation:
        self.calls += 1
        board = history[0]
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        for move in board.legal_moves:
            policy[encode_move(board, move)] = 1.0
        if self.preferred_uci:
            move = chess.Move.from_uci(self.preferred_uci)
            if move in board.legal_moves:
                policy[:] = 0
                policy[encode_move(board, move)] = 1.0
        policy /= policy.sum()
        return Evaluation(policy, self.value)


def test_search_visits_legal_children_exactly_n_times_and_preserves_environment() -> None:
    environment = ChessEnvironment()
    before = (environment.fen(), environment.history())
    mcts = MCTS(ScriptedEvaluator(), MCTSConfig(simulations=16))
    result = mcts.search(environment)
    assert result.move in environment.legal_moves()
    assert set(result.policy) == {encode_move(environment.board, move) for move in environment.legal_moves()}
    assert sum(entry.visits for entry in result.policy.values()) == 16
    assert sum(entry.prior for entry in result.policy.values()) == pytest.approx(1)
    assert environment.fen() == before[0]
    assert [board.fen() for board in environment.history()] == [board.fen() for board in before[1]]


def test_terminal_children_do_not_call_evaluator_and_mate_in_one_wins() -> None:
    environment = ChessEnvironment.from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    evaluator = ScriptedEvaluator(preferred_uci="f7g7")
    mcts = MCTS(evaluator, MCTSConfig(simulations=4))
    result = mcts.search(environment)
    assert result.move.uci() == "f7g7"
    # One bootstrap plus evaluations of non-terminal leaves only; the mating child is never evaluated.
    assert evaluator.calls < 5


@pytest.mark.parametrize(
    "fen",
    [
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        "7k/5Q2/7K/8/8/8/8/8 b - - 0 1",
        "8/8/8/8/8/8/K7/k7 w - - 100 1",
    ],
)
def test_terminal_root_is_rejected_without_evaluation(fen: str) -> None:
    evaluator = ScriptedEvaluator()
    with pytest.raises(ValueError, match="terminal"):
        MCTS(evaluator, MCTSConfig(simulations=1)).search(ChessEnvironment.from_fen(fen))
    assert evaluator.calls == 0


def test_temperature_zero_is_deterministic_and_positive_temperature_is_seeded() -> None:
    first = MCTS(ScriptedEvaluator(), MCTSConfig(simulations=20, seed=7)).search(ChessEnvironment())
    second = MCTS(ScriptedEvaluator(), MCTSConfig(simulations=20, seed=99)).search(ChessEnvironment())
    assert first.action == second.action
    config = MCTSConfig(simulations=20, temperature=1.0, seed=12)
    assert MCTS(ScriptedEvaluator(), config).search(ChessEnvironment()).action == MCTS(ScriptedEvaluator(), config).search(ChessEnvironment()).action


def test_root_noise_is_reproducible_and_only_applied_once() -> None:
    config = MCTSConfig(simulations=1, root_noise=True, seed=42)
    first = MCTS(ScriptedEvaluator(), config)
    first_result = first.search(ChessEnvironment())
    first_priors = {action: item.prior for action, item in first_result.policy.items()}
    second_result = MCTS(ScriptedEvaluator(), config).search(ChessEnvironment())
    assert first_priors == {action: item.prior for action, item in second_result.policy.items()}
    continued = first.search(ChessEnvironment())
    assert first_priors == {action: item.prior for action, item in continued.policy.items()}
    assert sum(first_priors.values()) == pytest.approx(1)


def test_tree_reuse_and_position_mismatch_reset() -> None:
    environment = ChessEnvironment()
    mcts = MCTS(ScriptedEvaluator(), MCTSConfig(simulations=8))
    result = mcts.search(environment)
    selected_visits = mcts.root.children[result.action].visit_count  # type: ignore[union-attr]
    environment.push(result.move)
    assert mcts.advance(environment, result.action)
    assert mcts.root is not None and mcts.root.visit_count == selected_visits
    mcts.search(environment)
    unrelated = ChessEnvironment()
    assert not mcts.advance(unrelated, result.action)
    assert mcts.root is not None and mcts.root.visit_count == 0


def test_returns_legal_moves_on_100_seeded_reachable_positions() -> None:
    source = random.Random(20260806)
    environment = ChessEnvironment()
    for _ in range(100):
        if environment.is_terminal():
            environment = ChessEnvironment()
        result = MCTS(ScriptedEvaluator(), MCTSConfig(simulations=2)).search(environment)
        assert result.move in environment.legal_moves()
        environment.push(source.choice(environment.legal_moves()))
