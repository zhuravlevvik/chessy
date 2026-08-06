from __future__ import annotations

import chess
import pytest

from chessy.chess import ChessEnvironment


def test_default_environment_and_legal_pushes() -> None:
    environment = ChessEnvironment()
    assert environment.fen() == chess.STARTING_FEN
    assert len(environment.legal_moves()) == 20

    move = environment.push_uci("e2e4")
    assert move == chess.Move.from_uci("e2e4")
    assert environment.board.peek() == move
    with pytest.raises(ValueError, match="illegal"):
        environment.push(chess.Move.from_uci("e2e5"))
    with pytest.raises(ValueError, match="invalid or illegal"):
        environment.push_uci("e7e4")


def test_board_property_and_copy_are_independent() -> None:
    environment = ChessEnvironment()
    exposed = environment.board
    exposed.push_uci("e2e4")
    assert environment.fen() == chess.STARTING_FEN

    clone = environment.copy()
    clone.push_uci("d2d4")
    assert environment.fen() == chess.STARTING_FEN
    assert clone.fen() != environment.fen()


def test_reset_and_fen_history() -> None:
    environment = ChessEnvironment.from_fen("8/8/8/8/8/8/4K3/7k w - - 12 42")
    initial_fen = environment.fen()
    environment.push_uci("e2e3")
    environment.reset()
    assert environment.fen() == initial_fen
    history = environment.history(3)
    assert len(history) == 3
    assert all(board.fen() == initial_fen for board in history)


def test_history_returns_current_to_past_and_pads_earliest() -> None:
    environment = ChessEnvironment()
    start = environment.fen()
    environment.push_uci("e2e4")
    after_e4 = environment.fen()
    environment.push_uci("e7e5")
    history = environment.history(5)
    assert history[0].fen() == environment.fen()
    assert history[1].fen() == after_e4
    assert history[2].fen() == start
    assert history[3].fen() == start
    assert history[4].fen() == start
    with pytest.raises(ValueError):
        environment.history(0)


@pytest.mark.parametrize(
    ("fen", "winner"),
    [
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", chess.WHITE),
        ("7k/5Q2/7K/8/8/8/8/8 b - - 0 1", None),
        ("8/8/8/8/8/8/4K3/7k w - - 0 1", None),
    ],
)
def test_terminal_positions_and_values(fen: str, winner: chess.Color | None) -> None:
    environment = ChessEnvironment.from_fen(fen)
    assert environment.is_terminal()
    assert environment.outcome() is not None
    assert environment.outcome().winner == winner
    if winner is None:
        assert environment.terminal_value(chess.WHITE) == 0.0
        assert environment.terminal_value(chess.BLACK) == 0.0
    else:
        assert environment.terminal_value(winner) == 1.0
        assert environment.terminal_value(not winner) == -1.0


def test_claimable_threefold_and_fifty_move_draws_are_terminal() -> None:
    repeated = ChessEnvironment()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        repeated.push_uci(uci)
    assert repeated.outcome() is not None
    assert repeated.outcome().termination == chess.Termination.THREEFOLD_REPETITION

    fifty_move = ChessEnvironment.from_fen("8/8/8/8/8/8/4K3/R6k w - - 100 1")
    assert fifty_move.outcome() is not None
    assert fifty_move.outcome().termination == chess.Termination.FIFTY_MOVES


def test_non_terminal_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-terminal"):
        ChessEnvironment().terminal_value(chess.WHITE)
