from __future__ import annotations

import chess
import numpy as np
import pytest

from chessy.chess import ChessEnvironment
from chessy.encoding import BOARD_ENCODING_VERSION, BOARD_PLANES, HISTORY_LENGTH, encode_board


def square_value(encoded: np.ndarray, plane: int, name: str) -> float:
    square = chess.parse_square(name)
    return encoded[plane, chess.square_rank(square), chess.square_file(square)]


def test_initial_position_shape_layout_pieces_and_metadata() -> None:
    encoded = encode_board([chess.Board()])
    assert BOARD_ENCODING_VERSION == "board119-v1"
    assert encoded.shape == (BOARD_PLANES, 8, 8)
    assert encoded.dtype == np.float32
    assert encoded.flags.c_contiguous
    assert np.all((encoded >= 0.0) & (encoded <= 1.0))
    assert square_value(encoded, 0, "a2") == 1.0
    assert square_value(encoded, 0, "h2") == 1.0
    assert square_value(encoded, 1, "b1") == 1.0
    assert square_value(encoded, 5, "e1") == 1.0
    assert square_value(encoded, 6, "a7") == 1.0
    assert square_value(encoded, 7, "g8") == 1.0
    assert square_value(encoded, 11, "e8") == 1.0
    assert encoded[0].sum() == 8.0
    assert encoded[6].sum() == 8.0
    assert np.all(encoded[112] == 1.0)
    assert all(np.all(encoded[plane] == 1.0) for plane in (113, 114, 115, 116))
    assert np.all(encoded[117] == 0.0)
    assert not encoded[118].any()


def test_absolute_orientation_turn_ep_and_halfmove_clock() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    encoded = encode_board([board])
    assert not encoded[112].any()
    assert square_value(encoded, 0, "e4") == 1.0
    assert square_value(encoded, 118, "e3") == 1.0

    for clock, expected in ((0, 0.0), (42, 0.42), (100, 1.0), (150, 1.0)):
        clock_board = chess.Board(f"8/8/8/8/8/8/4K3/7k w - - {clock} 1")
        assert np.all(encode_board([clock_board])[117] == expected)


def test_each_castling_right_is_encoded_separately() -> None:
    rights = (("K", 113), ("Q", 114), ("k", 115), ("q", 116))
    for kept, expected_plane in rights:
        board = chess.Board(f"r3k2r/8/8/8/8/8/8/R3K2R w {kept} - 0 1")
        encoded = encode_board([board])
        assert np.all(encoded[expected_plane] == 1.0)
        for plane in (113, 114, 115, 116):
            if plane != expected_plane:
                assert not encoded[plane].any()


def test_history_order_padding_and_no_mutation() -> None:
    environment = ChessEnvironment()
    start = environment.board
    environment.push_uci("e2e4")
    after_e4 = environment.board
    environment.push_uci("e7e5")
    current = environment.board
    history = (current, after_e4, start)
    move_stacks = [list(board.move_stack) for board in history]
    encoded = encode_board(history)
    assert square_value(encoded, 0, "e4") == 1.0
    assert square_value(encoded, 14 + 0, "e4") == 1.0
    assert square_value(encoded, 28 + 0, "e2") == 1.0
    assert np.array_equal(encoded[42:56], encoded[28:42])
    assert [list(board.move_stack) for board in history] == move_stacks
    assert encoded.flags.c_contiguous


def test_repetition_planes_and_determinism() -> None:
    environment = ChessEnvironment()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        environment.push_uci(uci)
    current = environment.board
    encoded_a = encode_board(environment.history())
    encoded_b = encode_board(environment.history())
    assert np.array_equal(encoded_a, encoded_b)
    assert np.all(encoded_a[12] == 1.0)
    assert np.all(encoded_a[13] == 1.0)
    assert current.is_repetition(3)


def test_empty_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        encode_board([])


def test_constants_match_contract() -> None:
    assert BOARD_PLANES == 119
    assert HISTORY_LENGTH == 8
