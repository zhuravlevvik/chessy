from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from chessy.encoding import (
    ACTION_ENCODING_VERSION,
    ACTION_PLANES,
    ACTION_SIZE,
    decode_action,
    encode_move,
    legal_action_mask,
)

KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def assert_round_trip_all_legal(board: chess.Board) -> None:
    actions = [encode_move(board, move) for move in board.legal_moves]
    assert len(actions) == len(set(actions))
    for action in actions:
        move = decode_action(board, action)
        assert encode_move(board, move) == action


def test_contract_and_exact_initial_indices() -> None:
    board = chess.Board()
    assert ACTION_ENCODING_VERSION == "az73-v1"
    assert ACTION_PLANES == 73
    assert ACTION_SIZE == 4672
    assert encode_move(board, chess.Move.from_uci("e2e3")) == 12
    assert encode_move(board, chess.Move.from_uci("e2e4")) == 76
    assert encode_move(board, chess.Move.from_uci("g1f3")) == 63 * 64 + 6


@pytest.mark.parametrize("board", [chess.Board(), chess.Board(KIWIPETE_FEN)])
def test_round_trip_all_legal_for_reference_positions(board: chess.Board) -> None:
    before = (board.fen(), list(board.move_stack))
    assert_round_trip_all_legal(board)
    assert (board.fen(), list(board.move_stack)) == before


def test_both_castles_for_both_colours() -> None:
    white = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    black = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    for board, moves in ((white, ("e1g1", "e1c1")), (black, ("e8g8", "e8c8"))):
        for uci in moves:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            assert decode_action(board, encode_move(board, move)) == move


@pytest.mark.parametrize(
    ("moves", "ep_move"),
    [
        (("e2e4", "a7a6", "e4e5", "d7d5"), "e5d6"),
        (("a2a3", "e7e5", "a3a4", "e5e4", "d2d4"), "e4d3"),
    ],
)
def test_en_passant_round_trip(moves: tuple[str, ...], ep_move: str) -> None:
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    move = chess.Move.from_uci(ep_move)
    assert board.is_en_passant(move)
    assert decode_action(board, encode_move(board, move)) == move


@pytest.mark.parametrize(
    ("fen", "moves"),
    [
        ("8/P7/8/8/8/8/8/k6K w - - 0 1", ("a7a8q", "a7a8r", "a7a8b", "a7a8n")),
        ("k6K/8/8/8/8/8/p7/8 b - - 0 1", ("a2a1q", "a2a1r", "a2a1b", "a2a1n")),
        ("r1r5/1P6/8/8/8/8/8/k6K w - - 0 1", ("b7a8q", "b7a8r", "b7a8b", "b7a8n", "b7c8q", "b7c8r", "b7c8b", "b7c8n")),
        ("7k/4K3/8/8/8/8/1p6/R1R5 b - - 0 1", ("b2a1q", "b2a1r", "b2a1b", "b2a1n", "b2c1q", "b2c1r", "b2c1b", "b2c1n")),
    ],
)
def test_all_promotion_types_and_directions(fen: str, moves: tuple[str, ...]) -> None:
    board = chess.Board(fen)
    for uci in moves:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        assert decode_action(board, encode_move(board, move)) == move


def test_random_reachable_positions_are_collision_free_and_round_trip() -> None:
    board = chess.Board()
    random_source = random.Random(20260806)
    positions_checked = 0
    moves_checked = 0
    while positions_checked < 100:
        if board.is_game_over():
            board.reset()
        moves = sorted(board.legal_moves, key=lambda move: move.uci())
        actions = [encode_move(board, move) for move in moves]
        assert len(actions) == len(set(actions))
        for move, action in zip(moves, actions, strict=True):
            assert decode_action(board, action) == move
        moves_checked += len(moves)
        positions_checked += 1
        board.push(random_source.choice(moves))
    assert positions_checked == 100
    assert moves_checked > 0


def test_legal_mask_matches_legal_moves_and_is_stable() -> None:
    board = chess.Board(KIWIPETE_FEN)
    before = (board.fen(), list(board.move_stack))
    mask = legal_action_mask(board)
    assert mask.shape == (ACTION_SIZE,)
    assert mask.dtype == np.bool_
    assert mask.flags.c_contiguous
    assert mask.sum() == board.legal_moves.count()
    assert np.array_equal(mask, legal_action_mask(board))
    for move in board.legal_moves:
        assert mask[encode_move(board, move)]
    assert (board.fen(), list(board.move_stack)) == before


@pytest.mark.parametrize(
    "fen",
    ["7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "7k/5Q2/7K/8/8/8/8/8 b - - 0 1"],
)
def test_terminal_positions_have_empty_mask(fen: str) -> None:
    mask = legal_action_mask(chess.Board(fen))
    assert not mask.any()


def test_invalid_actions_and_moves_are_rejected() -> None:
    board = chess.Board()
    for action in (-1, ACTION_SIZE, 1.5):
        with pytest.raises(ValueError):
            decode_action(board, action)
    with pytest.raises(ValueError, match="illegal"):
        encode_move(board, chess.Move.from_uci("e2e5"))
    with pytest.raises(ValueError, match="not legal"):
        decode_action(board, 0)
