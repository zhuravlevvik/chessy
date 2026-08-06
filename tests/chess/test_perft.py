from __future__ import annotations

import chess
import pytest

from chessy.chess import perft

KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


@pytest.mark.parametrize(
    ("fen", "depth", "nodes"),
    [
        (chess.STARTING_FEN, 1, 20),
        (chess.STARTING_FEN, 2, 400),
        (chess.STARTING_FEN, 3, 8902),
        (KIWIPETE_FEN, 1, 48),
        (KIWIPETE_FEN, 2, 2039),
        (KIWIPETE_FEN, 3, 97862),
    ],
)
def test_perft_matches_reference_positions(fen: str, depth: int, nodes: int) -> None:
    board = chess.Board(fen)
    before_fen = board.fen()
    before_stack = list(board.move_stack)
    assert perft(board, depth) == nodes
    assert board.fen() == before_fen
    assert board.move_stack == before_stack


def test_perft_depth_zero_and_negative_depth() -> None:
    board = chess.Board()
    assert perft(board, 0) == 1
    with pytest.raises(ValueError, match="non-negative"):
        perft(board, -1)
