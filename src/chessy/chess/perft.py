"""Reference move-tree counting for correctness tests."""

from __future__ import annotations

import chess


def perft(board: chess.Board, depth: int) -> int:
    """Count legal move-tree nodes at ``depth`` without changing ``board``."""
    if depth < 0:
        raise ValueError("perft depth must be non-negative")
    if depth == 0:
        return 1

    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes
