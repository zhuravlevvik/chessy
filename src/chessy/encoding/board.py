"""The versioned, history-aware ``board119-v1`` position encoding."""

from __future__ import annotations

from collections.abc import Sequence

import chess
import numpy as np

BOARD_ENCODING_VERSION = "board119-v1"
BOARD_PLANES = 119
HISTORY_LENGTH = 8

_PIECE_ORDER = (
    (chess.WHITE, chess.PAWN),
    (chess.WHITE, chess.KNIGHT),
    (chess.WHITE, chess.BISHOP),
    (chess.WHITE, chess.ROOK),
    (chess.WHITE, chess.QUEEN),
    (chess.WHITE, chess.KING),
    (chess.BLACK, chess.PAWN),
    (chess.BLACK, chess.KNIGHT),
    (chess.BLACK, chess.BISHOP),
    (chess.BLACK, chess.ROOK),
    (chess.BLACK, chess.QUEEN),
    (chess.BLACK, chess.KING),
)


def _fill_square(plane: np.ndarray, square: chess.Square) -> None:
    plane[chess.square_rank(square), chess.square_file(square)] = 1.0


def encode_board(history: Sequence[chess.Board]) -> np.ndarray:
    """Encode current-to-past board history into a ``[119, 8, 8]`` array."""
    if not history:
        raise ValueError("board history must contain at least one position")
    if any(not isinstance(board, chess.Board) for board in history):
        raise ValueError("board history must contain chess.Board instances")

    states = list(history[:HISTORY_LENGTH])
    while len(states) < HISTORY_LENGTH:
        states.append(states[-1])

    encoded = np.zeros((BOARD_PLANES, 8, 8), dtype=np.float32)
    for history_index, state in enumerate(states):
        offset = history_index * 14
        for piece_offset, (color, piece_type) in enumerate(_PIECE_ORDER):
            for square in state.pieces(piece_type, color):
                _fill_square(encoded[offset + piece_offset], square)
        if state.is_repetition(2):
            encoded[offset + 12].fill(1.0)
        if state.is_repetition(3):
            encoded[offset + 13].fill(1.0)

    current = states[0]
    if current.turn == chess.WHITE:
        encoded[112].fill(1.0)
    for plane, right in (
        (113, current.has_kingside_castling_rights(chess.WHITE)),
        (114, current.has_queenside_castling_rights(chess.WHITE)),
        (115, current.has_kingside_castling_rights(chess.BLACK)),
        (116, current.has_queenside_castling_rights(chess.BLACK)),
    ):
        if right:
            encoded[plane].fill(1.0)
    encoded[117].fill(min(max(current.halfmove_clock, 0), 100) / 100.0)
    if current.ep_square is not None:
        _fill_square(encoded[118], current.ep_square)
    return encoded
