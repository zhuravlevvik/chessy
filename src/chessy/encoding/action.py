"""The versioned ``az73-v1`` standard-chess action encoding."""

from __future__ import annotations

from numbers import Integral

import chess
import numpy as np

ACTION_ENCODING_VERSION = "az73-v1"
ACTION_PLANES = 73
ACTION_SIZE = 4672

_LINEAR_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
_KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
_UNDERPROMOTION_PIECES: tuple[chess.PieceType, ...] = (
    chess.KNIGHT, chess.BISHOP, chess.ROOK,
)


def _require_standard(board: chess.Board) -> None:
    if board.chess960:
        raise ValueError("az73-v1 supports standard chess only")


def _delta(move: chess.Move) -> tuple[int, int]:
    return (
        chess.square_file(move.to_square) - chess.square_file(move.from_square),
        chess.square_rank(move.to_square) - chess.square_rank(move.from_square),
    )


def _linear_plane(delta_file: int, delta_rank: int) -> int | None:
    for direction_index, (file_step, rank_step) in enumerate(_LINEAR_DIRECTIONS):
        for distance in range(1, 8):
            if (delta_file, delta_rank) == (file_step * distance, rank_step * distance):
                return direction_index * 7 + distance - 1
    return None


def _underpromotion_plane(board: chess.Board, move: chess.Move) -> int:
    try:
        piece_index = _UNDERPROMOTION_PIECES.index(move.promotion)
    except ValueError as exc:
        raise ValueError(f"unsupported underpromotion in move {move.uci()}") from exc
    forward = 1 if board.turn == chess.WHITE else -1
    directions = ((-forward, forward), (0, forward), (forward, forward))
    try:
        direction_index = directions.index(_delta(move))
    except ValueError as exc:
        raise ValueError(f"invalid underpromotion geometry for move {move.uci()}") from exc
    return 64 + piece_index * 3 + direction_index


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a legal ``move`` in ``board`` as a unique az73-v1 action index."""
    _require_standard(board)
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move.uci()} in position {board.fen()}")

    if move.promotion in _UNDERPROMOTION_PIECES:
        plane = _underpromotion_plane(board, move)
    else:
        delta_file, delta_rank = _delta(move)
        plane = _linear_plane(delta_file, delta_rank)
        if plane is None:
            try:
                plane = 56 + _KNIGHT_DELTAS.index((delta_file, delta_rank))
            except ValueError as exc:
                raise ValueError(f"unencodable move geometry {move.uci()}") from exc
    return plane * 64 + move.from_square


def _square_after(square: chess.Square, delta_file: int, delta_rank: int) -> chess.Square:
    file_index = chess.square_file(square) + delta_file
    rank_index = chess.square_rank(square) + delta_rank
    if not 0 <= file_index < 8 or not 0 <= rank_index < 8:
        raise ValueError("action geometry leaves the board")
    return chess.square(file_index, rank_index)


def decode_action(board: chess.Board, action: int) -> chess.Move:
    """Decode ``action`` to its legal move in ``board`` or raise ``ValueError``."""
    _require_standard(board)
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise ValueError(f"action must be an integer, got {action!r}")
    if not 0 <= action < ACTION_SIZE:
        raise ValueError(f"action index {action} is outside [0, {ACTION_SIZE})")

    action = int(action)
    plane, from_square = divmod(action, 64)
    promotion: chess.PieceType | None = None
    if plane < 56:
        direction_index, distance_index = divmod(plane, 7)
        file_step, rank_step = _LINEAR_DIRECTIONS[direction_index]
        to_square = _square_after(
            from_square, file_step * (distance_index + 1), rank_step * (distance_index + 1)
        )
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7):
            promotion = chess.QUEEN
    elif plane < 64:
        delta_file, delta_rank = _KNIGHT_DELTAS[plane - 56]
        to_square = _square_after(from_square, delta_file, delta_rank)
    else:
        promotion = _UNDERPROMOTION_PIECES[(plane - 64) // 3]
        direction_index = (plane - 64) % 3
        forward = 1 if board.turn == chess.WHITE else -1
        delta_file, delta_rank = ((-forward, forward), (0, forward), (forward, forward))[direction_index]
        to_square = _square_after(from_square, delta_file, delta_rank)

    move = chess.Move(from_square, to_square, promotion=promotion)
    if move not in board.legal_moves:
        raise ValueError(f"action index {action} is not legal in position {board.fen()}")
    return move


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Return a boolean az73-v1 mask with ``True`` exactly at legal actions."""
    _require_standard(board)
    mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        action = encode_move(board, move)
        if mask[action]:
            raise ValueError(f"legal action collision at index {action}")
        mask[action] = True
    return mask
