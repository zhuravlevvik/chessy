from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import chess
import numpy as np


@dataclass(frozen=True, slots=True)
class StartPosition:
    stage: str
    source_kind: str
    fen: str
    seed: int
    metadata: dict[str, object]
    max_plies: int


class PositionSource(Protocol):
    def sample(self, rng: np.random.Generator) -> StartPosition: ...


def _valid(board: chess.Board) -> bool:
    return not board.chess960 and board.is_valid() and board.outcome(claim_draw=True) is None and len(board.pieces(chess.KING, chess.WHITE)) == len(board.pieces(chess.KING, chess.BLACK)) == 1


def _empty_position(rng: np.random.Generator) -> tuple[chess.Board, list[int]]:
    board = chess.Board(None)
    squares = list(range(64))
    rng.shuffle(squares)
    # Kings cannot touch. Try a shuffled stream to preserve reproducibility.
    white = squares.pop()
    black = next(square for square in reversed(squares) if chess.square_distance(white, square) > 1)
    squares.remove(black)
    board.set_piece_at(white, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(black, chess.Piece(chess.KING, chess.BLACK))
    return board, squares


class EndgameSource:
    """KQK, KRK, KPvK and small pawn endings, with bounded rejection."""
    def __init__(self, *, max_plies: int = 160, attempts: int = 200) -> None:
        self.max_plies, self.attempts = max_plies, attempts

    def sample(self, rng: np.random.Generator) -> StartPosition:
        seed = int(rng.integers(0, 2**63))
        local = np.random.default_rng(seed)
        kind = ("kqk", "krk", "kpvk", "pawns")[int(local.integers(0, 4))]
        for _ in range(self.attempts):
            board, squares = _empty_position(local)
            strong = chess.WHITE if bool(local.integers(0, 2)) else chess.BLACK
            if kind == "kqk":
                board.set_piece_at(squares.pop(), chess.Piece(chess.QUEEN, strong))
            elif kind == "krk":
                board.set_piece_at(squares.pop(), chess.Piece(chess.ROOK, strong))
            elif kind == "kpvk":
                pawn_square = next(square for square in squares if 0 < chess.square_rank(square) < 7)
                board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, strong))
            else:
                for color in (chess.WHITE, chess.BLACK):
                    count = int(local.integers(1, 4))
                    candidates = [square for square in squares if 0 < chess.square_rank(square) < 7]
                    for square in candidates[:count]:
                        squares.remove(square)
                        board.set_piece_at(square, chess.Piece(chess.PAWN, color))
            board.turn = chess.WHITE if bool(local.integers(0, 2)) else chess.BLACK
            board.clear_stack()
            if _valid(board):
                return StartPosition("endgames", kind, board.fen(), seed, {"strong_color": "white" if strong else "black"}, self.max_plies)
        raise RuntimeError("endgame generator exhausted bounded rejection attempts")


class ReducedSource:
    def __init__(self, *, max_plies: int = 160, max_material_imbalance: int = 12, attempts: int = 500) -> None:
        self.max_plies, self.max_material_imbalance, self.attempts = max_plies, max_material_imbalance, attempts

    def sample(self, rng: np.random.Generator) -> StartPosition:
        seed = int(rng.integers(0, 2**63)); local = np.random.default_rng(seed)
        pieces = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        for _ in range(self.attempts):
            board, squares = _empty_position(local); totals = {chess.WHITE: 0, chess.BLACK: 0}
            target = int(local.integers(4, 15))  # plus two kings gives 6..16 pieces
            for _ in range(target):
                if not squares: break
                color = chess.WHITE if bool(local.integers(0, 2)) else chess.BLACK
                piece = pieces[int(local.integers(0, len(pieces)))]
                candidates = squares if piece != chess.PAWN else [s for s in squares if 0 < chess.square_rank(s) < 7]
                if not candidates: continue
                square = candidates[int(local.integers(0, len(candidates)))]; squares.remove(square)
                board.set_piece_at(square, chess.Piece(piece, color)); totals[color] += values[piece]
            board.turn = chess.WHITE if bool(local.integers(0, 2)) else chess.BLACK; board.clear_stack()
            if abs(totals[chess.WHITE] - totals[chess.BLACK]) <= self.max_material_imbalance and _valid(board):
                return StartPosition("reduced", "random-reduced", board.fen(), seed, {"material": {"white": totals[chess.WHITE], "black": totals[chess.BLACK]}}, self.max_plies)
        raise RuntimeError("reduced generator exhausted bounded rejection attempts")


_OPENING_PREFIXES: tuple[tuple[str, ...], ...] = (("e2e4", "e7e5", "g1f3", "b8c6"), ("d2d4", "d7d5", "c2c4"), ("c2c4", "e7e5", "b1c3"))


class FullSource:
    def __init__(self, *, max_plies: int = 160, random_prefix_plies: int = 6) -> None:
        self.max_plies, self.random_prefix_plies = max_plies, random_prefix_plies

    def sample(self, rng: np.random.Generator) -> StartPosition:
        seed = int(rng.integers(0, 2**63)); local = np.random.default_rng(seed); board = chess.Board()
        mode = int(local.integers(0, 3)); moves: list[str] = []
        if mode == 1:
            moves = list(_OPENING_PREFIXES[int(local.integers(0, len(_OPENING_PREFIXES)))])
            for uci in moves: board.push_uci(uci)
            kind = "opening-prefix"
        elif mode == 2:
            for _ in range(int(local.integers(1, self.random_prefix_plies + 1))):
                move = tuple(board.legal_moves)[int(local.integers(0, board.legal_moves.count()))]
                moves.append(move.uci()); board.push(move)
            kind = "random-prefix"
        else: kind = "initial"
        return StartPosition("full", kind, board.fen(), seed, {"moves": moves}, self.max_plies)


def source_for_stage(stage: str, *, max_plies: int = 160, max_material_imbalance: int = 12) -> PositionSource:
    if stage == "endgames": return EndgameSource(max_plies=max_plies)
    if stage == "reduced": return ReducedSource(max_plies=max_plies, max_material_imbalance=max_material_imbalance)
    if stage == "full": return FullSource(max_plies=max_plies)
    raise ValueError(f"unknown curriculum stage: {stage}")
