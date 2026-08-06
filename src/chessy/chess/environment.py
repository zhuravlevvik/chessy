"""Small, stateful wrapper around a standard :class:`chess.Board`."""

from __future__ import annotations

import chess


class ChessEnvironment:
    """Own a standard-chess board and its complete move history.

    The :attr:`board` property returns a copy, so callers cannot mutate the
    environment's state without going through its legality-checked methods.
    """

    def __init__(self, board: chess.Board | None = None) -> None:
        if board is not None and board.chess960:
            raise ValueError("ChessEnvironment supports standard chess only")
        self._initial_board = (
            chess.Board() if board is None else board.copy(stack=True)
        )
        self._board = self._initial_board.copy(stack=True)

    @classmethod
    def from_fen(cls, fen: str) -> "ChessEnvironment":
        """Create an environment whose history starts at ``fen``."""
        return cls(chess.Board(fen=fen, chess960=False))

    @property
    def board(self) -> chess.Board:
        """Return an independent copy of the current board and move stack."""
        return self._board.copy(stack=True)

    def reset(self) -> None:
        """Restore the position that was supplied at construction time."""
        self._board = self._initial_board.copy(stack=True)

    def copy(self) -> "ChessEnvironment":
        """Return an independent environment with the same current history."""
        duplicate = self.__class__.__new__(self.__class__)
        duplicate._initial_board = self._initial_board.copy(stack=True)
        duplicate._board = self._board.copy(stack=True)
        return duplicate

    def legal_moves(self) -> tuple[chess.Move, ...]:
        """Return all legal moves in the current position."""
        return tuple(self._board.legal_moves)

    def push(self, move: chess.Move) -> None:
        """Apply a legal move, rejecting moves unavailable in this position."""
        if move not in self._board.legal_moves:
            raise ValueError(f"illegal move {move.uci()} in position {self.fen()}")
        self._board.push(move)

    def push_uci(self, uci: str) -> chess.Move:
        """Parse, validate, apply, and return a UCI move for this position."""
        try:
            move = self._board.parse_uci(uci)
        except ValueError as exc:
            raise ValueError(f"invalid or illegal UCI move {uci!r}") from exc
        self.push(move)
        return move

    def fen(self) -> str:
        """Return the current FEN."""
        return self._board.fen()

    def is_terminal(self) -> bool:
        """Whether the game is over, including claimable draws."""
        return self._board.outcome(claim_draw=True) is not None

    def outcome(self) -> chess.Outcome | None:
        """Return the terminal outcome, including claimable draws when present."""
        return self._board.outcome(claim_draw=True)

    def terminal_value(self, perspective: chess.Color) -> float:
        """Return the terminal result from ``perspective`` as -1, 0, or +1."""
        outcome = self.outcome()
        if outcome is None:
            raise ValueError("terminal_value is undefined for a non-terminal position")
        if outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == perspective else -1.0

    def history(self, length: int = 8) -> tuple[chess.Board, ...]:
        """Return current-to-past boards, padding with the earliest position."""
        if length < 1:
            raise ValueError("history length must be at least 1")

        snapshot = self._board.copy(stack=True)
        positions: list[chess.Board] = []
        while len(positions) < length:
            positions.append(snapshot.copy(stack=True))
            if not snapshot.move_stack:
                break
            snapshot.pop()
        while len(positions) < length:
            positions.append(positions[-1].copy(stack=True))
        return tuple(positions)
