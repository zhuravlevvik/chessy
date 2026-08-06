"""Versioned reader matching the historical quality pipeline's game indexing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn


@dataclass(frozen=True)
class GameRecord:
    index: int
    source: str
    username: str
    headers: dict[str, str]
    moves: tuple[str, ...]


def read_records(
    source_path: Path, source: str, username: str, start_index: int
) -> tuple[list[GameRecord], dict[int, chess.pgn.Game], int]:
    """Read exactly the standard-start games accepted by quality filtering.

    The next source must receive ``start_index=len(previous_records)``.  This
    deliberately preserves the old, published game-index namespace.
    """
    records: list[GameRecord] = []
    games: dict[int, chess.pgn.Game] = {}
    skipped = 0
    with Path(source_path).open(encoding="utf-8", errors="replace") as pgn:
        while game := chess.pgn.read_game(pgn):
            if game.headers.get("Variant", "Standard").casefold() != "standard":
                skipped += 1
                continue
            names = (game.headers.get("White", "").casefold(), game.headers.get("Black", "").casefold())
            if username.casefold() not in names:
                skipped += 1
                continue
            board = game.board()
            if board.fen() != chess.Board().fen():
                skipped += 1
                continue
            moves: list[str] = []
            try:
                for move in game.mainline_moves():
                    moves.append(move.uci())
                    board.push(move)
            except (ValueError, AssertionError):
                skipped += 1
                continue
            index = start_index + len(records)
            records.append(GameRecord(index, source, username, dict(game.headers), tuple(moves)))
            games[index] = game
    return records, games, skipped
