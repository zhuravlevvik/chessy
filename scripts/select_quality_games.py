#!/usr/bin/env python3
"""Rebuild a filtered PGN from previously computed game-quality metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from filter_quality_games import _read_records, _write_pgn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--chess-com-pgn", type=Path, required=True)
    parser.add_argument("--chess-com-user", required=True)
    parser.add_argument("--lichess-pgn", type=Path, required=True)
    parser.add_argument("--lichess-user", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chess_records, chess_games, _ = _read_records(
        args.chess_com_pgn, "chess.com", args.chess_com_user, 0
    )
    _, lichess_games, _ = _read_records(
        args.lichess_pgn, "lichess", args.lichess_user, len(chess_records)
    )
    games = chess_games | lichess_games

    with args.quality_csv.open(encoding="utf-8") as source:
        selected = [
            int(row["index"])
            for row in csv.DictReader(source)
            if float(row["accuracy"]) >= args.threshold
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_pgn(args.output, [games[index] for index in selected])
    print(f"Selected {len(selected)} games -> {args.output}")


if __name__ == "__main__":
    main()
