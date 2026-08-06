#!/usr/bin/env python3
"""Evaluate a player's standard games with Stockfish and filter by accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn


@dataclass(frozen=True)
class GameRecord:
    index: int
    source: str
    username: str
    headers: dict[str, str]
    moves: tuple[str, ...]


@dataclass(frozen=True)
class GameQuality:
    index: int
    source: str
    date: str
    url: str
    color: str
    result: str
    time_control: str
    rating: str
    user_moves: int
    accuracy: float
    mean_move_accuracy: float
    acpl: float


@dataclass(frozen=True)
class MoveQuality:
    game_index: int
    source: str
    date: str
    url: str
    ply: int
    move_number: int
    color: str
    fen_before: str
    move_uci: str
    move_san: str
    accuracy: float
    cp_loss: int


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _win_percent(cp: int) -> float:
    cp = int(_clamp(cp, -100_000, 100_000))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def _move_accuracy(before: float, after: float) -> float:
    if after >= before:
        return 100.0
    win_diff = before - after
    raw = 103.1668100711649 * math.exp(-0.04354415386753951 * win_diff)
    raw -= 3.166924740191411
    return _clamp(raw + 1.0, 0.0, 100.0)


def _terminal_cp(board: chess.Board) -> int:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0
    return 100_000 if outcome.winner == chess.WHITE else -100_000


def _evaluate_cp(
    board: chess.Board, engine: chess.engine.SimpleEngine, nodes: int
) -> int:
    if board.is_game_over(claim_draw=True):
        return _terminal_cp(board)
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    return info["score"].pov(chess.WHITE).score(mate_score=100_000) or 0


def _game_accuracy(
    win_percents: list[float], user_color: chess.Color
) -> tuple[float, float, list[float]]:
    plies = len(win_percents) - 1
    window_size = max(2, min(8, plies // 10))
    window_size = min(window_size, len(win_percents))
    windows = [win_percents[:window_size]] * max(0, window_size - 2)
    windows.extend(
        win_percents[start : start + window_size]
        for start in range(len(win_percents) - window_size + 1)
    )
    weights = [
        _clamp(statistics.pstdev(window), 0.5, 12.0) for window in windows
    ]

    accuracies: list[float] = []
    selected_weights: list[float] = []
    for ply, (before_white, after_white, weight) in enumerate(
        zip(win_percents, win_percents[1:], weights)
    ):
        mover = chess.WHITE if ply % 2 == 0 else chess.BLACK
        if mover != user_color:
            continue
        if user_color == chess.WHITE:
            before, after = before_white, after_white
        else:
            before, after = 100 - before_white, 100 - after_white
        accuracies.append(_move_accuracy(before, after))
        selected_weights.append(weight)

    weighted = sum(a * w for a, w in zip(accuracies, selected_weights)) / sum(
        selected_weights
    )
    if any(accuracy == 0 for accuracy in accuracies):
        harmonic = 0.0
    else:
        harmonic = len(accuracies) / sum(1 / accuracy for accuracy in accuracies)
    return (weighted + harmonic) / 2, statistics.fmean(accuracies), accuracies


def _analyse(
    record: GameRecord, engine: chess.engine.SimpleEngine, nodes: int
) -> tuple[GameQuality, list[MoveQuality]]:
    board = chess.Board()
    evaluations = [0]
    positions: list[tuple[str, str, str]] = []
    for uci in record.moves:
        move = chess.Move.from_uci(uci)
        positions.append((board.fen(), uci, board.san(move)))
        board.push(move)
        evaluations.append(_evaluate_cp(board, engine, nodes))

    white_name = record.headers.get("White", "").casefold()
    user_color = chess.WHITE if white_name == record.username.casefold() else chess.BLACK
    win_percents = [_win_percent(cp) for cp in evaluations]
    accuracy, mean_accuracy, move_accuracies = _game_accuracy(win_percents, user_color)

    losses: list[int] = []
    for ply, (before_cp, after_cp) in enumerate(zip(evaluations, evaluations[1:])):
        mover = chess.WHITE if ply % 2 == 0 else chess.BLACK
        if mover != user_color:
            continue
        if user_color == chess.WHITE:
            losses.append(max(0, before_cp - after_cp))
        else:
            losses.append(max(0, after_cp - before_cp))

    rating_key = "WhiteElo" if user_color == chess.WHITE else "BlackElo"
    game_quality = GameQuality(
        index=record.index,
        source=record.source,
        date=record.headers.get("UTCDate", record.headers.get("Date", "")),
        url=record.headers.get("Link", record.headers.get("Site", "")),
        color="white" if user_color == chess.WHITE else "black",
        result=record.headers.get("Result", ""),
        time_control=record.headers.get("TimeControl", ""),
        rating=record.headers.get(rating_key, ""),
        user_moves=len(losses),
        accuracy=round(accuracy, 4),
        mean_move_accuracy=round(mean_accuracy, 4),
        acpl=round(statistics.fmean(losses), 4),
    )
    move_qualities: list[MoveQuality] = []
    user_move_index = 0
    for ply, (fen, uci, san) in enumerate(positions):
        mover = chess.WHITE if ply % 2 == 0 else chess.BLACK
        if mover != user_color:
            continue
        move_qualities.append(
            MoveQuality(
                game_index=record.index,
                source=record.source,
                date=game_quality.date,
                url=game_quality.url,
                ply=ply + 1,
                move_number=ply // 2 + 1,
                color="white" if user_color == chess.WHITE else "black",
                fen_before=fen,
                move_uci=uci,
                move_san=san,
                accuracy=round(move_accuracies[user_move_index], 4),
                cp_loss=losses[user_move_index],
            )
        )
        user_move_index += 1
    return game_quality, move_qualities


def _analyse_chunk(
    records: list[GameRecord], engine_path: str, nodes: int, hash_mb: int
) -> tuple[list[GameQuality], list[MoveQuality], list[dict[str, Any]]]:
    qualities: list[GameQuality] = []
    move_qualities: list[MoveQuality] = []
    errors: list[dict[str, Any]] = []
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 1, "Hash": hash_mb})
    try:
        for record in records:
            try:
                game_quality, game_moves = _analyse(record, engine, nodes)
                qualities.append(game_quality)
                move_qualities.extend(game_moves)
            except Exception as error:
                errors.append({"index": record.index, "error": repr(error)})
    finally:
        engine.quit()
    return qualities, move_qualities, errors


def _read_records(
    source_path: Path, source: str, username: str, start_index: int
) -> tuple[list[GameRecord], dict[int, chess.pgn.Game], int]:
    records: list[GameRecord] = []
    games: dict[int, chess.pgn.Game] = {}
    skipped = 0
    with source_path.open(encoding="utf-8", errors="replace") as pgn:
        while game := chess.pgn.read_game(pgn):
            variant = game.headers.get("Variant", "Standard").casefold()
            if variant != "standard":
                skipped += 1
                continue
            white = game.headers.get("White", "").casefold()
            black = game.headers.get("Black", "").casefold()
            if username.casefold() not in (white, black):
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
            except ValueError:
                skipped += 1
                continue
            index = start_index + len(records)
            records.append(
                GameRecord(
                    index=index,
                    source=source,
                    username=username,
                    headers=dict(game.headers),
                    moves=tuple(moves),
                )
            )
            games[index] = game
    return records, games, skipped


def _write_csv(path: Path, rows: list[GameQuality] | list[MoveQuality]) -> None:
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_pgn(path: Path, games: list[chess.pgn.Game]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for game in games:
            exporter = chess.pgn.StringExporter(
                headers=True, variations=False, comments=True
            )
            output.write(game.accept(exporter))
            output.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chess-com-pgn", type=Path, required=True)
    parser.add_argument("--chess-com-user", required=True)
    parser.add_argument("--lichess-pgn", type=Path, required=True)
    parser.add_argument("--lichess-user", required=True)
    parser.add_argument("--threshold", type=float, default=85.0)
    parser.add_argument("--min-user-moves", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--output", type=Path, default=Path("data/quality"))
    args = parser.parse_args()

    engine_path = shutil.which("stockfish")
    if engine_path is None:
        raise SystemExit("Stockfish was not found on PATH")

    chess_records, chess_games, chess_skipped = _read_records(
        args.chess_com_pgn, "chess.com", args.chess_com_user, 0
    )
    lichess_records, lichess_games, lichess_skipped = _read_records(
        args.lichess_pgn, "lichess", args.lichess_user, len(chess_records)
    )
    records = chess_records + lichess_records
    games: dict[int, chess.pgn.Game] = chess_games | lichess_games
    standard_games = len(records)
    records = [record for record in records if len(record.moves) // 2 >= args.min_user_moves]
    excluded_too_short = standard_games - len(records)
    if args.max_games is not None:
        records = records[: args.max_games]

    print(f"Analysing {len(records)} standard games with {args.workers} workers")
    qualities: list[GameQuality] = []
    move_qualities: list[MoveQuality] = []
    errors: list[dict[str, Any]] = []
    chunks = [
        records[start : start + args.chunk_size]
        for start in range(0, len(records), args.chunk_size)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _analyse_chunk, chunk, engine_path, args.nodes, args.hash_mb
            ): chunk
            for chunk in chunks
        }
        completed_games = 0
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                chunk_qualities, chunk_moves, chunk_errors = future.result()
                qualities.extend(chunk_qualities)
                move_qualities.extend(chunk_moves)
                errors.extend(chunk_errors)
            except Exception as error:
                errors.extend(
                    {"index": record.index, "error": repr(error)} for record in chunk
                )
            completed_games += len(chunk)
            print(f"Completed {completed_games}/{len(records)}", flush=True)

    qualities.sort(key=lambda quality: quality.index)
    move_qualities.sort(key=lambda move: (move.game_index, move.ply))
    retained = [quality for quality in qualities if quality.accuracy >= args.threshold]
    retained_games = [games[quality.index] for quality in retained]

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "game_quality.csv", qualities)
    _write_csv(args.output / "move_quality.csv", move_qualities)
    _write_pgn(args.output / f"personal_accuracy_{args.threshold:g}.pgn", retained_games)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "Stockfish 18",
        "nodes_per_position": args.nodes,
        "accuracy": "Lichess-compatible win-percentage accuracy",
        "threshold": args.threshold,
        "minimum_user_moves": args.min_user_moves,
        "standard_valid_games": standard_games,
        "excluded_too_short": excluded_too_short,
        "analysed_games": len(qualities),
        "analysed_user_moves": len(move_qualities),
        "retained_games": len(retained),
        "skipped_nonstandard_or_invalid": chess_skipped + lichess_skipped,
        "errors": errors,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Retained {len(retained)}/{len(qualities)} games at >= {args.threshold:g}%")


if __name__ == "__main__":
    main()
