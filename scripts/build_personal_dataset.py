#!/usr/bin/env python3
"""Build the Chessy personal-policy dataset from game and move quality."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from filter_quality_games import _read_records, _write_pgn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-quality", type=Path, required=True)
    parser.add_argument("--move-quality", type=Path, required=True)
    parser.add_argument("--full-game-threshold", type=float, default=82.0)
    parser.add_argument("--good-move-threshold", type=float, default=85.0)
    parser.add_argument("--chess-com-pgn", type=Path, required=True)
    parser.add_argument("--chess-com-user", required=True)
    parser.add_argument("--lichess-pgn", type=Path, required=True)
    parser.add_argument("--lichess-user", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/personal"))
    args = parser.parse_args()

    with args.game_quality.open(encoding="utf-8") as source:
        game_rows = list(csv.DictReader(source))
    game_accuracy = {int(row["index"]): float(row["accuracy"]) for row in game_rows}
    full_game_ids = {
        index
        for index, accuracy in game_accuracy.items()
        if accuracy >= args.full_game_threshold
    }

    chess_records, chess_games, _ = _read_records(
        args.chess_com_pgn, "chess.com", args.chess_com_user, 0
    )
    _, lichess_games, _ = _read_records(
        args.lichess_pgn, "lichess", args.lichess_user, len(chess_records)
    )
    games = chess_games | lichess_games

    args.output.mkdir(parents=True, exist_ok=True)
    full_games_path = args.output / "full_style_games.pgn"
    _write_pgn(full_games_path, [games[index] for index in sorted(full_game_ids)])

    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    samples_path = args.output / "policy_samples.jsonl"
    with args.move_quality.open(encoding="utf-8") as source, samples_path.open(
        "w", encoding="utf-8"
    ) as output:
        for row in csv.DictReader(source):
            game_index = int(row["game_index"])
            move_accuracy = float(row["accuracy"])
            if game_index in full_game_ids:
                sample_kind = "full_game"
            elif move_accuracy >= args.good_move_threshold:
                sample_kind = "good_move"
            else:
                continue
            sample = {
                "game_index": game_index,
                "source": row["source"],
                "date": row["date"],
                "url": row["url"],
                "ply": int(row["ply"]),
                "move_number": int(row["move_number"]),
                "color": row["color"],
                "fen": row["fen_before"],
                "move_uci": row["move_uci"],
                "move_san": row["move_san"],
                "move_accuracy": move_accuracy,
                "game_accuracy": game_accuracy[game_index],
                "sample_kind": sample_kind,
            }
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[sample_kind] += 1
            source_counts[row["source"]] += 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "full_game_accuracy_threshold": args.full_game_threshold,
        "good_move_accuracy_threshold": args.good_move_threshold,
        "full_games": len(full_game_ids),
        "policy_samples": sum(counts.values()),
        "samples_by_kind": dict(counts),
        "samples_by_source": dict(source_counts),
        "files": {
            "full_games": full_games_path.name,
            "policy_samples": samples_path.name,
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Built {len(full_game_ids)} full games and {sum(counts.values())} policy samples"
    )


if __name__ == "__main__":
    main()
