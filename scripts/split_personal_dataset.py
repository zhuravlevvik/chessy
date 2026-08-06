#!/usr/bin/env python3
"""Create chronological train/validation/test splits without game leakage."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")


def normalized_date(value: str) -> str:
    return value.replace(".", "-") if value else "0000-00-00"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/personal/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise SystemExit("train and validation ratios must be positive")
    if args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("train_ratio + val_ratio must be less than 1")

    samples_by_date: dict[str, int] = Counter()
    games_by_date: dict[str, set[int]] = defaultdict(set)
    game_date: dict[int, str] = {}
    total_samples = 0
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            sample = json.loads(line)
            game_index = int(sample["game_index"])
            date = normalized_date(sample.get("date", ""))
            previous_date = game_date.setdefault(game_index, date)
            if previous_date != date:
                raise ValueError(f"game {game_index} has conflicting dates")
            samples_by_date[date] += 1
            games_by_date[date].add(game_index)
            total_samples += 1

    ordered_dates = sorted(samples_by_date)
    train_target = total_samples * args.train_ratio
    val_target = total_samples * (args.train_ratio + args.val_ratio)
    date_split: dict[str, str] = {}
    cumulative = 0
    for date in ordered_dates:
        midpoint = cumulative + samples_by_date[date] / 2
        if midpoint < train_target:
            split = "train"
        elif midpoint < val_target:
            split = "val"
        else:
            split = "test"
        date_split[date] = split
        cumulative += samples_by_date[date]

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {name: args.output / f"{name}.jsonl" for name in SPLIT_NAMES}
    outputs = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    sample_counts: Counter[str] = Counter()
    kind_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_games: dict[str, set[int]] = defaultdict(set)
    split_dates: dict[str, set[str]] = defaultdict(set)
    try:
        with args.input.open(encoding="utf-8") as source:
            for line in source:
                sample = json.loads(line)
                game_index = int(sample["game_index"])
                date = game_date[game_index]
                split = date_split[date]
                outputs[split].write(line)
                sample_counts[split] += 1
                kind_counts[split][sample["sample_kind"]] += 1
                source_counts[split][sample["source"]] += 1
                split_games[split].add(game_index)
                split_dates[split].add(date)
    finally:
        for output in outputs.values():
            output.close()

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_games[left] & split_games[right]
        if overlap:
            raise RuntimeError(f"game leakage between {left} and {right}: {len(overlap)}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "chronological, grouped by game and calendar date",
        "requested_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(1 - args.train_ratio - args.val_ratio, 10),
        },
        "total_samples": total_samples,
        "splits": {},
    }
    for split in SPLIT_NAMES:
        dates = sorted(split_dates[split])
        manifest["splits"][split] = {
            "file": paths[split].name,
            "samples": sample_counts[split],
            "ratio": round(sample_counts[split] / total_samples, 10),
            "games": len(split_games[split]),
            "date_from": dates[0] if dates else None,
            "date_to": dates[-1] if dates else None,
            "samples_by_kind": dict(kind_counts[split]),
            "samples_by_source": dict(source_counts[split]),
        }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for split in SPLIT_NAMES:
        info = manifest["splits"][split]
        print(
            f"{split}: {info['samples']} samples, {info['games']} games, "
            f"{info['date_from']}..{info['date_to']}"
        )


if __name__ == "__main__":
    main()
