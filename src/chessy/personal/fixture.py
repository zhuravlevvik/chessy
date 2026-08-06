"""Generate ignored, runnable fixtures for the personal-training smoke preset."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import chess
import torch

from chessy.model import ChessyModel, ModelConfig, load_model_export
from chessy.model.export import export_model
from chessy.personal.builder import build_personal_dataset


def _pgn(white: str, black: str, result: str) -> str:
    return f'''[Event "personal smoke fixture"]
[White "{white}"]
[Black "{black}"]
[Result "{result}"]
[UTCDate "2026.01.01"]

1. e4 e5 2. Nf3 Nc6 {result}

'''


def _row(index: int, source: str, color: str, move: str) -> dict[str, object]:
    board = chess.Board()
    if color == "black":
        board.push_uci("e2e4")
    return {
        "game_index": index, "source": source, "date": "2026.01.01", "url": "fixture",
        "ply": 2 if color == "black" else 1, "move_number": 1, "color": color,
        "fen": board.fen(), "move_uci": move, "move_san": "e5" if color == "black" else "e4",
        "move_accuracy": 90.0, "game_accuracy": 90.0, "sample_kind": "full_game",
    }


def prepare_smoke_fixture(root: Path) -> dict[str, str]:
    root = Path(root).resolve()
    fixture = root / "runs" / "personal-smoke-fixture"
    raw = fixture / "raw"; splits = fixture / "splits"
    raw.mkdir(parents=True, exist_ok=True); splits.mkdir(parents=True, exist_ok=True)
    chess_com = raw / "chess_com.pgn"
    lichess = raw / "lichess.pgn"
    quality = raw / "game_quality.csv"
    chess_com.write_text(_pgn("mu1876", "other", "1-0") + _pgn("mu1876", "other", "0-1"), encoding="utf-8")
    lichess.write_text(_pgn("mu1878", "other", "1/2-1/2"), encoding="utf-8")
    with quality.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "result"]); writer.writeheader()
        writer.writerows([{"index": 0, "result": "1-0"}, {"index": 1, "result": "0-1"}, {"index": 2, "result": "1/2-1/2"}])
    rows = {
        "train": [_row(0, "chess.com", "white", "e2e4")],
        "val": [_row(1, "chess.com", "white", "e2e4")],
        "test": [_row(2, "lichess", "white", "e2e4")],
    }
    for split, values in rows.items():
        (splits / f"{split}.jsonl").write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
    split_manifest = splits / "manifest.json"
    split_manifest.write_text(json.dumps({"splits": {name: {"file": f"{name}.jsonl", "samples": 1, "games": 1} for name in rows}}), encoding="utf-8")
    immutable = build_personal_dataset(splits=split_manifest, chess_com_pgn=chess_com, lichess_pgn=lichess, game_quality=quality, output=fixture / "encoded", segment_samples=2)
    stable_manifest = immutable.parent / "personal-dataset-fixture.json"
    if not stable_manifest.exists() or stable_manifest.read_bytes() != immutable.read_bytes():
        shutil.copyfile(immutable, stable_manifest)

    base = fixture / "base"
    if base.exists():
        load_model_export(base, device="cpu")
        base_manifest = json.loads((base / "manifest.json").read_text())
        if base_manifest.get("metadata", {}).get("role") != "fixture":
            raise ValueError("existing smoke base is not marked as a fixture")
    else:
        torch.manual_seed(7)
        model = ChessyModel(ModelConfig(channels=8, residual_blocks=1, group_norm_groups=8, value_channels=8, value_hidden=16))
        export_model(model, base, metadata={"role": "fixture", "name": "personal smoke fixture"})
        base_manifest = json.loads((base / "manifest.json").read_text())
    return {
        "dataset_manifest": str(stable_manifest.relative_to(root)),
        "dataset_fingerprint": json.loads(stable_manifest.read_text())["content_fingerprint"],
        "base_export": str(base.relative_to(root)),
        "base_weights_sha256": base_manifest["weights"]["sha256"],
    }
