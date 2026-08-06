"""Rebuild historical samples with true PGN history, never from isolated FENs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chessy.chess import ChessEnvironment
from chessy.config.canonical import canonical_json, fingerprint
from chessy.data import read_records
from chessy.encoding import decode_action, encode_board, encode_move
from chessy.replay.codec import encode_board as pack_board
from chessy.personal.segment import ENUMS, FORMAT, sha256, write_segment


def _value_class(result: str, turn: chess.Color) -> int:
    if result == "1/2-1/2":
        return 1
    if result == "1-0":
        return 2 if turn == chess.WHITE else 0
    if result == "0-1":
        return 2 if turn == chess.BLACK else 0
    raise ValueError(f"unfinished or unknown game result: {result!r}")


def _phase(board: chess.Board, ply: int) -> int:
    if len(board.piece_map()) <= 10:
        return ENUMS["phase"]["endgame"]
    if ply <= 20:
        return ENUMS["phase"]["opening"]
    return ENUMS["phase"]["middlegame"]


def _sample_id(row: dict[str, Any], action: int, packed: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(f"{row['game_index']}:{row['ply']}:{action}:".encode())
    digest.update(packed.tobytes())
    return digest.hexdigest()


def _safe_input(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input must be a regular file: {path}")
    return path


def build_personal_dataset(*, splits: Path, chess_com_pgn: Path, lichess_pgn: Path, game_quality: Path, output: Path, segment_samples: int = 16384, chess_com_user: str = "mu1876", lichess_user: str = "mu1878") -> Path:
    """Create immutable encoded segments and return their immutable manifest path."""
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    splits, chess_com_pgn, lichess_pgn, game_quality = map(_safe_input, (splits, chess_com_pgn, lichess_pgn, game_quality))
    try:
        split_manifest = json.loads(splits.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid split manifest") from exc
    chess_records, chess_games, _ = read_records(chess_com_pgn, "chess.com", chess_com_user, 0)
    lichess_records, lichess_games, _ = read_records(lichess_pgn, "lichess", lichess_user, len(chess_records))
    records = {record.index: record for record in chess_records + lichess_records}
    games = chess_games | lichess_games
    quality: dict[int, dict[str, str]] = {}
    with game_quality.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            index = int(row["index"])
            if index in quality:
                raise ValueError(f"duplicate quality game index {index}")
            quality[index] = row

    all_rows: dict[str, list[dict[str, Any]]] = {}
    assigned_games: set[int] = set()
    for split in ("train", "val", "test"):
        spec = split_manifest.get("splits", {}).get(split, {})
        file_name = spec.get("file")
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise ValueError("unsafe split filename")
        rows: list[dict[str, Any]] = []
        split_file = _safe_input(splits.parent / file_name)
        if split_file.resolve().parent != splits.parent.resolve():
            raise ValueError("split file must remain beside its manifest")
        with split_file.open(encoding="utf-8") as source:
            for source_index, line in enumerate(source):
                row = json.loads(line)
                row["_split_row_index"] = source_index
                game_index = int(row["game_index"])
                rows.append(row)
        games_here = {int(row["game_index"]) for row in rows}
        if assigned_games & games_here:
            raise ValueError(f"game(s) appear in multiple splits: {sorted(assigned_games & games_here)[:3]}")
        assigned_games |= games_here
        if spec.get("samples") != len(rows) or spec.get("games") != len(games_here):
            raise ValueError(f"upstream {split} split counts do not match rows")
        all_rows[split] = sorted(rows, key=lambda item: (int(item["game_index"]), int(item["ply"])))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_segments: dict[str, list[dict[str, Any]]] = {name: [] for name in all_rows}
    split_info: dict[str, dict[str, Any]] = {}
    all_histograms: dict[str, Counter[str]] = {name: Counter() for name in ENUMS}
    for split, rows in all_rows.items():
        by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_game[int(row["game_index"])].append(row)
        encoded_by_game: list[list[dict[str, Any]]] = []
        for game_index in sorted(by_game):
            record, game = records.get(game_index), games.get(game_index)
            if record is None or game is None or game_index not in quality:
                raise ValueError(f"missing PGN or quality entry for game {game_index}")
            qrow = quality[game_index]
            header_result = game.headers.get("Result", "")
            if header_result != qrow.get("result"):
                raise ValueError(f"conflicting result for game {game_index}")
            target_rows = {int(row["ply"]): row for row in by_game[game_index]}
            if len(target_rows) != len(by_game[game_index]):
                raise ValueError(f"duplicate split ply in game {game_index}")
            env = ChessEnvironment(game.board())
            samples: list[dict[str, Any]] = []
            for ply, move in enumerate(game.mainline_moves(), start=1):
                board = env.board
                if ply in target_rows:
                    row = target_rows[ply]
                    if board.fen() != row["fen"]:
                        raise ValueError(f"FEN mismatch for game {game_index} ply {ply}")
                    if ("white" if board.turn else "black") != row["color"]:
                        raise ValueError(f"side-to-move mismatch for game {game_index} ply {ply}")
                    if move.uci() != row["move_uci"] or move not in board.legal_moves:
                        raise ValueError(f"illegal/mismatched target for game {game_index} ply {ply}")
                    action = encode_move(board, move)
                    if decode_action(board, action) != move:
                        raise ValueError(f"action round-trip failed for game {game_index} ply {ply}")
                    packed = pack_board(encode_board(env.history(8)))
                    legal = np.array(sorted(encode_move(board, legal_move) for legal_move in board.legal_moves), dtype=np.uint16)
                    value = _value_class(header_result, board.turn)
                    meta = {
                        "sample_id": _sample_id(row, action, packed), "split_row_index": row["_split_row_index"],
                        "game_index": game_index, "source": row["source"], "date": row["date"], "url": row.get("url", ""),
                        "ply": ply, "move_number": row["move_number"], "color": row["color"], "fen": row["fen"],
                        "move_uci": row["move_uci"], "move_san": row["move_san"], "move_accuracy": row["move_accuracy"],
                        "game_accuracy": row["game_accuracy"], "sample_kind": row["sample_kind"], "game_result": header_result,
                        "target_action": action, "value_class": value, "board_checksum": hashlib.sha256(packed.tobytes()).hexdigest(),
                    }
                    samples.append({"board": packed, "legal": legal, "target": action, "value": value, "meta": meta, "phase": _phase(board, ply)})
                env.push(move)
            if len(samples) != len(target_rows):
                raise ValueError(f"target ply absent from PGN game {game_index}")
            encoded_by_game.append(samples)
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for game_samples in encoded_by_game:
            if current and len(current) + len(game_samples) > segment_samples:
                chunks.append(current); current = []
            # An unusually long game forms its own legal hard-limit exception.
            current.extend(game_samples)
        if current:
            chunks.append(current)
        split_hist: dict[str, Counter[str]] = {name: Counter() for name in ENUMS}
        total_samples = 0
        for ordinal, chunk in enumerate(chunks):
            offsets = [0]; legal_actions: list[np.ndarray] = []
            for sample in chunk:
                legal_actions.append(sample["legal"]); offsets.append(offsets[-1] + len(sample["legal"]))
            arrays = {
                "boards": np.stack([sample["board"] for sample in chunk]).astype(np.uint8),
                "legal_offsets": np.asarray(offsets, dtype=np.int64), "legal_actions": np.concatenate(legal_actions).astype(np.uint16),
                "target_action": np.asarray([sample["target"] for sample in chunk], dtype=np.uint16), "value_class": np.asarray([sample["value"] for sample in chunk], dtype=np.uint8),
                "game_index": np.asarray([sample["meta"]["game_index"] for sample in chunk], dtype=np.uint32), "ply": np.asarray([sample["meta"]["ply"] for sample in chunk], dtype=np.uint16),
                "sample_kind": np.asarray([ENUMS["sample_kind"][sample["meta"]["sample_kind"]] for sample in chunk], dtype=np.uint8),
                "source": np.asarray([ENUMS["source"][sample["meta"]["source"]] for sample in chunk], dtype=np.uint8), "color": np.asarray([ENUMS["color"][sample["meta"]["color"]] for sample in chunk], dtype=np.uint8),
                "phase": np.asarray([sample["phase"] for sample in chunk], dtype=np.uint8),
            }
            segment = write_segment(output, split, ordinal, arrays, [sample["meta"] for sample in chunk])
            checked_manifest = json.loads((segment / "manifest.json").read_text())
            manifest_segments[split].append({"path": str(segment.relative_to(output)), "manifest_checksum": sha256(segment / "checksums.sha256"), "sample_count": len(chunk), "payload_fingerprint": checked_manifest["payload_fingerprint"]})
            total_samples += len(chunk)
            for name, mapping in ENUMS.items():
                inverse = {value: key for key, value in mapping.items()}
                split_hist[name].update(inverse[int(value)] for value in arrays[name])
        all_histograms = {name: all_histograms[name] + split_hist[name] for name in ENUMS}
        dates = [row["date"].replace(".", "-") for row in rows]
        split_info[split] = {"segments": manifest_segments[split], "sample_count": total_samples, "game_count": len(by_game), "histograms": {name: dict(sorted(values.items())) for name, values in split_hist.items()}, "date_from": min(dates), "date_to": max(dates)}

    sources = {"splits": str(splits), "splits_sha256": sha256(splits), "chess_com_pgn": str(chess_com_pgn), "chess_com_pgn_sha256": sha256(chess_com_pgn), "lichess_pgn": str(lichess_pgn), "lichess_pgn_sha256": sha256(lichess_pgn), "game_quality": str(game_quality), "game_quality_sha256": sha256(game_quality)}
    payload: dict[str, Any] = {"format": FORMAT, "encodings": {"board": "board119-v1", "action": "az73-v1"}, "enums": ENUMS, "sources": sources, "accounts": {"chess.com": chess_com_user, "lichess": lichess_user}, "thresholds": {"full_game": 82.0, "good_move": 85.0}, "splits": split_info, "histograms": {name: dict(sorted(values.items())) for name, values in all_histograms.items()}, "frozen_test_fingerprint": fingerprint(split_info["test"])}
    payload["content_fingerprint"] = fingerprint(payload)
    payload["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifests = output / "manifests"; manifests.mkdir(exist_ok=True)
    destination = manifests / f"personal-dataset-{payload['content_fingerprint']}.json"
    if destination.exists() and destination.read_bytes() != canonical_json(payload):
        # created_at is descriptive rather than content-addressed; immutable bytes must still agree.
        existing = json.loads(destination.read_text())
        if {key: value for key, value in existing.items() if key != "created_at"} != {key: value for key, value in payload.items() if key != "created_at"}:
            raise FileExistsError("existing personal manifest has conflicting contents")
    else:
        destination.write_bytes(canonical_json(payload))
    return destination
