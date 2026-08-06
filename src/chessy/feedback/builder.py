"""Build a deterministic encoded feedback dataset from verified raw games."""
from __future__ import annotations

import hashlib
import os
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from chessy.config.canonical import canonical_json, fingerprint
from chessy.encoding import encode_board, encode_move
from chessy.feedback.raw import inspect_feedback_root, verify_feedback_game
from chessy.feedback.segment import ENUMS, FORMAT, sha256, write_feedback_segment
from chessy.replay.codec import encode_board as pack_board


def _phase(board, ply: int) -> int:  # type: ignore[no-untyped-def]
    if len(board.piece_map()) <= 10: return ENUMS["phase"]["endgame"]
    return ENUMS["phase"]["opening"] if ply <= 20 else ENUMS["phase"]["middlegame"]


def _value(result: str, human_color: str) -> int:
    if result == "draw": return ENUMS["value_class"]["draw"]
    return ENUMS["value_class"][result]


def build_feedback_dataset(*, input: Path, output: Path, sample_weight: float = 4.0, max_positions_per_game: int = 16, segment_samples: int = 16384) -> Path:
    if not 0 < float(sample_weight) < float("inf") or max_positions_per_game <= 0 or segment_samples <= 0: raise ValueError("feedback build parameters must be positive")
    source = Path(input)
    games = inspect_feedback_root(source)
    if not games: raise ValueError("feedback input contains no confirmed games")
    encoded_games: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for info in games:
        checked = verify_feedback_game(source / str(info["game_id"])); manifest = checked["manifest"]; board = checked["game"].board()
        targets = {int(row["ply"]): row for row in checked["samples"]}; samples: list[dict[str, Any]] = []
        for ply, move in enumerate(checked["game"].mainline_moves(), start=1):
            if ply in targets:
                row = targets[ply]
                packed = pack_board(encode_board(tuple(_history(board))))
                legal = np.asarray(sorted(encode_move(board, item) for item in board.legal_moves), dtype=np.uint16)
                action = encode_move(board, move)
                sample_id = row.get("sample_id") or hashlib.sha256(f"{manifest['game_id']}:{ply}:{move.uci()}:{action}".encode()).hexdigest()
                samples.append({"board": packed, "legal": legal, "target": action, "value": _value(str(row["result"]), str(manifest["human_color"])), "ply": ply, "color": ENUMS["color"][str(manifest["human_color"])], "phase": _phase(board, ply), "meta": {"sample_id": sample_id, "game_id": manifest["game_id"], "ply": ply, "color": manifest["human_color"], "phase": next(name for name, value in ENUMS["phase"].items() if value == _phase(board, ply)), "model_id": manifest["model"]["id"], "model_checksum": manifest["model"]["checksum"], "time_control": manifest["time_control"], "result": manifest["result"], "termination": manifest["termination"], "source": "human_online"}})
            board.push(move)
        # Preserve every confirmed decision. The per-game cap is an epoch-level
        # sampling rule and must not irreversibly discard later moves here.
        encoded_games.append((manifest, samples))
    rows = [row for _, group in encoded_games for row in group]
    if not rows: raise ValueError("confirmed games contain no human targets")
    output = Path(output)
    if output.resolve() == source.resolve() or output.resolve().is_relative_to(source.resolve()): raise ValueError("feedback output must be separate from immutable raw input")
    if output.exists() and (output.is_symlink() or not stat.S_ISDIR(output.lstat().st_mode)): raise ValueError("feedback output must be an ordinary directory")
    output.mkdir(parents=True, exist_ok=True)
    segments: list[dict[str, Any]] = []
    for ordinal, start in enumerate(range(0, len(rows), segment_samples)):
        chunk = rows[start:start + segment_samples]; offsets = [0]
        for row in chunk: offsets.append(offsets[-1] + len(row["legal"]))
        arrays = {"boards": np.stack([row["board"] for row in chunk]).astype(np.uint8), "legal_offsets": np.asarray(offsets, dtype=np.int64), "legal_actions": np.concatenate([row["legal"] for row in chunk]).astype(np.uint16), "target_action": np.asarray([row["target"] for row in chunk], dtype=np.uint16), "value_class": np.asarray([row["value"] for row in chunk], dtype=np.uint8), "game_index": np.asarray([index for index, (_, group) in enumerate(encoded_games) for _ in group][start:start + len(chunk)], dtype=np.uint32), "ply": np.asarray([row["ply"] for row in chunk], dtype=np.uint16), "color": np.asarray([row["color"] for row in chunk], dtype=np.uint8), "phase": np.asarray([row["phase"] for row in chunk], dtype=np.uint8), "sample_weight": np.full(len(chunk), sample_weight, dtype=np.float32)}
        segment = write_feedback_segment(output, ordinal, arrays, [row["meta"] for row in chunk])
        checked = __import__("chessy.feedback.segment", fromlist=["verify_feedback_segment"]).verify_feedback_segment(segment)
        segments.append({"path": str(segment.relative_to(output)), "manifest_checksum": checked["checksum"], "sample_count": len(chunk), "payload_fingerprint": checked["manifest"]["payload_fingerprint"]})
    histograms: dict[str, dict[str, int]] = {}
    for key in ("color", "phase"):
        inverse = {value: name for name, value in ENUMS[key].items()}; counts = Counter(inverse[row[key]] for row in rows); histograms[key] = dict(sorted(counts.items()))
    histograms["result"] = dict(sorted(Counter(str(manifest["result"]) for manifest, _ in encoded_games).items()))
    histograms["model"] = dict(sorted(Counter(str(manifest["model"]["id"]) for manifest, _ in encoded_games).items()))
    histograms["time_control"] = dict(sorted(Counter(str(manifest["time_control"]) for manifest, _ in encoded_games).items()))
    game_table = [{"game_id": item["game_id"], "created_at": item["created_at"], "raw_manifest_checksum": sha256(source / item["game_id"] / "manifest.json"), "pgn_checksum": sha256(source / item["game_id"] / "game.pgn"), "samples_checksum": sha256(source / item["game_id"] / "samples.jsonl"), "model_id": item["model"]["id"], "model_checksum": item["model"]["checksum"], "human_color": item["human_color"], "result": item["result"], "termination": item["termination"], "time_control": item["time_control"]} for item, _ in encoded_games]
    payload: dict[str, Any] = {"format": FORMAT, "encodings": {"board": "board119-v1", "action": "az73-v1"}, "enums": ENUMS, "raw_games": [item["game_id"] for item in game_table], "games": game_table, "segments": segments, "game_count": len(game_table), "sample_count": len(rows), "histograms": histograms, "sample_weight": float(sample_weight), "max_positions_per_game": max_positions_per_game}
    payload["content_fingerprint"] = fingerprint(payload); payload["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifests = output / "manifests"; manifests.mkdir(exist_ok=True); destination = manifests / f"feedback-dataset-{payload['content_fingerprint']}.json"
    if destination.exists():
        import json
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if {key: value for key, value in existing.items() if key != "created_at"} != {key: value for key, value in payload.items() if key != "created_at"}: raise FileExistsError("conflicting feedback manifest")
    else:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            temporary.write_bytes(canonical_json(payload))
            with temporary.open("rb") as file: os.fsync(file.fileno())
            os.replace(temporary, destination)
            descriptor = os.open(manifests, os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        finally:
            if temporary.exists(): temporary.unlink()
    return destination


def _history(board):  # type: ignore[no-untyped-def]
    snapshot = board.copy(stack=True); result = []
    while len(result) < 8:
        result.append(snapshot.copy(stack=True))
        if not snapshot.move_stack: break
        snapshot.pop()
    while len(result) < 8: result.append(result[-1].copy(stack=True))
    return result
