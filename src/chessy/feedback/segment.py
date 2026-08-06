"""Immutable, checksummed segments for the train-only feedback stream."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import zipfile
import re
from collections import Counter
from pathlib import Path, PurePath
from typing import Any, Mapping

import numpy as np

from chessy.config.canonical import canonical_json, fingerprint
from chessy.encoding import ACTION_SIZE
from chessy.replay.codec import decode_board

FORMAT = "chessy-human-feedback-dataset-v1"
SEGMENT_FORMAT = "chessy-human-feedback-segment-v1"
ENUMS = {"color": {"black": 0, "white": 1}, "phase": {"opening": 0, "middlegame": 1, "endgame": 2}, "value_class": {"loss": 0, "draw": 1, "win": 2}}
_FILES = frozenset({"samples.npz", "metadata.jsonl", "manifest.json", "checksums.sha256"})
_ARRAYS = {"boards": (np.uint8, 4), "legal_offsets": (np.int64, 1), "legal_actions": (np.uint16, 1), "target_action": (np.uint16, 1), "value_class": (np.uint8, 1), "game_index": (np.uint32, 1), "ply": (np.uint16, 1), "color": (np.uint8, 1), "phase": (np.uint8, 1), "sample_weight": (np.float32, 1)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256(); digest.update(str(array.dtype).encode()); digest.update(repr(array.shape).encode()); digest.update(np.ascontiguousarray(array).tobytes()); return digest.hexdigest()


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            source = io.BytesIO(); np.lib.format.write_array(source, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    if set(arrays) != set(_ARRAYS): raise ValueError("feedback samples.npz has unexpected arrays")
    count: int | None = None
    for name, (dtype, rank) in _ARRAYS.items():
        value = arrays[name]
        if value.dtype != dtype or value.ndim != rank: raise ValueError(f"invalid feedback array {name}")
        if name == "boards":
            if value.shape[1:] != (119, 8, 8): raise ValueError("feedback boards must have shape [N,119,8,8]")
            count = value.shape[0]
        elif name not in {"legal_offsets", "legal_actions"} and value.shape[0] != count: raise ValueError(f"feedback array {name} has wrong sample count")
    assert count is not None
    offsets, actions = arrays["legal_offsets"], arrays["legal_actions"]
    if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(actions) or np.any(offsets[1:] < offsets[:-1]) or np.any(actions >= ACTION_SIZE): raise ValueError("invalid feedback legal actions")
    if not np.isfinite(arrays["sample_weight"]).all() or np.any(arrays["sample_weight"] <= 0): raise ValueError("invalid feedback sample weights")
    for index in range(count):
        legal = actions[int(offsets[index]):int(offsets[index + 1])]
        if not len(legal) or len(np.unique(legal)) != len(legal) or arrays["target_action"][index] not in legal: raise ValueError("feedback target is not uniquely legal")
    for name, values in ENUMS.items():
        if np.any(arrays[name] >= len(values)): raise ValueError(f"invalid feedback {name}")
    for board in arrays["boards"][: min(3, count)]: decode_board(board)
    return count


def _ordinary(path: Path) -> None:
    if path.is_symlink() or not path.is_dir() or {item.name for item in path.iterdir()} != _FILES: raise ValueError("feedback segment has unsafe or unexpected files")
    if any(item.is_symlink() or not stat.S_ISREG(item.lstat().st_mode) for item in path.iterdir()): raise ValueError("feedback segment entries must be regular files")


def verify_feedback_segment(path: Path) -> dict[str, Any]:
    path = Path(path); _ordinary(path)
    try:
        checks: dict[str, str] = {}
        for line in (path / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1); pure = PurePath(name)
            if name in checks or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1: raise ValueError("unsafe checksum entry")
            checks[name] = digest
        if set(checks) != {"samples.npz", "metadata.jsonl", "manifest.json"}: raise ValueError("checksums must cover payload exactly")
        if any(sha256(path / name) != digest for name, digest in checks.items()): raise ValueError("feedback checksum mismatch")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        with np.load(path / "samples.npz", allow_pickle=False) as source: arrays = {name: source[name].copy() for name in source.files}
        metadata = (path / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid feedback segment payload") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != SEGMENT_FORMAT or manifest.get("enums") != ENUMS: raise ValueError("invalid feedback segment manifest")
    count = _validate_arrays(arrays)
    if len(metadata) != count: raise ValueError("feedback metadata count mismatch")
    ids: set[str] = set()
    for line in metadata:
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) or re.fullmatch(r"[0-9a-f]{64}", row["sample_id"]) is None or row["sample_id"] in ids: raise ValueError("invalid feedback metadata")
        ids.add(row["sample_id"])
    payload = fingerprint({"arrays": {name: sha256_array(value) for name, value in arrays.items()}, "metadata": metadata})
    if manifest.get("sample_count") != count or manifest.get("payload_fingerprint") != payload: raise ValueError("feedback segment fingerprint mismatch")
    return {"manifest": manifest, "arrays": arrays, "metadata": metadata, "checksum": sha256(path / "checksums.sha256")}


def write_feedback_segment(root: Path, ordinal: int, arrays: Mapping[str, np.ndarray], metadata: list[dict[str, Any]]) -> Path:
    arrays = {name: np.asarray(value) for name, value in arrays.items()}; count = _validate_arrays(arrays)
    if len(metadata) != count: raise ValueError("feedback metadata must align with arrays")
    lines = [canonical_json(row).decode().rstrip("\n") for row in metadata]
    if len({row.get("sample_id") for row in metadata}) != count or any(not isinstance(row.get("sample_id"), str) for row in metadata): raise ValueError("feedback sample IDs must be unique")
    payload = fingerprint({"arrays": {name: sha256_array(value) for name, value in arrays.items()}, "metadata": lines})
    root = Path(root)
    if root.is_symlink() or not root.is_dir(): raise ValueError("feedback segment root must be an ordinary directory")
    segments = root / "segments"; segments.mkdir(parents=True, exist_ok=True)
    if segments.is_symlink() or not segments.is_dir(): raise ValueError("feedback segments path must be an ordinary directory")
    name = f"feedback-{ordinal:05d}-{payload[:12]}"; destination = segments / name
    if destination.exists():
        checked = verify_feedback_segment(destination)
        if checked["manifest"].get("payload_fingerprint") != payload: raise FileExistsError("conflicting feedback segment")
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=segments))
    try:
        _write_npz(temporary / "samples.npz", arrays); (temporary / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest = {"format": SEGMENT_FORMAT, "ordinal": ordinal, "sample_count": count, "enums": ENUMS, "payload_fingerprint": payload}
        (temporary / "manifest.json").write_bytes(canonical_json(manifest))
        for name in ("samples.npz", "metadata.jsonl", "manifest.json"):
            with (temporary / name).open("rb") as source: os.fsync(source.fileno())
        (temporary / "checksums.sha256").write_text("".join(f"{sha256(temporary / name)}  {name}\n" for name in ("manifest.json", "metadata.jsonl", "samples.npz")), encoding="utf-8")
        with (temporary / "checksums.sha256").open("rb") as source: os.fsync(source.fileno())
        verify_feedback_segment(temporary); temporary.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise


def load_feedback_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError("feedback manifest must be a regular file")
    try: data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc: raise ValueError("invalid feedback manifest") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT or data.get("enums") != ENUMS: raise ValueError("unsupported feedback manifest")
    return data


def verify_feedback_manifest(path: Path) -> dict[str, Any]:
    data = load_feedback_manifest(path); root = Path(path).parent.parent; seen: set[str] = set(); total = 0
    entries, games = data.get("segments"), data.get("games")
    if not isinstance(entries, list) or not entries or not isinstance(games, list) or not games: raise ValueError("feedback manifest requires games and segments")
    game_ids = [item.get("game_id") for item in games if isinstance(item, dict)]
    if len(game_ids) != len(games) or any(not isinstance(item, str) for item in game_ids) or len(set(game_ids)) != len(game_ids) or data.get("raw_games") != game_ids: raise ValueError("invalid feedback game table")
    for game in games:
        required = {"game_id", "created_at", "raw_manifest_checksum", "pgn_checksum", "samples_checksum", "model_id", "model_checksum", "human_color", "result", "termination", "time_control"}
        if set(game) != required or any(not isinstance(game.get(key), str) or not game[key] for key in required): raise ValueError("invalid feedback game metadata")
        if any(re.fullmatch(r"[0-9a-f]{64}", game[key]) is None for key in ("raw_manifest_checksum", "pgn_checksum", "samples_checksum", "model_checksum")): raise ValueError("invalid feedback game checksum")
        if game["human_color"] not in {"white", "black"} or game["result"] not in {"1-0", "0-1", "1/2-1/2"}: raise ValueError("invalid feedback game color/result")
    sample_histograms: dict[str, Counter[str]] = {"color": Counter(), "phase": Counter()}; used_games: set[int] = set()
    manifest_weight = data.get("sample_weight"); manifest_cap = data.get("max_positions_per_game")
    if not isinstance(manifest_weight, (int, float)) or not np.isfinite(manifest_weight) or manifest_weight <= 0 or not isinstance(manifest_cap, int) or isinstance(manifest_cap, bool) or manifest_cap <= 0: raise ValueError("invalid feedback manifest sampling settings")
    for entry in entries:
        relative = PurePath(entry.get("path", "")) if isinstance(entry, dict) else PurePath("")
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2 or relative.parts[0] != "segments": raise ValueError("unsafe feedback segment path")
        segment = root / relative
        if not segment.resolve().is_relative_to(root.resolve()): raise ValueError("feedback segment escapes dataset root")
        checked = verify_feedback_segment(segment)
        if entry.get("manifest_checksum") != checked["checksum"] or entry.get("payload_fingerprint") != checked["manifest"].get("payload_fingerprint") or entry.get("sample_count") != checked["manifest"].get("sample_count"): raise ValueError("feedback manifest segment pin mismatch")
        total += int(entry["sample_count"])
        arrays = checked["arrays"]
        if np.any(arrays["game_index"] >= len(games)) or not np.allclose(arrays["sample_weight"], float(manifest_weight), atol=1e-7, rtol=1e-6): raise ValueError("feedback segment does not match manifest games/weight")
        for offset, line in enumerate(checked["metadata"]):
            metadata = json.loads(line); identifier = metadata["sample_id"]; game_index = int(arrays["game_index"][offset]); used_games.add(game_index)
            if identifier in seen: raise ValueError("duplicate feedback sample across segments")
            seen.add(identifier)
            if metadata.get("game_id") != games[game_index]["game_id"] or metadata.get("ply") != int(arrays["ply"][offset]): raise ValueError("feedback metadata does not match encoded arrays")
        for key in ("color", "phase"):
            inverse = {value: name for name, value in ENUMS[key].items()}; sample_histograms[key].update(inverse[int(value)] for value in arrays[key])
    if total != data.get("sample_count") or len(games) != data.get("game_count") or used_games != set(range(len(games))): raise ValueError("feedback manifest counts mismatch")
    histograms = data.get("histograms")
    expected_sample = {key: dict(sorted(values.items())) for key, values in sample_histograms.items()}
    if not isinstance(histograms, dict) or any(histograms.get(key) != expected_sample[key] for key in expected_sample): raise ValueError("feedback manifest sample histogram mismatch")
    expected_games = {
        "result": dict(sorted(Counter(str(item.get("result")) for item in games).items())),
        "model": dict(sorted(Counter(str(item.get("model_id")) for item in games).items())),
        "time_control": dict(sorted(Counter(str(item.get("time_control")) for item in games).items())),
    }
    if any(histograms.get(key) != value for key, value in expected_games.items()): raise ValueError("feedback manifest game histogram mismatch")
    expected = fingerprint({key: value for key, value in data.items() if key not in {"created_at", "content_fingerprint"}})
    if data.get("content_fingerprint") != expected: raise ValueError("feedback dataset fingerprint mismatch")
    return data
