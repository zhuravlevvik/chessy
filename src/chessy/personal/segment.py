"""Safe immutable ``chessy-personal-dataset-v1`` segment format."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Mapping

import numpy as np

from chessy.config.canonical import canonical_json, fingerprint
from chessy.encoding import ACTION_SIZE
from chessy.replay.codec import decode_board

FORMAT = "chessy-personal-dataset-v1"
SEGMENT_FORMAT = "chessy-personal-segment-v1"
ENUMS = {
    "sample_kind": {"good_move": 0, "full_game": 1},
    "source": {"chess.com": 0, "lichess": 1},
    "color": {"black": 0, "white": 1},
    "phase": {"opening": 0, "middlegame": 1, "endgame": 2},
    "value_class": {"loss": 0, "draw": 1, "win": 2},
}
_FILES = frozenset({"samples.npz", "metadata.jsonl", "manifest.json", "checksums.sha256"})
_ARRAYS = {
    "boards": (np.uint8, 4), "legal_offsets": (np.int64, 1),
    "legal_actions": (np.uint16, 1), "target_action": (np.uint16, 1),
    "value_class": (np.uint8, 1), "game_index": (np.uint32, 1),
    "ply": (np.uint16, 1), "sample_kind": (np.uint8, 1),
    "source": (np.uint8, 1), "color": (np.uint8, 1), "phase": (np.uint8, 1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync(path: Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write fixed-order NPY members and fixed ZIP timestamps for reproducibility."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            stream = io.BytesIO()
            np.lib.format.write_array(stream, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, stream.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _checksums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (path / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        try:
            digest, name = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("invalid personal checksums.sha256") from exc
        pure = PurePath(name)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1 or name in values:
            raise ValueError("unsafe personal checksum entry")
        values[name] = digest
    if set(values) != {"samples.npz", "metadata.jsonl", "manifest.json"}:
        raise ValueError("personal checksums must cover payload exactly")
    return values


def _ordinary_dir(path: Path, expected: frozenset[str] = _FILES) -> None:
    if path.is_symlink() or not path.is_dir() or {item.name for item in path.iterdir()} != expected:
        raise ValueError("personal segment has unsafe or unexpected files")
    for item in path.iterdir():
        if not stat.S_ISREG(item.lstat().st_mode):
            raise ValueError("personal segment entries must be regular files")


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    if set(arrays) != set(_ARRAYS):
        raise ValueError("personal samples.npz has unexpected arrays")
    count: int | None = None
    for name, (dtype, rank) in _ARRAYS.items():
        array = arrays[name]
        if array.dtype != dtype or array.ndim != rank:
            raise ValueError(f"invalid personal array {name}")
        if name == "boards":
            if array.shape[1:] != (119, 8, 8):
                raise ValueError("personal boards shape must be [N,119,8,8]")
            count = array.shape[0]
        elif name != "legal_actions" and name != "legal_offsets" and array.shape[0] != count:
            raise ValueError(f"personal array {name} has wrong sample count")
    assert count is not None
    offsets, actions = arrays["legal_offsets"], arrays["legal_actions"]
    if offsets.shape != (count + 1,) or int(offsets[0]) != 0 or int(offsets[-1]) != len(actions) or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("invalid personal legal offsets")
    if np.any(actions >= ACTION_SIZE):
        raise ValueError("personal legal action out of bounds")
    for index in range(count):
        legal = actions[int(offsets[index]):int(offsets[index + 1])]
        if len(legal) == 0 or len(np.unique(legal)) != len(legal) or arrays["target_action"][index] not in legal:
            raise ValueError("personal target action is not uniquely legal")
    for name, mapping in ENUMS.items():
        if np.any(arrays[name] >= len(mapping)):
            raise ValueError(f"personal {name} enum is invalid")
    # This ensures the shared replay codec remains the one true packing format.
    for board in arrays["boards"][: min(3, count)]:
        decode_board(board)
    return count


def verify_segment(path: Path) -> dict[str, Any]:
    path = Path(path)
    _ordinary_dir(path)
    checks = _checksums(path)
    for name, digest in checks.items():
        if sha256(path / name) != digest:
            raise ValueError(f"checksum mismatch for personal {name}")
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        with np.load(path / "samples.npz", allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in source.files}
    except (UnicodeDecodeError, json.JSONDecodeError, OSError, ValueError) as exc:
        raise ValueError("invalid personal segment payload") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != SEGMENT_FORMAT or manifest.get("enums") != ENUMS:
        raise ValueError("invalid personal segment manifest")
    count = _validate_arrays(arrays)
    metadata = (path / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    if len(metadata) != count:
        raise ValueError("personal metadata count mismatch")
    sample_ids: set[str] = set()
    for line in metadata:
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get("sample_id"), str) or item["sample_id"] in sample_ids:
            raise ValueError("invalid or duplicate personal sample ID")
        sample_ids.add(item["sample_id"])
    if manifest.get("sample_count") != count or manifest.get("payload_fingerprint") != fingerprint({"arrays": {name: sha256_array(value) for name, value in arrays.items()}, "metadata": metadata}):
        raise ValueError("personal segment fingerprint mismatch")
    return {"manifest": manifest, "arrays": arrays, "metadata": metadata, "checksum": sha256(path / "checksums.sha256")}


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def write_segment(root: Path, split: str, ordinal: int, arrays: Mapping[str, np.ndarray], metadata: list[dict[str, Any]]) -> Path:
    """Publish one verified immutable segment using an atomic sibling rename."""
    if split not in {"train", "val", "test"}:
        raise ValueError("invalid personal split")
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    count = _validate_arrays(arrays)
    if len(metadata) != count:
        raise ValueError("metadata must have one row per sample")
    meta_lines = [canonical_json(row).decode("utf-8").rstrip("\n") for row in metadata]
    sample_ids = [row.get("sample_id") for row in metadata]
    if any(not isinstance(value, str) for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("metadata sample IDs must be unique strings")
    payload_fingerprint = fingerprint({"arrays": {name: sha256_array(value) for name, value in arrays.items()}, "metadata": meta_lines})
    segments_root = Path(root) / "segments"
    segments_root.mkdir(parents=True, exist_ok=True)
    name = f"{split}-{ordinal:05d}-{payload_fingerprint[:12]}"
    destination = segments_root / name
    if destination.exists():
        checked = verify_segment(destination)
        if checked["manifest"].get("payload_fingerprint") != payload_fingerprint:
            raise FileExistsError("existing personal segment does not match payload")
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=segments_root))
    try:
        _write_deterministic_npz(temporary / "samples.npz", arrays)
        (temporary / "metadata.jsonl").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
        manifest = {"format": SEGMENT_FORMAT, "split": split, "ordinal": ordinal, "sample_count": count, "enums": ENUMS, "payload_fingerprint": payload_fingerprint}
        (temporary / "manifest.json").write_bytes(canonical_json(manifest))
        for payload in ("samples.npz", "metadata.jsonl", "manifest.json"):
            _fsync(temporary / payload)
        (temporary / "checksums.sha256").write_text("".join(f"{sha256(temporary / item)}  {item}\n" for item in ("manifest.json", "metadata.jsonl", "samples.npz")), encoding="utf-8")
        _fsync(temporary / "checksums.sha256")
        verify_segment(temporary)
        temporary.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_personal_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("personal dataset manifest must be a regular file")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid personal dataset manifest") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT or data.get("enums") != ENUMS:
        raise ValueError("unsupported personal dataset manifest")
    return data


def verify_personal_manifest(path: Path) -> dict[str, Any]:
    data = load_personal_manifest(path)
    root = Path(path).parent.parent
    seen_games: set[int] = set()
    seen_samples: set[str] = set()
    global_histograms: dict[str, Counter[str]] = {name: Counter() for name in ENUMS}
    for split in ("train", "val", "test"):
        entries = data.get("splits", {}).get(split, {}).get("segments", [])
        if not isinstance(entries, list):
            raise ValueError("invalid personal manifest split")
        total = 0
        split_games: set[int] = set()
        split_histograms: dict[str, Counter[str]] = {name: Counter() for name in ENUMS}
        for entry in entries:
            relative = PurePath(entry.get("path", "")) if isinstance(entry, dict) else PurePath("")
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2 or relative.parts[0] != "segments":
                raise ValueError("unsafe personal segment path")
            segment = root / relative
            if not segment.resolve().is_relative_to(root.resolve()):
                raise ValueError("personal segment escapes dataset root")
            checked = verify_segment(segment)
            if entry.get("manifest_checksum") != checked["checksum"]:
                raise ValueError("personal manifest pin mismatch")
            if entry.get("sample_count") != checked["manifest"].get("sample_count") or entry.get("payload_fingerprint") != checked["manifest"].get("payload_fingerprint"):
                raise ValueError("personal manifest segment metadata mismatch")
            total += checked["manifest"]["sample_count"]
            for line in checked["metadata"]:
                row = json.loads(line)
                if row["sample_id"] in seen_samples:
                    raise ValueError("personal sample overlaps splits")
                seen_samples.add(row["sample_id"])
            split_games |= set(int(item) for item in checked["arrays"]["game_index"])
            for name, mapping in ENUMS.items():
                inverse = {value: key for key, value in mapping.items()}
                split_histograms[name].update(inverse[int(value)] for value in checked["arrays"][name])
        declared = data["splits"][split].get("sample_count")
        if declared != total:
            raise ValueError("personal manifest sample count mismatch")
        if data["splits"][split].get("game_count") != len(split_games):
            raise ValueError("personal manifest game count mismatch")
        expected_histograms = {name: dict(sorted(values.items())) for name, values in split_histograms.items()}
        if data["splits"][split].get("histograms") != expected_histograms:
            raise ValueError("personal manifest split histogram mismatch")
        for name in ENUMS:
            global_histograms[name].update(split_histograms[name])
        if seen_games & split_games:
            raise ValueError("personal game overlaps splits")
        seen_games |= split_games
    expected_global = {name: dict(sorted(values.items())) for name, values in global_histograms.items()}
    if data.get("histograms") != expected_global:
        raise ValueError("personal manifest global histogram mismatch")
    if data.get("frozen_test_fingerprint") != fingerprint(data["splits"]["test"]):
        raise ValueError("personal manifest frozen test fingerprint mismatch")
    if data.get("content_fingerprint") != fingerprint({key: value for key, value in data.items() if key not in {"created_at", "content_fingerprint"}}):
        raise ValueError("personal dataset content fingerprint mismatch")
    return data
