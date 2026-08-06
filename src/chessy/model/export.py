"""Safe, atomic playable exports in the ``chessy-model-v1`` format."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from chessy.encoding import ACTION_ENCODING_VERSION, BOARD_ENCODING_VERSION
from chessy.model.config import ModelConfig
from chessy.model.device import DeviceName, resolve_device
from chessy.model.network import ChessyModel

EXPORT_FORMAT = "chessy-model-v1"
_REQUIRED_FILES = frozenset({"checksums.sha256", "manifest.json", "model.safetensors"})
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  ([^\s]+)$")
_VALUE_CLASSES = ["loss", "draw", "win"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _ordinary_export_files(source: Path) -> dict[str, Path]:
    if not source.is_dir():
        raise ValueError(f"model export directory does not exist: {source}")
    contents = {path.name for path in source.iterdir()}
    if contents != _REQUIRED_FILES:
        raise ValueError("model export must contain exactly model.safetensors, manifest.json, checksums.sha256")
    files = {name: source / name for name in _REQUIRED_FILES}
    for name, path in files.items():
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"export entry must be a regular file: {name}")
    return files


def _parse_checksums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("checksums.sha256 must be UTF-8 text") from exc
    checksums: dict[str, str] = {}
    for line in lines:
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ValueError("invalid checksums.sha256 line")
        digest, filename = match.groups()
        relative = PurePath(filename)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError("checksums.sha256 contains an unsafe path")
        if filename in checksums:
            raise ValueError("checksums.sha256 contains duplicate filenames")
        checksums[filename] = digest
    expected = {"manifest.json", "model.safetensors"}
    if set(checksums) != expected:
        raise ValueError("checksums.sha256 must cover manifest.json and model.safetensors exactly")
    return checksums


def _read_validated_manifest(source: Path) -> tuple[dict[str, Any], Path]:
    files = _ordinary_export_files(source)
    checksums = _parse_checksums(files["checksums.sha256"])
    for name in ("manifest.json", "model.safetensors"):
        if _sha256(files[name]) != checksums[name]:
            raise ValueError(f"checksum mismatch for {name}")
    try:
        manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest.json is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    return manifest, files["model.safetensors"]


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest field {field!r} must be an object")
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> ModelConfig:
    required = {
        "architecture",
        "created_at",
        "encodings",
        "format",
        "framework",
        "model_config",
        "value_classes",
        "weights",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest is missing fields: {sorted(missing)}")
    if manifest["format"] != EXPORT_FORMAT:
        raise ValueError(f"unsupported model export format: {manifest['format']!r}")
    if manifest["architecture"] != "residual-cnn-v1":
        raise ValueError("incompatible model architecture")
    encodings = _require_mapping(manifest["encodings"], "encodings")
    if encodings.get("board") != BOARD_ENCODING_VERSION:
        raise ValueError("incompatible board encoding")
    if encodings.get("action") != ACTION_ENCODING_VERSION:
        raise ValueError("incompatible action encoding")
    if manifest["value_classes"] != _VALUE_CLASSES:
        raise ValueError("incompatible value class order")
    framework = _require_mapping(manifest["framework"], "framework")
    if framework.get("name") != "pytorch" or not isinstance(framework.get("version"), str):
        raise ValueError("incompatible framework metadata")
    config = ModelConfig.from_dict(_require_mapping(manifest["model_config"], "model_config"))
    if manifest["architecture"] != config.architecture:
        raise ValueError("manifest architecture does not match model_config")
    weights = _require_mapping(manifest["weights"], "weights")
    if weights.get("file") != "model.safetensors":
        raise ValueError("weights file must be model.safetensors")
    if weights.get("dtype") != "float32":
        raise ValueError("weights dtype must be float32")
    if not isinstance(weights.get("parameter_count"), int) or isinstance(
        weights.get("parameter_count"), bool
    ) or weights["parameter_count"] <= 0:
        raise ValueError("weights parameter_count must be a positive integer")
    checksum = weights.get("sha256")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ValueError("weights sha256 must be a lowercase SHA-256 hex digest")
    return config


def _resolve_load_device(device: DeviceName | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        if device.type == "cpu":
            return device
        if device.type == "mps":
            return resolve_device("mps")
        if device.type == "cuda":
            return resolve_device("cuda")
        raise ValueError(f"unsupported torch device: {device}")
    return resolve_device(device)


def _load_checked_state_dict(weights_path: Path, config: ModelConfig, manifest: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    try:
        state_dict = load_file(str(weights_path), device="cpu")
    except Exception as exc:  # Safetensors errors do not share one public base class.
        raise ValueError("could not read model.safetensors") from exc
    if not state_dict:
        raise ValueError("model.safetensors contains no tensors")
    if any(tensor.dtype != torch.float32 for tensor in state_dict.values()):
        raise ValueError("model.safetensors contains non-float32 tensors")
    parameter_count = sum(tensor.numel() for tensor in state_dict.values())
    if parameter_count != manifest["weights"]["parameter_count"]:
        raise ValueError("weights parameter count does not match manifest")
    if _sha256(weights_path) != manifest["weights"]["sha256"]:
        raise ValueError("weights sha256 does not match manifest")
    expected_model = ChessyModel(config)
    try:
        expected_model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError("weights do not strictly match the model architecture") from exc
    return state_dict


def export_model(
    model: ChessyModel,
    destination: Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Atomically export inference weights without serializing Python objects."""
    if not isinstance(model, ChessyModel):
        raise ValueError("model must be a ChessyModel")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"export destination already exists: {destination}")
    parent = destination.parent
    if not parent.is_dir():
        raise ValueError(f"export destination parent does not exist: {parent}")
    if metadata is not None and (
        not isinstance(metadata, Mapping)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items())
    ):
        raise ValueError("metadata must be a mapping of strings to strings")

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    try:
        weights_path = temporary / "model.safetensors"
        state_dict = {
            name: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for name, tensor in sorted(model.state_dict().items())
        }
        save_file(state_dict, str(weights_path))
        weights_checksum = _sha256(weights_path)
        manifest: dict[str, Any] = {
            "format": EXPORT_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "architecture": model.config.architecture,
            "model_config": model.config.to_dict(),
            "encodings": {"board": BOARD_ENCODING_VERSION, "action": ACTION_ENCODING_VERSION},
            "value_classes": _VALUE_CLASSES,
            "framework": {"name": "pytorch", "version": torch.__version__},
            "weights": {
                "file": "model.safetensors",
                "dtype": "float32",
                "parameter_count": sum(tensor.numel() for tensor in state_dict.values()),
                "sha256": weights_checksum,
            },
        }
        if metadata:
            manifest["metadata"] = dict(metadata)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        checksums_path = temporary / "checksums.sha256"
        checksums_path.write_text(
            "".join(
                f"{_sha256(temporary / name)}  {name}\n"
                for name in sorted(("manifest.json", "model.safetensors"))
            ),
            encoding="utf-8",
        )

        checked_manifest, checked_weights_path = _read_validated_manifest(temporary)
        config = _validate_manifest(checked_manifest)
        _load_checked_state_dict(checked_weights_path, config, checked_manifest)
        temporary.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_model_export(
    source: Path,
    *,
    device: DeviceName | torch.device = "auto",
) -> ChessyModel:
    """Verify and load a ``chessy-model-v1`` export with strict state-dict checks."""
    source = Path(source)
    manifest, weights_path = _read_validated_manifest(source)
    config = _validate_manifest(manifest)
    state_dict = _load_checked_state_dict(weights_path, config, manifest)
    model = ChessyModel(config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError("weights do not strictly match the model architecture") from exc
    model.to(_resolve_load_device(device))
    model.eval()
    return model
