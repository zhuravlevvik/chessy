"""Read-only discovery of playable model exports produced by local runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from chessy.play.agent import ModelInfo

_CHECKSUM = re.compile(r"^[0-9a-f]{64}$")


def safe_model_id(name: str, checksum: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower() or "model"
    return f"{stem[:48]}-{checksum[:12]}"


def model_info_from_export(path: Path) -> ModelInfo:
    path = Path(path)
    manifest_path = path / "manifest.json"
    required = (manifest_path, path / "model.safetensors", path / "checksums.sha256")
    if path.is_symlink() or any(item.is_symlink() or not item.is_file() for item in required) or manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("unsafe model export manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid model export manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("model export manifest must be an object")
    weights = manifest.get("weights")
    metadata = manifest.get("metadata", {})
    if manifest.get("format") != "chessy-model-v1" or manifest.get("architecture") != "residual-cnn-v1" or not isinstance(weights, dict) or not isinstance(metadata, dict):
        raise ValueError("incompatible model export manifest")
    checksum = weights.get("sha256")
    if not isinstance(checksum, str) or _CHECKSUM.fullmatch(checksum) is None:
        raise ValueError("invalid model export checksum")
    base = metadata.get("name", path.name)
    if not isinstance(base, str) or not base.strip(): base = path.name
    step = metadata.get("step")
    generation = metadata.get("generation")
    if isinstance(step, str) and step.isdigit(): name = f"{base} · step {int(step)}"
    elif isinstance(generation, str) and generation.isdigit(): name = f"{base} · generation {int(generation)}"
    else: name = f"{base} · {path.name}"
    return ModelInfo(id=safe_model_id(name, checksum), name=name[:100], checksum=checksum, architecture="residual-cnn-v1")


def discover_model_exports(runs_dir: Path, *, limit: int = 200) -> list[tuple[ModelInfo, Path]]:
    root = Path(runs_dir)
    if not root.is_dir() or root.is_symlink(): return []
    resolved_root = root.resolve(); manifests: list[tuple[int, Path]] = []
    for manifest in root.glob("*/exports/*/manifest.json"):
        try:
            export = manifest.parent
            if export.name.startswith(".") or export.is_symlink() or not export.resolve().is_relative_to(resolved_root): continue
            manifests.append((manifest.stat().st_mtime_ns, manifest))
        except OSError: continue
    manifests.sort(key=lambda item: item[0], reverse=True)
    found: list[tuple[ModelInfo, Path]] = []; seen: set[str] = set()
    for _, manifest in manifests[:limit]:
        try: info = model_info_from_export(manifest.parent)
        except (OSError, ValueError): continue
        if info.id in seen: continue
        seen.add(info.id); found.append((info, manifest.parent))
    return found
