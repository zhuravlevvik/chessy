from __future__ import annotations
from pathlib import Path
import yaml
from chessy.config.canonical import canonical_json, fingerprint_bytes
from chessy.config.schema import ChessyConfig

def load_config(path: Path) -> tuple[ChessyConfig, bytes, bytes, str]:
    source = Path(path).read_bytes()
    try: raw = yaml.safe_load(source)
    except yaml.YAMLError as exc: raise ValueError("invalid or unsafe YAML config") from exc
    if not isinstance(raw, dict): raise ValueError("config root must be a mapping")
    config = ChessyConfig.model_validate(raw)
    resolved = canonical_json(config.model_dump(mode="json"))
    return config, source, resolved, fingerprint_bytes(resolved)

def load_resolved(data: bytes) -> ChessyConfig:
    import json
    return ChessyConfig.model_validate(json.loads(data))
