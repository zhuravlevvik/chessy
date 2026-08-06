"""Canonical JSON used by all Chessy local artifact formats."""
from __future__ import annotations
import hashlib
import json
import math
from typing import Any

def _check(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON does not allow NaN or Infinity")
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("canonical JSON object keys must be strings")
        for item in value.values(): _check(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _check(item)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(f"not a JSON-compatible value: {type(value).__name__}")

def canonical_json(value: Any) -> bytes:
    _check(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")

def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).rstrip(b"\n")).hexdigest()


def fingerprint_bytes(resolved: bytes) -> str:
    """Fingerprint the *stored* canonical resolved-config bytes.

    This deliberately does not parse through the current schema.  A later
    schema may add defaults, while historical runs must retain their original
    identity.
    """
    if not isinstance(resolved, bytes) or not resolved:
        raise ValueError("resolved config must be non-empty bytes")
    return hashlib.sha256(resolved.rstrip(b"\n")).hexdigest()
