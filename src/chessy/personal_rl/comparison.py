"""Deterministic report assembly for the required personal model matrix."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
from chessy.config.canonical import fingerprint


def comparison_report(*, arenas: dict[str, Any], styles: dict[str, dict[str, Any]], checksums: dict[str, str], positions_fingerprint: str, config_fingerprint: str) -> dict[str, Any]:
    body = {"format": "chessy-personal-rl-comparison-v1", "arenas": {name: asdict(report) if hasattr(report, "__dataclass_fields__") else report for name, report in sorted(arenas.items())}, "style": styles, "checksums": checksums, "positions_fingerprint": positions_fingerprint, "config_fingerprint": config_fingerprint}
    return {**body, "content_fingerprint": fingerprint(body)}
