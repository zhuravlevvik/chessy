from __future__ import annotations

import json
from pathlib import Path

import pytest

from chessy.config.canonical import canonical_json
from chessy.config.loader import load_config
from chessy.run import Run
from chessy.run.logging import JsonlLog

CONFIG = """format: chessy-config-v1
name: safety-test
seed: 42
device: cpu
model: {architecture: residual-cnn-v1, input_planes: 119, action_planes: 73, board_size: 8, channels: 8, residual_blocks: 1, group_norm_groups: 8, value_channels: 8, value_hidden: 16, value_classes: 3}
optimizer: {type: adamw, learning_rate: 0.0003, weight_decay: 0.0001, beta1: 0.9, beta2: 0.999, epsilon: 0.00000001}
scheduler: {type: warmup-cosine, warmup_steps: 2, total_steps: 6, minimum_lr_ratio: 0.0}
training: {batch_size: 4, gradient_clip_norm: 1.0, snapshot_every_steps: 2, keep_last_periodic: 2}
artifacts: {runs_dir: runs, dataset_manifest: null, replay_manifest: null, league_manifest: null}
"""


def test_complete_jsonl_tail_without_newline_is_repaired(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    record = {
        "format": "chessy-events-v1",
        "sequence": 1,
        "timestamp": "2026-01-01T00:00:00Z",
        "type": "existing",
        "payload": {},
    }
    path.write_bytes(canonical_json(record).rstrip(b"\n"))

    log = JsonlLog(path, "events")
    log.append_event("next")

    parsed = [json.loads(line) for line in path.read_text().splitlines()]
    assert log.recovered is True
    assert [item["sequence"] for item in parsed] == [1, 2]


def test_existing_logs_must_be_strictly_monotonic(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    records = [
        {"format": "chessy-metrics-v1", "step": 2, "epoch": 0, "timestamp": "x", "metrics": {"loss": 1}},
        {"format": "chessy-metrics-v1", "step": 1, "epoch": 0, "timestamp": "x", "metrics": {"loss": 1}},
    ]
    path.write_bytes(b"".join(canonical_json(item) for item in records))

    with pytest.raises(ValueError, match="strictly increase"):
        JsonlLog(path, "metrics")


def test_run_open_rejects_changed_resolved_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG)
    config, source, resolved, config_fingerprint = load_config(config_path)
    run = Run.create(tmp_path, config, source, resolved, config_fingerprint)
    changed = json.loads((run.path / "config.resolved.json").read_text())
    changed["name"] = "tampered"
    (run.path / "config.resolved.json").write_bytes(canonical_json(changed))

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        Run.open(run.path)


def test_reference_manifest_must_not_be_symlink(tmp_path: Path) -> None:
    actual = tmp_path / "manifest.json"
    actual.write_text("{}")
    link = tmp_path / "manifest-link.json"
    link.symlink_to(actual)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG.replace("dataset_manifest: null", "dataset_manifest: manifest-link.json"))
    config, source, resolved, config_fingerprint = load_config(config_path)

    with pytest.raises(ValueError, match="must not be a symlink"):
        Run.create(tmp_path, config, source, resolved, config_fingerprint)


def test_artifact_paths_cannot_escape_project(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG.replace("dataset_manifest: null", "dataset_manifest: ../manifest.json"))

    with pytest.raises(Exception, match="relative safe paths"):
        load_config(config_path)
