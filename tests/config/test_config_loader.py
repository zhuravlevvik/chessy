from __future__ import annotations

import math

import pytest

from chessy.config.canonical import canonical_json, fingerprint
from chessy.config.loader import load_config


def test_config_is_strict_and_canonical(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_text("""format: chessy-config-v1
name: test
seed: 1
device: cpu
model: {architecture: residual-cnn-v1, input_planes: 119, action_planes: 73, board_size: 8, channels: 8, residual_blocks: 1, group_norm_groups: 8, value_channels: 8, value_hidden: 16, value_classes: 3}
optimizer: {type: adamw, learning_rate: 0.001, weight_decay: 0.0, beta1: 0.9, beta2: 0.99, epsilon: 0.00000001}
scheduler: {type: warmup-cosine, warmup_steps: 1, total_steps: 3, minimum_lr_ratio: 0.0}
training: {batch_size: 2, gradient_clip_norm: 1.0, snapshot_every_steps: 1, keep_last_periodic: 2}
artifacts: {runs_dir: runs, dataset_manifest: null, replay_manifest: null, league_manifest: null}
""")
    config, _, resolved, resolved_fp = load_config(source)
    assert b'"value_classes":3' in resolved
    assert resolved_fp == fingerprint(config.model_dump(mode="json"))
    source.write_text(source.read_text() + "unknown: 1\n")
    with pytest.raises(Exception):
        load_config(source)
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})
