from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from chessy.run import Run
from chessy.snapshot.loader import select_snapshot
from chessy.training.sampler import StatefulBatchSampler
from chessy.training.smoke import fork_smoke, run_smoke

CONFIG = """format: chessy-config-v1
name: test-smoke
seed: 42
device: cpu
model: {architecture: residual-cnn-v1, input_planes: 119, action_planes: 73, board_size: 8, channels: 8, residual_blocks: 1, group_norm_groups: 8, value_channels: 8, value_hidden: 16, value_classes: 3}
optimizer: {type: adamw, learning_rate: 0.0003, weight_decay: 0.0001, beta1: 0.9, beta2: 0.999, epsilon: 0.00000001}
scheduler: {type: warmup-cosine, warmup_steps: 2, total_steps: 6, minimum_lr_ratio: 0.0}
training: {batch_size: 4, gradient_clip_norm: 1.0, snapshot_every_steps: 2, keep_last_periodic: 2}
artifacts: {runs_dir: runs, dataset_manifest: null, replay_manifest: null, league_manifest: null}
"""


def _state(path: Path):
    run = Run.open(path)
    _, checked, _ = select_snapshot(run)
    return checked


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def test_cpu_resume_restores_complete_training_state(tmp_path):
    config = tmp_path / "smoke.yaml"
    config.write_text(CONFIG)
    uninterrupted = run_smoke(root=tmp_path, config_path=config)
    stopped = run_smoke(root=tmp_path, config_path=config, stop_after_steps=3)
    run_smoke(root=tmp_path, resume=stopped)
    first, second = _state(uninterrupted), _state(stopped)
    assert _equal(first["model_state"], second["model_state"])
    assert _equal(first["training_state"], second["training_state"])
    assert first["run_state"]["global_step"] == second["run_state"]["global_step"] == 6
    left = StatefulBatchSampler(97, 4, seed=42)
    right = StatefulBatchSampler(97, 4, seed=42)
    left.load_state_dict(first["training_state"]["sampler_state"])
    right.load_state_dict(second["training_state"]["sampler_state"])
    assert torch.equal(left.next_batch(), right.next_batch())


def test_corrupt_latest_falls_back_and_fork_is_independent(tmp_path):
    config = tmp_path / "smoke.yaml"
    config.write_text(CONFIG)
    run_path = run_smoke(root=tmp_path, config_path=config, stop_after_steps=4)
    run = Run.open(run_path)
    latest = run.path / "snapshots" / json.loads((run.path / "snapshots/index.json").read_text())["latest"]
    latest.joinpath("model.safetensors").write_bytes(b"broken")
    run_smoke(root=tmp_path, resume=run_path, stop_after_steps=5)
    assert any(json.loads(line)["type"] == "resume_fallback" for line in (run.path / "events.jsonl").read_text().splitlines())
    valid, _, _ = select_snapshot(run)
    before = {path.relative_to(run.path): path.read_bytes() for path in run.path.rglob("*") if path.is_file()}
    child = fork_smoke(root=tmp_path, snapshot_path=valid, config_path=config, mode="weights-only")
    after = {path.relative_to(run.path): path.read_bytes() for path in run.path.rglob("*") if path.is_file()}
    assert before == after and child != run_path


def test_subprocess_sigint_publishes_resumable_stop_snapshot(tmp_path):
    config = tmp_path / "smoke.yaml"
    config.write_text(CONFIG.replace("total_steps: 6", "total_steps: 40"))
    code = f"from pathlib import Path; from chessy.training.smoke import run_smoke; run_smoke(root=Path({str(tmp_path)!r}), config_path=Path({str(config)!r}))"
    child = subprocess.Popen([sys.executable, "-c", code])
    metrics = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        candidates = list((tmp_path / "runs").glob("*/metrics.jsonl"))
        if candidates and candidates[0].exists() and candidates[0].read_text():
            metrics = candidates[0]
            break
        time.sleep(0.01)
    assert metrics is not None, "trainer did not begin a batch in time"
    child.send_signal(__import__("signal").SIGINT)
    assert child.wait(timeout=10) == 0
    run = Run.open(metrics.parent)
    _, checked, _ = select_snapshot(run)
    assert checked["run_state"]["snapshot_reason"] == "stop"
    assert checked["run_state"]["global_step"] == run.metrics.last_step
