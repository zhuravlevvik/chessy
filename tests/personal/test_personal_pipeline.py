from __future__ import annotations

import csv
import json
from pathlib import Path

import chess
import pytest
import torch

from chessy.personal.builder import build_personal_dataset
from chessy.personal.dataset import PersonalDataset
from chessy.personal.sampler import PersonalBatchSampler
from chessy.personal.segment import verify_personal_manifest
from chessy.personal.validation import _summary
from chessy.model import ChessyModel, ModelConfig
from chessy.model.export import export_model
from chessy.run import Run
from chessy.snapshot.loader import select_snapshot
from chessy.training.personal_trainer import run_personal_training
from chessy.training.supervised_loss import supervised_policy_value_loss


def _pgn(white: str, black: str, result: str) -> str:
    return f'''[Event "fixture"]
[White "{white}"]
[Black "{black}"]
[Result "{result}"]
[UTCDate "2026.01.01"]

1. e4 e5 2. Nf3 Nc6 {result}

'''


def _row(index: int, source: str, color: str, result: str, move: str) -> dict[str, object]:
    board = chess.Board()
    if color == "black":
        board.push_uci("e2e4")
    return {"game_index": index, "source": source, "date": "2026.01.01", "url": "fixture", "ply": 2 if color == "black" else 1, "move_number": 1, "color": color, "fen": board.fen(), "move_uci": move, "move_san": "e5" if color == "black" else "e4", "move_accuracy": 90.0, "game_accuracy": 90.0, "sample_kind": "full_game"}


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    raw = tmp_path / "raw"; raw.mkdir()
    chess_com = raw / "chess.pgn"; chess_com.write_text(_pgn("mu1876", "other", "1-0") + _pgn("other", "mu1876", "0-1"))
    lichess = raw / "lichess.pgn"; lichess.write_text(_pgn("mu1878", "other", "1/2-1/2"))
    quality = tmp_path / "quality.csv"
    with quality.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "result"]); writer.writeheader()
        writer.writerows([{"index": 0, "result": "1-0"}, {"index": 1, "result": "0-1"}, {"index": 2, "result": "1/2-1/2"}])
    splits = tmp_path / "splits"; splits.mkdir()
    rows = {"train": [_row(0, "chess.com", "white", "1-0", "e2e4")], "val": [_row(1, "chess.com", "black", "0-1", "e7e5")], "test": [_row(2, "lichess", "white", "1/2-1/2", "e2e4")]}
    for split, values in rows.items():
        (splits / f"{split}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in values))
    (splits / "manifest.json").write_text(json.dumps({"splits": {key: {"file": f"{key}.jsonl", "samples": 1, "games": 1} for key in rows}}))
    return splits / "manifest.json", chess_com, lichess, quality


def test_build_load_firewall_and_stateful_sampler(tmp_path: Path) -> None:
    splits, chess_com, lichess, quality = _fixture_inputs(tmp_path)
    manifest = build_personal_dataset(splits=splits, chess_com_pgn=chess_com, lichess_pgn=lichess, game_quality=quality, output=tmp_path / "encoded", segment_samples=2)
    assert verify_personal_manifest(manifest)["splits"]["train"]["sample_count"] == 1
    train = PersonalDataset(manifest, split="train")
    assert train[0]["value_class"] == 2
    with pytest.raises(PermissionError):
        PersonalDataset(manifest, split="test")
    sampler = PersonalBatchSampler(train, batch_size=1, seed=4)
    first = sampler.next_batch(); state = sampler.state_dict()
    resumed = PersonalBatchSampler(train, batch_size=1, seed=4); resumed.load_state_dict(state)
    assert first == [0] and resumed.next_batch() is None


def test_supervised_loss_masks_illegal_logits() -> None:
    policy = torch.zeros((1, 4672)); value = torch.zeros((1, 3)); legal = torch.zeros((1, 4672), dtype=torch.bool)
    legal[0, 5] = True; legal[0, 7] = True
    target = torch.tensor([5]); classes = torch.tensor([2])
    first, _ = supervised_policy_value_loss(policy, value, target, legal, classes)
    policy[0, 100] = 10_000
    second, _ = supervised_policy_value_loss(policy, value, target, legal, classes)
    assert torch.equal(first, second)


def test_sampler_applies_weights_after_per_game_cap() -> None:
    class Dataset:
        split = "train"
        fingerprint = "fixture"
        indices_by_game = {**{game: [game] for game in range(20)}, **{game: [game] for game in range(20, 40)}}

        def __len__(self) -> int:
            return 40

        def __getitem__(self, index: int) -> dict[str, int]:
            return {"sample_kind": 0 if index < 20 else 1}

    sampler = PersonalBatchSampler(Dataset(), batch_size=64, seed=3, kind_weights={"good_move": .75, "full_game": 1.0})  # type: ignore[arg-type]
    kinds = [Dataset()[index]["sample_kind"] for index in sampler.pool]
    assert kinds.count(0) == 15
    assert kinds.count(1) == 20
    assert len(sampler.pool) == len(set(sampler.pool))


def test_validation_summary_uses_true_even_median() -> None:
    row = {"ce": 1.0, "top1": 1.0, "top3": 1.0, "top5": 1.0, "value_ce": 1.0, "value_acc": 1.0}
    assert _summary([{**row, "prob": .2}, {**row, "prob": .8}])["median_true_move_probability"] == pytest.approx(.5)


def test_personal_trainer_stop_and_resume(tmp_path: Path) -> None:
    splits, chess_com, lichess, quality = _fixture_inputs(tmp_path)
    manifest = build_personal_dataset(splits=splits, chess_com_pgn=chess_com, lichess_pgn=lichess, game_quality=quality, output=tmp_path / "encoded")
    base = tmp_path / "artifacts" / "base_rl"; base.parent.mkdir()
    torch.manual_seed(1)
    export_model(ChessyModel(ModelConfig(channels=8, residual_blocks=1, group_norm_groups=8, value_channels=8, value_hidden=16)), base, metadata={"role": "base_rl"})
    config = {
        "format": "chessy-config-v1", "name": "personal-test", "seed": 1, "device": "cpu",
        "model": {"architecture": "residual-cnn-v1", "input_planes": 119, "action_planes": 73, "board_size": 8, "channels": 8, "residual_blocks": 1, "group_norm_groups": 8, "value_channels": 8, "value_hidden": 16, "value_classes": 3},
        "optimizer": {"type": "adamw", "learning_rate": .001, "weight_decay": .0001, "beta1": .9, "beta2": .999, "epsilon": .0001},
        "scheduler": {"type": "warmup-cosine", "warmup_steps": 1, "total_steps": 3, "minimum_lr_ratio": .1},
        "training": {"batch_size": 1, "gradient_clip_norm": 1., "snapshot_every_steps": 1, "keep_last_periodic": 2},
        "artifacts": {"runs_dir": "runs", "dataset_manifest": str(manifest.relative_to(tmp_path)), "replay_manifest": None, "league_manifest": None},
        "personalization": {"base_export": "artifacts/base_rl", "dataset_manifest": str(manifest.relative_to(tmp_path)), "sample_kind_weights": {"good_move": .75, "full_game": 1.}, "max_positions_per_game": 16, "policy_loss_weight": 1., "value_loss_weight": .25, "max_epochs": 1, "early_stopping_patience": 1, "early_stopping_min_delta": .0001, "validation_every_epochs": 1, "selection_metric": "policy_cross_entropy", "batch_size": 1, "cache_segments": 1},
    }
    path = tmp_path / "personal.yaml"; path.write_text(json.dumps(config))
    uninterrupted = run_personal_training(root=tmp_path, config_path=path)
    stopped = run_personal_training(root=tmp_path, config_path=path, stop_after_steps=1)
    finished = run_personal_training(root=tmp_path, resume=stopped)
    assert finished == stopped
    assert (finished / "validation").is_dir()
    _, checked, _ = select_snapshot(Run.open(finished))
    _, expected, _ = select_snapshot(Run.open(uninterrupted))
    assert checked["training_state"]["personal_state"]["phase"] == "complete"
    assert checked["training_state"]["sampler_state"]["pool"] == expected["training_state"]["sampler_state"]["pool"]
    assert checked["training_state"]["sampler_state"]["cursor"] == expected["training_state"]["sampler_state"]["cursor"]
    assert all(torch.equal(checked["model_state"][name], expected["model_state"][name]) for name in checked["model_state"])
    snapshot_count = len(list((finished / "snapshots").glob("step-*")))
    assert run_personal_training(root=tmp_path, resume=finished) == finished
    assert len(list((finished / "snapshots").glob("step-*"))) == snapshot_count
