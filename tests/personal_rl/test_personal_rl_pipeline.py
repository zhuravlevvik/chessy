import json
from pathlib import Path

import torch
import pytest

from chessy.personal.dataset import PersonalDataset
from chessy.personal_rl.fixture import prepare_personal_rl_smoke_fixture
from chessy.personal_rl.gates import promotion_gate
from chessy.personal_rl.sampler import PersonalRLSamplers
from chessy.replay import ReplayDataset, load_manifest
from chessy.snapshot.writer import verify_snapshot
from chessy.training.personal_rl_trainer import run_personal_rl
from chessy.evaluation.arena import ArenaReport
from chessy.config.canonical import fingerprint


def _latest(run: Path) -> dict[str, object]:
    index = json.loads((run / "snapshots" / "index.json").read_text())
    return verify_snapshot(run / "snapshots" / index["latest"])


def test_stop_resume_matches_uninterrupted_and_never_opens_test(tmp_path: Path, monkeypatch) -> None:
    prepare_personal_rl_smoke_fixture(tmp_path)
    config = Path("configs/personal-rl-smoke.yaml").resolve()
    opened: list[str] = []
    original = PersonalDataset.__init__

    def recording_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        split = kwargs.get("split", args[1] if len(args) > 1 else None)
        opened.append(str(split)); original(self, *args, **kwargs)

    monkeypatch.setattr(PersonalDataset, "__init__", recording_init)
    resumed = run_personal_rl(root=tmp_path, config_path=config, device="cpu", stop_after_steps=1)
    stopped = _latest(resumed)
    assert stopped["training_state"]["sampler_state"]["format"] == "chessy-personal-rl-samplers-v1"
    run_personal_rl(root=tmp_path, resume=resumed, device="cpu")
    uninterrupted = run_personal_rl(root=tmp_path, config_path=config, device="cpu")
    left, right = _latest(resumed), _latest(uninterrupted)
    assert all(torch.equal(left["model_state"][name], right["model_state"][name]) for name in left["model_state"])
    assert left["training_state"]["scheduler_state"] == right["training_state"]["scheduler_state"]
    assert left["training_state"]["personal_rl_state"]["global_step"] == right["training_state"]["personal_rl_state"]["global_step"] == 4
    assert "test" not in opened
    manifest = json.loads((resumed / "run_manifest.json").read_text())
    assert set(manifest["inputs"]["personal_rl"]) == {"incumbent", "base_rl", "personal_supervised"}
    assert list((resumed / "league").glob("league-g*.json"))
    assert list((resumed / "validation").glob("comparison-g*.json"))
    assert _latest(resumed)["run_state"]["stage"] == "personal-rl-complete"


def test_all_sampler_streams_restore_the_exact_next_batch(tmp_path: Path) -> None:
    prepare_personal_rl_smoke_fixture(tmp_path)
    run = run_personal_rl(root=tmp_path, config_path=Path("configs/personal-rl-smoke.yaml").resolve(), device="cpu", stop_after_steps=1)
    historical_path = tmp_path / "runs/personal-rl-fixture/encoded/manifests/personal-dataset-fixture.json"
    state = _latest(run)["training_state"]["personal_rl_state"]
    replay = ReplayDataset(load_manifest(tmp_path / state["replay_manifest_path"]))
    historical = PersonalDataset(historical_path, split="train")
    kwargs = dict(replay=replay, historical=historical, feedback=None, rl_batch_size=2, historical_batch_size=2, feedback_batch_size=1, seed=91, recent_fraction=.5, recent_generations=1, sample_kind_weights={"good_move": .75, "full_game": 1.}, historical_max_positions_per_game=16, feedback_max_positions_per_game=16)
    first = PersonalRLSamplers.create(**kwargs); first.next(); sampler_state = first.state_dict(); expected = first.next()
    restored = PersonalRLSamplers.create(**kwargs); restored.load_state_dict(sampler_state); actual = restored.next()
    assert torch.equal(expected["rl"], actual["rl"])
    assert expected["historical"] == actual["historical"]


def test_promotion_requires_both_independent_gates() -> None:
    class Arena:
        promoted = True
        eligible_for_promotion = True
        score = .6
        confidence_interval = (.51, .7)

    assert promotion_gate(arena=Arena(), style={"passed": True})["passed"]
    assert not promotion_gate(arena=Arena(), style={"passed": False})["passed"]
    Arena.promoted = False
    assert not promotion_gate(arena=Arena(), style={"passed": True})["passed"]


def test_full_gate_publishes_versioned_export(tmp_path: Path, monkeypatch) -> None:
    prepare_personal_rl_smoke_fixture(tmp_path)
    import chessy.training.personal_rl_trainer as trainer

    def passing_arena(**kwargs):  # type: ignore[no-untyped-def]
        body = {"candidate": kwargs["candidate_checksum"], "opponent": kwargs["opponent_checksum"], "seed": kwargs["seed"]}
        return ArenaReport("chessy-arena-report-v1", body["candidate"], body["opponent"], 40, 24, 8, 8, .7, (.55, .8), True, True, {"fixture": 40}, ["fixture"], fingerprint(body))

    monkeypatch.setattr(trainer, "run_arena", passing_arena)
    monkeypatch.setattr(trainer, "style_gate", lambda **_: {"passed": True, "checks": {"fixture": True}})
    run = run_personal_rl(root=tmp_path, config_path=Path("configs/personal-rl-smoke.yaml").resolve(), device="cpu")
    exports = list((run / "exports").glob("personal_rl-g*-s*"))
    assert len(exports) == 1
    manifest = json.loads((exports[0] / "manifest.json").read_text())
    assert manifest["metadata"]["role"] == "personal_rl"
    assert _latest(run)["training_state"]["personal_rl_state"]["active_incumbent_generation"] == 1


def test_resume_rejects_changed_pinned_export_manifest(tmp_path: Path) -> None:
    prepare_personal_rl_smoke_fixture(tmp_path)
    run = run_personal_rl(root=tmp_path, config_path=Path("configs/personal-rl-smoke.yaml").resolve(), device="cpu", stop_after_steps=1)
    manifest = tmp_path / "runs/personal-rl-fixture/base-rl/manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="checksum|pinned input changed"):
        run_personal_rl(root=tmp_path, resume=run, device="cpu")
