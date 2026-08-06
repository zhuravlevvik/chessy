from __future__ import annotations
from pathlib import Path
import json
import pytest
import torch
from chessy.encoding import encode_move
from chessy.feedback import FeedbackDataset, MixedPersonalBatchSampler, build_feedback_dataset, verify_feedback_game
from chessy.feedback.fixture import prepare_feedback_smoke_fixture
from chessy.mcts import MCTSConfig, SearchAction, SearchResult
from chessy.personal.sampler import PersonalBatchSampler
from chessy.play import AgentDecision, GameSession, ModelInfo, save_human_feedback
from chessy.training.supervised_loss import supervised_policy_value_loss
from chessy.run import Run
from chessy.snapshot.loader import select_snapshot
from chessy.training.feedback_trainer import run_feedback_training

class Agent:
    def __init__(self) -> None:
        self.model = ModelInfo("test", "test", "a" * 64); self.config = MCTSConfig(simulations=1)
    def advance(self, *_: object) -> bool: return True
    def choose_move(self, environment):  # type: ignore[no-untyped-def]
        move = sorted(environment.legal_moves(), key=lambda item: item.uci())[0]; action = encode_move(environment.board, move)
        return AgentDecision(move, action, SearchResult(action, move, {action: SearchAction(1, 1, 1)}, 0, 1), 0, "test", "a" * 64, self.config)

def _game(root: Path, game_id: str) -> Path:
    session = GameSession(Agent(), human_color="white", feedback_opt_in=True, game_id=game_id)
    session.apply_human_move("e2e4"); session.play_bot_move(); session.resign()
    return save_human_feedback(session, root)

def test_verified_raw_build_is_content_addressed_and_tampering_fails(tmp_path: Path) -> None:
    raw = tmp_path / "raw"; game = _game(raw, "33333333-3333-4333-8333-333333333333")
    assert verify_feedback_game(game)["game_id"] == game.name
    manifest = build_feedback_dataset(input=raw, output=tmp_path / "encoded", segment_samples=1)
    assert build_feedback_dataset(input=raw, output=tmp_path / "encoded", segment_samples=1) == manifest
    assert FeedbackDataset(manifest)[0]["stream"] == "human_online"
    (game / "samples.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_feedback_game(game)


def test_feedback_writer_rejects_unsafe_or_conflicting_game_ids(tmp_path: Path) -> None:
    unsafe = GameSession(Agent(), human_color="white", feedback_opt_in=True, game_id="7" * 32)
    unsafe.resign()
    with pytest.raises(ValueError, match="UUID"):
        save_human_feedback(unsafe, tmp_path / "raw")
    assert not (tmp_path / "raw" / ("7" * 32)).exists()

    game_id = "77777777-7777-4777-8777-777777777777"
    _game(tmp_path / "raw", game_id)
    conflicting = GameSession(Agent(), human_color="black", feedback_opt_in=True, game_id=game_id)
    conflicting.resign()
    with pytest.raises(FileExistsError, match="different content"):
        save_human_feedback(conflicting, tmp_path / "raw")

class Historical:
    split = "train"; fingerprint = "historical-fixture"
    indices_by_game = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}
    def __len__(self) -> int: return 8
    def __getitem__(self, index: int) -> dict[str, int]: return {"sample_kind": index % 2}

def test_mixed_sampler_hard_cap_and_state_restore(tmp_path: Path) -> None:
    raw = tmp_path / "raw"; _game(raw, "44444444-4444-4444-8444-444444444444")
    feedback = FeedbackDataset(build_feedback_dataset(input=raw, output=tmp_path / "encoded"))
    first = MixedPersonalBatchSampler(PersonalBatchSampler(Historical(), 4, 3), feedback, .25, 16, 3)  # type: ignore[arg-type]
    batch = first.next_batch(); assert batch is not None and len(batch["feedback"]) <= 1
    state = first.state_dict(); second = MixedPersonalBatchSampler(PersonalBatchSampler(Historical(), 4, 3), feedback, .25, 16, 3)  # type: ignore[arg-type]
    second.load_state_dict(state); assert second.next_batch() == first.next_batch()


def test_mixed_sampler_never_exceeds_fraction_on_short_batch_and_does_not_recycle_within_epoch(tmp_path: Path) -> None:
    raw = tmp_path / "raw"; _game(raw, "55555555-5555-4555-8555-555555555555")
    feedback = FeedbackDataset(build_feedback_dataset(input=raw, output=tmp_path / "encoded"))
    historical = PersonalBatchSampler(Historical(), 3, 9)  # type: ignore[arg-type]
    sampler = MixedPersonalBatchSampler(historical, feedback, .5, 1, 9)
    batches = []
    while (batch := sampler.next_batch()) is not None:
        batches.append(batch)
        total = len(batch["historical"]) + len(batch["feedback"])
        assert len(batch["feedback"]) <= int(total * .5)
    assert sum(len(batch["feedback"]) for batch in batches) == 1
    sampler.next_epoch()
    assert len((sampler.next_batch() or {})["feedback"]) == 1

def test_weighted_loss_normalizes_by_sum_of_weights() -> None:
    policy = torch.zeros((2, 4672), requires_grad=True); value = torch.zeros((2, 3), requires_grad=True); legal = torch.zeros((2, 4672), dtype=torch.bool); legal[:, 0] = True
    loss, _ = supervised_policy_value_loss(policy, value, torch.tensor([0, 0]), legal, torch.tensor([1, 1]), sample_weight=torch.tensor([1.0, 4.0]))
    assert loss.item() == pytest.approx(.25 * torch.log(torch.tensor(3.0)).item())
    loss.backward(); assert torch.isfinite(policy.grad).all()


def test_builder_preserves_all_human_moves_and_sampler_owns_the_cap(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    session = GameSession(Agent(), human_color="white", feedback_opt_in=True, game_id="66666666-6666-4666-8666-666666666666")
    for _ in range(4):
        move = sorted(session.environment.legal_moves(), key=lambda item: item.uci())[0]
        session.apply_human_move(move.uci()); session.play_bot_move()
    session.resign(); save_human_feedback(session, raw)
    dataset = FeedbackDataset(build_feedback_dataset(input=raw, output=tmp_path / "encoded", max_positions_per_game=1))
    assert len(dataset) == 4
    sampler = MixedPersonalBatchSampler(PersonalBatchSampler(Historical(), 4, 2), dataset, .5, 1, 2)  # type: ignore[arg-type]
    batches = []
    while (batch := sampler.next_batch()) is not None: batches.append(batch)
    assert sum(len(batch["feedback"]) for batch in batches) == 1


def test_feedback_training_exact_resume_and_snapshot_pins_both_datasets(tmp_path: Path) -> None:
    prepare_feedback_smoke_fixture(tmp_path)
    config = json.loads(json.dumps(__import__("yaml").safe_load(Path("configs/personal-feedback-smoke.yaml").read_text())))
    config["device"] = "cpu"; config["personalization"]["max_epochs"] = 1; config["human_feedback"]["feedback_min_delta"] = 999.0
    path = tmp_path / "feedback.yaml"; path.write_text(__import__("yaml").safe_dump(config))
    uninterrupted = run_feedback_training(root=tmp_path, config_path=path)
    stopped = run_feedback_training(root=tmp_path, config_path=path, stop_after_steps=1)
    resumed = run_feedback_training(root=tmp_path, resume=stopped)
    _, expected, _ = select_snapshot(Run.open(uninterrupted)); latest, actual, _ = select_snapshot(Run.open(resumed))
    assert all(torch.equal(actual["model_state"][name], expected["model_state"][name]) for name in actual["model_state"])
    actual_sampler, expected_sampler = actual["training_state"]["sampler_state"], expected["training_state"]["sampler_state"]
    for key in ("feedback_pool", "feedback_cursor", "feedback_cycle"):
        assert actual_sampler[key] == expected_sampler[key]
    assert actual_sampler["historical_state"]["pool"] == expected_sampler["historical_state"]["pool"]
    assert actual_sampler["historical_state"]["cursor"] == expected_sampler["historical_state"]["cursor"]
    assert torch.equal(actual_sampler["generator_state"], expected_sampler["generator_state"])
    assert (latest / "feedback_manifest.json").is_file()
    assert set(json.loads((resumed / "run_manifest.json").read_text())["references"]) == {"dataset", "replay", "league", "feedback"}
