from types import SimpleNamespace
import torch

from chessy.personal_rl.gates import style_gate
from chessy.personal_rl.loss import personal_rl_loss


def _outputs(batch: int, action: int = 4672):
    return SimpleNamespace(policy_logits=torch.zeros(batch, action), value_logits=torch.zeros(batch, 3))


def _rl_batch(batch: int):
    mask = torch.zeros(batch, 4672, dtype=torch.bool); mask[:, :2] = True
    target = torch.zeros(batch, 4672); target[:, 0] = 1
    return {"policy_target": target, "legal_mask": mask, "value_class": torch.zeros(batch, dtype=torch.long)}


def _style_batch(batch: int):
    mask = torch.zeros(batch, 4672, dtype=torch.bool); mask[:, :2] = True
    return {"target_action": torch.zeros(batch, dtype=torch.long), "legal_mask": mask, "value_class": torch.zeros(batch, dtype=torch.long)}


def test_three_stream_loss_is_independently_normalised_and_feedback_is_weighted() -> None:
    common = dict(rl_output=_outputs(2), rl_batch=_rl_batch(2), historical_output=_outputs(3), historical_batch=_style_batch(3), style_strength=.2, style_policy_weight=1., style_value_weight=.25)
    without, _ = personal_rl_loss(**common)
    with_feedback, values = personal_rl_loss(**common, feedback_output=_outputs(1), feedback_batch=_style_batch(1), feedback_strength=.2, feedback_sample_weight=4.)
    assert values["feedback_loss"] > 0
    assert with_feedback > without
    doubled, _ = personal_rl_loss(**{**common, "historical_output": _outputs(6), "historical_batch": _style_batch(6)})
    assert torch.allclose(doubled, without)


def test_disabled_feedback_rejects_accidental_dataset_forward() -> None:
    try:
        personal_rl_loss(rl_output=_outputs(1), rl_batch=_rl_batch(1), historical_output=_outputs(1), historical_batch=_style_batch(1), feedback_output=_outputs(1))
    except ValueError as exc:
        assert "feedback tensors" in str(exc)
    else: raise AssertionError("feedback must not be consumed when disabled")


def test_style_gate_needs_ce_and_top1() -> None:
    baseline = {"metrics": {"policy_cross_entropy": 1., "top1": .6}}
    candidate = {"metrics": {"policy_cross_entropy": 1.01, "top1": .55}}
    gate = style_gate(baseline=baseline, candidate=candidate, historical_ce_tolerance=.02, minimum_top1_ratio=.95)
    assert not gate["passed"]
    assert gate["checks"]["historical_ce"]
    assert not gate["checks"]["historical_top1"]
