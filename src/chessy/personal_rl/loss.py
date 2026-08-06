"""Independently normalised mixed RL/style objective."""
from __future__ import annotations

from typing import Any
import torch

from chessy.training.rl_loss import policy_value_loss
from chessy.training.supervised_loss import supervised_policy_value_loss


def _supervised(outputs: Any, batch: dict[str, torch.Tensor], *, policy_weight: float, value_weight: float, sample_weight: float | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = None if sample_weight is None else torch.full_like(batch["target_action"], sample_weight, dtype=torch.float32)
    return supervised_policy_value_loss(outputs.policy_logits, outputs.value_logits, batch["target_action"], batch["legal_mask"], batch["value_class"], policy_weight=policy_weight, value_weight=value_weight, sample_weight=weights)


def personal_rl_loss(*, rl_output: Any, rl_batch: dict[str, torch.Tensor], historical_output: Any, historical_batch: dict[str, torch.Tensor], feedback_output: Any | None = None, feedback_batch: dict[str, torch.Tensor] | None = None, rl_policy_weight: float = 1.0, rl_value_weight: float = 1.0, style_strength: float = .2, style_policy_weight: float = 1.0, style_value_weight: float = .25, feedback_strength: float = 0.0, feedback_sample_weight: float = 4.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return documented total loss without coupling stream batch sizes.

    Every helper averages only its own mini-batch before the configured stream
    multiplier is applied.  The optional feedback tensors are intentionally
    rejected when disabled so its dataset is never implicitly consumed.
    """
    if style_strength <= 0: raise ValueError("style_strength must be positive")
    if feedback_strength < 0 or feedback_sample_weight <= 0: raise ValueError("invalid feedback weights")
    rl, rl_metrics = policy_value_loss(rl_output.policy_logits, rl_output.value_logits, rl_batch["policy_target"], rl_batch["legal_mask"], rl_batch["value_class"], policy_weight=rl_policy_weight, value_weight=rl_value_weight)
    historical, historical_metrics = _supervised(historical_output, historical_batch, policy_weight=style_policy_weight, value_weight=style_value_weight)
    if feedback_strength == 0:
        if feedback_output is not None or feedback_batch is not None: raise ValueError("feedback tensors supplied while feedback is disabled")
        feedback = rl.new_zeros(())
        feedback_metrics: dict[str, torch.Tensor] = {}
    else:
        if feedback_output is None or feedback_batch is None: raise ValueError("feedback tensors required when feedback is enabled")
        # The feedback dataset remains physically separate.  Its explicit
        # confidence weight scales this stream rather than manufacturing extra
        # rows; per-batch CE itself is still an independent mean.
        raw_feedback, feedback_metrics = _supervised(feedback_output, feedback_batch, policy_weight=style_policy_weight, value_weight=style_value_weight)
        feedback = raw_feedback * feedback_sample_weight
    total = rl + style_strength * historical + feedback_strength * feedback
    if not torch.isfinite(total): raise ValueError("non-finite personal RL loss")
    return total, {"total_loss": total, "rl_loss": rl, "historical_loss": historical, "feedback_loss": feedback, "rl_policy_loss": rl_metrics["policy_loss"], "rl_value_loss": rl_metrics["value_loss"], "rl_policy_entropy": rl_metrics["policy_entropy"], "historical_policy_loss": historical_metrics["policy_loss"], "historical_value_loss": historical_metrics["value_loss"], "historical_top1": historical_metrics["top1"].mean(), "historical_true_move_probability": historical_metrics["true_move_probability"].mean(), **({"feedback_policy_loss": feedback_metrics["policy_loss"], "feedback_value_loss": feedback_metrics["value_loss"], "feedback_top1": feedback_metrics["top1"].mean()} if feedback_metrics else {})}
