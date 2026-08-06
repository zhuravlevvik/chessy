"""Masked hard-move policy plus WDL loss used only by personalization."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_policy_value_loss(policy_logits: torch.Tensor, value_logits: torch.Tensor, target_action: torch.Tensor, legal_mask: torch.Tensor, value_class: torch.Tensor, *, policy_weight: float = 1.0, value_weight: float = .25) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if policy_logits.ndim != 2 or policy_logits.shape[1] != 4672 or value_logits.shape != (policy_logits.shape[0], 3):
        raise ValueError("incompatible supervised model outputs")
    if target_action.shape != (policy_logits.shape[0],) or value_class.shape != target_action.shape or legal_mask.shape != policy_logits.shape:
        raise ValueError("incompatible supervised targets")
    if target_action.dtype != torch.long or value_class.dtype != torch.long or legal_mask.dtype != torch.bool:
        raise ValueError("invalid supervised target dtypes")
    if not torch.isfinite(policy_logits).all() or not torch.isfinite(value_logits).all():
        raise ValueError("non-finite supervised logits")
    if (target_action < 0).any() or (target_action >= 4672).any() or (value_class < 0).any() or (value_class > 2).any() or not legal_mask.any(1).all():
        raise ValueError("invalid supervised targets")
    if not legal_mask.gather(1, target_action[:, None]).all():
        raise ValueError("supervised target action is illegal")
    masked = policy_logits.masked_fill(~legal_mask, float("-inf"))
    policy_per_sample = F.cross_entropy(masked, target_action, reduction="none")
    value_per_sample = F.cross_entropy(value_logits, value_class, reduction="none")
    policy_loss, value_loss = policy_per_sample.mean(), value_per_sample.mean()
    total = policy_weight * policy_loss + value_weight * value_loss
    if not torch.isfinite(total):
        raise ValueError("non-finite supervised loss")
    probabilities = masked.softmax(1)
    true_probability = probabilities.gather(1, target_action[:, None]).squeeze(1)
    return total, {"policy_loss": policy_loss, "value_loss": value_loss, "policy_per_sample": policy_per_sample, "value_per_sample": value_per_sample, "true_move_probability": true_probability, "top1": (masked.argmax(1) == target_action).float(), "value_accuracy": (value_logits.argmax(1) == value_class).float()}
