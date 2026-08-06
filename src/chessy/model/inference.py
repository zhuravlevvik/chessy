"""Inference-only policy masking and WDL value helpers."""

from __future__ import annotations

import torch

from chessy.encoding import ACTION_SIZE


def _validate_policy_inputs(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> None:
    if not isinstance(policy_logits, torch.Tensor) or not isinstance(legal_mask, torch.Tensor):
        raise ValueError("policy logits and legal mask must be torch.Tensor instances")
    if policy_logits.ndim != 2 or policy_logits.shape[1] != ACTION_SIZE:
        raise ValueError(f"policy_logits must have shape [B, {ACTION_SIZE}]")
    if legal_mask.shape != policy_logits.shape:
        raise ValueError("legal_mask must have the same shape as policy_logits")
    if legal_mask.dtype != torch.bool:
        raise ValueError("legal_mask must have dtype torch.bool")
    if legal_mask.device != policy_logits.device:
        raise ValueError("legal_mask must be on the same device as policy_logits")
    if not policy_logits.is_floating_point():
        raise ValueError("policy_logits must be floating-point")


def mask_policy_logits(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Return policy logits with illegal actions set to exactly negative infinity."""
    _validate_policy_inputs(policy_logits, legal_mask)
    return policy_logits.masked_fill(~legal_mask, float("-inf"))


def legal_policy_probabilities(
    policy_logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Softmax policy logits only across legal actions in every batch row."""
    _validate_policy_inputs(policy_logits, legal_mask)
    if not torch.all(legal_mask.any(dim=1)):
        raise ValueError("every policy row must contain at least one legal action")
    return torch.softmax(mask_policy_logits(policy_logits, legal_mask), dim=1)


def _validate_value_logits(value_logits: torch.Tensor) -> None:
    if not isinstance(value_logits, torch.Tensor):
        raise ValueError("value_logits must be a torch.Tensor")
    if value_logits.ndim != 2 or value_logits.shape[1] != 3:
        raise ValueError("value_logits must have shape [B, 3] in loss/draw/win order")
    if not value_logits.is_floating_point():
        raise ValueError("value_logits must be floating-point")


def value_probabilities(value_logits: torch.Tensor) -> torch.Tensor:
    """Convert loss/draw/win logits to probabilities."""
    _validate_value_logits(value_logits)
    return torch.softmax(value_logits, dim=1)


def expected_value(value_logits: torch.Tensor) -> torch.Tensor:
    """Return ``P(win) - P(loss)`` for every position in the batch."""
    probabilities = value_probabilities(value_logits)
    return probabilities[:, 2] - probabilities[:, 0]
