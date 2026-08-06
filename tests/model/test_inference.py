from __future__ import annotations

import chess
import pytest
import torch

from chessy.encoding import legal_action_mask
from chessy.model import (
    expected_value,
    legal_policy_probabilities,
    mask_policy_logits,
    value_probabilities,
)


def test_policy_mask_and_legal_probabilities() -> None:
    logits = torch.zeros((1, 4672), dtype=torch.float32)
    mask = torch.from_numpy(legal_action_mask(chess.Board())).unsqueeze(0)
    original_mask = mask.clone()
    masked = mask_policy_logits(logits, mask)
    assert torch.equal(masked[mask], logits[mask])
    assert torch.isneginf(masked[~mask]).all()
    probabilities = legal_policy_probabilities(logits, mask)
    assert probabilities[~mask].eq(0).all()
    assert probabilities[mask].gt(0).sum() == 20
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
    assert torch.equal(mask, original_mask)


def test_policy_input_validation_and_empty_masks() -> None:
    logits = torch.zeros((1, 4672))
    mask = torch.ones((1, 4672), dtype=torch.bool)
    with pytest.raises(ValueError, match="same shape"):
        mask_policy_logits(logits, mask[:, :-1])
    with pytest.raises(ValueError, match="torch.bool"):
        mask_policy_logits(logits, mask.to(torch.int64))
    with pytest.raises(ValueError, match="at least one"):
        legal_policy_probabilities(logits, torch.zeros_like(mask))


def test_wdl_helpers_have_expected_semantics() -> None:
    logits = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, -5.0], [-5.0, 0.0, 5.0]])
    probabilities = value_probabilities(logits)
    values = expected_value(logits)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3))
    assert values[0] == 0
    assert values[1] < 0
    assert values[2] > 0
    assert torch.all((-1 <= values) & (values <= 1))
    with pytest.raises(ValueError, match=r"\[B, 3\]"):
        value_probabilities(torch.zeros((1, 2)))
