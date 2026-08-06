from __future__ import annotations

import pytest
import torch

from chessy.model import ChessyModel


@pytest.mark.parametrize("batch_size", (1, 2, 8))
def test_cpu_forward_returns_raw_float32_logits(batch_size: int) -> None:
    torch.manual_seed(1)
    model = ChessyModel()
    output = model(torch.randn((batch_size, 119, 8, 8), dtype=torch.float32))
    assert output.policy_logits.shape == (batch_size, 4672)
    assert output.value_logits.shape == (batch_size, 3)
    assert output.policy_logits.dtype == torch.float32
    assert output.value_logits.dtype == torch.float32
    assert torch.isfinite(output.policy_logits).all()
    assert torch.isfinite(output.value_logits).all()
    assert not torch.allclose(output.policy_logits.sum(dim=1), torch.ones(batch_size))


@pytest.mark.parametrize("shape", ((119, 8, 8), (2, 118, 8, 8), (2, 119, 7, 8)))
def test_forward_rejects_wrong_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        ChessyModel()(torch.zeros(shape, dtype=torch.float32))


def test_parameter_count_backward_and_determinism() -> None:
    torch.manual_seed(7)
    first = ChessyModel()
    boards = torch.randn((2, 119, 8, 8), dtype=torch.float32)
    output = first(boards)
    (output.policy_logits.mean() + output.value_logits.mean()).backward()
    assert 1_500_000 <= sum(parameter.numel() for parameter in first.parameters()) <= 2_000_000
    for parameter in (
        first.stem_conv.weight,
        first.residual_blocks[0].conv1.weight,
        first.policy_conv.weight,
        first.value_linear2.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    torch.manual_seed(7)
    second = ChessyModel()
    for first_value, second_value in zip(first.state_dict().values(), second.state_dict().values()):
        # Gradients are not part of state_dict, so construction remains exactly reproducible.
        assert torch.equal(first_value, second_value)
    with torch.inference_mode():
        assert first(boards).policy_logits.shape == (2, 4672)
