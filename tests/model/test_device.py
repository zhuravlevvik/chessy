from __future__ import annotations

import pytest
import torch

from chessy.model import ChessyModel, resolve_device


def test_cpu_and_auto_are_resolvable() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}


def test_unavailable_explicit_backends_are_rejected() -> None:
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="MPS"):
            resolve_device("mps")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")
    with pytest.raises(ValueError, match="unknown"):
        resolve_device("wrong")  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
@pytest.mark.parametrize("batch_size", (1, 8))
def test_mps_forward(batch_size: int) -> None:
    model = ChessyModel().to("mps").eval()
    with torch.inference_mode():
        output = model(torch.randn((batch_size, 119, 8, 8), device="mps"))
    assert output.policy_logits.shape == (batch_size, 4672)
    assert output.value_logits.shape == (batch_size, 3)
    assert torch.isfinite(output.policy_logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward() -> None:
    model = ChessyModel().to("cuda").eval()
    with torch.inference_mode():
        output = model(torch.randn((1, 119, 8, 8), device="cuda"))
    assert output.policy_logits.shape == (1, 4672)
