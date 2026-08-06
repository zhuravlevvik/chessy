from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import chess
import numpy as np
import pytest
import torch

from chessy.chess import ChessEnvironment
from chessy.encoding import legal_action_mask
from chessy.mcts import BatchingInferenceService, DirectModelEvaluator, MCTS, MCTSConfig
from chessy.model import ChessyModel, ModelConfig


def tiny_model() -> ChessyModel:
    torch.manual_seed(0)
    return ChessyModel(ModelConfig(channels=8, residual_blocks=1, value_channels=8, value_hidden=16)).eval()


def test_direct_evaluator_masks_illegal_actions_and_cpu_search_integrates() -> None:
    evaluator = DirectModelEvaluator(tiny_model())
    environment = ChessEnvironment()
    evaluation = evaluator.evaluate(environment.history())
    mask = legal_action_mask(environment.board)
    assert evaluation.policy.shape == (4672,)
    assert np.all(evaluation.policy[~mask] == 0)
    assert evaluation.policy[mask].sum() == pytest.approx(1)
    assert -1 <= evaluation.value <= 1
    assert MCTS(evaluator, MCTSConfig(simulations=2)).search(environment).move in environment.legal_moves()
    assert all(parameter.grad is None for parameter in evaluator.model.parameters())


def test_batch_service_coalesces_concurrent_requests_and_closes() -> None:
    service = BatchingInferenceService(tiny_model(), max_batch_size=4, max_batch_wait_ms=50)
    barrier = threading.Barrier(4)

    def evaluate(index: int) -> float:
        environment = ChessEnvironment()
        for _ in range(index):
            environment.push(environment.legal_moves()[0])
        barrier.wait()
        return service.evaluate(environment.history()).value

    with service, ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(evaluate, range(4)))
    assert len(values) == 4
    assert max(service.batch_sizes) == 4
    assert max(service.batch_sizes) <= 4
    with pytest.raises(RuntimeError, match="closed"):
        service.evaluate(ChessEnvironment().history())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_integration_search() -> None:
    evaluator = DirectModelEvaluator(tiny_model().to("mps"))
    assert MCTS(evaluator, MCTSConfig(simulations=1)).search(ChessEnvironment()).move in ChessEnvironment().legal_moves()
