"""Model-independent and batched policy/value evaluators."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Protocol

import chess
import numpy as np
import torch

from chessy.encoding import ACTION_SIZE, encode_board, legal_action_mask
from chessy.model import ChessyModel, expected_value, legal_policy_probabilities


@dataclass(frozen=True, slots=True)
class Evaluation:
    policy: np.ndarray
    value: float


class Evaluator(Protocol):
    def evaluate(self, history: Sequence[chess.Board]) -> Evaluation: ...


def _model_device(model: ChessyModel) -> torch.device:
    return next(model.parameters()).device


class DirectModelEvaluator:
    """Inference adapter for a model already resident on its selected device."""

    def __init__(self, model: ChessyModel) -> None:
        self.model = model
        self.model.eval()
        self.device = _model_device(model)

    def evaluate(self, history: Sequence[chess.Board]) -> Evaluation:
        board = history[0]
        if board.outcome(claim_draw=True) is not None:
            raise ValueError("terminal positions must not be sent to the evaluator")
        encoded = torch.from_numpy(encode_board(history)).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(legal_action_mask(board)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(encoded)
            policy = legal_policy_probabilities(output.policy_logits, mask)[0].cpu().numpy()
            value = float(expected_value(output.value_logits)[0].cpu())
        return Evaluation(policy=policy, value=value)


@dataclass(slots=True)
class _Request:
    history: Sequence[chess.Board]
    future: Future[Evaluation]


class BatchingInferenceService:
    """One inference worker that coalesces concurrent callers into bounded batches."""

    _SENTINEL = object()

    def __init__(self, model: ChessyModel, *, max_batch_size: int = 32, max_batch_wait_ms: float = 2.0) -> None:
        if not 1 <= max_batch_size <= 32:
            raise ValueError("max_batch_size must be in [1, 32]")
        if not 0 <= max_batch_wait_ms <= 100:
            raise ValueError("max_batch_wait_ms must be in [0, 100]")
        self.model = model.eval()
        self.device = _model_device(model)
        self.max_batch_size = max_batch_size
        self.max_batch_wait_ms = max_batch_wait_ms
        self._queue: queue.Queue[_Request | object] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closing = False
        self.batch_sizes: list[int] = []

    def start(self) -> "BatchingInferenceService":
        with self._lock:
            if self._closing:
                raise RuntimeError("inference service is closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._worker, name="chessy-inference", daemon=True)
                self._thread.start()
        return self

    def evaluate(self, history: Sequence[chess.Board]) -> Evaluation:
        if history[0].outcome(claim_draw=True) is not None:
            raise ValueError("terminal positions must not be sent to the evaluator")
        future: Future[Evaluation] = Future()
        with self._lock:
            if self._closing:
                raise RuntimeError("inference service is closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._worker, name="chessy-inference", daemon=True)
                self._thread.start()
            self._queue.put(_Request(tuple(board.copy(stack=True) for board in history), future))
        return future.result()

    def _worker(self) -> None:
        while True:
            first = self._queue.get()
            if first is self._SENTINEL:
                return
            assert isinstance(first, _Request)
            batch = [first]
            deadline = time.monotonic() + self.max_batch_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is self._SENTINEL:
                    self._queue.put(self._SENTINEL)
                    break
                assert isinstance(item, _Request)
                batch.append(item)
            self.batch_sizes.append(len(batch))
            try:
                boards = torch.from_numpy(np.stack([encode_board(request.history) for request in batch])).to(self.device)
                masks = torch.from_numpy(np.stack([legal_action_mask(request.history[0]) for request in batch])).to(self.device)
                with torch.inference_mode():
                    output = self.model(boards)
                    policies = legal_policy_probabilities(output.policy_logits, masks).cpu().numpy()
                    values = expected_value(output.value_logits).cpu().numpy()
                for request, policy, value in zip(batch, policies, values, strict=True):
                    request.future.set_result(Evaluation(policy=policy, value=float(value)))
            except BaseException as exc:
                for request in batch:
                    request.future.set_exception(exc)

    def close(self) -> None:
        with self._lock:
            if self._closing:
                thread = self._thread
            else:
                self._closing = True
                thread = self._thread
                if thread is not None:
                    self._queue.put(self._SENTINEL)
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def __enter__(self) -> "BatchingInferenceService":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
