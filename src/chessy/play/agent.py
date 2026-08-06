"""Playable agent binding an MCTS tree to model metadata."""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess

from chessy.chess import ChessEnvironment
from chessy.mcts import MCTS, MCTSConfig, SearchResult


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    name: str
    checksum: str
    architecture: str = "residual-cnn-v1"
    untrained: bool = False
    random_seed: int | None = None

    def public_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "checksum": self.checksum,
            "architecture": self.architecture,
            "untrained": self.untrained,
        }


@dataclass(frozen=True, slots=True)
class AgentDecision:
    move: chess.Move
    action: int
    search_result: SearchResult
    elapsed_seconds: float
    model_id: str
    export_checksum: str
    mcts_config: MCTSConfig


class MCTSAgent:
    def __init__(self, mcts: MCTS, model: ModelInfo, *, clock: callable = time.monotonic) -> None:
        self.mcts = mcts
        self.model = model
        self.clock = clock

    @property
    def config(self) -> MCTSConfig:
        return self.mcts.config

    def choose_move(self, environment: ChessEnvironment) -> AgentDecision:
        if environment.is_terminal():
            raise ValueError("cannot choose a move in a terminal position")
        started = self.clock()
        result = self.mcts.search(environment)
        elapsed = max(0.0, self.clock() - started)
        if result.move not in environment.legal_moves():
            raise RuntimeError("MCTS returned an illegal move")
        return AgentDecision(
            move=result.move,
            action=result.action,
            search_result=result,
            elapsed_seconds=elapsed,
            model_id=self.model.id,
            export_checksum=self.model.checksum,
            mcts_config=self.config,
        )

    def advance(self, environment: ChessEnvironment, action: int) -> bool:
        return self.mcts.advance(environment, action)

    def reset(self) -> None:
        self.mcts.reset()
