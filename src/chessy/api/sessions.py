"""Bounded in-memory game and model registry."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from chessy.mcts import Evaluator, MCTS, profile_config
from chessy.play import GameSession, MCTSAgent, ModelInfo


@dataclass(frozen=True, slots=True)
class ModelRuntime:
    info: ModelInfo
    evaluator: Evaluator


class SessionRegistry:
    def __init__(
        self,
        models: list[ModelRuntime],
        *,
        feedback_dir: Path,
        max_active_sessions: int = 8,
        simulations_override: int | None = None,
    ) -> None:
        if not models:
            raise ValueError("at least one model is required")
        if max_active_sessions <= 0:
            raise ValueError("max_active_sessions must be positive")
        self.models = {runtime.info.id: runtime for runtime in models}
        if len(self.models) != len(models):
            raise ValueError("model IDs must be unique")
        self.feedback_dir = Path(feedback_dir)
        self.max_active_sessions = max_active_sessions
        self.simulations_override = simulations_override
        self.sessions: dict[str, GameSession] = {}
        self.lock = threading.RLock()

    def public_models(self) -> list[dict[str, object]]:
        return [runtime.info.public_dict() for runtime in self.models.values()]

    def create(self, *, model_id: str, color: str, time_control: str, profile: str, feedback_opt_in: bool) -> GameSession:
        with self.lock:
            runtime = self.models.get(model_id)
            if runtime is None:
                raise ValueError("unknown model ID")
            active = sum(session.status != "finished" for session in self.sessions.values())
            if active >= self.max_active_sessions:
                raise RuntimeError("active game limit reached")
            config = profile_config(profile, simulations=self.simulations_override)
            agent = MCTSAgent(MCTS(runtime.evaluator, config), runtime.info)
            session = GameSession(
                agent,
                human_color=color,
                time_control=time_control,
                profile=profile,
                feedback_opt_in=feedback_opt_in,
            )
            self.sessions[session.id] = session
            return session

    def get(self, game_id: str) -> GameSession:
        with self.lock:
            try:
                return self.sessions[game_id]
            except KeyError as exc:
                raise KeyError("game not found") from exc
