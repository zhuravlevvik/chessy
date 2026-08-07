"""Bounded in-memory game and model registry."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
        observer_runs_dir: Path | None = None,
        max_active_sessions: int = 8,
        simulations_override: int | None = None,
        model_catalog: Callable[[], list[tuple[ModelInfo, Path]]] | None = None,
        model_loader: Callable[[Path, ModelInfo], ModelRuntime] | None = None,
    ) -> None:
        if not models:
            raise ValueError("at least one model is required")
        if max_active_sessions <= 0:
            raise ValueError("max_active_sessions must be positive")
        self.models = {runtime.info.id: runtime for runtime in models}
        if len(self.models) != len(models):
            raise ValueError("model IDs must be unique")
        self.feedback_dir = Path(feedback_dir)
        self.observer_runs_dir = None if observer_runs_dir is None else Path(observer_runs_dir)
        self.max_active_sessions = max_active_sessions
        self.simulations_override = simulations_override
        self.model_catalog = model_catalog
        self.model_loader = model_loader
        self.model_sources: dict[str, tuple[ModelInfo, Path]] = {}
        self.sessions: dict[str, GameSession] = {}
        self.lock = threading.RLock()

    def _refresh_models(self) -> None:
        if self.model_catalog is None: return
        sources: dict[str, tuple[ModelInfo, Path]] = {}
        for info, path in self.model_catalog():
            existing = self.models.get(info.id)
            if existing is not None and existing.info.checksum != info.checksum: raise ValueError("model ID collision")
            sources.setdefault(info.id, (info, path))
        self.model_sources = sources

    def public_models(self) -> list[dict[str, object]]:
        with self.lock:
            self._refresh_models()
            infos = {runtime.info.id: runtime.info for runtime in self.models.values()}
            infos.update({model_id: source[0] for model_id, source in self.model_sources.items()})
            trained = sorted((info for info in infos.values() if not info.untrained), key=lambda info: info.name, reverse=True)
            untrained = sorted((info for info in infos.values() if info.untrained), key=lambda info: info.name)
            return [info.public_dict() for info in [*trained, *untrained]]

    def create(self, *, model_id: str, color: str, time_control: str, profile: str, feedback_opt_in: bool) -> GameSession:
        with self.lock:
            self._refresh_models()
            runtime = self.models.get(model_id)
            if runtime is None and model_id in self.model_sources:
                if self.model_loader is None: raise ValueError("model export cannot be loaded")
                info, path = self.model_sources[model_id]; runtime = self.model_loader(path, info)
                if runtime.info.id != info.id or runtime.info.checksum != info.checksum: raise ValueError("loaded model differs from catalog")
                self.models[model_id] = runtime
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
