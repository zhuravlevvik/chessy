from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from chessy.curriculum.sources import StartPosition, source_for_stage

@dataclass
class CurriculumState:
    format: str = "chessy-curriculum-v1"
    stage: str = "endgames"
    stage_mode: str = "manual"
    stage_mix: dict[str, float] | None = None
    transitions: list[dict[str, object]] | None = None
    def state_dict(self) -> dict[str, object]: return asdict(self)

class CurriculumManager:
    def __init__(self, state: CurriculumState, *, max_plies: int, max_material_imbalance: int = 12) -> None:
        self.state, self.max_plies, self.max_material_imbalance = state, max_plies, max_material_imbalance
        if state.format != "chessy-curriculum-v1": raise ValueError("invalid curriculum state")
        if state.stage_mix is None: state.stage_mix = {"endgames": 1.0, "reduced": 0.0, "full": 0.0}
        if state.transitions is None: state.transitions = []
    def sample(self, rng: np.random.Generator) -> StartPosition:
        names = ("endgames", "reduced", "full"); weights = np.array([self.state.stage_mix[name] for name in names], dtype=float)
        stage = names[int(rng.choice(len(names), p=weights / weights.sum()))]
        return source_for_stage(stage, max_plies=self.max_plies, max_material_imbalance=self.max_material_imbalance).sample(rng)
    def transition(self, stage: str, mix: dict[str, float], *, reason: str) -> None:
        if self.state.stage_mode != "gated": raise ValueError("manual curriculum requires an explicit fork")
        if stage not in mix or abs(sum(mix.values()) - 1.0) > 1e-6: raise ValueError("invalid stage transition")
        self.state.stage, self.state.stage_mix = stage, dict(mix); self.state.transitions.append({"stage": stage, "mix": dict(mix), "reason": reason})
