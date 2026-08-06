"""Versioned resumable state for the personal-RL state machine."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field


@dataclass
class PersonalRLState:
    phase: str = "train"
    generation: int = 0
    global_step: int = 0
    samples_seen: dict[str, int] = field(default_factory=lambda: {"rl": 0, "historical": 0, "feedback": 0})
    active_incumbent_export: str = ""
    active_incumbent_checksum: str = ""
    active_incumbent_generation: int = 0
    input_checksums: dict[str, str] = field(default_factory=dict)
    input_manifest_checksums: dict[str, str] = field(default_factory=dict)
    manifest_fingerprints: dict[str, str] = field(default_factory=dict)
    replay_manifest_path: str | None = None
    league_manifest_path: str | None = None
    training_block_boundary: int = 0
    sampler_epochs: dict[str, int] = field(default_factory=dict)
    best_style_metrics: dict[str, float] = field(default_factory=dict)
    best_strength_metrics: dict[str, float] = field(default_factory=dict)
    baseline_reports: dict[str, str] = field(default_factory=dict)
    pending_candidate_export: str | None = None
    pending_candidate_report: str | None = None
    generation_baseline_report: str | None = None
    generation_feedback_baseline_report: str | None = None
    elapsed_seconds: float = 0.0
    stop_reason: str | None = None

    def state_dict(self) -> dict[str, object]: return {"format": "chessy-personal-rl-state-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PersonalRLState":
        if value.get("format") != "chessy-personal-rl-state-v1": raise ValueError("invalid personal RL state")
        raw = {key: item for key, item in value.items() if key != "format"}
        if set(raw) != set(cls.__dataclass_fields__): raise ValueError("invalid personal RL state fields")
        if raw["phase"] not in {"selfplay", "train", "evaluate", "complete"}: raise ValueError("invalid personal RL phase")
        return cls(**raw)  # type: ignore[arg-type]
