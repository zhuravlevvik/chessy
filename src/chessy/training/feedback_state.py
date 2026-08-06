"""Serializable state for the separate human-feedback personalization run."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field

@dataclass
class FeedbackState:
    phase: str = "train"
    global_step: int = 0
    samples_seen: int = 0
    validation_epoch: int = 0
    baseline_historical_report: str | None = None
    baseline_feedback_report: str | None = None
    best_feedback_ce: float = float("inf")
    best_historical_ce: float | None = None
    best_step: int | None = None
    best_report: str | None = None
    patience: int = 0
    base_weights_sha256: str = ""
    historical_fingerprint: str = ""
    feedback_fingerprint: str = ""
    stream_counts: dict[str, int] = field(default_factory=lambda: {"historical": 0, "feedback": 0})
    elapsed_seconds: float = 0.0
    def state_dict(self) -> dict[str, object]: return {"format": "chessy-feedback-state-v1", **asdict(self)}
    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "FeedbackState":
        if value.get("format") != "chessy-feedback-state-v1": raise ValueError("invalid feedback training state")
        fields = set(cls.__dataclass_fields__); values = {key: item for key, item in value.items() if key != "format"}
        if set(values) != fields: raise ValueError("invalid feedback training state fields")
        return cls(**values)  # type: ignore[arg-type]
