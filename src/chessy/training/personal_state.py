"""Serializable state specific to historical supervised fine-tuning."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PersonalState:
    phase: str = "train"
    global_step: int = 0
    samples_seen: int = 0
    validation_epoch: int = 0
    best_metric: float = float("inf")
    best_step: int | None = None
    best_epoch: int | None = None
    best_report: str | None = None
    baseline_report: str | None = None
    patience: int = 0
    base_weights_sha256: str = ""
    dataset_fingerprint: str = ""
    elapsed_seconds: float = 0.0

    def state_dict(self) -> dict[str, object]:
        return {"format": "chessy-personal-state-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PersonalState":
        if value.get("format") != "chessy-personal-state-v1":
            raise ValueError("invalid personal training state")
        fields = set(cls.__dataclass_fields__)
        values = {key: item for key, item in value.items() if key != "format"}
        if set(values) != fields:
            raise ValueError("invalid personal training state fields")
        return cls(**values)
