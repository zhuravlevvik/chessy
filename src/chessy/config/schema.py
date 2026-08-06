"""Strict source-config contract for reproducible training runs."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from chessy.model import ModelConfig

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class ModelConfigSchema(StrictModel):
    architecture: Literal["residual-cnn-v1"] = "residual-cnn-v1"
    input_planes: int = 119; action_planes: int = 73; board_size: int = 8
    channels: int = 96; residual_blocks: int = 8; group_norm_groups: int = 8
    value_channels: int = 32; value_hidden: int = 128; value_classes: int = 3
    def to_model_config(self) -> ModelConfig: return ModelConfig(**self.model_dump())
    @model_validator(mode="after")
    def validate_model(self) -> "ModelConfigSchema":
        self.to_model_config(); return self

class OptimizerConfig(StrictModel):
    type: Literal["adamw"] = "adamw"; learning_rate: float = Field(gt=0); weight_decay: float = Field(ge=0)
    beta1: float = Field(ge=0, lt=1); beta2: float = Field(ge=0, lt=1); epsilon: float = Field(gt=0)

class SchedulerConfig(StrictModel):
    type: Literal["warmup-cosine"] = "warmup-cosine"; warmup_steps: int = Field(ge=0); total_steps: int = Field(gt=0)
    minimum_lr_ratio: float = Field(ge=0, le=1)
    @model_validator(mode="after")
    def warmup_before_total(self) -> "SchedulerConfig":
        if self.warmup_steps >= self.total_steps: raise ValueError("warmup_steps must be smaller than total_steps")
        return self

class TrainingConfig(StrictModel):
    batch_size: int = Field(gt=0); gradient_clip_norm: float = Field(gt=0)
    snapshot_every_steps: int = Field(gt=0); keep_last_periodic: int = Field(ge=2)

class ArtifactsConfig(StrictModel):
    runs_dir: str = "runs"; dataset_manifest: str | None = None; replay_manifest: str | None = None; league_manifest: str | None = None
    @field_validator("runs_dir", "dataset_manifest", "replay_manifest", "league_manifest")
    @classmethod
    def valid_artifact_path(cls, value: str | None) -> str | None:
        if value is None: return None
        if not value or value.startswith("/") or ".." in value.split("/"): raise ValueError("artifact paths must be relative safe paths")
        return value

class ChessyConfig(StrictModel):
    format: Literal["chessy-config-v1"] = "chessy-config-v1"; name: str = Field(min_length=1, max_length=80)
    seed: int = Field(ge=0); device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    model: ModelConfigSchema; optimizer: OptimizerConfig; scheduler: SchedulerConfig; training: TrainingConfig; artifacts: ArtifactsConfig
