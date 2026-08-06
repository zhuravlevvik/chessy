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

class TemperatureConfig(StrictModel):
    initial: float = Field(ge=0)
    cutoff_ply: int = Field(ge=0)
    final: float = Field(ge=0)

class SelfPlayConfig(StrictModel):
    actors: int = Field(default=1, ge=1, le=8)
    games_per_generation: int = Field(default=1, gt=0)
    simulations: int = Field(default=64, gt=0)
    c_puct: float = Field(default=1.5, gt=0)
    root_noise: Literal[True] = True
    dirichlet_alpha: float = Field(default=0.3, gt=0)
    dirichlet_epsilon: float = Field(default=0.25, ge=0, le=1)
    temperature: TemperatureConfig = Field(default_factory=lambda: TemperatureConfig(initial=1.0, cutoff_ply=20, final=0.0))
    max_game_plies: int = Field(default=160, gt=0)
    inference_batch_size: int = Field(default=32, gt=0, le=32)
    inference_wait_ms: float = Field(default=2.0, ge=0, le=100)
    graceful_timeout_seconds: float = Field(default=30.0, gt=0)

class ReplayConfig(StrictModel):
    root_dir: str = "replay"
    samples_per_segment: int = Field(default=16384, gt=0)
    active_max_samples: int = Field(default=250000, gt=0)
    recent_fraction: float = Field(default=0.5, ge=0, le=1)
    recent_generations: int = Field(default=2, gt=0)
    cache_segments: int = Field(default=2, gt=0)
    hard_disk_limit_bytes: int | None = Field(default=None, gt=0)
    @model_validator(mode="after")
    def replay_window_is_possible(self) -> "ReplayConfig":
        if self.active_max_samples < self.samples_per_segment:
            raise ValueError("active_max_samples must be >= samples_per_segment")
        if not self.root_dir or self.root_dir.startswith("/") or ".." in self.root_dir.split("/"):
            raise ValueError("replay root_dir must be a relative safe path")
        return self

class RLConfig(StrictModel):
    policy_loss_weight: float = Field(default=1.0, ge=0)
    value_loss_weight: float = Field(default=1.0, ge=0)
    train_steps_per_generation: int = Field(default=250, gt=0)
    batch_size: int = Field(default=256, gt=0)
    gradient_clip_norm: float = Field(default=1.0, gt=0)

class StageMixConfig(StrictModel):
    endgames: float = Field(default=1.0, ge=0, le=1)
    reduced: float = Field(default=0.0, ge=0, le=1)
    full: float = Field(default=0.0, ge=0, le=1)
    @model_validator(mode="after")
    def sums_to_one(self) -> "StageMixConfig":
        if abs(self.endgames + self.reduced + self.full - 1.0) > 1e-6:
            raise ValueError("stage_mix must sum to 1")
        return self

class CurriculumConfig(StrictModel):
    initial_stage: Literal["endgames", "reduced", "full"] = "endgames"
    stage_mode: Literal["manual", "gated"] = "manual"
    stage_mix: StageMixConfig = Field(default_factory=StageMixConfig)
    reduced_max_material_imbalance: int = Field(default=12, ge=0)

class EvaluationConfig(StrictModel):
    games_per_match: int = Field(default=40, gt=0)
    simulations: int = Field(default=96, gt=0)
    promotion_min_score: float = Field(default=0.55, ge=0, le=1)
    promotion_min_games: int = Field(default=40, gt=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    require_lower_confidence_above: float = Field(default=0.5, ge=0, le=1)

    @field_validator("games_per_match")
    @classmethod
    def paired_games(cls, value: int) -> int:
        if value % 2: raise ValueError("games_per_match must be even for paired colors")
        return value

class ChessyConfig(StrictModel):
    format: Literal["chessy-config-v1"] = "chessy-config-v1"; name: str = Field(min_length=1, max_length=80)
    seed: int = Field(ge=0); device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    model: ModelConfigSchema; optimizer: OptimizerConfig; scheduler: SchedulerConfig; training: TrainingConfig; artifacts: ArtifactsConfig
    # Optional on input keeps step-5 resolved configs readable. New RL configs
    # materialise all these defaults before their bytes are fingerprinted.
    self_play: SelfPlayConfig | None = None
    replay: ReplayConfig | None = None
    rl: RLConfig | None = None
    curriculum: CurriculumConfig | None = None
    evaluation: EvaluationConfig | None = None

    @model_validator(mode="after")
    def complete_rl_sections(self) -> "ChessyConfig":
        values = (self.self_play, self.replay, self.rl, self.curriculum, self.evaluation)
        if any(value is not None for value in values) and not all(value is not None for value in values):
            raise ValueError("self_play, replay, rl, curriculum, and evaluation must be configured together")
        return self
