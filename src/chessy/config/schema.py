"""Strict source-config contract for reproducible training runs."""
from __future__ import annotations
import math
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


class ObserverConfig(StrictModel):
    enabled: bool = True
    archive_every_generations: int = Field(default=1, gt=0)
    live_game_index: int = Field(default=0, ge=0)


class PersonalizationConfig(StrictModel):
    """Configuration for the deliberately separate historical fine-tuning run."""
    base_export: str
    allow_fixture_base: bool = False
    dataset_manifest: str
    train_split: Literal["train"] = "train"
    validation_split: Literal["val"] = "val"
    sample_kind_weights: dict[Literal["good_move", "full_game"], float] = Field(
        default_factory=lambda: {"good_move": 0.75, "full_game": 1.0}
    )
    max_positions_per_game: int = Field(default=16, gt=0)
    policy_loss_weight: float = Field(default=1.0, ge=0)
    value_loss_weight: float = Field(default=0.25, ge=0)
    max_epochs: int = Field(default=30, gt=0)
    early_stopping_patience: int = Field(default=5, gt=0)
    early_stopping_min_delta: float = Field(default=0.0001, ge=0)
    validation_every_epochs: int = Field(default=1, gt=0)
    selection_metric: Literal["policy_cross_entropy"] = "policy_cross_entropy"
    batch_size: int = Field(default=512, gt=0)
    cache_segments: int = Field(default=2, gt=0)
    drop_last: bool = False

    @field_validator("base_export", "dataset_manifest")
    @classmethod
    def safe_path(cls, value: str) -> str:
        if not value or value.startswith("/") or ".." in value.split("/"):
            raise ValueError("personalization paths must be relative safe paths")
        return value

    @model_validator(mode="after")
    def valid_objective(self) -> "PersonalizationConfig":
        if set(self.sample_kind_weights) != {"good_move", "full_game"}:
            raise ValueError("sample_kind_weights must specify good_move and full_game")
        if any(weight <= 0 for weight in self.sample_kind_weights.values()):
            raise ValueError("sample_kind_weights must be positive")
        if self.policy_loss_weight == 0 and self.value_loss_weight == 0:
            raise ValueError("at least one personal loss weight must be positive")
        return self

class HumanFeedbackConfig(StrictModel):
    enabled: bool = False
    dataset_manifest: str | None = None
    sample_weight: float = Field(default=4.0, gt=0)
    max_batch_fraction: float = Field(default=.25, gt=0, le=.5)
    max_positions_per_game: int = Field(default=16, gt=0)
    cache_segments: int = Field(default=1, gt=0)
    historical_regression_tolerance: float = Field(default=.01, ge=0)
    feedback_min_delta: float = Field(default=.0001, ge=0)
    @field_validator("sample_weight", "max_batch_fraction", "historical_regression_tolerance", "feedback_min_delta")
    @classmethod
    def finite_number(cls, value: float) -> float:
        if not math.isfinite(value): raise ValueError("human feedback numeric settings must be finite")
        return value
    @field_validator("dataset_manifest")
    @classmethod
    def safe_path(cls, value: str | None) -> str | None:
        if value is None: return None
        if not value or value.startswith("/") or ".." in value.split("/"): raise ValueError("feedback manifest path must be relative and safe")
        return value
    @model_validator(mode="after")
    def enabled_needs_manifest(self) -> "HumanFeedbackConfig":
        if self.enabled != (self.dataset_manifest is not None): raise ValueError("feedback dataset_manifest is required exactly when feedback is enabled")
        return self


class PersonalRLConfig(StrictModel):
    """Explicit contract for RL that preserves a frozen personal style anchor."""
    enabled: Literal[True] = True
    incumbent_export: str
    allowed_incumbent_roles: list[Literal["personal_supervised", "personal_feedback", "personal_rl"]] = Field(default_factory=lambda: ["personal_supervised", "personal_feedback", "personal_rl"])
    base_rl_export: str
    personal_supervised_export: str
    historical_dataset_manifest: str
    feedback_dataset_manifest: str | None = None
    rl_policy_weight: float = Field(default=1.0, ge=0)
    rl_value_weight: float = Field(default=1.0, ge=0)
    style_strength: float = Field(default=.20, gt=0)
    style_policy_weight: float = Field(default=1.0, ge=0)
    style_value_weight: float = Field(default=.25, ge=0)
    feedback_strength: float = Field(default=.20, ge=0)
    feedback_sample_weight: float = Field(default=4.0, gt=0)
    historical_batch_size: int = Field(default=16, gt=0)
    feedback_batch_size: int = Field(default=8, gt=0)
    sample_kind_weights: dict[Literal["good_move", "full_game"], float] = Field(default_factory=lambda: {"good_move": .75, "full_game": 1.0})
    historical_max_positions_per_game: int = Field(default=16, gt=0)
    feedback_max_positions_per_game: int = Field(default=16, gt=0)
    historical_ce_regression_tolerance: float = Field(default=.02, ge=0)
    feedback_ce_regression_tolerance: float = Field(default=.02, ge=0)
    minimum_style_top1_ratio: float = Field(default=.95, ge=0, le=1)

    @field_validator("incumbent_export", "base_rl_export", "personal_supervised_export", "historical_dataset_manifest", "feedback_dataset_manifest")
    @classmethod
    def safe_path(cls, value: str | None) -> str | None:
        if value is None: return None
        if not value or value.startswith("/") or ".." in value.split("/"):
            raise ValueError("personal_rl paths must be relative and safe")
        return value

    @field_validator("rl_policy_weight", "rl_value_weight", "style_strength", "style_policy_weight", "style_value_weight", "feedback_strength", "feedback_sample_weight", "historical_ce_regression_tolerance", "feedback_ce_regression_tolerance", "minimum_style_top1_ratio")
    @classmethod
    def finite_number(cls, value: float) -> float:
        if not math.isfinite(value): raise ValueError("personal_rl numeric settings must be finite")
        return value

    @model_validator(mode="after")
    def complete_feedback_and_weights(self) -> "PersonalRLConfig":
        if not self.allowed_incumbent_roles: raise ValueError("allowed_incumbent_roles cannot be empty")
        if len(set(self.allowed_incumbent_roles)) != len(self.allowed_incumbent_roles): raise ValueError("allowed_incumbent_roles contains duplicates")
        if set(self.sample_kind_weights) != {"good_move", "full_game"} or any(not math.isfinite(value) or value <= 0 for value in self.sample_kind_weights.values()):
            raise ValueError("sample_kind_weights must contain positive finite good_move and full_game weights")
        if self.rl_policy_weight == 0 and self.rl_value_weight == 0:
            raise ValueError("personal_rl requires a positive RL objective")
        if self.style_policy_weight == 0 and self.style_value_weight == 0:
            raise ValueError("personal_rl requires a positive style objective")
        if (self.feedback_dataset_manifest is None) != (self.feedback_strength == 0):
            raise ValueError("feedback manifest and positive feedback_strength are required together")
        return self

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
    personalization: PersonalizationConfig | None = None
    human_feedback: HumanFeedbackConfig | None = None
    personal_rl: PersonalRLConfig | None = None
    observer: ObserverConfig | None = None

    @model_validator(mode="after")
    def complete_rl_sections(self) -> "ChessyConfig":
        values = (self.self_play, self.replay, self.rl, self.curriculum, self.evaluation)
        if any(value is not None for value in values) and not all(value is not None for value in values):
            raise ValueError("self_play, replay, rl, curriculum, and evaluation must be configured together")
        if self.personalization is not None and any(value is not None for value in values):
            raise ValueError("personalization and RL sections cannot be mixed in one run config v1")
        if self.personalization is not None and self.artifacts.dataset_manifest != self.personalization.dataset_manifest:
            raise ValueError("artifacts.dataset_manifest must pin personalization.dataset_manifest")
        if self.personalization is not None and self.training.batch_size != self.personalization.batch_size:
            raise ValueError("training.batch_size must match personalization.batch_size")
        if self.human_feedback is not None and self.human_feedback.enabled and any(value is not None for value in values):
            raise ValueError("human feedback cannot be enabled in an RL config")
        if self.human_feedback is not None and self.human_feedback.enabled and self.personalization is None:
            raise ValueError("human feedback requires personalization")
        if self.personal_rl is not None:
            if not all(value is not None for value in values):
                raise ValueError("personal_rl requires complete self_play, replay, rl, curriculum, and evaluation sections")
            if self.personalization is not None or (self.human_feedback is not None and self.human_feedback.enabled):
                raise ValueError("personal_rl cannot be mixed with personalization or human_feedback sections")
            if self.training.batch_size != self.rl.batch_size:  # type: ignore[union-attr]
                raise ValueError("training.batch_size must match rl.batch_size for personal_rl")
            if self.training.gradient_clip_norm != self.rl.gradient_clip_norm:  # type: ignore[union-attr]
                raise ValueError("training.gradient_clip_norm must match rl.gradient_clip_norm for personal_rl")
            if self.personal_rl.rl_policy_weight != self.rl.policy_loss_weight or self.personal_rl.rl_value_weight != self.rl.value_loss_weight:  # type: ignore[union-attr]
                raise ValueError("personal_rl RL weights must match the rl section")
            if self.artifacts.dataset_manifest != self.personal_rl.historical_dataset_manifest:
                raise ValueError("artifacts.dataset_manifest must pin personal_rl.historical_dataset_manifest")
        if self.observer is not None and self.observer.enabled and not all(value is not None for value in values):
            raise ValueError("training observer requires complete RL sections")
        if self.observer is not None and self.self_play is not None and self.observer.live_game_index >= self.self_play.games_per_generation:
            raise ValueError("observer live_game_index must identify a generated game")
        return self
