"""Resumable supervised policy/value fine-tuning from an explicit base export."""
from __future__ import annotations

import json
import hashlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from chessy.config import load_config
from chessy.config.schema import ChessyConfig
from chessy.model import ChessyModel, load_model_export, resolve_device
from chessy.model.export import export_model
from chessy.personal.dataset import PersonalDataset
from chessy.personal.sampler import PersonalBatchSampler
from chessy.personal.validation import validate, write_validation_report
from chessy.run import Run
from chessy.snapshot import capture_rng, restore_rng, write_snapshot
from chessy.snapshot.loader import select_snapshot
from chessy.training.personal_state import PersonalState
from chessy.training.stop import StopController
from chessy.training.supervised_loss import supervised_policy_value_loss


def _seed(seed: int) -> np.random.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _optimizer(model: ChessyModel, config: ChessyConfig) -> torch.optim.Optimizer:
    value = config.optimizer
    return torch.optim.AdamW(model.parameters(), lr=value.learning_rate, weight_decay=value.weight_decay, betas=(value.beta1, value.beta2), eps=value.epsilon)


def _scheduler(optimizer: torch.optim.Optimizer, config: ChessyConfig) -> torch.optim.lr_scheduler.LambdaLR:
    value = config.scheduler
    def scale(step: int) -> float:
        if step < value.warmup_steps:
            return (step + 1) / max(1, value.warmup_steps)
        progress = min(1.0, max(0.0, (step - value.warmup_steps) / max(1, value.total_steps - value.warmup_steps)))
        return value.minimum_lr_ratio + (1 - value.minimum_lr_ratio) * .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for group in optimizer.state.values():
        for key, value in group.items():
            if isinstance(value, torch.Tensor):
                group[key] = value.to(device)


def _manifest(path: Path) -> dict[str, object]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _base(root: Path, config: ChessyConfig, device: torch.device) -> tuple[Path, ChessyModel, str, str]:
    assert config.personalization is not None
    unresolved = root / config.personalization.base_export
    if unresolved.is_symlink():
        raise ValueError("base export must not be a symlink")
    path = unresolved.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("base export must be inside the project")
    manifest = _manifest(path)
    metadata = manifest.get("metadata")
    role = metadata.get("role") if isinstance(metadata, dict) else None
    allowed_roles = {"base_rl", "fixture"} if config.personalization.allow_fixture_base else {"base_rl"}
    if role not in allowed_roles:
        raise ValueError("personalization requires an explicit model export with role base_rl")
    model = load_model_export(path, device=device)
    return path, model, str(manifest["weights"]["sha256"]), str(role)


def _model_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii")); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _run_state(run: Run, model: ChessyModel, state: PersonalState, sampler: PersonalBatchSampler, reason: str, stop_reason: str | None) -> dict[str, object]:
    return {"format": "chessy-run-state-v1", "run_id": run.id, "config_fingerprint": run.fingerprint, "global_step": state.global_step, "epoch": sampler.epoch, "samples_seen": state.samples_seen, "stage": "personal-supervised", "best_metric": None if math.isinf(state.best_metric) else state.best_metric, "best_step": state.best_step, "total_elapsed_seconds": state.elapsed_seconds, "last_completed_batch": state.global_step, "snapshot_reason": reason, "stop_reason": stop_reason, "created_at": __import__("chessy.run.logging", fromlist=["utcnow"]).utcnow(), "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()), "optimizer": "adamw", "scheduler": "warmup-cosine"}


def _snapshot(run: Run, model: ChessyModel, optimizer: torch.optim.Optimizer, scheduler: object, sampler: PersonalBatchSampler, generator: np.random.Generator, state: PersonalState, reason: str, tags: set[str], stop_reason: str | None = None) -> Path:
    payload = {"format": "chessy-training-state-v1", "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "sampler_state": sampler.state_dict(), "rng_state": capture_rng(generator), "gradient_scaler_state": None, "personal_state": state.state_dict()}
    return write_snapshot(run, model, payload, _run_state(run, model, state, sampler, reason, stop_reason), reason=reason, tags=tags)


def _prepare_new(root: Path, config_path: Path, device_override: str | None):
    config, source, resolved, fp = load_config(config_path)
    if config.personalization is None:
        raise ValueError("config has no personalization section")
    target = resolve_device(device_override or config.device)
    generator = _seed(config.seed)
    base_path, model, base_sha, base_role = _base(root, config, target)
    parent = {"kind": "model-export", "role": base_role, "path": str(base_path.relative_to(root)), "weights_sha256": base_sha, "mode": "weights-only"}
    run = Run.create(root, config, source, resolved, fp, parent=parent)
    dataset = PersonalDataset(root / config.personalization.dataset_manifest, split="train", cache_segments=config.personalization.cache_segments)
    validation = PersonalDataset(root / config.personalization.dataset_manifest, split="val", cache_segments=config.personalization.cache_segments)
    sampler = PersonalBatchSampler(dataset, config.personalization.batch_size, config.seed, dict(config.personalization.sample_kind_weights), config.personalization.max_positions_per_game, config.personalization.drop_last)
    optimizer = _optimizer(model, config); scheduler = _scheduler(optimizer, config)
    state = PersonalState(base_weights_sha256=base_sha, dataset_fingerprint=dataset.fingerprint)
    return run, config, generator, model, optimizer, scheduler, sampler, validation, state, target


def _prepare_resume(root: Path, resume: Path, device_override: str | None):
    run = Run.open(resume); config = run.config
    if config.personalization is None:
        raise ValueError("run is not a personalization run")
    snapshot, checked, rejected = select_snapshot(run)
    target = resolve_device(device_override or config.device); generator = np.random.default_rng(config.seed)
    _, base_model, base_sha, _ = _base(root, config, target)
    state = PersonalState.from_dict(checked["training_state"]["personal_state"])
    dataset = PersonalDataset(root / config.personalization.dataset_manifest, split="train", cache_segments=config.personalization.cache_segments)
    validation = PersonalDataset(root / config.personalization.dataset_manifest, split="val", cache_segments=config.personalization.cache_segments)
    if state.base_weights_sha256 != base_sha or state.dataset_fingerprint != dataset.fingerprint:
        raise ValueError("resume base export or personal dataset is incompatible")
    base_model.load_state_dict(checked["model_state"], strict=True)
    optimizer = _optimizer(base_model, config); scheduler = _scheduler(optimizer, config)
    optimizer.load_state_dict(checked["training_state"]["optimizer_state"]); _move_optimizer(optimizer, target); scheduler.load_state_dict(checked["training_state"]["scheduler_state"])
    sampler = PersonalBatchSampler(dataset, config.personalization.batch_size, config.seed, dict(config.personalization.sample_kind_weights), config.personalization.max_positions_per_game, config.personalization.drop_last)
    sampler.load_state_dict(checked["training_state"]["sampler_state"])
    restore_rng(checked["training_state"]["rng_state"], generator, target.type)
    if state.phase != "complete":
        run.events.append_event("resume_started", {"snapshot": snapshot.name, "rejected": rejected})
        if device_override:
            run.events.append_event("operational_override", {"device": device_override})
    return run, config, generator, base_model, optimizer, scheduler, sampler, validation, state, target


def run_personal_training(*, root: Path, config_path: Path | None = None, resume: Path | None = None, device: str | None = None, stop_after_steps: int | None = None) -> Path:
    if (config_path is None) == (resume is None):
        raise ValueError("provide exactly one config or resume run")
    root = Path(root).resolve()
    prepared = _prepare_resume(root, resume, device) if resume else _prepare_new(root, config_path, device)  # type: ignore[arg-type]
    run, config, generator, model, optimizer, scheduler, sampler, validation, state, target = prepared
    assert config.personalization is not None
    p = config.personalization
    if state.phase == "complete":
        return run.path
    reports = run.path / "validation"; reports.mkdir(exist_ok=True)
    if state.baseline_report is None:
        base_report = validate(model, validation, device=target, batch_size=p.batch_size, model_checksum=state.base_weights_sha256, config_fingerprint=run.fingerprint)
        base_path = write_validation_report(reports / f"baseline-{base_report['content_fingerprint'][:12]}.json", base_report)
        state.baseline_report = str(base_path.relative_to(run.path)); run.events.append_event("baseline_validation_completed", {"report": state.baseline_report})
    baseline = json.loads((run.path / state.baseline_report).read_text())
    started = time.monotonic()
    with StopController() as stopper:
        while state.phase == "train" and sampler.epoch < p.max_epochs:
            batch_indices = sampler.next_batch()
            if batch_indices is None:
                should_validate = (sampler.epoch + 1) % p.validation_every_epochs == 0 or sampler.epoch + 1 >= p.max_epochs
                if not should_validate:
                    sampler.next_epoch()
                    continue
                state.validation_epoch += 1
                report = validate(model, validation, device=target, batch_size=p.batch_size, model_checksum=_model_checksum(model), snapshot_step=state.global_step, config_fingerprint=run.fingerprint, baseline=baseline)
                report_path = write_validation_report(reports / f"epoch-{sampler.epoch:03d}-step-{state.global_step:012d}-{report['content_fingerprint'][:12]}.json", report)
                metric = float(report["metrics"][p.selection_metric])
                if not math.isfinite(metric):
                    raise ValueError("non-finite personal validation metric")
                if metric < state.best_metric - p.early_stopping_min_delta:
                    state.best_metric, state.best_step, state.best_epoch, state.best_report, state.patience = metric, state.global_step, sampler.epoch, str(report_path.relative_to(run.path)), 0
                    _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "best", {"best"})
                else:
                    state.patience += 1
                    _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "periodic", {"periodic"})
                run.events.append_event("validation_completed", {"epoch": sampler.epoch, "metric": metric, "report": str(report_path.relative_to(run.path)), "patience": state.patience})
                if state.patience >= p.early_stopping_patience:
                    state.phase = "export"; break
                sampler.next_epoch()
                continue
            model.train(); batch = sampler.dataset.batch(batch_indices)
            output = model(batch["boards"].to(target))
            loss, metrics = supervised_policy_value_loss(output.policy_logits, output.value_logits, batch["target_action"].to(target), batch["legal_mask"].to(target), batch["value_class"].to(target), policy_weight=p.policy_loss_weight, value_weight=p.value_loss_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm))
            if not math.isfinite(gradient):
                raise ValueError("non-finite personal gradient")
            optimizer.step(); scheduler.step(); state.global_step += 1; state.samples_seen += len(batch_indices)
            batch_seconds = time.monotonic() - started
            state.elapsed_seconds += batch_seconds; started = time.monotonic()
            metadata = batch["metadata"]; count = len(metadata)
            run.metrics.append_metric(state.global_step, sampler.epoch, {"total_loss": float(loss.detach().cpu()), "policy_loss": float(metrics["policy_loss"].detach().cpu()), "value_loss": float(metrics["value_loss"].detach().cpu()), "top1": float(metrics["top1"].mean().detach().cpu()), "true_move_probability": float(metrics["true_move_probability"].mean().detach().cpu()), "value_accuracy": float(metrics["value_accuracy"].mean().detach().cpu()), "gradient_norm": gradient, "lr": float(optimizer.param_groups[0]["lr"]), "samples_per_second": count / max(batch_seconds, 1e-12), "good_move_fraction": sum(row["sample_kind"] == 0 for row in metadata) / count, "chess_com_fraction": sum(row["source"] == 0 for row in metadata) / count, "white_fraction": sum(row["color"] == 1 for row in metadata) / count})
            stopping = stopper.requested.is_set() or (stop_after_steps is not None and state.global_step >= stop_after_steps)
            if stopping:
                _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "stop", {"stop"}, stopper.reason or "stop_after_steps")
                run.events.append_event("run_stopped", {"step": state.global_step}); return run.path
            if state.global_step % config.training.snapshot_every_steps == 0:
                _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "periodic", {"periodic"})
    state.phase = "export"
    _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "export", {"export"})
    if state.best_report is not None and float(baseline["metrics"][p.selection_metric]) - state.best_metric >= p.early_stopping_min_delta:
        # Export the actual best validation checkpoint, not whichever weights
        # happened to be resident after the final non-improving epoch.
        index = json.loads((run.path / "snapshots" / "index.json").read_text())
        from chessy.snapshot.writer import verify_snapshot
        best_snapshot = run.path / "snapshots" / index["best"]
        export_model_instance = ChessyModel(config.model.to_model_config()).to(target)
        export_model_instance.load_state_dict(verify_snapshot(best_snapshot, expected_run_id=run.id, expected_fingerprint=run.fingerprint)["model_state"], strict=True)
        export = run.path / "exports" / "personal-supervised"
        if not export.exists():
            staged = run.path / "exports" / ".personal-supervised-gate"
            if not staged.exists():
                export_model(export_model_instance, staged, metadata={"role": "personal_supervised", "owner_accounts": "mu1876,mu1878", "base_model_checksum": state.base_weights_sha256, "dataset_fingerprint": state.dataset_fingerprint, "best_validation_report": state.best_report, "best_epoch": str(state.best_epoch), "best_global_step": str(state.best_step), "selection_metric": p.selection_metric, "delta_from_base": str(float(baseline["metrics"][p.selection_metric]) - state.best_metric), "config_fingerprint": run.fingerprint})
            # The export remains hidden until it loads through the public model
            # API and completes two entirely legal arena games.
            from chessy.curriculum.sources import FullSource
            from chessy.evaluation import MCTSAgent, RandomAgent, run_arena
            from chessy.evaluation.arena import write_report
            from chessy.mcts import DirectModelEvaluator
            checked_model = load_model_export(staged, device="cpu")
            staged_manifest = json.loads((staged / "manifest.json").read_text())
            staged_metadata = staged_manifest.get("metadata", {})
            if staged_metadata.get("best_global_step") != str(state.best_step) or staged_metadata.get("dataset_fingerprint") != state.dataset_fingerprint:
                raise ValueError("unfinished personal export gate does not match resumable state")
            export_checksum = staged_manifest["weights"]["sha256"]
            arena = run_arena(candidate=MCTSAgent(DirectModelEvaluator(checked_model), simulations=1), opponent=RandomAgent(config.seed), positions=[FullSource().sample(np.random.default_rng(config.seed))], games=2, max_plies=80, candidate_checksum=export_checksum, opponent_checksum="random", config_fingerprint=run.fingerprint, seed=config.seed)
            write_report(reports / "arena-sanity.json", arena)
            staged.rename(export)
        run.events.append_event("personal_exported", {"path": str(export), "gate": "passed", "arena_report": "validation/arena-sanity.json"})
    else:
        run.events.append_event("personal_export_rejected", {"gate": "no validation improvement"})
    state.phase = "complete"
    _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "final", {"completed"})
    run.events.append_event("run_completed", {"step": state.global_step, "best_metric": state.best_metric})
    return run.path
