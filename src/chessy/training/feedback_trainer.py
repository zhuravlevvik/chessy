"""Separate resumable supervised update mixing historical and human feedback."""
from __future__ import annotations
import hashlib, json, math, random, statistics, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
from chessy.config import load_config
from chessy.config.schema import ChessyConfig
from chessy.feedback.dataset import FeedbackDataset
from chessy.feedback.sampler import MixedPersonalBatchSampler
from chessy.model import ChessyModel, load_model_export, resolve_device
from chessy.model.export import export_model
from chessy.personal.dataset import PersonalDataset
from chessy.personal.sampler import PersonalBatchSampler
from chessy.personal.validation import validate, write_validation_report
from chessy.run import Run
from chessy.snapshot import capture_rng, restore_rng, write_snapshot
from chessy.snapshot.loader import select_snapshot
from chessy.training.feedback_state import FeedbackState
from chessy.training.personal_trainer import _move_optimizer, _optimizer, _scheduler
from chessy.training.stop import StopController
from chessy.training.supervised_loss import supervised_policy_value_loss

def _seed(seed: int) -> np.random.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); return np.random.default_rng(seed)
def _checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        item = value.detach().cpu().contiguous(); digest.update(name.encode()); digest.update(str(item.dtype).encode()); digest.update(repr(tuple(item.shape)).encode()); digest.update(item.numpy().tobytes())
    return digest.hexdigest()
def _base(root: Path, config: ChessyConfig, device: torch.device) -> tuple[Path, ChessyModel, str, str]:
    assert config.personalization is not None
    path = (root / config.personalization.base_export).resolve()
    if (root / config.personalization.base_export).is_symlink() or not path.is_relative_to(root.resolve()): raise ValueError("base export must be a safe project-local directory")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8")); metadata = manifest.get("metadata", {}); role = metadata.get("role") if isinstance(metadata, dict) else None
    allowed = {"personal_supervised", "personal_feedback"} | ({"fixture"} if config.personalization.allow_fixture_base else set())
    if role not in allowed: raise ValueError("feedback training requires personal_supervised or personal_feedback base export")
    return path, load_model_export(path, device=device), str(manifest["weights"]["sha256"]), str(role)
def _feedback_report(model: torch.nn.Module, dataset: FeedbackDataset, *, device: torch.device, batch_size: int, checksum: str, config_fingerprint: str, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    model.eval(); started = time.monotonic(); rows: list[dict[str, float]] = []; slices: dict[str, dict[str, list[float]]] = {"color": {}, "phase": {}, "model": {}}
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = dataset.batch(list(range(start, min(start + batch_size, len(dataset))))); output = model(batch["boards"].to(device)); _, metrics = supervised_policy_value_loss(output.policy_logits, output.value_logits, batch["target_action"].to(device), batch["legal_mask"].to(device), batch["value_class"].to(device), sample_weight=batch["sample_weight"].to(device))
            masked = output.policy_logits.masked_fill(~batch["legal_mask"].to(device), float("-inf")); target = batch["target_action"].to(device)
            for index, metadata in enumerate(batch["metadata"]):
                probability = masked[index].softmax(0)[target[index]]
                item = {"ce": float(metrics["policy_per_sample"][index].cpu()), "value_ce": float(metrics["value_per_sample"][index].cpu()), "top1": float(metrics["top1"][index].cpu()), "top3": float((masked[index].topk(3).indices == target[index]).any().cpu()), "top5": float((masked[index].topk(5).indices == target[index]).any().cpu()), "prob": float(probability.cpu()), "value_acc": float(metrics["value_accuracy"][index].cpu())}; rows.append(item)
                for key, value in (("color", str(metadata["color"])), ("phase", str(metadata["phase"])), ("model", str(metadata["model_id"]))): slices[key].setdefault(value, []).append(item["ce"])
    metrics = {"count": len(rows), "policy_cross_entropy": sum(item["ce"] for item in rows) / len(rows), "value_cross_entropy": sum(item["value_ce"] for item in rows) / len(rows), "top1": sum(item["top1"] for item in rows) / len(rows), "top3": sum(item["top3"] for item in rows) / len(rows), "top5": sum(item["top5"] for item in rows) / len(rows), "mean_true_move_probability": sum(item["prob"] for item in rows) / len(rows), "median_true_move_probability": statistics.median(item["prob"] for item in rows), "value_accuracy": sum(item["value_acc"] for item in rows) / len(rows)}
    elapsed = time.monotonic() - started
    report: dict[str, Any] = {"format": "chessy-feedback-adaptation-report-v1", "diagnostic_only": True, "model_checksum": checksum, "dataset_manifest_fingerprint": dataset.fingerprint, "metrics": metrics, "slices": {key: {name: {"count": len(values), "policy_cross_entropy": sum(values) / len(values)} for name, values in groups.items()} for key, groups in slices.items()}, "elapsed_seconds": elapsed, "samples_per_second": len(rows) / max(elapsed, 1e-12), "config_fingerprint": config_fingerprint, "baseline_delta": None if baseline is None else metrics["policy_cross_entropy"] - baseline["metrics"]["policy_cross_entropy"], "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    from chessy.config.canonical import fingerprint
    report["content_fingerprint"] = fingerprint({key: value for key, value in report.items() if key not in {"created_at", "elapsed_seconds", "samples_per_second", "content_fingerprint"}}); return report
def _mixed_batch(historical: PersonalDataset, feedback: FeedbackDataset, indices: dict[str, list[int]]) -> dict[str, Any]:
    parts = []
    if indices["historical"]:
        value = historical.batch(indices["historical"]); value["sample_weight"] = torch.ones(len(indices["historical"]), dtype=torch.float32); value["stream"] = "historical"; parts.append(value)
    if indices["feedback"]: parts.append(feedback.batch(indices["feedback"]))
    if not parts: raise ValueError("empty mixed batch")
    return {"boards": torch.cat([item["boards"] for item in parts]), "legal_mask": torch.cat([item["legal_mask"] for item in parts]), "target_action": torch.cat([item["target_action"] for item in parts]), "value_class": torch.cat([item["value_class"] for item in parts]), "sample_weight": torch.cat([item["sample_weight"] for item in parts]), "streams": ["historical"] * len(indices["historical"]) + ["feedback"] * len(indices["feedback"])}
def _snapshot(run: Run, model: ChessyModel, optimizer: torch.optim.Optimizer, scheduler: object, sampler: MixedPersonalBatchSampler, generator: np.random.Generator, state: FeedbackState, reason: str, tags: set[str]) -> Path:
    payload = {"format": "chessy-training-state-v1", "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "sampler_state": sampler.state_dict(), "rng_state": capture_rng(generator), "gradient_scaler_state": None, "feedback_state": state.state_dict()}
    run_state = {"format": "chessy-run-state-v1", "run_id": run.id, "config_fingerprint": run.fingerprint, "global_step": state.global_step, "epoch": sampler.historical.epoch, "samples_seen": state.samples_seen, "stage": "personal-feedback", "best_metric": None if math.isinf(state.best_feedback_ce) else state.best_feedback_ce, "best_step": state.best_step, "total_elapsed_seconds": state.elapsed_seconds, "last_completed_batch": state.global_step, "snapshot_reason": reason, "stop_reason": None, "created_at": __import__("chessy.run.logging", fromlist=["utcnow"]).utcnow(), "model_parameter_count": sum(item.numel() for item in model.parameters()), "optimizer": "adamw", "scheduler": "warmup-cosine"}
    return write_snapshot(run, model, payload, run_state, reason=reason, tags=tags)
def _prepare_new(root: Path, config_path: Path, device_override: str | None):
    config, source, resolved, fp = load_config(config_path)
    if config.personalization is None or config.human_feedback is None or not config.human_feedback.enabled or not config.human_feedback.dataset_manifest: raise ValueError("config must explicitly enable human_feedback")
    device = resolve_device(device_override or config.device); generator = _seed(config.seed); base_path, model, sha, role = _base(root, config, device)
    historical = PersonalDataset(root / config.personalization.dataset_manifest, split="train", cache_segments=config.personalization.cache_segments); validation = PersonalDataset(root / config.personalization.dataset_manifest, split="val", cache_segments=config.personalization.cache_segments); feedback = FeedbackDataset(root / config.human_feedback.dataset_manifest, cache_segments=config.human_feedback.cache_segments)
    if float(feedback.manifest["sample_weight"]) != config.human_feedback.sample_weight or int(feedback.manifest["max_positions_per_game"]) != config.human_feedback.max_positions_per_game: raise ValueError("feedback dataset weighting/cap does not match training config")
    personal = PersonalBatchSampler(historical, config.personalization.batch_size, config.seed, dict(config.personalization.sample_kind_weights), config.personalization.max_positions_per_game, config.personalization.drop_last); sampler = MixedPersonalBatchSampler(personal, feedback, config.human_feedback.max_batch_fraction, config.human_feedback.max_positions_per_game, config.seed)
    run = Run.create(root, config, source, resolved, fp, parent={"kind": "model-export", "role": role, "path": str(base_path.relative_to(root)), "weights_sha256": sha, "mode": "weights-only"})
    if device_override: run.events.append_event("operational_override", {"device": device_override})
    optimizer = _optimizer(model, config)
    return run, config, generator, model, optimizer, _scheduler(optimizer, config), sampler, historical, validation, feedback, FeedbackState(base_weights_sha256=sha, historical_fingerprint=historical.fingerprint, feedback_fingerprint=feedback.fingerprint), device
def _prepare_resume(root: Path, resume: Path, device_override: str | None):
    run = Run.open(resume); config = run.config
    if config.personalization is None or config.human_feedback is None or not config.human_feedback.enabled or not config.human_feedback.dataset_manifest: raise ValueError("run is not a feedback training run")
    snapshot, checked, rejected = select_snapshot(run); device = resolve_device(device_override or config.device); generator = np.random.default_rng(config.seed); _, model, sha, _ = _base(root, config, device); state = FeedbackState.from_dict(checked["training_state"]["feedback_state"])
    historical = PersonalDataset(root / config.personalization.dataset_manifest, split="train", cache_segments=config.personalization.cache_segments); validation = PersonalDataset(root / config.personalization.dataset_manifest, split="val", cache_segments=config.personalization.cache_segments); feedback = FeedbackDataset(root / config.human_feedback.dataset_manifest, cache_segments=config.human_feedback.cache_segments)
    if float(feedback.manifest["sample_weight"]) != config.human_feedback.sample_weight or int(feedback.manifest["max_positions_per_game"]) != config.human_feedback.max_positions_per_game: raise ValueError("feedback dataset weighting/cap does not match training config")
    if (state.base_weights_sha256, state.historical_fingerprint, state.feedback_fingerprint) != (sha, historical.fingerprint, feedback.fingerprint): raise ValueError("feedback resume inputs are incompatible")
    personal = PersonalBatchSampler(historical, config.personalization.batch_size, config.seed, dict(config.personalization.sample_kind_weights), config.personalization.max_positions_per_game, config.personalization.drop_last); sampler = MixedPersonalBatchSampler(personal, feedback, config.human_feedback.max_batch_fraction, config.human_feedback.max_positions_per_game, config.seed); sampler.load_state_dict(checked["training_state"]["sampler_state"])
    model.load_state_dict(checked["model_state"], strict=True); optimizer = _optimizer(model, config); scheduler = _scheduler(optimizer, config); optimizer.load_state_dict(checked["training_state"]["optimizer_state"]); _move_optimizer(optimizer, device); scheduler.load_state_dict(checked["training_state"]["scheduler_state"]); restore_rng(checked["training_state"]["rng_state"], generator, device.type)
    if state.phase != "complete":
        run.events.append_event("resume_started", {"snapshot": snapshot.name, "rejected": rejected})
        if device_override: run.events.append_event("operational_override", {"device": device_override})
    return run, config, generator, model, optimizer, scheduler, sampler, historical, validation, feedback, state, device
def run_feedback_training(*, root: Path, config_path: Path | None = None, resume: Path | None = None, device: str | None = None, stop_after_steps: int | None = None) -> Path:
    if (config_path is None) == (resume is None): raise ValueError("provide exactly one config or resume run")
    root = Path(root).resolve(); prepared = _prepare_resume(root, resume, device) if resume else _prepare_new(root, config_path, device)  # type: ignore[arg-type]
    run, config, generator, model, optimizer, scheduler, sampler, historical, validation, feedback, state, target = prepared; assert config.personalization and config.human_feedback; p, h = config.personalization, config.human_feedback
    if state.phase == "complete": return run.path
    reports = run.path / "validation"; reports.mkdir(exist_ok=True)
    if state.baseline_historical_report is None:
        historic = validate(model, validation, device=target, batch_size=p.batch_size, model_checksum=state.base_weights_sha256, config_fingerprint=run.fingerprint); online = _feedback_report(model, feedback, device=target, batch_size=p.batch_size, checksum=state.base_weights_sha256, config_fingerprint=run.fingerprint)
        state.baseline_historical_report = str(write_validation_report(reports / f"historical-baseline-{historic['content_fingerprint'][:12]}.json", historic).relative_to(run.path)); state.baseline_feedback_report = str(write_validation_report(reports / f"feedback-baseline-{online['content_fingerprint'][:12]}.json", online).relative_to(run.path)); run.events.append_event("feedback_baselines_completed", {"historical": state.baseline_historical_report, "feedback": state.baseline_feedback_report})
    base_historical = json.loads((run.path / state.baseline_historical_report).read_text()); base_feedback = json.loads((run.path / state.baseline_feedback_report).read_text()); started = time.monotonic()
    with StopController() as stopper:
        while state.phase == "train" and sampler.historical.epoch < p.max_epochs:
            indices = sampler.next_batch()
            if indices is None:
                should_validate = (sampler.historical.epoch + 1) % p.validation_every_epochs == 0 or sampler.historical.epoch + 1 >= p.max_epochs
                if not should_validate: sampler.next_epoch(); continue
                state.validation_epoch += 1; historical_report = validate(model, validation, device=target, batch_size=p.batch_size, model_checksum=_checksum(model), snapshot_step=state.global_step, config_fingerprint=run.fingerprint, baseline=base_historical); feedback_report = _feedback_report(model, feedback, device=target, batch_size=p.batch_size, checksum=_checksum(model), config_fingerprint=run.fingerprint, baseline=base_feedback)
                combined = {"format": "chessy-feedback-validation-v1", "historical": historical_report, "feedback": feedback_report, "historical_delta": historical_report["baseline_delta"], "feedback_delta": feedback_report["baseline_delta"], "config_fingerprint": run.fingerprint, "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}; from chessy.config.canonical import fingerprint; combined["content_fingerprint"] = fingerprint({"format": combined["format"], "historical_report_fingerprint": historical_report["content_fingerprint"], "feedback_report_fingerprint": feedback_report["content_fingerprint"], "historical_delta": combined["historical_delta"], "feedback_delta": combined["feedback_delta"], "config_fingerprint": run.fingerprint})
                path = write_validation_report(reports / f"epoch-{sampler.historical.epoch:03d}-step-{state.global_step:012d}-{combined['content_fingerprint'][:12]}.json", combined); historical_ce, feedback_ce = float(historical_report["metrics"]["policy_cross_entropy"]), float(feedback_report["metrics"]["policy_cross_entropy"])
                allowed = (historical_ce <= float(base_historical["metrics"]["policy_cross_entropy"]) + h.historical_regression_tolerance and feedback_ce <= float(base_feedback["metrics"]["policy_cross_entropy"]) - h.feedback_min_delta and feedback_ce <= state.best_feedback_ce - h.feedback_min_delta)
                if allowed: state.best_feedback_ce, state.best_historical_ce, state.best_step, state.best_report, state.patience = feedback_ce, historical_ce, state.global_step, str(path.relative_to(run.path)), 0; _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "best", {"best"})
                else: state.patience += 1; _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "periodic", {"periodic"})
                if state.patience >= p.early_stopping_patience or sampler.historical.epoch + 1 >= p.max_epochs: state.phase = "export"; break
                sampler.next_epoch(); continue
            model.train(); batch = _mixed_batch(historical, feedback, indices); output = model(batch["boards"].to(target)); loss, metrics = supervised_policy_value_loss(output.policy_logits, output.value_logits, batch["target_action"].to(target), batch["legal_mask"].to(target), batch["value_class"].to(target), policy_weight=p.policy_loss_weight, value_weight=p.value_loss_weight, sample_weight=batch["sample_weight"].to(target)); optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm));
            if not math.isfinite(gradient): raise ValueError("non-finite feedback gradient")
            optimizer.step(); scheduler.step(); state.global_step += 1; state.samples_seen += len(batch["streams"]); state.stream_counts["historical"] += len(indices["historical"]); state.stream_counts["feedback"] += len(indices["feedback"]); elapsed = time.monotonic() - started; state.elapsed_seconds += elapsed; started = time.monotonic(); total = len(batch["streams"])
            historical_count, feedback_count = len(indices["historical"]), len(indices["feedback"])
            policy_rows, value_rows = metrics["policy_per_sample"], metrics["value_per_sample"]
            historical_policy = float(policy_rows[:historical_count].mean().detach().cpu()) if historical_count else 0.0; historical_value = float(value_rows[:historical_count].mean().detach().cpu()) if historical_count else 0.0
            feedback_policy = float(policy_rows[historical_count:].mean().detach().cpu()) if feedback_count else 0.0; feedback_value = float(value_rows[historical_count:].mean().detach().cpu()) if feedback_count else 0.0
            run.metrics.append_metric(state.global_step, sampler.historical.epoch, {"total_loss": float(loss.detach().cpu()), "policy_loss": float(metrics["policy_loss"].detach().cpu()), "value_loss": float(metrics["value_loss"].detach().cpu()), "historical_policy_loss": historical_policy, "historical_value_loss": historical_value, "feedback_policy_loss": feedback_policy, "feedback_value_loss": feedback_value, "historical_samples": historical_count, "feedback_samples": feedback_count, "feedback_fraction": feedback_count / total, "sample_weight_sum": float(batch["sample_weight"].sum()), "gradient_norm": gradient, "lr": float(optimizer.param_groups[0]["lr"]), "samples_per_second": total / max(elapsed, 1e-12)})
            if stopper.requested.is_set() or (stop_after_steps is not None and state.global_step >= stop_after_steps): _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "stop", {"stop"}); run.events.append_event("run_stopped", {"step": state.global_step}); return run.path
            if state.global_step % config.training.snapshot_every_steps == 0: _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "periodic", {"periodic"})
    state.phase = "export"; _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "export", {"export"})
    # A candidate is published only after both numerical gates.  The existing
    # personal trainer owns the detailed arena implementation; the strict load
    # here remains the first half of the publication gate.
    if state.best_report is not None and state.best_historical_ce is not None:
        export = run.path / "exports" / "personal-feedback"
        if not export.exists():
            # Export the weights frozen at the admissible validation boundary,
            # never later resident weights from a rejected epoch.
            index = json.loads((run.path / "snapshots" / "index.json").read_text())
            from chessy.snapshot.writer import verify_snapshot
            best = ChessyModel(config.model.to_model_config()).to(target)
            best.load_state_dict(verify_snapshot(run.path / "snapshots" / index["best"], expected_run_id=run.id, expected_fingerprint=run.fingerprint)["model_state"], strict=True)
            staged = run.path / "exports" / ".personal-feedback-gate"
            if not staged.exists(): export_model(best, staged, metadata={"role": "personal_feedback", "base_model_checksum": state.base_weights_sha256, "historical_dataset_fingerprint": state.historical_fingerprint, "feedback_dataset_fingerprint": state.feedback_fingerprint, "feedback_games": str(feedback.manifest["game_count"]), "feedback_samples": str(len(feedback)), "sample_weight": str(h.sample_weight), "max_batch_fraction": str(h.max_batch_fraction), "best_validation_report": state.best_report, "config_fingerprint": run.fingerprint})
            checked = load_model_export(staged, device="cpu")
            staged_metadata = json.loads((staged / "manifest.json").read_text()).get("metadata", {})
            if staged_metadata.get("best_validation_report") != state.best_report or staged_metadata.get("feedback_dataset_fingerprint") != state.feedback_fingerprint or staged_metadata.get("base_model_checksum") != state.base_weights_sha256: raise ValueError("staged feedback export does not match resumable state")
            from chessy.curriculum.sources import FullSource
            from chessy.evaluation import MCTSAgent, RandomAgent, run_arena
            from chessy.evaluation.arena import write_report
            from chessy.mcts import DirectModelEvaluator
            arena = run_arena(candidate=MCTSAgent(DirectModelEvaluator(checked), simulations=1), opponent=RandomAgent(config.seed), positions=[FullSource().sample(np.random.default_rng(config.seed))], games=2, max_plies=80, candidate_checksum=json.loads((staged / "manifest.json").read_text())["weights"]["sha256"], opponent_checksum="random", config_fingerprint=run.fingerprint, seed=config.seed)
            write_report(reports / "arena-sanity.json", arena); staged.rename(export)
        run.events.append_event("feedback_exported", {"path": str(export), "gate": "historical+feedback", "arena_report": "validation/arena-sanity.json"})
    else: run.events.append_event("feedback_export_rejected", {"gate": "no admissible candidate"})
    state.phase = "complete"; _snapshot(run, model, optimizer, scheduler, sampler, generator, state, "final", {"completed"}); run.events.append_event("run_completed", {"step": state.global_step}); return run.path
