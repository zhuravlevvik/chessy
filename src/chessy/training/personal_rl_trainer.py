"""Generation-based personal RL with a mandatory historical style anchor."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chessy.config import load_config
from chessy.config.canonical import canonical_json, fingerprint
from chessy.config.schema import ChessyConfig
from chessy.curriculum import CurriculumManager, CurriculumState
from chessy.evaluation import MCTSAgent, create_league, run_arena
from chessy.evaluation.arena import write_report
from chessy.feedback.dataset import FeedbackDataset
from chessy.mcts import BatchingInferenceService, DirectModelEvaluator, MCTSConfig
from chessy.model import ChessyModel, export_model, load_model_export, resolve_device
from chessy.personal.dataset import PersonalDataset
from chessy.personal.segment import verify_personal_manifest
from chessy.personal.validation import validate, write_validation_report
from chessy.personal_rl.gates import promotion_gate, style_gate
from chessy.personal_rl.comparison import comparison_report
from chessy.personal_rl.loss import personal_rl_loss
from chessy.personal_rl.sampler import PersonalRLSamplers
from chessy.replay import ReplayDataset, load_manifest, write_manifest, write_segment
from chessy.run import Run
from chessy.selfplay import SelfPlayCoordinator, TemperatureSchedule
from chessy.snapshot import capture_rng, restore_rng, write_snapshot
from chessy.snapshot.loader import select_snapshot
from chessy.training.personal_rl_state import PersonalRLState
from chessy.training.rl_trainer import _move_optimizer, _optimizer, _scheduler, _segment_chunks
from chessy.training.stop import StopController


def _seed(seed: int) -> np.random.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _safe_file(root: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts: raise ValueError(f"unsafe {label} path")
    unresolved = root / candidate
    if unresolved.is_symlink(): raise ValueError(f"unsafe {label} symlink")
    path = unresolved.resolve()
    if not path.is_relative_to(root) or not path.is_file(): raise ValueError(f"missing {label}")
    return path


def _export(root: Path, relative: str, *, roles: set[str], config: ChessyConfig, device: torch.device) -> tuple[Path, ChessyModel, str, str]:
    directory = _safe_file(root, f"{relative}/manifest.json", "model export").parent
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    role = metadata.get("role") if isinstance(metadata, dict) else None
    if role not in roles: raise ValueError(f"model export role must be one of {sorted(roles)}")
    if manifest.get("model_config") != config.model.to_model_config().to_dict(): raise ValueError("personal RL export architecture does not match config")
    # Strict loader verifies all files and the weights checksum before any run dir.
    return directory, load_model_export(directory, device=device), str(manifest["weights"]["sha256"]), str(role)


def _reference(root: Path, kind: str, relative: str | None) -> dict[str, object]:
    if relative is None: return {"format": "chessy-reference-v1", "kind": kind, "source": None, "source_sha256": None, "content": None}
    path = _safe_file(root, relative, f"{kind} manifest")
    raw = path.read_bytes()
    return {"format": "chessy-reference-v1", "kind": kind, "source": relative, "source_sha256": hashlib.sha256(raw).hexdigest(), "content": json.loads(raw)}


def _input_manifest(root: Path, values: dict[str, tuple[Path, str, str]]) -> dict[str, object]:
    return {
        name: {"path": str(directory.relative_to(root)), "role": role, "weights_sha256": checksum, "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()}
        for name, (directory, checksum, role) in sorted(values.items())
    }


def _write_json(path: Path, body: dict[str, Any]) -> Path:
    payload = canonical_json(body)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload: raise FileExistsError(f"report already exists with different content: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        with temporary.open("rb") as file: os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()
    return path


def _feedback_report(model: ChessyModel, dataset: FeedbackDataset, *, device: torch.device, checksum: str, config_fingerprint: str) -> dict[str, Any]:
    from chessy.training.supervised_loss import supervised_policy_value_loss
    rows: list[dict[str, float]] = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), 64):
            batch = dataset.batch(list(range(start, min(start + 64, len(dataset)))))
            output = model(batch["boards"].to(device)); _, values = supervised_policy_value_loss(output.policy_logits, output.value_logits, batch["target_action"].to(device), batch["legal_mask"].to(device), batch["value_class"].to(device))
            masked = output.policy_logits.masked_fill(~batch["legal_mask"].to(device), float("-inf")); target = batch["target_action"].to(device)
            for index in range(len(batch["target_action"])):
                rows.append({"ce": float(values["policy_per_sample"][index].cpu()), "top1": float(values["top1"][index].cpu()), "top3": float((masked[index].topk(3).indices == target[index]).any().cpu()), "top5": float((masked[index].topk(5).indices == target[index]).any().cpu()), "prob": float(values["true_move_probability"][index].cpu()), "value_acc": float(values["value_accuracy"][index].cpu())})
    metrics = {"count": len(rows), "policy_cross_entropy": sum(row["ce"] for row in rows) / len(rows), "top1": sum(row["top1"] for row in rows) / len(rows), "top3": sum(row["top3"] for row in rows) / len(rows), "top5": sum(row["top5"] for row in rows) / len(rows), "mean_true_move_probability": sum(row["prob"] for row in rows) / len(rows), "median_true_move_probability": statistics.median(row["prob"] for row in rows), "value_accuracy": sum(row["value_acc"] for row in rows) / len(rows)}
    body = {"format": "chessy-personal-rl-feedback-diagnostic-v1", "diagnostic_only": True, "model_checksum": checksum, "dataset_manifest_fingerprint": dataset.fingerprint, "metrics": metrics, "config_fingerprint": config_fingerprint}
    return {**body, "content_fingerprint": fingerprint(body)}


def _snapshot(run: Run, model: ChessyModel, optimizer: torch.optim.Optimizer, scheduler: object, samplers: PersonalRLSamplers | None, generator: np.random.Generator, state: PersonalRLState, reason: str, root: Path) -> Path:
    payload = {"format": "chessy-training-state-v1", "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "sampler_state": None if samplers is None else samplers.state_dict(), "rng_state": capture_rng(generator), "gradient_scaler_state": None, "personal_rl_state": state.state_dict()}
    run_state = {"format": "chessy-run-state-v1", "run_id": run.id, "config_fingerprint": run.fingerprint, "global_step": state.global_step, "epoch": state.sampler_epochs.get("historical", 0), "samples_seen": sum(state.samples_seen.values()), "stage": f"personal-rl-{state.phase}", "best_metric": state.best_style_metrics.get("policy_cross_entropy"), "best_step": state.global_step, "total_elapsed_seconds": state.elapsed_seconds, "last_completed_batch": state.global_step, "snapshot_reason": reason, "stop_reason": state.stop_reason, "created_at": __import__("chessy.run.logging", fromlist=["utcnow"]).utcnow(), "model_parameter_count": sum(item.numel() for item in model.parameters()), "optimizer": "adamw", "scheduler": "warmup-cosine", "generation": state.generation, "phase": state.phase, "replay_manifest": state.replay_manifest_path, "league_manifest": state.league_manifest_path}
    refs = {"dataset": _reference(root, "dataset", run.config.personal_rl.historical_dataset_manifest), "replay": _reference(root, "replay", state.replay_manifest_path), "league": _reference(root, "league", state.league_manifest_path)}  # type: ignore[union-attr]
    if run.config.personal_rl and run.config.personal_rl.feedback_dataset_manifest: refs["feedback"] = _reference(root, "feedback", run.config.personal_rl.feedback_dataset_manifest)
    return write_snapshot(run, model, payload, run_state, reason=reason, tags={reason}, references=refs)


def _verify_resume_inputs(root: Path, state: PersonalRLState, config: ChessyConfig) -> None:
    p = config.personal_rl; assert p
    specs = {"incumbent": (p.incumbent_export, set(p.allowed_incumbent_roles)), "base_rl": (p.base_rl_export, {"base_rl"}), "personal_supervised": (p.personal_supervised_export, {"personal_supervised"})}
    for name, (relative, roles) in specs.items():
        directory, _, checksum, _ = _export(root, relative, roles=roles, config=config, device=torch.device("cpu"))
        manifest_checksum = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        if checksum != state.input_checksums.get(name) or manifest_checksum != state.input_manifest_checksums.get(name): raise ValueError(f"personal RL pinned input changed: {name}")
    historical = verify_personal_manifest(_safe_file(root, p.historical_dataset_manifest, "historical manifest"))
    if str(historical["content_fingerprint"]) != state.manifest_fingerprints["historical"]: raise ValueError("personal RL historical manifest changed")
    if p.feedback_dataset_manifest:
        from chessy.feedback.segment import verify_feedback_manifest
        feedback = verify_feedback_manifest(_safe_file(root, p.feedback_dataset_manifest, "feedback manifest"))
        if str(feedback["content_fingerprint"]) != state.manifest_fingerprints["feedback"]: raise ValueError("personal RL feedback manifest changed")
    if state.replay_manifest_path:
        replay = load_manifest(_safe_file(root, state.replay_manifest_path, "replay manifest"))
        if replay.fingerprint != state.manifest_fingerprints.get("replay"): raise ValueError("personal RL replay manifest changed")
    if state.league_manifest_path:
        league = _reference(root, "league", state.league_manifest_path)["content"]
        if not isinstance(league, dict) or league.get("fingerprint") != state.manifest_fingerprints.get("league"): raise ValueError("personal RL league manifest changed")
    active = _safe_file(root, f"{state.active_incumbent_export}/manifest.json", "active incumbent").parent
    active_model = load_model_export(active, device="cpu")
    del active_model
    active_manifest = json.loads((active / "manifest.json").read_text())
    if str(active_manifest["weights"]["sha256"]) != state.active_incumbent_checksum: raise ValueError("personal RL active incumbent changed")


def _prepare_new(root: Path, config_path: Path, device_override: str | None):
    config, source, resolved, fp = load_config(config_path); p = config.personal_rl
    if p is None: raise ValueError("config must include personal_rl")
    device = resolve_device(device_override or config.device)
    incumbent_path, model, incumbent_sha, incumbent_role = _export(root, p.incumbent_export, roles=set(p.allowed_incumbent_roles), config=config, device=device)
    base_path, _, base_sha, _ = _export(root, p.base_rl_export, roles={"base_rl"}, config=config, device=device)
    supervised_path, _, supervised_sha, _ = _export(root, p.personal_supervised_export, roles={"personal_supervised"}, config=config, device=device)
    historical_manifest = verify_personal_manifest(_safe_file(root, p.historical_dataset_manifest, "historical manifest"))
    feedback: FeedbackDataset | None = None
    feedback_fp: str | None = None
    if p.feedback_dataset_manifest:
        feedback = FeedbackDataset(_safe_file(root, p.feedback_dataset_manifest, "feedback manifest")); feedback_fp = feedback.fingerprint
        if not math.isclose(float(feedback.manifest["sample_weight"]), p.feedback_sample_weight, rel_tol=1e-6, abs_tol=1e-7): raise ValueError("feedback sample weight does not match personal_rl config")
        if int(feedback.manifest["max_positions_per_game"]) != p.feedback_max_positions_per_game: raise ValueError("feedback per-game cap does not match personal_rl config")
    # Run.create is intentionally after every pinned input was verified.
    parent = {"kind": "model-export", "role": incumbent_role, "path": str(incumbent_path.relative_to(root)), "weights_sha256": incumbent_sha, "mode": "weights-only"}
    inputs = {"incumbent": (incumbent_path, incumbent_sha, incumbent_role), "base_rl": (base_path, base_sha, "base_rl"), "personal_supervised": (supervised_path, supervised_sha, "personal_supervised")}
    run = Run.create(root, config, source, resolved, fp, parent=parent, inputs={"personal_rl": _input_manifest(root, inputs)})
    if device_override: run.events.append_event("operational_override", {"device": device_override})
    historical_train = PersonalDataset(_safe_file(root, p.historical_dataset_manifest, "historical manifest"), split="train")
    historical_val = PersonalDataset(_safe_file(root, p.historical_dataset_manifest, "historical manifest"), split="val")
    # The test split is provenance only.  No PersonalDataset(test) is created.
    test_fp = fingerprint(historical_manifest["splits"]["test"])
    state = PersonalRLState(phase="selfplay", active_incumbent_export=str(incumbent_path.relative_to(root)), active_incumbent_checksum=incumbent_sha, input_checksums={name: value[1] for name, value in inputs.items()}, input_manifest_checksums={name: hashlib.sha256((value[0] / "manifest.json").read_bytes()).hexdigest() for name, value in inputs.items()}, manifest_fingerprints={"historical": historical_train.fingerprint, "historical_test": test_fp, **({"feedback": feedback_fp} if feedback_fp else {})})
    optimizer = _optimizer(model, config)
    return run, config, _seed(config.seed), model, optimizer, _scheduler(optimizer, config), historical_train, historical_val, feedback, state, device, (base_path, supervised_path)


def _prepare_resume(root: Path, resume: Path, device_override: str | None):
    run = Run.open(resume); config = run.config; p = config.personal_rl
    if p is None: raise ValueError("run is not a personal RL run")
    _, checked, rejected = select_snapshot(run); state = PersonalRLState.from_dict(checked["training_state"]["personal_rl_state"])
    _verify_resume_inputs(root, state, config); device = resolve_device(device_override or config.device); generator = np.random.default_rng(config.seed)
    active_path = _safe_file(root, f"{state.active_incumbent_export}/manifest.json", "active incumbent").parent
    model = load_model_export(active_path, device=device)
    model.load_state_dict(checked["model_state"], strict=True); optimizer = _optimizer(model, config); scheduler = _scheduler(optimizer, config); optimizer.load_state_dict(checked["training_state"]["optimizer_state"]); _move_optimizer(optimizer, device); scheduler.load_state_dict(checked["training_state"]["scheduler_state"]); restore_rng(checked["training_state"]["rng_state"], generator, device.type)
    historical_train = PersonalDataset(_safe_file(root, p.historical_dataset_manifest, "historical manifest"), split="train"); historical_val = PersonalDataset(_safe_file(root, p.historical_dataset_manifest, "historical manifest"), split="val")
    feedback = None if p.feedback_dataset_manifest is None else FeedbackDataset(_safe_file(root, p.feedback_dataset_manifest, "feedback manifest"))
    if device_override: run.events.append_event("operational_override", {"device": device_override})
    run.events.append_event("resume_completed", {"rejected_snapshots": rejected, "phase": state.phase, "step": state.global_step})
    return run, config, generator, model, optimizer, scheduler, historical_train, historical_val, feedback, state, device, None, checked["training_state"]["sampler_state"]


def inspect_personal_rl_run(*, root: Path, run_path: Path) -> dict[str, Any]:
    run = Run.open(run_path); config = run.config
    if config.personal_rl is None: raise ValueError("run is not a personal RL run")
    snapshot, checked, rejected = select_snapshot(run); state = PersonalRLState.from_dict(checked["training_state"]["personal_rl_state"])
    _verify_resume_inputs(root.resolve(), state, config)
    return {"format": "chessy-personal-rl-inspection-v1", "run": str(run.path), "snapshot": snapshot.name, "rejected_snapshots": rejected, "phase": state.phase, "generation": state.generation, "global_step": state.global_step, "active_incumbent": {"path": state.active_incumbent_export, "checksum": state.active_incumbent_checksum, "generation": state.active_incumbent_generation}, "manifest_fingerprints": state.manifest_fingerprints, "inputs_verified": True}


def personal_rl_evaluation_summary(run_path: Path) -> dict[str, Any]:
    run = Run.open(run_path)
    if run.config.personal_rl is None: raise ValueError("run is not a personal RL run")
    def reports(directory: str, pattern: str) -> list[dict[str, Any]]:
        return [json.loads(path.read_text()) for path in sorted((run.path / directory).glob(pattern))]
    return {"format": "chessy-personal-rl-evaluation-summary-v1", "run": str(run.path), "promotion_gates": reports("validation", "promotion-gate-*.json"), "comparisons": reports("validation", "comparison-*.json"), "arenas": reports("arena", "g*.json")}


def _generation_baseline(run: Run, model: ChessyModel, dataset: PersonalDataset, state: PersonalRLState, *, device: torch.device, batch_size: int) -> str:
    report = validate(model, dataset, device=device, batch_size=batch_size, model_checksum=state.active_incumbent_checksum, snapshot_step=state.global_step, config_fingerprint=run.fingerprint)
    path = write_validation_report(run.path / "validation" / f"incumbent-g{state.generation:04d}-historical-{report['content_fingerprint'][:12]}.json", report)
    return str(path.relative_to(run.path))


def _comparison(run: Run, *, root: Path, config: ChessyConfig, generation: int, incumbent_export: str, incumbent_checksum: str, incumbent_style: dict[str, Any], candidate_model: ChessyModel, candidate_checksum: str, candidate_style: dict[str, Any], positions: list[Any], device: torch.device, baseline_reports: dict[str, str]) -> Path:
    p, evaluation, sp = config.personal_rl, config.evaluation, config.self_play
    assert p and evaluation and sp
    _, base, base_checksum, _ = _export(root, p.base_rl_export, roles={"base_rl"}, config=config, device=device)
    _, supervised, supervised_checksum, _ = _export(root, p.personal_supervised_export, roles={"personal_supervised"}, config=config, device=device)
    models = {"base_rl": base, "personal_supervised": supervised, "personal_rl": candidate_model}
    checksums = {"base_rl": base_checksum, "personal_supervised": supervised_checksum, "personal_rl": candidate_checksum}
    styles = {
        "base_rl": json.loads((run.path / baseline_reports["base_rl"]).read_text())["metrics"],
        "personal_supervised": json.loads((run.path / baseline_reports["personal_supervised"]).read_text())["metrics"],
        "personal_rl": candidate_style["metrics"],
    }
    incumbent_path = _safe_file(root, f"{incumbent_export}/manifest.json", "comparison incumbent").parent
    incumbent_manifest = json.loads((incumbent_path / "manifest.json").read_text())
    if incumbent_manifest.get("metadata", {}).get("role") == "personal_feedback" and incumbent_checksum not in checksums.values():
        models["personal_feedback"] = load_model_export(incumbent_path, device=device); checksums["personal_feedback"] = incumbent_checksum; styles["personal_feedback"] = incumbent_style["metrics"]
    names = tuple(models)
    pairs = tuple((names[left], names[right]) for left in range(len(names)) for right in range(left + 1, len(names)))
    arenas: dict[str, Any] = {}
    for index, (left, right) in enumerate(pairs):
        report = run_arena(candidate=MCTSAgent(DirectModelEvaluator(models[left]), evaluation.simulations), opponent=MCTSAgent(DirectModelEvaluator(models[right]), evaluation.simulations), positions=positions, games=evaluation.games_per_match, max_plies=sp.max_game_plies, candidate_checksum=checksums[left], opponent_checksum=checksums[right], config_fingerprint=run.fingerprint, promotion_min_games=evaluation.promotion_min_games, promotion_min_score=evaluation.promotion_min_score, confidence_threshold=evaluation.require_lower_confidence_above, seed=config.seed + index)
        arenas[f"{left}_vs_{right}"] = report
        write_report(run.path / "arena" / f"comparison-{left}-vs-{right}-{report.fingerprint[:12]}.json", report)
    report = comparison_report(arenas=arenas, styles=styles, checksums=checksums, positions_fingerprint=fingerprint([position.fen for position in positions]), config_fingerprint=run.fingerprint)
    return _write_json(run.path / "validation" / f"comparison-g{generation:04d}-{report['content_fingerprint'][:12]}.json", report)


def run_personal_rl(*, root: Path, config_path: Path | None = None, resume: Path | None = None, device: str | None = None, stop_after_steps: int | None = None) -> Path:
    if (config_path is None) == (resume is None): raise ValueError("provide exactly one config or resume")
    root = root.resolve()
    prepared = _prepare_new(root, config_path, device) if config_path else _prepare_resume(root, resume, device)  # type: ignore[arg-type]
    if config_path:
        run, config, generator, model, optimizer, scheduler, historical_train, historical_val, feedback, state, target, _, = prepared
        saved_sampler = None
    else:
        run, config, generator, model, optimizer, scheduler, historical_train, historical_val, feedback, state, target, _, saved_sampler = prepared
    p, sp, replay_cfg, rl, curriculum_cfg, evaluation = config.personal_rl, config.self_play, config.replay, config.rl, config.curriculum, config.evaluation
    assert p and sp and replay_cfg and rl and curriculum_cfg and evaluation
    if state.phase == "complete": return run.path
    reports = run.path / "validation"; reports.mkdir(exist_ok=True); exports = run.path / "exports"; league_dir = run.path / "league"; league_dir.mkdir(exist_ok=True)
    # Baselines precede the first optimizer update and use val only.
    if not state.baseline_reports:
        initial = validate(model, historical_val, device=target, batch_size=p.historical_batch_size, model_checksum=state.active_incumbent_checksum, config_fingerprint=run.fingerprint)
        initial_path = write_validation_report(reports / f"incumbent-historical-{initial['content_fingerprint'][:12]}.json", initial); state.baseline_reports["incumbent_historical"] = str(initial_path.relative_to(run.path))
        for name, relative in (("base_rl", p.base_rl_export), ("personal_supervised", p.personal_supervised_export)):
            _, candidate, checksum, _ = _export(root, relative, roles={name}, config=config, device=target)
            report = validate(candidate, historical_val, device=target, batch_size=p.historical_batch_size, model_checksum=checksum, config_fingerprint=run.fingerprint)
            path = write_validation_report(reports / f"{name}-historical-{report['content_fingerprint'][:12]}.json", report); state.baseline_reports[name] = str(path.relative_to(run.path))
        if feedback:
            report = _feedback_report(model, feedback, device=target, checksum=state.active_incumbent_checksum, config_fingerprint=run.fingerprint); path = write_validation_report(reports / f"incumbent-feedback-{report['content_fingerprint'][:12]}.json", report); state.baseline_reports["incumbent_feedback"] = str(path.relative_to(run.path))
        run.events.append_event("personal_rl_baselines_completed", state.baseline_reports)
    if state.generation_baseline_report is None:
        state.generation_baseline_report = _generation_baseline(run, model, historical_val, state, device=target, batch_size=p.historical_batch_size)
    if feedback is not None and state.generation_feedback_baseline_report is None:
        report = _feedback_report(model, feedback, device=target, checksum=state.active_incumbent_checksum, config_fingerprint=run.fingerprint)
        path = write_validation_report(reports / f"incumbent-g{state.generation:04d}-feedback-{report['content_fingerprint'][:12]}.json", report); state.generation_feedback_baseline_report = str(path.relative_to(run.path))
    curriculum = CurriculumManager(CurriculumState(stage=curriculum_cfg.initial_stage, stage_mode=curriculum_cfg.stage_mode, stage_mix=curriculum_cfg.stage_mix.model_dump()), max_plies=sp.max_game_plies, max_material_imbalance=curriculum_cfg.reduced_max_material_imbalance)
    if state.league_manifest_path is None:
        league = create_league(league_dir / "league-0000-initial.json", incumbent=state.active_incumbent_generation, export_path=state.active_incumbent_export, export_checksum=state.active_incumbent_checksum, stage=curriculum.state.stage, tags=["personal-initial"])
        state.league_manifest_path = str(league.path.relative_to(root)); state.manifest_fingerprints["league"] = str(league.content["fingerprint"])
    samplers: PersonalRLSamplers | None = None
    started = time.monotonic()
    with StopController() as stopper:
        while state.global_step < config.scheduler.total_steps:
            incumbent = root / state.active_incumbent_export
            if state.phase == "selfplay":
                if state.generation_baseline_report is None:
                    state.generation_baseline_report = _generation_baseline(run, model, historical_val, state, device=target, batch_size=p.historical_batch_size)
                if feedback is not None and state.generation_feedback_baseline_report is None:
                    report = _feedback_report(model, feedback, device=target, checksum=state.active_incumbent_checksum, config_fingerprint=run.fingerprint)
                    path = write_validation_report(reports / f"incumbent-g{state.generation:04d}-feedback-{report['content_fingerprint'][:12]}.json", report); state.generation_feedback_baseline_report = str(path.relative_to(run.path))
                if replay_cfg.hard_disk_limit_bytes is not None and (root / replay_cfg.root_dir).exists():
                    used = sum(path.stat().st_size for path in (root / replay_cfg.root_dir).rglob("*") if path.is_file() and not path.is_symlink())
                    if used >= replay_cfg.hard_disk_limit_bytes: raise ValueError("replay hard disk limit reached")
                with BatchingInferenceService(model, max_batch_size=sp.inference_batch_size, max_batch_wait_ms=sp.inference_wait_ms) as service:
                    coordinator = SelfPlayCoordinator(run.id, config.seed, state.generation, sp.actors, service, curriculum, MCTSConfig(simulations=sp.simulations, c_puct=sp.c_puct, root_noise=True, dirichlet_alpha=sp.dirichlet_alpha, dirichlet_epsilon=sp.dirichlet_epsilon, max_batch_size=sp.inference_batch_size, max_batch_wait_ms=sp.inference_wait_ms), TemperatureSchedule(sp.temperature.initial, sp.temperature.cutoff_ply, sp.temperature.final), state.active_incumbent_checksum)
                    games, incomplete = coordinator.run(games=sp.games_per_generation, stop_requested=stopper.requested)
                if incomplete or stopper.requested.is_set():
                    state.stop_reason = stopper.reason or "stop_during_selfplay"; _snapshot(run, model, optimizer, scheduler, None, generator, state, "stop", root); return run.path
                old_segments: list[Path] = []
                if state.replay_manifest_path:
                    old = load_manifest(root / state.replay_manifest_path); old_segments = [old.path.parent.parent / entry["path"] for entry in old.content["segments"]]
                segments = [write_segment(root / replay_cfg.root_dir, generation=state.generation, ordinal=index, games=[game.sealed for game in chunk], run_id=run.id, model_checksum=state.active_incumbent_checksum) for index, chunk in enumerate(_segment_chunks(games, replay_cfg.samples_per_segment))]
                manifest = write_manifest(root / replay_cfg.root_dir, run_id=run.id, generation=state.generation, segments=old_segments + segments, active_max_samples=replay_cfg.active_max_samples, policy={"recent_fraction": replay_cfg.recent_fraction, "recent_generations": replay_cfg.recent_generations})
                state.replay_manifest_path = str(manifest.path.relative_to(root)); state.manifest_fingerprints["replay"] = manifest.fingerprint; state.training_block_boundary = min(state.global_step + rl.train_steps_per_generation, config.scheduler.total_steps); state.phase = "train"; saved_sampler = None; run.events.append_event("selfplay_completed", {"generation": state.generation, "replay": state.replay_manifest_path, "games": len(games)})
            if samplers is None:
                replay_data = ReplayDataset(load_manifest(root / state.replay_manifest_path), cache_segments=replay_cfg.cache_segments)  # type: ignore[arg-type]
                samplers = PersonalRLSamplers.create(replay=replay_data, historical=historical_train, feedback=feedback, rl_batch_size=rl.batch_size, historical_batch_size=p.historical_batch_size, feedback_batch_size=p.feedback_batch_size, seed=config.seed, recent_fraction=replay_cfg.recent_fraction, recent_generations=replay_cfg.recent_generations, sample_kind_weights=dict(p.sample_kind_weights), historical_max_positions_per_game=p.historical_max_positions_per_game, feedback_max_positions_per_game=p.feedback_max_positions_per_game)
                if saved_sampler is not None: samplers.load_state_dict(saved_sampler)
            while state.global_step < state.training_block_boundary:
                indices = samplers.next(); rb = samplers.replay.dataset.batch(indices["rl"]); hb = historical_train.batch(indices["historical"])  # type: ignore[arg-type]
                rb["policy_target"] = rb.pop("policy")
                def move(batch: dict[str, Any]) -> dict[str, Any]: return {key: value.to(target) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                rb, hb = move(rb), move(hb); fb = move(feedback.batch(indices["feedback"])) if feedback and "feedback" in indices else None  # type: ignore[arg-type]
                model.train(); optimizer.zero_grad(set_to_none=True)
                # Sequential forwards cap activation memory, followed by one backward/step.
                output_r, output_h = model(rb["boards"]), model(hb["boards"]); output_f = None if fb is None else model(fb["boards"])
                loss, metrics = personal_rl_loss(rl_output=output_r, rl_batch=rb, historical_output=output_h, historical_batch=hb, feedback_output=output_f, feedback_batch=fb, rl_policy_weight=p.rl_policy_weight, rl_value_weight=p.rl_value_weight, style_strength=p.style_strength, style_policy_weight=p.style_policy_weight, style_value_weight=p.style_value_weight, feedback_strength=p.feedback_strength, feedback_sample_weight=p.feedback_sample_weight)
                loss.backward(); gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), rl.gradient_clip_norm))
                if not math.isfinite(gradient): raise ValueError("non-finite personal RL gradient")
                optimizer.step(); scheduler.step(); state.global_step += 1; state.samples_seen["rl"] += len(rb["boards"]); state.samples_seen["historical"] += len(hb["boards"]); state.samples_seen["feedback"] += 0 if fb is None else len(fb["boards"]); state.sampler_epochs = {"historical": samplers.historical.epoch, "feedback": 0 if samplers.feedback is None else samplers.feedback.epoch}; elapsed = time.monotonic() - started; state.elapsed_seconds += elapsed; started = time.monotonic()
                metric_values = {name: float(value.detach().cpu()) for name, value in metrics.items() if isinstance(value, torch.Tensor) and value.ndim == 0}; metric_values.update({"rl_batch_size": len(rb["boards"]), "historical_batch_size": len(hb["boards"]), "feedback_batch_size": 0 if fb is None else len(fb["boards"]), "style_strength": p.style_strength, "feedback_strength": p.feedback_strength, "gradient_norm": gradient, "lr": float(optimizer.param_groups[0]["lr"]), "samples_per_second": (len(rb["boards"]) + len(hb["boards"]) + (0 if fb is None else len(fb["boards"]))) / max(elapsed, 1e-12), "generation": state.generation, "replay_draws": samplers.replay.draws})
                run.metrics.append_metric(state.global_step, samplers.historical.epoch, metric_values)
                if stopper.requested.is_set() or (stop_after_steps is not None and state.global_step >= stop_after_steps): state.stop_reason = stopper.reason or "stop_after_steps"; _snapshot(run, model, optimizer, scheduler, samplers, generator, state, "stop", root); run.events.append_event("run_stopped", {"step": state.global_step}); return run.path
                if state.global_step % config.training.snapshot_every_steps == 0: _snapshot(run, model, optimizer, scheduler, samplers, generator, state, "periodic", root)
            state.phase = "evaluate"; _snapshot(run, model, optimizer, scheduler, samplers, generator, state, "evaluate", root)
            candidate = exports / f"candidate-g{state.generation:04d}-s{state.global_step:012d}"
            if not candidate.exists(): export_model(model, candidate, metadata={"role": "personal_rl_candidate", "incumbent_checksum": state.active_incumbent_checksum, "config_fingerprint": run.fingerprint})
            candidate_manifest = json.loads((candidate / "manifest.json").read_text()); candidate_metadata = candidate_manifest.get("metadata", {})
            if candidate_metadata.get("role") != "personal_rl_candidate" or candidate_metadata.get("incumbent_checksum") != state.active_incumbent_checksum or candidate_metadata.get("config_fingerprint") != run.fingerprint: raise ValueError("candidate export provenance mismatch")
            candidate_checked = load_model_export(candidate, device="cpu")
            if any(not torch.equal(candidate_checked.state_dict()[name], tensor.detach().cpu()) for name, tensor in model.state_dict().items()): raise ValueError("candidate export differs from resumable model")
            candidate_checksum = str(candidate_manifest["weights"]["sha256"])
            assert state.generation_baseline_report is not None
            baseline = json.loads((run.path / state.generation_baseline_report).read_text()); current = validate(model, historical_val, device=target, batch_size=p.historical_batch_size, model_checksum=candidate_checksum, snapshot_step=state.global_step, config_fingerprint=run.fingerprint, baseline=baseline); current_path = write_validation_report(reports / f"candidate-historical-{current['content_fingerprint'][:12]}.json", current)
            feedback_baseline = feedback_current = None
            if feedback:
                assert state.generation_feedback_baseline_report is not None
                feedback_baseline = json.loads((run.path / state.generation_feedback_baseline_report).read_text()); feedback_current = _feedback_report(model, feedback, device=target, checksum=candidate_checksum, config_fingerprint=run.fingerprint); write_validation_report(reports / f"candidate-feedback-{feedback_current['content_fingerprint'][:12]}.json", feedback_current)
            style = style_gate(baseline=baseline, candidate=current, historical_ce_tolerance=p.historical_ce_regression_tolerance, minimum_top1_ratio=p.minimum_style_top1_ratio, feedback_baseline=feedback_baseline, feedback_candidate=feedback_current, feedback_ce_tolerance=p.feedback_ce_regression_tolerance)
            opponent = load_model_export(incumbent, device=target); positions = [curriculum.sample(np.random.default_rng(config.seed + state.global_step + index)) for index in range(evaluation.games_per_match // 2)]
            arena = run_arena(candidate=MCTSAgent(DirectModelEvaluator(model), evaluation.simulations), opponent=MCTSAgent(DirectModelEvaluator(opponent), evaluation.simulations), positions=positions, games=evaluation.games_per_match, max_plies=sp.max_game_plies, candidate_checksum=candidate_checksum, opponent_checksum=state.active_incumbent_checksum, config_fingerprint=run.fingerprint, promotion_min_games=evaluation.promotion_min_games, promotion_min_score=evaluation.promotion_min_score, confidence_threshold=evaluation.require_lower_confidence_above, seed=config.seed + state.global_step)
            arena_path = run.path / "arena" / f"g{state.generation:04d}-{arena.fingerprint[:12]}.json"; write_report(arena_path, arena); gate = promotion_gate(arena=arena, style=style)
            gate_body = {"format": "chessy-personal-rl-promotion-gate-v1", "generation": state.generation, "style": style, "promotion": gate, "candidate_report": str(current_path.relative_to(run.path)), "arena_report": str(arena_path.relative_to(run.path)), "config_fingerprint": run.fingerprint}
            gate_body["content_fingerprint"] = fingerprint(gate_body); gate_path = _write_json(reports / f"promotion-gate-g{state.generation:04d}.json", gate_body)
            comparison_path = _comparison(run, root=root, config=config, generation=state.generation, incumbent_export=state.active_incumbent_export, incumbent_checksum=state.active_incumbent_checksum, incumbent_style=baseline, candidate_model=model, candidate_checksum=candidate_checksum, candidate_style=current, positions=positions, device=target, baseline_reports=state.baseline_reports)
            state.best_style_metrics = dict(current["metrics"]); state.best_strength_metrics = {"score": arena.score, "ci_lower": arena.confidence_interval[0]}
            if gate["passed"]:
                final = exports / f"personal_rl-g{state.generation + 1:04d}-s{state.global_step:012d}"
                staged = exports / f".{final.name}-staging"
                publication = final if final.exists() else staged
                if not publication.exists(): export_model(model, publication, metadata={"role": "personal_rl", "incumbent_checksum": state.active_incumbent_checksum, "historical_dataset_fingerprint": historical_train.fingerprint, "config_fingerprint": run.fingerprint, "promotion_gate": str(gate_path.relative_to(run.path))})
                checked = load_model_export(publication, device="cpu")
                if torch.backends.mps.is_available(): load_model_export(publication, device="mps")
                staged_checksum = str(json.loads((publication / "manifest.json").read_text())["weights"]["sha256"])
                if staged_checksum != candidate_checksum: raise ValueError("staged personal RL export differs from candidate")
                # This post-gate pair verifies the newly written inference export,
                # not merely the in-memory candidate used by the strength arena.
                sanity = run_arena(candidate=MCTSAgent(DirectModelEvaluator(checked), 1), opponent=MCTSAgent(DirectModelEvaluator(opponent), 1), positions=[curriculum.sample(np.random.default_rng(config.seed + state.global_step))], games=2, max_plies=sp.max_game_plies, candidate_checksum=candidate_checksum, opponent_checksum=state.active_incumbent_checksum, config_fingerprint=run.fingerprint, promotion_min_games=40, seed=config.seed + state.global_step)
                write_report(run.path / "arena" / f"sanity-g{state.generation:04d}-{sanity.fingerprint[:12]}.json", sanity)
                if final.exists():
                    existing = load_model_export(final, device="cpu"); del existing
                    if str(json.loads((final / "manifest.json").read_text())["weights"]["sha256"]) != candidate_checksum: raise FileExistsError("published personal RL export conflicts with candidate")
                else: staged.rename(final)
                state.active_incumbent_export = str(final.relative_to(root)); state.active_incumbent_checksum = candidate_checksum; state.active_incumbent_generation += 1; run.events.append_event("personal_rl_promoted", {"export": state.active_incumbent_export, "arena": str(arena_path.relative_to(run.path))})
            else:
                # A rejected candidate never becomes the next self-play model.
                restored = load_model_export(incumbent, device=target); model.load_state_dict(restored.state_dict(), strict=True); optimizer.state.clear(); run.events.append_event("personal_rl_rejected", {"candidate": str(candidate.relative_to(run.path)), "gate": gate})
            old_league = json.loads((root / state.league_manifest_path).read_text())
            history = list(old_league.get("history", [])); history.append({"generation": state.generation, "candidate_export": str(candidate.relative_to(run.path)), "candidate_checksum": candidate_checksum, "arena_report": str(arena_path.relative_to(run.path)), "comparison_report": str(comparison_path.relative_to(run.path)), "strength_passed": bool(gate["strength_passed"]), "style_passed": bool(gate["style_passed"]), "promoted": bool(gate["passed"])})
            league = create_league(league_dir / f"league-g{state.generation:04d}-s{state.global_step:012d}.json", incumbent=state.active_incumbent_generation, export_path=state.active_incumbent_export, export_checksum=state.active_incumbent_checksum, history=history, stage=curriculum.state.stage, tags=["promoted" if gate["passed"] else "rejected"])
            state.league_manifest_path = str(league.path.relative_to(root)); state.manifest_fingerprints["league"] = str(league.content["fingerprint"]); state.generation += 1; state.phase = "selfplay"; state.generation_baseline_report = None; state.generation_feedback_baseline_report = None; state.training_block_boundary = state.global_step; samplers = None; state.stop_reason = None
            _snapshot(run, model, optimizer, scheduler, samplers, generator, state, "stage", root)
        state.phase = "complete"; _snapshot(run, model, optimizer, scheduler, samplers, generator, state, "final", root); run.events.append_event("run_completed", {"step": state.global_step}); return run.path
