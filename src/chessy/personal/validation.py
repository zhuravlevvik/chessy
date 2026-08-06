"""Full deterministic validation reports and compact slice metrics."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from chessy.config.canonical import canonical_json, fingerprint
from chessy.personal.dataset import PersonalDataset
from chessy.personal.segment import ENUMS
from chessy.training.supervised_loss import supervised_policy_value_loss


def _summary(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0}
    return {"count": len(rows), "policy_cross_entropy": sum(row["ce"] for row in rows) / len(rows), "top1": sum(row["top1"] for row in rows) / len(rows), "top3": sum(row["top3"] for row in rows) / len(rows), "top5": sum(row["top5"] for row in rows) / len(rows), "mean_true_move_probability": sum(row["prob"] for row in rows) / len(rows), "median_true_move_probability": statistics.median(row["prob"] for row in rows), "value_cross_entropy": sum(row["value_ce"] for row in rows) / len(rows), "value_accuracy": sum(row["value_acc"] for row in rows) / len(rows)}


@torch.no_grad()
def validate(model: torch.nn.Module, dataset: PersonalDataset, *, device: torch.device, batch_size: int, model_checksum: str = "", snapshot_step: int = 0, config_fingerprint: str = "", baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    if dataset.split != "val":
        raise ValueError("ordinary personal validation only accepts val split")
    started = time.monotonic()
    model.eval(); all_rows: list[dict[str, float]] = []; slices: dict[str, dict[str, list[dict[str, float]]]] = {key: defaultdict(list) for key in ("color", "sample_kind", "source", "phase")}
    for start in range(0, len(dataset), batch_size):
        batch = dataset.batch(list(range(start, min(start + batch_size, len(dataset)))))
        outputs = model(batch["boards"].to(device))
        _, metrics = supervised_policy_value_loss(outputs.policy_logits, outputs.value_logits, batch["target_action"].to(device), batch["legal_mask"].to(device), batch["value_class"].to(device))
        masked = outputs.policy_logits.masked_fill(~batch["legal_mask"].to(device), float("-inf"))
        target = batch["target_action"].to(device)
        for offset, metadata in enumerate(batch["metadata"]):
            values = {"ce": float(metrics["policy_per_sample"][offset].cpu()), "prob": float(metrics["true_move_probability"][offset].cpu()), "top1": float(metrics["top1"][offset].cpu()), "top3": float((masked[offset].topk(3).indices == target[offset]).any().cpu()), "top5": float((masked[offset].topk(5).indices == target[offset]).any().cpu()), "value_ce": float(metrics["value_per_sample"][offset].cpu()), "value_acc": float(metrics["value_accuracy"][offset].cpu())}
            all_rows.append(values)
            for key in slices:
                inverse = {value: name for name, value in ENUMS[key].items()}
                slices[key][inverse[metadata[key]]].append(values)
    overall = _summary(all_rows)
    elapsed = time.monotonic() - started
    report: dict[str, Any] = {"format": "chessy-personal-validation-v1", "model_checksum": model_checksum, "snapshot_step": snapshot_step, "dataset_manifest_fingerprint": dataset.fingerprint, "split": "val", "split_fingerprint": fingerprint(dataset.manifest["splits"]["val"]), "metrics": overall, "slices": {key: {name: _summary(rows) for name, rows in values.items()} for key, values in slices.items()}, "elapsed_seconds": elapsed, "samples_per_second": len(all_rows) / max(elapsed, 1e-12), "config_fingerprint": config_fingerprint, "selection_metric": "policy_cross_entropy", "baseline_delta": None if baseline is None else overall["policy_cross_entropy"] - baseline["metrics"]["policy_cross_entropy"], "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    report["content_fingerprint"] = fingerprint({key: value for key, value in report.items() if key not in {"created_at", "elapsed_seconds", "samples_per_second", "content_fingerprint"}})
    return report


def write_validation_report(path: Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("content_fingerprint") != report.get("content_fingerprint"):
            raise FileExistsError("validation report destination already has different content")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(canonical_json(report))
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
