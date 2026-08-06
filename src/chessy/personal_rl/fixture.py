"""Ignored tiny inputs for the honest personal-RL plumbing smoke."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import torch

from chessy.model import ChessyModel, ModelConfig
from chessy.model.export import export_model
from chessy.personal.fixture import prepare_smoke_fixture


def prepare_personal_rl_smoke_fixture(root: Path) -> dict[str, str]:
    root = Path(root).resolve(); source = prepare_smoke_fixture(root)
    fixture = root / "runs" / "personal-rl-fixture"; encoded = fixture / "encoded"
    if not encoded.exists(): shutil.copytree((root / source["dataset_manifest"]).parent.parent, encoded)
    source_manifest = encoded / "manifests" / "personal-dataset-fixture.json"
    # The smoke uses separate named roles even though the tiny weights are the
    # same deterministic network; it is plumbing, not a strength experiment.
    for name, role in (("base-rl", "base_rl"), ("personal-supervised", "personal_supervised")):
        destination = fixture / name
        if not destination.exists():
            torch.manual_seed(7)
            model = ChessyModel(ModelConfig(channels=8, residual_blocks=1, group_norm_groups=8, value_channels=8, value_hidden=16))
            export_model(model, destination, metadata={"role": role, "name": "personal RL smoke fixture"})
    base = json.loads((fixture / "base-rl" / "manifest.json").read_text())
    return {"dataset_manifest": str(source_manifest.relative_to(root)), "base_rl_export": str((fixture / "base-rl").relative_to(root)), "personal_supervised_export": str((fixture / "personal-supervised").relative_to(root)), "base_rl_checksum": str(base["weights"]["sha256"])}
