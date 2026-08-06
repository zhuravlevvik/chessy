from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from chessy.model import ChessyModel, export_model, load_model_export


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_manifest(export_dir: Path, manifest: dict[str, object]) -> None:
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (export_dir / "checksums.sha256").write_text(
        f"{_sha256(manifest_path)}  manifest.json\n"
        f"{_sha256(export_dir / 'model.safetensors')}  model.safetensors\n",
        encoding="utf-8",
    )


def test_export_round_trip_is_exact_and_does_not_mutate_source(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = ChessyModel()
    model.train()
    source_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    boards = torch.randn((2, 119, 8, 8))
    destination = export_model(model, tmp_path / "model")
    assert {path.name for path in destination.iterdir()} == {
        "model.safetensors",
        "manifest.json",
        "checksums.sha256",
    }
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["format"] == "chessy-model-v1"
    assert manifest["encodings"] == {"action": "az73-v1", "board": "board119-v1"}
    assert manifest["model_config"] == model.config.to_dict()
    assert manifest["weights"]["sha256"] == _sha256(destination / "model.safetensors")
    loaded = load_model_export(destination, device="cpu")
    assert not loaded.training
    for name, value in source_state.items():
        assert torch.equal(value, loaded.state_dict()[name])
        assert torch.equal(value, model.state_dict()[name])
    assert model.training
    with torch.inference_mode():
        original = model.eval()(boards)
        restored = loaded(boards)
    torch.testing.assert_close(original.policy_logits, restored.policy_logits, rtol=0, atol=0)
    torch.testing.assert_close(original.value_logits, restored.value_logits, rtol=0, atol=0)


def test_export_detects_corruption_and_incompatible_manifest(tmp_path: Path) -> None:
    destination = export_model(ChessyModel(), tmp_path / "model")
    weights = destination / "model.safetensors"
    contents = bytearray(weights.read_bytes())
    contents[-1] ^= 1
    weights.write_bytes(contents)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model_export(destination, device="cpu")

    destination = export_model(ChessyModel(), tmp_path / "other")
    manifest = json.loads((destination / "manifest.json").read_text())
    manifest["encodings"]["board"] = "wrong"
    _rewrite_manifest(destination, manifest)
    with pytest.raises(ValueError, match="board encoding"):
        load_model_export(destination, device="cpu")


def test_export_is_strict_and_never_overwrites(tmp_path: Path) -> None:
    destination = export_model(ChessyModel(), tmp_path / "model")
    with pytest.raises(FileExistsError):
        export_model(ChessyModel(), destination)
    state_dict = load_file(str(destination / "model.safetensors"))
    state_dict["unexpected"] = torch.zeros(1)
    save_file(state_dict, str(destination / "model.safetensors"))
    manifest = json.loads((destination / "manifest.json").read_text())
    manifest["weights"]["sha256"] = _sha256(destination / "model.safetensors")
    manifest["weights"]["parameter_count"] += 1
    _rewrite_manifest(destination, manifest)
    with pytest.raises(ValueError, match="strictly match"):
        load_model_export(destination, device="cpu")


def test_export_cleans_up_temporary_directory_after_a_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_module = importlib.import_module("chessy.model.export")

    def fail_save(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(export_module, "save_file", fail_save)
    with pytest.raises(OSError, match="simulated"):
        export_model(ChessyModel(), tmp_path / "model")
    assert not list(tmp_path.iterdir())
