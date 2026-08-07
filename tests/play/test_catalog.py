from pathlib import Path

from chessy.model import ChessyModel, export_model
from chessy.model.config import ModelConfig
from chessy.play.catalog import discover_model_exports, model_info_from_export


def _tiny_model() -> ChessyModel:
    return ChessyModel(ModelConfig(channels=8, residual_blocks=1, value_channels=8, value_hidden=16))


def test_catalog_discovers_playable_run_exports_and_skips_invalid_entries(tmp_path: Path) -> None:
    exports = tmp_path / "runs" / "run-1" / "exports"; exports.mkdir(parents=True)
    candidate = export_model(_tiny_model(), exports / "candidate-0000-step-000000000250", metadata={"name": "local", "generation": "0", "step": "250"})
    broken = exports / "candidate-broken"; broken.mkdir(); (broken / "manifest.json").write_text("{}", encoding="utf-8")
    found = discover_model_exports(tmp_path / "runs")
    assert len(found) == 1
    info, path = found[0]
    assert path == candidate
    assert info.name == "local · step 250"
    assert info.checksum == model_info_from_export(candidate).checksum


def test_catalog_ignores_missing_or_symlinked_roots(tmp_path: Path) -> None:
    assert discover_model_exports(tmp_path / "missing") == []
    target = tmp_path / "target"; target.mkdir(); link = tmp_path / "runs"; link.symlink_to(target, target_is_directory=True)
    assert discover_model_exports(link) == []
