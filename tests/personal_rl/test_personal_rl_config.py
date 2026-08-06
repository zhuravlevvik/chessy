from pathlib import Path
import pytest
from chessy.config import load_config


def test_personal_rl_smoke_config_round_trips() -> None:
    config, _, resolved, _ = load_config(Path("configs/personal-rl-smoke.yaml"))
    assert config.personal_rl is not None
    assert b"personal_rl" in resolved


@pytest.mark.parametrize("replacement", ["style_strength: 0.0", "style_strength: .nan"])
def test_style_strength_must_be_positive_and_finite(tmp_path: Path, replacement: str) -> None:
    text = Path("configs/personal-rl-smoke.yaml").read_text().replace("style_strength: 0.20", replacement)
    path = tmp_path / "invalid.yaml"; path.write_text(text)
    with pytest.raises(ValueError): load_config(path)


@pytest.mark.parametrize(
    "old,new",
    [
        ("rl_policy_weight: 1.0", "rl_policy_weight: 0.0"),
        ("style_policy_weight: 1.0\n  style_value_weight: 0.25", "style_policy_weight: 0.0\n  style_value_weight: 0.0"),
        ("feedback_strength: 0.0", "feedback_strength: 0.2"),
        ("historical_batch_size: 2", "historical_batch_size: 0"),
    ],
)
def test_personal_rl_rejects_conflicting_or_empty_objectives(tmp_path: Path, old: str, new: str) -> None:
    text = Path("configs/personal-rl-smoke.yaml").read_text().replace(old, new)
    path = tmp_path / "invalid.yaml"; path.write_text(text)
    with pytest.raises(ValueError): load_config(path)
