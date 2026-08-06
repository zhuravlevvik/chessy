"""Pure promotion gates, kept independent from arena/execution code."""
from __future__ import annotations
from typing import Any


def style_gate(*, baseline: dict[str, Any], candidate: dict[str, Any], historical_ce_tolerance: float, minimum_top1_ratio: float, feedback_baseline: dict[str, Any] | None = None, feedback_candidate: dict[str, Any] | None = None, feedback_ce_tolerance: float = 0.0) -> dict[str, object]:
    base = baseline["metrics"]; current = candidate["metrics"]
    historical_ce_delta = float(current["policy_cross_entropy"]) - float(base["policy_cross_entropy"])
    top1_floor = float(base["top1"]) * minimum_top1_ratio
    checks: dict[str, bool] = {"historical_ce": historical_ce_delta <= historical_ce_tolerance, "historical_top1": float(current["top1"]) >= top1_floor}
    feedback_delta: float | None = None
    if (feedback_baseline is None) != (feedback_candidate is None): raise ValueError("feedback reports must be supplied together")
    if feedback_baseline is not None and feedback_candidate is not None:
        feedback_delta = float(feedback_candidate["metrics"]["policy_cross_entropy"]) - float(feedback_baseline["metrics"]["policy_cross_entropy"])
        checks["feedback_ce"] = feedback_delta <= feedback_ce_tolerance
    return {"passed": all(checks.values()), "checks": checks, "historical_ce_delta": historical_ce_delta, "historical_top1_floor": top1_floor, "feedback_ce_delta": feedback_delta}


def promotion_gate(*, arena: Any, style: dict[str, object]) -> dict[str, object]:
    strength = bool(arena.promoted)
    return {"passed": strength and bool(style["passed"]), "strength_passed": strength, "style_passed": bool(style["passed"]), "arena_eligible": bool(arena.eligible_for_promotion), "arena_score": float(arena.score), "arena_confidence_interval": tuple(arena.confidence_interval)}
