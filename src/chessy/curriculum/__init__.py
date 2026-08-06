"""Reproducible starting-position curriculum for local self-play."""
from chessy.curriculum.sources import FullSource, EndgameSource, PositionSource, ReducedSource, StartPosition, source_for_stage
from chessy.curriculum.manager import CurriculumState, CurriculumManager

__all__ = ["CurriculumManager", "CurriculumState", "EndgameSource", "FullSource", "PositionSource", "ReducedSource", "StartPosition", "source_for_stage"]
