from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class TemperatureSchedule:
    initial: float
    cutoff_ply: int
    final: float
    def __post_init__(self)->None:
        if self.initial < 0 or self.final < 0 or self.cutoff_ply < 0: raise ValueError("invalid temperature schedule")
    def for_ply(self, ply:int)->float: return self.initial if ply < self.cutoff_ply else self.final
