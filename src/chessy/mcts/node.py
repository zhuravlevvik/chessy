"""Search-tree node and exact PUCT helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess


@dataclass(slots=True)
class SearchNode:
    prior: float
    to_play: chess.Color
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    expanded: bool = False
    noise_applied: bool = False
    fingerprint: tuple[str, ...] | None = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def puct_score(parent: SearchNode, child: SearchNode, c_puct: float) -> float:
    exploration = c_puct * child.prior * math.sqrt(max(1, parent.visit_count)) / (1 + child.visit_count)
    return -child.mean_value + exploration


def select_child(parent: SearchNode, c_puct: float) -> tuple[int, SearchNode]:
    if not parent.children:
        raise ValueError("cannot select a child from an unexpanded node")
    # max() preserves the desired lower-action tie break through the second key.
    return max(parent.children.items(), key=lambda item: (puct_score(parent, item[1], c_puct), -item[0]))


def backup(path: list[SearchNode], value: float) -> None:
    """Back up a leaf value, changing perspective at every ply."""
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        value = -value
