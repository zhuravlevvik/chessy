"""Lazy personal dataset loader with an explicit test-split firewall."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chessy.encoding import ACTION_SIZE
from chessy.personal.segment import load_personal_manifest, verify_personal_manifest, verify_segment
from chessy.replay.codec import decode_board


class PersonalDataset:
    """Read only the split explicitly allowed by the caller.

    ``test`` is intentionally unavailable to ordinary training/validation APIs;
    final evaluation must pass an explicit acknowledgement capability.
    """
    def __init__(self, manifest: Path, *, split: str, cache_segments: int = 2, acknowledge_test: bool = False) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("personal split must be train, val, or test")
        if split == "test" and not acknowledge_test:
            raise PermissionError("test split requires explicit final-test acknowledgement")
        if cache_segments <= 0:
            raise ValueError("cache_segments must be positive")
        self.path = Path(manifest)
        self.manifest = verify_personal_manifest(self.path)
        self.split, self.cache_segments = split, cache_segments
        self.root = self.path.parent.parent
        self._cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        self.locations: list[tuple[Path, int]] = []
        self.indices_by_game: dict[int, list[int]] = defaultdict(list)
        self.indices_by_kind: dict[int, list[int]] = defaultdict(list)
        for entry in self.manifest["splits"][split]["segments"]:
            segment = self.root / entry["path"]
            with np.load(segment / "samples.npz", allow_pickle=False) as raw:
                games = raw["game_index"].copy()
                kinds = raw["sample_kind"].copy()
            for offset, (game, kind) in enumerate(zip(games, kinds, strict=True)):
                index = len(self.locations)
                self.locations.append((segment, offset))
                self.indices_by_game[game].append(index)
                self.indices_by_kind[kind].append(index)
        if not self.locations:
            raise ValueError("personal split has no samples")

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["content_fingerprint"])

    def __len__(self) -> int:
        return len(self.locations)

    def _segment(self, path: Path) -> dict[str, Any]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        checked = verify_segment(path)
        self._cache[path] = checked
        while len(self._cache) > self.cache_segments:
            self._cache.popitem(last=False)
        return checked

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        path, offset = self.locations[index]
        checked = self._segment(path)
        arrays = checked["arrays"]
        start, end = int(arrays["legal_offsets"][offset]), int(arrays["legal_offsets"][offset + 1])
        actions = arrays["legal_actions"][start:end].astype(np.int64, copy=False)
        legal = np.zeros(ACTION_SIZE, dtype=np.bool_); legal[actions] = True
        return {
            "board": torch.from_numpy(decode_board(arrays["boards"][offset]).copy()),
            "legal_mask": torch.from_numpy(legal), "target_action": int(arrays["target_action"][offset]),
            "value_class": int(arrays["value_class"][offset]), "game_index": int(arrays["game_index"][offset]),
            "ply": int(arrays["ply"][offset]), "sample_kind": int(arrays["sample_kind"][offset]),
            "source": int(arrays["source"][offset]), "color": int(arrays["color"][offset]), "phase": int(arrays["phase"][offset]),
        }

    def batch(self, indices: list[int] | torch.Tensor) -> dict[str, Any]:
        rows = [self[int(index)] for index in indices]
        keys = ("game_index", "ply", "sample_kind", "source", "color", "phase")
        return {
            "boards": torch.stack([row["board"] for row in rows]),
            "legal_mask": torch.stack([row["legal_mask"] for row in rows]),
            "target_action": torch.tensor([row["target_action"] for row in rows], dtype=torch.long),
            "value_class": torch.tensor([row["value_class"] for row in rows], dtype=torch.long),
            "metadata": [{key: row[key] for key in keys} for row in rows],
        }
