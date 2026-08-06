"""Read-only lazy loader for encoded human-feedback samples."""
from __future__ import annotations
from collections import OrderedDict, defaultdict
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from chessy.encoding import ACTION_SIZE
from chessy.feedback.segment import verify_feedback_manifest, verify_feedback_segment
from chessy.replay.codec import decode_board


class FeedbackDataset:
    def __init__(self, manifest: Path, *, cache_segments: int = 1) -> None:
        if cache_segments <= 0: raise ValueError("cache_segments must be positive")
        self.path = Path(manifest); self.manifest = verify_feedback_manifest(self.path); self.root = self.path.parent.parent; self.cache_segments = cache_segments; self._cache: OrderedDict[Path, dict[str, Any]] = OrderedDict(); self.locations: list[tuple[Path, int]] = []; self.indices_by_game: dict[int, list[int]] = defaultdict(list)
        for entry in self.manifest["segments"]:
            path = self.root / entry["path"]
            with np.load(path / "samples.npz", allow_pickle=False) as raw: games = raw["game_index"].copy()
            for offset, game in enumerate(games): self.indices_by_game[int(game)].append(len(self.locations)); self.locations.append((path, offset))
        if not self.locations: raise ValueError("feedback dataset has no samples")
    @property
    def fingerprint(self) -> str: return str(self.manifest["content_fingerprint"])
    def __len__(self) -> int: return len(self.locations)
    def _segment(self, path: Path) -> dict[str, Any]:
        if path in self._cache: self._cache.move_to_end(path); return self._cache[path]
        value = verify_feedback_segment(path); self._cache[path] = value
        while len(self._cache) > self.cache_segments: self._cache.popitem(last=False)
        return value
    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self): raise IndexError(index)
        path, offset = self.locations[index]; checked = self._segment(path); arrays = checked["arrays"]; start, end = int(arrays["legal_offsets"][offset]), int(arrays["legal_offsets"][offset + 1]); mask = np.zeros(ACTION_SIZE, dtype=np.bool_); mask[arrays["legal_actions"][start:end].astype(np.int64, copy=False)] = True; meta = json.loads(checked["metadata"][offset])
        return {"board": torch.from_numpy(decode_board(arrays["boards"][offset]).copy()), "legal_mask": torch.from_numpy(mask), "target_action": int(arrays["target_action"][offset]), "value_class": int(arrays["value_class"][offset]), "sample_weight": float(arrays["sample_weight"][offset]), "stream": "human_online", "game_index": int(arrays["game_index"][offset]), "ply": int(arrays["ply"][offset]), "color": int(arrays["color"][offset]), "phase": int(arrays["phase"][offset]), "metadata": meta}
    def batch(self, indices: list[int]) -> dict[str, Any]:
        rows = [self[index] for index in indices]
        return {"boards": torch.stack([row["board"] for row in rows]), "legal_mask": torch.stack([row["legal_mask"] for row in rows]), "target_action": torch.tensor([row["target_action"] for row in rows], dtype=torch.long), "value_class": torch.tensor([row["value_class"] for row in rows], dtype=torch.long), "sample_weight": torch.tensor([row["sample_weight"] for row in rows], dtype=torch.float32), "stream": "human_online", "metadata": [row["metadata"] | {"game_index": row["game_index"], "ply": row["ply"], "color": row["color"], "phase": row["phase"]} for row in rows]}
