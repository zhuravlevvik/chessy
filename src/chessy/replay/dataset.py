from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from chessy.encoding import ACTION_SIZE
from chessy.replay.codec import decode_board
from chessy.replay.manifest import ReplayManifest, load_manifest
from chessy.replay.segment import verify_segment

class ReplayDataset:
    """Lazy immutable replay view; indices refer only to its verified manifest."""
    def __init__(self, manifest: ReplayManifest | Path, *, cache_segments: int = 2, active_max_samples: int | None = None) -> None:
        if cache_segments <= 0: raise ValueError("cache_segments must be positive")
        self.manifest = load_manifest(manifest) if isinstance(manifest, Path) else load_manifest(manifest.path)
        self.root = self.manifest.path.parent.parent; self.cache_segments = cache_segments; self._cache: OrderedDict[Path, dict[str,np.ndarray]] = OrderedDict()
        entries = list(self.manifest.content["segments"]); locations=[]
        for entry in entries:
            segment = self.root / entry["path"]; count=int(entry["sample_count"])
            locations.extend((segment, offset, int(entry["generation"])) for offset in range(count))
        limit=active_max_samples or int(self.manifest.content["active_window"]["max_samples"])
        self.locations = locations[-limit:]
        if not self.locations: raise ValueError("replay manifest has no active samples")
    def __len__(self) -> int: return len(self.locations)
    def generation_of(self, index: int) -> int: return self.locations[index][2]
    def _segment(self, path: Path) -> dict[str,np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path); return self._cache[path]
        checked=verify_segment(path); arrays=checked["arrays"]
        self._cache[path]=arrays
        while len(self._cache)>self.cache_segments: self._cache.popitem(last=False)
        return arrays
    def __getitem__(self, index: int) -> dict[str, object]:
        if not 0 <= index < len(self): raise IndexError(index)
        path, offset, generation = self.locations[index]; data=self._segment(path); start,end=(int(data["policy_offsets"][offset]),int(data["policy_offsets"][offset+1]))
        actions=data["policy_actions"][start:end].astype(np.int64); visits=data["policy_visits"][start:end].astype(np.float32)
        policy=np.zeros(ACTION_SIZE,dtype=np.float32); policy[actions]=visits/visits.sum(); legal=np.zeros(ACTION_SIZE,dtype=np.bool_); legal[actions]=True
        return {"board":torch.from_numpy(decode_board(data["boards"][offset])), "policy":torch.from_numpy(policy), "legal_mask":torch.from_numpy(legal), "value_class":int(data["value_class"][offset]), "generation":generation, "game_index":int(data["game_index"][offset]), "ply":int(data["ply"][offset])}
    def batch(self, indices: torch.Tensor | list[int]) -> dict[str, object]:
        records=[self[int(index)] for index in indices]
        return {"boards":torch.stack([r["board"] for r in records]), "policy":torch.stack([r["policy"] for r in records]), "legal_mask":torch.stack([r["legal_mask"] for r in records]), "value_class":torch.tensor([r["value_class"] for r in records],dtype=torch.long), "metadata":[{key:r[key] for key in ("generation","game_index","ply")} for r in records]}
