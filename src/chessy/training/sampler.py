"""A serializable, deterministic batch sampler independent of DataLoader."""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class StatefulBatchSampler:
    dataset_size: int; batch_size: int; shuffle: bool = True; drop_last: bool = False; seed: int = 0
    def __post_init__(self) -> None:
        if self.dataset_size <= 0 or self.batch_size <= 0 or self.seed < 0: raise ValueError("invalid sampler dimensions or seed")
        self.generator = torch.Generator(device="cpu"); self.generator.manual_seed(self.seed)
        self.epoch = 0; self.cursor = 0; self.permutation = self._make_permutation()
    def _make_permutation(self) -> torch.Tensor:
        return torch.randperm(self.dataset_size, generator=self.generator) if self.shuffle else torch.arange(self.dataset_size)
    def next_batch(self) -> torch.Tensor:
        if self.cursor >= self.dataset_size or (self.drop_last and self.cursor + self.batch_size > self.dataset_size):
            self.epoch += 1; self.cursor = 0; self.permutation = self._make_permutation()
        end = min(self.cursor + self.batch_size, self.dataset_size)
        if self.drop_last and end - self.cursor < self.batch_size:
            self.epoch += 1; self.cursor = 0; self.permutation = self._make_permutation(); end = self.batch_size
        batch = self.permutation[self.cursor:end].clone(); self.cursor = end
        return batch
    def state_dict(self) -> dict[str, object]:
        return {"format":"chessy-sampler-state-v1", "dataset_size":self.dataset_size, "batch_size":self.batch_size, "shuffle":self.shuffle, "drop_last":self.drop_last, "seed":self.seed, "epoch":self.epoch, "cursor":self.cursor, "permutation":self.permutation, "generator_state":self.generator.get_state()}
    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = {"format","dataset_size","batch_size","shuffle","drop_last","seed","epoch","cursor","permutation","generator_state"}
        if set(state) != expected or state["format"] != "chessy-sampler-state-v1": raise ValueError("invalid sampler state format")
        for key in ("dataset_size","batch_size","shuffle","drop_last","seed"):
            if state[key] != getattr(self, key): raise ValueError(f"sampler incompatible: {key}")
        permutation, generator_state = state["permutation"], state["generator_state"]
        if not isinstance(permutation, torch.Tensor) or permutation.dtype != torch.int64 or permutation.numel() != self.dataset_size: raise ValueError("invalid sampler permutation")
        if not isinstance(generator_state, torch.Tensor): raise ValueError("invalid sampler generator state")
        epoch, cursor = state["epoch"], state["cursor"]
        if isinstance(epoch,bool) or not isinstance(epoch,int) or isinstance(cursor,bool) or not isinstance(cursor,int) or epoch < 0 or not 0 <= cursor <= self.dataset_size: raise ValueError("invalid sampler cursor")
        self.epoch, self.cursor, self.permutation = epoch, cursor, permutation.clone(); self.generator.set_state(generator_state)
