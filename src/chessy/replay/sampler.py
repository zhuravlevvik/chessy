from __future__ import annotations
from dataclasses import dataclass
import torch
from chessy.replay.dataset import ReplayDataset

@dataclass
class ReplaySampler:
    dataset: ReplayDataset
    batch_size: int
    seed: int
    recent_fraction: float = 0.0
    recent_generations: int = 1
    def __post_init__(self) -> None:
        if self.batch_size <= 0 or not 0 <= self.recent_fraction <= 1 or self.recent_generations <= 0: raise ValueError("invalid replay sampler configuration")
        self.generator=torch.Generator(device="cpu"); self.generator.manual_seed(self.seed); self.draws=0
        all_generations=[self.dataset.generation_of(i) for i in range(len(self.dataset))]; newest=sorted(set(all_generations))[-self.recent_generations:]
        self.recent=torch.tensor([i for i,g in enumerate(all_generations) if g in newest],dtype=torch.long)
        self.all=torch.arange(len(self.dataset),dtype=torch.long)
    def next_batch(self) -> torch.Tensor:
        count_recent=int(round(self.batch_size*self.recent_fraction)); count_all=self.batch_size-count_recent
        def draw(pool:torch.Tensor, count:int)->torch.Tensor: return pool[torch.randint(len(pool),(count,),generator=self.generator)] if count else torch.empty(0,dtype=torch.long)
        values=torch.cat((draw(self.recent,count_recent),draw(self.all,count_all))); self.draws += 1
        return values[torch.randperm(len(values),generator=self.generator)]
    def state_dict(self) -> dict[str, object]:
        return {"format":"chessy-replay-sampler-v1", "batch_size":self.batch_size,"seed":self.seed,"recent_fraction":self.recent_fraction,"recent_generations":self.recent_generations,"dataset_fingerprint":self.dataset.manifest.fingerprint,"draws":self.draws,"generator_state":self.generator.get_state()}
    def load_state_dict(self,state:dict[str,object])->None:
        expected={"format","batch_size","seed","recent_fraction","recent_generations","dataset_fingerprint","draws","generator_state"}
        if set(state)!=expected or state["format"]!="chessy-replay-sampler-v1": raise ValueError("invalid replay sampler state")
        for key in ("batch_size","seed","recent_fraction","recent_generations"):
            if state[key] != getattr(self,key): raise ValueError(f"replay sampler incompatible: {key}")
        if state["dataset_fingerprint"] != self.dataset.manifest.fingerprint or not isinstance(state["generator_state"],torch.Tensor): raise ValueError("replay sampler manifest mismatch")
        self.draws=int(state["draws"]); self.generator.set_state(state["generator_state"])
