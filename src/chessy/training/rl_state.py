from __future__ import annotations
from dataclasses import asdict, dataclass, field
@dataclass
class RLState:
    generation:int=0
    phase:str="initialize"
    global_step:int=0
    samples_seen:int=0
    completed_game_indexes:list[int]=field(default_factory=list)
    training_block_end_step:int=0
    replay_manifest_path:str|None=None
    replay_manifest_fingerprint:str|None=None
    league_manifest_path:str|None=None
    league_manifest_fingerprint:str|None=None
    curriculum_state:dict[str,object]=field(default_factory=dict)
    best_loss:float=float("inf")
    def state_dict(self)->dict[str,object]: return {"format":"chessy-rl-state-v1",**asdict(self)}
    @classmethod
    def from_dict(cls,data:dict[str,object])->"RLState":
        if data.get("format")!="chessy-rl-state-v1":raise ValueError("invalid RL state")
        fields=set(cls.__dataclass_fields__)
        values={key:value for key,value in data.items() if key!="format"}
        if set(values)-fields: raise ValueError("unknown RL state fields")
        return cls(**values)
