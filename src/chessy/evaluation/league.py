from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from chessy.config.canonical import canonical_json, fingerprint
@dataclass(frozen=True)
class LeagueManifest:
    path:Path
    content:dict[str,object]
def create_league(path:Path,*,incumbent:int,export_path:str,export_checksum:str,history:list[dict[str,object]]|None=None,stage:str="endgames",tags:list[str]|None=None)->LeagueManifest:
    body={"format":"chessy-league-v1","incumbent_generation":incumbent,"incumbent":{"export_path":export_path,"checksum":export_checksum},"history":history or [],"curriculum_stage":stage,"tags":tags or ["initial"]}; body["fingerprint"]=fingerprint(body); path.parent.mkdir(parents=True,exist_ok=True)
    payload=canonical_json(body)
    if path.exists():
        if path.is_symlink() or path.read_bytes()!=payload: raise FileExistsError("league manifest already exists with different content")
    else:
        temporary=path.with_name(f".{path.name}.tmp-{__import__('os').getpid()}"); temporary.write_bytes(payload); __import__('os').replace(temporary,path)
    return LeagueManifest(path,body)
