from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, math, time
import chess
import numpy as np
from chessy.chess import ChessEnvironment
from chessy.config.canonical import canonical_json, fingerprint

@dataclass(frozen=True)
class ArenaReport:
    format: str
    candidate: str
    opponent: str
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    confidence_interval: tuple[float,float]
    eligible_for_promotion: bool
    promoted: bool
    termination: dict[str,int]
    game_ids: list[str]
    fingerprint: str

def _interval(values:list[float], seed:int)->tuple[float,float]:
    if not values:return (0.,0.)
    rng=np.random.default_rng(seed); means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))
def paired_schedule(positions:list, games:int)->list[tuple[object,bool]]:
    if games<=0 or games%2: raise ValueError("arena games must be a positive even number")
    if not positions: raise ValueError("arena requires starting positions")
    return [(positions[(index//2)%len(positions)],index%2==0) for index in range(games)]
def run_arena(*, candidate, opponent, positions:list, games:int, max_plies:int, candidate_checksum:str="candidate", opponent_checksum:str="opponent", config_fingerprint:str="", promotion_min_games:int=40, promotion_min_score:float=.55, confidence_threshold:float=.5, seed:int=0) -> ArenaReport:
    results=[]; term:dict[str,int]={}; ids=[]
    # A paired schedule means each listed start is played with reversed colours.
    schedule=paired_schedule(positions,games)
    for index,(start,candidate_white) in enumerate(schedule):
        env=ChessEnvironment.from_fen(start.fen if hasattr(start,"fen") else str(start)); plies=0
        while not env.is_terminal() and plies<max_plies:
            agent=candidate if (env.board.turn==chess.WHITE)==candidate_white else opponent
            env.push(agent.select(env)); plies+=1
        outcome=env.outcome(); reason="max-plies" if outcome is None else outcome.termination.name.lower(); term[reason]=term.get(reason,0)+1
        value=.5 if outcome is None or outcome.winner is None else (1.0 if (outcome.winner==chess.WHITE)==candidate_white else 0.0); results.append(value); ids.append(f"arena-{index}-{hashlib.sha256(f'{seed}|{index}|{env.fen()}'.encode()).hexdigest()[:12]}")
    wins=results.count(1.0); draws=results.count(.5); losses=results.count(0.0); score=sum(results)/games; ci=_interval(results,seed); eligible=games>=promotion_min_games; promoted=eligible and score>=promotion_min_score and ci[0]>confidence_threshold
    body={"format":"chessy-arena-report-v1","candidate":candidate_checksum,"opponent":opponent_checksum,"games":games,"wins":wins,"draws":draws,"losses":losses,"score":score,"confidence_interval":ci,"eligible_for_promotion":eligible,"promoted":promoted,"termination":term,"game_ids":ids,"config_fingerprint":config_fingerprint}
    return ArenaReport(**{key:body[key] for key in ArenaReport.__dataclass_fields__ if key!="fingerprint"},fingerprint=fingerprint(body))
def write_report(path, report:ArenaReport)->None:
    path.parent.mkdir(parents=True,exist_ok=True); payload=canonical_json(asdict(report))
    if path.exists():
        if path.is_symlink() or path.read_bytes()!=payload: raise FileExistsError("arena report already exists with different content")
    else: path.write_bytes(payload)
