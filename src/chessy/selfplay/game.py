from __future__ import annotations
from dataclasses import dataclass
import hashlib, time
from typing import Any, Callable
import chess, chess.pgn
import numpy as np
from chessy.chess import ChessEnvironment
from chessy.encoding import decode_action, encode_board
from chessy.mcts import MCTS, sample_action
from chessy.replay.segment import ReplaySample, SealedGame
from chessy.selfplay.temperature import TemperatureSchedule

def derive_seed(run_seed:int,generation:int,actor_id:int,game_index:int)->int:
    raw=f"{run_seed}|{generation}|{actor_id}|{game_index}".encode(); return int.from_bytes(hashlib.sha256(raw).digest()[:8],"big")
def derive_game_id(run_id:str,generation:int,game_index:int)->str:
    return f"game-{generation}-{game_index}-{hashlib.sha256(f'{run_id}|{generation}|{game_index}'.encode()).hexdigest()[:16]}"

@dataclass(frozen=True)
class SelfPlayGame:
    sealed: SealedGame
    duration_seconds: float

def _wdl(outcome: chess.Outcome, turn: chess.Color) -> int:
    if outcome.winner is None: return 1
    return 2 if outcome.winner == turn else 0

def play_game(*, run_id:str, run_seed:int, generation:int, game_index:int, actor_id:int, start, mcts:MCTS, schedule:TemperatureSchedule, model_checksum:str, stop_requested=None, observer_update:Callable[[dict[str,Any]],None]|None=None) -> SelfPlayGame | None:
    """Play one complete game. Interrupted games are intentionally not replay."""
    seed=derive_seed(run_seed,generation,actor_id,game_index); rng=np.random.default_rng(seed); environment=ChessEnvironment.from_fen(start.fen); began=time.monotonic(); pending=[]; moves=[]; move_records=[]
    if observer_update: observer_update({"status":"playing","generation":generation,"game_index":game_index,"model_checksum":model_checksum,"initial_fen":start.fen,"fen":start.fen,"result":"*","termination":None,"frames":[{"ply":0,"fen":start.fen,"uci":None,"san":None}]})
    while not environment.is_terminal() and len(pending) < start.max_plies:
        if stop_requested is not None and stop_requested.is_set(): return None
        board=environment.board; result=mcts.search(environment)
        # Store every legal root action so the trainer can reconstruct the
        # complete legal mask. Zero-visit actions simply receive zero target mass.
        actions=tuple(sorted(result.policy)); visits=tuple(result.policy[a].visits for a in actions)
        action=sample_action(result,temperature=schedule.for_ply(len(pending)),rng=rng)
        move=decode_action(board,action); san=board.san(move)
        pending.append((encode_board(environment.history()),actions,visits,action,board.turn))
        moves.append(move); environment.push(move); mcts.advance(environment,action); move_records.append({"ply":len(moves),"fen":environment.fen(),"uci":move.uci(),"san":san})
        if observer_update: observer_update({"status":"playing","generation":generation,"game_index":game_index,"model_checksum":model_checksum,"initial_fen":start.fen,"fen":environment.fen(),"result":"*","termination":None,"frames":[{"ply":0,"fen":start.fen,"uci":None,"san":None},*move_records]})
    outcome=environment.outcome(); termination="max-plies" if outcome is None else outcome.termination.name.lower().replace("_","-")
    result_text="1/2-1/2" if outcome is None or outcome.winner is None else ("1-0" if outcome.winner else "0-1")
    samples=tuple(ReplaySample(board=board,policy_actions=actions,policy_visits=visits,selected_action=action,value_class=(1 if outcome is None else _wdl(outcome,turn)),game_index=game_index,ply=ply,generation=generation) for ply,(board,actions,visits,action,turn) in enumerate(pending))
    game=chess.pgn.Game(); game.headers["Event"]="Chessy self-play"; game.headers["SetUp"]="1"; game.headers["FEN"]=start.fen; game.headers["Result"]=result_text; game.headers["Termination"]=termination
    node=game
    for move in moves: node=node.add_variation(move)
    sealed=SealedGame(game_id=derive_game_id(run_id,generation,game_index),game_index=game_index,generation=generation,stage=start.stage,source_kind=start.source_kind,initial_fen=start.fen,result=result_text,termination=termination,samples=samples,pgn=str(game),metadata={"actor_seed":seed,"start_metadata":start.metadata,"duration_seconds":time.monotonic()-began,"mcts":{"simulations":mcts.config.simulations,"c_puct":mcts.config.c_puct,"root_noise":mcts.config.root_noise}},complete=True)
    return SelfPlayGame(sealed,time.monotonic()-began)
