from __future__ import annotations

import io
from types import SimpleNamespace

import chess
import chess.pgn
import numpy as np

from chessy.curriculum.sources import EndgameSource, FullSource, ReducedSource, StartPosition
from chessy.encoding import encode_move
from chessy.mcts import SearchAction, SearchResult
from chessy.selfplay.game import play_game
from chessy.selfplay.temperature import TemperatureSchedule


def test_curriculum_sources_are_valid_and_seeded() -> None:
    for source in (EndgameSource(max_plies=8), ReducedSource(max_plies=8), FullSource(max_plies=8)):
        first=np.random.default_rng(123); second=np.random.default_rng(123)
        left=[source.sample(first).fen for _ in range(30)]
        right=[source.sample(second).fen for _ in range(30)]
        assert left==right
        for fen in left:
            board=chess.Board(fen)
            assert board.is_valid() and board.outcome(claim_draw=True) is None


def test_selfplay_executes_the_temperature_selected_action() -> None:
    board=chess.Board(); moves=sorted(board.legal_moves,key=lambda move:move.uci())[:2]
    actions=[encode_move(board,move) for move in moves]
    result=SearchResult(action=actions[0],move=moves[0],policy={actions[0]:SearchAction(1,1/11,.5),actions[1]:SearchAction(10,10/11,.5)},root_value=0.,simulations=11)
    class FakeMCTS:
        config=SimpleNamespace(simulations=11,c_puct=1.5,root_noise=True)
        def search(self, environment): return result
        def advance(self, environment, action): return True
    start=StartPosition("full","initial",board.fen(),1,{},1)
    played=play_game(run_id="run",run_seed=1,generation=0,game_index=0,actor_id=0,start=start,mcts=FakeMCTS(),schedule=TemperatureSchedule(0.,0,0.),model_checksum="x")
    assert played is not None
    assert played.sealed.samples[0].selected_action==actions[1]
    parsed=chess.pgn.read_game(io.StringIO(played.sealed.pgn))
    assert parsed is not None and next(iter(parsed.mainline_moves()))==moves[1]
    assert set(played.sealed.samples[0].policy_actions)==set(actions)
