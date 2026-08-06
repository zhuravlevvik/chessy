from __future__ import annotations
from dataclasses import dataclass
import chess, numpy as np
from chessy.chess import ChessEnvironment
from chessy.mcts import MCTS, MCTSConfig, Evaluator

class RandomAgent:
    def __init__(self, seed:int=0)->None: self.rng=np.random.default_rng(seed)
    def select(self, environment:ChessEnvironment)->chess.Move:
        moves=tuple(environment.legal_moves())
        if not moves: raise ValueError("no legal move")
        return moves[int(self.rng.integers(0,len(moves)))]

_VALUES={chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900,chess.KING:0}
class MaterialAgent:
    """Deterministic depth-limited negamax baseline; never used as a teacher."""
    def __init__(self, depth:int=2)->None:
        if depth not in (1,2): raise ValueError("material baseline depth must be 1 or 2")
        self.depth=depth
    def _score(self, board:chess.Board)->int:
        outcome=board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:return 0
            return 100000 if outcome.winner==board.turn else -100000
        material=sum((_VALUES[p]*len(board.pieces(p,chess.WHITE))-_VALUES[p]*len(board.pieces(p,chess.BLACK))) for p in _VALUES)
        return material if board.turn==chess.WHITE else -material
    def _negamax(self,board:chess.Board,depth:int,alpha:int,beta:int)->int:
        if depth==0 or board.outcome(claim_draw=True) is not None:return self._score(board)
        value=-100001
        for move in sorted(board.legal_moves,key=lambda m:m.uci()):
            board.push(move); value=max(value,-self._negamax(board,depth-1,-beta,-alpha)); board.pop(); alpha=max(alpha,value)
            if alpha>=beta:break
        return value
    def select(self,environment:ChessEnvironment)->chess.Move:
        board=environment.board; choices=[]; best=-100002
        for move in sorted(board.legal_moves,key=lambda m:m.uci()):
            board.push(move); score=-self._negamax(board,self.depth-1,-100001,100001); board.pop()
            if score>best:best,choices=score,[move]
        return choices[0]

@dataclass
class MCTSAgent:
    evaluator: Evaluator
    simulations: int=64
    c_puct: float=1.5
    def select(self,environment:ChessEnvironment)->chess.Move:
        # Arena is deterministic: no root noise and no sampling temperature.
        return MCTS(self.evaluator,MCTSConfig(simulations=self.simulations,c_puct=self.c_puct,temperature=0,root_noise=False)).search(environment).move
