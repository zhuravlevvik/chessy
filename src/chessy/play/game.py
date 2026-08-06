"""Authoritative, lock-protected local game session."""

from __future__ import annotations

import random
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import chess
import chess.pgn

from chessy.chess import ChessEnvironment
from chessy.encoding import encode_move
from chessy.play.agent import AgentDecision, MCTSAgent, ModelInfo

SessionStatus = Literal["waiting", "active", "bot_thinking", "finished"]


@dataclass(frozen=True, slots=True)
class TimeControl:
    id: str
    initial_seconds: float | None
    increment_seconds: float = 0.0


TIME_CONTROLS: dict[str, TimeControl] = {
    "untimed": TimeControl("untimed", None, 0),
    "3+2": TimeControl("3+2", 180, 2),
    "5+0": TimeControl("5+0", 300, 0),
    "10+0": TimeControl("10+0", 600, 0),
    "15+10": TimeControl("15+10", 900, 10),
}


@dataclass(frozen=True, slots=True)
class MoveRecord:
    ply: int
    uci: str
    san: str
    action: int
    color: chess.Color
    human: bool
    fen_before: str
    history_fens: tuple[str, ...]


class GameSession:
    """Own rules, clocks, result, move history, MCTS state, and PGN."""

    def __init__(
        self,
        agent: MCTSAgent,
        *,
        human_color: str = "white",
        time_control: str = "untimed",
        profile: str = "normal",
        feedback_opt_in: bool = False,
        seed: int | None = None,
        clock: callable = time.monotonic,
        game_id: str | None = None,
        max_plies: int = 300,
    ) -> None:
        if human_color not in {"white", "black", "random"}:
            raise ValueError("human_color must be white, black, or random")
        if time_control not in TIME_CONTROLS:
            raise ValueError(f"unknown time control: {time_control!r}")
        if profile not in {"fast", "normal", "deep"}:
            raise ValueError(f"unknown MCTS profile: {profile!r}")
        self.id = game_id or str(uuid.uuid4())
        self.agent = agent
        self.model: ModelInfo = agent.model
        self.profile = profile
        self.time_control = TIME_CONTROLS[time_control]
        self.feedback_opt_in = bool(feedback_opt_in)
        self.feedback_saved = False
        self.created_at = datetime.now(timezone.utc)
        self.environment = ChessEnvironment()
        self.lock = threading.RLock()
        self.clock = clock
        self.max_plies = max_plies
        rng = random.Random(seed if seed is not None else uuid.UUID(self.id).int)
        self.human_color = (
            rng.choice((chess.WHITE, chess.BLACK))
            if human_color == "random"
            else human_color == "white"
        )
        self.bot_color = not self.human_color
        initial = self.time_control.initial_seconds
        self.remaining: dict[chess.Color, float | None] = {chess.WHITE: initial, chess.BLACK: initial}
        self.turn_started_at = self.clock()
        self.status: SessionStatus = "active" if self.human_color == chess.WHITE else "bot_thinking"
        self.result = "*"
        self.termination: str | None = None
        self.moves: list[MoveRecord] = []
        self.decisions: list[AgentDecision] = []
        self.sequence = 0

    def _tick(self) -> int:
        self.sequence += 1
        return self.sequence

    def _consume_clock(self, color: chess.Color, now: float) -> bool:
        remaining = self.remaining[color]
        if remaining is None:
            return True
        elapsed = max(0.0, now - self.turn_started_at)
        remaining -= elapsed
        self.remaining[color] = max(0.0, remaining)
        if remaining <= 0:
            self._finish_timeout(color)
            return False
        return True

    def _finish_timeout(self, expired: chess.Color) -> None:
        opponent = not expired
        if self.environment.board.has_insufficient_material(opponent):
            self._finish("1/2-1/2", "timeout_insufficient_material")
        else:
            self._finish("0-1" if expired == chess.WHITE else "1-0", "timeout")

    def expire_if_needed(self) -> bool:
        with self.lock:
            if self.status == "finished" or self.time_control.initial_seconds is None:
                return False
            color = self.environment.board.turn
            remaining = self.remaining[color]
            assert remaining is not None
            if self.clock() - self.turn_started_at < remaining:
                return False
            self._consume_clock(color, self.clock())
            self._tick()
            return True

    def deadline_seconds(self) -> float | None:
        with self.lock:
            if self.status == "finished":
                return None
            remaining = self.remaining[self.environment.board.turn]
            if remaining is None:
                return None
            return max(0.0, remaining - (self.clock() - self.turn_started_at))

    def _finish(self, result: str, termination: str) -> None:
        self.result = result
        self.termination = termination
        self.status = "finished"

    def _check_game_over(self) -> None:
        outcome = self.environment.outcome()
        if outcome is not None:
            result = outcome.result()
            termination = outcome.termination.name.lower()
            self._finish(result, termination)
        elif len(self.moves) >= self.max_plies:
            self._finish("1/2-1/2", "max_plies")

    def _apply_move(self, move: chess.Move, *, human: bool, decision: AgentDecision | None = None) -> MoveRecord:
        board = self.environment.board
        color = board.turn
        now = self.clock()
        if not self._consume_clock(color, now):
            raise TimeoutError("clock expired before the move")
        if move not in board.legal_moves:
            raise ValueError("illegal move")
        record = MoveRecord(
            ply=len(self.moves) + 1,
            uci=move.uci(),
            san=board.san(move),
            action=encode_move(board, move),
            color=color,
            human=human,
            # Preserve the raw en-passant target because board119-v1 encodes
            # ``ep_square`` even when no legal capture currently exists.
            fen_before=board.fen(en_passant="fen"),
            history_fens=tuple(
                history.fen(en_passant="fen") for history in self.environment.history()
            ),
        )
        self.environment.push(move)
        self.moves.append(record)
        if decision is not None:
            self.decisions.append(decision)
        remaining = self.remaining[color]
        if remaining is not None:
            self.remaining[color] = remaining + self.time_control.increment_seconds
        self.agent.advance(self.environment, record.action)
        self.turn_started_at = now
        self._check_game_over()
        if self.status != "finished":
            self.status = "bot_thinking" if self.environment.board.turn == self.bot_color else "active"
        self._tick()
        return record

    def apply_human_move(self, uci: str) -> MoveRecord:
        with self.lock:
            if self.status == "finished":
                raise RuntimeError("game is finished")
            if self.status == "bot_thinking":
                raise RuntimeError("bot is thinking")
            if self.environment.board.turn != self.human_color:
                raise RuntimeError("not the human turn")
            try:
                move = self.environment.board.parse_uci(uci)
            except ValueError as exc:
                raise ValueError("invalid or illegal move") from exc
            return self._apply_move(move, human=True)

    def play_bot_move(self) -> tuple[MoveRecord, AgentDecision] | None:
        with self.lock:
            if self.status == "finished":
                return None
            if self.environment.board.turn != self.bot_color:
                raise RuntimeError("not the bot turn")
            self.status = "bot_thinking"
            search_environment = self.environment.copy()
            expected_fen = self.environment.fen()
        # Search deliberately runs without the session lock: health, snapshots,
        # rejection of duplicate moves, and deadline handling remain responsive.
        decision = self.agent.choose_move(search_environment)
        with self.lock:
            if self.status == "finished":
                return None
            if self.environment.fen() != expected_fen or self.environment.board.turn != self.bot_color:
                raise RuntimeError("game changed while the bot was thinking")
            try:
                record = self._apply_move(decision.move, human=False, decision=decision)
            except TimeoutError:
                self._tick()
                return None
            return record, decision

    def resign(self) -> None:
        with self.lock:
            if self.status == "finished":
                raise RuntimeError("game is finished")
            self._finish("0-1" if self.human_color == chess.WHITE else "1-0", "resignation")
            self._tick()

    def offer_draw(self) -> None:
        with self.lock:
            if self.status == "finished":
                raise RuntimeError("game is finished")
            self._tick()  # v1 bot deterministically declines; API emits the corresponding event.

    def pgn(self) -> str:
        with self.lock:
            game = chess.pgn.Game.from_board(self.environment.board)
            game.headers["Date"] = self.created_at.strftime("%Y.%m.%d")
            game.headers["White"] = "Human" if self.human_color == chess.WHITE else f"Chessy:{self.model.id}"
            game.headers["Black"] = "Human" if self.human_color == chess.BLACK else f"Chessy:{self.model.id}"
            game.headers["Result"] = self.result
            game.headers["Termination"] = self.termination or "unterminated"
            game.headers["ChessyModel"] = self.model.id
            game.headers["ChessyWeights"] = self.model.checksum
            game.headers["ChessyMCTS"] = self.agent.config.version
            game.headers["ChessyProfile"] = self.profile
            game.headers["ChessySimulations"] = str(self.agent.config.simulations)
            game.headers["TimeControl"] = (
                "-" if self.time_control.initial_seconds is None
                else f"{int(self.time_control.initial_seconds)}+{int(self.time_control.increment_seconds)}"
            )
            return str(game) + "\n"

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            board = self.environment.board
            now = self.clock()
            remaining: dict[str, float | None] = {}
            for color, key in ((chess.WHITE, "white"), (chess.BLACK, "black")):
                value = self.remaining[color]
                if value is not None and color == board.turn and self.status != "finished":
                    value = max(0.0, value - (now - self.turn_started_at))
                remaining[key] = value
            return {
                "game_id": self.id,
                "sequence": self.sequence,
                "status": self.status,
                "fen": board.fen(),
                "turn": "white" if board.turn else "black",
                "human_color": "white" if self.human_color else "black",
                "bot_color": "white" if self.bot_color else "black",
                "result": self.result,
                "termination": self.termination,
                "moves": [{"ply": move.ply, "uci": move.uci, "san": move.san, "human": move.human} for move in self.moves],
                "legal_moves": [move.uci() for move in board.legal_moves] if self.status != "finished" else [],
                "clocks": remaining,
                "server_monotonic": now,
                "model": self.model.public_dict(),
                "profile": self.profile,
                "simulations": self.agent.config.simulations,
                "time_control": self.time_control.id,
                "feedback_opt_in": self.feedback_opt_in,
                "feedback_saved": self.feedback_saved,
            }
