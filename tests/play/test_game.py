from __future__ import annotations

import io
from pathlib import Path

import chess
import chess.pgn
import pytest

from chessy.encoding import encode_move
from chessy.mcts import MCTSConfig, SearchAction, SearchResult
from chessy.play import AgentDecision, GameSession, ModelInfo, save_human_feedback


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FirstMoveAgent:
    def __init__(self, clock: FakeClock | None = None) -> None:
        self.model = ModelInfo("test-model", "Test model", "a" * 64)
        self.config = MCTSConfig(simulations=1)
        self.clock = clock or FakeClock()
        self.advanced: list[int] = []

    def choose_move(self, environment):  # type: ignore[no-untyped-def]
        move = sorted(environment.legal_moves(), key=lambda candidate: candidate.uci())[0]
        action = encode_move(environment.board, move)
        result = SearchResult(action, move, {action: SearchAction(1, 1, 1)}, 0, 1)
        return AgentDecision(move, action, result, 0, self.model.id, self.model.checksum, self.config)

    def advance(self, environment, action: int) -> bool:  # type: ignore[no-untyped-def]
        self.advanced.append(action)
        return True


@pytest.mark.parametrize("choice", ("white", "black", "random"))
def test_color_choice_and_bot_first_move(choice: str) -> None:
    session = GameSession(FirstMoveAgent(), human_color=choice, seed=3)
    assert session.human_color in (chess.WHITE, chess.BLACK)
    if session.bot_color == chess.WHITE:
        assert session.status == "bot_thinking"
        assert session.play_bot_move() is not None
        assert len(session.moves) == 1 and not session.moves[0].human
    else:
        assert session.status == "active"


def test_moves_are_authoritative_and_out_of_turn_is_rejected() -> None:
    session = GameSession(FirstMoveAgent(), human_color="white")
    with pytest.raises(ValueError, match="invalid"):
        session.apply_human_move("e2e5")
    record = session.apply_human_move("e2e4")
    assert record.san == "e4"
    assert session.status == "bot_thinking"
    with pytest.raises(RuntimeError, match="thinking"):
        session.apply_human_move("d2d4")
    assert session.play_bot_move() is not None
    assert session.status == "active"


def test_resign_draw_decline_clock_increment_and_timeout() -> None:
    clock = FakeClock()
    session = GameSession(FirstMoveAgent(clock), human_color="white", time_control="3+2", clock=clock)
    clock.advance(10)
    session.apply_human_move("e2e4")
    assert session.remaining[chess.WHITE] == pytest.approx(172)
    session.offer_draw()
    assert session.status == "bot_thinking"
    clock.advance(181)
    assert session.expire_if_needed()
    assert session.status == "finished"
    assert session.result == "1-0"
    assert session.termination == "timeout"

    resigned = GameSession(FirstMoveAgent(), human_color="black")
    resigned.resign()
    assert resigned.result == "1-0"
    assert resigned.termination == "resignation"


def test_pgn_round_trips_before_and_after_finish() -> None:
    session = GameSession(FirstMoveAgent(), human_color="white", profile="fast")
    session.apply_human_move("e2e4")
    session.play_bot_move()
    ongoing = chess.pgn.read_game(io.StringIO(session.pgn()))
    assert ongoing is not None and ongoing.headers["Result"] == "*"
    session.resign()
    finished = chess.pgn.read_game(io.StringIO(session.pgn()))
    assert finished is not None
    assert finished.headers["ChessyMCTS"] == "mcts-puct-v1"
    assert finished.headers["Termination"] == "resignation"
    assert len(list(finished.mainline_moves())) == 2


def test_move_history_preserves_raw_en_passant_square_for_board_encoding() -> None:
    session = GameSession(FirstMoveAgent(), human_color="white")
    session.apply_human_move("e2e4")
    session.play_bot_move()  # The deterministic fake agent chooses a7a5.
    record = session.apply_human_move("h2h3")
    assert record.fen_before.split()[3] == "a6"
    assert record.history_fens[0].split()[3] == "a6"


def test_feedback_requires_opt_in_and_finish_then_is_atomic_and_idempotent(tmp_path: Path) -> None:
    declined = GameSession(FirstMoveAgent(), human_color="white", feedback_opt_in=False)
    declined.resign()
    with pytest.raises(PermissionError):
        save_human_feedback(declined, tmp_path / "feedback")
    assert not (tmp_path / "feedback").exists()

    session = GameSession(FirstMoveAgent(), human_color="white", feedback_opt_in=True)
    with pytest.raises(RuntimeError, match="after"):
        save_human_feedback(session, tmp_path / "feedback")
    session.apply_human_move("e2e4")
    session.play_bot_move()
    session.resign()
    destination = save_human_feedback(session, tmp_path / "feedback")
    assert {path.name for path in destination.iterdir()} == {"game.pgn", "manifest.json", "samples.jsonl"}
    lines = (destination / "samples.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert '"move_uci":"e2e4"' in lines[0]
    assert save_human_feedback(session, tmp_path / "feedback") == destination
    assert len(list((tmp_path / "feedback").iterdir())) == 1
