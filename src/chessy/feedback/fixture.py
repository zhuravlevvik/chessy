"""Ignored tiny fixture used by the feedback smoke preset and tests."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import chess
from chessy.encoding import encode_move
from chessy.feedback.builder import build_feedback_dataset
from chessy.play import AgentDecision, GameSession, ModelInfo, save_human_feedback
from chessy.mcts import MCTSConfig, SearchAction, SearchResult
from chessy.personal.fixture import prepare_smoke_fixture

class _FirstMoveAgent:
    def __init__(self) -> None:
        self.model = ModelInfo("feedback-fixture", "feedback fixture", "b" * 64); self.config = MCTSConfig(simulations=1)
    def advance(self, *_: object) -> bool: return True
    def choose_move(self, environment):  # type: ignore[no-untyped-def]
        move = sorted(environment.legal_moves(), key=lambda item: item.uci())[0]; action = encode_move(environment.board, move)
        return AgentDecision(move, action, SearchResult(action, move, {action: SearchAction(1, 1, 1)}, 0, 1), 0, self.model.id, self.model.checksum, self.config)

def _game(color: str) -> GameSession:
    game_id = "11111111-1111-4111-8111-111111111111" if color == "white" else "22222222-2222-4222-8222-222222222222"
    session = GameSession(_FirstMoveAgent(), human_color=color, feedback_opt_in=True, game_id=game_id)
    session.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    if color == "black": session.play_bot_move()
    session.apply_human_move("e7e5" if color == "black" else "e2e4")
    session.play_bot_move(); session.resign(); return session

def prepare_feedback_smoke_fixture(root: Path) -> dict[str, str]:
    base = prepare_smoke_fixture(root); fixture = Path(root).resolve() / "runs" / "personal-smoke-fixture"; raw = fixture / "feedback-raw"
    raw.mkdir(parents=True, exist_ok=True)
    for color in ("white", "black"):
        session = _game(color); save_human_feedback(session, raw)
    immutable = build_feedback_dataset(input=raw, output=fixture / "feedback-encoded", sample_weight=4.0, max_positions_per_game=16, segment_samples=8)
    stable = immutable.parent / "feedback-dataset-fixture.json"
    if not stable.exists() or stable.read_bytes() != immutable.read_bytes(): stable.write_bytes(immutable.read_bytes())
    return base | {"feedback_manifest": str(stable.relative_to(root)), "feedback_fingerprint": json.loads(stable.read_text())["content_fingerprint"]}
