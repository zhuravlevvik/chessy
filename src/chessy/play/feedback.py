"""Atomic writer for confirmed ``chessy-human-feedback-v1`` games."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import chess

from chessy.play.game import GameSession

FEEDBACK_FORMAT = "chessy-human-feedback-v1"
SAMPLE_WEIGHT = 4.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _human_result(session: GameSession) -> str:
    if session.result == "1/2-1/2":
        return "draw"
    won = (session.result == "1-0") == (session.human_color == chess.WHITE)
    return "win" if won else "loss"


def save_human_feedback(session: GameSession, root: Path) -> Path:
    """Persist confirmed human targets once, using sibling temp dir + rename."""
    with session.lock:
        if not session.feedback_opt_in:
            raise PermissionError("feedback opt-in was not enabled for this game")
        if session.status != "finished":
            raise RuntimeError("feedback can only be saved after the game")
        root = Path(root)
        destination = root / session.id
        if destination.is_dir():
            session.feedback_saved = True
            return destination
        root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{session.id}.tmp-", dir=root))
        try:
            pgn_path = temporary / "game.pgn"
            pgn_path.write_text(session.pgn(), encoding="utf-8")
            samples_path = temporary / "samples.jsonl"
            human_result = _human_result(session)
            with samples_path.open("w", encoding="utf-8") as file:
                for move in session.moves:
                    if not move.human:
                        continue
                    sample = {
                        "format": FEEDBACK_FORMAT,
                        "game_id": session.id,
                        "ply": move.ply,
                        "fen": move.fen_before,
                        "history_fens": list(move.history_fens),
                        "move_uci": move.uci,
                        "action": move.action,
                        "human_color": "white" if session.human_color else "black",
                        "result": human_result,
                        "source": "human_online",
                        "weight": SAMPLE_WEIGHT,
                    }
                    file.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
            count = sum(move.human for move in session.moves)
            manifest = {
                "format": FEEDBACK_FORMAT,
                "game_id": session.id,
                "created_at": session.created_at.isoformat().replace("+00:00", "Z"),
                "human_color": "white" if session.human_color else "black",
                "result": session.result,
                "termination": session.termination,
                "time_control": session.time_control.id,
                "model": {
                    "id": session.model.id,
                    "checksum": session.model.checksum,
                    "random_seed": session.model.random_seed,
                },
                "mcts": session.agent.config.to_dict(),
                "sample_weight": SAMPLE_WEIGHT,
                "human_samples": count,
                "hashes": {"game.pgn": _sha256(pgn_path), "samples.jsonl": _sha256(samples_path)},
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            temporary.rename(destination)
            session.feedback_saved = True
            session._tick()
            return destination
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
