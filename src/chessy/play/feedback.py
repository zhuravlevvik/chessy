"""Atomic writer for confirmed ``chessy-human-feedback-v1`` games."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

import chess

from chessy.encoding import ACTION_ENCODING_VERSION, BOARD_ENCODING_VERSION
from chessy.play.game import GameSession

FEEDBACK_FORMAT = "chessy-human-feedback-v1"
SAMPLE_WEIGHT = 4.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ordinary_directory(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink() and stat.S_ISDIR(path.lstat().st_mode)


def _human_result(session: GameSession) -> str:
    if session.result == "1/2-1/2":
        return "draw"
    won = (session.result == "1-0") == (session.human_color == chess.WHITE)
    return "win" if won else "loss"


def save_human_feedback(session: GameSession, root: Path, *, confirmed: bool = False) -> Path:
    """Persist confirmed human targets once, using sibling temp dir + rename."""
    with session.lock:
        if not session.feedback_opt_in and not confirmed:
            raise PermissionError("feedback opt-in was not enabled for this game")
        if session.status != "finished":
            raise RuntimeError("feedback can only be saved after the game")
        if confirmed:
            session.feedback_opt_in = True
        if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", session.id, re.I) is None:
            raise ValueError("feedback game ID must be a UUID")
        if re.fullmatch(r"[0-9a-f]{64}", session.model.checksum) is None:
            raise ValueError("feedback requires a checksummed model export")
        root = Path(root)
        if root.exists() and not _ordinary_directory(root):
            raise ValueError("feedback root must be an ordinary directory")
        destination = root / session.id
        if destination.exists():
            if not _ordinary_directory(destination):
                raise ValueError("feedback destination must be an ordinary directory")
            # A pre-existing ID is safe only when it is a complete, verifiable
            # artifact.  This avoids treating a torn/manual directory as success.
            from chessy.feedback import verify_feedback_game
            checked = verify_feedback_game(destination)
            manifest = checked["manifest"]
            if (destination / "game.pgn").read_text(encoding="utf-8") != session.pgn() or manifest["model"]["checksum"] != session.model.checksum or manifest["result"] != session.result or manifest["human_color"] != ("white" if session.human_color else "black"):
                raise FileExistsError("feedback game ID already belongs to different content")
            session.feedback_saved = True
            return destination
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise ValueError("feedback root must not be a symlink")
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
                        "sample_id": hashlib.sha256(f"{session.id}:{move.ply}:{move.uci}:{move.action}".encode()).hexdigest(),
                        "encodings": {"board": BOARD_ENCODING_VERSION, "action": ACTION_ENCODING_VERSION},
                    }
                    file.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
            _fsync(pgn_path)
            _fsync(samples_path)
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
                "encodings": {"board": BOARD_ENCODING_VERSION, "action": ACTION_ENCODING_VERSION},
                "human_samples": count,
                "hashes": {"game.pgn": _sha256(pgn_path), "samples.jsonl": _sha256(samples_path)},
            }
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            _fsync(manifest_path)
            _fsync_directory(temporary)
            temporary.rename(destination)
            _fsync_directory(root)
            session.feedback_saved = True
            session._tick()
            return destination
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
