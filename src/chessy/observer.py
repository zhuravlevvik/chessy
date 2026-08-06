"""Filesystem bridge between self-play workers and the local spectator UI."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import chess.pgn

from chessy.config.canonical import canonical_json, fingerprint


_ID = re.compile(r"^[a-zA-Z0-9_.-]{1,160}$")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        temporary.write_bytes(canonical_json(value))
        with temporary.open("rb") as file: os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def _frames(pgn: str) -> tuple[str, list[dict[str, object]]]:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None: raise ValueError("observer PGN is empty")
    board = game.board(); initial = board.fen(); frames: list[dict[str, object]] = [{"ply": 0, "fen": initial, "uci": None, "san": None}]
    for ply, move in enumerate(game.mainline_moves(), 1):
        san = board.san(move); board.push(move); frames.append({"ply": ply, "fen": board.fen(), "uci": move.uci(), "san": san})
    return initial, frames


class TrainingObserver:
    def __init__(self, run_path: Path, *, enabled: bool, archive_every_generations: int, live_game_index: int) -> None:
        self.run_path = Path(run_path); self.enabled = enabled; self.archive_every_generations = archive_every_generations; self.live_game_index = live_game_index
        if archive_every_generations <= 0 or live_game_index < 0: raise ValueError("invalid observer configuration")

    def live_update(self, value: dict[str, Any]) -> None:
        if not self.enabled or value.get("game_index") != self.live_game_index: return
        body = {"format": "chessy-observer-game-v1", "id": f"{self.run_path.name}-live", "run_id": self.run_path.name, "kind": "live", **value}
        body["content_fingerprint"] = fingerprint(body)
        _atomic_json(self.run_path / "showcase" / "live.json", body)

    def archive(self, game: Any, model_checksum: str) -> Path | None:
        sealed = game.sealed
        if not self.enabled or sealed.game_index != self.live_game_index: return None
        initial, frames = _frames(sealed.pgn)
        complete = {"status": "complete", "generation": sealed.generation, "game_index": sealed.game_index, "model_checksum": model_checksum, "initial_fen": initial, "fen": frames[-1]["fen"], "result": sealed.result, "termination": sealed.termination, "frames": frames}
        self.live_update(complete)
        if sealed.generation % self.archive_every_generations: return None
        directory = self.run_path / "showcase" / f"generation-{sealed.generation:04d}"
        pgn_bytes = sealed.pgn.encode("utf-8")
        body: dict[str, Any] = {"format": "chessy-observer-game-v1", "id": f"{self.run_path.name}-g{sealed.generation:04d}", "run_id": self.run_path.name, "kind": "archive", **complete, "pgn": "game.pgn", "pgn_sha256": hashlib.sha256(pgn_bytes).hexdigest()}
        body["content_fingerprint"] = fingerprint(body)
        payload = canonical_json(body); manifest = directory / "manifest.json"; pgn_path = directory / "game.pgn"
        if manifest.exists():
            if manifest.is_symlink() or manifest.read_bytes() != payload: raise FileExistsError("observer archive already exists with different content")
            if pgn_path.is_symlink() or not pgn_path.is_file() or hashlib.sha256(pgn_path.read_bytes()).hexdigest() != body["pgn_sha256"]: raise FileExistsError("observer archive PGN is missing or corrupt")
            return manifest
        directory.mkdir(parents=True, exist_ok=True)
        temporary = pgn_path.with_name(f".{pgn_path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_bytes(pgn_bytes)
            with temporary.open("rb") as file: os.fsync(file.fileno())
            os.replace(temporary, pgn_path); _atomic_json(manifest, body)
        finally:
            if temporary.exists(): temporary.unlink()
        return manifest


def discover_observer_games(runs_dir: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    root = Path(runs_dir)
    if not root.is_dir() or root.is_symlink(): return []
    paths = sorted(root.glob("*/showcase/generation-*/manifest.json"), reverse=True)[:limit]
    live = sorted(root.glob("*/showcase/live.json"), reverse=True)
    result: list[dict[str, Any]] = []
    for path in [*live, *paths]:
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > 5 * 1024 * 1024: continue
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError): continue
        if isinstance(value, dict) and value.get("format") == "chessy-observer-game-v1" and isinstance(value.get("id"), str) and _ID.fullmatch(value["id"]):
            expected = dict(value); actual = expected.pop("content_fingerprint", None)
            if actual != fingerprint(expected): continue
            if value.get("kind") == "archive":
                pgn = path.parent / str(value.get("pgn", "")); checksum = value.get("pgn_sha256")
                try:
                    if pgn.is_symlink() or not pgn.is_file() or not isinstance(checksum, str) or hashlib.sha256(pgn.read_bytes()).hexdigest() != checksum: continue
                except OSError: continue
            result.append(value)
    return result


def observer_game(runs_dir: Path, game_id: str) -> dict[str, Any]:
    if _ID.fullmatch(game_id) is None: raise KeyError(game_id)
    for game in discover_observer_games(runs_dir):
        if game["id"] == game_id: return game
    raise KeyError(game_id)
