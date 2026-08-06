"""Fail-closed reader for immutable ``chessy-human-feedback-v1`` games."""
from __future__ import annotations

import hashlib
import io
import json
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from chessy.encoding import ACTION_ENCODING_VERSION, BOARD_ENCODING_VERSION, decode_action, encode_move
from chessy.mcts import MCTSConfig

FORMAT = "chessy-human-feedback-v1"
_FILES = frozenset({"game.pgn", "samples.jsonl", "manifest.json"})
_HEX = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)
_RESULTS = {"1-0", "0-1", "1/2-1/2"}
_MAX_BYTES = {"game.pgn": 2 * 1024 * 1024, "samples.jsonl": 20 * 1024 * 1024, "manifest.json": 1024 * 1024}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error(game_id: str, reason: str) -> ValueError:
    return ValueError(f"feedback game {game_id}: {reason}")


def _ordinary_game_dir(path: Path) -> None:
    if path.is_symlink() or not path.is_dir() or not _UUID.fullmatch(path.name):
        raise ValueError("feedback game path must be an ordinary UUID directory")
    entries = {item.name for item in path.iterdir()}
    if entries != _FILES:
        raise _error(path.name, "unexpected files")
    for item in path.iterdir():
        if item.is_symlink() or not stat.S_ISREG(item.lstat().st_mode):
            raise _error(path.name, "payload entries must be regular files")
        if item.stat().st_size > _MAX_BYTES[item.name]:
            raise _error(path.name, f"oversized {item.name}")


def _human_result(result: str, human_color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    return "win" if (result == "1-0") == bool(human_color) else "loss"


def _read_json(path: Path, game_id: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(game_id, "invalid manifest JSON") from exc
    if not isinstance(data, dict):
        raise _error(game_id, "manifest must be an object")
    return data


def verify_feedback_game(path: Path) -> dict[str, Any]:
    """Verify a raw artifact by hashes and exact legal PGN replay.

    v1 rows did not contain IDs/encoding labels; they remain readable while
    newer writer fields are validated whenever present.
    """
    path = Path(path)
    _ordinary_game_dir(path)
    game_id = path.name
    manifest = _read_json(path / "manifest.json", game_id)
    if manifest.get("format") != FORMAT or manifest.get("game_id") != game_id:
        raise _error(game_id, "format or game ID mismatch")
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict) or set(hashes) != {"game.pgn", "samples.jsonl"}:
        raise _error(game_id, "invalid hashes")
    for name, digest in hashes.items():
        if not isinstance(digest, str) or not _HEX.fullmatch(digest) or _sha256(path / name) != digest:
            raise _error(game_id, f"checksum mismatch for {name}")
    color_name = manifest.get("human_color")
    if color_name not in {"white", "black"} or manifest.get("result") not in _RESULTS:
        raise _error(game_id, "invalid color or result")
    if not isinstance(manifest.get("termination"), str) or not isinstance(manifest.get("time_control"), str):
        raise _error(game_id, "invalid termination or time control")
    model = manifest.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("id"), str) or not isinstance(model.get("checksum"), str) or not _HEX.fullmatch(model["checksum"]):
        raise _error(game_id, "invalid model checksum")
    try:
        created_at = datetime.fromisoformat(str(manifest.get("created_at", "")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise _error(game_id, "invalid created_at") from exc
    if created_at.tzinfo is None:
        raise _error(game_id, "created_at must include a timezone")
    try:
        MCTSConfig.from_dict(manifest.get("mcts"))
    except (TypeError, ValueError) as exc:
        raise _error(game_id, "invalid MCTS configuration") from exc
    suggested_weight = manifest.get("sample_weight")
    if not isinstance(suggested_weight, (int, float)) or not float(suggested_weight) > 0 or not float(suggested_weight) < float("inf"):
        raise _error(game_id, "invalid sample weight")
    encodings = manifest.get("encodings")
    if encodings is not None and encodings != {"board": BOARD_ENCODING_VERSION, "action": ACTION_ENCODING_VERSION}:
        raise _error(game_id, "unsupported encodings")
    try:
        pgn_text = (path / "game.pgn").read_text(encoding="utf-8")
        source = io.StringIO(pgn_text)
        game = chess.pgn.read_game(source)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise _error(game_id, "invalid PGN") from exc
    if game is None or game.errors or chess.pgn.read_game(source) is not None:
        raise _error(game_id, "PGN must contain exactly one game")
    if game.headers.get("Result") != manifest["result"] or game.headers.get("Result") not in _RESULTS:
        raise _error(game_id, "PGN result mismatch or unfinished game")
    expected_human = "Human"
    if game.headers.get("White") == expected_human:
        pgn_color = "white"
    elif game.headers.get("Black") == expected_human:
        pgn_color = "black"
    else:
        raise _error(game_id, "PGN has no human side")
    if pgn_color != color_name:
        raise _error(game_id, "human color mismatch")
    bot_header = game.headers.get("Black" if pgn_color == "white" else "White")
    if bot_header != f"Chessy:{model['id']}" or game.headers.get("ChessyModel") != model["id"] or game.headers.get("ChessyWeights") != model["checksum"]:
        raise _error(game_id, "PGN model identity mismatch")
    try:
        lines = (path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise _error(game_id, "invalid samples encoding") from exc
    if len(lines) != manifest.get("human_samples"):
        raise _error(game_id, "human sample count mismatch")
    samples: dict[int, dict[str, Any]] = {}
    sample_ids: set[str] = set()
    for line in lines:
        if len(line) > 1_000_000:
            raise _error(game_id, "oversized sample row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _error(game_id, "invalid sample JSON") from exc
        if not isinstance(row, dict) or row.get("format") != FORMAT or row.get("game_id") != game_id:
            raise _error(game_id, "sample identity mismatch")
        ply = row.get("ply")
        if not isinstance(ply, int) or ply <= 0 or ply in samples:
            raise _error(game_id, "duplicate or invalid sample ply")
        if row.get("human_color") != color_name or row.get("result") != _human_result(manifest["result"], color_name == "white"):
            raise _error(game_id, "sample human result mismatch")
        if row.get("source") != "human_online":
            raise _error(game_id, "sample source must be human_online")
        if not isinstance(row.get("weight"), (int, float)) or not 0 < float(row["weight"]) < float("inf"):
            raise _error(game_id, "invalid row weight")
        if float(row["weight"]) != float(suggested_weight):
            raise _error(game_id, "sample weight does not match manifest")
        if row.get("encodings") is not None and row["encodings"] != {"board": BOARD_ENCODING_VERSION, "action": ACTION_ENCODING_VERSION}:
            raise _error(game_id, "unsupported sample encodings")
        identifier = row.get("sample_id")
        if identifier is not None:
            if not isinstance(identifier, str) or not _HEX.fullmatch(identifier) or identifier in sample_ids:
                raise _error(game_id, "invalid or duplicate sample ID")
            sample_ids.add(identifier)
        samples[ply] = row
    board = game.board()
    human_color = color_name == "white"
    human_plies: set[int] = set()
    for ply, move in enumerate(game.mainline_moves(), start=1):
        if move not in board.legal_moves:
            raise _error(game_id, f"illegal PGN move at ply {ply}")
        if board.turn == human_color:
            human_plies.add(ply)
            row = samples.get(ply)
            if row is None:
                raise _error(game_id, f"missing human sample at ply {ply}")
            fen = board.fen(en_passant="fen")
            snapshot = board.copy(stack=True)
            boards: list[chess.Board] = []
            while len(boards) < 8:
                boards.append(snapshot.copy(stack=True))
                if not snapshot.move_stack:
                    break
                snapshot.pop()
            while len(boards) < 8:
                boards.append(boards[-1].copy(stack=True))
            histories = [item.fen(en_passant="fen") for item in boards]
            if row.get("fen") != fen or row.get("history_fens") != histories:
                raise _error(game_id, f"FEN/history mismatch at ply {ply}")
            action = row.get("action")
            if row.get("move_uci") != move.uci() or not isinstance(action, int) or action != encode_move(board, move):
                raise _error(game_id, f"move/action mismatch at ply {ply}")
            try:
                if decode_action(board, action) != move:
                    raise _error(game_id, f"action decode mismatch at ply {ply}")
            except ValueError as exc:
                raise _error(game_id, f"invalid action at ply {ply}") from exc
        elif ply in samples:
            raise _error(game_id, f"bot move injected as target at ply {ply}")
        board.push(move)
    if set(samples) != human_plies:
        raise _error(game_id, "sample plies do not equal human plies")
    return {"manifest": manifest, "game": game, "samples": [samples[key] for key in sorted(samples)], "game_id": game_id}


def inspect_feedback_root(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("feedback root must be an ordinary directory")
    rows: list[dict[str, Any]] = []
    for item in root.iterdir():
        if item.name.startswith(".tmp-") or item.name == ".DS_Store":
            continue
        checked = verify_feedback_game(item)
        manifest = checked["manifest"]
        rows.append({"game_id": checked["game_id"], "created_at": manifest["created_at"], "human_samples": manifest["human_samples"], "result": manifest["result"], "human_color": manifest["human_color"]})
    return sorted(rows, key=lambda row: (str(row["created_at"]), str(row["game_id"])))
