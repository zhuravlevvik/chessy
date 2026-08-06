from pathlib import Path
from types import SimpleNamespace

import chess
import pytest

from chessy.observer import TrainingObserver, discover_observer_games, observer_game


def test_live_updates_and_periodic_archive_are_replayable(tmp_path: Path) -> None:
    observer = TrainingObserver(tmp_path / "runs" / "run", enabled=True, archive_every_generations=2, live_game_index=0)
    observer.live_update({"status": "playing", "generation": 2, "game_index": 0, "model_checksum": "b" * 64, "initial_fen": chess.STARTING_FEN, "fen": chess.STARTING_FEN, "result": "*", "termination": None, "frames": [{"ply": 0, "fen": chess.STARTING_FEN, "uci": None, "san": None}]})
    sealed = SimpleNamespace(game_index=0, generation=2, pgn='[Result "1/2-1/2"]\n\n1. Nf3 Nf6 1/2-1/2\n', result="1/2-1/2", termination="fixture")
    path = observer.archive(SimpleNamespace(sealed=sealed), "b" * 64)
    assert path is not None and path.is_file()
    games = discover_observer_games(tmp_path / "runs")
    assert {game["kind"] for game in games} == {"live", "archive"}
    archived = next(game for game in games if game["kind"] == "archive")
    assert archived["frames"][-1]["san"] == "Nf6"
    assert observer_game(tmp_path / "runs", archived["id"])["result"] == "1/2-1/2"
    (path.parent / "game.pgn").write_text("corrupt", encoding="utf-8")
    assert all(game["kind"] != "archive" for game in discover_observer_games(tmp_path / "runs"))
    with pytest.raises(FileExistsError, match="PGN is missing or corrupt"):
        observer.archive(SimpleNamespace(sealed=sealed), "b" * 64)


def test_disabled_or_non_selected_games_leave_no_artifacts(tmp_path: Path) -> None:
    disabled = TrainingObserver(tmp_path / "disabled", enabled=False, archive_every_generations=1, live_game_index=0)
    disabled.live_update({"game_index": 0})
    selected = TrainingObserver(tmp_path / "selected", enabled=True, archive_every_generations=1, live_game_index=1)
    selected.live_update({"game_index": 0})
    assert not (tmp_path / "disabled").exists()
    assert not (tmp_path / "selected").exists()


def test_completed_live_game_is_updated_between_archive_generations(tmp_path: Path) -> None:
    observer = TrainingObserver(tmp_path / "runs" / "run", enabled=True, archive_every_generations=5, live_game_index=0)
    sealed = SimpleNamespace(game_index=0, generation=2, pgn='[Result "1-0"]\n\n1. e4 1-0\n', result="1-0", termination="fixture")
    assert observer.archive(SimpleNamespace(sealed=sealed), "c" * 64) is None
    games = discover_observer_games(tmp_path / "runs")
    assert len(games) == 1
    assert games[0]["kind"] == "live"
    assert games[0]["status"] == "complete"
    assert games[0]["result"] == "1-0"
