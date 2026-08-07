from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import chess
import numpy as np
from fastapi.testclient import TestClient

from chessy.api import ModelRuntime, SessionRegistry, create_app
from chessy.encoding import ACTION_SIZE, encode_move
from chessy.mcts import Evaluation
from chessy.play import ModelInfo


class UniformEvaluator:
    def evaluate(self, history: Sequence[chess.Board]) -> Evaluation:
        board = history[0]
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        for move in board.legal_moves:
            policy[encode_move(board, move)] = 1
        policy /= policy.sum()
        return Evaluation(policy, 0)


def make_client(tmp_path: Path, *, static: bool = False) -> tuple[TestClient, SessionRegistry]:
    info = ModelInfo("random-untrained-seed-0", "Random untrained (seed 0)", "0" * 64, untrained=True, random_seed=0)
    registry = SessionRegistry(
        [ModelRuntime(info, UniformEvaluator())],
        feedback_dir=tmp_path / "feedback",
        observer_runs_dir=tmp_path / "runs",
        simulations_override=2,
    )
    static_dir = None
    if static:
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<main>Chessy</main>")
    return TestClient(create_app(registry, static_dir=static_dir)), registry


def create_game(client: TestClient, **changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_id": "random-untrained-seed-0",
        "color": "white",
        "time_control": "untimed",
        "profile": "fast",
        "feedback_opt_in": False,
    }
    payload.update(changes)
    response = client.post("/api/games", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_health_models_validation_and_static(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, static=True)
    with client:
        assert client.get("/api/health").json() == {"status": "ok", "version": "play-api-v1"}
        model = client.get("/api/models").json()["models"][0]
        assert model["untrained"] is True
        assert "path" not in model
        invalid = client.post("/api/games", json={"model_id": "missing", "fen": chess.STARTING_FEN})
        assert invalid.status_code == 422
        assert "Chessy" in client.get("/").text


def test_discovered_model_is_listed_and_loaded_only_when_selected(tmp_path: Path) -> None:
    initial = ModelInfo("random-untrained-seed-0", "Random", "0" * 64, untrained=True)
    discovered = ModelInfo("candidate-abc", "Candidate · step 250", "a" * 64)
    loaded: list[Path] = []
    source = tmp_path / "candidate"
    def loader(path: Path, info: ModelInfo) -> ModelRuntime:
        loaded.append(path); return ModelRuntime(info, UniformEvaluator())
    registry = SessionRegistry([ModelRuntime(initial, UniformEvaluator())], feedback_dir=tmp_path / "feedback", model_catalog=lambda: [(discovered, source)], model_loader=loader, simulations_override=1)
    client = TestClient(create_app(registry))
    with client:
        models = client.get("/api/models").json()["models"]
        assert [model["id"] for model in models] == [discovered.id, initial.id]
        assert loaded == []
        response = client.post("/api/games", json={"model_id": discovered.id, "color": "white", "time_control": "untimed", "profile": "fast", "feedback_opt_in": False})
        assert response.status_code == 201
        assert loaded == [source]


def test_websocket_authoritative_flow_reconnect_errors_pgn_and_feedback(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        game = create_game(client, feedback_opt_in=False)
        game_id = game["game_id"]
        assert game["feedback_opt_in"] is False
        with client.websocket_connect(f"/api/games/{game_id}/ws") as websocket:
            initial = websocket.receive_json()
            assert initial["type"] == "state"
            websocket.send_json({"version": "play-ws-v1", "type": "unknown", "payload": {}})
            error = websocket.receive_json()
            assert error["type"] == "error" and error["payload"]["code"] == "invalid_message"
            websocket.send_json({"version": "play-ws-v1", "type": "move", "payload": {"uci": "e2e4"}})
            events = [websocket.receive_json() for _ in range(4)]
            assert [event["type"] for event in events] == ["move_applied", "bot_thinking", "move_applied", "state"]
            sequences = [initial["sequence"], error["sequence"], *(event["sequence"] for event in events)]
            assert sequences == sorted(set(sequences))
            websocket.send_json({"version": "play-ws-v1", "type": "offer_draw", "payload": {}})
            assert websocket.receive_json()["type"] == "draw_declined"
            websocket.send_json({"version": "play-ws-v1", "type": "resign", "payload": {}})
            assert websocket.receive_json()["type"] == "game_over"

        with client.websocket_connect(f"/api/games/{game_id}/ws") as websocket:
            assert websocket.receive_json()["type"] == "state"
        state = client.get(f"/api/games/{game_id}").json()
        assert state["status"] == "finished"
        pgn = client.get(f"/api/games/{game_id}/pgn")
        assert pgn.status_code == 200 and "e4" in pgn.text
        declined = client.post(f"/api/games/{game_id}/feedback", json={"confirm": False})
        assert declined.status_code == 422
        saved = client.post(f"/api/games/{game_id}/feedback", json={"confirm": True})
        assert saved.json()["saved"] is True
        assert client.get(f"/api/games/{game_id}").json()["feedback_opt_in"] is True
        assert len(list((tmp_path / "feedback" / game_id).iterdir())) == 3


def test_black_game_starts_bot_outside_request_loop(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        game = create_game(client, color="black")
        game_id = game["game_id"]
        with client.websocket_connect(f"/api/games/{game_id}/ws") as websocket:
            state = websocket.receive_json()
            assert state["type"] == "state"
            # The bot task may already have completed; either authoritative state is valid.
            assert state["payload"]["human_color"] == "black"


def test_training_observer_api_lists_live_and_archive(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from chessy.observer import TrainingObserver

    run = tmp_path / "runs" / "run-one"; observer = TrainingObserver(run, enabled=True, archive_every_generations=1, live_game_index=0)
    observer.live_update({"status": "playing", "generation": 2, "game_index": 0, "model_checksum": "a" * 64, "initial_fen": chess.STARTING_FEN, "fen": chess.STARTING_FEN, "result": "*", "termination": None, "frames": [{"ply": 0, "fen": chess.STARTING_FEN, "uci": None, "san": None}]})
    game = SimpleNamespace(sealed=SimpleNamespace(game_index=0, generation=2, pgn='[Result "1-0"]\n\n1. e4 e5 1-0\n', result="1-0", termination="fixture"))
    observer.archive(game, "a" * 64)
    client, _ = make_client(tmp_path)
    with client:
        games = client.get("/api/observer/games").json()["games"]
        assert {item["kind"] for item in games} == {"live", "archive"}
        assert "frames" not in games[0] and games[0]["plies"] == 2
        detail = client.get(f"/api/observer/games/{games[0]['id']}")
        assert detail.status_code == 200 and detail.json()["frames"][-1]["san"] == "e5"
        assert client.get("/api/observer/games/../../etc").status_code == 404
