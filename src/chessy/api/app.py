"""Same-origin FastAPI/REST/WebSocket application for local play."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from chessy.api.schemas import ClientEnvelope, CreateGameRequest, EmptyPayload, FeedbackRequest, MovePayload
from chessy.api.sessions import SessionRegistry
from chessy.play import GameSession, save_human_feedback
from chessy.observer import discover_observer_games, observer_game

MAX_WS_PAYLOAD_BYTES = 16 * 1024


class SessionCoordinator:
    def __init__(self) -> None:
        self.connections: dict[str, set[WebSocket]] = defaultdict(set)
        self.sequences: dict[str, int] = defaultdict(int)
        self.bot_tasks: dict[str, asyncio.Task[None]] = {}
        self.deadline_tasks: dict[str, asyncio.Task[None]] = {}

    def event(self, session: GameSession, event_type: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        self.sequences[session.id] += 1
        return {
            "version": "play-ws-v1",
            "type": event_type,
            "sequence": self.sequences[session.id],
            "payload": payload if payload is not None else session.snapshot(),
        }

    async def send(self, websocket: WebSocket, session: GameSession, event_type: str, payload: dict[str, object] | None = None) -> None:
        await websocket.send_json(self.event(session, event_type, payload))

    async def broadcast(self, session: GameSession, event_type: str, payload: dict[str, object] | None = None) -> None:
        message = self.event(session, event_type, payload)
        stale: list[WebSocket] = []
        for websocket in tuple(self.connections[session.id]):
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.connections[session.id].discard(websocket)

    def schedule_deadline(self, session: GameSession) -> None:
        old = self.deadline_tasks.pop(session.id, None)
        if old is not None:
            old.cancel()
        delay = session.deadline_seconds()
        if delay is None:
            return

        async def expire() -> None:
            try:
                await asyncio.sleep(delay)
                if await asyncio.to_thread(session.expire_if_needed):
                    await self.broadcast(session, "game_over")
            except asyncio.CancelledError:
                pass

        self.deadline_tasks[session.id] = asyncio.create_task(expire())

    def schedule_bot(self, session: GameSession) -> None:
        existing = self.bot_tasks.get(session.id)
        if existing is not None and not existing.done():
            return

        async def play() -> None:
            await self.broadcast(session, "bot_thinking")
            try:
                result = await asyncio.to_thread(session.play_bot_move)
                if result is not None:
                    record, _ = result
                    await self.broadcast(session, "move_applied", {"move": record.uci, "state": session.snapshot()})
                if session.status == "finished":
                    await self.broadcast(session, "game_over")
                else:
                    await self.broadcast(session, "state")
            except Exception:
                await self.broadcast(session, "error", {"code": "bot_error", "message": "Chessy could not complete its move."})
            finally:
                self.schedule_deadline(session)

        self.bot_tasks[session.id] = asyncio.create_task(play())

    async def close(self) -> None:
        tasks = list(self.bot_tasks.values()) + list(self.deadline_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def create_app(registry: SessionRegistry, *, static_dir: Path | None = None) -> FastAPI:
    coordinator = SessionCoordinator()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await coordinator.close()

    app = FastAPI(title="Chessy", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.registry = registry
    app.state.coordinator = coordinator

    def get_session(game_id: str) -> GameSession:
        try:
            return registry.get(game_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="game not found") from exc

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "version": "play-api-v1"}

    @app.get("/api/models")
    async def models() -> dict[str, object]:
        return {"models": await asyncio.to_thread(registry.public_models)}

    @app.get("/api/observer/games")
    async def observer_games() -> dict[str, object]:
        if registry.observer_runs_dir is None: return {"games": []}
        games = await asyncio.to_thread(discover_observer_games, registry.observer_runs_dir)
        summaries = [{key: value for key, value in game.items() if key != "frames"} | {"plies": max(0, len(game.get("frames", [])) - 1)} for game in games]
        return {"games": summaries}

    @app.get("/api/observer/games/{game_id}")
    async def observer_game_detail(game_id: str) -> dict[str, object]:
        if registry.observer_runs_dir is None: raise HTTPException(status_code=404, detail="observer game not found")
        try: return await asyncio.to_thread(observer_game, registry.observer_runs_dir, game_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="observer game not found") from exc

    @app.post("/api/games", status_code=201)
    async def create_game(request: CreateGameRequest) -> dict[str, object]:
        try:
            session = await asyncio.to_thread(registry.create,
                model_id=request.model_id,
                color=request.color,
                time_control=request.time_control,
                profile=request.profile,
                feedback_opt_in=request.feedback_opt_in,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        coordinator.schedule_deadline(session)
        if session.status == "bot_thinking":
            coordinator.schedule_bot(session)
        return session.snapshot()

    @app.get("/api/games/{game_id}")
    async def game_state(game_id: str) -> dict[str, object]:
        return get_session(game_id).snapshot()

    @app.get("/api/games/{game_id}/pgn")
    async def game_pgn(game_id: str) -> PlainTextResponse:
        session = get_session(game_id)
        return PlainTextResponse(
            session.pgn(),
            media_type="application/x-chess-pgn",
            headers={"Content-Disposition": f'attachment; filename="chessy-{session.id}.pgn"'},
        )

    @app.post("/api/games/{game_id}/feedback")
    async def feedback(game_id: str, request: FeedbackRequest) -> dict[str, object]:
        session = get_session(game_id)
        try:
            await asyncio.to_thread(save_human_feedback, session, registry.feedback_dir, confirmed=request.confirm)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"saved": True, "game_id": session.id}

    @app.websocket("/api/games/{game_id}/ws")
    async def game_socket(websocket: WebSocket, game_id: str) -> None:
        try:
            session = registry.get(game_id)
        except KeyError:
            await websocket.close(code=4404, reason="game not found")
            return
        await websocket.accept()
        coordinator.connections[game_id].add(websocket)
        await coordinator.send(websocket, session, "state")
        try:
            while True:
                raw = await websocket.receive_text()
                if len(raw.encode("utf-8")) > MAX_WS_PAYLOAD_BYTES:
                    await coordinator.send(websocket, session, "error", {"code": "payload_too_large", "message": "Message is too large."})
                    continue
                try:
                    envelope = ClientEnvelope.model_validate_json(raw)
                    if envelope.type == "move":
                        payload = MovePayload.model_validate(envelope.payload)
                        record = await asyncio.to_thread(session.apply_human_move, payload.uci)
                        await coordinator.broadcast(session, "move_applied", {"move": record.uci, "state": session.snapshot()})
                        coordinator.schedule_deadline(session)
                        if session.status == "bot_thinking":
                            coordinator.schedule_bot(session)
                    elif envelope.type == "resign":
                        EmptyPayload.model_validate(envelope.payload)
                        await asyncio.to_thread(session.resign)
                        coordinator.schedule_deadline(session)
                        await coordinator.broadcast(session, "game_over")
                    elif envelope.type == "offer_draw":
                        EmptyPayload.model_validate(envelope.payload)
                        await asyncio.to_thread(session.offer_draw)
                        await coordinator.broadcast(session, "draw_declined", {"message": "Chessy declines the draw offer."})
                    else:
                        EmptyPayload.model_validate(envelope.payload)
                        await coordinator.send(websocket, session, "pong", {})
                except (ValidationError, json.JSONDecodeError):
                    await coordinator.send(websocket, session, "error", {"code": "invalid_message", "message": "Invalid WebSocket message."})
                except TimeoutError:
                    await coordinator.broadcast(session, "game_over")
                except ValueError:
                    await coordinator.send(websocket, session, "error", {"code": "illegal_move", "message": "That move is not legal in the current position."})
                except RuntimeError as exc:
                    code = "bot_thinking" if "thinking" in str(exc) else "invalid_state"
                    await coordinator.send(websocket, session, "error", {"code": code, "message": "The game is not ready for that action."})
        except WebSocketDisconnect:
            pass
        finally:
            coordinator.connections[game_id].discard(websocket)

    if static_dir is not None:
        static_dir = Path(static_dir)
        assets = static_dir / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def frontend(path: str) -> FileResponse:
            index = static_dir / "index.html"
            if not index.is_file():
                raise HTTPException(status_code=503, detail="frontend is not built")
            return FileResponse(index)

    return app
