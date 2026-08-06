import { useEffect, useRef, useState } from "react";
import { createGame, getModels, saveFeedback } from "./api";
import { GameView } from "./components/GameView";
import { StartScreen } from "./components/StartScreen";
import type { CreateGamePayload, GameState, ModelInfo, ServerEvent } from "./types";

function stateFromEvent(event: ServerEvent): GameState | null {
  if ("state" in event.payload && event.payload.state) return event.payload.state;
  if ("game_id" in event.payload) return event.payload as GameState;
  return null;
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [game, setGame] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connection, setConnection] = useState<"connected" | "reconnecting" | "disconnected">("disconnected");
  const [feedbackSaving, setFeedbackSaving] = useState(false);
  const [feedbackDeclined, setFeedbackDeclined] = useState(false);
  const socket = useRef<WebSocket | null>(null);
  const lastSequence = useRef(0);

  useEffect(() => { getModels().then(setModels).catch((reason) => setError(reason.message)).finally(() => setLoading(false)); }, []);

  function connect(gameId: string) {
    socket.current?.close();
    setConnection("reconnecting");
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${location.host}/api/games/${gameId}/ws`);
    socket.current = ws;
    ws.onopen = () => setConnection("connected");
    ws.onmessage = ({ data }) => {
      const event = JSON.parse(data) as ServerEvent;
      if (event.sequence <= lastSequence.current) return;
      lastSequence.current = event.sequence;
      const next = stateFromEvent(event);
      if (next) setGame(next);
      if (event.type === "error" || event.type === "draw_declined") setError("message" in event.payload ? event.payload.message ?? null : null);
      else setError(null);
    };
    ws.onclose = () => {
      setConnection("disconnected");
      if (socket.current === ws) window.setTimeout(() => connect(gameId), 1200);
    };
  }

  async function start(payload: CreateGamePayload) {
    setLoading(true); setError(null); setFeedbackDeclined(false); lastSequence.current = 0;
    try { const next = await createGame(payload); setGame(next); connect(next.game_id); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось начать партию"); }
    finally { setLoading(false); }
  }

  function send(type: string, payload: object = {}) {
    if (socket.current?.readyState !== WebSocket.OPEN) { setError("Нет соединения с сервером"); return; }
    socket.current.send(JSON.stringify({ version: "play-ws-v1", type, payload }));
  }

  if (!game) return <StartScreen models={models} loading={loading} error={error} onStart={start} />;
  const visibleGame = feedbackDeclined ? { ...game, feedback_opt_in: false } : game;
  return <GameView
    state={visibleGame} connection={connection} error={error}
    onMove={(uci) => send("move", { uci })} onResign={() => send("resign")} onDraw={() => send("offer_draw")}
    onNew={() => { socket.current?.close(); socket.current = null; setGame(null); setError(null); }}
    feedbackSaving={feedbackSaving} onDeclineFeedback={() => setFeedbackDeclined(true)}
    onSaveFeedback={async () => { setFeedbackSaving(true); try { await saveFeedback(game.game_id); setGame({ ...game, feedback_saved: true }); } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось сохранить"); } finally { setFeedbackSaving(false); } }}
  />;
}

export { stateFromEvent };
