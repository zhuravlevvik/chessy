import { useEffect, useRef, useState } from "react";
import { createGame, getModels, getObserverGame, getObserverGames, saveFeedback } from "./api";
import { GameView } from "./components/GameView";
import { StartScreen } from "./components/StartScreen";
import { ObserverView } from "./components/ObserverView";
import type { CreateGamePayload, GameState, ModelInfo, ObserverGame, ServerEvent } from "./types";

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
  const [observing, setObserving] = useState(false); const [observerGames, setObserverGames] = useState<ObserverGame[]>([]); const [observed, setObserved] = useState<ObserverGame | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const lastSequence = useRef(0);

  useEffect(() => { Promise.all([getModels().then(setModels), getObserverGames().then(setObserverGames)]).catch((reason) => setError(reason.message)).finally(() => setLoading(false)); }, []);
  useEffect(() => { if (!observing) return; const refresh = async () => { const games = await getObserverGames(); setObserverGames(games); const live = observed?.kind === "live" ? observed.id : games.find((item) => item.kind === "live")?.id; if (live) setObserved(await getObserverGame(live)); }; refresh().catch(() => {}); const timer = window.setInterval(() => refresh().catch(() => {}), 1000); return () => window.clearInterval(timer); }, [observing, observed?.id, observed?.kind]);

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

  if (observing) return <ObserverView games={observerGames} selected={observed} onSelect={(id) => getObserverGame(id).then(setObserved).catch((reason) => setError(reason.message))} onBack={() => { setObserving(false); setObserved(null); }} />;
  if (!game) return <StartScreen models={models} loading={loading} error={error} onStart={start} observerCount={observerGames.length} onObserve={() => setObserving(true)} />;
  return <GameView
    state={game} connection={connection} error={error}
    onMove={(uci) => send("move", { uci })} onResign={() => send("resign")} onDraw={() => send("offer_draw")}
    onNew={() => { socket.current?.close(); socket.current = null; setGame(null); setError(null); }}
    feedbackSaving={feedbackSaving} feedbackDeclined={feedbackDeclined} onDeclineFeedback={() => setFeedbackDeclined(true)}
    onSaveFeedback={async () => { setFeedbackSaving(true); try { await saveFeedback(game.game_id); setGame({ ...game, feedback_saved: true }); } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось сохранить"); } finally { setFeedbackSaving(false); } }}
  />;
}

export { stateFromEvent };
