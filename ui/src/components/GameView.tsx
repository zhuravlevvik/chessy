import { useEffect, useMemo, useRef, useState } from "react";
import { Chessboard } from "react-chessboard";
import type { GameState } from "../types";
import { pgnUrl } from "../api";

interface Props {
  state: GameState;
  connection: "connected" | "reconnecting" | "disconnected";
  error: string | null;
  onMove: (uci: string) => void;
  onResign: () => void;
  onDraw: () => void;
  onNew: () => void;
  onSaveFeedback: () => void;
  onDeclineFeedback: () => void;
  feedbackSaving: boolean;
}

function formatClock(seconds: number | null): string {
  if (seconds === null) return "—:—";
  const safe = Math.max(0, Math.ceil(seconds));
  return `${Math.floor(safe / 60)}:${String(safe % 60).padStart(2, "0")}`;
}

export function GameView(props: Props) {
  const { state } = props;
  const [selected, setSelected] = useState<string | null>(null);
  const [promotion, setPromotion] = useState<{ from: string; to: string; choices: string[] } | null>(null);
  const [clientNow, setClientNow] = useState(() => performance.now());
  const clockBaseline = useRef({
    receivedAt: performance.now(),
    clocks: { ...state.clocks },
    turn: state.turn,
    status: state.status,
  });
  useEffect(() => {
    const interval = window.setInterval(() => setClientNow(performance.now()), 250);
    return () => window.clearInterval(interval);
  }, []);
  useEffect(() => {
    const receivedAt = performance.now();
    clockBaseline.current = {
      receivedAt,
      clocks: { ...state.clocks },
      turn: state.turn,
      status: state.status,
    };
    setClientNow(receivedAt);
  }, [state]);
  useEffect(() => setSelected(null), [state.fen]);

  function visibleClock(color: "white" | "black"): number | null {
    const baseline = clockBaseline.current;
    const remaining = baseline.clocks[color];
    if (remaining === null || baseline.status === "finished" || baseline.turn !== color) {
      return remaining;
    }
    return Math.max(0, remaining - (clientNow - baseline.receivedAt) / 1000);
  }

  const canMove = state.status === "active" && state.turn === state.human_color && props.connection === "connected";
  const destinations = useMemo(() => selected ? state.legal_moves.filter((move) => move.startsWith(selected)).map((move) => move.slice(2, 4)) : [], [selected, state.legal_moves]);
  const squareStyles = useMemo(() => {
    const styles: Record<string, React.CSSProperties> = {};
    if (selected) styles[selected] = { boxShadow: "inset 0 0 0 4px #d9a84e" };
    destinations.forEach((square) => { styles[square] = { background: "radial-gradient(circle, rgba(217,168,78,.8) 0 14%, transparent 16%)" }; });
    return styles;
  }, [selected, destinations]);

  function attempt(from: string, to: string): boolean {
    if (!canMove) return false;
    const candidates = state.legal_moves.filter((move) => move.startsWith(from + to));
    if (!candidates.length) return false;
    if (candidates.some((move) => move.length === 5)) {
      setPromotion({ from, to, choices: candidates.map((move) => move[4]) });
      return true;
    }
    props.onMove(from + to);
    return true;
  }

  const resultText = state.result === "1-0" ? "Белые победили" : state.result === "0-1" ? "Чёрные победили" : "Ничья";
  return <main className="game-shell">
    <header className="game-header">
      <div className="mini-brand"><span>C</span><div><strong>CHESSY</strong><small>LOCAL MATCH</small></div></div>
      <div className={`connection ${props.connection}`}>{props.connection === "connected" ? "Подключено" : props.connection === "reconnecting" ? "Переподключение…" : "Нет связи"}</div>
    </header>
    <section className="board-column">
      <div className={`player-strip ${state.turn === state.bot_color ? "active" : ""}`}>
        <div className="avatar bot">C</div><div><strong>Chessy</strong><small>{state.model.name}</small></div>
        <time>{formatClock(visibleClock(state.bot_color))}</time>
      </div>
      <div className="board-wrap">
        <Chessboard options={{
          id: "chessy-board",
          position: state.fen,
          boardOrientation: state.human_color,
          allowDragging: canMove,
          squareStyles,
          darkSquareStyle: { backgroundColor: "#52634d" },
          lightSquareStyle: { backgroundColor: "#d6d2bd" },
          boardStyle: { borderRadius: 4, boxShadow: "0 24px 65px rgba(0,0,0,.35)" },
          canDragPiece: ({ square }) => Boolean(square && state.legal_moves.some((move) => move.startsWith(square))),
          onPieceDrop: ({ sourceSquare, targetSquare }) => Boolean(targetSquare && attempt(sourceSquare, targetSquare)),
          onSquareClick: ({ square }) => {
            if (!canMove) return;
            if (selected && attempt(selected, square)) setSelected(null);
            else setSelected(state.legal_moves.some((move) => move.startsWith(square)) ? square : null);
          },
        }} />
        {state.status === "bot_thinking" && <div className="thinking"><span /><span /><span /> Chessy считает варианты</div>}
      </div>
      <div className={`player-strip ${state.turn === state.human_color ? "active" : ""}`}>
        <div className="avatar human">ВЫ</div><div><strong>Вы</strong><small>{state.human_color === "white" ? "Белые" : "Чёрные"}</small></div>
        <time>{formatClock(visibleClock(state.human_color))}</time>
      </div>
    </section>
    <aside className="side-panel">
      <div className="match-meta"><span>{state.profile.toUpperCase()} · {state.simulations} симуляций</span><span>{state.time_control}</span></div>
      <h2>{state.status === "finished" ? resultText : state.status === "bot_thinking" ? "Ход Chessy" : "Ваш ход"}</h2>
      {state.status === "finished" && <p className="termination">Причина: {state.termination}</p>}
      {props.error && <p className="form-error" role="alert">{props.error}</p>}
      <div className="moves" aria-label="История ходов">
        {!state.moves.length && <p>Партия только начинается.</p>}
        {Array.from({ length: Math.ceil(state.moves.length / 2) }, (_, index) => <div className="move-row" key={index}>
          <span>{index + 1}.</span><b>{state.moves[index * 2]?.san}</b><b>{state.moves[index * 2 + 1]?.san}</b>
        </div>)}
      </div>
      {state.status !== "finished" ? <div className="actions">
        <button onClick={props.onDraw}>Предложить ничью</button><button className="danger" onClick={props.onResign}>Сдаться</button>
      </div> : <div className="post-game">
        <a className="download" href={pgnUrl(state.game_id)} download>Скачать PGN</a>
        {state.feedback_opt_in && !state.feedback_saved && <div className="feedback-confirm">
          <strong>Сохранить ваши ходы для обучения?</strong><p>Будут записаны только ваши решения; ходы Chessy останутся контекстом.</p>
          <button onClick={props.onSaveFeedback} disabled={props.feedbackSaving}>{props.feedbackSaving ? "Сохраняю…" : "Сохранить"}</button>
          <button onClick={props.onDeclineFeedback}>Не сохранять</button>
        </div>}
        {state.feedback_saved && <p className="success">Ходы сохранены ✓</p>}
        <button className="primary" onClick={props.onNew}>Новая партия</button>
      </div>}
    </aside>
    {promotion && <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Выбор превращения пешки">
      <div className="promotion"><h3>В кого превратить пешку?</h3><div>
        {(["q", "r", "b", "n"] as const).filter((piece) => promotion.choices.includes(piece)).map((piece) => <button key={piece} aria-label={`Превращение ${piece}`} onClick={() => { props.onMove(promotion.from + promotion.to + piece); setPromotion(null); }}>{({ q: "♛", r: "♜", b: "♝", n: "♞" })[piece]}</button>)}
      </div><button onClick={() => setPromotion(null)}>Отмена</button></div>
    </div>}
  </main>;
}

export { formatClock };
