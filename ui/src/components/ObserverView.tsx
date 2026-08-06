import { useEffect, useMemo, useState } from "react";
import { Chessboard } from "react-chessboard";
import type { ObserverGame } from "../types";

interface Props { games: ObserverGame[]; selected: ObserverGame | null; onSelect: (id: string) => void; onBack: () => void; }

export function ObserverView({ games, selected, onSelect, onBack }: Props) {
  const [ply, setPly] = useState(0); const [playing, setPlaying] = useState(false);
  const frames = selected?.frames ?? [];
  useEffect(() => { setPly(selected?.kind === "live" ? Math.max(0, frames.length - 1) : 0); setPlaying(false); }, [selected?.id]);
  useEffect(() => { if (selected?.kind === "live") setPly(Math.max(0, frames.length - 1)); }, [selected?.kind, frames.length]);
  useEffect(() => {
    if (!playing || ply >= frames.length - 1) { if (ply >= frames.length - 1) setPlaying(false); return; }
    const timer = window.setTimeout(() => setPly((value) => value + 1), 650); return () => window.clearTimeout(timer);
  }, [playing, ply, frames.length]);
  const frame = frames[Math.min(ply, Math.max(0, frames.length - 1))];
  const moves = useMemo(() => frames.slice(1), [frames]);
  return <main className="observer-shell">
    <header className="game-header"><div className="mini-brand"><span>C</span><div><strong>CHESSY</strong><small>TRAINING OBSERVER</small></div></div><button className="observer-button" onClick={onBack}>Вернуться к игре</button></header>
    <aside className="observer-list"><p className="eyebrow">ПАРТИИ ОБУЧЕНИЯ</p>{!games.length && <p>Архив появится после первой self-play партии.</p>}{games.map((game) => <button className={selected?.id === game.id ? "selected" : ""} key={game.id} onClick={() => onSelect(game.id)}><strong>{game.kind === "live" ? (game.status === "playing" ? "● LIVE" : "Последняя live") : `Поколение ${game.generation}`}</strong><small>{game.run_id}<br />{game.result} · {game.plies ?? 0} полуходов</small></button>)}</aside>
    <section className="observer-board"><div className="observer-title"><div><p className="eyebrow">{selected?.kind === "live" ? (selected.status === "playing" ? "ИДЁТ СЕЙЧАС" : "ПОСЛЕДНЯЯ LIVE") : "АРХИВ"}</p><h1>{selected ? `Поколение ${selected.generation}` : "Выберите партию"}</h1></div>{selected && <span>{selected.status === "playing" ? "Боты думают…" : `${selected.result} · ${selected.termination}`}</span>}</div>
      {frame ? <><div className="spectator-board"><Chessboard options={{ id: "observer-board", position: frame.fen, allowDragging: false, darkSquareStyle: { backgroundColor: "#52634d" }, lightSquareStyle: { backgroundColor: "#d6d2bd" } }} /></div><div className="observer-controls"><button onClick={() => setPly(0)}>«</button><button onClick={() => setPly(Math.max(0, ply - 1))}>‹</button><button onClick={() => setPlaying(!playing)}>{playing ? "Пауза" : "Смотреть"}</button><button onClick={() => setPly(Math.min(frames.length - 1, ply + 1))}>›</button><button onClick={() => setPly(frames.length - 1)}>»</button><span>Ход {ply}/{Math.max(0, frames.length - 1)} {frame.san ? `· ${frame.san}` : ""}</span></div><div className="observer-moves">{moves.map((move) => <button className={move.ply === ply ? "active" : ""} key={move.ply} onClick={() => setPly(move.ply)}>{move.ply}. {move.san}</button>)}</div></> : <div className="observer-empty">Выберите live-партию или поколение слева.</div>}
    </section>
  </main>;
}
