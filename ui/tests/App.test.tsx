import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App, { stateFromEvent } from "../src/App";
import { GameView } from "../src/components/GameView";
import type { GameState } from "../src/types";

const boardCapture = vi.hoisted(() => ({ options: null as any }));
vi.mock("react-chessboard", () => ({ Chessboard: ({ options }: { options: unknown }) => { boardCapture.options = options; return <div data-testid="board" />; } }));

const model = { id: "random-untrained-seed-0", name: "Random", checksum: "random", architecture: "residual-cnn-v1", untrained: true };
const state: GameState = {
  game_id: "game-1", sequence: 0, status: "active", fen: "start", turn: "white", human_color: "white", bot_color: "black",
  result: "*", termination: null, moves: [], legal_moves: ["e2e4"], clocks: { white: null, black: null }, server_monotonic: 0,
  model, profile: "normal", simulations: 128, time_control: "untimed", feedback_opt_in: false, feedback_saved: false,
};

class FakeWebSocket {
  static OPEN = 1; readyState = 1; onopen: (() => void) | null = null; onmessage: ((event: { data: string }) => void) | null = null; onclose: (() => void) | null = null;
  constructor(public url: string) { setTimeout(() => this.onopen?.(), 0); }
  send = vi.fn(); close = vi.fn();
}

describe("Chessy UI", () => {
  beforeEach(() => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      if (url === "/api/models") return new Response(JSON.stringify({ models: [model] }), { status: 200 });
      if (url === "/api/observer/games") return new Response(JSON.stringify({ games: [] }), { status: 200 });
      if (url === "/api/games" && init?.method === "POST") return new Response(JSON.stringify(state), { status: 201 });
      return new Response(JSON.stringify({ saved: true }), { status: 200 });
    }));
  });

  it("shows the start form, random warning, and feedback unchecked", async () => {
    render(<App />);
    expect(await screen.findByText("НЕ ОБУЧЕНА")).toBeVisible();
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    expect(screen.getByText("Начать партию")).toBeEnabled();
    expect(screen.getByText("Смотреть обучение")).toBeEnabled();
  });

  it("opens the training observer and renders a live position", async () => {
    const live = { id: "run-live", run_id: "run", kind: "live", status: "playing", generation: 3, game_index: 0, model_checksum: "a".repeat(64), initial_fen: "start", fen: "after-e4", result: "*", termination: null, plies: 1 };
    vi.mocked(fetch).mockImplementation(async (url: string) => {
      if (url === "/api/models") return new Response(JSON.stringify({ models: [model] }), { status: 200 });
      if (url === "/api/observer/games") return new Response(JSON.stringify({ games: [live] }), { status: 200 });
      if (url === "/api/observer/games/run-live") return new Response(JSON.stringify({ ...live, frames: [{ ply: 0, fen: "start", uci: null, san: null }, { ply: 1, fen: "after-e4", uci: "e2e4", san: "e4" }] }), { status: 200 });
      return new Response("{}", { status: 404 });
    });
    render(<App />);
    fireEvent.click(await screen.findByText("Смотреть обучение · 1"));
    expect(await screen.findByText("● LIVE")).toBeVisible();
    await waitFor(() => expect(screen.getByText(/Ход 1\/1/)).toBeVisible());
    expect(boardCapture.options.position).toBe("after-e4");
  });

  it("sends the exact create-game payload", async () => {
    render(<App />);
    fireEvent.click(await screen.findByText("Начать партию"));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith("/api/games", expect.objectContaining({ method: "POST" })));
    const request = vi.mocked(fetch).mock.calls.find(([url]) => url === "/api/games")![1]!;
    expect(JSON.parse(String(request.body))).toEqual({ model_id: model.id, color: "white", time_control: "untimed", profile: "normal", feedback_opt_in: false });
  });

  it("renders thinking, disconnected, game-over, PGN and feedback confirmation states", () => {
    const props = { connection: "disconnected" as const, error: null, onMove: vi.fn(), onResign: vi.fn(), onDraw: vi.fn(), onNew: vi.fn(), onSaveFeedback: vi.fn(), onDeclineFeedback: vi.fn(), feedbackSaving: false };
    const { rerender } = render(<GameView {...props} state={{ ...state, status: "bot_thinking", turn: "black" }} />);
    expect(screen.getByText("Chessy считает варианты")).toBeVisible();
    expect(screen.getByText("Нет связи")).toBeVisible();
    rerender(<GameView {...props} state={{ ...state, status: "finished", result: "0-1", termination: "resignation", feedback_opt_in: true }} />);
    expect(screen.getByText("Чёрные победили")).toBeVisible();
    expect(screen.getByText("Скачать PGN")).toHaveAttribute("href", "/api/games/game-1/pgn");
    expect(screen.getByText("Сохранить ваши ходы для обучения?")).toBeVisible();
  });

  it("uses authoritative server state and blocks input outside the human turn", () => {
    const authoritative = { ...state, fen: "server-fen", moves: [{ ply: 1, uci: "e2e4", san: "e4", human: true }] };
    expect(stateFromEvent({ version: "play-ws-v1", type: "state", sequence: 4, payload: authoritative })).toEqual(authoritative);
    const onMove = vi.fn();
    render(<GameView state={{ ...state, status: "bot_thinking", turn: "black" }} connection="connected" error={null} onMove={onMove} onResign={vi.fn()} onDraw={vi.fn()} onNew={vi.fn()} onSaveFeedback={vi.fn()} onDeclineFeedback={vi.fn()} feedbackSaving={false} />);
    expect(boardCapture.options.onPieceDrop({ sourceSquare: "e2", targetSquare: "e4", piece: {} })).toBe(false);
    expect(onMove).not.toHaveBeenCalled();
  });

  it("opens promotion choice and sends full promotion UCI", () => {
    const onMove = vi.fn();
    render(<GameView state={{ ...state, fen: "promotion", legal_moves: ["e7e8q", "e7e8r", "e7e8b", "e7e8n"] }} connection="connected" error={null} onMove={onMove} onResign={vi.fn()} onDraw={vi.fn()} onNew={vi.fn()} onSaveFeedback={vi.fn()} onDeclineFeedback={vi.fn()} feedbackSaving={false} />);
    act(() => { expect(boardCapture.options.onPieceDrop({ sourceSquare: "e7", targetSquare: "e8", piece: {} })).toBe(true); });
    fireEvent.click(screen.getByRole("button", { name: "Превращение n" }));
    expect(onMove).toHaveBeenCalledWith("e7e8n");
  });

  it("interpolates the active clock between authoritative server states", () => {
    vi.useFakeTimers();
    const timed = {
      ...state,
      clocks: { white: 120, black: 120 },
      time_control: "3+2",
    };
    render(<GameView state={timed} connection="connected" error={null} onMove={vi.fn()} onResign={vi.fn()} onDraw={vi.fn()} onNew={vi.fn()} onSaveFeedback={vi.fn()} onDeclineFeedback={vi.fn()} feedbackSaving={false} />);
    expect(screen.getAllByText("2:00")).toHaveLength(2);
    act(() => vi.advanceTimersByTime(1100));
    expect(screen.getByText("1:59")).toBeVisible();
    expect(screen.getByText("2:00")).toBeVisible();
    vi.useRealTimers();
  });
});
