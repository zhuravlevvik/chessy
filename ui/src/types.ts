export type Color = "white" | "black";

export interface ModelInfo {
  id: string;
  name: string;
  checksum: string;
  architecture: string;
  untrained: boolean;
}

export interface MoveInfo {
  ply: number;
  uci: string;
  san: string;
  human: boolean;
}

export interface GameState {
  game_id: string;
  sequence: number;
  status: "waiting" | "active" | "bot_thinking" | "finished";
  fen: string;
  turn: Color;
  human_color: Color;
  bot_color: Color;
  result: string;
  termination: string | null;
  moves: MoveInfo[];
  legal_moves: string[];
  clocks: Record<Color, number | null>;
  server_monotonic: number;
  model: ModelInfo;
  profile: "fast" | "normal" | "deep";
  simulations: number;
  time_control: string;
  feedback_opt_in: boolean;
  feedback_saved: boolean;
}

export interface ServerEvent {
  version: "play-ws-v1";
  type: string;
  sequence: number;
  payload: GameState | { state?: GameState; code?: string; message?: string };
}

export interface CreateGamePayload {
  model_id: string;
  color: "white" | "black" | "random";
  time_control: "untimed" | "3+2" | "5+0" | "10+0" | "15+10";
  profile: "fast" | "normal" | "deep";
  feedback_opt_in: boolean;
}

export interface ObserverFrame { ply: number; fen: string; uci: string | null; san: string | null; }
export interface ObserverGame {
  id: string; run_id: string; kind: "live" | "archive"; status: "playing" | "complete";
  generation: number; game_index: number; model_checksum: string; initial_fen: string; fen: string;
  result: string; termination: string | null; plies?: number; frames?: ObserverFrame[];
}
