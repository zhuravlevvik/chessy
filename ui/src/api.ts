import type { CreateGamePayload, GameState, ModelInfo } from "./types";

async function checked<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Ошибка сервера" }));
    throw new Error(body.detail ?? "Ошибка сервера");
  }
  return response.json() as Promise<T>;
}

export async function getModels(): Promise<ModelInfo[]> {
  const response = await fetch("/api/models");
  return (await checked<{ models: ModelInfo[] }>(response)).models;
}

export async function createGame(payload: CreateGamePayload): Promise<GameState> {
  return checked<GameState>(await fetch("/api/games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }));
}

export async function saveFeedback(gameId: string): Promise<void> {
  await checked(await fetch(`/api/games/${gameId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm: true }),
  }));
}

export function pgnUrl(gameId: string): string {
  return `/api/games/${gameId}/pgn`;
}
