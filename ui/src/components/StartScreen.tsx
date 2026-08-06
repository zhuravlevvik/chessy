import { useEffect, useState, type FormEvent } from "react";
import type { CreateGamePayload, ModelInfo } from "../types";

interface Props {
  models: ModelInfo[];
  loading: boolean;
  error: string | null;
  onStart: (payload: CreateGamePayload) => void;
}

export function StartScreen({ models, loading, error, onStart }: Props) {
  const [modelId, setModelId] = useState("");
  useEffect(() => { if (!modelId && models[0]) setModelId(models[0].id); }, [modelId, models]);
  const selectedModel = models.find((model) => model.id === modelId) ?? models[0];

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    onStart({
      model_id: String(data.get("model_id")),
      color: String(data.get("color")) as CreateGamePayload["color"],
      time_control: String(data.get("time_control")) as CreateGamePayload["time_control"],
      profile: String(data.get("profile")) as CreateGamePayload["profile"],
      feedback_opt_in: data.get("feedback_opt_in") === "on",
    });
  }

  return <main className="start-shell">
    <section className="hero">
      <div className="brand-mark" aria-hidden="true">C</div>
      <p className="eyebrow">PERSONAL CHESS LAB</p>
      <h1>Сыграем партию?</h1>
      <p className="lead">Chessy думает через нейросеть и MCTS. Правила, часы и результат всегда контролирует локальный сервер.</p>
    </section>
    <form className="start-card" onSubmit={submit}>
      <label>Модель
        <select name="model_id" required disabled={!models.length} value={modelId} onChange={(event) => setModelId(event.target.value)}>
          {models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
        </select>
      </label>
      {selectedModel?.untrained && <div className="warning" role="status">
        <span>НЕ ОБУЧЕНА</span>
        Случайная сеть знает только правила через MCTS. Это первая точка отсчёта, не заявка на шахматную силу.
      </div>}
      <div className="field-grid">
        <label>Ваш цвет
          <select name="color" defaultValue="white">
            <option value="white">Белые</option><option value="black">Чёрные</option><option value="random">Случайно</option>
          </select>
        </label>
        <label>Контроль
          <select name="time_control" defaultValue="untimed">
            <option value="untimed">Без часов</option><option value="3+2">3 + 2</option><option value="5+0">5 + 0</option><option value="10+0">10 + 0</option><option value="15+10">15 + 10</option>
          </select>
        </label>
      </div>
      <fieldset>
        <legend>Глубина поиска</legend>
        <div className="segments">
          <label><input type="radio" name="profile" value="fast" /><span>Быстро<small>32</small></span></label>
          <label><input type="radio" name="profile" value="normal" defaultChecked /><span>Нормально<small>128</small></span></label>
          <label><input type="radio" name="profile" value="deep" /><span>Глубоко<small>512</small></span></label>
        </div>
      </fieldset>
      <label className="check-row"><input name="feedback_opt_in" type="checkbox" />
        <span><strong>Добавить мои ходы в обучение</strong><small>После партии Chessy ещё раз спросит подтверждение.</small></span>
      </label>
      {error && <p className="form-error" role="alert">{error}</p>}
      <button className="primary" disabled={loading || !models.length}>{loading ? "Подготавливаю доску…" : "Начать партию"}</button>
    </form>
  </main>;
}
