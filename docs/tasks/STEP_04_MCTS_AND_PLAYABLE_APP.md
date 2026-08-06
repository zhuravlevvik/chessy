# Chessy — шаг 4: MCTS и первый playable agent

Статус: готово к реализации
Дата постановки: 2026-08-06
Рабочая директория: `/Users/zhuravlevvikt/Documents/codex_projects/chessy`

## 1. Цель задачи

Собрать первый полный вертикальный срез Chessy: от позиции и случайного либо
загруженного checkpoint до выбранного MCTS хода и локальной партии в браузере.

После выполнения шага пользователь должен запускать одну команду, открывать
локальную страницу и играть полноценную партию против Chessy. Backend остаётся
единственным источником правил, состояния игры, часов, MCTS и PGN.

В рамках шага требуется:

- реализовать версионированный поиск `mcts-puct-v1`;
- подключить `ChessyModel` и legal policy;
- поддержать пакетный inference на MPS/CPU/CUDA;
- реализовать переиспользование дерева после хода;
- создать игровой агент и управляемую backend-сессию партии;
- создать FastAPI/WebSocket API;
- создать React/TypeScript/Vite UI с `react-chessboard`;
- запускать production UI командой `uv run chessy play` только на loopback;
- скачивать PGN;
- опционально сохранять подтверждённые ходы пользователя как
  `chessy-human-feedback-v1`.

Модель в этом шаге не обучается. При отсутствии export пользователь может
играть против детерминированно инициализированной случайной сети; UI должен
честно помечать её как untrained.

## 2. Архитектурные источники

Перед началом полностью прочитать:

- `AGENTS.md`;
- `docs/PROJECT_PLAN.md`, особенно разделы 5–9, 14–15, 19, 20 и 22;
- `docs/decisions/0002-compute-backend.md`;
- `docs/decisions/0003-position-encoding.md`;
- `docs/decisions/0004-action-encoding.md`;
- `docs/decisions/0005-artifact-formats.md`;
- `docs/decisions/0007-play-interface.md`;
- `docs/decisions/0008-mcts-v1.md`;
- текущие `chess`, `encoding` и `model` packages и их тесты;
- `docs/tasks/STEP_03_POLICY_VALUE_MODEL.md`.

Зафиксированные версии контрактов:

- `board119-v1`;
- `az73-v1`;
- `residual-cnn-v1`;
- `chessy-model-v1`;
- `mcts-puct-v1`;
- `chessy-human-feedback-v1` — вводится этим шагом.

## 3. Границы задачи

### 3.1. Обязательно входит

- корректный однопоточный PUCT tree search;
- единый batching inference service для конкурентных запросов;
- direct evaluator для простых тестов и отладки;
- профили `fast`, `normal`, `deep`;
- выбор checkpoint из явно переданных CLI путей;
- fallback на случайную модель;
- партия человека против бота с выбором цвета;
- untimed и базовые часы с increment;
- рокировка, en passant и promotion через существующую среду;
- resignation, draw offer, restart и завершение по правилам;
- PGN download;
- opt-in и post-game confirmation human feedback;
- production build frontend, доступный из Python package.

### 3.2. Не входит

- self-play workers и replay buffer;
- обучение, optimizer и scheduler;
- training snapshot/resume;
- evaluation arena и Elo;
- opening book;
- tablebases;
- multi-user authentication;
- внешний сервер или облачный deployment;
- игра через Chess.com/Lichess API;
- бот-resignation по value;
- сохранение ходов бота как человеческих policy targets.

## 4. Безопасность и обязательные ограничения

1. Сервер слушает только `127.0.0.1`. CLI не должен предоставлять флаг,
   позволяющий поставить `0.0.0.0`.
2. Порт по умолчанию выбирается ОС через bind к порту `0`.
3. Не включать permissive CORS: frontend и API имеют один origin.
4. Backend проверяет каждое действие; browser никогда не является источником
   легальности или результата.
5. UI не принимает произвольный filesystem path. Модели задаются только при
   запуске CLI и затем выбираются по безопасному ID.
6. WebSocket payload имеет ограниченный набор типов и разумный предел размера.
7. Ошибки API не раскрывают абсолютные пути, stack traces или содержимое весов.
8. Human feedback выключен по умолчанию и требует отдельного подтверждения
   после партии.
9. Не писать в `data/human_feedback/`, если пользователь не подтвердил запись.
10. Не изменять существующие raw/quality/personal datasets.
11. Не коммитить веса, node_modules или локальные feedback-партии.
12. Не создавать Git-коммит без отдельного запроса пользователя.

## 5. Зависимости

### 5.1. Python

Добавить через `uv` runtime dependencies:

- `fastapi`;
- `uvicorn[standard]`.

Добавить development dependency:

- `httpx` для FastAPI TestClient.

Использовать совместимые стабильные диапазоны, разрешённые текущим Python 3.12,
и зафиксировать точные версии в `uv.lock`. Не добавлять отдельный chess package,
WebSocket framework или production process manager.

### 5.2. Frontend

В `ui/package.json` нужны runtime dependencies:

- `react`;
- `react-dom`;
- `react-chessboard`.

Development dependencies:

- `typescript`;
- `vite`;
- `@vitejs/plugin-react`;
- `vitest`;
- `jsdom`;
- `@testing-library/react`;
- `@testing-library/jest-dom`;
- React type packages.

Использовать npm и коммитить `package-lock.json`. `ui/node_modules/` остаётся в
`.gitignore`. Не использовать CDN scripts.

Проверенные на машине инструменты на момент постановки:

```text
node v26.5.0
npm 11.17.0
```

Не обновлять глобальные Node/npm/Homebrew packages в рамках задачи.

## 6. Целевая структура

```text
src/chessy/
  mcts/
    __init__.py
    config.py
    node.py
    evaluator.py
    search.py
  play/
    __init__.py
    agent.py
    game.py
    feedback.py
  api/
    __init__.py
    app.py
    schemas.py
    sessions.py
    static/                 # production Vite build, tracked
  cli.py
ui/
  package.json
  package-lock.json
  tsconfig.json
  vite.config.ts
  index.html
  src/
    main.tsx
    App.tsx
    api.ts
    types.ts
    components/
    styles.css
  tests/
scripts/
  build_ui.py              # optional thin reproducible build helper
tests/
  mcts/
  play/
  api/
```

Не создавать training/selfplay/replay modules.

## 7. Конфигурация `mcts-puct-v1`

Создать frozen dataclass `MCTSConfig`:

```python
version: str = "mcts-puct-v1"
simulations: int = 128
c_puct: float = 1.5
temperature: float = 0.0
root_noise: bool = False
dirichlet_alpha: float = 0.3
dirichlet_epsilon: float = 0.25
max_batch_size: int = 32
max_batch_wait_ms: float = 2.0
seed: int = 0
```

Строго валидировать:

- `simulations > 0`;
- `c_puct > 0`;
- `temperature >= 0`;
- `alpha > 0`;
- `epsilon` в `[0,1]`;
- batch size в `[1,32]`;
- wait в `[0,100]` ms;
- неизвестные dict fields запрещены.

Игровые профили:

| Profile | Simulations | Root noise | Temperature |
|---|---:|---:|---:|
| `fast` | 32 | off | 0 |
| `normal` | 128 | off | 0 |
| `deep` | 512 | off | 0 |

В UI по умолчанию выбран `normal`. Root noise в партии с человеком всегда
выключен. Параметры self-play noise присутствуют в config и покрыты unit tests,
но UI не должен включать их случайно.

Resignation бота выключен. На уровне игровой сессии действует защитный предел
300 полуходов; при достижении партия завершается ничьёй с отдельной termination
reason `max_plies`.

## 8. Инварианты дерева и формула PUCT

### 8.1. SearchNode

Каждый узел представляет позицию со стороной `to_play` и хранит:

```python
prior: float
visit_count: int
value_sum: float
to_play: chess.Color
children: dict[int, SearchNode]
```

Action key — индекс `az73-v1`, а не UCI-строка.

`mean_value = value_sum / visit_count` при `visit_count > 0`, иначе `0.0`.
Значение узла всегда хранится относительно стороны, которой принадлежит ход в
позиции этого узла.

### 8.2. Selection

Родитель выбирает ребёнка с максимальным:

```text
score(parent, child) = -Q(child) + U(parent, child)

U = c_puct × prior(child) × sqrt(max(1, N(parent))) / (1 + N(child))
```

Минус перед `Q(child)` обязателен: value ребёнка задано с точки зрения
соперника. При равенстве score выбирать меньший action index для
детерминированного режима.

Не добавлять material bonus, handcrafted evaluation или exploration term,
которого нет в формуле.

### 8.3. Expansion

- Terminal node не отправляется в модель и не получает children.
- Non-terminal leaf оценивается моделью один раз.
- Children создаются только для `board.legal_moves`.
- Priors берутся из legal policy probabilities и суммируются в 1 с численной
  погрешностью.
- Каждый child получает корректный `to_play` после соответствующего хода.
- Коллизии action encoding считаются ошибкой.

### 8.4. Backup

Leaf value приходит относительно `leaf.to_play`. При движении к родителю знак
меняется на каждом ply:

```text
node.value_sum += value
node.visit_count += 1
value = -value
```

Тесты обязаны отдельно доказать знак на путях нечётной и чётной длины.

### 8.5. Terminal value

- Победа стороны, которой принадлежит ход в terminal board: `+1`.
- Поражение: `-1`.
- Любая ничья: `0`.

На практике checkmate означает `-1` для стороны хода. Использовать outcome
существующей среды, а не определять мат вручную.

## 9. Root noise, temperature и выбор хода

### 9.1. Dirichlet noise

При `root_noise=True` один раз после expansion root:

```text
P' = (1 - epsilon) × P + epsilon × Dirichlet(alpha)
```

- Шум применяется только к legal root children.
- После смешивания priors нормализованы.
- Один root не получает noise повторно при каждом simulation.
- RNG — локальный `numpy.random.Generator` с config seed.
- При игре с человеком noise всегда off.

### 9.2. Итоговая policy поиска

Search result содержит для каждого legal root action:

- visit count;
- нормализованную visit policy;
- prior;
- root value;
- число фактически выполненных simulations.

При `temperature == 0` выбирается максимальный visit count, tie-break — меньший
action index.

При `temperature > 0`:

```text
p(a) ∝ N(a) ** (1 / temperature)
```

Ход выбирается локальным seeded RNG. Не использовать global NumPy random.

## 10. Search lifecycle и tree reuse

Публичный API допускает эквивалентный дизайн, но должен поддерживать:

```python
class MCTS:
    def search(self, environment: ChessEnvironment) -> SearchResult: ...
    def advance(self, environment: ChessEnvironment, action: int) -> bool: ...
    def reset(self) -> None: ...
```

Требования:

- Root привязан к fingerprint позиции и истории, влияющей на `board119-v1`.
- Повторный `search` той же позиции продолжает существующее дерево.
- После хода `advance` делает соответствующего child новым root и отбрасывает
  недостижимые ветви.
- Если action отсутствует или fingerprint не совпал, дерево безопасно
  пересоздаётся и возвращается `False`.
- Переиспользование работает после хода человека и после хода бота.
- Дерево не сериализуется в этом шаге.
- Board/move stack не мутируются после завершения search.

Bootstrap-оценка root не входит в число requested simulations. После поиска
сумма visit counts root children должна равняться `simulations` для обычной
non-terminal позиции.

## 11. Model evaluator и batching service

### 11.1. Единый evaluator protocol

MCTS не должен напрямую зависеть от `ChessyModel`. Ввести небольшой protocol:

```python
class Evaluator(Protocol):
    def evaluate(self, history: Sequence[chess.Board]) -> Evaluation: ...
```

`Evaluation` содержит:

- policy probabilities `[4672]` только на legal actions;
- scalar value в `[-1,1]` относительно стороны хода.

Это позволяет использовать scripted evaluator в unit tests.

### 11.2. DirectModelEvaluator

- Кодирует `board119-v1` существующим encoder.
- Строит legal mask через существующий `az73-v1` helper.
- Выполняет `model.eval()` и `torch.inference_mode()`.
- Не меняет device модели на каждый запрос.
- Возвращает CPU NumPy/простые значения без autograd graph.
- Terminal position отклоняется: MCTS обязан обработать её до evaluator.

### 11.3. BatchingInferenceService

Единый worker владеет model inference на выбранном device:

- thread-safe queue запросов;
- batch до `max_batch_size`, максимум 32;
- первый запрос ждёт не более `max_batch_wait_ms`, по умолчанию 2 ms;
- один tensor transfer и один model forward на batch;
- каждый caller получает только свой result или своё exception;
- `start`, идемпотентный `close`, context manager;
- после close новые запросы отклоняются;
- при фатальной ошибке batch все futures завершаются ошибкой, worker не зависает;
- shutdown не теряет уже принятые requests.

Сервис нужен для будущих конкурентных партий/self-play. В одной последовательной
партии batch часто будет равен 1 — это ожидаемо и не считается дефектом.

## 12. Playable agent

`MCTSAgent` связывает environment, MCTS и evaluator.

```python
def choose_move(environment: ChessEnvironment) -> AgentDecision: ...
```

`AgentDecision` содержит минимум:

- `move`;
- `action`;
- `search_result`;
- elapsed time;
- model ID/export checksum;
- MCTS config.

Агент:

- никогда не возвращает нелегальный ход;
- не применяет ход к environment сам, если это не оговорено API;
- отказывается выбирать ход в terminal position;
- не использует resignation;
- остаётся детерминированным при temperature 0, noise off и одинаковых весах.

## 13. Модели при запуске

CLI принимает повторяемый параметр:

```text
--model <path-to-chessy-model-v1>
```

- Каждый export полностью валидируется при старте.
- Backend назначает безопасный model ID и показывает UI имя, checksum и
  architecture без абсолютного пути.
- При нескольких `--model` UI показывает dropdown.
- При отсутствии `--model` создаётся `ChessyModel` с `torch.manual_seed(0)` и
  ID `random-untrained-seed-0`.
- Случайная модель живёт только в памяти и не экспортируется автоматически.
- UI явно показывает предупреждение, что модель не обучена и играет случайно
  осмысленным только в рамках MCTS образом.

## 14. Игровая сессия

### 14.1. GameSession

Backend-сессия владеет:

- UUID game ID;
- `ChessEnvironment`;
- выбранной моделью и MCTS config;
- цветом человека и бота;
- часами;
- статусом `waiting / active / bot_thinking / finished`;
- termination reason и результатом;
- последовательностью ходов и agent decisions;
- feedback opt-in state;
- PGN.

Все изменения одной сессии защищены lock. Второй пользовательский ход во время
`bot_thinking` отклоняется, а не ставится в очередь.

### 14.2. Цвет

Стартовый выбор:

- white;
- black;
- random.

Random color выбирается seeded session RNG и сразу фиксируется в state. Если
бот играет белыми, его первый поиск запускается после создания сессии.

### 14.3. Часы

Поддержать:

- `untimed` — default;
- `3+2`;
- `5+0`;
- `10+0`;
- `15+10`.

Backend использует `time.monotonic()`. Время размышления бота учитывается.
Increment добавляется после завершённого легального хода. UI лишь отображает
server timestamps и периодически интерполирует их; итог timeout определяет
backend.

Для timed game backend запускает отменяемую deadline-task текущего хода. Она
завершает партию и отправляет `game_over`, даже если от browser больше не
приходит сообщений. После каждого хода, завершения или restart старая task
обязательно отменяется, чтобы исключить запоздалый timeout из прошлой позиции.

Timeout обычно означает поражение истёкшей стороны. Если соперник не имеет
достаточного материала для мата согласно `python-chess`, результат — ничья.

Тесты времени используют injected fake clock, а не `sleep`.

### 14.4. Draw и resignation

- Пользователь может сдаться: немедленная победа бота.
- Пользователь может предложить ничью.
- В v1 untrained/checkpoint bot детерминированно отклоняет draw offer; событие
  и ответ отображаются в UI.
- Claimable draw и автоматические ничьи обрабатываются правилами backend.
- Bot resignation выключен.
- Restart создаёт новую game ID, а не переписывает завершённую сессию.

### 14.5. PGN

PGN создаётся backend и включает:

- Date;
- White/Black (`Human` и `Chessy:<model-id>`);
- Result;
- Termination;
- model ID и weights checksum;
- MCTS version/profile/simulations;
- time control;
- полный movetext.

Незавершённая партия также может быть скачана с `Result "*"`. PGN download не
зависит от feedback opt-in.

## 15. Human feedback `chessy-human-feedback-v1`

### 15.1. UX-контракт

- На стартовом экране checkbox выключен.
- Значение checkbox означает только намерение рассмотреть сохранение.
- После завершения партии UI отдельно спрашивает подтверждение.
- До подтверждения ничего не записывается.
- Отказ не создаёт пустых директорий или временных файлов.
- Повторное подтверждение идемпотентно и не дублирует samples.

### 15.2. Файлы

После подтверждения атомарно создать:

```text
data/human_feedback/<game-id>/
  game.pgn
  manifest.json
  samples.jsonl
```

`manifest.json` содержит минимум:

- `format: chessy-human-feedback-v1`;
- game ID и UTC timestamp;
- human color;
- result и termination;
- time control;
- model ID, export checksum или random seed;
- полный MCTS config;
- `sample_weight: 4.0`;
- количество human samples;
- hashes `game.pgn` и `samples.jsonl`.

Каждая JSONL-строка соответствует только ходу пользователя и содержит:

- game ID и ply;
- FEN непосредственно перед ходом;
- до восьми FEN истории current-to-past;
- human UCI move;
- action index `az73-v1`;
- human color;
- финальный result относительно человека;
- source `human_online`;
- weight `4.0`.

Ходы бота остаются контекстом PGN, но не являются policy targets.

Запись выполняется через временную sibling-директорию и atomic rename. Корень
feedback задаётся backend config/CLI для тестируемости; browser не выбирает
путь.

## 16. FastAPI и WebSocket API

### 16.1. REST endpoints

Минимальный API:

```text
GET  /api/health
GET  /api/models
POST /api/games
GET  /api/games/{game_id}
GET  /api/games/{game_id}/pgn
POST /api/games/{game_id}/feedback
WS   /api/games/{game_id}/ws
```

`POST /api/games` принимает только model ID, color, time-control ID, MCTS
profile и feedback opt-in. Произвольные paths и raw FEN не принимаются.

### 16.2. WebSocket client messages

Версионировать envelope, например:

```json
{"version":"play-ws-v1","type":"move","payload":{"uci":"e2e4"}}
```

Поддержать типы:

- `move`;
- `resign`;
- `offer_draw`;
- `ping`.

Promotion передаётся полным UCI (`e7e8q`, `e7e8r`, ...).

### 16.3. Server events

- `state` — полный authoritative snapshot;
- `bot_thinking`;
- `move_applied`;
- `draw_declined`;
- `game_over`;
- `error` с безопасным code/message;
- `pong`.

После подключения и reconnect первым событием всегда идёт полный `state`.
События содержат монотонный session sequence number, чтобы UI мог игнорировать
устаревшее сообщение.

Bot search запускать вне event loop (`asyncio.to_thread` или эквивалент), иначе
WebSocket/health не должны зависать на 512 simulations.

### 16.4. Session registry

- In-memory registry достаточно для v1.
- Не более конфигурируемого небольшого числа активных сессий, default 8.
- Finished sessions сохраняются в памяти до завершения процесса либо явной
  bounded cleanup policy.
- Не удалять session во время активного WebSocket/search.
- Restart создаёт новую session через тот же validated factory.

## 17. React UI

### 17.1. Стартовый экран

- model dropdown;
- заметная маркировка untrained random model;
- выбор цвета white/black/random;
- time control;
- strength profile fast/normal/deep с числом simulations;
- checkbox «Добавить мои ходы в обучение», выключенный по умолчанию;
- кнопка начала партии.

### 17.2. Во время партии

- `react-chessboard` с ориентацией цвета человека;
- drag-and-drop и click-to-move;
- подсветка выбранного поля и легальных destinations;
- promotion dialog Q/R/B/N до отправки UCI;
- часы обоих игроков;
- история ходов в SAN;
- текущий status и индикатор размышления;
- model/profile badge;
- кнопки resign, offer draw, new game;
- понятная ошибка при отклонённом/устаревшем ходе;
- блокировка пользовательского ввода не в свой ход и во время bot thinking.

UI может использовать локальную копию state для рендера, но не применяет ход
окончательно до server event.

### 17.3. После партии

- результат и termination reason;
- полный movelist;
- кнопка Download PGN;
- checkpoint/MCTS summary;
- если pre-game opt-in включён — отдельные Save/Do not save;
- явное подтверждение успешной записи feedback;
- кнопка новой партии.

### 17.4. Качество интерфейса

- Рабочий desktop layout для ноутбука и простой responsive layout.
- Не использовать внешние изображения/CDN.
- Состояния loading, disconnected и reconnect видимы.
- Основные элементы доступны с клавиатуры и имеют labels.
- Не изображать силу случайной модели как реальную шахматную силу.

## 18. Production frontend и одна команда

Vite production output направить в:

```text
src/chessy/api/static/
```

Собранные `index.html` и hashed assets коммитятся, чтобы установленный Python
package мог запустить UI без Node. Source maps в production выключены.

Настроить Hatchling package data и тестом проверить, что wheel содержит static
assets.

Frontend source остаётся источником истины. `npm run build` должен полностью
пересоздавать static directory. В PR не должно быть stale bundle: тест/скрипт
сравнивает production build с отслеживаемыми файлами.

## 19. CLI `chessy play`

Добавить entry point:

```toml
[project.scripts]
chessy = "chessy.cli:main"
```

Команда:

```bash
uv run chessy play [--model PATH ...] [--device auto] [--port 0]
                   [--no-open] [--feedback-dir PATH]
```

Требования:

- host жёстко равен `127.0.0.1`;
- default port `0`;
- socket bind выполняется до запуска uvicorn, после чего печатается фактический
  URL;
- browser автоматически открывается через стандартный `webbrowser`, если не
  передан `--no-open`;
- startup валидирует модели, static assets и writable feedback root до запуска;
- SIGINT корректно останавливает server и inference service;
- повторный запуск не зависит от фиксированного порта;
- `--help` не загружает модель и не запускает сервер.

Параметр `--simulations` можно предоставить как явный expert override, но он
должен валидироваться и отображаться в UI/PGN/feedback manifest.

## 20. Тестирование MCTS

Использовать scripted/fake evaluator для математических unit tests; случайная
нейросеть не является oracle.

Обязательные тесты:

1. Точная PUCT formula на вручную заданных statistics.
2. Tie-break по action index.
3. Backup sign на путях длины 1, 2 и 3.
4. Terminal mate/draw не вызывают evaluator.
5. Expansion создаёт ровно legal children без коллизий.
6. Priors legal children суммируются в 1.
7. После N simulations сумма root child visits равна N.
8. Search не мутирует переданную environment.
9. Temperature 0 детерминирована.
10. Temperature >0 воспроизводима при seed.
11. Root noise применяется один раз, только при включении и воспроизводимо.
12. Noise выключен в игровых профилях.
13. Tree reuse сохраняет visits выбранного child.
14. Несовпадение позиции сбрасывает дерево.
15. MCTS возвращает легальный ход минимум на 100 детерминированных достижимых
    позициях при малом числе simulations.
16. Mate-in-one/forced terminal fixture выбирается корректно с scripted
    priors/value.
17. Повторения и правило 50 ходов дают terminal draw value 0.

## 21. Тестирование batching и модели

- Direct evaluator выдаёт zero probability illegal actions.
- Value соответствует `P(win)-P(loss)`.
- Batch service сохраняет порядок результатов callers.
- Batch никогда не превышает 32.
- Несколько конкурентных CPU запросов действительно объединяются хотя бы в
  один batch; тест использует barrier/event, а не случайный `sleep`.
- Ошибка одного model forward доставляется всем requests этого batch.
- Close завершает принятые requests и отклоняет новые.
- CPU integration search с реальным `ChessyModel` проходит.
- MPS integration search проходит при доступном MPS, иначе skip.
- Нет autograd graphs и gradients после inference.

Не задавать throughput threshold как критерий.

## 22. Тестирование game/API/feedback

### Game

- выбор всех вариантов цвета;
- bot first move за белых;
- illegal/out-of-turn move отклоняется;
- promotion Q/R/B/N;
- мат, пат, repetition, 50-move и insufficient material;
- resign и draw decline;
- fake-clock timeout и increment;
- max 300 plies;
- PGN корректно читается `python-chess` обратно.

### API/WebSocket

- health и models endpoints;
- создание game с validation errors;
- initial state при connect/reconnect;
- legal move → bot thinking → bot reply;
- второй move во время thinking отклоняется;
- stale/unknown message type даёт safe error, не падение socket;
- PGN endpoint до и после завершения;
- event sequence строго возрастает;
- event loop остаётся responsive во время fake slow search;
- static index и assets раздаются;
- API не принимает path/FEN от browser.

### Feedback

- opt-in false: запись невозможна;
- opt-in true до завершения: запись невозможна;
- post-game confirm создаёт ровно три файла;
- samples содержат только human moves;
- hashes и counts совпадают;
- повторный confirm идемпотентен;
- отказ и ошибка не оставляют temporary directories;
- test пишет только в `tmp_path`, не в реальный `data/human_feedback`.

## 23. Frontend tests

Vitest/Testing Library должны проверить минимум:

- стартовая форма и выключенный feedback checkbox;
- random model warning;
- отправка корректного create-game payload;
- authoritative state обновляет доску и ходы;
- ход блокируется не в очередь пользователя;
- promotion dialog формирует правильный UCI;
- thinking/disconnected/game-over states;
- post-game feedback confirmation;
- PGN download link доступен независимо от opt-in.

Дополнительно выполнить production build и ручной smoke-test в настоящем
браузере. Не добавлять Playwright только ради одного smoke-test в этом шаге.

## 24. Ручная приёмка

После автоматических тестов:

1. Запустить production UI:

   ```bash
   uv run chessy play --device mps --port 0
   ```

2. Убедиться, что URL использует `127.0.0.1` и случайный порт.
3. Начать игру случайной моделью за белых и сделать несколько ходов.
4. Начать игру за чёрных и проверить первый ход бота.
5. Проверить fast и normal profiles.
6. Проверить resign, draw offer, restart и PGN download.
7. Завершить короткую тестовую партию и проверить оба сценария feedback:
   отказ и подтверждение.
8. Подтвердить, что без opt-in файлы не создаются.
9. Остановить сервер через Ctrl+C и убедиться в корректном shutdown.

Не использовать результаты игры случайной сети как оценку силы.

## 25. Порядок реализации

1. Проверить актуальный clean `main` и прочитать ADR.
2. Добавить Python и frontend dependencies с lock-файлами.
3. Реализовать MCTS config, node, PUCT, expansion и backup с fake tests.
4. Добавить temperature, root noise и tree reuse.
5. Реализовать direct evaluator и batching service.
6. Добавить integration search с CPU/MPS моделью.
7. Реализовать MCTSAgent и GameSession с fake clock/agent tests.
8. Реализовать PGN и human feedback writer.
9. Реализовать FastAPI/session registry/WebSocket с fake agent для быстрых tests.
10. Реализовать CLI и loopback random-port startup.
11. Создать React UI и frontend tests.
12. Собрать production assets внутрь Python package и проверить wheel.
13. Запустить все Python/frontend tests и ручную приёмку.

Не начинать UI до доказанной корректности PUCT/backup. Красивой доской нельзя
компенсировать поиск, который перепутал знак value.

## 26. Команды финальной проверки

```bash
uv lock --check
uv sync --locked
uv run python --version
uv run pytest
uv run chessy --help
uv run chessy play --help
npm --prefix ui ci
npm --prefix ui test -- --run
npm --prefix ui run build
uv build
git diff --check
git diff --exit-code -- data/raw data/quality data/personal
git status --short
```

Отдельный automated smoke может запускать server с `--no-open`, читать
напечатанный URL, проверять `/api/health` и корректно завершать процесс. Он не
должен зависеть от фиксированного порта.

## 27. Критерии приёмки

Задача выполнена только если одновременно выполнено всё ниже:

- [ ] Реализован точный контракт `mcts-puct-v1` с `c_puct=1.5`.
- [ ] PUCT selection, expansion, terminal evaluation и alternating backup
      покрыты математическими unit tests.
- [ ] Search создаёт только legal children и всегда возвращает legal move.
- [ ] Temperature и Dirichlet noise воспроизводимы через локальный seed.
- [ ] Root noise выключен при игре с человеком.
- [ ] Профили fast/normal/deep используют 32/128/512 simulations.
- [ ] Дерево переиспользуется после ходов человека и бота.
- [ ] Direct evaluator использует `board119-v1`, `az73-v1` и WDL модели.
- [ ] Batching service поддерживает batch до 32 и wait до 2 ms.
- [ ] CPU и доступный MPS проходят integration search.
- [ ] При отсутствии export доступна явно помеченная random untrained model.
- [ ] Valid `chessy-model-v1` можно выбрать через CLI/UI без передачи path из
      browser.
- [ ] GameSession является источником правил, часов, результата и PGN.
- [ ] Поддержаны выбор цвета, untimed и четыре clock preset.
- [ ] Поддержаны promotion, resign, draw offer, restart и max 300 plies.
- [ ] FastAPI/WebSocket API валидирует сообщения и остаётся responsive во время
      поиска.
- [ ] Production server слушает только `127.0.0.1` на автоматически выбранном
      порту.
- [ ] React UI позволяет завершить полноценную партию против бота.
- [ ] UI показывает часы, movelist, thinking, result и reconnect states.
- [ ] PGN скачивается независимо от feedback.
- [ ] Feedback checkbox выключен по умолчанию и требует post-game confirmation.
- [ ] `chessy-human-feedback-v1` атомарно сохраняет только human policy targets.
- [ ] Без подтверждения реальные feedback-файлы не создаются.
- [ ] Python tests и frontend tests проходят.
- [ ] Production frontend собран и входит в wheel.
- [ ] `uv run chessy play` работает без отдельного Node/Vite процесса.
- [ ] Ctrl+C корректно останавливает uvicorn и inference worker.
- [ ] Existing datasets и model contracts не изменены.
- [ ] Training, self-play, snapshots и arena не добавлены преждевременно.

## 28. Что не делать даже при наличии времени

- Не обучать случайную модель.
- Не добавлять handcrafted material evaluation в MCTS.
- Не включать root noise в human play.
- Не реализовывать distributed workers.
- Не сохранять дерево поиска в snapshot.
- Не давать browser выбирать произвольный путь к модели/feedback.
- Не открывать сервер на LAN.
- Не использовать localStorage как authoritative game state.
- Не копировать bot moves в human targets.
- Не заявлять, что random checkpoint играет сильно.

## 29. Финальный отчёт агента

Агент должен сообщить:

1. Какие файлы и зависимости добавлены.
2. Фактические версии Python/FastAPI/Uvicorn/Node/npm/frontend packages.
3. Результаты PUCT, backup, terminal и tree-reuse tests.
4. Число Python и frontend tests, skips и время.
5. Результаты CPU и MPS integration search.
6. Фактический batch behavior inference service.
7. Результат `uv build` и наличие static assets в wheel.
8. URL ручного smoke-test и подтверждение loopback/random port.
9. Какие сценарии партии проверены вручную.
10. Проверены ли PGN download и оба feedback-сценария.
11. Были ли изменены existing datasets.
12. Все отклонения от плана и причины.

Не считать задачу завершённой, если математические MCTS tests зелёные, но
пользователь не может запустить UI одной командой и сделать легальный ход.
Не переходить к training snapshots/self-play без нового задания пользователя.
