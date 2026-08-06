# Chessy — шаг 6: self-play, replay buffer и RL-curriculum

Статус: готово к реализации
Дата постановки: 2026-08-06
Рабочая директория: `/Users/zhuravlevvikt/Documents/codex_projects/chessy`

## 1. Цель задачи

Собрать первый настоящий замкнутый цикл обучения Chessy без человеческих партий:

```text
текущая generation
      ↓
self-play с MCTS
      ↓
неизменяемый replay segment
      ↓
policy/value training
      ↓
arena против baseline и предыдущей generation
      ↓
promotion или продолжение обучения
```

Шаг должен превратить уже существующие шахматную среду, `board119-v1`,
`az73-v1`, policy/value model, MCTS и training snapshots в единый локальный
RL-pipeline. Pipeline запускается на macOS, использует MPS при наличии, имеет
CPU fallback, корректно останавливается и продолжается через полный snapshot.

Главная цель PR — не заявить, что за короткий smoke-run модель уже «научилась
шахматам», а доказать корректность обучающего контура и возможность измерять
прогресс. Реальный `base_rl` будет получен последующими долгими run.

## 2. Архитектурные источники

Перед реализацией полностью прочитать:

- `AGENTS.md`;
- `docs/PROJECT_PLAN.md`, особенно разделы 8–11, 16–18, 20–22;
- `docs/decisions/0002-compute-backend.md`;
- `docs/decisions/0003-position-encoding.md`;
- `docs/decisions/0004-action-encoding.md`;
- `docs/decisions/0005-artifact-formats.md`;
- `docs/decisions/0006-config-and-runs.md`;
- `docs/decisions/0008-mcts-v1.md`;
- `docs/tasks/STEP_04_MCTS_AND_PLAYABLE_APP.md`;
- `docs/tasks/STEP_05_RUNS_AND_TRAINING_SNAPSHOTS.md`;
- текущие модули `chessy.chess`, `chessy.encoding`, `chessy.model`,
  `chessy.mcts`, `chessy.run`, `chessy.snapshot`, `chessy.training`.

Нельзя менять семантику утверждённых форматов:

- `board119-v1`;
- `az73-v1`;
- `residual-cnn-v1`;
- `mcts-puct-v1`;
- `chessy-model-v1`;
- `chessy-config-v1`;
- `chessy-run-v1`;
- `chessy-snapshot-v1`.

Новые форматы этого шага:

- replay segment: `chessy-replay-segment-v1`;
- replay manifest: `chessy-replay-manifest-v1`;
- self-play game metadata: `chessy-selfplay-game-v1`;
- curriculum state: `chessy-curriculum-v1`;
- league manifest: `chessy-league-v1`;
- arena report: `chessy-arena-report-v1`.

Добавить ADR `docs/decisions/0009-self-play-replay-and-rl.md`, кратко фиксирующий
архитектуру поколений, immutable replay segments, outcome-only reward и правила
promotion.

## 3. Принцип реализации: поколения, а не непрерывная гонка

Для v1 использовать последовательные поколения:

1. Зафиксировать веса `generation-N`.
2. Сыграть ими заданное число self-play партий.
3. Атомарно закрыть один или несколько replay segments.
4. Обновить active replay manifest.
5. Выполнить заданное число training steps.
6. Сохранить candidate snapshot/export.
7. Провести arena.
8. При выполнении promotion gate сделать candidate новой generation.
9. Иначе сохранить отчёт и продолжить по конфигурации без ложного promotion.

Причины:

- self-play всегда знает точную версию весов;
- trainer не меняет модель под уже идущими партиями;
- границы остановки и resume понятны;
- replay и arena можно воспроизвести отдельно;
- проще диагностировать деградацию;
- MPS batching всё равно работает между несколькими параллельными игровыми
  actors.

Не строить в этом шаге распределённую систему, multiprocessing cluster,
Ray, Celery, Redis, Kafka или базу данных.

## 4. Границы задачи

### 4.1. Обязательно входит

- расширение строгой конфигурации для RL;
- безопасная эволюция fingerprint старых resolved configs;
- генераторы стартовых позиций curriculum A/B/C;
- один self-play game actor;
- локальный coordinator нескольких actors;
- общее batching inference service на CPU/MPS;
- динамическая temperature schedule;
- sparse MCTS visit targets;
- immutable replay segments и checksums;
- active replay manifest;
- replay dataset и stateful sampler;
- policy/value RL loss;
- настоящий trainer поверх replay;
- полный resume trainer/curriculum/replay/league state;
- random baseline;
- material alpha-beta baseline;
- arena с paired colors/positions;
- generation league и promotion gate;
- CLI для self-play, train, replay inspect/verify и arena;
- CPU unit/integration tests;
- короткий CPU smoke-run;
- короткая MPS-проверка, если MPS доступен;
- документация ручного запуска долгого локального run.

### 4.2. Не входит

- исторические партии владельца;
- supervised personalization;
- human-feedback из UI;
- смешанный RL/style loss;
- Stockfish как учитель или источник target value;
- material reward, piece-square reward или reward shaping;
- resignation;
- opening book из личных партий;
- test split владельца;
- распределённый self-play;
- CUDA-specific оптимизация;
- MLX;
- автоматический background daemon;
- автоматическое удаление архивных replay segments;
- доказательство конкретного Elo после smoke-run.

## 5. Награда и target semantics

Единственная обучающая награда — исход партии:

- победа с точки зрения стороны в позиции: class `win`, index `2`;
- ничья: class `draw`, index `1`;
- поражение: class `loss`, index `0`.

Не добавлять промежуточную награду за материал, шах, взятия, promotion или
длину партии.

Для каждого сыгранного ply сохранить:

- `board119-v1` до хода;
- полный список легальных root actions;
- число посещений каждого root action;
- выбранный action;
- side to move;
- итоговый WDL target с точки зрения side to move;
- game ID, ply и generation ID.

Policy target строится как нормализованные MCTS visit counts. Он не должен
зависеть от temperature, использованной для выбора фактического хода.

Если сумма visits неожиданно равна нулю, segment writer обязан отклонить sample.

## 6. Изменения MCTS

Текущий `MCTS.search()` одновременно рассчитывает visit distribution и выбирает
ход через фиксированную temperature из `MCTSConfig`. Для self-play разделить эти
ответственности, не ломая игровой API.

Предпочтительный контракт:

```python
result = mcts.search(environment)
action = sample_action(
    result,
    temperature=temperature_schedule.for_ply(ply),
    rng=actor_rng,
)
```

Допустимо оставить поле `SearchResult.action` для обратной совместимости игры с
человеком, но self-play обязан явно выбирать action из raw visits.

Добавить и протестировать:

- `visit_policy(result)` — нормализует raw visits независимо от temperature;
- `sample_action(..., temperature=0)` — детерминированно выбирает максимальный
  visit count, при равенстве минимальный action;
- `sample_action(..., temperature>0)` — использует `visits ** (1/T)`;
- отказ при NaN, отрицательных visits, пустой legal policy и некорректной T;
- сохранение tree reuse через `MCTS.advance()` после фактического хода;
- root Dirichlet noise только для self-play;
- отсутствие root noise в arena и игре с человеком.

Не пытаться реализовать virtual loss или параллельный поиск внутри одного дерева.
Параллелизм v1 — несколько независимых игр, использующих один
`BatchingInferenceService`.

## 7. RL-конфигурация

Расширить `chessy-config-v1` вложенными строгими секциями. Smoke-конфиги шага 5
должны продолжать загружаться.

Рекомендуемые секции:

```yaml
self_play:
  actors: 2
  games_per_generation: 8
  simulations: 64
  c_puct: 1.5
  root_noise: true
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25
  temperature:
    initial: 1.0
    cutoff_ply: 20
    final: 0.0
  max_game_plies: 160
  inference_batch_size: 32
  inference_wait_ms: 2.0

replay:
  root_dir: replay
  samples_per_segment: 16384
  active_max_samples: 250000
  recent_fraction: 0.5
  recent_generations: 2
  cache_segments: 2

rl:
  policy_loss_weight: 1.0
  value_loss_weight: 1.0
  train_steps_per_generation: 250
  batch_size: 256
  gradient_clip_norm: 1.0

curriculum:
  initial_stage: endgames
  stage_mode: manual
  stage_mix:
    endgames: 1.0
    reduced: 0.0
    full: 0.0

evaluation:
  games_per_match: 40
  simulations: 96
  promotion_min_score: 0.55
  promotion_min_games: 40
  require_lower_confidence_above: 0.50
```

Все числа строго валидировать. В частности:

- actors `1..8`;
- simulations > 0;
- batch size > 0;
- temperature >= 0;
- cutoff ply >= 0;
- fractions в `[0, 1]`;
- сумма `stage_mix` равна 1 с разумным tolerance;
- `active_max_samples >= samples_per_segment`;
- promotion threshold в `[0, 1]`;
- max game plies > 0;
- `root_noise=true` разрешён только в self-play config;
- evaluation принудительно использует temperature 0 и root noise false.

### 7.1. Совместимость fingerprint

Добавление optional/default полей в Pydantic не должно сделать старые
`config.resolved.json` нечитаемыми.

Сейчас некоторые проверки повторно считают fingerprint из
`model_dump()`, который добавит новые defaults. Исправить контракт:

- fingerprint существующего run считается по canonical JSON, фактически
  сохранённому в `config.resolved.json`;
- новый run сначала полностью materialize-ит resolved config, затем хеширует
  именно эти bytes;
- snapshot verification сравнивает fingerprint сохранённых resolved bytes;
- старый snapshot без новых секций остаётся валиден;
- новый RL-run содержит секции явно в resolved config.

Добавить migration regression test: resolved config шага 5 открывается после
расширения schema и сохраняет прежний fingerprint.

## 8. Curriculum

### 8.1. Общий контракт source

```python
class PositionSource(Protocol):
    def sample(self, rng: np.random.Generator) -> StartPosition: ...
```

`StartPosition` содержит:

- stage;
- source kind;
- FEN;
- стабильный seed/index;
- metadata генератора;
- максимум plies для этой позиции.

Любая позиция должна:

- проходить `chess.Board(fen).is_valid()`;
- быть standard chess, не Chess960;
- быть нетерминальной;
- иметь ровно двух королей;
- корректно работать в `ChessEnvironment`;
- воспроизводиться по seed.

### 8.2. Стадия A: `endgames`

Реализовать минимум четыре смеси:

- KQK;
- KRK;
- KPvK;
- простые pawn endings с пешками обеих сторон.

Для KQK/KRK случайно варьировать:

- сильную сторону;
- side to move;
- расположение фигур;
- расстояние между королями.

Отбрасывать invalid/terminal позиции. Не выдавать только «мат в один»: позиции
должны покрывать и координацию фигур.

Для pawn endings исключать пешки на первой и восьмой горизонтали. Castling и en
passant отсутствуют.

### 8.3. Стадия B: `reduced`

Генерировать позиции с 6–16 фигурами:

- оба короля обязательны;
- material imbalance ограничен конфигом;
- positions valid и нетерминальны;
- side to move случайна;
- castling rights соответствуют фактическим фигурам либо отсутствуют;
- генератор имеет bounded rejection attempts и понятную ошибку при исчерпании.

Не утверждать равномерность этого распределения. Логировать material histogram.

### 8.4. Стадия C: `full`

Смесь:

- стандартная начальная позиция;
- небольшой публичный/синтетический список нейтральных opening prefixes;
- случайные короткие legal prefixes из начальной позиции.

Opening prefixes хранить в репозитории как маленький versioned fixture. Не
читать `data/personal`, train/val/test владельца и human feedback.

### 8.5. Переходы между стадиями

Поддержать два режима:

- `manual` — stage/mix меняется только через явный `fork`;
- `gated` — stage manager предлагает переход после настроенных arena gates.

В v1 автоматический gate не должен незаметно менять конфигурацию. Событие
`curriculum_gate_passed` записывается, затем новая смесь фиксируется в
curriculum state и stage snapshot. Любой переход логируется.

Для smoke использовать manual mode. Реальные пороги оставить конфигурируемыми,
не выдавать предварительные значения за научно обоснованные.

## 9. Self-play game actor

Actor получает:

- immutable generation model;
- общий thread-safe evaluator;
- position source;
- actor-specific RNG seed;
- MCTS config;
- stop token.

Actor:

1. Получает стартовую позицию.
2. На каждом ply выполняет MCTS.
3. Сохраняет raw visit counts до выбора хода.
4. Выбирает ход по temperature schedule.
5. Продвигает environment и MCTS root.
6. Завершает игру по `ChessEnvironment.outcome(claim_draw=True)`.
7. При `max_game_plies` фиксирует truncation как draw с отдельной причиной.
8. После результата заполняет WDL target для всех samples.
9. Возвращает завершённую игру coordinator-у.

Resignation полностью выключен.

### 9.1. Детерминированные IDs и seed derivation

Не использовать Python `hash()`.

Получать seed через SHA-256 от:

```text
run_seed | generation | actor_id | game_index
```

Game ID также детерминирован и включает run ID/generation/game index. После
resume coordinator не должен повторно записать уже sealed game ID.

Порядок завершения потоков может различаться. Перед записью segment сортировать
игры по game index, чтобы одинаковый набор завершённых игр давал стабильный
manifest.

### 9.2. Остановка

После SIGINT/SIGTERM:

1. Не назначать новые игры.
2. Уже начатые игры либо корректно завершить, либо пометить incomplete и не
   включать их samples в replay.
3. Не ждать бесконечно: configurable graceful timeout.
4. Закрыть inference queue после actors.
5. Sealed completed games записать атомарно.
6. Обновить replay manifest.
7. Завершить текущий trainer batch, если он идёт.
8. Создать stop snapshot с phase/curriculum/replay/league state.

Incomplete игры разрешено хранить только как диагностические metadata вне
active replay. Их нельзя обучающе интерпретировать как ничью.

## 10. Replay segment format

Каждый segment — атомарно опубликованная директория:

```text
replay/
  segments/
    segment-<generation>-<ordinal>-<sha-prefix>/
      samples.npz
      games.pgn
      games.jsonl
      manifest.json
      checksums.sha256
  manifests/
    replay-<generation>-<fingerprint>.json
```

Формат segment manifest: `chessy-replay-segment-v1`.

### 10.1. `samples.npz`

Использовать только NumPy arrays, загружаемые с `allow_pickle=False`:

- `boards`: `uint8 [N, 119, 8, 8]`;
- `policy_offsets`: `int64 [N + 1]`;
- `policy_actions`: `uint16 [K]`;
- `policy_visits`: `uint32 [K]`;
- `value_class`: `uint8 [N]`;
- `selected_action`: `uint16 [N]`;
- `game_index`: `uint32 [N]`;
- `ply`: `uint16 [N]`;
- `generation`: `uint32 [N]`.

`boards` кодируются losslessly для `board119-v1`:

- бинарные planes хранятся как 0/1;
- halfmove plane хранит integer `0..100` и при чтении делится на 100;
- writer проверяет, что исходное float32 encoding точно восстанавливается.

Не хранить dense `[N, 4672]` policy: это неоправданно увеличит replay.

### 10.2. PGN и metadata

`games.pgn` содержит полные партии со стартовым FEN и Variant/SetUp headers при
необходимости. `games.jsonl` содержит для каждой игры:

- format;
- game ID/index;
- generation;
- stage/source;
- initial FEN;
- result;
- termination (`checkmate`, `stalemate`, repetition, 50-move, insufficient,
  max-plies и т.д.);
- ply count;
- model/export checksum;
- actor seed;
- timestamps/durations;
- MCTS settings.

PGN и metadata должны согласовываться с arrays. Verifier переигрывает все партии
или детерминированную выборку в больших segments и проверяет actions/value.
Для тестовых segments всегда проверять все партии.

### 10.3. Безопасность и атомарность

- писать во временную sibling-директорию;
- fsync payload и директории;
- checksums покрывают каждый payload file;
- запрет symlink, extra files, absolute paths и `..`;
- проверить segment до atomic rename;
- существующий segment никогда не перезаписывать;
- loader проверяет dtype, rank, shapes, offsets и bounds;
- `policy_offsets[0] == 0`, последний offset равен K;
- actions внутри sample уникальны и `< 4672`;
- visits > 0;
- selected action legal и присутствует в sparse policy;
- value class только 0/1/2;
- ни один incomplete game не попадает в arrays.

## 11. Replay manifest и retention

`chessy-replay-manifest-v1` содержит:

- created_at;
- run ID;
- generation;
- ordered список segments;
- path только относительно project/run root;
- SHA-256 manifest/checksums каждого segment;
- sample/game counts;
- stage/generation histograms;
- active window policy;
- total logical/physical bytes;
- fingerprint всего manifest.

Manifest immutable: каждое изменение создаёт новый файл. Snapshot ссылается на
точный manifest и встраивает его содержимое/checksum.

`active_max_samples` ограничивает sampler window, но в этом шаге не удаляет
старые segment directories. Старые snapshots могут на них ссылаться. Физическое
удаление/архивация появится только с отдельным reference-aware GC.

При превышении configurable hard disk limit pipeline останавливается безопасно
до запуска новых self-play игр и сообщает точный объём/пути.

## 12. Replay dataset и sampler

Dataset лениво читает immutable segments и имеет небольшой LRU cache.

Batch возвращает:

- boards float32 `[B,119,8,8]`;
- sparse или собранный dense policy target `[B,4672]`;
- legal mask `[B,4672]`, построенный из полного списка root actions;
- value class `[B]`;
- sample metadata для метрик.

Sampler:

- полностью stateful и сериализуем;
- использует собственный `torch.Generator` или NumPy Generator;
- поддерживает uniform sampling по active samples;
- поддерживает `recent_fraction` из последних N generations;
- не допускает segment/sample вне manifest;
- после resume выдаёт тот же следующий batch в однопоточном CPU-тесте;
- логирует распределение generations/stages в batches.

Не копировать sample физически ради weighting.

## 13. RL loss и trainer

Policy loss:

```text
L_policy = -sum(pi_mcts * log_softmax(masked_policy_logits))
```

Illegal logits маскируются до softmax. Legal mask должен содержать все legal root
actions, а не только выбранный action.

Value loss:

```text
L_value = cross_entropy(value_logits, wdl_class)
```

Итог:

```text
L = policy_loss_weight * L_policy
  + value_loss_weight * L_value
```

Weight decay реализуется AdamW. Material/reward auxiliary losses отсутствуют.

Trainer обязан:

- обучать model на selected device;
- использовать float32 по умолчанию;
- не включать mixed precision без отдельного benchmark/test;
- проверять finite loss/gradients;
- применять gradient clipping;
- писать optimizer LR и gradient norm;
- создавать periodic/best/stage/stop snapshots;
- хранить replay manifest и league manifest текущего шага;
- сохранять curriculum/generation/phase state;
- корректно продолжать sampler/optimizer/scheduler/global step;
- не использовать synthetic smoke dataset из шага 5.

Минимальные metrics на training step/window:

- total/policy/value loss;
- policy entropy;
- MCTS target entropy;
- top-1 agreement с максимальным MCTS action;
- value class accuracy;
- predicted expected value mean;
- gradient norm;
- learning rate;
- samples/sec;
- replay generation age;
- stage mix.

Best snapshot выбирать по явно заданной metric, например rolling total loss или
arena score. Не смешивать разные критерии под одним именем.

## 14. Полный snapshot и resume

Расширить training payload/state без небезопасных Python objects. Snapshot RL-run
должен восстанавливать:

- model/optimizer/scheduler;
- replay sampler;
- Python/NumPy/Torch RNG;
- global step/samples seen;
- generation;
- текущую phase (`selfplay`, `train`, `arena`, `promotion`);
- curriculum state;
- assigned/completed game indexes;
- active replay manifest fingerprint/path;
- league manifest fingerprint/path;
- best metrics;
- generation candidate/incumbent identity;
- arena progress, если resume разрешён внутри arena.

Для простоты допустимо возобновлять arena с начала, если:

- незавершённый отчёт не считается финальным;
- seed/opening schedule тот же;
- это явно записано событием;
- уже завершённые результаты не смешиваются дважды.

Snapshot writer должен принимать актуальные replay/league references. Нельзя
копировать в каждый snapshot все replay bytes.

Resume проверяет существование и checksums всех active segments до продолжения.
Повреждённый или отсутствующий replay segment — явная ошибка, а не тихое
исключение sample.

## 15. League и generations

League manifest `chessy-league-v1` содержит:

- incumbent generation;
- immutable playable export path/checksum;
- historical promoted generations;
- parent generation;
- creation/promotion arena report;
- curriculum stage;
- optional tags (`initial`, `candidate`, `promoted`, `base_rl`).

Generation 0 — исходные случайные веса run. До первого self-play создать
playable export и зафиксировать checksum.

Historical opponents выбирать детерминированно по конфигу. В первой реализации
достаточно:

- incumbent;
- предыдущая promoted generation;
- опционально одна более старая generation.

Self-play stage C может задавать смесь current-vs-current и
current-vs-historical. Для стадий A/B current-vs-current достаточно.

Ни один candidate не получает тег `promoted` без завершённого arena report.
Тег `base_rl` никогда не ставится автоматически коротким smoke-run. Для него
нужна отдельная ручная команда после выполнения критериев проекта.

## 16. Baselines

### 16.1. Random legal baseline

- выбирает равномерно из legal moves;
- имеет собственный детерминированный RNG;
- не использует model/MCTS;
- корректно играет из curriculum positions.

### 16.2. Material alpha-beta baseline

Реализовать небольшой фиксированный baseline:

- negamax/alpha-beta;
- depth 1–2, конфигурируемо;
- terminal score доминирует над material;
- material values фиксированы и документированы;
- no opening book;
- no Stockfish;
- deterministic UCI tie-break;
- необязательный маленький transposition cache только если не усложняет
  корректность.

Это только измерительная линейка, не teacher и не источник training targets.

Добавить tactical/terminal tests: мат выбирается вместо material gain, legal
move всегда возвращается, одинаковая позиция даёт одинаковый ход.

## 17. Arena

Arena запускается без Dirichlet noise и с temperature 0.

Матчи используют paired schedule:

- одна и та же стартовая позиция играется дважды;
- candidate меняет цвет;
- seed/start position фиксированы в report;
- оба агента получают одинаковый лимит MCTS simulations своего типа.

Report содержит:

- candidate/opponent checksums;
- config fingerprint;
- stage/source distribution;
- wins/draws/losses;
- score `(W + 0.5D) / N`;
- average plies;
- termination distribution;
- elapsed/throughput;
- seeded confidence interval для mean game score;
- список game IDs/PGN paths;
- promotion decision и причины.

Promotion gate:

- сыграно не меньше `promotion_min_games`;
- point score >= `promotion_min_score`;
- нижняя граница configured confidence interval > configured threshold;
- нет invalid/crashed games.

При малой выборке report валиден, но promotion запрещён. Не пересчитывать Elo как
доказанный рейтинг из нескольких партий. Если выводится Elo difference, рядом
обязательно показывать широкий CI и метод расчёта.

Arena smoke может использовать 2–4 партии и проверяет plumbing, но всегда имеет
`eligible_for_promotion=false`.

## 18. Coordinator state machine

Реализовать явную state machine:

```text
INITIALIZE
  → SELF_PLAY
  → SEAL_REPLAY
  → TRAIN
  → EXPORT_CANDIDATE
  → ARENA
  → PROMOTE | REJECT
  → NEXT_GENERATION | COMPLETE
```

Каждый переход:

- валидируется;
- пишется в events JSONL;
- обновляет serializable state;
- имеет безопасную точку snapshot;
- не повторяет уже атомарно завершённый side effect после resume.

Использовать idempotency keys для segment/export/report names. Если destination
уже существует, проверить checksum и state; не перезаписывать его.

События минимум:

- `generation_started`;
- `selfplay_started/completed/stopped`;
- `replay_segment_sealed`;
- `replay_manifest_updated`;
- `training_started/completed`;
- `candidate_exported`;
- `arena_started/completed`;
- `generation_promoted/rejected`;
- `curriculum_gate_passed`;
- `run_completed/stopped`.

## 19. CLI

Добавить понятные команды:

```bash
uv run chessy selfplay smoke --config configs/rl-smoke.yaml
uv run chessy train rl --config configs/rl-smoke.yaml
uv run chessy train rl --resume runs/<run-id>
uv run chessy replay inspect replay/manifests/<manifest>.json
uv run chessy replay verify replay/manifests/<manifest>.json
uv run chessy arena run --candidate <export-or-snapshot> --opponent random --games 4
uv run chessy arena run --candidate <export-or-snapshot> --opponent material --games 4
```

Конкретная группировка argparse может отличаться, но:

- `--help` работает на всех уровнях;
- invalid config даёт короткое понятное сообщение и non-zero exit;
- `--resume` и `--config` mutually exclusive;
- operational `--device` override логируется;
- destructive cleanup команд нет;
- test split персональных данных нигде не принимается;
- команды печатают абсолютные пути созданных run/report/manifest.

Добавить `configs/rl-smoke.yaml`:

- tiny model 8 channels / 1 block;
- CPU;
- stage A;
- 1 actor;
- 2–4 self-play games;
- 4–8 MCTS simulations;
- маленький batch;
- 2–4 training steps;
- 2 arena games;
- promotion невозможен из-за min games.

Добавить `configs/rl-local-mps.yaml` как безопасный стартовый, но не
«оптимальный» пресет:

- текущая основная model 96×8;
- MPS;
- 2 actors сначала;
- 64 simulations для stage A;
- inference batch до 32;
- консервативные segment/train размеры;
- явные комментарии в соседней документации, что параметры требуют benchmark.

## 20. Целевая структура

```text
configs/
  rl-smoke.yaml
  rl-local-mps.yaml
src/chessy/
  curriculum/
    __init__.py
    schema.py
    sources.py
    endgames.py
    reduced.py
    full.py
    manager.py
  selfplay/
    __init__.py
    temperature.py
    game.py
    actor.py
    coordinator.py
  replay/
    __init__.py
    schema.py
    codec.py
    segment.py
    manifest.py
    dataset.py
    sampler.py
    verify.py
  training/
    rl_loss.py
    rl_trainer.py
    rl_state.py
  evaluation/
    __init__.py
    agents.py
    material.py
    arena.py
    statistics.py
    league.py
docs/decisions/
  0009-self-play-replay-and-rl.md
tests/
  curriculum/
  selfplay/
  replay/
  training/
  evaluation/
```

Небольшие файлы допустимо объединять. Не складывать codecs, filesystem I/O,
actors, trainer и arena в один огромный модуль.

## 21. Тесты

### 21.1. Curriculum

- 100+ позиций каждого generator valid и нетерминальны;
- KQK/KRK имеют точный material;
- pawn positions не имеют пешек на 1/8 rank;
- одинаковый seed даёт одинаковую последовательность FEN;
- reduced соблюдает piece/material bounds;
- full prefixes legal;
- bounded rejection корректно падает.

### 21.2. Temperature/MCTS targets

- raw visits нормализуются в policy;
- T=0 deterministic;
- seeded T>0 reproducible;
- temperature не меняет сохранённый target;
- noise только self-play;
- chosen action legal;
- tree advance соответствует сыгранному ходу.

### 21.3. Self-play

- scripted mate создаёт win/loss targets с правильной перспективой;
- draw создаёт class 1 для обеих сторон;
- max plies помечается truncation, а не естественной ничьёй;
- incomplete game не даёт samples;
- PGN переигрывается;
- deterministic game IDs/seeds;
- parallel completion sorting стабильно.

Для быстрых тестов использовать scripted/fake evaluators, а не тысячи real MCTS
simulations.

### 21.4. Replay

- round-trip arrays без потери board119;
- `allow_pickle=False`;
- sparse offsets/actions/visits валидируются;
- checksum corruption обнаруживается;
- symlink/extra file/path traversal отклоняются;
- atomic writer не публикует partial segment;
- manifest counts/histograms совпадают;
- missing referenced segment блокирует resume;
- active window не удаляет старые bytes.

### 21.5. Sampler и loss

- sampler resume возвращает тот же следующий batch;
- recent fraction работает статистически на seeded sample;
- samples только из manifest;
- policy loss совпадает с ручным маленьким примером;
- illegal logits не влияют на masked policy loss;
- WDL perspective корректна;
- finite guards ловят NaN/Inf;
- один optimizer step изменяет веса;
- tiny fixed replay overfit уменьшает loss.

### 21.6. Baseline и arena

- random всегда legal и seeded;
- material agent deterministic;
- material agent выбирает forced mate;
- color-paired schedule симметричен;
- W/D/L и score считаются правильно;
- confidence calculation reproducible;
- smoke sample не проходит min-games promotion gate;
- report checksum/format валидируются.

### 21.7. End-to-end и resume

CPU integration test:

1. Создать generation 0.
2. Сыграть tiny deterministic self-play batch.
3. Запечатать replay segment.
4. Сделать минимум два trainer steps.
5. Остановиться.
6. Открыть snapshot и resume.
7. Завершить generation и arena.
8. Проверить run/replay/league/report manifests.

Отдельный deterministic single-actor CPU test сравнивает uninterrupted и
stop/resume:

- model state;
- optimizer/scheduler;
- sampler;
- global step/generation/phase;
- следующий batch;
- replay manifest fingerprint.

Для parallel/MPS режима побитовая идентичность не требуется, но не должно быть
дубликатов или потерянных sealed games.

Subprocess SIGINT test отправляет сигнал во время self-play или training и
проверяет валидный stop snapshot.

## 22. Производительность и ограничения тестов

Unit suite должна оставаться разумной для CPU:

- tiny model;
- fake evaluators там, где проверяется orchestration;
- реальные MCTS smoke — единицы simulations/games;
- performance benchmark не является flaky assertion CI;
- MPS tests помечены skip, если backend недоступен;
- долгий `rl-local-mps` никогда не запускается из pytest.

Добавить отдельную benchmark-команду/скрипт, измеряющий:

- self-play positions/sec;
- inference batch-size histogram;
- MCTS moves/sec;
- trainer samples/sec;
- replay bytes/sample;
- peak process RSS, если доступно без новой тяжёлой зависимости.

Результаты benchmark информативны и не используются как жёсткие тестовые
пороги.

## 23. Ручная приёмка

Перед сдачей агент обязан выполнить:

```bash
uv lock --check
uv sync --locked
uv run pytest
uv run chessy --help
uv run chessy selfplay smoke --config configs/rl-smoke.yaml
uv run chessy train rl --config configs/rl-smoke.yaml
```

Затем:

1. Остановить или ограничить RL smoke на промежуточной фазе.
2. Выполнить `train rl --resume`.
3. Проверить run через `run inspect`.
4. Проверить каждый replay manifest/segment.
5. Проверить финальный snapshot.
6. Запустить короткую arena против random и material baseline.
7. На Mac выполнить короткий MPS smoke с tiny model.
8. Зафиксировать реальные пути и размеры созданных артефактов в отчёте.

Smoke outputs остаются в gitignored `runs/`/`replay/`. Не удалять существующие
пользовательские артефакты.

## 24. Критерии готовности

Шаг готов, когда одновременно выполнено:

1. Curriculum A/B/C выдаёт валидные воспроизводимые позиции.
2. Self-play сохраняет legal sparse MCTS targets и правильные WDL targets.
3. Несколько actors используют общий batched evaluator без deadlock.
4. Replay segment immutable, атомарен и полностью проверяется checksums/schema.
5. Replay manifest точно фиксирует active training data.
6. Trainer реально оптимизирует policy/value model на replay.
7. Snapshot содержит полное RL/curriculum/replay/league состояние.
8. Stop/resume не сбрасывает optimizer, sampler, generation или phase.
9. Arena воспроизводимо сравнивает candidate с random/material/previous model.
10. Promotion невозможен без достаточного числа игр и confidence gate.
11. CPU end-to-end smoke проходит.
12. MPS tiny smoke проходит на доступном Mac.
13. Старые smoke configs/snapshots шага 5 остаются читаемыми.
14. Полный pytest зелёный.
15. Никакие большие replay/run artifacts не попали в Git.

## 25. Что агент должен вернуть

В финальном отчёте указать:

- краткое описание архитектуры;
- список новых форматов и CLI-команд;
- какие curriculum sources реализованы;
- точную формулу loss;
- как устроены replay segments/manifests;
- как resume восстанавливает generation и phase;
- результаты полного pytest;
- результаты CPU и MPS smoke;
- реальные размеры replay bytes/sample и snapshots;
- arena smoke W/D/L без заявлений о силе;
- пути созданных локальных run/replay/report;
- известные ограничения и следующий безопасный long-run command;
- `git status --short`.

Коммит не создавать: ревью, feature-ветку, commit и push выполняет основной
агент после отдельного запроса пользователя.

## 26. Запрещённые упрощения

Нельзя считать задачу выполненной, если:

- self-play хранит только выбранный ход вместо visit distribution;
- value target берётся из оценки сети, а не из результата партии;
- material score используется как reward;
- replay — один изменяемый JSONL без manifest/checksums;
- snapshot копирует гигабайты replay внутрь себя;
- resume начинает generation заново и создаёт дубликаты;
- MCTS weights меняются во время незавершённой self-play generation;
- arena включает root noise или stochastic temperature;
- candidate автоматически объявляется сильнее после 2–4 smoke games;
- corrupted segment тихо пропускается;
- старые segments удаляются без анализа ссылок snapshots;
- личные train/val/test данные используются в RL curriculum;
- long training запускается автоматически из тестов;
- MPS является единственным поддерживаемым backend;
- PR содержит generated run, replay segment, snapshot или arena PGN.

Главный инженерный принцип шага: каждый обучающий пример должен иметь понятное
происхождение, каждый долгий run — точку безопасного продолжения, а каждое
утверждение об усилении — воспроизводимый матч и достаточную статистику.
