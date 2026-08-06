# Chessy — шаг 7: персональный датасет и supervised fine-tuning

Статус: готово к реализации
Дата постановки: 2026-08-06
Рабочая директория: `/Users/zhuravlevvikt/Documents/codex_projects/chessy`

## 1. Цель задачи

Научить Chessy предсказывать шахматные решения владельца поверх уже обученной
универсальной модели `base_rl`.

Полный контур шага:

```text
исторические PGN + зафиксированные train/val/test split
                         ↓
восстановление полной истории позиции и результата партии
                         ↓
immutable personal dataset segments
                         ↓
base_rl playable export
                         ↓
supervised policy/value fine-tuning
                         ↓
validation и early stopping
                         ↓
personal_supervised playable export
```

Критерий успеха — не просто уменьшение train loss. Лучшая персональная версия
должна на неизменяемом validation split:

- лучше исходного `base_rl` предсказывать фактические ходы владельца;
- иметь более низкую policy cross-entropy;
- иметь более высокие top-1/top-3/top-5 accuracy;
- назначать фактическому ходу большую среднюю вероятность;
- загружаться игровым интерфейсом и играть только легальные полноценные партии.

Test split в этом шаге не используется для настройки, early stopping или выбора
лучшего snapshot.

## 2. Обязательные источники

Перед реализацией полностью прочитать:

- `AGENTS.md`;
- `docs/PROJECT_PLAN.md`, особенно разделы 4, 7, 12, 16–18 и 20–22;
- `docs/decisions/0003-position-encoding.md`;
- `docs/decisions/0004-action-encoding.md`;
- `docs/decisions/0005-artifact-formats.md`;
- `docs/decisions/0006-config-and-runs.md`;
- `docs/decisions/0009-self-play-replay-and-rl.md`;
- `docs/tasks/STEP_05_RUNS_AND_TRAINING_SNAPSHOTS.md`;
- `docs/tasks/STEP_06_SELF_PLAY_RL_CURRICULUM.md`;
- `scripts/filter_quality_games.py`;
- `scripts/build_personal_dataset.py`;
- `scripts/split_personal_dataset.py`;
- текущие `chessy.encoding`, `chessy.model`, `chessy.run`, `chessy.snapshot`,
  `chessy.replay` и `chessy.training`.

Исходные данные:

- Chess.com user: `mu1876`;
- Lichess user: `mu1878`;
- `data/raw/chess_com_mu1876.pgn`;
- `data/raw/lichess_mu1878.pgn`;
- `data/quality/game_quality.csv`;
- `data/personal/splits/train.jsonl` — 106104 samples;
- `data/personal/splits/val.jsonl` — 13337 samples;
- `data/personal/splits/test.jsonl` — 13293 samples;
- `data/personal/splits/manifest.json`.

## 3. Зафиксированные решения

### 3.1. Начальная модель

Fine-tuning начинается только из явно указанного валидного
`chessy-model-v1` export с тегом/ролью `base_rl`.

- Не брать случайную модель по умолчанию.
- Не брать автоматически «последний snapshot» из `runs/`.
- Проверить Safetensors, manifest, checksum, architecture и encodings.
- Зафиксировать путь и checksum base export в run parent metadata.
- В smoke-тесте разрешён tiny export со случайными весами, но он обязательно
  маркируется `fixture`, а не `base_rl`.
- Никакой smoke-run не получает настоящий тег `personal_supervised` без
  validation gate.

### 3.2. Supervised objective

```text
L_total = L_policy + 0.25 × L_value
```

- `L_policy` — cross-entropy фактического хода владельца среди всех легальных
  ходов позиции;
- `L_value` — WDL cross-entropy по результату партии с точки зрения side to move;
- illegal logits маскируются до softmax;
- weight decay реализуется AdamW;
- material/Stockfish evaluation не используется как value target;
- move accuracy не превращается в soft label.

### 3.3. Веса источников

Начальные веса:

- `good_move`: `0.75`;
- `full_game`: `1.0`;
- `human_online`: в этот шаг не входит.

Реализовать их через weighted selection внутри sampler. Не применять тот же вес
ещё раз к loss, чтобы не получить двойное усиление.

На одну историческую партию в одной эпохе выбирать не более 16 позиций.

### 3.4. Разделение данных

- Использовать существующее хронологическое split без переразбиения.
- Game ID не пересекаются между train/val/test.
- Одна календарная дата не разделяется между split.
- Validation доступен trainer-у.
- Test не доступен обычной train/evaluate-команде.

## 4. Границы задачи

### 4.1. Обязательно входит

- воспроизводимое enrichment текущих split;
- полная `board119-v1` история каждой позиции;
- legal action mask и target `az73-v1`;
- WDL target из результата партии;
- immutable personal dataset segments;
- dataset manifest и checksums;
- lazy dataset loader;
- stateful weighted game-capped sampler;
- supervised policy/value loss;
- baseline validation `base_rl` до обучения;
- train/validation loop;
- detailed validation slices;
- early stopping;
- best/periodic/stop snapshots;
- полный resume;
- `personal_supervised` playable export;
- smoke game/export validation;
- CLI и документация;
- CPU tests и короткий MPS smoke.

### 4.2. Не входит

- human-feedback из web UI;
- использование `data/human_feedback`;
- персональный RL после fine-tuning;
- смешанный RL/style loss;
- отдельный style prior во время MCTS;
- изменение web-интерфейса;
- повторный Stockfish-анализ;
- изменение порогов 82%/85%;
- новое train/val/test разбиение;
- подбор параметров на test;
- автоматический запуск долгого обучения;
- заявление, что модель играет сильнее `base_rl`, только по move prediction;
- коммит больших derived segments, weights, snapshots или runs.

## 5. Новый формат personal dataset

Добавить формат `chessy-personal-dataset-v1`.

Рекомендуемая структура:

```text
data/personal/encoded/
  segments/
    train-00000-<sha>/
      samples.npz
      metadata.jsonl
      manifest.json
      checksums.sha256
    val-00000-<sha>/
      ...
    test-00000-<sha>/
      ...
  manifests/
    personal-dataset-<fingerprint>.json
```

`data/personal/encoded/` добавить в `.gitignore`. В Git входят только код,
форматы и маленькие fixture manifests, но не реальные encoded samples.

Каждый segment immutable и публикуется через temporary sibling directory,
fsync, verification и atomic rename.

### 5.1. `samples.npz`

Только NumPy arrays, загрузка строго с `allow_pickle=False`:

- `boards`: `uint8 [N,119,8,8]`;
- `legal_offsets`: `int64 [N+1]`;
- `legal_actions`: `uint16 [K]`;
- `target_action`: `uint16 [N]`;
- `value_class`: `uint8 [N]`;
- `game_index`: `uint32 [N]`;
- `ply`: `uint16 [N]`;
- `sample_kind`: `uint8 [N]`;
- `source`: `uint8 [N]`;
- `color`: `uint8 [N]`;
- `phase`: `uint8 [N]`.

Переиспользовать lossless board codec из `chessy.replay.codec`, не создавать
второй несовместимый формат упаковки `board119-v1`.

Fixed enums записать в manifest, например:

```json
{
  "sample_kind": {"good_move": 0, "full_game": 1},
  "source": {"chess.com": 0, "lichess": 1},
  "color": {"black": 0, "white": 1},
  "phase": {"opening": 0, "middlegame": 1, "endgame": 2},
  "value_class": {"loss": 0, "draw": 1, "win": 2}
}
```

### 5.2. `metadata.jsonl`

На sample сохранить audit metadata:

- исходный split row index;
- game index;
- source/date/url;
- ply/move number;
- color;
- original FEN;
- move UCI/SAN;
- move accuracy;
- game accuracy;
- sample kind;
- game result;
- target action;
- value class;
- encoded board checksum или стабильный sample ID.

Не хранить usernames, cookies, API tokens или лишние персональные данные.

## 6. Enrichment pipeline

Добавить воспроизводимую команду, например:

```bash
uv run chessy dataset personal build \
  --splits data/personal/splits/manifest.json \
  --chess-com-pgn data/raw/chess_com_mu1876.pgn \
  --lichess-pgn data/raw/lichess_mu1878.pgn \
  --game-quality data/quality/game_quality.csv \
  --output data/personal/encoded
```

### 6.1. Восстановление индексов партий

Сохранить точную существующую индексацию:

1. Chess.com games начинаются с index 0.
2. Lichess games продолжают индексацию после числа Chess.com records.
3. Использовать ту же filtering/standard-chess логику, что
   `filter_quality_games._read_records`.
4. Не импортировать исполняемый script как скрытую library dependency;
   вынести общий безопасный PGN reader в `chessy.data` либо реализовать
   versioned reader и покрыть parity-тестом.

Для каждого split row:

- найти соответствующую партию;
- переиграть её с начальной позиции;
- перед target ply получить `ChessEnvironment.history(8)`;
- сверить текущий `board.fen()` с сохранённым `fen`;
- проверить, что `move_uci` легален;
- проверить `encode_move`/`decode_action` round-trip;
- сохранить полный `encode_board(history)`;
- сохранить все legal actions;
- определить WDL относительно `board.turn`.

Простое создание восьми независимых boards из FEN недостаточно: оно потеряет
move stack и признаки повторения. История должна строиться при последовательном
переигрывании исходной PGN.

### 6.2. Value target

Результат брать из `data/quality/game_quality.csv` и сверять с PGN headers.

- `1-0`: win для белых, loss для чёрных;
- `0-1`: win для чёрных, loss для белых;
- `1/2-1/2`: draw;
- `*`, отсутствующий или конфликтующий результат — ошибка build, не draw.

Поскольку sample сделан перед ходом владельца, `board.turn` должен совпадать с
полем `color`.

### 6.3. Phase labels

Зафиксировать простое правило:

- opening: `ply <= 20`;
- endgame: на доске не более 10 фигур;
- middlegame: всё остальное.

Порядок проверки: сначала endgame по material, затем opening, затем middlegame,
чтобы ранняя искусственная эндшпильная позиция не считалась opening.

### 6.4. Segment boundaries

- Не смешивать split в одном segment.
- Размер по умолчанию: 16384 samples.
- Не разделять одну партию между segments, если это не превышает hard limit.
- Сортировка стабильна: `(game_index, ply)`.
- Повторный build с теми же inputs даёт те же payload fingerprints.
- `created_at` не участвует в content fingerprint.

## 7. Dataset manifest

Manifest содержит:

- format/version;
- encodings `board119-v1` и `az73-v1`;
- source PGN paths и SHA-256;
- source split manifest path и SHA-256;
- game-quality path и SHA-256;
- account/source mapping;
- thresholds 82%/85% из upstream manifest;
- ordered segments для каждого split;
- checksum каждого segment manifest/checksums;
- sample/game counts;
- histograms kind/source/color/phase/value;
- min/max dates;
- enum mappings;
- frozen test fingerprint;
- общий content fingerprint.

При загрузке:

- запретить symlink и unsafe paths;
- проверить exact checksums;
- проверить shapes/dtypes/offsets/action bounds;
- проверить target action внутри legal actions;
- проверить уникальность sample IDs;
- проверить отсутствие game overlap между split;
- проверить counts/histograms;
- отклонить missing/corrupt segment;
- не пропускать плохие строки молча.

## 8. Test firewall

Test split должен быть технически защищён от случайного использования.

Обычные API:

```python
PersonalDataset(manifest, split="train")
PersonalDataset(manifest, split="val")
```

Попытка `split="test"` без специального capability/acknowledgement должна
завершаться ошибкой.

Train CLI вообще не принимает test split.

Опциональная финальная команда:

```bash
uv run chessy personalize evaluate-test \
  --model <personal-supervised-export> \
  --dataset <manifest> \
  --acknowledge-final-test
```

Она:

- требует точный явный флаг;
- никогда не запускается из обычных тестов/CI;
- пишет audit report с model/dataset checksums и временем;
- не изменяет model, config или best selection.

В рамках реализации шага команду можно добавить и протестировать только на tiny
fixture test split. На реальном `test.jsonl` её не запускать.

## 9. Personal dataset loader

Loader должен:

- лениво читать immutable NPZ segments;
- использовать bounded LRU cache;
- возвращать float32 boards;
- собирать legal mask `[4672]`;
- возвращать scalar target action;
- возвращать WDL class;
- возвращать game/kind/source/color/phase metadata;
- не мутировать cached arrays;
- поддерживать deterministic iteration validation split;
- предоставлять быстрый lookup indices per game/kind.

Не создавать dense one-hot policy targets на диске.

## 10. Weighted game-capped sampler

Добавить `PersonalBatchSampler` со state format
`chessy-personal-sampler-v1`.

Алгоритм эпохи:

1. Для каждой train game детерминированно перемешать её samples.
2. Взять не более `max_positions_per_game=16`.
3. Выполнить weighted random ordering без replacement:
   - `good_move=0.75`;
   - `full_game=1.0`.
4. Перемешать общий epoch pool своим generator.
5. Нарезать batches.
6. Последний неполный batch либо вернуть, либо drop по явной config option.

Требования:

- длинные партии не доминируют;
- sample не повторяется в одной эпохе;
- одна партия даёт максимум 16 samples;
- одинаковые seed/manifest дают одинаковую эпоху;
- state содержит epoch, cursor, pool/permutation и RNG;
- resume возвращает точно следующий batch;
- смена manifest/config несовместима с resume;
- логируется фактическое распределение sample kinds/sources/colors.

Не копировать строки физически ради веса.

## 11. Конфигурация

Расширить `chessy-config-v1` optional-секцией `personalization`. Старые configs
и snapshots должны сохранять fingerprint по фактически сохранённым resolved
bytes, как в шаге 6.

Пример:

```yaml
personalization:
  base_export: artifacts/base_rl
  dataset_manifest: data/personal/encoded/manifests/personal-dataset-....json
  train_split: train
  validation_split: val
  sample_kind_weights:
    good_move: 0.75
    full_game: 1.0
  max_positions_per_game: 16
  policy_loss_weight: 1.0
  value_loss_weight: 0.25
  max_epochs: 30
  early_stopping_patience: 5
  early_stopping_min_delta: 0.0001
  validation_every_epochs: 1
  selection_metric: policy_cross_entropy
  batch_size: 512
  cache_segments: 2
```

Валидация:

- safe relative paths;
- weights > 0;
- `train_split` строго `train`;
- `validation_split` строго `val`;
- max positions > 0;
- loss weights >= 0 и не оба zero;
- max epochs/patience/interval > 0;
- batch/cache > 0;
- selection metric из фиксированного enum;
- personalization и RL sections не смешиваются в одном run config v1.

Добавить:

- `configs/personal-supervised-smoke.yaml`;
- `configs/personal-supervised-local-mps.yaml`.

Smoke config использует tiny fixture dataset/model и не ссылается на реальные
test данные. Local MPS config — стартовая гипотеза, не оптимальный preset.

## 12. Supervised loss

```python
masked_logits = policy_logits.masked_fill(~legal_mask, -inf)
policy_loss = cross_entropy(masked_logits, target_action)
value_loss = cross_entropy(value_logits, value_class)
total = policy_loss + 0.25 * value_loss
```

Проверить:

- target action legal;
- target/value dtypes;
- finite logits/loss/gradients;
- illegal logits не влияют на policy loss;
- корректность WDL class order loss/draw/win;
- per-sample losses доступны для slice metrics;
- source weights не применяются повторно в loss.

Метрики train:

- total/policy/value loss;
- policy top-1;
- true move probability;
- value accuracy;
- gradient norm;
- learning rate;
- samples/sec;
- epoch/cursor;
- effective kind/source/color distribution.

## 13. Validation

Перед первым optimizer step вычислить baseline metrics `base_rl` на полном val.

После каждой validation epoch вычислять без sampling и dropout:

- policy cross-entropy;
- top-1/top-3/top-5 accuracy;
- mean/median probability фактического хода;
- value cross-entropy и accuracy;
- число samples;
- elapsed и samples/sec.

Slices:

- color: white/black;
- sample kind: full_game/good_move;
- source: chess.com/lichess;
- phase: opening/middlegame/endgame.

Для редких slices всегда показывать count. Не сравнивать проценты без размера
выборки.

Validation report format: `chessy-personal-validation-v1`.

Report immutable и содержит:

- model checksum/snapshot step;
- dataset manifest fingerprint;
- split fingerprint;
- metrics overall/slices;
- config fingerprint;
- baseline delta;
- selection metric;
- created_at;
- content fingerprint.

## 14. Best model и early stopping

Primary metric по умолчанию: validation policy cross-entropy overall.

- Lower is better.
- Improvement требует `min_delta`.
- Patience считается validation epochs, не batches.
- Baseline report не тратит patience.
- При improvement создать best snapshot/tag и обновить best pointer.
- При исчерпании patience завершить текущую validation boundary, сохранить
  completed/early-stop snapshot и выйти с кодом 0.
- NaN/Inf validation — ошибка run, не improvement.

Дополнительный export gate:

- best personal metric должна быть лучше base metric минимум на `min_delta`;
- export должен успешно загрузиться;
- legal-move smoke должен пройти.

Если gate не пройден, сохранить candidate и отчёт, но не присваивать ему роль
`personal_supervised`.

## 15. Training run и snapshots

Run создаётся как явный weights-only child `base_rl`:

```json
{
  "parent": {
    "kind": "model-export",
    "role": "base_rl",
    "path": "...",
    "weights_sha256": "...",
    "mode": "weights-only"
  }
}
```

Новый optimizer/scheduler создаются с нуля.

Snapshot должен восстанавливать:

- model;
- optimizer/scheduler;
- sampler epoch/cursor/permutation/RNG;
- Python/NumPy/Torch CPU/MPS RNG;
- global step/samples seen;
- validation epoch;
- best metric/step/report;
- patience counter;
- current phase (`train`, `validation`, `export`, `complete`);
- dataset manifest/fingerprint;
- base model checksum;
- elapsed state.

Разрешить `personal_state` как проверяемое расширение
`chessy-training-state-v1`, аналогично `rl_state`.

Snapshot встраивает актуальный dataset reference. Replay/league references для
чистого supervised run остаются валидными empty references.

Resume:

- проверяет base/dataset checksums;
- не перечитывает test;
- не начинает эпоху заново;
- не повторяет уже зафиксированный validation report;
- даёт тот же следующий batch в deterministic CPU test;
- operational device override логируется.

## 16. Playable export

Лучший прошедший gate checkpoint экспортировать в:

```text
runs/<run-id>/exports/personal-supervised/
```

Использовать существующий `chessy-model-v1`.

Metadata минимум:

- role `personal_supervised`;
- owner accounts (`mu1876`, `mu1878`) без секретов;
- base model checksum;
- dataset fingerprint;
- best validation report fingerprint;
- best epoch/global step;
- selection metric и delta относительно base;
- Git commit/config fingerprint.

Export должен открываться существующей командой `chessy play --model ...` без
изменений игрового API.

## 17. Проверка игровой пригодности

Move-prediction improvement не доказывает игровую силу.

После export выполнить только safety/sanity checks:

- модель загружается;
- inference finite;
- MCTS всегда возвращает legal move;
- минимум две paired игры `personal_supervised` против `base_rl` завершаются
  корректно либо по max plies;
- root noise off, temperature 0;
- report явно помечен `sanity_only=true` и `strength_claim=false`.

Не блокировать export по результату 2–4 игр. Блокировать только по invalid move,
crash, corrupt export или non-finite output.

## 18. CLI

Предлагаемые команды:

```bash
uv run chessy dataset personal build --config configs/personal-supervised-local-mps.yaml
uv run chessy dataset personal inspect --manifest <path>
uv run chessy dataset personal verify --manifest <path>

uv run chessy personalize train --config <config>
uv run chessy personalize train --resume runs/<run-id>
uv run chessy personalize validate --model <export> --dataset <manifest>
uv run chessy personalize compare --base <export> --personal <export> --dataset <manifest>
```

Требования:

- `--config` и `--resume` mutually exclusive;
- `--device` operational override логируется;
- `--stop-after-steps` доступен для smoke/tests;
- все созданные paths печатаются абсолютно;
- ошибки manifests/config короткие и non-zero;
- обычная validate принимает только val;
- никакая команда по умолчанию не читает test;
- cleanup/delete команд нет.

## 19. Целевая структура

```text
configs/
  personal-supervised-smoke.yaml
  personal-supervised-local-mps.yaml
src/chessy/
  data/
    __init__.py
    pgn.py
  personal/
    __init__.py
    schema.py
    builder.py
    segment.py
    manifest.py
    dataset.py
    sampler.py
    metrics.py
    validation.py
  training/
    supervised_loss.py
    personal_state.py
    personal_trainer.py
docs/
  PERSONAL_TRAINING.md
docs/decisions/
  0010-personal-supervised-training.md
tests/
  personal/
  training/
```

Допустимо объединять маленькие модули. Не объединять PGN parsing, filesystem
formats, sampler, trainer и metrics в один файл.

## 20. Обязательные тесты: build

- исходная game indexing совпадает с существующим pipeline;
- FEN каждого fixture sample совпадает при replay;
- history содержит current + семь previous positions;
- repetition planes сохраняются на повторяющейся партии;
- halfmove/en-passant/castling planes корректны;
- target action legal и round-trip;
- WDL меняет перспективу для white/black;
- draw class = 1;
- unfinished/conflicting result отклоняется;
- split game overlap отклоняется;
- deterministic rebuild даёт одинаковые fingerprints;
- game не разделяется между segments без hard-limit причины.

## 21. Обязательные тесты: format/security

- NPZ загружается с `allow_pickle=False`;
- exact filenames/checksums;
- symlink/extra file/path traversal отклоняются;
- dtype/shape/offset bounds проверяются;
- target присутствует в legal actions;
- corrupt payload обнаруживается;
- manifest pin защищает от internally re-checksummed replacement segment;
- duplicate sample IDs/game overlap обнаруживаются;
- train/val loader не открывает test segment;
- test firewall требует explicit acknowledgement.

## 22. Обязательные тесты: sampler/loss

- максимум 16 samples на game/epoch;
- no duplicate sample within epoch;
- kind weights статистически влияют на selection;
- seeded epoch deterministic;
- resume возвращает тот же следующий batch;
- manifest/weights/cap incompatibility отклоняется;
- illegal logits не влияют на policy loss;
- loss совпадает с ручным примером;
- value weight ровно 0.25;
- source weight не применяется второй раз;
- optimizer step меняет weights;
- tiny dataset overfit уменьшает policy loss.

## 23. Обязательные тесты: validation/trainer

- top-1/top-3/top-5 считаются правильно;
- true-move probability и CE корректны;
- slice counts суммируются в overall;
- white/black и phase mappings корректны;
- baseline report создаётся до первого step;
- best metric/min_delta/patience работают;
- test split не участвует в selection;
- export gate не проходит без improvement;
- успешный gate создаёт loadable `personal_supervised` export;
- stop/resume сохраняет optimizer/scheduler/sampler/patience;
- uninterrupted и resumed tiny CPU run дают одинаковые финальные weights,
  следующий batch и best selection;
- subprocess SIGINT создаёт resumable snapshot;
- старые smoke/RL snapshots остаются валидными.

## 24. Tiny fixtures

Не использовать реальные 132k samples в pytest.

Создать маленькие PGN fixtures:

- white win;
- black win;
- draw;
- castling;
- en passant;
- repetition;
- promotion;
- одна deliberately invalid/conflicting запись.

Сделать tiny chronological train/val/test manifest с несколькими games и
samples. Fixture model — 8 channels / 1 residual block.

Integration test должен укладываться в секунды/десятки секунд на CPU.

## 25. Ручная приёмка

Перед сдачей агент выполняет:

```bash
uv lock --check
uv sync --locked
uv run pytest
uv run chessy --help
uv run chessy dataset personal build --help
uv run chessy personalize train --help
```

Затем на tiny fixtures:

1. Создать encoded dataset.
2. Inspect и verify manifest.
3. Создать tiny base export.
4. Запустить baseline validation.
5. Запустить CPU training и остановить после нескольких steps.
6. Выполнить resume до early stop/complete.
7. Проверить snapshots и validation reports.
8. Проверить candidate/personal export.
9. Запустить две sanity arena games.
10. Повторить короткий stop/resume на MPS, если доступен.

После tiny приёмки отдельно собрать реальный encoded train/val/test dataset.
Это разрешённое производное преобразование уже предоставленных данных, но:

- не изменять raw/quality/current split files;
- не запускать реальный long training автоматически;
- не запускать real test evaluation;
- сообщить время build, counts, fingerprints и disk size;
- derived files оставить в gitignored path.

## 26. Реальная data acceptance

Для реального build проверить точные ожидаемые counts:

- train: 106104 samples, 3809 games;
- val: 13337 samples, 478 games;
- test: 13293 samples, 487 games;
- total: 132734 samples;
- no game overlap;
- target illegal count: 0;
- FEN mismatch count: 0;
- unknown/conflicting result count: 0;
- duplicate sample ID count: 0.

Histograms manifest должны совпасть с upstream split manifest по kind/source.

Если реальные данные не проходят эти инварианты, не «чинить» строки молча.
Остановить build с диагностикой game index/ply/source.

## 27. Критерии готовности

Шаг готов, когда одновременно выполнено:

1. Все 132734 примера воспроизводимо обогащаются `board119-v1` history и WDL.
2. Реальные split counts и upstream histograms совпадают.
3. Personal segments immutable, checksummed и безопасно загружаются.
4. Test технически исключён из обычного train/validation.
5. Weighted sampler соблюдает cap 16 positions/game/epoch.
6. Fine-tuning начинается из точного base export checksum.
7. Baseline validation сохранён до обучения.
8. Trainer использует `L_policy + 0.25 * L_value`.
9. Best model выбирается только по validation.
10. Early stopping работает по validation epochs.
11. Stop/resume не сбрасывает sampler/optimizer/scheduler/patience.
12. Tiny uninterrupted и resumed CPU runs эквивалентны.
13. MPS smoke проходит.
14. Export gate не выдаёт роль без validation improvement.
15. Прошедший gate export загружается игровым интерфейсом.
16. Sanity arena не содержит illegal/crashed games.
17. Полный pytest зелёный.
18. Большие dataset/model/run artifacts не попали в Git.

## 28. Что агент должен вернуть

В отчёте указать:

- архитектуру personal dataset;
- exact real build counts и fingerprint;
- размеры encoded train/val/test;
- время build;
- base export checksum/role;
- формулу loss и sampler semantics;
- baseline validation metrics на fixture;
- best fixture validation metrics и delta;
- результаты CPU stop/resume;
- результаты MPS smoke;
- export gate result;
- sanity arena W/D/L без strength claim;
- результаты полного pytest;
- пути созданных локальных artifacts;
- известные ограничения;
- `git status --short`.

Коммит не создавать. Ревью, feature-ветку, commit и push выполнит основной
агент по отдельному запросу пользователя.

## 29. Запрещённые упрощения

Нельзя считать шаг выполненным, если:

- fine-tuning стартует со случайных weights вместо явного base export;
- dataset хранит только FEN без корректной восьмипозиционной history;
- repetition history теряется;
- value target берётся из Stockfish evaluation;
- одна длинная партия доминирует эпоху;
- веса 0.75/1.0 применяются и sampler-ом, и loss одновременно;
- validation вычисляется на случайной подвыборке без явного режима;
- test читается обычным trainer-ом;
- best выбирается по train loss;
- export получает роль без improvement относительно base;
- resume начинает эпоху заново;
- corrupt/missing segment тихо пропускается;
- исходные split/raw/quality файлы переписываются;
- реальные test metrics запускаются в процессе разработки;
- smoke arena объявляется доказательством усиления;
- generated NPZ, Safetensors, snapshots или runs попадают в PR.

Главный принцип шага: Chessy должен перенять твои решения измеримо и
воспроизводимо, не подглядывая в test и не забывая, из какой именно базовой
модели и каких именно партий он получился.
