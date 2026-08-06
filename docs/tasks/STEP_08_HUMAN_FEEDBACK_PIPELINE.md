# Шаг 8. Human-feedback: от подтверждённой партии до следующей персональной модели

## 1. Контекст и обязательная предпосылка

Этот шаг начинается только после merge PR шага 7 и обновления локального
`main`. Рабочую ветку создать от актуального `main`:

```bash
git switch main
git pull --ff-only
git switch -c feature/human-feedback-pipeline
```

Не продолжать работу поверх незамерженной ветки
`feature/personal-supervised-training`.

К началу шага в проекте уже есть:

- локальный игровой интерфейс и выбор model export;
- выключенная по умолчанию галочка добавления ходов в обучение;
- повторное подтверждение после окончания партии;
- атомарная запись `chessy-human-feedback-v1`;
- `game.pgn`, `samples.jsonl` и `manifest.json` для подтверждённой партии;
- история FEN перед каждым человеческим ходом;
- immutable historical personal dataset;
- supervised personalization trainer;
- validation/test firewall;
- полные snapshots и точный stop/resume.

Нельзя заново реализовывать игровой UI или дублировать существующий writer.
Нужно достроить отсутствующий путь от сохранённой партии до следующего
обучающего run.

## 2. Цель шага

Реализовать полный проверяемый цикл:

```text
игра против Chessy
    -> явное подтверждение пользователя
    -> immutable raw feedback game
    -> verify/replay
    -> versioned feedback dataset
    -> mixed historical + feedback batches
    -> supervised update от personal model
    -> validation gate
    -> playable personal_feedback export
```

Главный пользовательский инвариант:

- если сохранение не было включено и подтверждено, партия не создаёт ни одного
  обучающего примера;
- если сохранение подтверждено, только ходы пользователя входят в targets и
  гарантированно появляются в явно выбранном следующем training run.

## 3. Что не входит в шаг

- персональный self-play RL и смешанный RL/style loss;
- изменение MCTS ради имитации стиля;
- автоматический запуск обучения сразу после партии;
- использование Stockfish для разметки live-feedback;
- добавление ходов бота как human policy targets;
- изменение замороженного historical test;
- облачный сервер, база данных или открытие внешнего порта;
- заявление об усилении модели по tiny smoke;
- автоматическое объявление каждой сыгранной партии ground truth без consent.

Персональный RL остаётся следующим отдельным шагом.

## 4. Зафиксированные продуктовые решения

### 4.1. Consent

Для записи нужны оба действия:

1. пользователь включил галочку до начала партии;
2. пользователь подтвердил сохранение после завершения партии.

Значение по умолчанию — `false`. PGN можно скачать независимо от consent.
Нажатие «Не сохранять», закрытие вкладки или новая партия не должны создавать
feedback artifact.

### 4.2. Какие ходы являются targets

- позиции и ходы пользователя — policy/value samples;
- ходы Chessy сохраняются только как контекст PGN и history;
- target action всегда должен быть легален в восстановленной позиции;
- WDL берётся из результата партии с точки зрения пользователя;
- evaluation движка не используется как target;
- незавершённая партия не допускается в dataset.

### 4.3. Приоритет feedback

Стартовые параметры:

```yaml
sample_weight: 4.0
max_batch_fraction: 0.25
max_positions_per_game: 16
```

Вес применяется к per-sample loss, а не физическим копированием строки.
Feedback занимает не более 25% любого mixed batch. Одна длинная партия даёт не
более 16 targets за эпоху.

### 4.4. Test firewall

Human-feedback существует отдельным train-only потоком. Он никогда:

- не добавляется в historical `train.jsonl`, `val.jsonl` или `test.jsonl`;
- не меняет frozen test fingerprint;
- не читается командой final historical test evaluation;
- не используется как причина менять содержимое исторических split.

## 5. Форматы артефактов

### 5.1. Raw game

Сохранить обратную совместимость с существующим
`chessy-human-feedback-v1`:

```text
data/human_feedback/<game-id>/
  game.pgn
  samples.jsonl
  manifest.json
```

Существующие v1 artifacts должны остаться читаемыми. Формат нельзя молча
переопределять. Если для новых записей нужен расширенный формат, ввести
`chessy-human-feedback-v2` и явный migration/reader, но не переписывать старые
директории.

Raw directory после публикации считается immutable. Допустима только
идемпотентная повторная попытка сохранить те же байты под тем же game ID.

### 5.2. Feedback dataset

Добавить отдельный формат:

```text
chessy-human-feedback-dataset-v1
chessy-human-feedback-segment-v1
```

Не расширять enum `source` внутри уже опубликованного
`chessy-personal-dataset-v1`: это изменило бы смысл старых manifests. Feedback
должен иметь собственный loader/segment format и объединяться с historical
dataset на уровне mixed sampler/trainer.

Рекомендуемая структура:

```text
data/human_feedback_encoded/
  segments/
    feedback-00000-<fingerprint>/
      samples.npz
      metadata.jsonl
      manifest.json
      checksums.sha256
  manifests/
    feedback-dataset-<fingerprint>.json
```

Generated NPZ и локальные manifests должны быть gitignored.

### 5.3. Arrays segment

Минимальный payload:

- `boards`: `uint8 [N,119,8,8]`;
- `legal_offsets`: `int64 [N+1]`;
- `legal_actions`: `uint16 [sum legal]`;
- `target_action`: `uint16 [N]`;
- `value_class`: `uint8 [N]`, порядок loss/draw/win;
- `game_index`: `uint32 [N]`, локальный индекс внутри manifest;
- `ply`: `uint16 [N]`;
- `color`: `uint8 [N]`;
- `phase`: `uint8 [N]`;
- `sample_weight`: `float32 [N]`.

UUID игры хранится в metadata и в manifest game table. Не пытаться упаковать
UUID напрямую в `uint32`.

### 5.4. Dataset manifest

Manifest должен содержать:

- format и encoding versions;
- ordered список raw games;
- для каждой игры: ID, created_at, raw manifest checksum, PGN checksum,
  samples checksum, model ID/checksum, human color, result, termination;
- segments с checksum, sample count и payload fingerprint;
- game/sample counts;
- histograms по color/result/phase/model/time control;
- применённые `sample_weight` и `max_positions_per_game`;
- content fingerprint;
- created_at, исключённый из content fingerprint.

Одинаковый набор raw games и одинаковые параметры дают тот же content
fingerprint. Добавление новой подтверждённой партии создаёт новый manifest, не
изменяя старый.

## 6. Строгая проверка raw feedback

Добавить модуль `chessy.feedback` и функцию уровня
`verify_feedback_game(path)`.

Проверка должна fail closed:

- path — обычная директория, не symlink;
- имя директории — безопасный UUID/game ID;
- только ожидаемые обычные файлы;
- hashes из manifest совпадают;
- manifest и каждая строка имеют поддерживаемый format;
- `game_id` совпадает между директорией, manifest и samples;
- `human_samples` совпадает с числом строк;
- `sample_weight` конечный и положительный;
- model checksum имеет валидный формат;
- result/termination/time control/color/MCTS metadata валидны;
- PGN читается как одна standard-chess партия;
- PGN завершён и result совпадает с manifest;
- human color определяется одинаково по PGN и manifest;
- партия replay-ится от начальной позиции без illegal moves;
- samples соответствуют только человеческим plies;
- `fen`, raw en-passant square и `history_fens` совпадают с replay;
- `move_uci` и `action` совпадают с общим `az73-v1` codec;
- action legal и round-trip decode возвращает тот же move;
- дубликатов ply/sample ID нет;
- bot plies отсутствуют среди targets;
- human WDL согласован с итогом PGN.

Временные директории `.tmp-*` при scan игнорируются. Любая обычная на вид, но
повреждённая feedback directory должна останавливать build с точным game ID и
причиной, а не пропускаться.

## 7. Усиление writer без нарушения совместимости

Проверить текущий `save_human_feedback` и сохранить его свойства:

- sibling temporary directory;
- fsync payload перед rename;
- atomic publish;
- idempotence;
- session lock;
- отсутствие записи до окончания партии;
- отсутствие записи без opt-in.

Доработать при необходимости:

- валидировать уже существующий destination перед признанием idempotent;
- отклонять symlink destination/root;
- fsync `game.pgn`, `samples.jsonl`, `manifest.json` и parent directory;
- включить deterministic sample ID;
- явно записать board/action encoding versions;
- не брать training weight из глобальной константы как единственный источник
  истины: raw хранит suggested/default weight для аудита, а фактический weight
  pin-ится training config/dataset manifest;
- сохранять duration/clock metadata, если оно уже доступно без изменения
  игрового поведения.

Не сохранять IP, browser fingerprint, cookies или другие ненужные персональные
данные.

## 8. Dataset builder

Добавить API и CLI, которые:

1. безопасно сканируют raw root;
2. сортируют игры детерминированно по `(created_at, game_id)`;
3. полностью verify каждую игру;
4. replay-ят PGN и историю;
5. кодируют только human moves в `board119-v1` и `az73-v1`;
6. строят sparse legal actions;
7. назначают WDL;
8. создают immutable segments;
9. публикуют content-addressed manifest;
10. повторный build того же набора возвращает тот же artifact.

Команды:

```bash
uv run chessy feedback inspect --input data/human_feedback
uv run chessy feedback verify --game data/human_feedback/<game-id>
uv run chessy feedback build \
  --input data/human_feedback \
  --output data/human_feedback_encoded
uv run chessy feedback dataset-verify --manifest <feedback-manifest>
```

`inspect` не создаёт файлов. `build` не переписывает raw games. Пустой input
должен завершаться понятной ошибкой, а не создавать фиктивный dataset.

## 9. FeedbackDataset

Добавить ленивый read-only loader с:

- строгой manifest/segment verification;
- bounded segment cache;
- dense legal mask только в памяти batch;
- float32 board decode;
- metadata game/color/phase/model/time control;
- быстрыми indices per game;
- запретом mutation cached arrays;
- отсутствием API для historical test.

Loader возвращает те же training keys, что historical `PersonalDataset`, плюс
`sample_weight` и `stream="human_online"`.

## 10. Mixed sampler

Добавить resumable `MixedPersonalBatchSampler` с отдельным versioned state.

Правила каждого batch:

- historical поток остаётся основой;
- feedback samples занимают не более
  `floor(batch_size * max_batch_fraction)`;
- при ненулевом feedback и достаточно большом batch выделяется хотя бы одно
  feedback место;
- feedback выбирается без replacement внутри своего cycle;
- максимум 16 human positions одной игры за historical epoch;
- при исчерпании feedback pool начинается новый детерминированно перемешанный
  cycle;
- одна и та же строка не дублируется физически ради веса;
- порядок полностью определяется seed, manifests и sampler state;
- incomplete final historical batch не должен нарушать fraction cap;
- `drop_last` имеет однозначную семантику;
- batch metadata показывает фактическое число samples каждого потока.

State должен сохранять:

- оба manifest fingerprints;
- historical sampler state;
- feedback pool/permutation/cursor/cycle;
- generator state;
- batch size, fraction, caps и seed;
- epoch и следующий точный batch.

Любое изменение manifest или sampler config несовместимо с full-state resume.

## 11. Loss

Для historical samples weight равен `1.0`. Для feedback по умолчанию `4.0`.

Per-sample objective:

```text
L_i = policy_weight * CE_policy_i + value_weight * CE_value_i
L_batch = sum(sample_weight_i * L_i) / sum(sample_weight_i)
```

Нормализация на сумму weights обязательна, чтобы изменение числа feedback rows
не превращалось в неявное изменение learning rate.

Проверить:

- illegal logits не влияют на loss;
- weights конечны и положительны;
- historical weight не применяется дважды;
- cap реализован sampler-ом, weight — loss-ом;
- per-stream policy/value loss логируются отдельно;
- нулевой feedback batch корректен;
- mixed batch с одним feedback sample корректен;
- gradients finite.

## 12. Конфигурация

Добавить optional strict-секцию, например:

```yaml
human_feedback:
  enabled: true
  dataset_manifest: data/human_feedback_encoded/manifests/feedback-dataset-<fp>.json
  sample_weight: 4.0
  max_batch_fraction: 0.25
  max_positions_per_game: 16
  cache_segments: 1
  historical_regression_tolerance: 0.01
  feedback_min_delta: 0.0001
```

Для training preset:

```yaml
personalization:
  base_export: artifacts/personal_supervised
  dataset_manifest: <historical-personal-manifest>
```

Валидация config:

- safe relative paths;
- feedback manifest обязателен при `enabled=true`;
- weight > 0;
- fraction строго в `(0, 0.5]`, default 0.25;
- cap/cache > 0;
- tolerances конечны и >= 0;
- feedback нельзя включать в RL config этого шага;
- historical dataset reference остаётся обязательным;
- resolved config и snapshot pin-ят оба manifests.

Добавить:

- `configs/personal-feedback-smoke.yaml`;
- `configs/personal-feedback-local-mps.yaml`.

Smoke fixture генерируется в ignored path и не зависит от пользовательских
реальных feedback games.

## 13. Новый training run

Не возобновлять completed run шага 7 как будто feedback существовал в нём.
Создавать новый weights-only child от явного export с ролью:

- `personal_supervised`; или
- предыдущий проверенный `personal_feedback`.

Run parent фиксирует path, role и weights checksum. Run references фиксируют:

- historical personal manifest;
- feedback dataset manifest;
- base export checksum.

Новый optimizer/scheduler создаются с нуля. Full-state resume продолжает только
тот же feedback run с теми же manifests.

Предлагаемая команда:

```bash
uv run chessy personalize feedback --config configs/personal-feedback-local-mps.yaml
uv run chessy personalize feedback --resume runs/<run-id>
```

Не перегружать поведение существующей команды неоднозначным auto-detection.

## 14. Validation и выбор best

Historical validation остаётся основной защитой от забывания и никогда не
семплируется. Historical test не читается.

Перед первым optimizer step сохранить два baseline report:

1. полный historical `val`;
2. полный feedback dataset как adaptation diagnostic.

Feedback diagnostic измеряется на обучающих feedback rows и потому не является
оценкой generalization. Это должно быть явно записано в report/UI/docs.

На validation boundary вычислять:

- historical val policy/value metrics и slices;
- feedback policy/value metrics;
- delta к обоим baseline;
- per-model/per-color/per-phase feedback slices;
- effective mixed distribution с начала run.

Best candidate допустим, если одновременно:

```text
historical_val_ce <= base_historical_val_ce + regression_tolerance
feedback_ce <= best_feedback_ce - feedback_min_delta
```

Среди допустимых candidates выбирать минимальный feedback CE. Если historical
regression превышен, candidate не становится best независимо от train loss.

Early stopping считает validation boundaries. NaN/Inf — ошибка run.

## 15. Export gate

Успешный export получает роль `personal_feedback` и metadata:

- generation/version;
- parent role/path/checksum;
- historical dataset fingerprint;
- feedback dataset fingerprint;
- число feedback games/samples;
- sample weight и max batch fraction;
- best historical/feedback reports;
- deltas и tolerances;
- config fingerprint.

Gate требует:

- feedback CE improvement минимум на `feedback_min_delta`;
- historical regression не хуже tolerance;
- export checksum и строгую загрузку;
- две legal sanity arena games;
- отсутствие обращения к historical test.

Если gate не пройден, best candidate/snapshots/reports сохраняются, но export с
ролью `personal_feedback` не публикуется.

## 16. Snapshots и resume

Расширить training state отдельным versioned feedback state:

- phase;
- global step/samples seen;
- historical epoch;
- feedback cycle/cursor;
- baseline reports;
- best metrics/report/step;
- patience;
- base checksum;
- оба dataset fingerprints;
- cumulative stream counts;
- elapsed time.

Snapshot должен по-прежнему сохранять model, optimizer, scheduler, RNG и mixed
sampler. Stop возможен на любом completed batch.

Проверить:

- uninterrupted CPU run и stop/resume дают побитово одинаковые weights;
- следующий mixed batch совпадает;
- feedback cycle не начинается заново;
- cap/fraction не меняются после resume;
- completed resume — idempotent no-op;
- crash между staged export и final publish безопасно возобновляется;
- старые smoke/RL/personal snapshots читаются.

## 17. Метрики

Train metrics:

- total/policy/value loss;
- historical policy/value loss;
- feedback policy/value loss;
- feedback weighted contribution;
- batch historical/feedback counts;
- actual feedback fraction;
- sample weight sum;
- top-1 и true-move probability per stream;
- value accuracy per stream;
- gradient norm, LR, samples/sec;
- feedback game distribution.

Validation report должен быть immutable и содержать model checksum, snapshot
step, оба manifest fingerprints, counts, deltas, config fingerprint и content
fingerprint.

## 18. UI/API scope

Существующий интерфейс уже реализует нужный UX. В этом шаге разрешены только
точечные исправления:

- checkbox остаётся выключенным по умолчанию;
- post-game confirmation обязателен;
- после успешной записи показывается game ID и понятный статус;
- повторный запрос идемпотентен;
- отказ не отправляет confirm API;
- PGN download работает независимо;
- ошибки диска показываются пользователю без ложного статуса «сохранено».

Не добавлять кнопку «обучить сейчас»: dataset build и training остаются явными
локальными командами, чтобы пользователь контролировал момент и base model.

## 19. Безопасность файлов

Для raw games, encoded segments и manifests:

- запрещены absolute paths, `..`, symlinks и special files;
- проверяется containment после resolve;
- checksum list покрывает payload ровно один раз;
- `np.load(..., allow_pickle=False)`;
- `torch.load` для dataset не используется;
- JSON/JSONL имеют bounds на размер и число строк;
- publication через sibling temp + fsync + rename;
- destructive cleanup применяется только к собственной временной директории;
- неизвестные файлы вызывают ошибку;
- corrupt game/segment не пропускается.

## 20. Тесты

### 20.1. Consent/API

- default opt-out не создаёт directory;
- opt-in без post-confirmation не создаёт directory;
- confirm до окончания запрещён;
- confirm после окончания создаёт ровно один artifact;
- повторный confirm идемпотентен;
- decline не создаёт samples;
- PGN доступен во всех случаях;
- disk error не выставляет `feedback_saved=true`.

### 20.2. Raw verifier

- white win, black win, draw;
- resignation, timeout, mate и max-plies;
- castling, en passant, promotion, repetition history;
- human white и black;
- tampered PGN/sample/manifest/hash;
- illegal move/action mismatch;
- wrong model checksum/result/history;
- duplicated/missing human ply;
- bot move injected as target;
- symlink/path traversal/special file;
- incomplete temp directory.

### 20.3. Builder/dataset

- deterministic fingerprint;
- adding one game creates a new manifest;
- old manifest remains unchanged;
- exact game/sample counts and histograms;
- board119/action round-trip;
- WDL perspective;
- segment corruption detected;
- no mutation of raw data;
- historical frozen test bytes/fingerprint unchanged.

### 20.4. Sampler/loss

- no batch exceeds feedback fraction;
- one game never exceeds cap;
- no duplicate inside feedback cycle;
- weight changes loss but not sample multiplicity;
- weighted normalization correct;
- deterministic seed;
- exact next batch after state restore;
- tiny/incomplete batches;
- zero feedback rejected at build/config boundary.

### 20.5. Trainer

- exact uninterrupted/resumed CPU equivalence;
- MPS stop/resume smoke;
- baseline reports before step 1;
- historical regression gate;
- feedback improvement gate;
- no export when either gate fails;
- successful loadable `personal_feedback` export;
- two legal arena games;
- historical test never opened;
- completed resume no-op;
- backwards compatibility старых snapshots/configs.

Не использовать реальные пользовательские feedback games в pytest.

## 21. Tiny fixtures

Создать генерируемые fixtures в ignored `runs/` или pytest `tmp_path`:

- минимум три партии с человеком белыми/чёрными и ничьёй;
- отдельные партии/позиции для castling, en passant, promotion, repetition;
- одна повреждённая запись;
- tiny 8-channel model export с ролью fixture;
- tiny historical train/val dataset;
- feedback dataset из нескольких human moves.

Fixture config может явно разрешать fixture base, production config — нет.

## 22. Ручная приёмка

Перед сдачей выполнить:

```bash
uv lock --check
uv sync --locked
uv run pytest
uv run chessy --help
uv run chessy feedback --help
uv run chessy personalize feedback --help
```

Затем на tiny fixtures:

1. Проверить opt-out и отсутствие файлов.
2. Сохранить подтверждённую игру.
3. Verify raw artifact.
4. Построить feedback dataset.
5. Inspect/verify manifest.
6. Запустить CPU training и остановить после нескольких steps.
7. Resume до complete.
8. Сравнить с uninterrupted run.
9. Проверить оба baseline и best reports.
10. Проверить успешный и отклонённый export gate.
11. Загрузить export игровым API.
12. Сыграть две sanity arena games.
13. Повторить короткий stop/resume на MPS.

Если в `data/human_feedback/` уже есть реальные подтверждённые партии, разрешено
только read-only inspect/verify. Не включать их в training автоматически без
отдельной команды пользователя. Не запускать historical test evaluation.

## 23. Предлагаемые файлы

```text
configs/
  personal-feedback-smoke.yaml
  personal-feedback-local-mps.yaml
docs/
  HUMAN_FEEDBACK_TRAINING.md
  decisions/0011-human-feedback-pipeline.md
src/chessy/
  feedback/
    __init__.py
    raw.py
    builder.py
    segment.py
    dataset.py
    sampler.py
    validation.py
  training/
    feedback_state.py
    feedback_trainer.py
tests/
  feedback/
    test_raw_feedback.py
    test_feedback_dataset.py
    test_mixed_sampler.py
    test_feedback_training.py
```

Точный layout можно скорректировать, но raw verification, dataset format,
sampler и trainer не должны сливаться в один монолитный файл.

## 24. Порядок реализации

1. Провести gap analysis текущего writer/UI и зафиксировать ADR.
2. Реализовать strict raw verifier и негативные security tests.
3. Усилить writer, сохранив чтение v1.
4. Реализовать immutable feedback segments/manifest.
5. Реализовать builder и CLI inspect/verify/build.
6. Реализовать `FeedbackDataset`.
7. Реализовать mixed sampler и exact resume tests.
8. Добавить weighted per-sample loss и stream metrics.
9. Расширить strict config schema/presets.
10. Реализовать отдельный feedback trainer и snapshots.
11. Реализовать dual validation и export gate.
12. Добавить load/arena sanity gate.
13. Выполнить CPU tests и MPS smoke.
14. Обновить документацию и отчёт.

После каждого блока запускать targeted tests; перед сдачей — полный suite.

## 25. Критерии готовности

Шаг готов только если одновременно выполнено:

1. Opt-out/decline не оставляют training artifacts.
2. Confirmed finished game сохраняется атомарно и идемпотентно.
3. Raw verifier replay-ит PGN/history/action и обнаруживает corruption.
4. Только human moves становятся targets.
5. Versioned feedback manifest воспроизводим и immutable.
6. Historical test не меняется и не читается.
7. Mixed batch никогда не превышает feedback fraction.
8. Одна feedback game не превышает per-epoch cap.
9. Weight 4.0 применяется в loss без физического дублирования.
10. Новый run стартует от точного personal export checksum.
11. Оба baseline report созданы до optimizer step.
12. Best выбирается по feedback improvement при ограничении historical regression.
13. Stop/resume восстанавливает точный следующий mixed batch и weights.
14. Gate не публикует export при забывании historical style.
15. Успешный export загружается UI и играет legal games.
16. MPS smoke проходит.
17. Полный pytest зелёный.
18. Generated PGN/NPZ/weights/runs не попадают в Git.

## 26. Что агент должен вернуть

В итоговом отчёте указать:

- найденные gaps текущего feedback writer/UI;
- raw и encoded artifact formats;
- fixture game/sample counts и fingerprints;
- sampler semantics и фактическую feedback fraction;
- формулу weighted loss;
- base export role/checksum;
- historical и feedback baseline/best metrics;
- export gate result;
- CPU uninterrupted/resume equivalence;
- MPS smoke result;
- arena W/D/L без заявления о силе;
- полный pytest result;
- созданные ignored artifact paths и размеры;
- подтверждение, что historical test fingerprint не изменился;
- известные ограничения;
- `git status --short`.

Коммит не создавать. Основной агент после ревью сам создаст feature-ветку,
коммит и push по принятому процессу.

## 27. Запрещённые упрощения

Нельзя считать шаг выполненным, если:

- наличие checkbox само по себе считается готовым pipeline;
- training читает raw JSONL без immutable manifest/checksums;
- bot moves становятся human targets;
- feedback rows копируются четыре раза;
- batch fraction соблюдается только «в среднем»;
- resume начинает feedback pool заново;
- новая партия молча меняет manifest уже начатого run;
- historical val/test объединяются с feedback;
- best выбирается только по feedback train loss без regression gate;
- роль `personal_feedback` ставится до load/legal-game gate;
- corrupted game пропускается;
- decline создаёт пустой marker или samples;
- реальные пользовательские партии автоматически запускают обучение;
- smoke результат объявляется доказательством силы.

Главный принцип шага: пользователь явно решает, какая новая партия становится
обучающим сигналом, а Chessy усваивает её сильно, измеримо и обратимо, не
забывая уже изученный стиль и не касаясь замороженного test.
