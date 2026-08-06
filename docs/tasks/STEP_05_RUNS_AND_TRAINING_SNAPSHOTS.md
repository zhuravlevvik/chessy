# Chessy — шаг 5: инфраструктура run и полные training snapshots

Статус: готово к реализации
Дата постановки: 2026-08-06
Рабочая директория: `/Users/zhuravlevvikt/Documents/codex_projects/chessy`

## 1. Цель задачи

Создать надёжную инфраструктуру экспериментов до начала долгого self-play и
обучения. Любой будущий trainer должен уметь:

- получить строгую resolved-конфигурацию;
- создать идентифицируемый run;
- писать события и метрики;
- атомарно сохранять полный `chessy-snapshot-v1`;
- корректно остановиться после текущего batch;
- продолжить тот же run через `resume` без сброса состояния;
- создать новый эксперимент через явный `fork`;
- пережить повреждение последнего snapshot, используя предыдущий валидный;
- соблюдать retention без потери best/stage snapshots.

Главный критерий шага: детерминированный короткий CPU training run после
остановки и `resume` должен дать то же состояние, что и непрерывный run:

- веса модели;
- optimizer;
- scheduler;
- sampler;
- global step и epoch;
- Python/NumPy/PyTorch RNG;
- следующий batch и следующий набор случайных чисел.

В этом шаге используется маленький synthetic smoke trainer. Он проверяет
инфраструктуру, но не обучает модель шахматам и не читает личный датасет.

## 2. Архитектурные источники

Перед началом полностью прочитать:

- `AGENTS.md`;
- `docs/PROJECT_PLAN.md`, особенно разделы 16, 17, 19, 20 и 22;
- `docs/decisions/0002-compute-backend.md`;
- `docs/decisions/0005-artifact-formats.md`;
- `docs/decisions/0006-config-and-runs.md`;
- `docs/tasks/STEP_03_POLICY_VALUE_MODEL.md`;
- текущие `chessy.model` export/config modules;
- текущий CLI в `src/chessy/cli.py`.

Зафиксированные новые форматы:

- source config: `chessy-config-v1`;
- run manifest: `chessy-run-v1`;
- snapshot: `chessy-snapshot-v1`;
- run state: `chessy-run-state-v1`;
- metrics stream: `chessy-metrics-v1`;
- events stream: `chessy-events-v1`.

Названия `chessy-model-v1`, `board119-v1`, `az73-v1` и
`residual-cnn-v1` не меняются.

## 3. Границы задачи

### 3.1. Обязательно входит

- YAML config loader;
- строгие Pydantic schemas;
- canonical resolved JSON и fingerprint;
- run ID и manifest среды;
- JSONL metrics/events;
- stateful deterministic sampler;
- RNG capture/restore для Python, NumPy, CPU, MPS и CUDA;
- model Safetensors + local trusted training state;
- atomic snapshot directory;
- checksum verification;
- `resume` и `fork`;
- periodic/best/stage tags;
- retention: два последних periodic плюс pinned snapshots;
- graceful SIGINT/SIGTERM;
- corruption fallback;
- synthetic CPU/MPS smoke trainer;
- CLI для smoke, inspect и verify.

### 3.2. Не входит

- self-play actors;
- replay buffer segments;
- настоящий chess dataset/DataLoader;
- RL или supervised loss над шахматными примерами;
- arena/Elo;
- league opponents;
- distributed checkpoints;
- cloud/object storage;
- изменение игрового UI;
- автоматический upload артефактов.

Dataset/replay/league manifests уже входят в snapshot schema, но в smoke run они
будут валидными пустыми reference manifests. Реальные producers появятся на
следующем шаге.

## 4. Безопасность и ограничения

1. Snapshot считается локальным training artifact, а не безопасным файлом для
   обмена. Playable export по-прежнему использует только Safetensors.
2. `training_state.pt` не содержит model weights — они лежат отдельно в
   `model.safetensors`.
3. Содержимое `training_state.pt` ограничить tensors и простыми контейнерами,
   чтобы загружать через `torch.load(..., weights_only=True)`.
4. До загрузки любого файла проверить структуру директории и SHA-256.
5. Не принимать абсолютные или содержащие `..` пути из checksum/index files.
6. Не перезаписывать существующий run/snapshot.
7. Все временные директории создаются только sibling к точному destination.
8. Retention удаляет только проверенные snapshot-директории внутри конкретного
   `runs/<run-id>/snapshots/`. Никогда не применять recursive deletion к
   неразрешённому пути, symlink или run root.
9. Старый run и его snapshots неизменяемы при `fork`.
10. Не читать и не изменять существующие `data/raw`, `data/quality`,
    `data/personal` и `data/human_feedback`.
11. Не коммитить `runs/`, snapshots или synthetic outputs.
12. Не создавать Git-коммит без отдельного запроса пользователя.

## 5. Зависимости

Добавить прямые runtime dependencies через `uv`:

- `pydantic` — сейчас он приходит транзитивно через FastAPI, но config contract
  не должен зависеть от случайной транзитивной зависимости;
- `pyyaml` для YAML source config.

Предпочтительно:

```bash
uv add "pydantic>=2.11,<3"
uv add "pyyaml>=6,<7"
```

Не добавлять Hydra, OmegaConf, MLflow, Weights & Biases, Lightning, DVC или
database. Для персонального локального проекта достаточно прозрачных файловых
форматов.

## 6. Целевая структура

```text
configs/
  smoke.yaml
src/chessy/
  config/
    __init__.py
    schema.py
    loader.py
    canonical.py
  run/
    __init__.py
    identity.py
    manifest.py
    logging.py
    manager.py
  snapshot/
    __init__.py
    schema.py
    rng.py
    writer.py
    loader.py
    retention.py
  training/
    __init__.py
    sampler.py
    state.py
    stop.py
    smoke.py
tests/
  config/
  run/
  snapshot/
  training/
```

Допустимо объединить небольшие файлы, но не складывать config, filesystem I/O,
signals и trainer в один модуль.

## 7. Source config `chessy-config-v1`

### 7.1. Общий YAML

Создать `configs/smoke.yaml` как небольшой, быстро выполняемый пример:

```yaml
format: chessy-config-v1
name: snapshot-smoke
seed: 42
device: cpu

model:
  architecture: residual-cnn-v1
  input_planes: 119
  action_planes: 73
  board_size: 8
  channels: 8
  residual_blocks: 1
  group_norm_groups: 8
  value_channels: 8
  value_hidden: 16
  value_classes: 3

optimizer:
  type: adamw
  learning_rate: 0.0003
  weight_decay: 0.0001
  beta1: 0.9
  beta2: 0.999
  epsilon: 0.00000001

scheduler:
  type: warmup-cosine
  warmup_steps: 2
  total_steps: 12
  minimum_lr_ratio: 0.0

training:
  batch_size: 4
  gradient_clip_norm: 1.0
  snapshot_every_steps: 4
  keep_last_periodic: 2

artifacts:
  runs_dir: runs
  dataset_manifest: null
  replay_manifest: null
  league_manifest: null
```

### 7.2. Pydantic schemas

Использовать Pydantic v2 и `ConfigDict(extra="forbid", strict=True)` на каждом
уровне.

Обязательные модели:

- `ChessyConfig`;
- `ModelConfigSchema`;
- `OptimizerConfig`;
- `SchedulerConfig`;
- `TrainingConfig`;
- `ArtifactsConfig`.

Правила:

- `format` строго `chessy-config-v1`;
- `name` 1–80 символов;
- seed — неотрицательный int;
- device: `auto/cpu/mps/cuda`;
- model schema конвертируется в существующий `ModelConfig` и проходит его
  validation;
- optimizer в v1 только AdamW;
- learning rate/epsilon положительны;
- weight decay неотрицателен;
- beta в `[0,1)`;
- scheduler только warmup-cosine;
- `0 <= warmup_steps < total_steps`;
- batch size и snapshot interval положительны;
- keep periodic минимум 2;
- неизвестные поля запрещены;
- YAML aliases допустимы только если итоговый object проходит schema;
- custom Python YAML tags запрещены через `yaml.safe_load`.

Не подставлять environment variables и не выполнять шаблоны/выражения из YAML.

### 7.3. Source и resolved config

Source YAML сохраняется в run без переписывания пользовательского текста.
Resolved config содержит все default values и сериализуется в canonical JSON:

- UTF-8;
- `sort_keys=True`;
- separators `(",", ":")`;
- `ensure_ascii=False`;
- один завершающий newline;
- только JSON-compatible primitives;
- float NaN/Infinity запрещены.

Функции canonical serialization должны быть общими для config, manifests и
state JSON, а не копироваться из model export.

## 8. Fingerprint и run ID

Config fingerprint:

```text
sha256(canonical_resolved_config_without_trailing_newline)
```

Run ID:

```text
YYYYMMDD-HHMMSS-<slug>-<first-10-fingerprint>
```

- Timestamp UTC.
- Slug lowercase ASCII `[a-z0-9-]`, максимум 40 символов.
- Двойные/краевые дефисы удаляются.
- Одновременная коллизия не перезаписывается: добавить безопасный числовой
  suffix либо создать новый timestamp.
- ID валидируется перед использованием как directory name.

Одинаковый resolved config даёт одинаковый fingerprint, но разные run IDs из-за
timestamp. Изменение любого поля resolved config меняет fingerprint.

## 9. Run directory `chessy-run-v1`

```text
runs/<run-id>/
  config.source.yaml
  config.resolved.json
  run_manifest.json
  events.jsonl
  metrics.jsonl
  snapshots/
    index.json
    step-000000000004/
    step-000000000008/
  exports/
```

Run создаётся через temporary sibling directory и atomic rename. Существующий
run ID никогда не переиспользуется.

### 9.1. run_manifest.json

Минимальные поля:

```text
format = chessy-run-v1
run_id
created_at UTC
name
config_fingerprint
git.commit
git.dirty
uv_lock.sha256
python.version
python.executable
platform.system/machine/release
torch.version
requested_device
resolved_device
project_version
parent = null | {run_id, snapshot, mode}
dataset/replay/league reference hashes
```

Не сохранять username, home directory, access tokens или полный environment.
Абсолютный Python executable допустим только в локальном manifest, но не должен
показываться через игровой API.

Git information получается read-only. Отсутствующий `.git` не ломает запуск:
manifest пишет `commit: null`, `dirty: null`.

Dirty worktree разрешён для локальной разработки, но явно фиксируется. Не
объявлять run воспроизводимым без commit/lock hash.

## 10. Reference manifests

Snapshot всегда содержит:

- `dataset_manifest.json`;
- `replay_manifest.json`;
- `league_manifest.json`.

Если source path равен null, создавать строгий пустой manifest:

```json
{"format":"chessy-reference-v1","kind":"dataset","source":null,"source_sha256":null,"content":null}
```

Для указанного source:

- файл должен быть обычным UTF-8 JSON;
- symlink отклоняется;
- вычисляется SHA-256 исходных bytes;
- parsed JSON сохраняется в `content`;
- kind должен соответствовать ожидаемому;
- snapshot хранит копию manifest, но не копирует dataset/replay segments.

Resume проверяет, что snapshot reference manifests не изменены. Существование
всех внешних replay segments будет проверяться на self-play этапе.

## 11. JSONL events и metrics

### 11.1. events.jsonl

Каждая строка:

```json
{"format":"chessy-events-v1","sequence":1,"timestamp":"...Z","type":"run_created","payload":{}}
```

Типы минимум:

- `run_created`;
- `snapshot_started`;
- `snapshot_completed`;
- `snapshot_failed`;
- `resume_started`;
- `resume_completed`;
- `resume_fallback`;
- `fork_created`;
- `operational_override`;
- `stop_requested`;
- `run_stopped`;
- `run_completed`;
- `log_recovered`.

Sequence строго возрастает и продолжается после resume.

### 11.2. metrics.jsonl

Каждая строка:

```json
{"format":"chessy-metrics-v1","step":4,"epoch":1,"timestamp":"...Z","metrics":{"loss":1.23,"lr":0.0002}}
```

- Только конечные int/float values.
- Step не убывает.
- Resume append-ит, не перезаписывает историю.
- Дублирование уже завершённого step запрещено.
- Logs flush + fsync непосредственно перед публикацией snapshot.

Writer кодирует строку полностью и выполняет один низкоуровневый append под
lock. Если после crash обнаружен оборванный последний JSON fragment, сохранить
его как `*.recovered-fragment`, обрезать только этот fragment до последнего
валидного newline и записать `log_recovered`. Malformed строка в середине файла
считается corruption и не чинится молча.

## 12. Stateful sampler

Создать минимальный `StatefulBatchSampler`, пригодный для будущего dataset:

- принимает dataset size, batch size, shuffle, drop_last и seed;
- использует отдельный `torch.Generator` на CPU;
- хранит epoch, permutation, cursor и generator state;
- `state_dict()` содержит только primitives/tensors;
- `load_state_dict()` строго проверяет version и совместимость размеров;
- после restore следующий batch индексов точно совпадает;
- смена batch size/dataset size требует fork, а не resume.

Synthetic smoke dataset строит данные детерминированно из sample index, поэтому
sampler/RNG tests не зависят от файлов в `data/`.

## 13. Полный snapshot `chessy-snapshot-v1`

### 13.1. Структура

```text
step-000000000004/
  model.safetensors
  training_state.pt
  run_state.json
  config.resolved.json
  dataset_manifest.json
  replay_manifest.json
  league_manifest.json
  checksums.sha256
```

Snapshot directory name использует global step, padded до 12 цифр. На одном
step существует только один snapshot; best/stage/periodic являются tags в
index, а не дублирующими копиями файлов.

### 13.2. model.safetensors

- Полный model state dict на CPU float32;
- sorted keys;
- detached contiguous tensors;
- strict load;
- исходная модель не меняет device или train/eval flag.

Можно вынести общие Safetensors helpers из `chessy.model.export`, не меняя
формат `chessy-model-v1`.

### 13.3. training_state.pt

Содержит:

```text
format = chessy-training-state-v1
optimizer_state
scheduler_state
sampler_state
rng_state
gradient_scaler_state = null (v1 float32)
```

Не содержит:

- model object;
- arbitrary class instances;
- lambdas/callables;
- filesystem paths;
- dataset/replay contents.

Сохранять через `torch.save`, загружать только через
`torch.load(..., map_location="cpu", weights_only=True)`.

### 13.4. run_state.json

Формат `chessy-run-state-v1`:

- run ID и config fingerprint;
- global step;
- epoch;
- samples seen;
- stage/curriculum state (для smoke: `smoke`);
- best metric и best step;
- total elapsed seconds;
- last completed batch;
- snapshot reason: periodic/best/stage/stop/manual/completed;
- stop reason либо null;
- created_at UTC;
- model parameter count;
- optimizer/scheduler identifiers.

Global step означает число полностью завершённых optimizer steps. Snapshot
никогда не фиксирует полуобработанный batch.

### 13.5. checksums.sha256

Хеширует все семь остальных файлов. Формат совпадает с model export:

```text
<64 lowercase hex><two spaces><relative filename>\n
```

Строки сортируются. Checksums file не хеширует себя.

## 14. RNG state

Snapshot фиксирует:

- `random.getstate()`;
- legacy `numpy.random` state, если код его затронет;
- состояние явного `numpy.random.Generator` run;
- `torch.get_rng_state()` для CPU;
- `torch.mps.get_rng_state()` при доступном/инициализированном MPS;
- `torch.cuda.get_rng_state_all()` при доступной CUDA.

Формат должен состоять из tensors/primitives и быть совместим с
`weights_only=True`. Tuple/NumPy arrays при необходимости преобразуются в
versioned dict + tensors/lists и восстанавливаются обратно.

Restore происходит до создания следующего batch или stochastic operation.

При resume на том же backend соответствующий device RNG восстанавливается.
При разрешённой эксплуатационной смене backend:

- CPU/Python/NumPy state восстанавливается;
- state исходного accelerator сохраняется в snapshot, но не применяется к
  другому backend;
- пишется `operational_override`;
- побитовая идентичность не обещается.

На текущем PyTorch проверены API:

```text
torch.mps.get_rng_state / set_rng_state
torch.cuda.get_rng_state_all / set_rng_state_all
torch.load(weights_only=True)
```

## 15. Атомарная запись snapshot

Алгоритм:

1. Проверить destination и отсутствие snapshot этого step.
2. Создать sibling temp directory `.step-...tmp-<random>`.
3. Записать все payload files.
4. Flush/fsync каждого файла.
5. Записать checksums последним и fsync.
6. Полностью вызвать snapshot verifier на temp directory.
7. Fsync temp directory.
8. Atomic rename temp → final snapshot directory.
9. Fsync parent snapshots directory.
10. Атомарно обновить `snapshots/index.json`.
11. Только после этого записать `snapshot_completed`.

При исключении:

- final directory не появляется;
- temp directory удаляется;
- старый index не меняется;
- записывается `snapshot_failed` с безопасным типом ошибки, без tensor dump.

Запись existing step запрещена даже с флагом force.

## 16. Snapshot index и tags

`snapshots/index.json` имеет формат `chessy-snapshot-index-v1`:

```json
{
  "latest":"step-000000000012",
  "best":"step-000000000008",
  "stages":{"smoke":"step-000000000012"},
  "snapshots":[
    {"name":"step-000000000008","step":8,"tags":["periodic","best"],"sha256":"..."}
  ]
}
```

Index checksum entry для snapshot — SHA-256 canonical map его
`checksums.sha256`, а не recursive directory hash.

Index обновляется через temp file + `os.replace`. Loader не доверяет index без
проверки snapshot.

## 17. Retention policy

По умолчанию хранить:

- два последних snapshot с tag `periodic`;
- snapshot с tag `best`;
- каждый snapshot, на который ссылается `stages`;
- `latest`;
- snapshot с tag `manual` или `stop`.

Один snapshot с несколькими tags хранится одной директорией.

Retention запускается только после успешной публикации нового snapshot и
index. Перед удалением каждого кандидата:

- resolved path обязан быть прямым child snapshots directory;
- имя соответствует `step-[0-9]{12}`;
- path не symlink;
- snapshot присутствует в старом/new index;
- он не pinned;
- минимум два валидных snapshot останутся.

Corrupt/unindexed/temporary directories автоматически не удалять; `inspect`
показывает их для ручного решения.

## 18. Snapshot verifier и corruption fallback

Verifier проверяет до загрузки:

1. Directory обычная, не symlink.
2. Ровно восемь обязательных файлов.
3. Безопасный checksum file покрывает ровно семь payload files.
4. Все hashes совпадают.
5. Все JSON форматы/версии/типы валидны.
6. Resolved config fingerprint совпадает с run state.
7. Run ID совпадает.
8. Model config совместим и Safetensors strict-loadable.
9. Parameter count совпадает.
10. `training_state.pt` загружается с `weights_only=True`.
11. Optimizer/scheduler/sampler state имеют ожидаемые version/keys.
12. Reference manifest kinds корректны.

Resume сначала проверяет index.latest. Если он повреждён:

- сканирует snapshot directories по step убыванию;
- выбирает первый полностью валидный и совместимый;
- не изменяет повреждённую директорию;
- пишет `resume_fallback` с именами rejected/selected;
- если валидного snapshot нет, завершает работу без мутации run.

CLI verify возвращает ненулевой exit code и понятное сообщение для corruption.

## 19. Resume

### 19.1. Семантика

`resume` продолжает тот же run ID и использует сохранённый
`config.resolved.json`. Новый YAML config при resume не принимается.

Разрешённые эксплуатационные overrides:

- `--device auto|cpu|mps|cuda`;
- `--stop-after-steps N` как ограничение текущего процесса, не изменение
  training total steps;
- verbosity.

Каждый override записывается событием. Нельзя override:

- model architecture;
- optimizer/scheduler;
- batch size;
- sampler/dataset/replay;
- seed;
- total steps;
- config fingerprint.

### 19.2. Порядок восстановления

1. Проверить run manifest, config и выбранный snapshot.
2. Создать model/optimizer/scheduler/sampler из сохранённого config.
3. Strict-load model state.
4. Load optimizer и перенести его tensor states на target device.
5. Load scheduler.
6. Load sampler.
7. Восстановить global counters.
8. Восстановить RNG последним, непосредственно перед следующим batch.
9. Открыть logs в append mode после проверки/recovery.
10. Записать `resume_completed`.

Никаких random/model initializations после RNG restore до следующего training
step.

## 20. Fork

Fork всегда создаёт новый run с новым ID и parent metadata:

```text
parent.run_id
parent.snapshot
parent.snapshot_checksum
parent.mode
```

Поддержать два явных режима:

### `full-state`

- Копирует model, optimizer, scheduler, sampler, counters и RNG.
- Разрешён только при полностью совместимой критической конфигурации.
- Можно менять только device/operational settings и run name/artifacts root.
- Новый run начинает собственные logs/snapshot index.

### `weights-only`

- Strict-load только model weights.
- Optimizer/scheduler/sampler/RNG создаются заново из нового config/seed.
- Global step и epoch начинают с 0.
- Допускает новый optimizer/scheduler/batch size при совместимой архитектуре.
- Изменение model architecture требует отдельной migration и здесь запрещено.

Исходный run и snapshot должны остаться byte-for-byte неизменными после fork.

## 21. Graceful stop

Создать `StopController`:

- временно устанавливает handlers SIGINT и SIGTERM в main thread;
- handler только ставит thread-safe flag и фиксирует reason;
- не пишет snapshot и не выполняет тяжёлую работу внутри signal handler;
- trainer проверяет flag после полностью завершённого optimizer step;
- flush-ит metrics/events;
- сохраняет snapshot с tag `stop`;
- пишет `run_stopped`;
- закрывает ресурсы;
- восстанавливает прежние signal handlers.

Первый сигнал просит graceful stop. Второй сигнал может поднять
`KeyboardInterrupt` для аварийного выхода, но уже опубликованные snapshots не
затрагиваются.

Для будущего self-play предусмотреть hooks:

```python
request_stop_producers()
drain_pending_results()
flush_replay()
```

В smoke trainer они no-op, но порядок graceful shutdown документирован и
тестируется.

## 22. Synthetic smoke trainer

Smoke trainer нужен только для end-to-end доказательства инфраструктуры.

Требования:

- маленький `ChessyModel` из smoke config;
- synthetic float32 boards/targets детерминированно строятся по sample indices;
- policy CE + value CE;
- AdamW;
- warmup + cosine scheduler;
- gradient clipping;
- один metrics record на optimizer step;
- periodic snapshots;
- optional best tag по loss;
- completed snapshot на последнем total step;
- работает на CPU и MPS;
- не читает `data/` и не создаёт playable claims.

Synthetic data может быть шахматно нереалистичной. Это plumbing test, не
демонстрация силы или корректного RL objective.

## 23. CLI

Расширить существующий `chessy` CLI:

```text
chessy train smoke --config configs/smoke.yaml
                   [--stop-after-steps N]

chessy train smoke --resume runs/<run-id>
                   [--device cpu|mps|cuda|auto]
                   [--stop-after-steps N]

chessy run inspect runs/<run-id>
chessy snapshot verify runs/<run-id>/snapshots/<snapshot>

chessy run fork --snapshot <path>
                 --config <yaml>
                 --mode full-state|weights-only
```

Команды `--help` не создают runs и не инициализируют MPS.

`inspect` read-only выводит:

- run ID/status/config fingerprint;
- git/lock provenance;
- latest/best/stage refs;
- список valid/corrupt/unindexed snapshots;
- last metric/step;
- disk usage;
- parent/fork info.

Не печатать полный optimizer state или RNG bytes.

## 24. Обязательные unit tests

### Config/canonical

- valid YAML → strict resolved config;
- defaults присутствуют в resolved JSON;
- unknown field на любом уровне отклоняется;
- неправильные strict types отклоняются (`"4"` не int);
- unsafe YAML tag отклоняется;
- canonical serialization детерминирована;
- NaN/Infinity отклоняются;
- fingerprint стабилен и меняется при изменении config;
- slug/run ID безопасны.

### Sampler/RNG

- sampler restore даёт точно тот же следующий batch;
- несовместимые dataset/batch параметры отклоняются;
- Python random совпадает после restore;
- NumPy legacy и explicit Generator совпадают;
- CPU torch RNG совпадает;
- MPS RNG round-trip при доступном MPS;
- CUDA test conditional skip;
- serialized RNG загружается с `weights_only=True`.

### Logs

- sequence/step monotonic;
- append после resume;
- concurrent writers не смешивают строки;
- invalid metric values отклоняются;
- trailing fragment восстанавливается с отдельным artifact/event;
- malformed middle line вызывает corruption.

## 25. Snapshot tests

1. Snapshot содержит ровно восемь обязательных файлов.
2. Все checksums совпадают.
3. Model state до/после strict-load идентичен.
4. Optimizer tensors/state идентичны.
5. Scheduler, sampler, counters и RNG идентичны.
6. Snapshot writer не меняет model device/train flag/parameters.
7. Existing destination не перезаписывается.
8. Искусственная ошибка каждого этапа записи не оставляет final/temp garbage.
9. Corrupted byte обнаруживается до `torch.load`.
10. Unsafe checksum path отклоняется.
11. Missing/extra file отклоняется.
12. Wrong format/config/run ID/parameter count отклоняется.
13. `training_state.pt` реально загружается с `weights_only=True`.
14. Index update атомарен и указывает только на valid snapshot.
15. Corrupt latest приводит к fallback на предыдущий valid snapshot.
16. Если все snapshots corrupt, run не мутируется.
17. Retention оставляет два periodic, best, stage, stop и latest.
18. Retention не следует symlink и не удаляет неизвестные директории.

## 26. Resume equivalence test

Обязательный end-to-end CPU test:

### Непрерывный путь

```text
seed 42 → steps 1..12 → final state A
```

### Resume путь

```text
seed 42 → steps 1..5 → stop snapshot
new process-like reconstruction
resume → steps 6..12 → final state B
```

Сравнить A и B:

- каждый model tensor: `torch.equal`;
- полный optimizer state recursively;
- scheduler state;
- sampler state;
- global step/epoch/samples seen;
- metrics по steps 1..12;
- следующий sampler batch;
- следующие значения Python/NumPy/Torch RNG;
- итоговый loss с нулевой либо строго обоснованной tolerance.

Test должен создавать новые Python objects для resume, а не продолжать теми же
instances.

Отдельный MPS smoke проверяет сохранение/загрузку и продолжение step, но не
требует побитового равенства с непрерывным MPS run.

## 27. Fork tests

- full-state fork с совместимым config переносит весь state;
- несовместимый full-state config отклоняется до создания run;
- weights-only переносит только model tensors;
- weights-only сбрасывает optimizer/scheduler/sampler/counters/RNG по новому
  config;
- parent metadata точна;
- исходный run tree byte-for-byte не меняется;
- fork не наследует старые metrics/events как собственные;
- architecture mismatch отклоняется.

## 28. Graceful stop tests

- request flag не прерывает середину optimizer step;
- snapshot имеет последний полностью завершённый global step;
- metrics flush выполнен до snapshot;
- stop reason сохраняется;
- periodic и stop snapshot на одном step не дублируются, а объединяют tags;
- handlers восстанавливаются;
- subprocess SIGINT завершается за ограниченное время и оставляет valid
  resumable snapshot;
- повторный resume после SIGINT проходит.

Не использовать долгие sleep; синхронизация tests через events/barriers/fake
trainer hooks.

## 29. Команды финальной проверки

```bash
uv lock --check
uv sync --locked
uv run pytest
uv run chessy train smoke --config configs/smoke.yaml --stop-after-steps 5
uv run chessy run inspect runs/<smoke-run-id>
uv run chessy snapshot verify runs/<smoke-run-id>/snapshots/<latest>
uv run chessy train smoke --resume runs/<smoke-run-id>
uv run chessy run fork --snapshot <latest> --config configs/smoke.yaml --mode full-state
git diff --check
git diff --exit-code -- data
git status --short
```

Реальные smoke runs создаются в ignored `runs/`. Test suite использует только
`tmp_path`.

После ручного smoke не удалять артефакты destructive-командой без отдельной
просьбы; сообщить пользователю путь и размер, чтобы он мог изучить snapshot.

## 30. Критерии приёмки

Задача выполнена только если одновременно выполнено всё ниже:

- [ ] Pydantic и PyYAML объявлены прямыми runtime dependencies.
- [ ] `chessy-config-v1` строго валидирует YAML и запрещает unknown fields.
- [ ] Resolved config canonical и имеет стабильный fingerprint.
- [ ] Run ID безопасен и включает timestamp/name/fingerprint.
- [ ] `chessy-run-v1` фиксирует Git, `uv.lock`, Python, PyTorch, device и
      reference manifests.
- [ ] Metrics/events streams append-only, versioned и восстанавливают только
      оборванный trailing fragment.
- [ ] Stateful sampler точно продолжает следующий batch.
- [ ] Snapshot содержит model, optimizer, scheduler, sampler, counters, config,
      references и все RNG states.
- [ ] Model хранится в Safetensors отдельно от training state.
- [ ] `training_state.pt` загружается с `weights_only=True`.
- [ ] Snapshot публикуется только после полной verify и atomic rename.
- [ ] Corruption обнаруживается до state loading.
- [ ] Повреждённый latest даёт явный fallback на предыдущий valid snapshot.
- [ ] Retention хранит минимум два periodic и все pinned snapshots.
- [ ] Graceful SIGINT/SIGTERM сохраняет последний завершённый step.
- [ ] Resume не принимает новый critical config.
- [ ] Fork всегда создаёт новый run и не мутирует parent.
- [ ] Full-state и weights-only fork имеют различную проверенную семантику.
- [ ] CPU uninterrupted/resume paths дают идентичное полное состояние.
- [ ] MPS smoke продолжает обучение с сохранённого step.
- [ ] CLI smoke/inspect/verify/fork работают и имеют безопасный `--help`.
- [ ] Snapshot/run outputs остаются ignored Git artifacts.
- [ ] Все tests проходят одной командой `uv run pytest`.
- [ ] Existing datasets, playable export, MCTS и UI contracts не сломаны.
- [ ] Self-play, replay и реальное обучение не добавлены преждевременно.

## 31. Что не делать даже при наличии времени

- Не запускать self-play.
- Не обучаться на personal train split.
- Не добавлять reward/material heuristics.
- Не копировать replay buffer внутрь snapshot.
- Не сохранять model object через pickle.
- Не разрешать resume с тихо изменённым learning rate/batch size.
- Не перезаписывать corrupt snapshot «исправленной» версией.
- Не удалять corrupt/unindexed artifacts автоматически.
- Не обещать bitwise identity между CPU и MPS.
- Не интегрировать внешние experiment trackers.

## 32. Финальный отчёт агента

Агент должен сообщить:

1. Какие модули, configs и зависимости добавлены.
2. Точные версии Pydantic/PyYAML/PyTorch.
3. Структуру созданного smoke run и snapshot.
4. Размер model/training/full snapshot и всего smoke run.
5. Число tests, skips, warnings и время.
6. Результат CPU uninterrupted-vs-resume сравнения.
7. Результат MPS smoke и восстановление RNG.
8. Результаты corruption/fallback/retention tests.
9. Результат full-state и weights-only fork tests.
10. Результат subprocess graceful SIGINT test.
11. Команды и пути ручного smoke run.
12. Были ли изменены existing datasets или контракты.
13. Все отклонения от плана и причины.

Не считать задачу завершённой только потому, что model weights загрузились.
`resume` считается доказанным лишь после восстановления optimizer, scheduler,
sampler, counters и RNG с эквивалентным следующим training step.
