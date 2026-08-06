# Chessy — шаг 3: policy/value-модель и playable export

Статус: готово к реализации
Дата постановки: 2026-08-06
Рабочая директория: `/Users/zhuravlevvikt/Documents/codex_projects/chessy`

## 1. Цель задачи

Реализовать первую нейросеть Chessy, способную принимать позиции формата
`board119-v1` и выдавать:

- policy logits для 4672 действий `az73-v1`;
- value logits для исходов loss/draw/win;
- корректно замаскированное распределение только по легальным действиям.

Модель должна:

- работать в PyTorch на CPU;
- работать на MPS при его наличии;
- оставаться переносимой на CUDA без отдельной реализации;
- сохраняться в безопасный игровой формат `chessy-model-v1`;
- воспроизводимо загружаться и давать те же результаты inference;
- иметь небольшой benchmark для batch inference.

После выполнения шага случайно инициализированная модель должна оценивать пакет
позиций, сохраняться в export-директорию, загружаться обратно и побитово или с
оговорённой численной точностью воспроизводить CPU-выходы.

Задача не включает обучение, optimizer, training snapshot, MCTS, self-play,
игровой API или UI.

## 2. Архитектурные источники

Перед началом полностью прочитать:

- `AGENTS.md`;
- `docs/PROJECT_PLAN.md`, особенно разделы 8, 16, 17, 19, 20 и 22;
- `docs/decisions/0002-compute-backend.md`;
- `docs/decisions/0003-position-encoding.md`;
- `docs/decisions/0004-action-encoding.md`;
- `docs/decisions/0005-artifact-formats.md`;
- реализацию `src/chessy/encoding/` и её тесты;
- `docs/tasks/STEP_02_CHESS_ENVIRONMENT_AND_ENCODING.md`.

Утверждённые контракты нельзя переименовывать:

- board encoding: `board119-v1`;
- action encoding: `az73-v1`;
- архитектура модели: `residual-cnn-v1`;
- игровой export: `chessy-model-v1`.

## 3. Обязательные ограничения

1. Использовать PyTorch. Не добавлять MLX, TensorFlow, JAX или ONNX.
2. CPU является обязательным backend для тестов и fallback.
3. MPS поддерживается штатными PyTorch-операциями без отдельной версии сети.
4. Не добавлять Transformer, attention, squeeze-excitation и иные усложнения.
5. Не менять `board119-v1` и `az73-v1`.
6. Не обучать модель и не добавлять optimizer/scheduler.
7. Не реализовывать training snapshot `chessy-snapshot-v1`; здесь создаётся
   только playable export.
8. Не реализовывать MCTS, API или UI.
9. Не читать и не изменять `data/`.
10. Не добавлять большие веса в Git. Временные экспорты тестов должны жить в
    pytest `tmp_path`.
11. Не создавать Git-коммит без отдельного запроса пользователя.

## 4. Зависимости

Добавить через `uv` runtime-зависимости:

- `torch`;
- `safetensors`.

Предпочтительные команды:

```bash
uv add "torch>=2.7,<3"
uv add "safetensors>=0.5,<1"
```

Если доступная стабильная версия PyTorch для Python 3.12 требует другой нижней
границы, допустимо выбрать совместимую версию и объяснить отклонение. Не
редактировать `uv.lock` вручную.

Не добавлять NumPy повторно: зависимость уже существует. Не добавлять Pydantic,
YAML или benchmark-framework — для этого шага достаточно стандартной библиотеки
и `time.perf_counter`.

После изменения зависимостей проверить:

```bash
uv lock --check
uv sync --locked
```

## 5. Целевая структура

```text
src/chessy/
  model/
    __init__.py
    config.py
    device.py
    network.py
    inference.py
    export.py
scripts/
  benchmark_model.py
tests/
  model/
    test_config.py
    test_device.py
    test_network.py
    test_inference.py
    test_export.py
```

Допустимо объединить `config.py` и `device.py` с соседними модулями, если API
останется небольшим и тестируемым. Не создавать директории training или MCTS.

## 6. Конфигурация `residual-cnn-v1`

### 6.1. ModelConfig

Создать неизменяемую dataclass `ModelConfig` со значениями по умолчанию:

```python
architecture: str = "residual-cnn-v1"
input_planes: int = 119
action_planes: int = 73
board_size: int = 8
channels: int = 96
residual_blocks: int = 8
group_norm_groups: int = 8
value_channels: int = 32
value_hidden: int = 128
value_classes: int = 3
```

Конфигурация должна:

- валидироваться при создании;
- запрещать неизвестные поля при загрузке из dict/JSON;
- иметь `to_dict()` с JSON-совместимыми значениями;
- иметь `from_dict()` со строгой проверкой;
- проверять совместимость с константами `BOARD_PLANES`, `ACTION_PLANES` и
  `ACTION_SIZE`;
- требовать положительные размеры;
- требовать делимость `channels` и `value_channels` на
  `group_norm_groups`;
- требовать `board_size == 8` и `value_classes == 3` в версии v1.

Полный YAML/Pydantic config loader появится на этапе инфраструктуры run. Не
строить его в этой задаче.

### 6.2. Семантика value

Порядок классов фиксирован:

```text
0 = loss
1 = draw
2 = win
```

Исход всегда задаётся относительно стороны, которой принадлежит ход во входной
позиции. Модель возвращает logits, а не softmax probabilities.

Ожидаемое скалярное значение для MCTS:

```text
P(win) - P(loss)
```

Оно должно лежать в `[-1, +1]`.

## 7. Архитектура сети

### 7.1. Общий контракт

Главный класс:

```python
class ChessyModel(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None: ...
    def forward(self, boards: torch.Tensor) -> PolicyValueOutput: ...
```

`PolicyValueOutput` — `NamedTuple` или frozen dataclass с полями:

```python
policy_logits: torch.Tensor  # [B, 4672]
value_logits: torch.Tensor   # [B, 3]
```

Forward принимает только `[B, 119, 8, 8]`. При неверной размерности, числе
плоскостей или размере доски выдавать понятный `ValueError`.

### 7.2. Stem

```text
Conv2d(119, 96, kernel_size=3, padding=1, bias=False)
GroupNorm(8, 96)
ReLU(inplace=False)
```

Размер доски остаётся `8 × 8`.

### 7.3. Residual trunk

Восемь одинаковых residual-блоков:

```text
x ── Conv2d(96, 96, 3, padding=1, bias=False)
     GroupNorm(8, 96)
     ReLU
     Conv2d(96, 96, 3, padding=1, bias=False)
     GroupNorm(8, 96)
     + skip(x)
     ReLU ── output
```

- Stride всегда равен 1.
- Projection skip не нужен: число каналов не меняется.
- BatchNorm не использовать: локальное обучение и MCTS будут иметь разные и
  часто небольшие batch sizes.
- Dropout не использовать в v1.
- `inplace=False` упрощает отладку и не создаёт скрытых конфликтов autograd.

### 7.4. Policy head

```text
Conv2d(96, 73, kernel_size=1, bias=True)
reshape [B, 73, 8, 8] -> [B, 4672]
```

Порядок flatten обязан совпадать с `action = plane * 64 + from_square`.
NumPy board использует `[rank, file]`; contiguous flatten последних двух
измерений даёт `rank * 8 + file`, что совпадает с нумерацией `python-chess`.

Policy head возвращает сырые logits. Внутри `forward` не должно быть softmax и
не должна применяться legal mask.

### 7.5. Value head

```text
Conv2d(96, 32, kernel_size=1, bias=False)
GroupNorm(8, 32)
ReLU
Flatten
Linear(32 * 8 * 8, 128)
ReLU
Linear(128, 3)
```

Value head возвращает W/D/L logits без softmax.

### 7.6. Размер модели

Для конфигурации по умолчанию ожидается примерно 1.7 млн обучаемых параметров.
Тест должен требовать диапазон:

```text
1_500_000 <= trainable_parameters <= 2_000_000
```

Не подгонять точное число ценой изменения утверждённой архитектуры.

### 7.7. Инициализация

- Conv2d: Kaiming normal для ReLU, bias нулями.
- Linear: Kaiming uniform или стандартная явная PyTorch-инициализация,
  одинаковая для всех устройств.
- GroupNorm weight единицами, bias нулями.
- Инициализация должна зависеть от текущего PyTorch RNG и воспроизводиться при
  одинаковом `torch.manual_seed` на CPU.

Не делать residual zero-init в первой версии: это можно исследовать отдельно
после появления обучения.

## 8. Device policy

В `device.py` реализовать:

```python
DeviceName = Literal["auto", "cpu", "mps", "cuda"]

def resolve_device(requested: DeviceName = "auto") -> torch.device: ...
```

Семантика:

- `cpu` всегда возвращает CPU;
- `mps` требует `torch.backends.mps.is_available()`, иначе `RuntimeError`;
- `cuda` требует `torch.cuda.is_available()`, иначе `RuntimeError`;
- `auto`: MPS, затем CUDA, затем CPU;
- неизвестное значение отклоняется через `ValueError`.

MPS имеет приоритет в `auto`, потому что основной компьютер проекта — Mac.

В v1 используется `float32`. Не включать autocast, float16 или bfloat16 без
отдельного benchmark и решения.

## 9. Inference helpers и legal mask

В `inference.py` реализовать отдельные функции, не смешивая их с сетью:

```python
def mask_policy_logits(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor: ...

def legal_policy_probabilities(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor: ...

def value_probabilities(value_logits: torch.Tensor) -> torch.Tensor: ...

def expected_value(value_logits: torch.Tensor) -> torch.Tensor: ...
```

Требования:

- policy logits и mask имеют одинаковую форму `[B, 4672]`;
- mask имеет dtype `torch.bool` и находится на том же device;
- нелегальные logits заменяются ровно на `-inf`;
- легальные logits не меняются;
- `legal_policy_probabilities` применяет softmax после маски;
- сумма вероятностей каждой строки равна 1;
- вероятность нелегальных действий равна 0;
- строка без единого легального действия отклоняется через `ValueError`, чтобы
  не получить NaN. Терминальные позиции MCTS обязан обрабатывать до inference;
- `value_probabilities` возвращает softmax по трём классам;
- `expected_value` возвращает `[B]` как `P(win) - P(loss)`.

Функции не должны менять входные tensors in-place.

Добавить небольшой helper для преобразования NumPy-результатов существующих
кодировщиков в batch tensors допустимо, но не создавать DataLoader или dataset.

## 10. Playable export `chessy-model-v1`

### 10.1. Структура директории

```text
<export-dir>/
  model.safetensors
  manifest.json
  checksums.sha256
```

Экспорт содержит только inference-веса и метаданные. Pickle и
`torch.save(model)` запрещены.

### 10.2. Manifest

`manifest.json` записывается как UTF-8 canonical JSON:

- `sort_keys=True`;
- компактные separators;
- один завершающий newline;
- числа и строки без платформозависимого форматирования.

Минимальная схема:

```json
{
  "format": "chessy-model-v1",
  "created_at": "<UTC ISO-8601>",
  "architecture": "residual-cnn-v1",
  "model_config": {},
  "encodings": {
    "board": "board119-v1",
    "action": "az73-v1"
  },
  "value_classes": ["loss", "draw", "win"],
  "framework": {
    "name": "pytorch",
    "version": "<torch.__version__>"
  },
  "weights": {
    "file": "model.safetensors",
    "dtype": "float32",
    "parameter_count": 0,
    "sha256": "<hex>"
  }
}
```

`model_config` содержит полный `ModelConfig.to_dict()`. Допустимо добавить
`project_version` и произвольную заметку только как документированные
опциональные поля. Не добавлять training metrics, optimizer или dataset paths.

### 10.3. Checksums

`checksums.sha256` содержит SHA-256 для:

- `manifest.json`;
- `model.safetensors`.

Формат каждой строки:

```text
<64 lowercase hex><two spaces><relative filename>
```

Строки сортируются по имени файла и завершаются newline. Сам файл checksums не
хеширует себя.

### 10.4. Сохранение

Публичный API:

```python
def export_model(
    model: ChessyModel,
    destination: Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path: ...
```

Требования:

- destination не должна уже существовать, чтобы исключить тихую смесь версий;
- сначала создаётся временная sibling-директория;
- веса переносятся на CPU, приводятся к contiguous float32 и сохраняются через
  `safetensors.torch.save_file`;
- исходная модель не переносится на другой device и не мутируется;
- после записи проверяются файлы, checksums и возможность прочитать safetensors;
- готовая директория публикуется атомарным rename;
- при ошибке временная директория очищается;
- ключи state dict сохраняются детерминированно;
- export не содержит optimizer или произвольного pickle.

### 10.5. Загрузка

Публичный API:

```python
def load_model_export(
    source: Path,
    *,
    device: DeviceName | torch.device = "auto",
) -> ChessyModel: ...
```

Порядок проверки до использования модели:

1. Ровно три обязательных файла существуют и являются обычными файлами.
2. `checksums.sha256` корректно парсится без абсолютных путей и `..`.
3. SHA-256 manifest и весов совпадают.
4. `format == chessy-model-v1`.
5. Architecture, board/action encoding и порядок value classes совместимы.
6. `ModelConfig` строго валиден.
7. Parameter count и dtype соответствуют manifest.
8. Safetensors загружается с `strict=True` без отсутствующих или лишних ключей.

Возвращаемая модель:

- находится на запрошенном device;
- переведена в `eval()`;
- содержит те же параметры, что экспортированная модель.

Не разрешать `strict=False` и не игнорировать несовместимые версии.

## 11. Benchmark batch inference

Создать `scripts/benchmark_model.py` с CLI:

```text
--device auto|cpu|mps|cuda
--batch-sizes 1 8 32
--warmup 5
--iterations 20
--seed 0
```

Скрипт:

- создаёт модель с конфигурацией по умолчанию;
- использует случайный float32 input `[B,119,8,8]`;
- включает `model.eval()` и `torch.inference_mode()`;
- выполняет warmup отдельно для каждого batch size;
- синхронизирует MPS/CUDA непосредственно перед началом и после измеряемого
  блока;
- сообщает device, batch size, среднюю latency одного batch и positions/sec;
- не записывает файлы;
- не запускается как часть pytest;
- возвращает ненулевой exit code для недоступного явно выбранного device.

Не устанавливать минимальный throughput как критерий приёмки: результат зависит
от конкретного Mac и фоновой нагрузки. Нужна измерительная точка, а не обещание
силы или скорости.

## 12. Обязательные тесты

### 12.1. Конфигурация

- Значения `ModelConfig()` совпадают с контрактом.
- `to_dict()`/`from_dict()` дают round-trip.
- Неизвестное поле и несовместимые размеры отклоняются.
- Конфигурация совместима с `board119-v1` и `az73-v1`.

### 12.2. Network

- Forward для batch sizes `1`, `2` и `8` на CPU.
- Формы `[B,4672]` и `[B,3]`.
- Выходы и параметры float32.
- Policy/value outputs содержат конечные logits до маскирования.
- Forward не применяет softmax: суммы logits не обязаны равняться 1.
- Неверные shapes отклоняются.
- Parameter count находится в диапазоне 1.5–2.0 млн.
- Backward smoke-test подтверждает ненулевые конечные gradients хотя бы в stem,
  одном residual block и обеих головах.
- Одинаковый seed создаёт одинаковые CPU state dict и outputs.
- `eval()` + `inference_mode()` работают без изменения shapes.

### 12.3. Inference helpers

- Legal logits сохраняются, illegal становятся `-inf`.
- После softmax illegal probabilities равны нулю, legal суммируются в 1.
- Начальная позиция даёт ровно 20 ненулевых policy probabilities.
- Mask не мутируется.
- Несовместимые shape, dtype и device отклоняются.
- Пустая legal mask отклоняется до softmax.
- WDL softmax суммируется в 1.
- Для искусственных logits проверяются знак и границы expected value.

### 12.4. Export

- В export ровно три обязательных файла.
- Manifest содержит точные версии форматов и полный config.
- Checksums совпадают с реальными файлами.
- State dict до и после загрузки совпадает tensor-by-tensor.
- CPU outputs исходной и загруженной модели совпадают через
  `torch.testing.assert_close` с `rtol=0`, `atol=0`.
- Загруженная модель находится в eval mode.
- Повреждение одного байта весов обнаруживается checksum-проверкой до загрузки.
- Изменённый manifest с пересчитанным checksum, но неверной версией encoding,
  отклоняется как несовместимый.
- Лишний/пропущенный state-dict key отклоняется.
- Existing destination не перезаписывается.
- После искусственной ошибки временная директория не остаётся рядом.
- Экспорт модели не меняет её device, training/eval flag и значения параметров.

### 12.5. Devices

- CPU всегда разрешается.
- `auto` всегда возвращает доступный device.
- Недоступный явно запрошенный backend выдаёт понятную ошибку.
- MPS forward запускается отдельным тестом с `skip`, только если MPS недоступен.
- При доступном MPS проверить batch `1` и `8`, конечность outputs и правильные
  shapes.
- CUDA-тест аналогично допускает skip; наличие CUDA на Mac не требуется.

Тесты не должны скачивать готовые веса и не должны требовать сеть.

## 13. Требования к качеству

- Публичный API должен экспортироваться из `chessy.model` без wildcard imports.
- Внутренние tensors не хранить как глобальное изменяемое состояние.
- Не использовать `.data` для изменения параметров.
- В тестах явно задавать seed там, где сравниваются числа.
- Не считать MPS побитово идентичным CPU; проверять формы, конечность и разумный
  tolerance только там, где сравнение действительно нужно.
- Не подавлять PyTorch warnings без объяснения.
- Не выполнять device transfer внутри каждого residual block.
- Не применять legal mask внутри `ChessyModel.forward`: terminal handling и
  mask принадлежат inference/MCTS-слою.
- Не создавать зависимости от файловой системы в `network.py`.
- Не загружать pickle из model export.

## 14. Порядок реализации

1. Проверить `git status` и убедиться, что исходная точка — актуальный `main`.
2. Прочитать ADR и существующие encoding contracts.
3. Добавить PyTorch и Safetensors через `uv`.
4. Реализовать и протестировать `ModelConfig`.
5. Реализовать residual block, trunk и обе головы.
6. Добавить CPU shape, parameter-count, gradient и determinism tests.
7. Реализовать device resolution и условные MPS/CUDA tests.
8. Реализовать masking/WDL helpers и тесты с реальными legal masks.
9. Реализовать export, checksums, строгую загрузку и corruption tests.
10. Добавить benchmark CLI и проверить его на CPU с минимальными итерациями.
11. Запустить весь suite, lock-проверки и `git diff --check`.

## 15. Команды финальной проверки

```bash
uv lock --check
uv sync --locked
uv run python --version
uv run python -c "import torch; import safetensors; import chessy.model"
uv run pytest
uv run python scripts/benchmark_model.py --device cpu --batch-sizes 1 8 --warmup 1 --iterations 2
git diff --check
git diff --exit-code -- data/
git status --short
```

Дополнительная smoke-проверка должна вывести:

```text
architecture = residual-cnn-v1
board encoding = board119-v1
action encoding = az73-v1
policy shape = (2, 4672)
value shape = (2, 3)
parameter count = <1.5–2.0 million>
cpu export round-trip = ok
mps = available|unavailable
```

## 16. Критерии приёмки

Задача считается выполненной, только если одновременно выполнено всё ниже:

- [ ] PyTorch и Safetensors объявлены runtime-зависимостями и зафиксированы в
      `uv.lock`.
- [ ] `uv lock --check` и `uv sync --locked` проходят.
- [ ] `ModelConfig` строго валидируется и совместим с encoding constants.
- [ ] Реализована точная архитектура из восьми residual blocks и 96 каналов.
- [ ] Используются GroupNorm и ReLU, без BatchNorm и Dropout.
- [ ] Forward принимает `[B,119,8,8]` и возвращает `[B,4672]` и `[B,3]`.
- [ ] Policy и value возвращаются как logits без встроенного softmax.
- [ ] Порядок value classes — loss/draw/win относительно стороны хода.
- [ ] Число обучаемых параметров находится в диапазоне 1.5–2.0 млн.
- [ ] CPU forward, backward и deterministic initialization покрыты тестами.
- [ ] MPS forward проходит на текущем Mac или корректно skip-ается, если backend
      действительно недоступен.
- [ ] CUDA остаётся поддерживаемым через общий PyTorch-код и условный тест.
- [ ] Legal masking ставит `-inf` только нелегальным действиям.
- [ ] Legal policy не содержит вероятности на нелегальных ходах и суммируется в
      1.
- [ ] Expected value корректно вычисляется из WDL.
- [ ] Export содержит `model.safetensors`, `manifest.json` и
      `checksums.sha256`.
- [ ] Manifest фиксирует architecture, config, encoding versions, value order,
      dtype, parameter count и framework version.
- [ ] Checksums проверяются до загрузки весов.
- [ ] Повреждённый или несовместимый export отклоняется.
- [ ] CPU export/load round-trip воспроизводит state dict и outputs.
- [ ] Экспорт не мутирует исходную модель.
- [ ] Benchmark CLI работает на CPU и поддерживает MPS/CUDA.
- [ ] Все тесты проходят одной командой `uv run pytest`.
- [ ] `git diff --check` не находит ошибок.
- [ ] `data/` не изменена.
- [ ] MCTS, обучение, snapshots, API и UI не добавлены преждевременно.

## 17. Что не делать даже при наличии времени

- Не запускать обучение на персональном датасете.
- Не добавлять optimizer, scheduler или loss-функции training loop.
- Не реализовывать MCTS и batching service.
- Не создавать полный `chessy-snapshot-v1`.
- Не сохранять `.pt`, `.pth` или pickle как playable export.
- Не добавлять ONNX/ CoreML export.
- Не включать mixed precision.
- Не заявлять об игровой силе случайной модели.
- Не коммитить benchmark exports или веса.

## 18. Финальный отчёт агента

В завершении агент должен сообщить:

1. Какие файлы созданы и изменены.
2. Точные версии PyTorch и Safetensors.
3. Фактическое число параметров модели.
4. Формы policy/value outputs на CPU и MPS.
5. Доступен ли MPS на текущем Mac и прошёл ли тест.
6. Количество тестов и время выполнения.
7. Результат CPU export/load round-trip и corruption tests.
8. Результаты benchmark для batch sizes 1, 8 и 32 отдельно по доступным
   devices, без интерпретации как показателя силы.
9. Были ли изменены данные или добавлено что-либо вне области задачи.
10. Все отклонения от плана и причины.

Не утверждать, что задача завершена, если хотя бы один критерий приёмки не
выполнен. Не переходить к MCTS или игровому интерфейсу без нового задания
пользователя.
