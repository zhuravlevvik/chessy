# Шаг 9. Персональный RL: усиление без потери стиля

## 1. Контекст и обязательная предпосылка

Это последний этап утверждённого roadmap. Он начинается только после merge PR
шага 8 и обновления локального `main`:

```bash
git switch main
git pull --ff-only
git switch -c feature/personal-rl
```

Не продолжать работу поверх незамерженной ветки
`feature/human-feedback-pipeline`.

К началу шага в проекте уже должны существовать:

- `base_rl` export и обычный self-play/RL pipeline;
- immutable RL replay segments и manifests;
- curriculum, league и arena promotion gate;
- `personal_supervised` export;
- опциональный `personal_feedback` export;
- immutable historical personal dataset с frozen train/val/test splits;
- отдельный train-only human-feedback dataset;
- full snapshots, graceful stop и exact resume;
- игровой UI, способный загрузить любой совместимый model export.

Не нужно заново обучать правила шахмат с нуля, переписывать MCTS, создавать
новый UI или объединять исторические и feedback manifests. Задача шага —
скомпоновать уже существующие механизмы в отдельный, проверяемый personal-RL
run.

## 2. Цель

Реализовать полный цикл:

```text
personal incumbent
    -> self-play
    -> immutable RL replay
    -> RL policy/value update
       + historical style anchor
       + optional human-feedback anchor
    -> strength arena gate
    -> style-retention gate
    -> load/legal-game gate
    -> playable personal_rl export
```

`personal incumbent` выбирается явно в config:

1. `personal_feedback`, если пользователь уже запускал шаг 8;
2. иначе `personal_supervised`.

Автоматически искать «самый новый» export запрещено. Путь, роль и checksum
каждой исходной модели должны быть закреплены в resolved config и run metadata.

Главный критерий результата: кандидат становится сильнее текущего персонального
предка в статистически достаточном paired arena и при этом не выходит за
заданные пределы ухудшения сходства с ходами владельца.

## 3. Что не входит в шаг

- обучение шахматным правилам с нуля;
- online-learning прямо во время партии пользователя;
- автоматический запуск обучения после сохранения feedback;
- изменение MCTS ради стилистического выбора хода;
- добавление ходов бота в human targets;
- изменение frozen historical test;
- использование test для выбора checkpoint или настройки коэффициентов;
- Stockfish как teacher или обязательный внешний runtime;
- distributed/cloud training;
- открытие внешних портов;
- утверждение о силе по smoke-конфигурации.

## 4. Зафиксированная схема обучения

### 4.1. Три независимых потока

Trainer читает три логически независимых источника:

1. `rl` — позиции из self-play replay с мягким MCTS policy target и WDL;
2. `historical_style` — позиции владельца из historical train split с hard
   policy target;
3. `feedback_style` — опциональные подтверждённые ходы пользователя.

Historical `val` используется только для style gate. Historical `test` не
открывается trainer-ом ни при каких обстоятельствах. Feedback остаётся
train-only сигналом и отдельной диагностикой, а не generalization test.

Потоки не объединяются физически в один manifest и не копируют строки для
реализации веса.

### 4.2. Objective

Для одного optimizer step:

```text
L_rl = w_rl_policy * CE(MCTS policy, policy logits)
     + w_rl_value  * CE(self-play WDL, value logits)

L_historical = w_style_policy * CE(owner move, policy logits)
             + w_style_value  * CE(game WDL, value logits)

L_feedback = weighted supervised loss шага 8

L_total = L_rl
        + style_strength * L_historical
        + feedback_strength * L_feedback
```

Каждый компонент сначала нормализуется внутри собственного mini-batch. Размер
потока не должен неявно менять его коэффициент. Если feedback отключён,
`L_feedback` равен нулю и feedback dataset не открывается.

Стартовые значения для локального эксперимента:

```yaml
rl_policy_weight: 1.0
rl_value_weight: 1.0
style_strength: 0.20
style_policy_weight: 1.0
style_value_weight: 0.25
feedback_strength: 0.20
feedback_sample_weight: 4.0
```

Это начальная конфигурация, а не универсальный optimum. Нулевой
`style_strength` запрещён: такой run является обычным RL и не решает задачу
этого шага.

### 4.3. Батчи и ограниченные ресурсы

На каждом update обязателен RL mini-batch и historical-style mini-batch.
Feedback mini-batch добавляется только при включённом feedback.

Для Mac/MPS начать с:

```yaml
rl_batch_size: 32
historical_batch_size: 16
feedback_batch_size: 8
gradient_accumulation_steps: 1
```

Потоки разрешено прогонять последовательно до одного `backward/step`, чтобы не
держать все forward activations одновременно. Реализация должна складывать
корректно масштабированные gradients и вызывать optimizer/scheduler ровно один
раз на global step.

При нехватке unified memory уменьшать размеры батчей, но не менять objective
молча. Фактические размеры и коэффициенты логируются на каждом шаге.

### 4.4. Детерминированное семплирование

- RL stream использует существующую recent/older replay policy.
- Historical stream использует существующие `sample_kind_weights` и
  `max_positions_per_game`.
- Feedback stream использует отдельный sampler и
  `max_positions_per_game`; партии не должны циклически повторяться внутри
  одной style epoch сверх cap.
- Каждый sampler имеет независимый seed, детерминированно полученный из run
  seed и имени потока.
- Полное состояние всех sampler-ов входит в snapshot.
- Исчерпание короткого style-потока начинает следующую явно учитываемую style
  epoch; оно не меняет число optimizer updates.

### 4.5. Self-play и replay

Self-play начинается от точного personal incumbent, а после успешной promotion
— от нового personal-RL incumbent. Root noise остаётся включённым только в
self-play. Replay segments сохраняют checksum модели-генератора.

Replay manifest должен различать поколения и не смешивать данные другого run
без явного `seed_replay_manifest` в config. Для первого варианта seed replay
по умолчанию выключен: так проще доказать происхождение данных. Если поддержка
seed replay реализуется, его fingerprint и policy смешивания обязательны в
snapshot и отчёте.

Неудачный кандидат не становится self-play incumbent следующего поколения.
Trainer должен восстановить promoted incumbent перед следующей генерацией.

## 5. Strict config contract

Добавить отдельную секцию `personal_rl`. Не снимать существующий запрет на
совместное использование обычной `personalization` и RL секций и не
переопределять смысл старых configs.

Рекомендуемый контракт:

```yaml
personal_rl:
  enabled: true
  incumbent_export: artifacts/personal_feedback
  allowed_incumbent_roles: [personal_supervised, personal_feedback, personal_rl]
  base_rl_export: artifacts/base_rl
  personal_supervised_export: artifacts/personal_supervised
  historical_dataset_manifest: data/personal/encoded/manifests/personal-dataset-REPLACE.json
  feedback_dataset_manifest: null

  rl_policy_weight: 1.0
  rl_value_weight: 1.0
  style_strength: 0.20
  style_policy_weight: 1.0
  style_value_weight: 0.25
  feedback_strength: 0.20
  feedback_sample_weight: 4.0

  historical_batch_size: 16
  feedback_batch_size: 8
  sample_kind_weights: {good_move: 0.75, full_game: 1.0}
  historical_max_positions_per_game: 16
  feedback_max_positions_per_game: 16

  historical_ce_regression_tolerance: 0.02
  feedback_ce_regression_tolerance: 0.02
  minimum_style_top1_ratio: 0.95
```

Точные имена можно улучшить, но смысл должен остаться явным.

Validation requirements:

- все пути относительные, project-local и без symlink escape;
- все float конечны;
- все веса неотрицательны;
- `style_strength > 0`;
- `feedback_strength > 0` требует feedback manifest;
- feedback manifest требует `feedback_strength > 0`;
- batch sizes и caps положительны;
- regression tolerances конечны и неотрицательны;
- роли exports проверяются по manifest, а не по имени каталога;
- архитектура всех exports совпадает с config;
- `training.batch_size == rl.batch_size` либо это различие явно устранено в
  schema, без двух конфликтующих источников истины.

Добавить:

- `configs/personal-rl-smoke.yaml`;
- `configs/personal-rl-local-mps.yaml`.

Smoke config обязан быть маленьким и честно называться plumbing check.

## 6. Run provenance и immutable inputs

До создания run проверить и закрепить:

- incumbent export role/path/weights checksum;
- base RL export role/path/weights checksum;
- personal-supervised export role/path/weights checksum;
- historical train/val manifest fingerprint;
- frozen historical test fingerprint как provenance-only значение;
- optional feedback manifest fingerprint;
- optional seed replay manifest fingerprint;
- resolved config fingerprint.

Невалидный input не должен оставлять пустой run directory.

`Run.references` и snapshot должны содержать все реально используемые
manifests. Resume запрещён, если изменились байты config, export или любого
закреплённого manifest. Snapshot verification обязан покрывать checksums всех
reference-файлов.

## 7. Состояние personal-RL run

Добавить versioned state, например `chessy-personal-rl-state-v1`:

- `phase`: selfplay/train/evaluate/complete;
- generation;
- global step и samples seen по каждому потоку;
- active incumbent export/checksum/generation;
- исходные три model checksums;
- replay/league manifest paths и fingerprints;
- completed self-play game indexes;
- граница текущего training block;
- состояние curriculum;
- состояние RL, historical и feedback samplers;
- RNG Python/NumPy/PyTorch/MPS;
- optimizer и scheduler;
- лучшие style/strength metrics;
- pending candidate export/report paths;
- elapsed time и stop reason.

Остановка допустима в self-play, между потоками одного update, после update и
перед/после arena. Опубликованный snapshot должен соответствовать только
завершённой атомарной границе. Частично выполненный optimizer step нельзя
считать завершённым; после resume он повторяется детерминированно целиком.

Не завершённая self-play game не попадает в replay. Уже sealed game не должна
генерироваться повторно после resume.

## 8. Метрики

### 8.1. Training metrics на каждом update

Логировать как минимум:

- total loss;
- RL policy/value loss и policy entropy;
- historical policy/value loss, top-1 и true-move probability;
- feedback policy/value loss и top-1, если включён;
- фактические размеры каждого потока;
- эффективные коэффициенты каждого компонента;
- gradient norm, learning rate, samples/sec;
- generation, replay draws и style epochs.

Нельзя логировать только общий loss: иначе невозможно понять, усиливается ли
модель ценой забывания стиля.

### 8.2. Baseline до первого optimizer step

До обучения создать immutable baseline reports для:

- incumbent на historical val;
- incumbent на feedback, если включён;
- base RL на historical val;
- personal-supervised на historical val.

Baseline reports должны иметь model checksum, dataset fingerprint, config
fingerprint и детерминированный content fingerprint. Время выполнения не входит
в content fingerprint.

### 8.3. Style validation

На каждой evaluation boundary измерять на historical val:

- policy cross-entropy;
- top-1/top-3/top-5 owner-move accuracy;
- mean/median probability истинного хода;
- value accuracy.

Если feedback включён, считать тот же диагностический набор отдельно. Feedback
не выдавать за held-out validation.

## 9. Двойной promotion gate

Кандидат публикуется только при одновременном прохождении всех gates.

### 9.1. Strength gate

Главный paired arena:

```text
candidate personal_rl vs current personal incumbent
```

- одинаковые стартовые позиции;
- обе расстановки цветов;
- одинаковые MCTS budgets;
- не менее 40 партий для реальной promotion;
- score не ниже `promotion_min_score`;
- нижняя граница confidence interval выше заданного threshold.

Smoke arena может иметь 2 партии, но всегда получает
`eligible_for_promotion=false` и не публикует production export.

### 9.2. Style-retention gate

Относительно incumbent baseline одновременно требуется:

- historical CE regression не больше tolerance;
- historical top-1 не ниже `baseline_top1 * minimum_style_top1_ratio`;
- при feedback: feedback CE regression не больше отдельного tolerance.

Округлять метрики до проверки запрещено.

### 9.3. Load/legal-game gate

После двух основных gates candidate export нужно строго загрузить с CPU и MPS
(если MPS доступен), проверить manifest/checksum и сыграть минимум две legal
sanity games с перестановкой цветов. Только после этого staging directory
атомарно переименовывается в export с ролью `personal_rl`.

Если любой gate провален:

- candidate и отчёты сохраняются для анализа;
- роль `personal_rl` не публикуется;
- league incumbent не меняется;
- run может перейти к следующей генерации только по явно определённой policy.

## 10. Сравнение трёх моделей

Финальный evaluation report должен содержать матрицу:

```text
base_rl
personal_supervised
personal_rl candidate/incumbent
```

Для каждой пары выполнить paired arena с одинаковым набором стартовых позиций и
цветов. Если training incumbent был `personal_feedback`, добавить его отдельной
четвёртой строкой, но не подменять им обязательные три роли.

Отчёт включает W/D/L, score, confidence interval, число партий, MCTS budget,
checksums моделей, fingerprints позиций и config. Никаких Elo-выводов по
маленькому числу партий.

Отдельная style table показывает historical-val метрики всех трёх моделей.
Так видно компромисс между шахматной силой и сходством с владельцем.

## 11. CLI

Добавить отдельные команды, не меняя смысл обычного `rl train`:

```bash
uv run chessy personal-rl train --config configs/personal-rl-local-mps.yaml
uv run chessy personal-rl resume --run runs/<run-id>
uv run chessy personal-rl evaluate --run runs/<run-id>
uv run chessy personal-rl inspect --run runs/<run-id>
```

Допустимо сохранить существующее дерево `personalize`, если итоговые команды
последовательны с CLI проекта. Обязательны:

- `--device auto|cpu|mps|cuda` как operational override, записанный в events;
- test-only `--stop-after-steps`;
- понятная ошибка при несовместимом export/manifest;
- help без импорта или инициализации MPS;
- resume только существующего run, без изменения config.

## 12. Предлагаемая структура кода

```text
src/chessy/
  personal_rl/
    __init__.py
    config.py                 # только если schema.py становится неудобным
    sampler.py
    validation.py
    comparison.py
  training/
    personal_rl_state.py
    personal_rl_trainer.py
tests/
  personal_rl/
    test_config.py
    test_objective.py
    test_sampler_resume.py
    test_training_resume.py
    test_gates.py
    test_comparison.py
```

Переиспользовать существующие replay, self-play, validation, arena, export и
snapshot primitives. Не копировать целиком `rl_trainer.py` и
`feedback_trainer.py`: общие безопасные операции лучше вынести в небольшие
функции после тестов на сохранение поведения.

## 13. Обязательные тесты

### 13.1. Config и provenance

1. Полный personal-RL config проходит strict load/canonical round-trip.
2. Неизвестные поля, NaN/Inf, unsafe paths и нулевой style strength отклоняются.
3. Feedback manifest и strength должны включаться только вместе.
4. Неподходящая export role или архитектура отклоняется до создания run.
5. Подмена байтов любого pinned input ломает resume.

### 13.2. Objective

6. Формула total loss совпадает с ручным расчётом на маленьких tensors.
7. Нормализация каждого потока не зависит от размера другого потока.
8. Feedback sample weight влияет на loss, но не дублирует rows.
9. Нелегальные logits не получают probability mass в policy loss.
10. Нулевой/отключённый feedback не выполняет чтение feedback dataset.

### 13.3. Sampling и replay

11. Все три sampler-а детерминированы для одинакового seed.
12. Historical/feedback per-game caps соблюдаются на каждой epoch.
13. Recent replay fraction соблюдается в доступных границах.
14. Sealed self-play games после resume не дублируются.
15. Не завершённая game не появляется в manifest.

### 13.4. Exact stop/resume

16. CPU uninterrupted N steps и stop K + resume до N дают:
    - одинаковые model tensors;
    - одинаковые optimizer/scheduler states;
    - одинаковый следующий batch каждого потока;
    - одинаковые replay/league fingerprints;
    - одинаковый global step и sample counters.
17. Stop между stream forwards повторяет весь update и не делает двойной step.
18. Corrupted newest snapshot приводит к проверенному fallback либо явной
    ошибке согласно существующей policy.

### 13.5. Gates

19. Strength pass + style fail не публикует export.
20. Style pass + strength fail не публикует export.
21. Оба pass + load failure не публикуют export.
22. Только полный pass создаёт role `personal_rl` и меняет league incumbent.
23. Smoke arena никогда не eligible для production promotion.
24. Historical test не открывается ни train, ни gate кодом.

### 13.6. Comparison и UI

25. Матрица трёх моделей использует paired colors и одинаковые positions.
26. Report checksum/fingerprint воспроизводимы.
27. Успешный `personal_rl` export загружается игровым API.
28. Две sanity games заканчиваются только legal moves.

## 14. Ручная MPS-проверка

Перед сдачей выполнить маленький, но настоящий MPS run на fixtures:

1. Создать fixture exports для обязательных ролей.
2. Создать маленькие historical и optional feedback manifests.
3. Запустить personal RL с `--device mps`.
4. Остановить после первого optimizer step.
5. Продолжить тот же run минимум до четвёртого шага.
6. Проверить events, snapshots и точные sample counters.
7. Довести до evaluation boundary.
8. Проверить оба gates и load/legal-game stage.
9. Открыть export через тот же loader, который использует игровой API.

Зафиксировать peak memory, длительность, размеры replay/run/export. Tiny arena
нужна для plumbing, а не для утверждения о силе.

## 15. Документация

Добавить:

- `docs/PERSONAL_RL.md` с командами train/resume/evaluate/play;
- ADR о mixed RL/style objective и двойном gate;
- описание всех коэффициентов и безопасной настройки на Mac;
- процедуру отката к `personal_supervised`/`personal_feedback`;
- предупреждение, что test запускается один раз для финального отчёта и не
  влияет на выбор модели;
- известные ограничения локального self-play.

Обновить `docs/PROJECT_PLAN.md`, отметив завершение последнего этапа только
после фактического прохождения критериев ниже.

## 16. Порядок реализации

1. Провести gap analysis RL, personalization, feedback, snapshot и arena кода.
2. Зафиксировать ADR и точный config/state/artifact contract.
3. Расширить strict schema и добавить presets.
4. Реализовать multi-stream objective с unit tests.
5. Реализовать детерминированную композицию sampler-ов.
6. Добавить personal-RL state и pinned run references.
7. Реализовать self-play/replay/train state machine.
8. Реализовать atomic stop/resume на всех phase boundaries.
9. Добавить baseline и periodic style validation.
10. Реализовать strength и style-retention gates.
11. Реализовать staging/load/legal-game publication gate.
12. Реализовать трёхсторонний comparison report.
13. Добавить CLI и inspect output.
14. Выполнить targeted tests и security/corruption tests.
15. Выполнить полный pytest.
16. Выполнить ручной MPS stop/resume smoke.
17. Обновить документацию и итоговый отчёт.

## 17. Критерии готовности

Шаг готов только если одновременно выполнено:

1. Run стартует от явно закреплённого personal incumbent checksum.
2. Self-play replay содержит только complete sealed games и provenance модели.
3. RL, historical и optional feedback потоки остаются раздельными.
4. Mixed objective совпадает с документированной формулой.
5. Historical style anchor присутствует на каждом optimizer update.
6. Feedback отключается без чтения или создания feedback artifacts.
7. Frozen historical test не читается trainer/gates.
8. Все sampler states и RNG входят в snapshot.
9. CPU uninterrupted/resume equivalence доказана тестом.
10. MPS stop/resume проходит на реальном backend.
11. Training metrics позволяют отдельно видеть strength/style losses.
12. Baselines записаны до первого optimizer step.
13. Strength gate сравнивает кандидата с текущим personal incumbent.
14. Style gate ограничивает historical и optional feedback regression.
15. Провал любого gate не публикует `personal_rl`.
16. Успешный export строго загружается и играет legal games.
17. Матрица `base_rl`/`personal_supervised`/`personal_rl` сформирована.
18. Export доступен в существующем игровом интерфейсе.
19. Полный pytest зелёный.
20. Generated replay/snapshots/weights/PGN не попадают в Git.

## 18. Что агент должен вернуть

В итоговом отчёте указать:

- найденные архитектурные gaps и переиспользованные компоненты;
- точную формулу loss и фактические coefficients/batch sizes;
- роли, пути и checksums исходных exports;
- fingerprints historical/feedback/replay/league manifests;
- baseline и лучшие style metrics;
- arena W/D/L, score и confidence interval для обязательных сравнений;
- результат каждого promotion gate;
- CPU uninterrupted/resume equivalence;
- MPS stop/resume результат, время и peak memory;
- итоговый export role/path/checksum либо явную причину rejection;
- полный pytest result;
- созданные ignored artifact paths и размеры;
- подтверждение, что frozen test fingerprint и contents не изменились;
- известные ограничения;
- `git status --short`.

Коммит не создавать. Основной агент после ревью сам создаст feature-ветку,
коммит и push по принятому процессу.

## 19. Запрещённые упрощения

Нельзя считать шаг выполненным, если:

- обычный RL trainer просто стартует с personal weights без style loss;
- стиль оценивается по train loss;
- strength arena заменена двумя smoke games;
- кандидат публикуется при прохождении только одного gate;
- historical test используется для checkpoint selection;
- feedback физически добавляется в historical manifest;
- потери суммируются без независимой нормализации потоков;
- resume начинает любой sampler/replay generation заново;
- неудачный кандидат становится self-play incumbent;
- model role определяется по имени каталога;
- config ищет «последний» export неявно;
- corrupted input/snapshot молча пропускается;
- MPS smoke заменён CPU-тестом;
- tiny run объявляется доказательством роста силы.

Главный принцип финального шага: Chessy должен учиться побеждать лучше, не
переставая быть твоим ботом. Усиление и сохранение стиля — два независимых
условия публикации, а не одна усреднённая метрика.
