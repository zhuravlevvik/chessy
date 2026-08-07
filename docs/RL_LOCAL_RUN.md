# Локальный self-play RL-run

Чтобы во время run смотреть выбранную self-play партию и архив поколений,
запустите обычный локальный UI во втором терминале и выберите **Смотреть
обучение**. Настройки описаны в `docs/TRAINING_OBSERVER.md`.

Короткая проверка контура на CPU:

```bash
uv run chessy train rl --config configs/rl-smoke.yaml
```

Она создаёт run в `runs/` и immutable replay в `replay/`; оба пути игнорируются
Git. Проверить артефакты можно так:

```bash
uv run chessy run inspect runs/<run-id>
uv run chessy replay verify replay/manifests/<manifest>.json
uv run chessy snapshot verify runs/<run-id>/snapshots/<snapshot>
```

Для первого ручного MPS-замера используйте консервативный пресет (он не
запускается тестами и не является заявлением о силе):

```bash
uv run chessy train rl --config configs/rl-local-mps.yaml
```

Терминал показывает текущую фазу, прогресс self-play и arena по партиям,
периодические training-метрики и путь каждого сохранённого snapshot. Вывод
печатается сразу, поэтому длительный self-play не выглядит зависшим.

Перед длительным запуском сначала подтвердите доступность MPS и измерьте
positions/sec. Возобновление использует последний проверенный snapshot:

```bash
uv run chessy train rl --resume runs/<run-id>
```
