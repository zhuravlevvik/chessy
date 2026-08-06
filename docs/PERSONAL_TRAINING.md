# Personal supervised training

Для быстрого локального smoke сначала создайте tiny fixture dataset/model
(они останутся в игнорируемом `runs/`), затем запустите preset:

```bash
uv run chessy dataset personal prepare-smoke
uv run chessy personalize train --config configs/personal-supervised-smoke.yaml
```

Соберите неизменяемый датасет:

```bash
uv run chessy dataset personal build
uv run chessy dataset personal verify --manifest data/personal/encoded/manifests/personal-dataset-<fingerprint>.json
```

В `configs/personal-supervised-local-mps.yaml` замените placeholder manifest и
укажите проверенный export с metadata `role: base_rl`. Затем запускайте:

```bash
uv run chessy personalize train --config configs/personal-supervised-local-mps.yaml
```

Обычная validation и compare всегда используют только `val`:

```bash
uv run chessy personalize validate --model runs/<run>/exports/personal-supervised --dataset <manifest>
uv run chessy personalize compare --base artifacts/base_rl --personal runs/<run>/exports/personal-supervised --dataset <manifest>
```

`test` намеренно не имеет обычной CLI-команды: final evaluation требует отдельной
явной процедуры и не должна использоваться для настройки или выбора snapshot.
