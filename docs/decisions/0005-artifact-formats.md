# ADR-0005: Форматы модели и training snapshot

Статус: утверждено
Дата: 2026-08-06

## Решение

### Playable export

Использовать формат `chessy-model-v1`:

- `model.safetensors`;
- `manifest.json`;
- `checksums.sha256`.

Export содержит только веса и достаточные метаданные для создания модели,
проверки `board119-v1`/`az73-v1` и запуска inference.

### Training snapshot

Использовать формат директории `chessy-snapshot-v1`:

- `model.safetensors`;
- `training_state.pt`;
- `run_state.json`;
- resolved config;
- dataset, replay и league manifests;
- `checksums.sha256`.

Snapshot хранит optimizer, scheduler, sampler, global step и RNG-состояния,
включая MPS. Replay-сегменты указываются ссылками и не копируются.

Запись выполняется во временную директорию, проверяется и завершается атомарным
переименованием. `resume` требует совместимости форматов; несовместимые изменения
выполняются через новый `fork` run.
