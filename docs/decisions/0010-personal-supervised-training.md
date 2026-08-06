# ADR-0010: персональный supervised fine-tuning

Статус: утверждено

Исторические позиции публикуются только как immutable `chessy-personal-dataset-v1`
segments. Каждая позиция содержит lossless `board119-v1` history, sparse legal
actions, фактический az73 action и WDL с точки зрения side to move.

Fine-tuning допускается только из явно указанного checksummed `base_rl` export.
Цель: `policy CE + 0.25 * WDL CE`, с mask до softmax. Вес 0.75/1.0 действует
исключительно при weighted game-capped sampling, максимум 16 позиций на игру за
эпоху. Test split закрыт capability-проверкой и не участвует в trainer-е.

Best snapshot выбирается только по полной validation policy cross-entropy;
экспорт `personal_supervised` возможен лишь при improvement над baseline минимум
на configured `min_delta` и после строгой загрузки export.
