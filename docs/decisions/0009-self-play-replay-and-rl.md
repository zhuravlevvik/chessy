# ADR 0009 — self-play, replay и RL-поколения

Дата: 2026-08-06
Статус: принято

Chessy обучается последовательными поколениями. Веса `generation-N` неизменны
во время self-play; после завершения партий coordinator атомарно публикует
checksummed replay segment и новый immutable replay manifest. Только samples,
связанные с указанным manifest, доступны trainer-у.

Каждый sample хранит `board119-v1`, sparse root MCTS visits, выбранный ход и
WDL-исход партии с точки зрения стороны хода. Единственная награда — исход
партии (loss/draw/win); material и другие shaping-сигналы не используются.
Policy loss — cross-entropy с нормализованными visit counts под legal mask,
value loss — WDL cross-entropy.

Candidate экспортируется отдельно и оценивается в deterministic paired-color
arena: root noise выключен, temperature равна нулю. Promotion возможен только
после завершённого отчёта, достаточного числа игр, score threshold и нижней
границы доверительного интервала. Короткий smoke-run не может поставить тег
`promoted` или `base_rl`.

Snapshot хранит model/optimizer/scheduler/RNG/sampler и сериализуемое состояние
generation, phase, curriculum, replay и league; replay bytes в snapshot не
копируются. Resume сначала повторно проверяет referenced immutable artifacts.
