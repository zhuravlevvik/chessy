# ADR-0011: Immutable human-feedback pipeline

Confirmed games remain immutable `chessy-human-feedback-v1` directories. They
are independently replayed and then encoded into a separate,
content-addressed `chessy-human-feedback-dataset-v1`; historical personal
splits, especially the frozen test split, are never extended.

Training mixes the streams with a hard per-batch cap of 25% feedback positions.
The 4.0 feedback priority is an explicit per-sample loss weight, normalized by
the sum of all batch weights, rather than duplicated rows. A `personal_feedback`
export is gated on feedback CE improvement and bounded historical validation
regression, then strict loading plus two legal sanity games.
