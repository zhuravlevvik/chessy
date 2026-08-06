# ADR-0012: Personal RL keeps strength and style as independent signals

`personal_rl` is a separate run type.  It starts from an explicitly named
personal export and pins its role, model checksum, historical manifest and
optional feedback manifest before a run directory exists.  It never discovers
"the latest" model.

Each optimizer update has three independently averaged mini-batches:

```text
L = L_rl + style_strength * L_historical + feedback_strength * L_feedback
```

`L_rl` is MCTS-policy plus WDL cross-entropy.  Historical and feedback terms
are masked hard-move policy plus WDL cross-entropy.  The feedback confidence
weight scales its distinct stream; it does not duplicate rows into history.

Promotion requires both a paired strength arena against the current personal
incumbent and the historical validation retention gate.  A smoke arena is
never promotion-eligible.  Export publication happens only after both gates,
strict CPU/MPS loading, and legal arena games.  A rejected candidate is kept
for analysis but cannot become the next self-play incumbent.

The frozen historical test split is recorded only as a fingerprint during
training.  It is not opened by training or gate code; it is reserved for one
final report.
