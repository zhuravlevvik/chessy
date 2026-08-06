# Personal RL

Personal RL strengthens a named personal model while anchoring it to the
owner's frozen historical training data.  It is not online learning: saving
feedback never starts training automatically.

Start a small plumbing check:

```bash
uv run chessy personal-rl train --config configs/personal-rl-smoke.yaml
```

Run the local Mac profile (replace all placeholder artifact paths first):

```bash
uv run chessy personal-rl train --config configs/personal-rl-local-mps.yaml --device mps
uv run chessy personal-rl resume --run runs/<run-id> --device mps
uv run chessy personal-rl inspect --run runs/<run-id>
uv run chessy personal-rl evaluate --run runs/<run-id>
uv run chessy play --model runs/<run-id>/exports/personal_rl-g<generation>-s<step>
```

The objective is `L_rl + style_strength * L_historical + feedback_strength *
L_feedback`.  The initial local values are RL batches 32, historical batches
16, feedback batches 8, and `style_strength: 0.20`.  On MPS, reduce batch
sizes first if memory is tight; do not silently remove the style term.

The candidate must pass both the paired arena against the active personal
incumbent and historical style retention (CE and top-1).  Feedback diagnostics,
when enabled, are train-only and not a generalisation result.  The frozen test
split is intentionally not opened by this workflow; use it once for a final
report only.

If a candidate is rejected, use the explicitly pinned `personal_supervised` or
`personal_feedback` export in the game command.  All inputs are checksummed,
and changing their bytes makes resume fail rather than silently changing an
experiment.  Local self-play is deliberately small and slow; smoke results do
not establish playing strength.
