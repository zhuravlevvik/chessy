# Human-feedback training

After a finished and explicitly confirmed game, inspect and verify it before
building a dataset:

```bash
uv run chessy feedback inspect --input data/human_feedback
uv run chessy feedback verify --game data/human_feedback/<game-id>
uv run chessy feedback build --input data/human_feedback --output data/human_feedback_encoded
uv run chessy feedback dataset-verify --manifest data/human_feedback_encoded/manifests/feedback-dataset-<fingerprint>.json
uv run chessy personalize feedback --config configs/personal-feedback-local-mps.yaml
```

The feedback diagnostic is measured on the same confirmed feedback samples and
is an adaptation check, not a generalization claim. Historical validation stays
separate; historical test is not loaded by this workflow.
