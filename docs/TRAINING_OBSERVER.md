# Training observer

Chessy can publish one self-play game per generation for local spectating. It
uses the existing loopback play server and does not open another port.

Enable it in an RL or personal-RL config:

```yaml
observer:
  enabled: true
  archive_every_generations: 5
  live_game_index: 0
```

During self-play, `live.json` is atomically replaced after every move. When the
selected game finishes, its PGN and replayable FEN frames are archived:

```text
runs/<run-id>/showcase/
  live.json
  generation-0000/
    game.pgn
    manifest.json
```

Start the normal local interface while training is running:

```bash
uv run chessy play
```

Choose **Смотреть обучение** on the start screen. The observer polls the local
filesystem through the same FastAPI application once per second. It can follow
the current game automatically or replay any archived generation with manual
move controls.

The play screen also discovers `chessy-model-v1` exports below
`runs/*/exports/` automatically. Its model list refreshes while the start screen
is open; weights are loaded only after a discovered model is selected for a
game, so merely listing checkpoints does not consume model memory.

Observed self-play includes root noise and temperature because it is a real
training game, not a strength evaluation. Arena reports remain the source of
measured progress. Observer files live below ignored `runs/` directories and
do not enter Git or the training replay manifest.
