from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, json, os, shutil, stat, tempfile
from pathlib import Path, PurePath
from typing import Iterable
import numpy as np
from chessy.config.canonical import canonical_json
from chessy.encoding import ACTION_SIZE
from chessy.replay.codec import encode_board

FORMAT = "chessy-replay-segment-v1"
_FILES = {"samples.npz", "games.pgn", "games.jsonl", "manifest.json", "checksums.sha256"}

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()
def _fsync(path: Path) -> None:
    with path.open("rb") as file: os.fsync(file.fileno())
def _safe_name(value: str) -> None:
    pure = PurePath(value)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1: raise ValueError("unsafe replay path")

@dataclass(frozen=True)
class ReplaySample:
    board: np.ndarray
    policy_actions: tuple[int, ...]
    policy_visits: tuple[int, ...]
    selected_action: int
    value_class: int
    game_index: int
    ply: int
    generation: int

    def validate(self) -> None:
        encode_board(self.board)
        if not self.policy_actions or len(self.policy_actions) != len(self.policy_visits): raise ValueError("sparse policy is empty or malformed")
        if len(set(self.policy_actions)) != len(self.policy_actions) or any(not 0 <= int(a) < ACTION_SIZE for a in self.policy_actions): raise ValueError("policy actions must be unique az73 actions")
        if any(not isinstance(v, (int, np.integer)) or v < 0 for v in self.policy_visits) or sum(self.policy_visits) <= 0: raise ValueError("policy visits must be non-negative with positive total")
        if self.selected_action not in self.policy_actions: raise ValueError("selected action must appear in sparse policy")
        if self.value_class not in (0, 1, 2): raise ValueError("value class must be loss/draw/win")
        if min(self.game_index, self.ply, self.generation) < 0: raise ValueError("sample identifiers must be non-negative")

@dataclass(frozen=True)
class SealedGame:
    game_id: str
    game_index: int
    generation: int
    stage: str
    source_kind: str
    initial_fen: str
    result: str
    termination: str
    samples: tuple[ReplaySample, ...]
    pgn: str
    metadata: dict[str, object]
    complete: bool = True

def _arrays(games: Iterable[SealedGame]) -> dict[str, np.ndarray]:
    samples = [sample for game in games for sample in game.samples]
    if not samples: raise ValueError("cannot seal an empty replay segment")
    for sample in samples: sample.validate()
    offsets = [0]; actions: list[int] = []; visits: list[int] = []
    for sample in samples:
        actions.extend(sample.policy_actions); visits.extend(sample.policy_visits); offsets.append(len(actions))
    return {"boards": np.stack([encode_board(sample.board) for sample in samples]), "policy_offsets": np.asarray(offsets, dtype=np.int64), "policy_actions": np.asarray(actions, dtype=np.uint16), "policy_visits": np.asarray(visits, dtype=np.uint32), "value_class": np.asarray([s.value_class for s in samples], dtype=np.uint8), "selected_action": np.asarray([s.selected_action for s in samples], dtype=np.uint16), "game_index": np.asarray([s.game_index for s in samples], dtype=np.uint32), "ply": np.asarray([s.ply for s in samples], dtype=np.uint16), "generation": np.asarray([s.generation for s in samples], dtype=np.uint32)}

def _validate_arrays(data: dict[str, np.ndarray]) -> None:
    names = {"boards", "policy_offsets", "policy_actions", "policy_visits", "value_class", "selected_action", "game_index", "ply", "generation"}
    if set(data) != names: raise ValueError("invalid replay array names")
    n = data["boards"].shape[0]
    checks = (("boards", np.uint8, (n,119,8,8)), ("policy_offsets", np.int64, (n+1,)), ("policy_actions", np.uint16, None), ("policy_visits", np.uint32, None), ("value_class", np.uint8, (n,)), ("selected_action", np.uint16, (n,)), ("game_index", np.uint32, (n,)), ("ply", np.uint16, (n,)), ("generation", np.uint32, (n,)))
    for name, dtype, shape in checks:
        a = data[name]
        if a.dtype != dtype or (shape is not None and a.shape != shape) or a.ndim != (4 if name == "boards" else 1): raise ValueError(f"invalid {name} array")
    off, actions, visits = data["policy_offsets"], data["policy_actions"], data["policy_visits"]
    if n == 0 or off[0] != 0 or off[-1] != len(actions) or len(actions) != len(visits) or np.any(off[1:] < off[:-1]): raise ValueError("invalid sparse policy offsets")
    if np.any(actions >= ACTION_SIZE) or not np.issubdtype(visits.dtype,np.unsignedinteger) or np.any(data["value_class"] > 2): raise ValueError("invalid sparse policy values")
    for i in range(n):
        current = actions[off[i]:off[i+1]]
        current_visits=visits[off[i]:off[i+1]]
        if len(current) == 0 or int(current_visits.sum()) <= 0 or len(np.unique(current)) != len(current) or data["selected_action"][i] not in current: raise ValueError("invalid sample legal policy")

def _checksums(path: Path) -> dict[str, str]:
    lines = (path / "checksums.sha256").read_text().splitlines(); result: dict[str, str] = {}
    for line in lines:
        try: digest, name = line.split("  ", 1)
        except ValueError as exc: raise ValueError("invalid replay checksums") from exc
        _safe_name(name)
        if len(digest) != 64 or name in result: raise ValueError("invalid replay checksums")
        result[name] = digest
    if set(result) != _FILES - {"checksums.sha256"}: raise ValueError("replay checksums do not cover payload")
    return result

def verify_segment(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_dir() or path.is_symlink() or {p.name for p in path.iterdir()} != _FILES: raise ValueError("invalid replay segment directory")
    for entry in path.iterdir():
        if not stat.S_ISREG(entry.lstat().st_mode): raise ValueError("replay segment files must be regular")
    checks = _checksums(path)
    if any(_sha(path / name) != digest for name, digest in checks.items()): raise ValueError("replay checksum mismatch")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != FORMAT: raise ValueError("invalid replay segment manifest")
    with np.load(path / "samples.npz", allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    _validate_arrays(data)
    records = [json.loads(line) for line in (path / "games.jsonl").read_text().splitlines() if line]
    if len(records) != manifest.get("game_count") or any(record.get("format") != "chessy-selfplay-game-v1" or not record.get("complete", False) for record in records): raise ValueError("invalid replay game metadata")
    if manifest.get("sample_count") != data["boards"].shape[0]: raise ValueError("replay manifest sample count mismatch")
    return {"manifest": manifest, "arrays": data, "checksum": _sha(path / "checksums.sha256")}

def write_segment(root: Path, *, generation: int, ordinal: int, games: Iterable[SealedGame], run_id: str, model_checksum: str) -> Path:
    games = tuple(sorted(games, key=lambda game: game.game_index))
    if not games or any(not game.complete for game in games) or len({g.game_id for g in games}) != len(games): raise ValueError("only unique completed games may be sealed")
    arrays = _arrays(games); root = Path(root); segments = root / "segments"; segments.mkdir(parents=True, exist_ok=True)
    digest_seed = hashlib.sha256("|".join(game.game_id for game in games).encode()).hexdigest()[:12]
    final = segments / f"segment-{generation}-{ordinal}-{digest_seed}"
    if final.exists():
        checked = verify_segment(final)
        if checked["manifest"].get("games") == [g.game_id for g in games]: return final
        raise FileExistsError("existing replay segment has a different idempotency key")
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=segments))
    try:
        np.savez_compressed(temporary / "samples.npz", **arrays)
        (temporary / "games.pgn").write_text("\n\n".join(game.pgn.strip() for game in games) + "\n", encoding="utf-8")
        records = []
        for game in games:
            reserved={"format","game_id","game_index","generation","stage","source_kind","initial_fen","result","termination","ply_count","model_checksum","complete"}
            if reserved & set(game.metadata): raise ValueError("game metadata overrides reserved fields")
            record = {"format":"chessy-selfplay-game-v1", "game_id":game.game_id, "game_index":game.game_index, "generation":game.generation, "stage":game.stage, "source_kind":game.source_kind, "initial_fen":game.initial_fen, "result":game.result, "termination":game.termination, "ply_count":len(game.samples), "model_checksum":model_checksum, "complete":True, **game.metadata}
            records.append(record)
        (temporary / "games.jsonl").write_text("".join(canonical_json(record).decode() for record in records), encoding="utf-8")
        manifest = {"format":FORMAT, "generation":generation, "ordinal":ordinal, "run_id":run_id, "model_checksum":model_checksum, "games":[g.game_id for g in games], "game_count":len(games), "sample_count":int(arrays["boards"].shape[0]), "arrays": {name:{"dtype":str(value.dtype),"shape":list(value.shape)} for name,value in arrays.items()}}
        (temporary / "manifest.json").write_bytes(canonical_json(manifest))
        for name in _FILES - {"checksums.sha256"}: _fsync(temporary / name)
        (temporary / "checksums.sha256").write_text("".join(f"{_sha(temporary / name)}  {name}\n" for name in sorted(_FILES - {"checksums.sha256"})), encoding="utf-8"); _fsync(temporary / "checksums.sha256")
        verify_segment(temporary); temporary.rename(final)
        return final
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise
