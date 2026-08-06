from __future__ import annotations
import hashlib, json, os
from dataclasses import dataclass
from pathlib import Path
from chessy.config.canonical import canonical_json, fingerprint
from chessy.replay.segment import verify_segment

@dataclass(frozen=True)
class ReplayManifest:
    path: Path
    content: dict[str, object]
    @property
    def fingerprint(self) -> str: return str(self.content["fingerprint"])

def write_manifest(root: Path, *, run_id: str, generation: int, segments: list[Path], active_max_samples: int, policy: dict[str, object] | None = None) -> ReplayManifest:
    root = Path(root); entries=[]; sample_count=game_count=logical=0; generations: dict[str,int]={}; stages: dict[str,int]={}
    for segment in segments:
        checked = verify_segment(segment); item = checked["manifest"]; rel = segment.resolve().relative_to(root.resolve()).as_posix()
        entries.append({"path":rel, "manifest_sha256":hashlib.sha256((segment / "manifest.json").read_bytes()).hexdigest(), "checksums_sha256":checked["checksum"], "sample_count":item["sample_count"], "game_count":item["game_count"], "generation":item["generation"]})
        sample_count += int(item["sample_count"]); game_count += int(item["game_count"]); logical += sum(p.stat().st_size for p in segment.iterdir())
        generations[str(item["generation"])] = generations.get(str(item["generation"]),0)+int(item["sample_count"])
        for line in (segment / "games.jsonl").read_text().splitlines():
            stage=json.loads(line)["stage"]; stages[stage]=stages.get(stage,0)+1
    content={"format":"chessy-replay-manifest-v1", "run_id":run_id, "generation":generation, "segments":entries, "sample_count":sample_count, "game_count":game_count, "generation_histogram":generations, "stage_histogram":stages, "active_window":{"max_samples":active_max_samples, **(policy or {})}, "logical_bytes":logical, "physical_bytes":logical}
    content["fingerprint"] = fingerprint(content)
    manifests=root / "manifests"; manifests.mkdir(parents=True, exist_ok=True); path=manifests / f"replay-{generation}-{content['fingerprint'][:12]}.json"
    if not path.exists():
        temporary=path.with_name(f".{path.name}.tmp-{os.getpid()}"); temporary.write_bytes(canonical_json(content)); os.replace(temporary,path)
    return ReplayManifest(path, content)

def load_manifest(path: Path, *, verify: bool = True) -> ReplayManifest:
    path=Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError("replay manifest must be a regular file")
    content=json.loads(path.read_text())
    if not isinstance(content,dict) or not isinstance(content.get("segments"),list) or not isinstance(content.get("active_window"),dict): raise ValueError("invalid replay manifest structure")
    expected=dict(content); actual=expected.pop("fingerprint",None)
    if content.get("format")!="chessy-replay-manifest-v1" or actual != fingerprint(expected): raise ValueError("invalid replay manifest fingerprint")
    root=path.parent.parent
    for entry in content["segments"]:
        if not isinstance(entry,dict) or not isinstance(entry.get("path"),str): raise ValueError("invalid replay segment reference")
        unresolved=root / entry["path"]
        if unresolved.is_symlink(): raise ValueError("replay manifest has unsafe segment path")
        candidate=unresolved.resolve()
        if not candidate.is_relative_to(root.resolve()) or not candidate.is_dir(): raise ValueError("replay manifest has unsafe segment path")
        if verify:
            checked=verify_segment(candidate)
            manifest_sha=hashlib.sha256((candidate/"manifest.json").read_bytes()).hexdigest()
            if manifest_sha != entry.get("manifest_sha256") or checked["checksum"] != entry.get("checksums_sha256"): raise ValueError("replay segment does not match manifest reference")
    return ReplayManifest(path, content)
