from __future__ import annotations
import json
from pathlib import Path
from chessy.snapshot.writer import verify_snapshot
def select_snapshot(run:object) -> tuple[Path,dict[str,object],list[str]]:
    snapshots=run.path/"snapshots"; index=json.loads((snapshots/"index.json").read_text()); candidates=[]
    if index.get("latest"): candidates.append(index["latest"])
    candidates += [p.name for p in sorted(snapshots.glob("step-*"),reverse=True) if p.name not in candidates]
    rejected=[]
    for name in candidates:
        try: return snapshots/name,verify_snapshot(snapshots/name,expected_run_id=run.id,expected_fingerprint=run.fingerprint),rejected
        except ValueError: rejected.append(name)
    raise ValueError("no valid snapshot available")
