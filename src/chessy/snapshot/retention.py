"""Conservative retention for indexed, verified snapshot directories only."""
from __future__ import annotations
import os, shutil
from pathlib import Path
from typing import Any
from chessy.snapshot.writer import STEP_RE, verify_snapshot, _atomic_json

def apply_retention(snapshots: Path, index: dict[str, Any], keep_periodic: int) -> list[str]:
    entries=list(index["snapshots"]); periodic=sorted((e for e in entries if "periodic" in e["tags"]),key=lambda e:e["step"],reverse=True)
    pinned={e["name"] for e in periodic[:keep_periodic]}
    # Even best-only/manual schedules must always retain two verified recovery points.
    valid=[]
    for entry in sorted(entries,key=lambda e:e["step"],reverse=True):
        path=snapshots/entry["name"]
        try: verify_snapshot(path)
        except ValueError: continue
        valid.append(entry["name"])
    pinned |= set(valid[:2])
    pinned |= {name for name in (index.get("latest"),index.get("best")) if name}
    pinned |= set(index.get("stages",{}).values())
    pinned |= {e["name"] for e in entries if {"manual","stop"}&set(e["tags"])}
    removed=[]
    for entry in entries:
        name=entry["name"]; path=snapshots/name
        if name in pinned: continue
        # Resolve the exact child before any destructive operation.
        if not STEP_RE.fullmatch(name) or path.parent.resolve()!=snapshots.resolve() or path.is_symlink() or not path.is_dir(): continue
        try: verify_snapshot(path)
        except ValueError: continue
        shutil.rmtree(path); removed.append(name)
    if removed:
        index["snapshots"]=[e for e in entries if e["name"] not in removed]
        _atomic_json(snapshots/"index.json",index)
        fd=os.open(snapshots,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    return removed
