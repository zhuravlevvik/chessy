"""Append-only, fsynced JSONL logs with conservative crash recovery."""
from __future__ import annotations
import json, math, os, threading
from datetime import datetime, timezone
from pathlib import Path
from chessy.config.canonical import canonical_json
def utcnow() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def recover_log(path: Path) -> bool:
    if not path.exists(): return False
    data=path.read_bytes()
    if not data or data.endswith(b"\n"): return False
    prefix, _, tail=data.rpartition(b"\n")
    try:
        json.loads(tail)
    except (UnicodeDecodeError,json.JSONDecodeError):
        recovered=path.with_name(path.name+".recovered-fragment"); recovered.write_bytes(tail)
        with path.open("wb") as f: f.write(prefix+(b"\n" if prefix else b"")); f.flush(); os.fsync(f.fileno())
        return True
    with path.open("ab") as f: f.write(b"\n"); f.flush(); os.fsync(f.fileno())
    return True
class JsonlLog:
    def __init__(self,path: Path, kind: str) -> None:
        self.path,self.kind=path,kind; path.parent.mkdir(parents=True,exist_ok=True)
        self._lock=threading.Lock(); self.recovered=recover_log(path); self.sequence=0; self.last_step=-1
        if path.exists():
            for no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
                try: record=json.loads(line)
                except json.JSONDecodeError as exc: raise ValueError(f"corrupt JSONL line {no}") from exc
                if record.get("format") != f"chessy-{kind}-v1": raise ValueError("wrong log format")
                if kind=="events":
                    sequence=record.get("sequence")
                    if not isinstance(sequence,int) or isinstance(sequence,bool) or sequence != self.sequence+1: raise ValueError("event sequence must increase by one")
                    self.sequence=sequence
                else:
                    step=record.get("step")
                    if not isinstance(step,int) or isinstance(step,bool) or step <= self.last_step: raise ValueError("metric steps must strictly increase")
                    self.last_step=step
    def append_event(self,event_type: str,payload: dict[str,object]|None=None) -> None:
        if self.kind!="events": raise ValueError("not event log")
        with self._lock:
            self.sequence+=1; self._append({"format":"chessy-events-v1","sequence":self.sequence,"timestamp":utcnow(),"type":event_type,"payload":payload or {}})
    def append_metric(self,step:int,epoch:int,metrics:dict[str,float|int]) -> None:
        if self.kind!="metrics": raise ValueError("not metric log")
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in metrics.values()): raise ValueError("metrics must be finite numbers")
        with self._lock:
            if step<=self.last_step: raise ValueError("metric step must increase")
            self.last_step=step; self._append({"format":"chessy-metrics-v1","step":step,"epoch":epoch,"timestamp":utcnow(),"metrics":metrics})
    def _append(self,record:dict[str,object]) -> None:
        data=canonical_json(record)
        fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)
        try:
            view=memoryview(data)
            while view: view=view[os.write(fd,view):]
            os.fsync(fd)
        finally: os.close(fd)
