from __future__ import annotations
import hashlib,json,os,platform,subprocess,tempfile
from pathlib import Path
import torch
from chessy.config.canonical import canonical_json
from chessy.config.loader import load_resolved
from chessy.config.schema import ChessyConfig
from chessy.model import resolve_device
from chessy.run.identity import make_run_id,valid_run_id
from chessy.run.logging import JsonlLog,utcnow
def _sha(path:Path)->str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def _git(root:Path)->tuple[str|None,bool|None]:
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
        dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True))
        return commit,dirty
    except (OSError,subprocess.CalledProcessError): return None,None
def references(config:ChessyConfig,root:Path)->dict[str,dict[str,object]]:
    out={}
    root=root.resolve()
    for kind,source in (("dataset",config.artifacts.dataset_manifest),("replay",config.artifacts.replay_manifest),("league",config.artifacts.league_manifest)):
        if source is None: out[kind]={"format":"chessy-reference-v1","kind":kind,"source":None,"source_sha256":None,"content":None}; continue
        unresolved=root/source
        if unresolved.is_symlink(): raise ValueError(f"{kind} manifest must not be a symlink")
        path=unresolved.resolve()
        if not path.is_relative_to(root) or not path.is_file(): raise ValueError(f"{kind} manifest must be a regular file inside the project")
        raw=path.read_bytes()
        try: content=json.loads(raw)
        except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError(f"invalid {kind} manifest JSON") from exc
        out[kind]={"format":"chessy-reference-v1","kind":kind,"source":source,"source_sha256":hashlib.sha256(raw).hexdigest(),"content":content}
    return out
class Run:
    def __init__(self,path:Path,config:ChessyConfig,fingerprint:str) -> None:
        self.path,self.config,self.fingerprint=path,config,fingerprint; self.id=path.name
        self.events=JsonlLog(path/"events.jsonl","events"); self.metrics=JsonlLog(path/"metrics.jsonl","metrics")
        if self.events.recovered: self.events.append_event("log_recovered",{"log":"events.jsonl"})
        if self.metrics.recovered: self.events.append_event("log_recovered",{"log":"metrics.jsonl"})
    @classmethod
    def create(cls,root:Path,config:ChessyConfig,source:bytes,resolved:bytes,fingerprint:str,parent:dict[str,object]|None=None)->"Run":
        runs=(root/config.artifacts.runs_dir); runs.mkdir(parents=True,exist_ok=True)
        identifier=make_run_id(config.name,fingerprint); candidate=runs/identifier; suffix=1
        while candidate.exists(): suffix+=1; candidate=runs/f"{identifier}-{suffix}"
        temp=Path(tempfile.mkdtemp(prefix=f".{candidate.name}.tmp-",dir=runs))
        try:
            (temp/"snapshots").mkdir(); (temp/"exports").mkdir(); (temp/"config.source.yaml").write_bytes(source); (temp/"config.resolved.json").write_bytes(resolved)
            commit,dirty=_git(root); device=resolve_device(config.device)
            manifest={"format":"chessy-run-v1","run_id":candidate.name,"created_at":utcnow(),"name":config.name,"config_fingerprint":fingerprint,"git":{"commit":commit,"dirty":dirty},"uv_lock":{"sha256":_sha(root/"uv.lock") if (root/"uv.lock").is_file() else None},"python":{"version":platform.python_version(),"executable":os.sys.executable},"platform":{"system":platform.system(),"machine":platform.machine(),"release":platform.release()},"torch":{"version":torch.__version__},"requested_device":config.device,"resolved_device":device.type,"project_version":"0.1.0","parent":parent,"references":references(config,root)}
            (temp/"run_manifest.json").write_bytes(canonical_json(manifest)); temp.rename(candidate)
        except Exception:
            import shutil; shutil.rmtree(temp,ignore_errors=True); raise
        run=cls(candidate,config,fingerprint); run.events.append_event("run_created",{}); return run
    @classmethod
    def open(cls,path:Path)->"Run":
        unresolved=Path(path)
        if unresolved.is_symlink(): raise ValueError("invalid run path")
        path=unresolved.resolve()
        if not valid_run_id(path.name) or not path.is_dir(): raise ValueError("invalid run path")
        manifest=json.loads((path/"run_manifest.json").read_text())
        if manifest.get("format")!="chessy-run-v1" or manifest.get("run_id")!=path.name: raise ValueError("invalid run manifest")
        raw=(path/"config.resolved.json").read_bytes(); config=load_resolved(raw)
        from chessy.config.canonical import fingerprint_bytes
        actual=fingerprint_bytes(raw)
        if manifest.get("config_fingerprint") != actual: raise ValueError("run config fingerprint mismatch")
        return cls(path,config,actual)
