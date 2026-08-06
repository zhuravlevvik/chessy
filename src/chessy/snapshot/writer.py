"""Atomic, checksummed ``chessy-snapshot-v1`` directories."""
from __future__ import annotations
import hashlib,json,os,re,shutil,stat,tempfile
from pathlib import Path,PurePath
from typing import Any
import torch
from safetensors.torch import save_file,load_file
from chessy.config.canonical import canonical_json,fingerprint_bytes
from chessy.config.loader import load_resolved
from chessy.model import ChessyModel
REQUIRED={"model.safetensors","training_state.pt","run_state.json","config.resolved.json","dataset_manifest.json","replay_manifest.json","league_manifest.json","checksums.sha256"}
FEEDBACK_FILE="feedback_manifest.json"; PAYLOAD=REQUIRED-{"checksums.sha256"}; STEP_RE=re.compile(r"step-\d{12}(?:-\d+)?$"); CHECK_RE=re.compile(r"^([0-9a-f]{64})  ([^\s]+)$")
def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
    return h.hexdigest()
def _fsync(path:Path)->None:
    with path.open("rb") as f: os.fsync(f.fileno())
def _atomic_json(path:Path,value:Any)->None:
    temp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temp.write_bytes(canonical_json(value)); _fsync(temp); os.replace(temp,path)
    finally:
        if temp.exists(): temp.unlink()
def _parse_checksums(path:Path, expected:set[str])->dict[str,str]:
    result={}
    for line in path.read_text(encoding="utf-8").splitlines():
        match=CHECK_RE.fullmatch(line)
        if not match: raise ValueError("invalid checksums.sha256")
        digest,name=match.groups(); relative=PurePath(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts)!=1 or name in result: raise ValueError("unsafe checksum path")
        result[name]=digest
    if set(result)!=expected: raise ValueError("checksums must cover snapshot payload exactly")
    return result
def verify_snapshot(path:Path, *, expected_run_id:str|None=None, expected_fingerprint:str|None=None)->dict[str,Any]:
    path=Path(path)
    if not path.is_dir() or path.is_symlink() or not (STEP_RE.fullmatch(path.name) or re.fullmatch(r"\.step-\d{12}(?:-\d+)?\.tmp-[A-Za-z0-9_-]+", path.name)): raise ValueError("invalid snapshot directory")
    names={p.name for p in path.iterdir()}; allowed=(REQUIRED,REQUIRED|{FEEDBACK_FILE})
    if names not in allowed: raise ValueError("snapshot has unexpected or missing files")
    for name in names:
        if not stat.S_ISREG((path/name).lstat().st_mode): raise ValueError(f"snapshot entry must be regular: {name}")
    checks=_parse_checksums(path/"checksums.sha256",names-{"checksums.sha256"})
    for name,digest in checks.items():
        if sha256(path/name)!=digest: raise ValueError(f"checksum mismatch for {name}")
    try:
        state=json.loads((path/"run_state.json").read_text()); config=load_resolved((path/"config.resolved.json").read_bytes())
    except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError("invalid snapshot JSON") from exc
    if state.get("format")!="chessy-run-state-v1": raise ValueError("wrong run state format")
    config_fp=fingerprint_bytes((path/"config.resolved.json").read_bytes())
    if state.get("config_fingerprint")!=config_fp or (expected_fingerprint and config_fp!=expected_fingerprint): raise ValueError("snapshot config fingerprint mismatch")
    if expected_run_id and state.get("run_id")!=expected_run_id: raise ValueError("snapshot run ID mismatch")
    reference_kinds=("dataset","replay","league")+(("feedback",) if FEEDBACK_FILE in names else ())
    for kind in reference_kinds:
        ref=json.loads((path/f"{kind}_manifest.json").read_text())
        if ref.get("format")!="chessy-reference-v1" or ref.get("kind")!=kind: raise ValueError("invalid reference manifest")
        if ref.get("source") is None:
            if ref.get("source_sha256") is not None or ref.get("content") is not None: raise ValueError("invalid empty reference manifest")
        else:
            source=ref.get("source"); checksum=ref.get("source_sha256")
            if not isinstance(source,str) or Path(source).is_absolute() or ".." in Path(source).parts or not isinstance(checksum,str) or re.fullmatch(r"[0-9a-f]{64}",checksum) is None or ref.get("content") is None: raise ValueError("invalid populated reference manifest")
    try: model_state=load_file(str(path/"model.safetensors"),device="cpu")
    except Exception as exc: raise ValueError("invalid safetensors model") from exc
    # Verification must not advance the caller's training RNG stream.
    with torch.random.fork_rng(devices=[]):
        model=ChessyModel(config.model.to_model_config())
        try: model.load_state_dict(model_state,strict=True)
        except RuntimeError as exc: raise ValueError("model does not match config") from exc
    if sum(t.numel() for t in model_state.values())!=state.get("model_parameter_count"): raise ValueError("model parameter count mismatch")
    try: training=torch.load(path/"training_state.pt",map_location="cpu",weights_only=True)
    except Exception as exc: raise ValueError("training state cannot be safely loaded") from exc
    required_training={"format","optimizer_state","scheduler_state","sampler_state","rng_state","gradient_scaler_state"}
    if not isinstance(training,dict) or training.get("format")!="chessy-training-state-v1" or not required_training.issubset(training) or set(training)-required_training-{"rl_state","personal_state","feedback_state"}: raise ValueError("invalid training state")
    return {"run_state":state,"config":config,"model_state":model_state,"training_state":training,"checksum":sha256(path/"checksums.sha256")}
def _index(path:Path)->dict[str,Any]:
    if not path.exists(): return {"format":"chessy-snapshot-index-v1","latest":None,"best":None,"stages":{},"snapshots":[]}
    result=json.loads(path.read_text())
    if result.get("format")!="chessy-snapshot-index-v1": raise ValueError("invalid snapshot index")
    return result
def write_snapshot(run:Any, model:ChessyModel, training_state:dict[str,Any], run_state:dict[str,Any], *, reason:str, tags:set[str], references:dict[str,dict[str,Any]]|None=None) -> Path:
    snapshots=run.path/"snapshots"; name=f"step-{run_state['global_step']:012d}"; final=snapshots/name
    existing=_index(snapshots/"index.json")
    if final.exists() and reason not in {"stop","completed"}:
        # Validation can occur immediately after a stop snapshot, without an
        # optimizer step in between. Preserve both immutable states.
        suffix=2
        while (snapshots/f"{name}-{suffix}").exists(): suffix+=1
        name=f"{name}-{suffix}"; final=snapshots/name
    if final.exists():
        entry=next((x for x in existing["snapshots"] if x["name"]==name),None)
        if entry is None: raise ValueError("existing unindexed snapshot")
        entry["tags"]=sorted(set(entry["tags"])|tags); existing["latest"]=name
        if "best" in entry["tags"]: existing["best"]=name
        existing["stages"][run_state["stage"]]=name; _atomic_json(snapshots/"index.json",existing); return final
    temporary=Path(tempfile.mkdtemp(prefix=f".{name}.tmp-",dir=snapshots))
    try:
        model_state={n:t.detach().to("cpu",torch.float32).contiguous() for n,t in sorted(model.state_dict().items())}; save_file(model_state,str(temporary/"model.safetensors"))
        torch.save(training_state,temporary/"training_state.pt")
        (temporary/"run_state.json").write_bytes(canonical_json(run_state)); (temporary/"config.resolved.json").write_bytes((run.path/"config.resolved.json").read_bytes())
        refs=references or json.loads((run.path/"run_manifest.json").read_text())["references"]
        if set(refs) not in ({"dataset","replay","league"},{"dataset","replay","league","feedback"}): raise ValueError("snapshot references contain unsupported kinds")
        for kind in sorted(refs): (temporary/f"{kind}_manifest.json").write_bytes(canonical_json(refs[kind]))
        payload=PAYLOAD|({FEEDBACK_FILE} if "feedback" in refs else set())
        for name2 in payload: _fsync(temporary/name2)
        (temporary/"checksums.sha256").write_text("".join(f"{sha256(temporary/n)}  {n}\n" for n in sorted(payload)),encoding="utf-8"); _fsync(temporary/"checksums.sha256")
        verify_snapshot(temporary,expected_run_id=run.id,expected_fingerprint=run.fingerprint); temporary.rename(final)
        entry={"name":name,"step":run_state["global_step"],"tags":sorted(tags),"sha256":sha256(final/"checksums.sha256")}; existing["snapshots"].append(entry); existing["snapshots"].sort(key=lambda x:x["step"]); existing["latest"]=name; existing["stages"][run_state["stage"]]=name
        if "best" in tags: existing["best"]=name
        _atomic_json(snapshots/"index.json",existing)
        from chessy.snapshot.retention import apply_retention
        apply_retention(snapshots, existing, run.config.training.keep_last_periodic)
        return final
    except Exception:
        shutil.rmtree(temporary,ignore_errors=True); raise
