"""Small synthetic trainer proving run and snapshot plumbing, not chess strength."""
from __future__ import annotations
import math,random,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from chessy.config import load_config
from chessy.config.schema import ChessyConfig
from chessy.model import ChessyModel,resolve_device
from chessy.run import Run
from chessy.snapshot import capture_rng,restore_rng,write_snapshot
from chessy.snapshot.loader import select_snapshot
from chessy.training import StatefulBatchSampler,StopController

DATASET_SIZE=97
@dataclass
class SmokeState:
    step:int=0; samples_seen:int=0; best_loss:float=float("inf"); best_step:int|None=None; elapsed:float=0.0
def _seed(seed:int)->np.random.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); return np.random.default_rng(seed)
def _scheduler(optimizer:torch.optim.Optimizer, config:ChessyConfig)->torch.optim.lr_scheduler.LambdaLR:
    warmup,total,minimum=config.scheduler.warmup_steps,config.scheduler.total_steps,config.scheduler.minimum_lr_ratio
    def factor(step:int)->float:
        if step < warmup: return (step+1)/max(1,warmup)
        ratio=(step-warmup)/max(1,total-warmup); return minimum+(1-minimum)*0.5*(1+math.cos(math.pi*ratio))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
def _synthetic(indices:torch.Tensor, device:torch.device)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    # Every item is a pure function of its index; no random source is consumed here.
    base=indices.to(torch.float32).view(-1,1,1,1); coords=torch.arange(119*8*8,dtype=torch.float32).view(1,119,8,8)
    boards=torch.remainder(base*17+coords*13,101).div(50.0).sub(1.0).to(device)
    policy=torch.remainder(indices*37+11,4672).to(device); value=torch.remainder(indices,3).to(device)
    return boards,policy,value
def _optimizer(model:ChessyModel, config:ChessyConfig)->torch.optim.Optimizer:
    c=config.optimizer; return torch.optim.AdamW(model.parameters(),lr=c.learning_rate,weight_decay=c.weight_decay,betas=(c.beta1,c.beta2),eps=c.epsilon)
def _move_optimizer(optimizer:torch.optim.Optimizer, device:torch.device)->None:
    for value in optimizer.state.values():
        for key,item in value.items():
            if isinstance(item,torch.Tensor): value[key]=item.to(device)
def _run_state(run:Run,state:SmokeState,sampler:StatefulBatchSampler,model:ChessyModel,reason:str,stop_reason:str|None)->dict[str,object]:
    return {"format":"chessy-run-state-v1","run_id":run.id,"config_fingerprint":run.fingerprint,"global_step":state.step,"epoch":sampler.epoch,"samples_seen":state.samples_seen,"stage":"smoke","best_metric":None if math.isinf(state.best_loss) else state.best_loss,"best_step":state.best_step,"total_elapsed_seconds":state.elapsed,"last_completed_batch":state.step,"snapshot_reason":reason,"stop_reason":stop_reason,"created_at":__import__("chessy.run.logging",fromlist=["utcnow"]).utcnow(),"model_parameter_count":sum(p.numel() for p in model.parameters()),"optimizer":"adamw","scheduler":"warmup-cosine"}
def _snapshot(run:Run,model:ChessyModel,optimizer:torch.optim.Optimizer,scheduler:object,sampler:StatefulBatchSampler,np_generator:np.random.Generator,state:SmokeState,reason:str,stop_reason:str|None,tags:set[str])->Path:
    run.events.append_event("snapshot_started",{"step":state.step,"reason":reason})
    # A corrupted historical directory is immutable. Continue training and let a
    # later step become the new valid latest rather than replacing evidence.
    destination = run.path / "snapshots" / f"step-{state.step:012d}"
    if destination.exists():
        run.events.append_event("snapshot_failed", {"step": state.step, "error": "ExistingSnapshot"})
        return destination
    try:
        payload={"format":"chessy-training-state-v1","optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"sampler_state":sampler.state_dict(),"rng_state":capture_rng(np_generator),"gradient_scaler_state":None}
        result=write_snapshot(run,model,payload,_run_state(run,state,sampler,model,reason,stop_reason),reason=reason,tags=tags)
    except Exception as exc:
        run.events.append_event("snapshot_failed",{"step":state.step,"error":type(exc).__name__}); raise
    run.events.append_event("snapshot_completed",{"step":state.step,"snapshot":result.name}); return result
def _prepare_new(root:Path, config_path:Path)->tuple[Run,ChessyConfig,np.random.Generator,ChessyModel,torch.optim.Optimizer,object,StatefulBatchSampler,SmokeState,torch.device]:
    config,source,resolved,fp=load_config(config_path); np_gen=_seed(config.seed); run=Run.create(root,config,source,resolved,fp); device=resolve_device(config.device); model=ChessyModel(config.model.to_model_config()).to(device); optimizer=_optimizer(model,config); scheduler=_scheduler(optimizer,config); sampler=StatefulBatchSampler(DATASET_SIZE,config.training.batch_size,seed=config.seed); return run,config,np_gen,model,optimizer,scheduler,sampler,SmokeState(),device
def _prepare_resume(path:Path, requested_device:str|None)->tuple[Run,ChessyConfig,np.random.Generator,ChessyModel,torch.optim.Optimizer,object,StatefulBatchSampler,SmokeState,torch.device]:
    run=Run.open(path); snapshot,checked,rejected=select_snapshot(run)
    if rejected: run.events.append_event("resume_fallback",{"rejected":rejected,"selected":snapshot.name})
    run.events.append_event("resume_started",{"snapshot":snapshot.name})
    config=run.config; target=resolve_device(requested_device or config.device); np_gen=np.random.default_rng(config.seed); model=ChessyModel(config.model.to_model_config()).to(target); model.load_state_dict(checked["model_state"],strict=True)
    # LambdaLR initialization changes the optimizer LR, so construct it first
    # and restore optimizer state afterwards.
    optimizer=_optimizer(model,config); scheduler=_scheduler(optimizer,config); optimizer.load_state_dict(checked["training_state"]["optimizer_state"]); _move_optimizer(optimizer,target); scheduler.load_state_dict(checked["training_state"]["scheduler_state"]); sampler=StatefulBatchSampler(DATASET_SIZE,config.training.batch_size,seed=config.seed); sampler.load_state_dict(checked["training_state"]["sampler_state"])
    saved=checked["run_state"]; state=SmokeState(saved["global_step"],saved["samples_seen"],float("inf") if saved["best_metric"] is None else saved["best_metric"],saved["best_step"],saved["total_elapsed_seconds"])
    saved_device=json_load(run.path/"run_manifest.json").get("resolved_device")
    restore_rng(checked["training_state"]["rng_state"],np_gen,target.type)
    if saved_device!=target.type: run.events.append_event("operational_override",{"device":target.type,"saved_device":saved_device})
    if requested_device: run.events.append_event("operational_override",{"device":requested_device})
    run.events.append_event("resume_completed",{"snapshot":snapshot.name}); return run,config,np_gen,model,optimizer,scheduler,sampler,state,target
def json_load(path:Path)->dict[str,object]:
    import json; return json.loads(path.read_text())
def run_smoke(*,root:Path,config_path:Path|None=None,resume:Path|None=None,device:str|None=None,stop_after_steps:int|None=None)->Path:
    if (config_path is None)==(resume is None): raise ValueError("provide exactly one config or resume run")
    if resume: run,config,np_gen,model,optimizer,scheduler,sampler,state,target=_prepare_resume(resume,device)
    else: run,config,np_gen,model,optimizer,scheduler,sampler,state,target=_prepare_new(root,config_path) # type: ignore[arg-type]
    started=time.monotonic(); stop_at=stop_after_steps
    with StopController() as stopper:
        while state.step < config.scheduler.total_steps:
            indices=sampler.next_batch(); boards,policy,value=_synthetic(indices,target); model.train(); output=model(boards); loss=F.cross_entropy(output.policy_logits,policy)+F.cross_entropy(output.value_logits,value)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config.training.gradient_clip_norm); optimizer.step(); scheduler.step()
            state.step+=1; state.samples_seen+=len(indices); state.elapsed+=time.monotonic()-started; started=time.monotonic(); loss_value=float(loss.detach().cpu()); best=loss_value < state.best_loss
            if best: state.best_loss=loss_value; state.best_step=state.step
            # A corruption fallback can replay steps already present in the
            # append-only log; preserve that forensic history rather than
            # writing duplicate step records.
            if state.step > run.metrics.last_step:
                run.metrics.append_metric(state.step,sampler.epoch,{"loss":loss_value,"lr":float(optimizer.param_groups[0]["lr"])})
            stopping=stopper.requested.is_set() or (stop_at is not None and state.step>=stop_at)
            periodic=state.step%config.training.snapshot_every_steps==0
            completed=state.step==config.scheduler.total_steps
            if periodic or best or stopping or completed:
                reason="stop" if stopping else "completed" if completed else "best" if best else "periodic"; tags=({"stop"} if stopping else set()) | ({"periodic"} if periodic else set()) | ({"best"} if best else set()) | ({"completed"} if completed else set())
                _snapshot(run,model,optimizer,scheduler,sampler,np_gen,state,reason,stopper.reason or ("stop_after_steps" if stopping else None),tags)
            if stopping:
                run.events.append_event("stop_requested",{"reason":stopper.reason or "stop_after_steps"}); run.events.append_event("run_stopped",{"step":state.step}); return run.path
    run.events.append_event("run_completed",{"step":state.step}); return run.path

def fork_smoke(*,root: Path, snapshot_path: Path, config_path: Path, mode: str) -> Path:
    """Create an explicit independent fork; parent files are only read."""
    from chessy.snapshot.writer import verify_snapshot
    parent_path = Path(snapshot_path).resolve(); parent_run = Run.open(parent_path.parent.parent)
    checked = verify_snapshot(parent_path, expected_run_id=parent_run.id, expected_fingerprint=parent_run.fingerprint)
    config, source, resolved, fp = load_config(config_path)
    if config.model.model_dump() != parent_run.config.model.model_dump(): raise ValueError("fork model architecture/configuration is incompatible")
    parent = {"run_id":parent_run.id,"snapshot":parent_path.name,"snapshot_checksum":checked["checksum"],"mode":mode}
    if mode == "full-state":
        for field in ("seed","optimizer","scheduler","training"):
            if getattr(config,field) != getattr(parent_run.config,field): raise ValueError(f"full-state fork incompatible: {field}")
    run = Run.create(root,config,source,resolved,fp,parent=parent); target=resolve_device(config.device)
    if mode == "full-state":
        model=ChessyModel(config.model.to_model_config()).to(target); model.load_state_dict(checked["model_state"],strict=True); optimizer=_optimizer(model,config); optimizer.load_state_dict(checked["training_state"]["optimizer_state"]); _move_optimizer(optimizer,target); scheduler=_scheduler(optimizer,config); scheduler.load_state_dict(checked["training_state"]["scheduler_state"]); sampler=StatefulBatchSampler(DATASET_SIZE,config.training.batch_size,seed=config.seed); sampler.load_state_dict(checked["training_state"]["sampler_state"]); np_gen=np.random.default_rng(config.seed); restore_rng(checked["training_state"]["rng_state"],np_gen,target.type)
        old=checked["run_state"]; state=SmokeState(old["global_step"],old["samples_seen"],float("inf") if old["best_metric"] is None else old["best_metric"],old["best_step"],old["total_elapsed_seconds"])
        payload=checked["training_state"]
    elif mode == "weights-only":
        np_gen=_seed(config.seed); model=ChessyModel(config.model.to_model_config()).to(target); model.load_state_dict(checked["model_state"],strict=True); optimizer=_optimizer(model,config); scheduler=_scheduler(optimizer,config); sampler=StatefulBatchSampler(DATASET_SIZE,config.training.batch_size,seed=config.seed); state=SmokeState(); payload={"format":"chessy-training-state-v1","optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"sampler_state":sampler.state_dict(),"rng_state":capture_rng(np_gen),"gradient_scaler_state":None}
    else: raise ValueError("unknown fork mode")
    run_state=_run_state(run,state,sampler,model,"manual",None)
    if mode == "full-state":
        # State is complete, but its serializable payload is intentionally copied without model weights.
        payload={**payload, "format":"chessy-training-state-v1"}
    write_snapshot(run,model,payload,run_state,reason="manual",tags={"manual"})
    run.events.append_event("fork_created",parent); return run.path
