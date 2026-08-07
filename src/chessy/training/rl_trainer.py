"""Local generation-based self-play RL trainer with resumable phase state."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chessy.config import load_config
from chessy.config.schema import ChessyConfig
from chessy.curriculum import CurriculumManager, CurriculumState
from chessy.evaluation import MCTSAgent, create_league, run_arena
from chessy.mcts import BatchingInferenceService, DirectModelEvaluator, MCTSConfig
from chessy.model import ChessyModel, export_model, load_model_export, resolve_device
from chessy.observer import TrainingObserver
from chessy.replay import ReplayDataset, ReplaySampler, load_manifest, write_manifest, write_segment
from chessy.run import Run
from chessy.selfplay import SelfPlayCoordinator, TemperatureSchedule
from chessy.snapshot import capture_rng, restore_rng, write_snapshot
from chessy.snapshot.loader import select_snapshot
from chessy.training.rl_loss import policy_value_loss
from chessy.training.rl_state import RLState
from chessy.training.stop import StopController


class _Progress:
    def __init__(self) -> None: self.started=time.monotonic()
    def write(self, phase:str, message:str) -> None:
        elapsed=int(time.monotonic()-self.started); hours,remainder=divmod(elapsed,3600); minutes,seconds=divmod(remainder,60)
        print(f"[chessy {hours:02d}:{minutes:02d}:{seconds:02d}] {phase}: {message}",flush=True)


def _seed(seed: int) -> np.random.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _optimizer(model: ChessyModel, config: ChessyConfig) -> torch.optim.Optimizer:
    c=config.optimizer
    return torch.optim.AdamW(model.parameters(),lr=c.learning_rate,weight_decay=c.weight_decay,betas=(c.beta1,c.beta2),eps=c.epsilon)


def _scheduler(optimizer: torch.optim.Optimizer, config: ChessyConfig) -> torch.optim.lr_scheduler.LambdaLR:
    warmup,total,minimum=config.scheduler.warmup_steps,config.scheduler.total_steps,config.scheduler.minimum_lr_ratio
    def factor(step: int) -> float:
        if step < warmup: return (step+1)/max(1,warmup)
        ratio=min(1.0,max(0.0,(step-warmup)/max(1,total-warmup)))
        return minimum+(1-minimum)*.5*(1+math.cos(math.pi*ratio))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for group in optimizer.state.values():
        for key,value in group.items():
            if isinstance(value,torch.Tensor): group[key]=value.to(device)


def _reference(root: Path, kind: str, relative: str | None) -> dict[str,object]:
    if relative is None:
        return {"format":"chessy-reference-v1","kind":kind,"source":None,"source_sha256":None,"content":None}
    pure=Path(relative)
    if pure.is_absolute() or ".." in pure.parts: raise ValueError(f"unsafe {kind} manifest path")
    unresolved=root/pure
    if unresolved.is_symlink(): raise ValueError(f"unsafe {kind} manifest symlink")
    path=unresolved.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError(f"missing {kind} manifest")
    raw=path.read_bytes(); content=json.loads(raw)
    return {"format":"chessy-reference-v1","kind":kind,"source":pure.as_posix(),"source_sha256":hashlib.sha256(raw).hexdigest(),"content":content}


def _snapshot(run:Run,model:ChessyModel,optimizer:torch.optim.Optimizer,scheduler:object,sampler_state:dict[str,object],npgen:np.random.Generator,state:RLState,reason:str,stop_reason:str|None=None)->Path:
    payload={"format":"chessy-training-state-v1","optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"sampler_state":sampler_state,"rng_state":capture_rng(npgen),"gradient_scaler_state":None,"rl_state":state.state_dict()}
    run_state={"format":"chessy-run-state-v1","run_id":run.id,"config_fingerprint":run.fingerprint,"global_step":state.global_step,"epoch":int(sampler_state.get("draws",0)),"samples_seen":state.samples_seen,"stage":"rl-"+state.phase,"best_metric":None if math.isinf(state.best_loss) else state.best_loss,"best_step":None,"total_elapsed_seconds":0.,"last_completed_batch":state.global_step,"snapshot_reason":reason,"stop_reason":stop_reason,"created_at":__import__("chessy.run.logging",fromlist=["utcnow"]).utcnow(),"model_parameter_count":sum(p.numel() for p in model.parameters()),"optimizer":"adamw","scheduler":"warmup-cosine","generation":state.generation,"phase":state.phase,"replay_manifest":state.replay_manifest_path,"league_manifest":state.league_manifest_path}
    root=run.path.parent.parent
    references={"dataset":_reference(root,"dataset",run.config.artifacts.dataset_manifest),"replay":_reference(root,"replay",state.replay_manifest_path),"league":_reference(root,"league",state.league_manifest_path)}
    return write_snapshot(run,model,payload,run_state,reason=reason,tags={reason},references=references)


def _prepare(root:Path,config_path:Path|None,resume:Path|None,device:str|None):
    if (config_path is None)==(resume is None): raise ValueError("provide exactly one config or resume")
    if config_path is not None:
        config,source,resolved,fp=load_config(config_path)
        if not all((config.self_play,config.replay,config.rl,config.curriculum,config.evaluation)): raise ValueError("config is missing RL sections")
        run=Run.create(root,config,source,resolved,fp); npgen=_seed(config.seed); target=resolve_device(device or config.device)
        model=ChessyModel(config.model.to_model_config()).to(target); optimizer=_optimizer(model,config)
        state=RLState(curriculum_state=CurriculumState(stage=config.curriculum.initial_stage,stage_mode=config.curriculum.stage_mode,stage_mix=config.curriculum.stage_mix.model_dump()).state_dict())
        return run,config,npgen,model,optimizer,_scheduler(optimizer,config),None,state,target
    run=Run.open(resume); config=run.config
    if not all((config.self_play,config.replay,config.rl,config.curriculum,config.evaluation)): raise ValueError("resume run is not an RL run")
    _,checked,rejected=select_snapshot(run)
    if rejected: run.events.append_event("resume_fallback",{"rejected":rejected})
    target=resolve_device(device or config.device); npgen=np.random.default_rng(config.seed)
    model=ChessyModel(config.model.to_model_config()).to(target); model.load_state_dict(checked["model_state"],strict=True)
    optimizer=_optimizer(model,config); scheduler=_scheduler(optimizer,config)
    optimizer.load_state_dict(checked["training_state"]["optimizer_state"]); _move_optimizer(optimizer,target); scheduler.load_state_dict(checked["training_state"]["scheduler_state"])
    restore_rng(checked["training_state"]["rng_state"],npgen,target.type)
    state=RLState.from_dict(checked["training_state"].get("rl_state",{}))
    root=root.resolve()
    if state.replay_manifest_path:
        replay_manifest=load_manifest(root/state.replay_manifest_path)
        if replay_manifest.fingerprint!=state.replay_manifest_fingerprint: raise ValueError("resume replay manifest fingerprint mismatch")
    if state.league_manifest_path:
        league=_reference(root,"league",state.league_manifest_path)["content"]
        if not isinstance(league,dict) or league.get("fingerprint")!=state.league_manifest_fingerprint: raise ValueError("resume league manifest fingerprint mismatch")
    run.events.append_event("resume_completed",{"phase":state.phase,"step":state.global_step})
    return run,config,npgen,model,optimizer,scheduler,checked["training_state"].get("sampler_state"),state,target


def _export_checksum(path: Path) -> str:
    return str(json.loads((path/"manifest.json").read_text())["weights"]["sha256"])


def _segment_chunks(games:list[Any],limit:int)->list[list[Any]]:
    chunks=[]; current=[]; samples=0
    for game in games:
        size=len(game.sealed.samples)
        if current and samples+size>limit: chunks.append(current); current=[]; samples=0
        current.append(game); samples+=size
    if current: chunks.append(current)
    return chunks


def run_rl(*,root:Path,config_path:Path|None=None,resume:Path|None=None,device:str|None=None,stop_after_steps:int|None=None)->Path:
    root=root.resolve(); run,config,npgen,model,optimizer,scheduler,saved_sampler,state,target=_prepare(root,config_path,resume,device)
    progress=_Progress(); progress.write("run",f"id={run.id} device={target.type} generation={state.generation} step={state.global_step}/{config.scheduler.total_steps} phase={state.phase}")
    sp,replay,rl,cur,evaluation=config.self_play,config.replay,config.rl,config.curriculum,config.evaluation
    assert sp and replay and rl and cur and evaluation
    if state.phase=="complete": progress.write("run",f"already complete at step={state.global_step}"); return run.path
    if state.global_step>=config.scheduler.total_steps:
        run.events.append_event("run_completed",{"generation":state.generation,"step":state.global_step,"recovered_from_final_stop":True})
        return run.path
    exports=run.path/"exports"; league_dir=run.path/"league"; league_dir.mkdir(exist_ok=True)
    curriculum=CurriculumManager(CurriculumState(**{k:v for k,v in state.curriculum_state.items() if k!="format"}),max_plies=sp.max_game_plies,max_material_imbalance=cur.reduced_max_material_imbalance)
    observer_config=config.observer
    observer=TrainingObserver(run.path,enabled=bool(observer_config and observer_config.enabled),archive_every_generations=1 if observer_config is None else observer_config.archive_every_generations,live_game_index=0 if observer_config is None else observer_config.live_game_index)
    sampler:ReplaySampler|None=None
    with StopController() as stopper:
        while state.global_step < config.scheduler.total_steps:
            incumbent=exports/f"generation-{state.generation:04d}"
            if not incumbent.exists(): export_model(model,incumbent,metadata={"generation":str(state.generation),"name":config.name})
            incumbent_checksum=_export_checksum(incumbent)
            if state.league_manifest_path is None:
                league=create_league(league_dir/f"league-{state.generation:04d}-initial.json",incumbent=state.generation,export_path=str(incumbent.relative_to(run.path)),export_checksum=incumbent_checksum,stage=curriculum.state.stage)
                state.league_manifest_path=str(league.path.relative_to(root)); state.league_manifest_fingerprint=str(league.content["fingerprint"])

            if state.phase in {"initialize","selfplay"}:
                state.phase="selfplay"; run.events.append_event("selfplay_started",{"generation":state.generation})
                progress.write("self-play",f"generation={state.generation} games={sp.games_per_generation} actors={sp.actors} simulations={sp.simulations}")
                if replay.hard_disk_limit_bytes is not None and (root/replay.root_dir).exists():
                    used=sum(p.stat().st_size for p in (root/replay.root_dir).rglob("*") if p.is_file() and not p.is_symlink())
                    if used>=replay.hard_disk_limit_bytes: raise ValueError("replay hard disk limit reached")
                with BatchingInferenceService(model,max_batch_size=sp.inference_batch_size,max_batch_wait_ms=sp.inference_wait_ms) as service:
                    def selfplay_progress(done:int,total:int,item:Any)->None:
                        sealed=item.sealed; progress.write("self-play",f"{done}/{total} game={sealed.game_index} plies={len(sealed.samples)} result={sealed.result} termination={sealed.termination} duration={item.duration_seconds:.1f}s")
                    coordinator=SelfPlayCoordinator(run.id,config.seed,state.generation,sp.actors,service,curriculum,MCTSConfig(simulations=sp.simulations,c_puct=sp.c_puct,root_noise=True,dirichlet_alpha=sp.dirichlet_alpha,dirichlet_epsilon=sp.dirichlet_epsilon,max_batch_size=sp.inference_batch_size,max_batch_wait_ms=sp.inference_wait_ms),TemperatureSchedule(sp.temperature.initial,sp.temperature.cutoff_ply,sp.temperature.final),incumbent_checksum,observer,selfplay_progress)
                    games,incomplete=coordinator.run(games=sp.games_per_generation,stop_requested=stopper.requested)
                if incomplete or stopper.requested.is_set():
                    run.events.append_event("selfplay_stopped",{"completed_discarded":len(games),"incomplete":incomplete})
                    snapshot=_snapshot(run,model,optimizer,scheduler,{"format":"chessy-replay-sampler-pending-v1","draws":0},npgen,state,"stop",stopper.reason); progress.write("snapshot",f"saved={snapshot} reason={stopper.reason or 'stop_during_selfplay'}")
                    run.events.append_event("run_stopped",{"phase":state.phase}); progress.write("run",f"stopped safely during self-play; resume with --resume {run.path}"); return run.path
                old_segments=[]
                if state.replay_manifest_path:
                    old=load_manifest(root/state.replay_manifest_path); old_segments=[old.path.parent.parent/entry["path"] for entry in old.content["segments"]]
                new_segments=[]
                for ordinal,chunk in enumerate(_segment_chunks(games,replay.samples_per_segment)):
                    new_segments.append(write_segment(root/replay.root_dir,generation=state.generation,ordinal=ordinal,games=[g.sealed for g in chunk],run_id=run.id,model_checksum=incumbent_checksum))
                manifest=write_manifest(root/replay.root_dir,run_id=run.id,generation=state.generation,segments=old_segments+new_segments,active_max_samples=replay.active_max_samples,policy={"recent_fraction":replay.recent_fraction,"recent_generations":replay.recent_generations})
                state.replay_manifest_path=str(manifest.path.relative_to(root)); state.replay_manifest_fingerprint=manifest.fingerprint; state.completed_game_indexes=[g.sealed.game_index for g in games]; state.phase="train"; state.training_block_end_step=min(state.global_step+rl.train_steps_per_generation,config.scheduler.total_steps)
                run.events.append_event("selfplay_completed",{"generation":state.generation,"games":len(games)}); run.events.append_event("replay_manifest_updated",{"manifest":state.replay_manifest_path})
                progress.write("self-play",f"complete generation={state.generation}; replay={state.replay_manifest_path}")
                saved_sampler=None

            if sampler is None:
                manifest=load_manifest(root/state.replay_manifest_path) # type: ignore[arg-type]
                dataset=ReplayDataset(manifest,cache_segments=replay.cache_segments)
                sampler=ReplaySampler(dataset,rl.batch_size,config.seed+state.generation,replay.recent_fraction,replay.recent_generations)
                if saved_sampler and saved_sampler.get("format")=="chessy-replay-sampler-v1": sampler.load_state_dict(saved_sampler)
                saved_sampler=None
            else:
                dataset=sampler.dataset
            if state.training_block_end_step<=state.global_step: state.training_block_end_step=min(state.global_step+rl.train_steps_per_generation,config.scheduler.total_steps)
            state.phase="train"; run.events.append_event("training_started",{"generation":state.generation,"until_step":state.training_block_end_step})
            visible_end=min(state.training_block_end_step,stop_after_steps) if stop_after_steps is not None else state.training_block_end_step; report_every=max(1,min(25,max(1,visible_end-state.global_step)//10))
            progress.write("training",f"steps={state.global_step + 1}..{visible_end} generation={state.generation}")
            while state.global_step < state.training_block_end_step:
                batch=dataset.batch(sampler.next_batch()); boards=batch["boards"].to(target); policy=batch["policy"].to(target); mask=batch["legal_mask"].to(target); value=batch["value_class"].to(target)
                model.train(); output=model(boards); loss,metrics=policy_value_loss(output.policy_logits,output.value_logits,policy,mask,value,policy_weight=rl.policy_loss_weight,value_weight=rl.value_loss_weight)
                optimizer.zero_grad(set_to_none=True); loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),rl.gradient_clip_norm))
                if not math.isfinite(grad): raise ValueError("non-finite gradient")
                optimizer.step(); scheduler.step(); state.global_step+=1; state.samples_seen+=len(boards); loss_value=float(loss.detach().cpu()); state.best_loss=min(state.best_loss,loss_value)
                run.metrics.append_metric(state.global_step,sampler.draws,{"loss":loss_value,"policy_loss":float(metrics["policy_loss"].detach().cpu()),"value_loss":float(metrics["value_loss"].detach().cpu()),"policy_entropy":float(metrics["policy_entropy"].detach().cpu()),"gradient_norm":grad,"lr":float(optimizer.param_groups[0]["lr"]),"top1_agreement":float(metrics["top1_agreement"].detach().cpu()),"value_accuracy":float(metrics["value_accuracy"].detach().cpu())})
                if state.global_step%report_every==0 or state.global_step==visible_end: progress.write("training",f"step={state.global_step}/{visible_end} loss={loss_value:.4f} policy={float(metrics['policy_loss'].detach().cpu()):.4f} value={float(metrics['value_loss'].detach().cpu()):.4f} lr={float(optimizer.param_groups[0]['lr']):.6g}")
                stopping=stopper.requested.is_set() or (stop_after_steps is not None and state.global_step>=stop_after_steps)
                if stopping or (state.global_step%config.training.snapshot_every_steps==0 and state.global_step<config.scheduler.total_steps):
                    snapshot=_snapshot(run,model,optimizer,scheduler,sampler.state_dict(),npgen,state,"stop" if stopping else "periodic",stopper.reason or ("stop_after_steps" if stopping else None)); progress.write("snapshot",f"saved={snapshot} reason={'stop' if stopping else 'periodic'}")
                if stopping:
                    run.events.append_event("run_stopped",{"phase":state.phase,"step":state.global_step}); progress.write("run",f"stopped safely at step={state.global_step}; resume with --resume {run.path}"); return run.path

            state.phase="arena"; candidate=exports/f"candidate-{state.generation:04d}-step-{state.global_step:012d}"
            if not candidate.exists(): export_model(model,candidate,metadata={"generation":str(state.generation),"step":str(state.global_step),"name":config.name})
            candidate_checksum=_export_checksum(candidate)
            opponent_model=load_model_export(incumbent,device=target)
            positions=[curriculum.sample(np.random.default_rng(config.seed+state.generation*1000+i)) for i in range(evaluation.games_per_match//2)]
            progress.write("arena",f"generation={state.generation} games={evaluation.games_per_match} simulations={evaluation.simulations}")
            def arena_progress(done:int,total:int,value:float,reason:str)->None: progress.write("arena",f"{done}/{total} candidate_score={value:.1f} termination={reason}")
            report=run_arena(candidate=MCTSAgent(DirectModelEvaluator(model),evaluation.simulations),opponent=MCTSAgent(DirectModelEvaluator(opponent_model),evaluation.simulations),positions=positions,games=evaluation.games_per_match,max_plies=sp.max_game_plies,candidate_checksum=candidate_checksum,opponent_checksum=incumbent_checksum,config_fingerprint=run.fingerprint,promotion_min_games=evaluation.promotion_min_games,promotion_min_score=evaluation.promotion_min_score,confidence_threshold=evaluation.require_lower_confidence_above,seed=config.seed+state.global_step,progress_update=arena_progress)
            report_path=run.path/"arena"/f"arena-{state.generation:04d}-step-{state.global_step:012d}-{report.fingerprint[:12]}.json"
            from chessy.evaluation.arena import write_report
            write_report(report_path,report); run.events.append_event("arena_completed",{"report":str(report_path),"score":report.score,"promoted":report.promoted})
            progress.write("arena",f"complete score={report.score:.3f} ci={report.confidence_interval[0]:.3f}..{report.confidence_interval[1]:.3f} promoted={report.promoted}")
            old_league=json.loads((root/state.league_manifest_path).read_text()) # type: ignore[arg-type]
            history=list(old_league.get("history",[])); history.append({"candidate_generation":state.generation+1,"candidate_export":str(candidate.relative_to(run.path)),"candidate_checksum":candidate_checksum,"arena_report":str(report_path.relative_to(run.path)),"promoted":report.promoted})
            active_export=candidate if report.promoted else incumbent; active_checksum=candidate_checksum if report.promoted else incumbent_checksum; active_generation=state.generation+1 if report.promoted else state.generation
            league=create_league(league_dir/f"league-{state.generation:04d}-step-{state.global_step:012d}.json",incumbent=active_generation,export_path=str(active_export.relative_to(run.path)),export_checksum=active_checksum,history=history,stage=curriculum.state.stage,tags=["promoted"] if report.promoted else ["candidate"])
            state.league_manifest_path=str(league.path.relative_to(root)); state.league_manifest_fingerprint=str(league.content["fingerprint"])
            run.events.append_event("generation_promoted" if report.promoted else "generation_rejected",{"generation":state.generation,"step":state.global_step})
            if report.promoted:
                state.generation+=1; state.completed_game_indexes=[]; state.phase="selfplay"; state.training_block_end_step=0; sampler=None
            else:
                state.phase="train"; state.training_block_end_step=0

        state.phase="complete"
        assert sampler is not None
        _snapshot(run,model,optimizer,scheduler,sampler.state_dict(),npgen,state,"completed")
        run.events.append_event("run_completed",{"generation":state.generation,"step":state.global_step}); progress.write("run",f"complete generation={state.generation} step={state.global_step}")
        return run.path
