"""Command-line entry point for the loopback-only playable app."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import threading
import webbrowser
from pathlib import Path

import torch
import uvicorn

from chessy.api import ModelRuntime, SessionRegistry, create_app
from chessy.mcts import BatchingInferenceService
from chessy.model import ChessyModel, load_model_export, resolve_device
from chessy.play import ModelInfo

LOOPBACK_HOST = "127.0.0.1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessy", description="Chessy personal chess bot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    play = subparsers.add_parser("play", help="play locally against Chessy")
    play.add_argument("--model", action="append", type=Path, default=[], metavar="PATH", help="chessy-model-v1 export directory (repeatable)")
    play.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    play.add_argument("--port", type=int, default=0, help="loopback port; 0 asks the OS for a free port")
    play.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    play.add_argument("--feedback-dir", type=Path, default=Path("data/human_feedback"))
    play.add_argument("--simulations", type=int, help="expert override for all strength profiles")
    train = subparsers.add_parser("train", help="training commands")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    smoke = train_sub.add_parser("smoke", help="run the synthetic snapshot smoke trainer")
    smoke_group = smoke.add_mutually_exclusive_group(required=True)
    smoke_group.add_argument("--config", type=Path)
    smoke_group.add_argument("--resume", type=Path)
    smoke.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    smoke.add_argument("--stop-after-steps", type=int)
    run = subparsers.add_parser("run", help="inspect or fork local training runs")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    inspect = run_sub.add_parser("inspect", help="show local run health")
    inspect.add_argument("path", type=Path)
    fork = run_sub.add_parser("fork", help="make a new run from a snapshot")
    fork.add_argument("--snapshot", required=True, type=Path); fork.add_argument("--config", required=True, type=Path)
    fork.add_argument("--mode", required=True, choices=("full-state", "weights-only"))
    snapshot = subparsers.add_parser("snapshot", help="snapshot commands")
    snapshot_sub = snapshot.add_subparsers(dest="snapshot_command", required=True)
    verify = snapshot_sub.add_parser("verify", help="verify checksums and training state")
    verify.add_argument("path", type=Path)
    return parser


def _safe_model_id(name: str, checksum: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower() or "model"
    return f"{stem[:48]}-{checksum[:12]}"


def _export_runtime(path: Path, device: torch.device) -> tuple[ModelRuntime, BatchingInferenceService]:
    model = load_model_export(path, device=device)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    checksum = manifest["weights"]["sha256"]
    metadata = manifest.get("metadata", {})
    name = metadata.get("name", path.name) if isinstance(metadata, dict) else path.name
    if not isinstance(name, str):
        name = path.name
    info = ModelInfo(
        id=_safe_model_id(name, checksum),
        name=name[:100],
        checksum=checksum,
        architecture=manifest["architecture"],
    )
    service = BatchingInferenceService(model).start()
    return ModelRuntime(info, service), service


def _random_runtime(device: torch.device) -> tuple[ModelRuntime, BatchingInferenceService]:
    torch.manual_seed(0)
    model = ChessyModel().to(device).eval()
    info = ModelInfo(
        id="random-untrained-seed-0",
        name="Random untrained network (seed 0)",
        checksum="random-seed-0",
        untrained=True,
        random_seed=0,
    )
    service = BatchingInferenceService(model).start()
    return ModelRuntime(info, service), service


def _validate_feedback_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        if not path.is_dir() or not os.access(path, os.W_OK):
            raise ValueError("feedback directory is not writable")
        return path
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise ValueError("feedback directory parent is not writable")
    return path


def _run_play(args: argparse.Namespace) -> int:
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be in [0, 65535]")
    if args.simulations is not None and args.simulations <= 0:
        raise SystemExit("--simulations must be positive")
    device = resolve_device(args.device)
    static_dir = Path(__file__).parent / "api" / "static"
    if not (static_dir / "index.html").is_file() or not (static_dir / "assets").is_dir():
        raise SystemExit("Chessy frontend assets are missing; rebuild the package")
    feedback_dir = _validate_feedback_root(args.feedback_dir)
    services: list[BatchingInferenceService] = []
    runtimes: list[ModelRuntime] = []
    listener: socket.socket | None = None
    try:
        if args.model:
            for path in args.model:
                runtime, service = _export_runtime(path.expanduser().resolve(), device)
                runtimes.append(runtime)
                services.append(service)
        else:
            runtime, service = _random_runtime(device)
            runtimes.append(runtime)
            services.append(service)
        if len({runtime.info.id for runtime in runtimes}) != len(runtimes):
            raise SystemExit("model exports resolve to duplicate safe IDs")
        registry = SessionRegistry(
            runtimes,
            feedback_dir=feedback_dir,
            simulations_override=args.simulations,
        )
        app = create_app(registry, static_dir=static_dir)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((LOOPBACK_HOST, args.port))
        listener.listen(128)
        actual_port = listener.getsockname()[1]
        url = f"http://{LOOPBACK_HOST}:{actual_port}/"
        print(f"Chessy is ready at {url}", flush=True)
        if not args.no_open:
            threading.Timer(0.5, webbrowser.open, args=(url,)).start()
        config = uvicorn.Config(app, host=LOOPBACK_HOST, port=actual_port, log_level="info")
        try:
            uvicorn.Server(config).run(sockets=[listener])
        except KeyboardInterrupt:
            # Uvicorn has already completed its lifespan shutdown at this point.
            pass
        return 0
    finally:
        if listener is not None:
            listener.close()
        for service in services:
            service.close()


def _project_root() -> Path:
    return Path.cwd()


def _run_smoke_command(args: argparse.Namespace) -> int:
    from chessy.training.smoke import run_smoke
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise SystemExit("--stop-after-steps must be positive")
    path = run_smoke(root=_project_root(), config_path=args.config, resume=args.resume, device=args.device, stop_after_steps=args.stop_after_steps)
    print(path)
    return 0


def _inspect_run(path: Path) -> int:
    from chessy.run import Run
    from chessy.snapshot.writer import verify_snapshot
    run = Run.open(path); snapshots = run.path / "snapshots"; index_path = snapshots / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    status=[]
    for item in sorted(snapshots.glob("step-*")):
        try: verify_snapshot(item, expected_run_id=run.id, expected_fingerprint=run.fingerprint); state="valid"
        except ValueError as exc: state=f"corrupt ({exc})"
        status.append({"name":item.name,"status":state})
    size=sum(p.stat().st_size for p in run.path.rglob("*") if p.is_file() and not p.is_symlink())
    print(json.dumps({"run_id":run.id,"config_fingerprint":run.fingerprint,"latest":index.get("latest"),"best":index.get("best"),"stages":index.get("stages"),"snapshots":status,"last_metric_step":run.metrics.last_step,"disk_bytes":size,"parent":json.loads((run.path/"run_manifest.json").read_text()).get("parent")},indent=2,ensure_ascii=False))
    return 0


def _verify_snapshot(path: Path) -> int:
    from chessy.snapshot.writer import verify_snapshot
    try:
        checked = verify_snapshot(path)
    except (OSError, ValueError) as exc:
        print(f"invalid snapshot: {exc}")
        return 1
    print(f"valid {path} step={checked['run_state']['global_step']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "play":
        return _run_play(args)
    if args.command == "train" and args.train_command == "smoke":
        return _run_smoke_command(args)
    if args.command == "run" and args.run_command == "inspect":
        return _inspect_run(args.path)
    if args.command == "run" and args.run_command == "fork":
        from chessy.training.smoke import fork_smoke
        print(fork_smoke(root=_project_root(), snapshot_path=args.snapshot, config_path=args.config, mode=args.mode)); return 0
    if args.command == "snapshot" and args.snapshot_command == "verify":
        return _verify_snapshot(args.path)
    raise SystemExit(2)
