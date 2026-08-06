"""Command-line entry point for the loopback-only playable app."""

from __future__ import annotations

import argparse
import hashlib
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
    rl = train_sub.add_parser("rl", help="run one local self-play RL generation")
    rl_group = rl.add_mutually_exclusive_group(required=True)
    rl_group.add_argument("--config", type=Path)
    rl_group.add_argument("--resume", type=Path)
    rl.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    rl.add_argument("--stop-after-steps", type=int)
    selfplay = subparsers.add_parser("selfplay", help="self-play convenience commands")
    selfplay_sub = selfplay.add_subparsers(dest="selfplay_command", required=True)
    selfplay_smoke = selfplay_sub.add_parser("smoke", help="run the tiny end-to-end self-play smoke generation")
    selfplay_smoke.add_argument("--config", required=True, type=Path)
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
    replay = subparsers.add_parser("replay", help="inspect or verify immutable replay")
    replay_sub = replay.add_subparsers(dest="replay_command", required=True)
    for name in ("inspect", "verify"):
        command = replay_sub.add_parser(name, help=f"{name} a replay manifest")
        command.add_argument("path", type=Path)
    arena = subparsers.add_parser("arena", help="run a small deterministic arena")
    arena_sub = arena.add_subparsers(dest="arena_command", required=True)
    arena_run = arena_sub.add_parser("run", help="compare an export against a baseline")
    arena_run.add_argument("--candidate", required=True, type=Path)
    arena_run.add_argument("--opponent", required=True, choices=("random", "material"))
    arena_run.add_argument("--games", type=int, default=4)
    dataset = subparsers.add_parser("dataset", help="immutable dataset commands")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    personal = dataset_sub.add_parser("personal", help="build and inspect personal historical data")
    personal_sub = personal.add_subparsers(dest="personal_command", required=True)
    build = personal_sub.add_parser("build", help="encode split PGN positions with full history")
    build.add_argument("--splits", type=Path, default=Path("data/personal/splits/manifest.json"))
    build.add_argument("--chess-com-pgn", type=Path, default=Path("data/raw/chess_com_mu1876.pgn"))
    build.add_argument("--lichess-pgn", type=Path, default=Path("data/raw/lichess_mu1878.pgn"))
    build.add_argument("--game-quality", type=Path, default=Path("data/quality/game_quality.csv"))
    build.add_argument("--output", type=Path, default=Path("data/personal/encoded"))
    build.add_argument("--segment-samples", type=int, default=16384)
    personal_sub.add_parser("prepare-smoke", help="generate the ignored tiny dataset and fixture model")
    for name in ("inspect", "verify"):
        command = personal_sub.add_parser(name, help=f"{name} a personal dataset manifest")
        command.add_argument("--manifest", required=True, type=Path)
    feedback = subparsers.add_parser("feedback", help="inspect, verify, and encode confirmed human feedback")
    feedback_sub = feedback.add_subparsers(dest="feedback_command", required=True)
    feedback_sub.add_parser("prepare-smoke", help="generate ignored tiny feedback fixtures")
    feedback_inspect = feedback_sub.add_parser("inspect", help="inspect raw confirmed games without writing files")
    feedback_inspect.add_argument("--input", type=Path, default=Path("data/human_feedback"))
    feedback_verify = feedback_sub.add_parser("verify", help="verify one raw confirmed game")
    feedback_verify.add_argument("--game", required=True, type=Path)
    feedback_build = feedback_sub.add_parser("build", help="build immutable encoded feedback dataset")
    feedback_build.add_argument("--input", type=Path, default=Path("data/human_feedback")); feedback_build.add_argument("--output", type=Path, default=Path("data/human_feedback_encoded")); feedback_build.add_argument("--sample-weight", type=float, default=4.0); feedback_build.add_argument("--max-positions-per-game", type=int, default=16); feedback_build.add_argument("--segment-samples", type=int, default=16384)
    feedback_dataset_verify = feedback_sub.add_parser("dataset-verify", help="verify an encoded feedback manifest")
    feedback_dataset_verify.add_argument("--manifest", required=True, type=Path)
    personalize = subparsers.add_parser("personalize", help="supervised personal fine-tuning")
    personalize_sub = personalize.add_subparsers(dest="personalize_command", required=True)
    personal_train = personalize_sub.add_parser("train", help="fine-tune an explicit base_rl export")
    personal_group = personal_train.add_mutually_exclusive_group(required=True)
    personal_group.add_argument("--config", type=Path); personal_group.add_argument("--resume", type=Path)
    personal_train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda")); personal_train.add_argument("--stop-after-steps", type=int)
    personal_feedback = personalize_sub.add_parser("feedback", help="fine-tune a personal model with explicit human feedback")
    feedback_group = personal_feedback.add_mutually_exclusive_group(required=True)
    feedback_group.add_argument("--config", type=Path); feedback_group.add_argument("--resume", type=Path)
    personal_feedback.add_argument("--device", choices=("auto", "cpu", "mps", "cuda")); personal_feedback.add_argument("--stop-after-steps", type=int)
    personal_validate = personalize_sub.add_parser("validate", help="validate only the frozen val split")
    personal_validate.add_argument("--model", required=True, type=Path); personal_validate.add_argument("--dataset", required=True, type=Path); personal_validate.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    personal_compare = personalize_sub.add_parser("compare", help="compare base and personal models on validation")
    personal_compare.add_argument("--base", required=True, type=Path); personal_compare.add_argument("--personal", required=True, type=Path); personal_compare.add_argument("--dataset", required=True, type=Path)
    personal_rl = subparsers.add_parser("personal-rl", help="personal RL with a mandatory style-retention gate")
    personal_rl_sub = personal_rl.add_subparsers(dest="personal_rl_command", required=True)
    personal_rl_sub.add_parser("prepare-smoke", help="create ignored tiny fixtures for the plumbing smoke")
    personal_rl_train = personal_rl_sub.add_parser("train", help="start a personal-RL run from an explicit incumbent")
    personal_rl_train.add_argument("--config", required=True, type=Path); personal_rl_train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda")); personal_rl_train.add_argument("--stop-after-steps", type=int)
    personal_rl_resume = personal_rl_sub.add_parser("resume", help="resume a verified personal-RL snapshot")
    personal_rl_resume.add_argument("--run", required=True, type=Path); personal_rl_resume.add_argument("--device", choices=("auto", "cpu", "mps", "cuda")); personal_rl_resume.add_argument("--stop-after-steps", type=int)
    personal_rl_evaluate = personal_rl_sub.add_parser("evaluate", help="show persisted gate and validation reports")
    personal_rl_evaluate.add_argument("--run", required=True, type=Path)
    personal_rl_inspect = personal_rl_sub.add_parser("inspect", help="show personal-RL run and pinned-input health")
    personal_rl_inspect.add_argument("--run", required=True, type=Path)
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
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous(); digest.update(name.encode()); digest.update(value.numpy().tobytes())
    info = ModelInfo(
        id="random-untrained-seed-0",
        name="Random untrained network (seed 0)",
        checksum=digest.hexdigest(),
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
            observer_runs_dir=_project_root() / "runs",
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


def _run_rl_command(args: argparse.Namespace) -> int:
    from chessy.training.rl_trainer import run_rl
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise SystemExit("--stop-after-steps must be positive")
    print(run_rl(root=_project_root(), config_path=args.config, resume=args.resume, device=args.device, stop_after_steps=args.stop_after_steps))
    return 0


def _replay_command(args: argparse.Namespace) -> int:
    from chessy.replay import load_manifest
    try:
        manifest = load_manifest(args.path, verify=args.replay_command == "verify")
    except (OSError, ValueError) as exc:
        print(f"invalid replay manifest: {exc}")
        return 1
    print(json.dumps(manifest.content, indent=2, ensure_ascii=False))
    return 0


def _arena_command(args: argparse.Namespace) -> int:
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    from chessy.curriculum.sources import FullSource
    from chessy.evaluation import MCTSAgent, MaterialAgent, RandomAgent, run_arena
    from chessy.mcts import DirectModelEvaluator
    model = load_model_export(args.candidate, device="cpu")
    candidate_checksum = json.loads((args.candidate / "manifest.json").read_text(encoding="utf-8"))["weights"]["sha256"]
    opponent = RandomAgent(0) if args.opponent == "random" else MaterialAgent()
    report = run_arena(candidate=MCTSAgent(DirectModelEvaluator(model), 4), opponent=opponent, positions=[FullSource().sample(__import__("numpy").random.default_rng(0))], games=args.games, max_plies=160, candidate_checksum=candidate_checksum, opponent_checksum=args.opponent, promotion_min_games=40)
    print(json.dumps(__import__("dataclasses").asdict(report), indent=2))
    return 0


def _personal_dataset_command(args: argparse.Namespace) -> int:
    from chessy.personal.builder import build_personal_dataset
    from chessy.personal.segment import load_personal_manifest, verify_personal_manifest
    if args.personal_command == "prepare-smoke":
        from chessy.personal.fixture import prepare_smoke_fixture
        print(json.dumps(prepare_smoke_fixture(_project_root()), indent=2)); return 0
    if args.personal_command == "build":
        if args.segment_samples <= 0:
            raise SystemExit("--segment-samples must be positive")
        path = build_personal_dataset(splits=args.splits, chess_com_pgn=args.chess_com_pgn, lichess_pgn=args.lichess_pgn, game_quality=args.game_quality, output=args.output, segment_samples=args.segment_samples)
        print(path.resolve()); return 0
    try:
        payload = verify_personal_manifest(args.manifest) if args.personal_command == "verify" else load_personal_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        print(f"invalid personal dataset: {exc}"); return 1
    print(json.dumps(payload, indent=2, ensure_ascii=False)); return 0


def _feedback_command(args: argparse.Namespace) -> int:
    from chessy.feedback import build_feedback_dataset, inspect_feedback_root, verify_feedback_game
    from chessy.feedback.segment import verify_feedback_manifest
    try:
        if args.feedback_command == "prepare-smoke":
            from chessy.feedback.fixture import prepare_feedback_smoke_fixture
            print(json.dumps(prepare_feedback_smoke_fixture(_project_root()), indent=2)); return 0
        if args.feedback_command == "inspect": payload = inspect_feedback_root(args.input)
        elif args.feedback_command == "verify": payload = verify_feedback_game(args.game)["manifest"]
        elif args.feedback_command == "dataset-verify": payload = verify_feedback_manifest(args.manifest)
        else:
            print(build_feedback_dataset(input=args.input, output=args.output, sample_weight=args.sample_weight, max_positions_per_game=args.max_positions_per_game, segment_samples=args.segment_samples).resolve()); return 0
    except (OSError, ValueError) as exc:
        print(f"invalid feedback artifact: {exc}"); return 1
    print(json.dumps(payload, indent=2, ensure_ascii=False)); return 0


def _personalize_command(args: argparse.Namespace) -> int:
    from chessy.personal.dataset import PersonalDataset
    from chessy.personal.validation import validate
    from chessy.training.personal_trainer import run_personal_training
    if args.personalize_command in {"train", "feedback"}:
        if args.stop_after_steps is not None and args.stop_after_steps <= 0:
            raise SystemExit("--stop-after-steps must be positive")
        if args.personalize_command == "feedback":
            from chessy.training.feedback_trainer import run_feedback_training
            print(run_feedback_training(root=_project_root(), config_path=args.config, resume=args.resume, device=args.device, stop_after_steps=args.stop_after_steps)); return 0
        print(run_personal_training(root=_project_root(), config_path=args.config, resume=args.resume, device=args.device, stop_after_steps=args.stop_after_steps)); return 0
    dataset = PersonalDataset(args.dataset, split="val")
    if args.personalize_command == "validate":
        model = load_model_export(args.model, device=args.device)
        print(json.dumps(validate(model, dataset, device=resolve_device(args.device), batch_size=512, model_checksum=json.loads((args.model / "manifest.json").read_text())["weights"]["sha256"]), indent=2)); return 0
    base = load_model_export(args.base, device="cpu"); personal = load_model_export(args.personal, device="cpu")
    base_report = validate(base, dataset, device=torch.device("cpu"), batch_size=512)
    personal_report = validate(personal, dataset, device=torch.device("cpu"), batch_size=512, baseline=base_report)
    print(json.dumps({"base": base_report, "personal": personal_report}, indent=2)); return 0


def _personal_rl_command(args: argparse.Namespace) -> int:
    if args.personal_rl_command == "prepare-smoke":
        from chessy.personal_rl.fixture import prepare_personal_rl_smoke_fixture
        print(json.dumps(prepare_personal_rl_smoke_fixture(_project_root()), ensure_ascii=False, indent=2)); return 0
    if args.personal_rl_command in {"train", "resume"}:
        if args.stop_after_steps is not None and args.stop_after_steps <= 0: raise SystemExit("--stop-after-steps must be positive")
        from chessy.training.personal_rl_trainer import run_personal_rl
        print(run_personal_rl(root=_project_root(), config_path=getattr(args, "config", None), resume=getattr(args, "run", None), device=args.device, stop_after_steps=args.stop_after_steps)); return 0
    run = args.run.resolve()
    from chessy.training.personal_rl_trainer import inspect_personal_rl_run, personal_rl_evaluation_summary
    result = inspect_personal_rl_run(root=_project_root(), run_path=run) if args.personal_rl_command == "inspect" else personal_rl_evaluation_summary(run)
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "play":
        return _run_play(args)
    if args.command == "train" and args.train_command == "smoke":
        return _run_smoke_command(args)
    if args.command == "train" and args.train_command == "rl":
        return _run_rl_command(args)
    if args.command == "selfplay" and args.selfplay_command == "smoke":
        # The smoke command intentionally completes the generation so its
        # replay segment is trained and then checked by arena plumbing.
        args.resume = None
        args.device = None
        args.stop_after_steps = None
        return _run_rl_command(args)
    if args.command == "run" and args.run_command == "inspect":
        return _inspect_run(args.path)
    if args.command == "run" and args.run_command == "fork":
        from chessy.training.smoke import fork_smoke
        print(fork_smoke(root=_project_root(), snapshot_path=args.snapshot, config_path=args.config, mode=args.mode)); return 0
    if args.command == "snapshot" and args.snapshot_command == "verify":
        return _verify_snapshot(args.path)
    if args.command == "replay":
        return _replay_command(args)
    if args.command == "arena" and args.arena_command == "run":
        return _arena_command(args)
    if args.command == "dataset" and args.dataset_command == "personal":
        return _personal_dataset_command(args)
    if args.command == "feedback":
        return _feedback_command(args)
    if args.command == "personalize":
        return _personalize_command(args)
    if args.command == "personal-rl":
        return _personal_rl_command(args)
    raise SystemExit(2)
