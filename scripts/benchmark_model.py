"""Measure batch inference latency for the default Chessy model."""

from __future__ import annotations

import argparse
import time

import torch

from chessy.model import ChessyModel, resolve_device


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if any(size <= 0 for size in arguments.batch_sizes):
        parser.error("--batch-sizes values must be positive")
    if arguments.warmup < 0:
        parser.error("--warmup must be non-negative")
    if arguments.iterations <= 0:
        parser.error("--iterations must be positive")

    device = resolve_device(arguments.device)
    torch.manual_seed(arguments.seed)
    model = ChessyModel().to(device).eval()
    print(f"device = {device}")
    with torch.inference_mode():
        for batch_size in arguments.batch_sizes:
            boards = torch.randn((batch_size, 119, 8, 8), dtype=torch.float32, device=device)
            for _ in range(arguments.warmup):
                model(boards)
            _synchronize(device)
            started = time.perf_counter()
            for _ in range(arguments.iterations):
                model(boards)
            _synchronize(device)
            elapsed = time.perf_counter() - started
            latency_ms = elapsed / arguments.iterations * 1000
            positions_per_second = batch_size * arguments.iterations / elapsed
            print(
                f"batch_size = {batch_size}; latency_ms = {latency_ms:.3f}; "
                f"positions_per_second = {positions_per_second:.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
