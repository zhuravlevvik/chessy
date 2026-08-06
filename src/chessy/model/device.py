"""Portable device selection for CPU, Apple MPS, and CUDA."""

from __future__ import annotations

from typing import Literal

import torch

DeviceName = Literal["auto", "cpu", "mps", "cuda"]


def resolve_device(requested: DeviceName = "auto") -> torch.device:
    """Resolve a requested backend, preferring MPS over CUDA in ``auto`` mode."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"unknown device request: {requested!r}")
