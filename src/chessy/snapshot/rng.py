"""Weights-only-safe capture of all random sources used by a run."""
from __future__ import annotations
import random
from typing import Any
import numpy as np
import torch

def _py_to_json(value: object) -> object:
    if isinstance(value, tuple): return {"tuple":[_py_to_json(x) for x in value]}
    if isinstance(value, (str,int,float,type(None))): return value
    raise ValueError("unsupported Python RNG state")
def _py_from_json(value: object) -> object:
    if isinstance(value, dict) and set(value) == {"tuple"}: return tuple(_py_from_json(x) for x in value["tuple"])
    return value
def capture_rng(np_generator: np.random.Generator) -> dict[str, Any]:
    legacy = np.random.get_state()
    state: dict[str, Any] = {"format":"chessy-rng-v1", "python":_py_to_json(random.getstate()), "numpy_legacy":{"name":legacy[0],"keys":torch.from_numpy(legacy[1].astype(np.uint32)),"pos":legacy[2],"has_gauss":legacy[3],"cached_gaussian":legacy[4]}, "numpy_generator":np_generator.bit_generator.state, "torch_cpu":torch.get_rng_state()}
    if torch.backends.mps.is_available() and hasattr(torch.mps,"get_rng_state"):
        try: state["torch_mps"] = torch.mps.get_rng_state()
        except RuntimeError: pass
    if torch.cuda.is_available(): state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state
def restore_rng(state: dict[str, Any], np_generator: np.random.Generator, target_device: str) -> None:
    if state.get("format") != "chessy-rng-v1": raise ValueError("invalid RNG state")
    random.setstate(_py_from_json(state["python"]))
    legacy = state["numpy_legacy"]
    if not isinstance(legacy, dict) or not isinstance(legacy.get("keys"),torch.Tensor): raise ValueError("invalid legacy NumPy state")
    np.random.set_state((legacy["name"], legacy["keys"].cpu().numpy().astype(np.uint32), legacy["pos"], legacy["has_gauss"], legacy["cached_gaussian"]))
    np_generator.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"])
    if target_device == "mps" and "torch_mps" in state and hasattr(torch.mps,"set_rng_state"): torch.mps.set_rng_state(state["torch_mps"])
    if target_device == "cuda" and "torch_cuda" in state: torch.cuda.set_rng_state_all(state["torch_cuda"])
