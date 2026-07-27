"""LoRA adapter interpolation and model-soup utilities used for ACOmni variants."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file


def _adapter_file(directory: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No adapter weights found in {directory}")


def _load(directory: Path) -> dict[str, torch.Tensor]:
    path = _adapter_file(directory)
    state = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu")
    return state.get("state_dict", state) if isinstance(state, dict) else state


def _copy_metadata(reference: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in reference.iterdir():
        if path.is_file() and path.name not in {
            "adapter_model.safetensors", "adapter_model.bin", "optimizer.pt",
            "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin",
        }:
            shutil.copy2(path, output / path.name)
    if not (output / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter_config.json missing in {reference}")


def _validate(states: list[Mapping[str, torch.Tensor]]) -> None:
    keys = set(states[0])
    for state in states[1:]:
        if set(state) != keys:
            raise ValueError("Adapter key sets differ")
    for key in keys:
        shapes = {tuple(state[key].shape) for state in states}
        if len(shapes) != 1:
            raise ValueError(f"Shape mismatch for {key}: {sorted(shapes)}")


def interpolate_adapters(source_a: Path, source_b: Path, output: Path, alpha: float) -> Path:
    """Write ``(1-alpha) * source_a + alpha * source_b``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    left, right = _load(source_a), _load(source_b)
    _validate([left, right])
    merged = {}
    for key, tensor in left.items():
        if torch.is_floating_point(tensor):
            merged[key] = ((1 - alpha) * tensor.float() + alpha * right[key].float()).to(tensor.dtype)
        else:
            merged[key] = right[key]
    _copy_metadata(source_b, output)
    save_file(merged, str(output / "adapter_model.safetensors"))
    (output / "merge_manifest.json").write_text(
        json.dumps({"type": "interpolation", "source_a": str(source_a), "source_b": str(source_b), "alpha": alpha}, indent=2),
        encoding="utf-8",
    )
    return output


def adapter_soup(sources: Mapping[Path, float], output: Path) -> Path:
    if not sources:
        raise ValueError("At least one source adapter is required")
    directories = list(sources)
    states = [_load(path) for path in directories]
    _validate(states)
    total = sum(sources.values())
    if total <= 0:
        raise ValueError("Soup weights must sum to a positive value")
    weights = [sources[path] / total for path in directories]
    merged = {}
    for key, reference in states[0].items():
        if torch.is_floating_point(reference):
            value = torch.zeros_like(reference, dtype=torch.float32)
            for weight, state in zip(weights, states):
                value += weight * state[key].float()
            merged[key] = value.to(reference.dtype)
        else:
            merged[key] = reference
    _copy_metadata(directories[-1], output)
    save_file(merged, str(output / "adapter_model.safetensors"))
    (output / "merge_manifest.json").write_text(
        json.dumps({"type": "soup", "sources": {str(path): weight for path, weight in zip(directories, weights)}}, indent=2),
        encoding="utf-8",
    )
    return output
