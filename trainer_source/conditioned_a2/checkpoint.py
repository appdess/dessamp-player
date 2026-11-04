"""Restricted checkpoint loading for the isolated conditioned-A2 trainer."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import math
from pathlib import Path
import re


_MINIMUM_SAFE_TORCH_VERSION = (2, 6, 0)
_STABLE_TORCH_VERSION = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:\+[^\s]+)?$"
)


def load_weights_only_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str = "cpu",
    torch_module=None,
):
    """Load a Lightning checkpoint without allowing pickle object construction."""
    path = Path(checkpoint_path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    if torch_module is None:
        import torch as torch_module

    version_text = str(getattr(torch_module, "__version__", ""))
    version_match = _STABLE_TORCH_VERSION.fullmatch(version_text)
    if version_match is None:
        raise RuntimeError("Cannot prove that this is a stable, supported PyTorch version")
    version = tuple(
        int(version_match.group(part)) for part in ("major", "minor", "patch")
    )
    if version < _MINIMUM_SAFE_TORCH_VERSION:
        raise RuntimeError(
            "Checkpoint loading requires PyTorch 2.6.0 or newer because older "
            "versions are affected by CVE-2025-32434"
        )

    try:
        load_parameters = inspect.signature(torch_module.load).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot prove that this PyTorch loader supports weights_only=True"
        ) from exc
    if "weights_only" not in load_parameters:
        raise RuntimeError(
            "This operation requires a PyTorch version with weights_only=True support"
        )

    checkpoint = torch_module.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint root must be a mapping")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint must contain a non-empty state_dict mapping")
    if not all(isinstance(name, str) for name in state_dict):
        raise ValueError("Checkpoint state_dict keys must be strings")
    is_tensor = getattr(torch_module, "is_tensor", None)
    if not callable(is_tensor) or not all(is_tensor(value) for value in state_dict.values()):
        raise ValueError("Checkpoint state_dict values must be tensors")
    sample_rate = checkpoint.get("sample_rate")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float))
        or not math.isfinite(sample_rate)
        or sample_rate <= 0
    ):
        raise ValueError("Checkpoint must contain a positive numeric sample_rate")
    return checkpoint
