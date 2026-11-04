"""Fail-closed loading for PyTorch Lightning checkpoint files."""

from collections.abc import Mapping
import inspect
import math
from pathlib import Path
import re


_PICKLE_CHECKPOINT_OPTIONS = frozenset(("ckpt_path", "resume_from_checkpoint"))
_MINIMUM_SAFE_TORCH_VERSION = (2, 6, 0)
_STABLE_TORCH_VERSION = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:\+[^\s]+)?$"
)


def validated_trainer_options(options, *, context):
    """Copy trainer options after rejecting framework-managed pickle loading."""
    if not isinstance(options, Mapping):
        raise TypeError(f"{context} must be a mapping")
    forbidden = sorted(_PICKLE_CHECKPOINT_OPTIONS.intersection(options))
    if forbidden:
        raise ValueError(
            f"{context} contains disabled checkpoint option(s): {', '.join(forbidden)}"
        )
    return dict(options)


def load_weights_only_checkpoint(
    checkpoint_path,
    *,
    map_location="cpu",
    torch_module=None,
):
    """Load tensor state without allowing pickle to construct arbitrary objects."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    if torch_module is None:
        import torch as torch_module

    version_text = str(getattr(torch_module, "__version__", ""))
    version_match = _STABLE_TORCH_VERSION.fullmatch(version_text)
    if version_match is None:
        raise RuntimeError(
            "Cannot prove that this is a stable, supported PyTorch version"
        )
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

    checkpoint = torch_module.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint root must be a mapping")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint must contain a non-empty state_dict mapping")
    if not all(isinstance(name, str) for name in state_dict):
        raise ValueError("Checkpoint state_dict keys must be strings")
    is_tensor = getattr(torch_module, "is_tensor", None)
    if not callable(is_tensor) or not all(
        is_tensor(value) for value in state_dict.values()
    ):
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
