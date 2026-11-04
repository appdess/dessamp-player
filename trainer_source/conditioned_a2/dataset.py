"""Shared-input, multi-output dataset for conditioned A2 training."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from nam.data import AbstractDataset, wav_to_tensor


class ConditionedCaptureDataset(AbstractDataset):
    """Index matched capture WAVs without materializing duplicate audio chunks."""

    def __init__(
        self,
        *,
        x: torch.Tensor,
        outputs: list[torch.Tensor],
        conditions: list[torch.Tensor],
        nx: int,
        ny: int,
        sample_rate: int,
        split: str,
        chunks_per_output: int | None = None,
    ) -> None:
        if not outputs or len(outputs) != len(conditions):
            raise ValueError("Outputs and conditions must be non-empty and aligned")
        if nx < 1 or ny < 1 or len(x) < nx + ny - 1:
            raise ValueError("Audio is too short for the requested receptive field/chunk")
        self._x = x.contiguous()
        self._outputs = [output.contiguous() for output in outputs]
        self._conditions = [condition.contiguous() for condition in conditions]
        self._nx = int(nx)
        self._ny = int(ny)
        self._sample_rate = int(sample_rate)
        self._split = split
        self._available_steps_per_output = (len(self._x) - self._nx + 1) // self._ny
        if self._available_steps_per_output < 1:
            raise ValueError("Dataset has no complete chunks")
        if chunks_per_output is None:
            self._step_indices = tuple(range(self._available_steps_per_output))
            self._chunk_selection_mode = "all"
        else:
            if (
                isinstance(chunks_per_output, bool)
                or not isinstance(chunks_per_output, int)
                or not 1 <= chunks_per_output <= self._available_steps_per_output
            ):
                raise ValueError(
                    "chunks_per_output must be between one and the available chunk count"
                )
            # Select the midpoint of each equal-width temporal stratum. This is
            # deterministic, covers the complete stimulus, and avoids the severe
            # content bias of taking one contiguous prefix.
            self._step_indices = tuple(
                ((2 * index + 1) * self._available_steps_per_output)
                // (2 * chunks_per_output)
                for index in range(chunks_per_output)
            )
            if len(set(self._step_indices)) != chunks_per_output:
                raise RuntimeError("Uniform chunk selection produced duplicate indices")
            self._chunk_selection_mode = "uniform-stratified-v1"
        self._steps_per_output = len(self._step_indices)

    def __len__(self) -> int:
        return len(self._outputs) * self._steps_per_output

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        output_index, selected_step = divmod(index, self._steps_per_output)
        step = self._step_indices[selected_step]
        start = step * self._ny
        target_start = start + self._nx - 1
        return (
            self._x[start : start + self._nx + self._ny - 1],
            self._conditions[output_index],
            self._outputs[output_index][target_start : target_start + self._ny],
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return self._ny

    @property
    def condition_size(self) -> int:
        return int(self._conditions[0].numel())

    @property
    def chunk_selection(self) -> dict[str, Any]:
        encoded = json.dumps(
            self._step_indices, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return {
            "mode": self._chunk_selection_mode,
            "available_chunks_per_output": self._available_steps_per_output,
            "selected_chunks_per_output": self._steps_per_output,
            "selected_fraction": self._steps_per_output
            / self._available_steps_per_output,
            "selected_audio_seconds_per_output": self._steps_per_output
            * self._ny
            / self._sample_rate,
            "first_step_index": self._step_indices[0],
            "last_step_index": self._step_indices[-1],
            "step_indices_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def teardown(self) -> None:
        self._x = torch.empty(0)
        self._outputs.clear()
        self._conditions.clear()


def _checked_condition(raw: Any, *, index: int, expected_size: int | None) -> torch.Tensor:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"outputs[{index}].g_vector must be a non-empty list")
    condition = torch.tensor(raw, dtype=torch.float32)
    if condition.ndim != 1 or not torch.isfinite(condition).all():
        raise ValueError(f"outputs[{index}].g_vector must be a finite vector")
    if expected_size is not None and condition.numel() != expected_size:
        raise ValueError(
            f"outputs[{index}].g_vector has {condition.numel()} values; "
            f"expected {expected_size}"
        )
    if torch.any(condition < 0.0) or torch.any(condition > 1.0):
        raise ValueError(f"outputs[{index}].g_vector must be normalized to [0,1]")
    return condition


def load_capture_dataset(
    config: dict[str, Any],
    *,
    split: str,
    nx: int,
    ny: int,
    expected_condition_size: int = 7,
    chunks_per_output: int | None = None,
) -> ConditionedCaptureDataset:
    """Load one split from the existing DessMetal capture-data contract."""
    config = copy.deepcopy(config)
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    common = config.get("common")
    split_config = config.get(split)
    if not isinstance(common, dict) or not isinstance(split_config, dict):
        raise ValueError("Data config requires common, train and validation objects")
    if common.get("allow_unequal_lengths", False):
        raise ValueError("Conditioned A2 requires exactly aligned capture lengths")
    if common.get("delay", 0) != 0:
        raise ValueError("Conditioned A2 expects latency-aligned captures (delay=0)")

    raw_x_path = split_config.get("x_path", common.get("x_path", ""))
    if not isinstance(raw_x_path, str) or not raw_x_path:
        raise ValueError(f"Data config {split}.x_path or common.x_path must be a path")
    x_path = Path(raw_x_path).expanduser().resolve()
    if not x_path.is_file():
        raise FileNotFoundError(x_path)
    x, info = wav_to_tensor(x_path, info=True)
    input_gain_db = float(common.get("input_gain", 0.0))
    if not math.isfinite(input_gain_db):
        raise ValueError("common.input_gain must be finite")
    if input_gain_db != 0.0:
        x = x * (10.0 ** (input_gain_db / 20.0))

    raw_outputs = split_config.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ValueError(f"Data config {split}.outputs must be non-empty")
    outputs: list[torch.Tensor] = []
    conditions: list[torch.Tensor] = []
    for index, item in enumerate(raw_outputs):
        if not isinstance(item, dict):
            raise ValueError(f"{split}.outputs[{index}] must be an object")
        y_path = Path(item.get("y_path", "")).expanduser().resolve()
        if not y_path.is_file():
            raise FileNotFoundError(y_path)
        y, y_info = wav_to_tensor(y_path, info=True, rate=info.rate)
        if y_info.rate != info.rate or len(y) != len(x):
            raise ValueError(
                f"Capture {y_path} does not exactly match input rate/length "
                f"({y_info.rate}/{len(y)} versus {info.rate}/{len(x)})"
            )
        y_scale = float(item.get("y_scale", 1.0))
        if not math.isfinite(y_scale):
            raise ValueError(f"{split}.outputs[{index}].y_scale must be finite")
        outputs.append(y * y_scale)
        conditions.append(
            _checked_condition(
                item.get("g_vector"),
                index=index,
                expected_size=expected_condition_size,
            )
        )

    return ConditionedCaptureDataset(
        x=x,
        outputs=outputs,
        conditions=conditions,
        nx=nx,
        ny=ny,
        sample_rate=info.rate,
        split=split,
        chunks_per_output=chunks_per_output,
    )
