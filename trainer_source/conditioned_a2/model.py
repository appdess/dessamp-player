"""Pinned NAM 0.13 A2 children with block-constant global controls."""

from __future__ import annotations

import copy
import importlib.resources
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import nam
from nam.models.base import BaseNet
from nam.models.wavenet import WaveNet


A2_CHANNELS = {"lite": 3, "full": 8}
EXPECTED_NAM_VERSION = "0.13.0"
MAX_GLOBAL_CONDITIONS = 64


def _require_supported_nam() -> None:
    version = getattr(nam, "__version__", None)
    if version != EXPECTED_NAM_VERSION:
        raise RuntimeError(
            f"Conditioned A2 is pinned to neural-amp-modeler {EXPECTED_NAM_VERSION}; "
            f"found {version!r}"
        )


def _upstream_child_config(channels: int, condition_size: int) -> dict[str, Any]:
    """Read the exact A2 child recipe shipped by pinned upstream NAM."""
    _require_supported_nam()
    if channels not in A2_CHANNELS.values():
        raise ValueError(f"A2 channels must be one of {sorted(A2_CHANNELS.values())}")
    resource = importlib.resources.files("nam.train._resources").joinpath(
        "config_model_packed.json"
    )
    with resource.open("r", encoding="utf-8") as source:
        packed = json.load(source)
    children = packed["net"]["config"]["submodels"]
    name = f"channels_{channels}"
    matches = [child for child in children if child.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"Pinned upstream A2 recipe has no unique {name!r} child")
    config = copy.deepcopy(matches[0]["config"])
    layers = config.get("layers_configs")
    if not isinstance(layers, list) or len(layers) != 1:
        raise RuntimeError("Pinned upstream A2 recipe changed its single-array shape")
    if layers[0].get("condition_size") != 1:
        raise RuntimeError("Pinned upstream A2 recipe no longer has one audio condition row")
    layers[0]["condition_size"] = condition_size
    return config


class ConditionedA2Net(BaseNet):
    """One A2 child whose input mixer sees audio plus global control rows.

    Upstream NAM 0.13 already implements the exact A2 convolutions and export
    order.  We instantiate that model with ``1 + G`` mixer inputs, while the
    residual audio path remains mono, and supply the block-constant rows in a
    small pinned wrapper.  Export rewrites ``condition_size`` back to the base
    audio row and records ``global_condition_size`` for DessMetal Core.
    """

    def __init__(self, channels: int, global_condition_size: int, sample_rate: float | None = None):
        if global_condition_size < 1 or global_condition_size > MAX_GLOBAL_CONDITIONS:
            raise ValueError(
                f"global_condition_size must be between 1 and {MAX_GLOBAL_CONDITIONS}"
            )
        super().__init__(sample_rate=sample_rate)
        self.channels = int(channels)
        self.global_condition_size = int(global_condition_size)
        config = _upstream_child_config(
            channels=self.channels,
            condition_size=1 + self.global_condition_size,
        )
        self._audio_net = WaveNet.init_from_config(config)

        # This experiment intentionally relies on a pinned private boundary.
        # Fail clearly if an accidental dependency upgrade changes it.
        inner = getattr(self._audio_net, "_net", None)
        if inner is None or not hasattr(inner, "_layer_arrays"):
            raise RuntimeError("Pinned upstream WaveNet internals changed")

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._audio_net.receptive_field

    def forward(
        self,
        x: torch.Tensor,
        global_condition: torch.Tensor,
        pad_start: bool | None = None,
    ) -> torch.Tensor:
        pad_start = self.pad_start_default if pad_start is None else pad_start
        scalar = x.ndim == 1
        if scalar:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError(f"Audio input must have shape (L,) or (B,L), got {tuple(x.shape)}")

        if global_condition.ndim == 1:
            global_condition = global_condition[None, :]
        if global_condition.ndim != 2:
            raise ValueError(
                "Global condition must have shape (G,) or (B,G), "
                f"got {tuple(global_condition.shape)}"
            )
        if global_condition.shape[1] != self.global_condition_size:
            raise ValueError(
                f"Expected {self.global_condition_size} global controls, "
                f"got {global_condition.shape[1]}"
            )
        if global_condition.shape[0] == 1 and x.shape[0] > 1:
            global_condition = global_condition.expand(x.shape[0], -1)
        if global_condition.shape[0] != x.shape[0]:
            raise ValueError("Audio and global-condition batch sizes differ")
        if not torch.isfinite(global_condition).all():
            raise ValueError("Global conditions must be finite")

        if pad_start:
            x = torch.cat(
                (
                    torch.zeros(
                        (x.shape[0], self.receptive_field - 1),
                        dtype=x.dtype,
                        device=x.device,
                    ),
                    x,
                ),
                dim=1,
            )
        if x.shape[1] < self.receptive_field:
            raise ValueError(
                f"Input has {x.shape[1]} samples, fewer than receptive field "
                f"{self.receptive_field}"
            )
        y = self._forward_mps_safe(x, global_condition=global_condition)
        return y[0] if scalar else y

    def _forward(self, x: torch.Tensor, global_condition: torch.Tensor) -> torch.Tensor:
        audio = x[:, None, :]
        controls = global_condition.to(dtype=x.dtype, device=x.device)
        controls = controls[:, :, None].expand(-1, -1, x.shape[1])
        condition = torch.cat((audio, controls), dim=1)

        inner = self._audio_net._net
        layer_output = audio
        head_input = None
        for layer_array in inner._layer_arrays:
            head_input, layer_output = layer_array(
                layer_output,
                condition,
                head_input=head_input,
            )
        if head_input is None:
            raise RuntimeError("A2 layer stack produced no head input")
        head_input = inner._head_scale * head_input
        output = head_input if inner._head is None else inner._head(head_input)
        if output.shape[1] != 1:
            raise RuntimeError(f"A2 child produced {output.shape[1]} output channels")
        return output[:, 0, :]

    def _at_nominal_settings(self, x: torch.Tensor) -> torch.Tensor:
        controls = torch.full(
            (self.global_condition_size,),
            0.5,
            dtype=x.dtype,
            device=x.device,
        )
        return self(x, controls)

    def export_config(self) -> dict[str, Any]:
        config = copy.deepcopy(self._audio_net._export_config())
        layers = config.get("layers")
        if not isinstance(layers, list) or len(layers) != 1:
            raise RuntimeError("A2 export no longer contains one layer array")
        expected = 1 + self.global_condition_size
        if layers[0].get("condition_size") != expected:
            raise RuntimeError(
                f"A2 export condition_size changed: expected {expected}, "
                f"got {layers[0].get('condition_size')!r}"
            )
        layers[0]["condition_size"] = 1
        layers[0]["global_condition_size"] = self.global_condition_size
        config["global_condition_size"] = self.global_condition_size
        return config

    def export_weights(self) -> np.ndarray:
        return self._audio_net._export_weights()

    # BaseNet's abstract export hooks are implemented so the object remains a
    # well-formed NAM model, while the explicit helpers below control metadata.
    def _export_config(self) -> dict[str, Any]:
        return self.export_config()

    def _export_weights(self) -> np.ndarray:
        return self.export_weights()

    def _export_input_output(self) -> tuple[np.ndarray, np.ndarray]:
        if self.sample_rate is None:
            raise RuntimeError("Cannot export an A2 child without a sample rate")
        samples = int(self.sample_rate)
        x = torch.zeros(samples, dtype=torch.float32, device=self.device)
        controls = torch.full(
            (self.global_condition_size,), 0.5, dtype=x.dtype, device=x.device
        )
        with torch.no_grad():
            y = self(x, controls, pad_start=True)
        return x.cpu().numpy(), y.cpu().numpy()

    def _get_export_architecture(self) -> str:
        return "WaveNet"


def build_conditioned_a2(
    child: str,
    global_condition_size: int,
    sample_rate: float | None = None,
) -> ConditionedA2Net:
    try:
        channels = A2_CHANNELS[child]
    except KeyError as error:
        raise ValueError(f"child must be one of {sorted(A2_CHANNELS)}") from error
    return ConditionedA2Net(
        channels=channels,
        global_condition_size=global_condition_size,
        sample_rate=sample_rate,
    )


def _date_metadata() -> dict[str, int]:
    now = datetime.now(timezone.utc)
    return {
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
    }


def export_child_dict(
    model: ConditionedA2Net,
    control_order: Sequence[str],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if model.sample_rate is None or not math.isfinite(model.sample_rate):
        raise ValueError("A finite sample rate is required for export")
    if len(control_order) != model.global_condition_size:
        raise ValueError("Control order length does not match global_condition_size")
    if len(set(control_order)) != len(control_order) or any(
        not isinstance(name, str) or not name for name in control_order
    ):
        raise ValueError("Control order must contain unique non-empty strings")

    export_metadata: dict[str, Any] = {
        "date": _date_metadata(),
        "control_order": list(control_order),
        "conditioned_a2": True,
        "a2_channels": model.channels,
        "receptive_field": model.receptive_field,
        "upstream_nam_version": EXPECTED_NAM_VERSION,
    }
    if metadata:
        export_metadata.update(copy.deepcopy(metadata))
    return {
        "version": "0.7.0",
        "metadata": export_metadata,
        "architecture": "WaveNet",
        "config": model.export_config(),
        "weights": model.export_weights().tolist(),
        "sample_rate": float(model.sample_rate),
    }


def write_model_dict(model_dict: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model_dict, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )


def pack_slimmable_container(
    lite: dict[str, Any],
    full: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for name, child, channels in (("lite", lite, 3), ("full", full, 8)):
        if child.get("architecture") != "WaveNet":
            raise ValueError(f"{name} child is not a WaveNet")
        child_metadata = child.get("metadata", {})
        if child_metadata.get("a2_channels") != channels:
            raise ValueError(f"{name} child does not have {channels} A2 channels")
    if lite.get("sample_rate") != full.get("sample_rate"):
        raise ValueError("A2 children have different sample rates")
    if lite["config"].get("global_condition_size") != full["config"].get(
        "global_condition_size"
    ):
        raise ValueError("A2 children have different global-condition sizes")
    if lite.get("metadata", {}).get("control_order") != full.get("metadata", {}).get(
        "control_order"
    ):
        raise ValueError("A2 children have different control orders")

    container_metadata: dict[str, Any] = {
        "date": _date_metadata(),
        "conditioned_a2": True,
        "control_order": copy.deepcopy(lite["metadata"]["control_order"]),
        "selection": "lite_below_0.5_full_at_or_above_0.5",
    }
    if metadata:
        container_metadata.update(copy.deepcopy(metadata))
    return {
        "version": "0.7.0",
        "metadata": container_metadata,
        "architecture": "SlimmableContainer",
        "config": {
            "submodels": [
                {"max_value": 0.5, "model": copy.deepcopy(lite)},
                {"max_value": 1.0, "model": copy.deepcopy(full)},
            ]
        },
        "weights": [],
        "sample_rate": lite["sample_rate"],
    }
