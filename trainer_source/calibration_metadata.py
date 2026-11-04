"""Bounded propagation of measured capture calibration into NAM metadata."""

from __future__ import annotations

import math
from typing import Any, Mapping


def calibration_model_metadata(data_config: Mapping[str, Any]) -> dict[str, float]:
    """Return only calibration fields understood by the native NAM loader."""

    config_metadata = data_config.get("metadata")
    if not isinstance(config_metadata, Mapping):
        return {}
    calibration = config_metadata.get("calibration")
    if calibration is None:
        if config_metadata.get("physical_amplifier") is True:
            raise ValueError("physical data config is missing calibration metadata")
        return {}
    if not isinstance(calibration, Mapping):
        raise ValueError("data config calibration must be an object")
    if (
        config_metadata.get("physical_amplifier") is True
        and calibration.get("status") != "measured"
    ):
        raise ValueError("physical data config calibration status must be measured")

    exported: dict[str, float] = {}
    for field in ("input_level_dbu", "output_level_dbu"):
        value = calibration.get(field)
        if value is None:
            if (
                field == "input_level_dbu"
                and config_metadata.get("physical_amplifier") is True
            ):
                raise ValueError(
                    "physical data config calibration requires input_level_dbu"
                )
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"calibration {field} must be finite")
        exported[field] = float(value)
    return exported
