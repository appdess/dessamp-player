"""Create one generic, evidence-bearing conditioned capture project from JSON."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

from dessmetal_capture.domain import ControlDefinition
from dessmetal_capture.store import CaptureStore
from dessmetal_capture.wav_io import inspect_wav


def load_project_config(path: Path) -> dict[str, Any]:
    if not path.is_file() or not 0 < path.stat().st_size <= 2 * 1024 * 1024:
        raise ValueError(f"project config is missing, empty, or oversized: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("project config root must be an object")
    controls_raw = payload.get("controls")
    candidates = payload.get("candidate_points")
    metadata = payload.get("metadata")
    if not isinstance(controls_raw, list) or not isinstance(candidates, list):
        raise ValueError("project config requires controls and explicit candidate_points lists")
    if not isinstance(metadata, dict):
        raise ValueError("project config requires an experiment metadata object")
    controls = [ControlDefinition.from_mapping(control) for control in controls_raw]
    if any(control.minimum < 0.0 or control.maximum > 1.0 for control in controls):
        raise ValueError(
            "conditioned A2 model ranges must remain inside [0, 1]; use display_minimum/display_maximum for amp labels"
        )
    control_ids = [control.id for control in controls]
    if metadata.get("control_order") != control_ids:
        raise ValueError("metadata.control_order must exactly match the controls array")
    if not isinstance(metadata.get("physical_amplifier"), bool):
        raise ValueError("metadata.physical_amplifier must be explicitly true or false")
    defaults = metadata.get("base_settings")
    if not isinstance(defaults, Mapping) or set(defaults) != set(control_ids):
        raise ValueError("metadata.base_settings must contain every learned control exactly once")
    for control in controls:
        value = float(defaults[control.id])
        if not control.minimum <= value <= control.maximum:
            raise ValueError(f"default for {control.id!r} lies outside its model range")
    frozen = metadata.get("frozen_controls")
    if not isinstance(frozen, Mapping):
        raise ValueError("metadata.frozen_controls must record every non-conditioned amp setting")
    delay = metadata.get("training_delay_samples")
    if isinstance(delay, bool) or not isinstance(delay, int):
        raise ValueError("metadata.training_delay_samples must be a measured integer")
    calibration = metadata.get("calibration")
    if metadata["physical_amplifier"]:
        if not isinstance(calibration, Mapping):
            raise ValueError("a physical project requires metadata.calibration evidence")
        serialized_calibration = json.dumps(calibration, sort_keys=True).upper()
        if "REPLACE" in serialized_calibration or calibration.get("status") != "measured":
            raise ValueError(
                "replace every calibration placeholder and set calibration.status to 'measured'"
            )
        input_level_dbu = calibration.get("input_level_dbu")
        if (
            isinstance(input_level_dbu, bool)
            or not isinstance(input_level_dbu, (int, float))
            or not math.isfinite(float(input_level_dbu))
        ):
            raise ValueError(
                "a measured physical calibration requires a finite input_level_dbu"
            )
    return payload


def create_project(args: argparse.Namespace) -> dict[str, Any]:
    source_config = args.project_config.expanduser().resolve()
    config = load_project_config(source_config)
    input_source = args.input.expanduser().resolve()
    input_info = inspect_wav(input_source)
    if (
        input_info["sample_rate"] != 48_000
        or input_info["channels"] != 1
        or input_info["sample_width_bits"] != 24
    ):
        raise ValueError(
            "DessAmp conditioned capture requires a 48 kHz mono PCM24 standardized input"
        )
    if input_info["peak"] == 0.0:
        raise ValueError("The standardized input is silent; choose the dry capture stimulus")
    store = CaptureStore(args.data_dir.expanduser().resolve())
    input_destination = store.input_dir / f"{input_info['sha256']}.wav"
    if input_destination.exists():
        if inspect_wav(input_destination)["sha256"] != input_info["sha256"]:
            raise ValueError("existing standardized input destination has different bytes")
    else:
        temporary = input_destination.with_name(f".{input_destination.name}.tmp")
        shutil.copy2(input_source, temporary)
        temporary.replace(input_destination)
    project = store.create_project(
        name=str(config.get("name", "")),
        amp_name=str(config.get("amp_name", "")),
        description=str(config.get("description", "")),
        controls=config["controls"],
        candidate_points=config["candidate_points"],
        acquisition_strategy=str(config.get("acquisition_strategy", "coverage-v1")),
        sample_rate=48_000,
        metadata={
            **config["metadata"],
            "project_config_path": str(source_config),
            "project_config_schema_version": int(config.get("schema_version", 1)),
        },
    )
    project = store.set_input_signal(project["id"], input_destination, input_info)
    return {
        "status": "created",
        "project_id": project["id"],
        "data_dir": str(store.data_dir),
        "database": str(store.database_path),
        "input": {"path": str(input_destination), **input_info},
        "control_order": [control["id"] for control in project["controls"]],
        "candidate_count": len(project["candidate_points"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(create_project(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
