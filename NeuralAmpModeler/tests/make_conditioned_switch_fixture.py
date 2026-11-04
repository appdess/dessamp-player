#!/usr/bin/env python3
"""Create a deterministic three-control A2 Full .dess integration fixture.

Run this with the trainer's pinned neural-amp-modeler 0.13 environment. The
fixture is generated outside Git and exists only to exercise current-source
APP/AUv2/VST3 builds with one knob and two independently switchable controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "capture_companion" / "src"))

from trainer_source.conditioned_a2.model import (  # noqa: E402
    build_conditioned_a2,
    export_child_dict,
    write_model_dict,
)
from dessmetal_capture.dess_package import create_dess_package  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="New .dess fixture path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".dess":
        raise ValueError("output must use the .dess extension")
    if output.exists():
        raise FileExistsError(f"refusing to replace fixture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20260820)
    model = build_conditioned_a2("full", 3, sample_rate=48_000.0)
    model_path = output.parent / "model.nam"
    write_model_dict(
        export_child_dict(
            model,
            ("lead_gain", "bright", "bottom"),
            metadata={"purpose": "exact-au-switch-integration-fixture"},
        ),
        model_path,
    )
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    controls = [
        {
            "id": "lead_gain",
            "label": "Lead Gain",
            "minimum": 0.0,
            "maximum": 1.0,
            "default_value": 0.5,
            "display_minimum": 0.0,
            "display_maximum": 100.0,
            "unit": "%",
            "taper": "linear",
            "model_parameter_index": 0,
        },
        {
            "id": "bright",
            "label": "Air Switch",
            "minimum": 0.0,
            "maximum": 1.0,
            "default_value": 0.0,
            "unit": "",
            "taper": "switch",
            "model_parameter_index": 1,
        },
        {
            "id": "bottom",
            "label": "Low Resonance",
            "minimum": 0.0,
            "maximum": 1.0,
            "default_value": 0.0,
            "unit": "",
            "taper": "switch",
            "model_parameter_index": 2,
        },
    ]
    manifest = {
        "schema_version": 1,
        "artifact_id": "exact-au-switch-fixture",
        "name": "Exact AU switch fixture",
        "presentation": {"head_shell": "dual-red-modern-v1"},
        "model": {
            "path": "model.nam",
            "format": "nam",
            "sha256": model_sha256,
            "format_version": "0.7.0",
            "architecture": "WaveNet",
            "sample_rate": 48_000.0,
            "parameter_mode": "global_condition",
            "global_condition_size": 3,
            "runtime_quality": {
                "mode": "fixed",
                "default_size": 1.0,
                "breakpoints": [],
            },
        },
        "controls": controls,
        "provenance": {"workflow_validation_only": True, "physical_amplifier": False},
    }
    manifest_path = output.parent / "player-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = create_dess_package(manifest_path, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
