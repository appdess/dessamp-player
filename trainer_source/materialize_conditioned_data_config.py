"""Materialize a conditioned NAM data config from one persistent capture project."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from dessmetal_capture.store import CaptureStore


DEFAULT_VALIDATION_BASENAME = "starter-00__train-pcm24.wav"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_data_config(
    *,
    input_path: Path,
    control_order: Sequence[str],
    captures: Sequence[Mapping[str, Any]],
    validation_capture_id: str | None = None,
    validation_basename: str | None = DEFAULT_VALIDATION_BASENAME,
    delay_samples: int = 0,
    metadata: Mapping[str, Any] | None = None,
    allow_capture_warnings: bool = False,
) -> dict[str, Any]:
    if isinstance(delay_samples, bool) or not isinstance(delay_samples, int):
        raise ValueError("delay_samples must be an integer")
    controls = [str(value).strip() for value in control_order]
    if not controls or len(controls) != len(set(controls)) or any(not value for value in controls):
        raise ValueError("control_order must contain unique non-empty ids")
    if validation_capture_id is not None:
        validation_capture_id = validation_capture_id.strip()
        if not validation_capture_id:
            raise ValueError("validation_capture_id must be non-empty")
    if validation_basename is not None and (
        Path(validation_basename).name != validation_basename or not validation_basename
    ):
        raise ValueError("validation_basename must be one filename")
    if validation_capture_id is None and validation_basename is None:
        raise ValueError("select one validation capture by id or basename")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    warning_ids = [
        str(capture.get("id", "<unknown>"))
        for capture in captures
        if capture.get("quality_status") == "warning"
    ]
    if warning_ids and not allow_capture_warnings:
        raise ValueError(
            "warning captures require --allow-capture-warnings for an explicitly labelled workflow proof: "
            + ", ".join(warning_ids)
        )
    accepted = [
        capture
        for capture in captures
        if capture.get("quality_status") == "pass"
        or (allow_capture_warnings and capture.get("quality_status") == "warning")
    ]
    if len(accepted) < 2:
        raise ValueError("at least two accepted captures are required")
    ordered = sorted(
        accepted,
        key=lambda capture: (str(capture.get("created_at", "")), str(capture.get("wav_path", ""))),
    )
    outputs: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    seen_vectors: set[tuple[float, ...]] = set()
    for capture in ordered:
        settings = capture.get("settings")
        if not isinstance(settings, Mapping) or set(settings) != set(controls):
            raise ValueError("capture settings do not match the fixed control order")
        vector = tuple(float(settings[control_id]) for control_id in controls)
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("capture settings contain a non-finite value")
        if vector in seen_vectors:
            raise ValueError(f"duplicate accepted capture vector: {vector}")
        seen_vectors.add(vector)
        wav_path = Path(str(capture.get("wav_path", ""))).expanduser().resolve()
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        accepted_sha256 = capture.get("wav_sha256")
        if not isinstance(accepted_sha256, str) or _SHA256_RE.fullmatch(accepted_sha256) is None:
            raise ValueError(
                f"accepted capture {capture.get('id', '<unknown>')!r} has no valid SQLite wav_sha256"
            )
        current_sha256 = _sha256(wav_path)
        if current_sha256 != accepted_sha256:
            raise ValueError(
                f"accepted capture {capture.get('id', '<unknown>')!r} WAV hash no longer matches SQLite"
            )
        output = {"g_vector": list(vector), "y_path": str(wav_path)}
        is_validation = (
            str(capture.get("id", "")) == validation_capture_id
            if validation_capture_id is not None
            else wav_path.name == validation_basename
        )
        if is_validation:
            validation.append(output)
        else:
            outputs.append(output)
    if len(validation) != 1:
        raise ValueError(
            "expected exactly one accepted validation capture selected by "
            f"{('id ' + repr(validation_capture_id)) if validation_capture_id is not None else ('basename ' + repr(validation_basename))}; "
            f"found {len(validation)}"
        )
    if not outputs:
        raise ValueError("training split contains no accepted captures")
    config_metadata = dict(metadata or {})
    declared_order = config_metadata.get("control_order")
    if declared_order is not None and list(declared_order) != controls:
        raise ValueError("metadata control_order does not match the materialized vector order")
    config_metadata["control_order"] = controls
    config_metadata["validation_capture_id"] = validation_capture_id
    return {
        "schema_version": 1,
        "metadata": config_metadata,
        "type": "shared_in_multi_out",
        "common": {
            "allow_unequal_lengths": False,
            "delay": delay_samples,
            "input_gain": 0.0,
            "x_path": str(input_path.resolve()),
        },
        "train": {"outputs": outputs},
        "validation": {"outputs": validation},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-root",
        type=Path,
        help="Legacy regression-harness root containing 00_WORKFLOW and 04_TRAINER_DATA",
    )
    source.add_argument(
        "--data-dir",
        type=Path,
        help="Persistent capture-companion data directory for a generic project",
    )
    parser.add_argument("--project-id", help="Required with --data-dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-capture-id")
    parser.add_argument("--validation-basename")
    parser.add_argument("--delay-samples", type=int)
    parser.add_argument("--allow-capture-warnings", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise RuntimeError(f"Refusing to replace existing data config: {output_path}")
    if args.run_root is not None:
        run_root = args.run_root.expanduser().resolve()
        state = json.loads(
            (run_root / "00_WORKFLOW" / "run-state.json").read_text(encoding="utf-8")
        )
        store = CaptureStore(run_root / "04_TRAINER_DATA")
        project_id = str(state["project_id"])
        project = store.get_project(project_id)
        input_path = Path(state["input_path"])
        control_order = list(state["control_order"])
        validation_capture_id = args.validation_capture_id
        validation_basename = args.validation_basename or DEFAULT_VALIDATION_BASENAME
        delay_samples = (
            args.delay_samples
            if args.delay_samples is not None
            else int(state.get("training_delay_samples", 0))
        )
        physical_amplifier = bool(state.get("physical_amplifier", False))
        source_kind = "regression-run-root"
        declared_source_kind = str(state.get("source_kind") or source_kind)
        workflow_validation_only = bool(state.get("workflow_validation_only", False))
        source_provenance = state.get("source")
        calibration = state.get("calibration") or (project.get("metadata") or {}).get(
            "calibration"
        )
    else:
        if not args.project_id:
            raise ValueError("--project-id is required with --data-dir")
        if not args.validation_capture_id:
            raise ValueError(
                "generic projects require --validation-capture-id; basenames are not a durable identity"
            )
        store = CaptureStore(args.data_dir.expanduser().resolve())
        project_id = args.project_id
        project = store.get_project(project_id)
        if not project.get("input_signal_path"):
            raise ValueError("the persistent project has no standardized input signal")
        input_path = Path(project["input_signal_path"])
        control_order = [control["id"] for control in project["controls"]]
        if any(
            float(control["minimum"]) < 0.0 or float(control["maximum"]) > 1.0
            for control in project["controls"]
        ):
            raise ValueError(
                "generic A2 project model ranges must stay inside [0, 1]; native amp ranges belong in display fields"
            )
        validation_capture_id = args.validation_capture_id
        validation_basename = None
        project_metadata = project.get("metadata") or {}
        delay_samples = (
            args.delay_samples
            if args.delay_samples is not None
            else int(project_metadata.get("training_delay_samples", 0))
        )
        physical_amplifier = bool(project_metadata.get("physical_amplifier", False))
        source_kind = "persistent-capture-project"
        declared_source_kind = str(project_metadata.get("source_kind") or source_kind)
        workflow_validation_only = bool(
            project_metadata.get("workflow_validation_only", False)
        )
        source_provenance = project_metadata.get("source")
        calibration = project_metadata.get("calibration")
    captures = store.list_captures(project_id)
    config = build_data_config(
        input_path=input_path,
        control_order=control_order,
        captures=captures,
        validation_capture_id=validation_capture_id,
        validation_basename=validation_basename,
        delay_samples=delay_samples,
        metadata={
            "project_id": project_id,
            "project_name": project["name"],
            "amp_name": project["amp_name"],
            "controls": project["controls"],
            "control_order": control_order,
            "physical_amplifier": physical_amplifier,
            "source": source_kind,
            "source_kind": declared_source_kind,
            "source_provenance": source_provenance,
            "calibration": calibration,
            "workflow_validation_only": workflow_validation_only,
            "capture_warnings_admitted": bool(args.allow_capture_warnings),
        },
        allow_capture_warnings=args.allow_capture_warnings,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "train_count": len(config["train"]["outputs"]),
                "validation_count": len(config["validation"]["outputs"]),
                "control_count": len(control_order),
                "control_order": control_order,
                "project_id": project_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
