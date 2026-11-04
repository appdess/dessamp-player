"""Materialize all-train teacher data plus a disjoint-DI validation reservation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from dessmetal_capture.domain import settings_key
from dessmetal_capture.store import CaptureStore
from dessmetal_capture.wav_io import inspect_wav


MAX_JSON_BYTES = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_JSON_BYTES:
        raise ValueError(f"JSON evidence is missing, empty, or oversized: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence root must be an object: {path}")
    return payload


def _checked_controls(project: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[str]]:
    controls = project.get("controls")
    if not isinstance(controls, list) or not controls:
        raise ValueError("teacher project requires controls")
    control_ids = [str(control.get("id", "")) for control in controls]
    if (
        len(control_ids) != len(set(control_ids))
        or any(not control_id for control_id in control_ids)
        or len(control_ids) > 12
    ):
        raise ValueError("teacher project has an invalid control order")
    for control in controls:
        minimum = float(control.get("minimum"))
        maximum = float(control.get("maximum"))
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("teacher project model ranges must stay inside [0, 1]")
    return controls, control_ids


def _vector_key(values: Sequence[float], control_order: Sequence[str]) -> str:
    return settings_key(
        {
            control_id: float(value)
            for control_id, value in zip(control_order, values, strict=True)
        }
    )


def _checked_wav(
    path: Path,
    *,
    expected_sha256: str,
    expected_frames: int,
    expected_sample_rate: int,
    label: str,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 no longer matches its immutable evidence")
    info = inspect_wav(path)
    if (
        info["sample_rate"] != expected_sample_rate
        or info["channels"] != 1
        or info["sample_width_bits"] != 24
        or info["frames"] != expected_frames
    ):
        raise ValueError(f"{label} must exactly match its split input as mono PCM24")
    peak = float(info["peak"])
    if not math.isfinite(peak) or not 0.0 < peak < 1.0:
        raise ValueError(f"{label} is silent, non-finite, or clipped")
    return info


def build_teacher_data_config(
    *,
    project: Mapping[str, Any],
    captures: Sequence[Mapping[str, Any]],
    validation_reservation: Mapping[str, Any],
    validation_reservation_path: Path,
) -> dict[str, Any]:
    metadata = project.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("teacher project metadata is missing")
    if (
        metadata.get("source_kind") != "teacher_model_distillation"
        or metadata.get("physical_amplifier") is not False
        or metadata.get("workflow_validation_only") is not True
    ):
        raise ValueError(
            "project must be a private workflow-validation-only teacher distillation"
        )
    controls, control_order = _checked_controls(project)
    declared_order = metadata.get("control_order")
    if declared_order is not None and list(declared_order) != control_order:
        raise ValueError("project metadata control_order does not match project controls")

    input_info = project.get("input_signal_info")
    input_path_raw = project.get("input_signal_path")
    if not isinstance(input_info, Mapping) or not isinstance(input_path_raw, str):
        raise ValueError("teacher project has no standardized training input")
    training_input = Path(input_path_raw).expanduser().resolve()
    training_input_sha256 = str(input_info.get("sha256", ""))
    training_frames = int(input_info.get("frames", 0))
    sample_rate = int(project.get("sample_rate", 0))
    _checked_wav(
        training_input,
        expected_sha256=training_input_sha256,
        expected_frames=training_frames,
        expected_sample_rate=sample_rate,
        label="training input",
    )

    candidate_points = project.get("candidate_points")
    if not isinstance(candidate_points, list) or not candidate_points:
        raise ValueError("teacher project has no reviewed candidate points")
    candidate_keys = {
        settings_key(
            {
                control_id: float(point[control_id])
                for control_id in control_order
            }
        )
        for point in candidate_points
    }
    if len(candidate_keys) != len(candidate_points):
        raise ValueError("teacher project candidate points are duplicated")

    training_by_key: dict[str, dict[str, Any]] = {}
    for capture in captures:
        if capture.get("quality_status") != "pass":
            raise ValueError("teacher distillation materialization admits only pass captures")
        settings = capture.get("settings")
        if not isinstance(settings, Mapping) or set(settings) != set(control_order):
            raise ValueError("teacher capture settings do not match control_order")
        vector = [float(settings[control_id]) for control_id in control_order]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in vector):
            raise ValueError("teacher capture contains a non-finite or unnormalized setting")
        key = _vector_key(vector, control_order)
        if key in training_by_key:
            raise ValueError(f"duplicate teacher training vector: {key}")
        validation = capture.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError("teacher capture provenance is missing")
        teacher_model = validation.get("teacher_model")
        if (
            validation.get("source_kind") != "teacher_model_distillation"
            or validation.get("role") != "train"
            or validation.get("shared_input_sha256") != training_input_sha256
            or not isinstance(teacher_model, Mapping)
            or not isinstance(teacher_model.get("sha256"), str)
        ):
            raise ValueError("teacher capture is not bound to the all-train import contract")
        wav_path = Path(str(capture.get("wav_path", ""))).expanduser().resolve()
        wav_sha256 = str(capture.get("wav_sha256", ""))
        _checked_wav(
            wav_path,
            expected_sha256=wav_sha256,
            expected_frames=training_frames,
            expected_sample_rate=sample_rate,
            label=f"training capture {capture.get('id', '<unknown>')}",
        )
        training_by_key[key] = {
            "capture_id": str(capture.get("id", "")),
            "g_vector": vector,
            "y_path": str(wav_path),
            "teacher_sha256": str(teacher_model["sha256"]),
        }
    if set(training_by_key) != candidate_keys:
        raise ValueError("accepted teacher captures do not cover every reviewed candidate point")

    if validation_reservation.get("schema_version") != 1:
        raise ValueError("validation reservation must use schema version 1")
    if validation_reservation.get("target_audio_available_to_acquisition") is not False:
        raise ValueError("internal-validation targets must remain unavailable to acquisition")
    scope = validation_reservation.get("evaluation_scope")
    if not isinstance(scope, Mapping) or scope.get("role") != "internal-validation-on-disjoint-di":
        raise ValueError("reservation role must be internal-validation-on-disjoint-di")
    if validation_reservation.get("control_order") != control_order:
        raise ValueError("validation reservation control_order does not match the project")
    provenance = validation_reservation.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("source_kind") != "teacher_model_distillation"
        or provenance.get("project_id") != project.get("id")
    ):
        raise ValueError("validation reservation is not bound to this teacher project")
    validation_input_record = validation_reservation.get("evaluation_input")
    if not isinstance(validation_input_record, Mapping):
        raise ValueError("validation reservation has no input record")
    validation_input = Path(str(validation_input_record.get("path", ""))).expanduser().resolve()
    validation_input_sha256 = str(validation_input_record.get("sha256", ""))
    validation_frames = int(validation_input_record.get("frames", 0))
    if validation_input_sha256 == training_input_sha256:
        raise ValueError("training and validation inputs must be byte-distinct DIs")
    _checked_wav(
        validation_input,
        expected_sha256=validation_input_sha256,
        expected_frames=validation_frames,
        expected_sample_rate=sample_rate,
        label="validation input",
    )

    references = validation_reservation.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("validation reservation requires references")
    validation_by_key: dict[str, dict[str, Any]] = {}
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise ValueError(f"validation reference {index} must be an object")
        vector_raw = reference.get("g_vector")
        if not isinstance(vector_raw, list) or len(vector_raw) != len(control_order):
            raise ValueError(f"validation reference {index} does not match control_order")
        vector = [float(value) for value in vector_raw]
        key = _vector_key(vector, control_order)
        if key in validation_by_key:
            raise ValueError(f"duplicate validation vector: {key}")
        training = training_by_key.get(key)
        if training is None:
            raise ValueError("validation vector set differs from the all-train vector set")
        if reference.get("teacher_sha256") != training["teacher_sha256"]:
            raise ValueError("validation reference was not rendered by the matching teacher")
        setting_id = reference.get("setting_id")
        if setting_id is not None and setting_id != training["capture_id"]:
            raise ValueError("validation setting_id does not match the training capture")
        path = Path(str(reference.get("path", ""))).expanduser().resolve()
        reference_sha256 = str(reference.get("sha256", ""))
        if int(reference.get("frames", 0)) != validation_frames:
            raise ValueError("validation reference length differs from its bound input")
        _checked_wav(
            path,
            expected_sha256=reference_sha256,
            expected_frames=validation_frames,
            expected_sample_rate=sample_rate,
            label=f"validation reference {index}",
        )
        validation_by_key[key] = {"g_vector": vector, "y_path": str(path)}
    if set(validation_by_key) != set(training_by_key):
        raise ValueError("validation reservation must cover exactly the all-train vector set")

    ordered_keys = sorted(training_by_key, key=lambda key: training_by_key[key]["g_vector"])
    reservation_path = validation_reservation_path.expanduser().resolve()
    return {
        "schema_version": 1,
        "type": "shared_in_multi_out",
        "metadata": {
            "project_id": project["id"],
            "project_name": project["name"],
            "amp_name": project["amp_name"],
            "controls": controls,
            "control_order": control_order,
            "physical_amplifier": False,
            "source": "persistent-capture-project-plus-validation-reservation",
            "source_kind": "teacher_model_distillation",
            "source_provenance": metadata.get("source"),
            "workflow_validation_only": True,
            "validation_strategy": "separate_disjoint_di_reservation",
            "training_input_sha256": training_input_sha256,
            "validation_input_sha256": validation_input_sha256,
            "validation_reservation_path": str(reservation_path),
            "validation_reservation_sha256": sha256_file(reservation_path),
            "train_capture_count": len(training_by_key),
            "validation_reference_count": len(validation_by_key),
            "per_teacher_normalization": False,
            "per_teacher_alignment": False,
        },
        "common": {
            "allow_unequal_lengths": False,
            "delay": 0,
            "input_gain": 0.0,
            "x_path": str(training_input),
        },
        "train": {
            "x_path": str(training_input),
            "outputs": [
                {
                    "g_vector": training_by_key[key]["g_vector"],
                    "y_path": training_by_key[key]["y_path"],
                }
                for key in ordered_keys
            ],
        },
        "validation": {
            "x_path": str(validation_input),
            "outputs": [validation_by_key[key] for key in ordered_keys],
        },
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace teacher data config: {output}")
    reservation_path = args.validation_reservation.expanduser().resolve()
    reservation = _load_object(reservation_path)
    store = CaptureStore(args.data_dir.expanduser().resolve())
    project = store.get_project(args.project_id)
    config = build_teacher_data_config(
        project=project,
        captures=store.list_captures(args.project_id),
        validation_reservation=reservation,
        validation_reservation_path=reservation_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--validation-reservation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    config = materialize(build_parser().parse_args())
    print(
        json.dumps(
            {
                "train_count": len(config["train"]["outputs"]),
                "validation_count": len(config["validation"]["outputs"]),
                "control_order": config["metadata"]["control_order"],
                "validation_strategy": config["metadata"]["validation_strategy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
