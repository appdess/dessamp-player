"""Materialize an audit-safe disagreement candidate config from one capture project."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from dessmetal_capture.store import CaptureStore


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(settings: Mapping[str, Any], control_order: Sequence[str]) -> list[float]:
    if set(settings) != set(control_order):
        raise ValueError("settings do not match the persistent control schema")
    values = [float(settings[control_id]) for control_id in control_order]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("settings contain a non-finite value")
    return values


def build_candidate_config(
    *,
    project: Mapping[str, Any],
    captures: Sequence[Mapping[str, Any]],
    reservation: Mapping[str, Any],
    reservation_path: Path,
) -> dict[str, Any]:
    control_order = [control["id"] for control in project["controls"]]
    if reservation.get("target_audio_available_to_acquisition") is not False:
        raise ValueError("reservation must keep target audio unavailable to acquisition")
    declared_order = reservation.get("control_order")
    if declared_order is not None and declared_order != control_order:
        raise ValueError("reservation control_order does not match the persistent project")
    references = reservation.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("reservation requires at least one hidden reference")
    reserved = []
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping) or not isinstance(reference.get("g_vector"), list):
            raise ValueError(f"hidden reference {index} has no g_vector")
        values = [float(value) for value in reference["g_vector"]]
        if len(values) != len(control_order) or any(not math.isfinite(value) for value in values):
            raise ValueError(f"hidden reference {index} does not match the control schema")
        reserved.append(values)
    candidate_vectors = [vector(point, control_order) for point in project["candidate_points"]]
    captured_vectors = [
        vector(capture["settings"], control_order)
        for capture in captures
        if capture.get("quality_status") != "failed"
    ]
    keyed_candidates = {tuple(values) for values in candidate_vectors}
    if len(keyed_candidates) != len(candidate_vectors):
        raise ValueError("persistent candidate set contains duplicates")
    keyed_reserved = {tuple(values) for values in reserved}
    if len(keyed_reserved) != len(reserved):
        raise ValueError("reservation contains duplicate vectors")
    if keyed_reserved & keyed_candidates:
        raise ValueError(
            "a frozen audit vector appears in the acquirable candidate set"
        )
    if keyed_reserved & {tuple(values) for values in captured_vectors}:
        raise ValueError("a frozen audit vector was already captured into the training project")
    return {
        "schema_version": 1,
        "project_id": project["id"],
        "project_name": project["name"],
        "control_order": control_order,
        "candidate_vectors": candidate_vectors,
        "captured_vectors": captured_vectors,
        "reserved_evaluation_vectors": reserved,
        "reservation": {
            "path": str(reservation_path),
            "sha256": sha256_file(reservation_path),
            "target_audio_paths_included": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--evaluation-reservation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace existing candidate config: {output}")
    reservation_path = args.evaluation_reservation.expanduser().resolve()
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    if not isinstance(reservation, dict):
        raise ValueError("evaluation reservation root must be an object")
    store = CaptureStore(args.data_dir.expanduser().resolve())
    project = store.get_project(args.project_id)
    config = build_candidate_config(
        project=project,
        captures=store.list_captures(args.project_id),
        reservation=reservation,
        reservation_path=reservation_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "candidate_count": len(config["candidate_vectors"]),
                "captured_count": len(config["captured_vectors"]),
                "reserved_count": len(config["reserved_evaluation_vectors"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
