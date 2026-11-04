"""Import one audit-safe native disagreement batch into a persistent Trainer queue."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dessmetal_capture.domain import ControlDefinition
from dessmetal_capture.panama import parse_new_points, source_digest
from dessmetal_capture.store import CaptureStore

from materialize_conditioned_candidate_config import build_candidate_config, sha256_file


def normalized_distance(
    left: Sequence[float], right: Sequence[float], controls: Sequence[ControlDefinition]
) -> float:
    return math.sqrt(
        sum(
            ((float(a) - float(b)) / (control.maximum - control.minimum)) ** 2
            for a, b, control in zip(left, right, controls)
        )
    )


def select_batch(
    ranking: Sequence[Mapping[str, Any]],
    *,
    controls: Sequence[ControlDefinition],
    occupied: Sequence[Sequence[float]],
    maximum: int,
    minimum_distance: float,
) -> list[dict[str, Any]]:
    if not 1 <= maximum <= 64:
        raise ValueError("maximum must be between 1 and 64")
    if not math.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError("minimum_distance must be finite and non-negative")
    selected: list[dict[str, Any]] = []
    selected_vectors: list[list[float]] = []
    for entry in ranking:
        if not isinstance(entry, Mapping):
            raise ValueError("ranking entries must be objects")
        vector = entry.get("g_vector")
        if not isinstance(vector, list) or len(vector) != len(controls):
            raise ValueError("ranking vector does not match the project controls")
        values = [float(value) for value in vector]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("ranking vector contains a non-finite value")
        if any(
            normalized_distance(values, other, controls) < minimum_distance
            for other in [*occupied, *selected_vectors]
        ):
            continue
        selected.append(
            {
                "index": int(entry["index"]),
                "g_vector": values,
                "disagreement_score": float(entry["disagreement_score"]),
            }
        )
        selected_vectors.append(values)
        if len(selected) == maximum:
            break
    if not selected:
        raise ValueError("no ranked point satisfies the minimum-distance contract")
    return selected


def import_batch(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.selection_report.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("selection report root must be an object")
    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping) or any(
        protocol.get(flag) is not True
        for flag in (
            "reserved_vectors_excluded_before_render",
            "captured_vectors_excluded_before_render",
        )
    ) or protocol.get("target_audio_read") is not False:
        raise ValueError("selection report is not audit-safe")
    candidate_path = Path(str(report.get("candidate_config_path", ""))).expanduser().resolve()
    if not candidate_path.is_file() or report.get("candidate_config_sha256") != sha256_file(candidate_path):
        raise ValueError("selection report candidate-config evidence is missing or changed")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(candidate, dict):
        raise ValueError("candidate config root must be an object")

    store = CaptureStore(args.data_dir.expanduser().resolve())
    project = store.get_project(args.project_id)
    if candidate.get("project_id") != project["id"]:
        raise ValueError("candidate config belongs to another project")
    reservation_record = candidate.get("reservation")
    if not isinstance(reservation_record, Mapping):
        raise ValueError("candidate config has no reservation evidence")
    reservation_path = Path(str(reservation_record.get("path", ""))).expanduser().resolve()
    if (
        not reservation_path.is_file()
        or reservation_record.get("sha256") != sha256_file(reservation_path)
    ):
        raise ValueError("candidate config reservation evidence is missing or changed")
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    current = build_candidate_config(
        project=project,
        captures=store.list_captures(project["id"]),
        reservation=reservation,
        reservation_path=reservation_path,
    )
    for field in (
        "control_order",
        "candidate_vectors",
        "captured_vectors",
        "reserved_evaluation_vectors",
        "reservation",
    ):
        if current[field] != candidate.get(field):
            raise ValueError(f"candidate config is stale: {field} changed")

    controls = [ControlDefinition.from_mapping(control) for control in project["controls"]]
    occupied = [
        *candidate["captured_vectors"],
        *candidate["reserved_evaluation_vectors"],
    ]
    ranking = report.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("selection report has no ranking")
    points = select_batch(
        ranking,
        controls=controls,
        occupied=occupied,
        maximum=args.maximum,
        minimum_distance=args.minimum_distance,
    )
    suggestions = parse_new_points(
        points,
        controls=controls,
        control_order=candidate["control_order"],
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = store.backup(
        store.data_dir / "backups" / f"capture-before-disagreement-round-{args.round_index}-{timestamp}.sqlite3"
    )
    queued = store.enqueue_panama_suggestions(
        project["id"],
        round_index=args.round_index,
        source_sha256=source_digest(points),
        control_order=candidate["control_order"],
        suggestions=suggestions,
    )
    return {
        "status": "queued",
        "project_id": project["id"],
        "round_index": args.round_index,
        "queued_count": sum(item["status"] == "queued" for item in queued),
        "points": points,
        "selection_report": str(report_path),
        "selection_report_sha256": sha256_file(report_path),
        "database_backup": str(backup),
        "database_backup_sha256": sha256_file(backup),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--maximum", type=int, default=8)
    parser.add_argument("--minimum-distance", type=float, default=0.35)
    return parser


def main() -> None:
    print(json.dumps(import_batch(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
