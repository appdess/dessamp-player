"""Import hash-bound teacher renders into a persistent conditioned project.

The teacher models and rendered WAVs remain outside Git.  This importer does
not render or normalize audio: it verifies an explicit plan, records one
ordinary capture-companion recommendation/capture for every non-audit setting,
and preserves teacher-model provenance in SQLite.  Frozen audit teachers must
be handled through an evaluation reservation instead of this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from dessmetal_capture.domain import settings_key
from dessmetal_capture.manifests import inspect_nam_model
from dessmetal_capture.store import CaptureStore
from dessmetal_capture.wav_io import inspect_wav


MAX_PLAN_BYTES = 2 * 1024 * 1024
ALLOWED_ROLES = {"train", "validation"}
VALIDATION_STRATEGIES = {
    "capture_holdout",
    "separate_disjoint_di_reservation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_plan(path: Path) -> dict[str, Any]:
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_PLAN_BYTES:
        raise ValueError(f"teacher import plan is missing, empty, or oversized: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("teacher import plan must be a schema-version-1 object")
    return payload


def _resolve_inside(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"{label} escapes the experiment root")
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {candidate}")
    return candidate


def _required_hash(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def validate_plan_entries(
    plan: Mapping[str, Any],
    *,
    project: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metadata = project.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("source_kind") != "teacher_model_distillation":
        raise ValueError("project is not declared as teacher_model_distillation")
    if metadata.get("physical_amplifier") is not False:
        raise ValueError("a teacher-distillation project must not claim direct physical-amplifier capture")
    if plan.get("source_kind") != "teacher_model_distillation":
        raise ValueError("plan source_kind must be teacher_model_distillation")
    if plan.get("project_id") != project.get("id"):
        raise ValueError("teacher import plan project_id does not match the persistent project")

    root_raw = plan.get("experiment_root")
    if not isinstance(root_raw, str) or not root_raw.strip():
        raise ValueError("teacher import plan requires experiment_root")
    root = Path(root_raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"experiment root is unavailable: {root}")

    input_path = _resolve_inside(root, plan.get("shared_input_relative"), "shared input")
    expected_input_hash = _required_hash(plan.get("shared_input_sha256"), "shared input hash")
    if sha256_file(input_path) != expected_input_hash:
        raise ValueError("shared input SHA-256 no longer matches the import plan")
    project_input = project.get("input_signal_info")
    if not isinstance(project_input, Mapping) or project_input.get("sha256") != expected_input_hash:
        raise ValueError("persistent project input does not match the teacher import plan")

    validation_strategy = str(plan.get("validation_strategy", "capture_holdout"))
    if validation_strategy not in VALIDATION_STRATEGIES:
        raise ValueError(
            "teacher import validation_strategy must be capture_holdout or "
            "separate_disjoint_di_reservation"
        )
    entries_raw = plan.get("entries")
    if not isinstance(entries_raw, list) or len(entries_raw) < 2:
        raise ValueError("teacher import plan requires at least two entries")
    controls = list(project.get("controls") or [])
    control_ids = [str(control["id"]) for control in controls]
    candidate_keys = {
        settings_key({control_id: float(point[control_id]) for control_id in control_ids})
        for point in project.get("candidate_points") or []
    }
    roles: list[str] = []
    seen_settings: set[str] = set()
    seen_capture_ids: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for raw in entries_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("teacher import entries must be objects")
        role = raw.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError("teacher import roles are train or validation; frozen audit targets stay separate")
        roles.append(str(role))
        capture_id = str(raw.get("capture_id", "")).strip()
        if not capture_id or capture_id in seen_capture_ids:
            raise ValueError("teacher capture IDs must be non-empty and unique")
        seen_capture_ids.add(capture_id)
        settings = {control_id: float(raw[control_id]) for control_id in control_ids}
        if any(not math.isfinite(value) for value in settings.values()):
            raise ValueError("teacher settings must be finite")
        for control, value in zip(controls, settings.values(), strict=True):
            if value < float(control["minimum"]) or value > float(control["maximum"]):
                raise ValueError(f"teacher setting for {control['id']!r} is outside the project range")
        key = settings_key(settings)
        if key in seen_settings:
            raise ValueError(f"duplicate teacher setting: {key}")
        if key not in candidate_keys:
            raise ValueError("teacher setting is not in the reviewed project candidate set")
        seen_settings.add(key)

        teacher_path = _resolve_inside(root, raw.get("teacher_relative"), "teacher model")
        teacher_hash = _required_hash(raw.get("teacher_sha256"), "teacher model hash")
        if sha256_file(teacher_path) != teacher_hash:
            raise ValueError(f"teacher model SHA-256 mismatch for {capture_id}")
        teacher_metadata = inspect_nam_model(teacher_path)
        if teacher_metadata["sample_rate"] != int(project["sample_rate"]):
            raise ValueError(f"teacher model sample rate mismatch for {capture_id}")
        if teacher_metadata["global_condition_size"] != 0:
            raise ValueError(f"teacher {capture_id} is already conditioned; expected a snapshot model")

        render_path = _resolve_inside(root, raw.get("render_relative"), "teacher render")
        render_hash = _required_hash(raw.get("render_sha256"), "teacher render hash")
        if sha256_file(render_path) != render_hash:
            raise ValueError(f"teacher render SHA-256 mismatch for {capture_id}")
        wav_info = inspect_wav(render_path)
        if (
            wav_info["sample_rate"] != int(project["sample_rate"])
            or wav_info["channels"] != 1
            or wav_info["sample_width_bits"] != 24
            or wav_info["frames"] != int(project_input["frames"])
        ):
            raise ValueError(f"teacher render format/length mismatch for {capture_id}")
        if not math.isfinite(float(wav_info["peak"])) or not 0.0 < float(wav_info["peak"]) < 1.0:
            raise ValueError(f"teacher render is silent, non-finite, or clipped for {capture_id}")
        resolved.append(
            {
                "capture_id": capture_id,
                "gain_label": str(raw.get("gain_label", "")),
                "role": role,
                "settings": settings,
                "settings_key": key,
                "teacher_path": teacher_path,
                "teacher_sha256": teacher_hash,
                "teacher_metadata": teacher_metadata,
                "render_path": render_path,
                "render_sha256": render_hash,
                "wav_info": wav_info,
            }
        )
    validation_count = roles.count("validation")
    if validation_strategy == "capture_holdout" and validation_count != 1:
        raise ValueError("teacher import plan must contain exactly one internal-validation setting")
    if validation_strategy == "separate_disjoint_di_reservation" and validation_count != 0:
        raise ValueError(
            "a separate disjoint-DI reservation requires every imported teacher capture to be train"
        )
    if seen_settings != candidate_keys:
        raise ValueError("teacher import entries must cover the reviewed non-audit candidate set exactly")
    return resolved


def import_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.expanduser().resolve()
    plan = read_plan(plan_path)
    store = CaptureStore(args.data_dir.expanduser().resolve())
    project_id = str(plan.get("project_id", ""))
    project = store.get_project(project_id)
    entries = validate_plan_entries(plan, project=project)
    if store.list_captures(project_id):
        raise RuntimeError("persistent project already contains captures; refusing a duplicate teacher import")
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise RuntimeError(f"Refusing to replace teacher import report: {output_path}")
    backup_path = store.data_dir / "backups" / "capture-before-teacher-import.sqlite3"
    if backup_path.exists():
        raise RuntimeError(f"Refusing to replace teacher import backup: {backup_path}")
    store.backup(backup_path)

    pending = {entry["settings_key"]: entry for entry in entries}
    imported: list[dict[str, Any]] = []
    while pending:
        recommendation = store.create_recommendation(project_id)
        key = settings_key(recommendation["settings"])
        entry = pending.pop(key, None)
        if entry is None:
            raise RuntimeError("capture companion recommended a setting outside the hash-bound teacher plan")
        capture = store.add_capture(
            project_id=project_id,
            recommendation_id=recommendation["id"],
            source="teacher_model_distillation",
            wav_path=entry["render_path"],
            wav_info=entry["wav_info"],
            quality_status="pass",
            validation={
                "status": "passed",
                "source_kind": "teacher_model_distillation",
                "role": entry["role"],
                "gain_label": entry["gain_label"],
                "shared_input_sha256": plan["shared_input_sha256"],
                "teacher_model": {
                    "path": str(entry["teacher_path"]),
                    "sha256": entry["teacher_sha256"],
                    "architecture": entry["teacher_metadata"]["architecture"],
                    "runtime_quality": entry["teacher_metadata"]["runtime_quality"],
                },
                "render": {
                    "path": str(entry["render_path"]),
                    "sha256": entry["render_sha256"],
                    "per_teacher_normalization": False,
                    "per_teacher_alignment": False,
                },
                "plan_path": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
            },
            notes=f"{entry['role']} teacher at creator gain label {entry['gain_label']}",
            capture_id=entry["capture_id"],
        )
        imported.append(
            {
                "capture_id": capture["id"],
                "role": entry["role"],
                "gain_label": entry["gain_label"],
                "settings": entry["settings"],
                "wav_sha256": capture["wav_sha256"],
                "teacher_sha256": entry["teacher_sha256"],
            }
        )
    validation_id = next(
        (item["capture_id"] for item in imported if item["role"] == "validation"),
        None,
    )
    result = {
        "status": "imported",
        "source_kind": "teacher_model_distillation",
        "project_id": project_id,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "capture_count": len(imported),
        "train_count": sum(item["role"] == "train" for item in imported),
        "validation_strategy": str(plan.get("validation_strategy", "capture_holdout")),
        "validation_capture_id": validation_id,
        "captures": imported,
        "database_backup": str(backup_path),
        "database_backup_sha256": sha256_file(backup_path),
        "frozen_audit_imported": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    result = import_plan(build_parser().parse_args())
    print(json.dumps({key: result[key] for key in ("status", "project_id", "capture_count", "validation_capture_id")}, sort_keys=True))


if __name__ == "__main__":
    main()
