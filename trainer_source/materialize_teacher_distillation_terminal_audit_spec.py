"""Materialize a leakage-safe terminal teacher-distillation audit spec.

This helper starts only after an independent member has already been selected
from internal validation.  It never renders targets, opens models, or chooses a
member.  Instead it binds the existing selection report, current native
renderer, frozen unseen DI, and already-rendered terminal teacher targets into
one spec consumable by ``create_evaluation_reservation.py``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from dessmetal_capture.wav_io import inspect_wav


LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _checked_pcm24_mono(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    info = inspect_wav(resolved)
    if (
        info["sample_rate"] != 48_000
        or info["channels"] != 1
        or info["sample_width_bits"] != 24
    ):
        raise ValueError(f"{label} must be 48 kHz mono PCM24")
    peak = float(info["peak"])
    if not math.isfinite(peak) or not 0.0 < peak < 1.0:
        raise ValueError(f"{label} is silent, non-finite, or clipped")
    return {
        "path": str(resolved),
        "sha256": str(info["sha256"]),
        "size_bytes": int(info["size_bytes"]),
        "frames": int(info["frames"]),
        "sample_rate": 48_000,
        "channels": 1,
        "sample_width_bits": 24,
        "peak": peak,
    }


def _checked_selection_report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    report = _read_json_object(resolved, "selected-member report")
    if report.get("selection_opened_hidden_evidence") is not False:
        raise ValueError(
            "selected-member report must declare selection_opened_hidden_evidence=false"
        )
    criterion = report.get("criterion")
    if criterion != "minimum best_internal_validation_esr":
        raise ValueError(
            "selected-member report must use the internal-only selection criterion"
        )
    selected_member = report.get("selected_member")
    selected_model_sha256 = report.get("selected_model_sha256")
    members = report.get("members")
    if not isinstance(selected_member, str) or not selected_member.strip():
        raise ValueError("selected-member report has no selected_member")
    if not isinstance(selected_model_sha256, str) or LOWER_SHA256.fullmatch(
        selected_model_sha256
    ) is None:
        raise ValueError("selected-member report has no valid selected_model_sha256")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError(
            "selected-member report must contain at least two independent members"
        )
    selected_records = [
        member
        for member in members
        if isinstance(member, Mapping) and member.get("label") == selected_member
    ]
    if len(selected_records) != 1:
        raise ValueError(
            "selected-member report must identify exactly one listed selected member"
        )
    if selected_records[0].get("model_sha256") != selected_model_sha256:
        raise ValueError(
            "selected-member report model hash does not match its selected member"
        )
    return report, {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "selected_member": selected_member.strip(),
        "selected_model_sha256": selected_model_sha256,
        "criterion": criterion,
        "selection_opened_hidden_evidence": False,
    }


def _checked_external_scope(overrides: Mapping[str, Any]) -> dict[str, Any]:
    scope = dict(overrides)
    role = scope.get("role")
    if (
        not isinstance(role, str)
        or not role.strip()
        or "external" not in role.lower()
        or "internal" in role.lower()
    ):
        raise ValueError(
            "external evaluation scope requires a non-internal external role"
        )
    for key in (
        "release_gate",
        "disjoint_from_training_input",
        "disjoint_from_internal_validation_input",
    ):
        if key in scope and scope[key] is not True:
            raise ValueError(f"external evaluation scope cannot disable {key}")
    if (
        "selection_opened_hidden_evidence" in scope
        and scope["selection_opened_hidden_evidence"] is not False
    ):
        raise ValueError(
            "external evaluation scope cannot claim hidden evidence was opened during "
            "selection"
        )
    scope.update(
        {
            "role": role.strip(),
            "release_gate": True,
            "disjoint_from_training_input": True,
            "disjoint_from_internal_validation_input": True,
            "selection_opened_hidden_evidence": False,
        }
    )
    return scope


def _checked_source_references(
    source_spec: Mapping[str, Any], control_order: Sequence[str]
) -> list[dict[str, Any]]:
    references = source_spec.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("internal-validation source spec has no references")
    normalized: list[dict[str, Any]] = []
    seen_setting_ids: set[str] = set()
    for index, raw in enumerate(references):
        if not isinstance(raw, Mapping):
            raise ValueError(f"source reference {index} must be an object")
        vector = raw.get("g_vector")
        setting_id = raw.get("setting_id")
        teacher_sha256 = raw.get("teacher_sha256")
        raw_path = raw.get("path")
        if not isinstance(vector, list) or len(vector) != len(control_order):
            raise ValueError(f"source reference {index} does not match control_order")
        if any(not math.isfinite(float(value)) for value in vector):
            raise ValueError(f"source reference {index} has a non-finite g_vector")
        if not isinstance(setting_id, str) or not setting_id.strip():
            raise ValueError(f"source reference {index} has no setting_id")
        setting_id = setting_id.strip()
        if setting_id in seen_setting_ids:
            raise ValueError(f"source reference setting_id is duplicated: {setting_id}")
        seen_setting_ids.add(setting_id)
        if not isinstance(teacher_sha256, str) or LOWER_SHA256.fullmatch(
            teacher_sha256
        ) is None:
            raise ValueError(f"source reference {index} has no valid teacher_sha256")
        if not isinstance(raw_path, str):
            raise ValueError(f"source reference {index} has no internal target path")
        normalized.append(
            {
                "g_vector": copy.deepcopy(vector),
                "setting_id": setting_id,
                "teacher_sha256": teacher_sha256,
                "internal_target": _checked_pcm24_mono(
                    Path(raw_path), f"internal-validation target {setting_id}"
                ),
            }
        )
    return normalized


def build_terminal_audit_spec(
    *,
    source_spec_path: Path,
    selection_report_path: Path,
    training_input_paths: Sequence[Path],
    evaluation_input_path: Path,
    target_paths: Sequence[Path],
    renderer_path: Path,
    evaluation_scope_overrides: Mapping[str, Any],
    provenance_overrides: Mapping[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build a terminal spec without rendering targets or opening a model."""

    source_path = source_spec_path.expanduser().resolve()
    source_spec = _read_json_object(source_path, "internal-validation source spec")
    control_order = source_spec.get("control_order")
    if (
        not isinstance(control_order, list)
        or not control_order
        or len(control_order) != len(set(control_order))
        or any(not isinstance(value, str) or not value for value in control_order)
    ):
        raise ValueError("source spec control_order must contain unique non-empty ids")
    source_scope = source_spec.get("evaluation_scope")
    source_role = (
        source_scope.get("role") if isinstance(source_scope, Mapping) else None
    )
    if not isinstance(source_role, str) or "internal" not in source_role.lower():
        raise ValueError("source spec must be an internal-validation reservation spec")
    source_provenance = source_spec.get("provenance")
    if not isinstance(source_provenance, Mapping):
        raise ValueError("source spec provenance must be an object")
    if source_provenance.get("source_kind") != "teacher_model_distillation":
        raise ValueError(
            "source spec must declare source_kind=teacher_model_distillation"
        )
    if source_spec.get("target_audio_available_to_acquisition") is True:
        raise ValueError("source spec violates the acquisition boundary")

    source_references = _checked_source_references(source_spec, control_order)
    if len(target_paths) != len(source_references):
        raise ValueError(
            "terminal target count does not match the internal-validation source "
            "target count"
        )
    _, selection_record = _checked_selection_report(selection_report_path)

    renderer = renderer_path.expanduser().resolve()
    if not renderer.is_file():
        raise FileNotFoundError(f"current renderer is unavailable: {renderer}")
    if not os.access(renderer, os.X_OK):
        raise ValueError(f"current renderer is not executable: {renderer}")
    renderer_record = {"path": str(renderer), "sha256": sha256_file(renderer)}

    if not training_input_paths:
        raise ValueError("at least one immutable training input is required")
    training_inputs = [
        _checked_pcm24_mono(path, f"training input {index}")
        for index, path in enumerate(training_input_paths)
    ]
    training_hashes = [record["sha256"] for record in training_inputs]
    if len(training_hashes) != len(set(training_hashes)):
        raise ValueError("training inputs must be byte-distinct")

    raw_internal_input = source_spec.get("evaluation_input")
    if not isinstance(raw_internal_input, str):
        raise ValueError(
            "source spec requires one internal-validation evaluation_input path"
        )
    internal_input = _checked_pcm24_mono(
        Path(raw_internal_input), "internal-validation input"
    )
    if internal_input["sha256"] in set(training_hashes):
        raise ValueError("internal-validation input overlaps a training input")

    evaluation_input = _checked_pcm24_mono(
        evaluation_input_path, "frozen external evaluation input"
    )
    protected_input_hashes = set(training_hashes) | {internal_input["sha256"]}
    if evaluation_input["sha256"] in protected_input_hashes:
        raise ValueError(
            "frozen external evaluation input overlaps train/internal input"
        )

    internal_target_hashes = {
        reference["internal_target"]["sha256"] for reference in source_references
    }
    terminal_references: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    seen_target_paths: set[str] = set()
    for source_reference, raw_target in zip(
        source_references, target_paths, strict=True
    ):
        setting_id = source_reference["setting_id"]
        target = _checked_pcm24_mono(raw_target, f"terminal target {setting_id}")
        if target["path"] in seen_target_paths:
            raise ValueError(f"terminal target path is duplicated: {target['path']}")
        seen_target_paths.add(target["path"])
        if target["frames"] != evaluation_input["frames"]:
            raise ValueError(
                f"terminal target {setting_id} does not match the frozen "
                "evaluation-input length"
            )
        if target["sha256"] in protected_input_hashes:
            raise ValueError(
                f"terminal target {setting_id} overlaps a train/internal input"
            )
        if target["sha256"] == evaluation_input["sha256"]:
            raise ValueError(
                f"terminal target {setting_id} is identical to the unseen DI"
            )
        if target["sha256"] in internal_target_hashes:
            raise ValueError(
                f"terminal target {setting_id} overlaps an internal-validation target"
            )
        terminal_references.append(
            {
                "g_vector": copy.deepcopy(source_reference["g_vector"]),
                "path": target["path"],
                "setting_id": setting_id,
                "teacher_sha256": source_reference["teacher_sha256"],
            }
        )
        target_records.append({"setting_id": setting_id, **target})

    scope = _checked_external_scope(evaluation_scope_overrides)
    supplied_provenance = dict(provenance_overrides or {})
    if supplied_provenance.get("source_kind", "teacher_model_distillation") != (
        "teacher_model_distillation"
    ):
        raise ValueError(
            "provenance cannot change teacher_model_distillation source_kind"
        )
    if (
        supplied_provenance.get(
            "target_generation_available_to_acquisition", False
        )
        is not False
    ):
        raise ValueError("provenance cannot expose target generation to acquisition")
    provenance = {**dict(source_provenance), **supplied_provenance}
    provenance.update(
        {
            "source_kind": "teacher_model_distillation",
            "target_generation_available_to_acquisition": False,
            "internal_validation_spec": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
            "member_selection_report": selection_record,
            "renderer": renderer_record,
            "training_inputs": training_inputs,
            "internal_validation_input": internal_input,
            "external_evaluation_input": evaluation_input,
            "terminal_audit_targets": target_records,
            "terminal_audit_target_set_sha256": canonical_json_sha256(target_records),
        }
    )
    return {
        "schema_version": 1,
        "control_order": copy.deepcopy(control_order),
        "target_audio_available_to_acquisition": False,
        "evaluation_scope": scope,
        "evaluation_input": evaluation_input["path"],
        "references": terminal_references,
        "provenance": provenance,
        "notes": notes,
    }


def write_terminal_audit_spec(spec: Mapping[str, Any], output_path: Path) -> Path:
    output = output_path.expanduser().resolve()
    if output.exists():
        raise RuntimeError(
            f"Refusing to replace existing terminal audit spec: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(spec, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return output


def _parse_json_object_argument(raw: str, label: str) -> dict[str, Any]:
    if raw.startswith("@"):
        return _read_json_object(Path(raw[1:]), label)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must decode to an object")
    return value


def _target_paths_from_args(
    args: argparse.Namespace, source_spec: Mapping[str, Any]
) -> list[Path]:
    references = source_spec.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("internal-validation source spec has no references")
    if args.target_dir is not None:
        if args.target:
            raise ValueError("use --target-dir/--target-basename or --target, not both")
        if len(args.target_basename) != len(references):
            raise ValueError(
                "ordered --target-basename count must match the source reference count"
            )
        directory = args.target_dir.expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(
                f"terminal target directory is unavailable: {directory}"
            )
        paths: list[Path] = []
        for basename in args.target_basename:
            relative = Path(basename)
            if relative.name != basename or relative.suffix.lower() != ".wav":
                raise ValueError("each --target-basename must be one WAV basename")
            paths.append(directory / relative)
        return paths

    if args.target_basename:
        raise ValueError("--target-basename requires --target-dir")
    mappings: dict[str, Path] = {}
    for raw in args.target:
        setting_id, separator, path_raw = raw.partition("=")
        setting_id = setting_id.strip()
        path_raw = path_raw.strip()
        if separator != "=" or not setting_id or not path_raw:
            raise ValueError("--target must be SETTING_ID=/absolute/target.wav")
        if setting_id in mappings:
            raise ValueError(f"duplicate --target setting id: {setting_id}")
        mappings[setting_id] = Path(path_raw)
    source_ids = []
    for reference in references:
        if not isinstance(reference, Mapping) or not isinstance(
            reference.get("setting_id"), str
        ):
            raise ValueError("every source reference must declare setting_id")
        source_ids.append(reference["setting_id"].strip())
    if set(mappings) != set(source_ids) or len(mappings) != len(source_ids):
        raise ValueError(
            "explicit --target mappings must match every source setting_id exactly"
        )
    return [mappings[setting_id] for setting_id in source_ids]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--training-input", type=Path, action="append", required=True)
    parser.add_argument("--evaluation-input", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--target-basename", action="append", default=[])
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="SETTING_ID=PATH",
        help="Explicit target mapping alternative to --target-dir/--target-basename.",
    )
    parser.add_argument(
        "--evaluation-scope-json",
        required=True,
        help="External scope JSON object, or @/path/to/object.json.",
    )
    parser.add_argument(
        "--provenance-json",
        default="{}",
        help="Additional provenance JSON object, or @/path/to/object.json.",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_spec = _read_json_object(args.source_spec, "internal-validation source spec")
    targets = _target_paths_from_args(args, source_spec)
    spec = build_terminal_audit_spec(
        source_spec_path=args.source_spec,
        selection_report_path=args.selection_report,
        training_input_paths=args.training_input,
        evaluation_input_path=args.evaluation_input,
        target_paths=targets,
        renderer_path=args.renderer,
        evaluation_scope_overrides=_parse_json_object_argument(
            args.evaluation_scope_json, "evaluation scope override"
        ),
        provenance_overrides=_parse_json_object_argument(
            args.provenance_json, "provenance override"
        ),
        notes=args.notes,
    )
    output = write_terminal_audit_spec(spec, args.output)
    print(
        json.dumps(
            {
                "status": "materialized",
                "output": str(output),
                "sha256": sha256_file(output),
                "reference_count": len(spec["references"]),
                "selected_member": spec["provenance"]["member_selection_report"][
                    "selected_member"
                ],
                "rendered_targets": False,
                "selected_member_changed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
