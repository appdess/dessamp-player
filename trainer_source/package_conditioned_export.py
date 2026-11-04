"""Select one externally evaluated conditioned export and package it as .dess.

The arithmetic ensemble is an audit/acquisition tool.  This command binds one
member's byte hash to its fixed external-evaluation record, preserves a SQLite
backup before importing the model into the persistent Trainer, and labels any
failed numerical gate as workflow validation only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dessmetal_capture.dess_package import create_dess_package
from dessmetal_capture.manifests import import_model_artifact
from dessmetal_capture.store import CaptureStore
from dessmetal_player.artifact import PlayerArtifact

from run_fortin_conditioned_active_learning import (
    read_json,
    sha256_file,
    write_json,
)


def _finite_metric(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


def _member_contract(
    contract: Mapping[str, Any],
    *,
    mean_esr: float,
    max_esr: float,
    label: str,
) -> dict[str, Any]:
    mean_gate = _finite_metric(contract.get("required_mean_esr_lte"), f"{label} mean ESR gate")
    max_gate = _finite_metric(contract.get("required_max_esr_lte"), f"{label} max ESR gate")
    return {
        "passed": mean_esr <= mean_gate and max_esr <= max_gate,
        "required_mean_esr_lte": mean_gate,
        "required_max_esr_lte": max_gate,
    }


def project_provenance_fields(
    state: Mapping[str, Any], *, physical_amplifier: bool
) -> dict[str, Any]:
    """Preserve the persistent project's source and release boundary."""

    raw_source_kind = state.get("source_kind")
    if raw_source_kind is None:
        source_kind = "physical_amplifier" if physical_amplifier else "amp_simulator"
    elif not isinstance(raw_source_kind, str) or not raw_source_kind.strip():
        raise ValueError("project metadata source_kind must be a non-empty string")
    else:
        source_kind = raw_source_kind.strip()

    raw_validation_only = state.get("workflow_validation_only", False)
    if not isinstance(raw_validation_only, bool):
        raise ValueError("project metadata workflow_validation_only must be boolean")

    raw_source_provenance = state.get("source_provenance")
    if raw_source_provenance is None and isinstance(state.get("source"), Mapping):
        # Persistent capture projects historically call this object ``source``;
        # materialized cloud configs expose the same object as source_provenance.
        raw_source_provenance = state["source"]
    if raw_source_provenance is not None and not isinstance(raw_source_provenance, Mapping):
        raise ValueError("project metadata source_provenance must be an object")

    fields: dict[str, Any] = {
        "purpose": (
            "teacher-model-distillation-private-validation"
            if source_kind == "teacher_model_distillation"
            else "conditioned-cloud-quality-validation"
        ),
        "source_kind": source_kind,
        "project_workflow_validation_only": raw_validation_only,
    }
    if raw_source_provenance is not None:
        fields["source_provenance"] = dict(raw_source_provenance)
    return fields


def checked_head_shell_style(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("head shell style must be a string")
    style = value.strip()
    if re.fullmatch(r"[a-z0-9-]{1,64}", style) is None:
        raise ValueError("head shell style must be a short lowercase presentation token")
    return style


def checked_product_token(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    token = value.strip()
    if token and re.fullmatch(r"[a-z0-9-]{1,64}", token) is None:
        raise ValueError(f"{field} must be an optional short lowercase presentation token")
    return token


def checked_control_label_overrides(values: Sequence[str]) -> dict[str, str]:
    """Parse presentation-only ``control_id=Label`` overrides.

    The model-vector IDs and indices remain authoritative.  This only lets a
    private/listening package replace a capture-oriented label such as
    ``Gain Index`` with a concise product label such as ``Gain``.
    """

    overrides: dict[str, str] = {}
    for raw_value in values:
        if not isinstance(raw_value, str):
            raise ValueError("control label overrides must be strings")
        control_id, separator, label = raw_value.partition("=")
        control_id = control_id.strip()
        label = label.strip()
        if (
            separator != "="
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", control_id) is None
            or not label
            or len(label) > 64
            or any(ord(character) < 32 for character in label)
        ):
            raise ValueError("control label override must be CONTROL_ID=LABEL with a short label")
        if control_id in overrides:
            raise ValueError(f"control label override is duplicated for {control_id!r}")
        overrides[control_id] = label
    return overrides


def select_member_evidence(
    report: Mapping[str, Any],
    *,
    member: str,
    model_sha256: str,
) -> dict[str, Any]:
    models = report.get("models")
    individual = report.get("individual")
    acceptance = report.get("acceptance")
    if not isinstance(models, Mapping) or not isinstance(individual, Mapping):
        raise ValueError("evaluation report has no model/member records")
    if not isinstance(acceptance, Mapping):
        raise ValueError("evaluation report has no acceptance contract")
    model_record = models.get(member)
    member_record = individual.get(member)
    if not isinstance(model_record, Mapping) or not isinstance(member_record, Mapping):
        raise ValueError(f"evaluation report has no member {member!r}")
    if model_record.get("sha256") != model_sha256:
        raise ValueError("selected model SHA-256 does not match the evaluation report")
    summary = member_record.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("selected member has no external metric summary")
    mean_esr = _finite_metric(summary.get("mean_esr"), "mean_esr")
    max_esr = _finite_metric(summary.get("max_esr"), "max_esr")
    acceptable = _member_contract(
        acceptance,
        mean_esr=mean_esr,
        max_esr=max_esr,
        label="acceptance",
    )
    target_source = report.get("quality_target")
    target = _member_contract(
        target_source if isinstance(target_source, Mapping) else acceptance,
        mean_esr=mean_esr,
        max_esr=max_esr,
        label="quality target",
    )
    preferred_source = report.get("preferred")
    preferred = (
        _member_contract(
            preferred_source,
            mean_esr=mean_esr,
            max_esr=max_esr,
            label="preferred",
        )
        if isinstance(preferred_source, Mapping)
        else None
    )
    tier = (
        "preferred"
        if preferred is not None and preferred["passed"]
        else "project-target"
        if target["passed"]
        else "nam-normal"
        if acceptable["passed"]
        else "failed"
    )
    return {
        "member": member,
        "model": dict(model_record),
        "summary": dict(summary),
        "required_mean_esr_lte": acceptable["required_mean_esr_lte"],
        "required_max_esr_lte": acceptable["required_max_esr_lte"],
        "passed": acceptable["passed"],
        "quality_tier": tier,
        "quality_target": target,
        "preferred": preferred,
    }


def package(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model.expanduser().resolve()
    evaluation_path = args.evaluation_report.expanduser().resolve()
    loader_path = args.loader.expanduser().resolve()
    manifest_loader_path = args.manifest_loader.expanduser().resolve()
    renderer_path = args.renderer.expanduser().resolve()
    for path, label in (
        (model_path, "model"),
        (evaluation_path, "evaluation report"),
        (loader_path, "loader"),
        (manifest_loader_path, "manifest loader"),
        (renderer_path, "renderer"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is unavailable: {path}")

    report = read_json(evaluation_path)
    evidence = select_member_evidence(
        report,
        member=args.member,
        model_sha256=sha256_file(model_path),
    )
    if not evidence["passed"] and not args.allow_failed_gate:
        raise RuntimeError(
            "selected member fails the external ESR gate; pass --allow-failed-gate only for a "
            "workflow-validation listening artifact"
        )
    package_filename = args.package_filename.strip()
    package_relative = Path(package_filename)
    if (
        not package_filename
        or package_relative.name != package_filename
        or package_relative.suffix.lower() != ".dess"
    ):
        raise ValueError("--package-filename must be one .dess basename")
    output_report = args.output.expanduser().resolve()
    if output_report.exists():
        raise RuntimeError(f"Refusing to replace existing package report: {output_report}")

    if args.run_root is not None:
        run_root = args.run_root.expanduser().resolve()
        state = read_json(run_root / "00_WORKFLOW" / "run-state.json")
        store = CaptureStore(run_root / "04_TRAINER_DATA")
        project = store.get_project(state["project_id"])
        control_order = list(state["control_order"])
        defaults = dict(state["base_settings"])
        uniform_safety_trim_db = state.get("uniform_safety_trim_db")
    else:
        if not args.project_id:
            raise ValueError("--project-id is required with --data-dir")
        store = CaptureStore(args.data_dir.expanduser().resolve())
        project = store.get_project(args.project_id)
        state = project.get("metadata") or {}
        control_order = [control["id"] for control in project["controls"]]
        configured_defaults = state.get("base_settings")
        if configured_defaults is None:
            defaults = {
                control["id"]: (float(control["minimum"]) + float(control["maximum"])) / 2.0
                for control in project["controls"]
            }
        elif not isinstance(configured_defaults, Mapping):
            raise ValueError("project metadata base_settings must be an object")
        else:
            defaults = {str(key): float(value) for key, value in configured_defaults.items()}
        uniform_safety_trim_db = state.get("uniform_safety_trim_db")
    declared_physical = state.get("physical_amplifier")
    if declared_physical is not None and bool(declared_physical) != bool(args.physical_amplifier):
        raise ValueError(
            "packaging provenance conflicts with the persistent project's physical_amplifier declaration"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = (
        store.data_dir
        / "backups"
        / f"capture-before-model-import-{args.member}-{timestamp}.sqlite3"
    )
    store.backup(backup_path)
    project_provenance = project_provenance_fields(
        state, physical_amplifier=bool(args.physical_amplifier)
    )
    provenance = {
        **project_provenance,
        "workflow_validation_only": (
            project_provenance["project_workflow_validation_only"]
            or not evidence["passed"]
        ),
        "control_order": control_order,
        "cloud_run_id": args.cloud_run_id,
        "selected_member": args.member,
        "external_evaluation_report": str(evaluation_path),
        "external_evaluation_report_sha256": sha256_file(evaluation_path),
        "external_member_metrics": evidence["summary"],
        "external_gate": {
            "passed": evidence["passed"],
            "required_mean_esr_lte": evidence["required_mean_esr_lte"],
            "required_max_esr_lte": evidence["required_max_esr_lte"],
            "quality_tier": evidence["quality_tier"],
            "quality_target": evidence["quality_target"],
            "preferred": evidence["preferred"],
        },
        "ensemble_metrics": report.get("ensemble", {}).get("summary"),
        "uniform_safety_trim_db": uniform_safety_trim_db,
        "physical_amplifier": bool(args.physical_amplifier),
        "human_listening_passed": False,
    }
    artifact = import_model_artifact(
        store,
        project_id=project["id"],
        source_model_path=model_path,
        name=args.name,
        provenance=provenance,
        loader_path=loader_path,
    )
    if artifact["validation_status"] != "valid":
        raise RuntimeError(f"current NAM Core rejected the imported model: {artifact}")

    manifest_path = Path(artifact["manifest_path"])
    manifest = read_json(manifest_path)
    head_shell_style = checked_head_shell_style(args.head_shell_style)
    product_family = checked_product_token(args.product_family, "product family")
    product_variant = checked_product_token(args.product_variant, "product variant")
    if product_variant and not product_family:
        raise ValueError("product variant requires product family")
    control_label_overrides = checked_control_label_overrides(args.control_label)
    manifest_control_ids = {str(control["id"]) for control in manifest["controls"]}
    unknown_label_ids = sorted(set(control_label_overrides) - manifest_control_ids)
    if unknown_label_ids:
        raise ValueError(
            "control label override references unknown model control(s): "
            + ", ".join(unknown_label_ids)
        )
    manifest["presentation"] = {"head_shell": head_shell_style}
    if product_family:
        manifest["presentation"]["product_family"] = product_family
    if product_variant:
        manifest["presentation"]["product_variant"] = product_variant
    for control in manifest["controls"]:
        if control["id"] not in defaults:
            raise RuntimeError(f"run state has no default for {control['id']!r}")
        control["default_value"] = float(defaults[control["id"]])
        if control["id"] in control_label_overrides:
            control["label"] = control_label_overrides[control["id"]]
    write_json(manifest_path, manifest)
    canonical_package = create_dess_package(manifest_path, manifest_path.parent / "model.dess")
    friendly_path = manifest_path.parent / package_filename
    if friendly_path.exists():
        raise RuntimeError(f"Refusing to replace existing friendly package: {friendly_path}")
    friendly_package = create_dess_package(manifest_path, friendly_path)

    manifest_loader = subprocess.run(
        [str(manifest_loader_path), str(friendly_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if manifest_loader.returncode != 0:
        detail = (manifest_loader.stderr or manifest_loader.stdout or "manifest loader failed")[-4_000:]
        raise RuntimeError(detail)
    player = PlayerArtifact(friendly_path, renderer_path)
    result = {
        "status": "accepted" if evidence["passed"] else "workflow-validation-only",
        "artifact_id": artifact["id"],
        "name": args.name,
        "selected_member": args.member,
        "model_path": artifact["model_path"],
        "model_sha256": artifact["model_sha256"],
        "manifest_path": str(manifest_path),
        "canonical_package": canonical_package,
        "package": friendly_package,
        "player_description": player.describe(),
        "presentation": dict(manifest["presentation"]),
        "control_label_overrides": control_label_overrides,
        "native_model_loader": artifact["validation_message"],
        "native_manifest_loader": (manifest_loader.stdout + manifest_loader.stderr).strip(),
        "external_evidence": evidence,
        "database_backup": str(backup_path),
        "database_backup_sha256": sha256_file(backup_path),
        "shipped_model_replaced": False,
    }
    write_json(output_report, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-root",
        type=Path,
        help="Legacy regression-harness root",
    )
    source.add_argument(
        "--data-dir",
        type=Path,
        help="Generic persistent capture-companion data directory",
    )
    parser.add_argument("--project-id", help="Required with --data-dir")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--package-filename", required=True)
    parser.add_argument(
        "--head-shell-style",
        default="studio-black-v1",
        choices=(
            "studio-black-v1",
            "dual-red-modern-v1",
            "forge-streamlined-v1",
            "fire-chrome-v1",
        ),
        help="Select one original shell compiled into the player; no image path is accepted.",
    )
    parser.add_argument(
        "--product-family",
        default="",
        help="Optional bounded catalog family token used to group separately packaged product voices.",
    )
    parser.add_argument(
        "--product-variant",
        default="",
        help="Optional bounded voice token; requires --product-family.",
    )
    parser.add_argument(
        "--control-label",
        action="append",
        default=[],
        metavar="CONTROL_ID=LABEL",
        help=(
            "Override one code-rendered control label without changing its manifest ID, "
            "model index, range, or default; may be repeated."
        ),
    )
    parser.add_argument("--cloud-run-id", required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--manifest-loader", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-failed-gate", action="store_true")
    provenance = parser.add_mutually_exclusive_group(required=True)
    provenance.add_argument(
        "--physical-amplifier",
        action="store_true",
        help="Record that the selected captures came from a physical amplifier",
    )
    provenance.add_argument(
        "--amp-simulator",
        action="store_true",
        help="Record that the selected captures came from an amp simulator/plugin",
    )
    return parser


def main() -> None:
    result = package(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "member": result["selected_member"],
                "package": result["package"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
