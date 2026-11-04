#!/usr/bin/env python3
"""Assemble the gated private FIRE -> FORGE -> REDLINE listening catalogue.

This helper is deliberately narrower than the generic Xcode resource copier.
It accepts four already-created ``.dess`` packages, verifies their immutable
hashes and product contracts, requires the accepted Fire packaging report,
loads every package through one exact current-Core executable, and atomically
materializes an external private catalogue.  It never writes model bytes into
the repository and never grants signing, listening, or redistribution approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SOURCE = REPOSITORY_ROOT / "capture_companion" / "src"
sys.path.insert(0, str(CAPTURE_SOURCE))

from dessmetal_capture.dess_package import read_dess_package  # noqa: E402


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MAX_EVIDENCE_JSON_BYTES = 4 * 1024 * 1024
FIRE_CONTROLS = ("lead_gain", "bass", "middle", "treble", "presence")


@dataclass(frozen=True)
class SlotContract:
    key: str
    filename: str
    display_name: str
    control_ids: tuple[str, ...]
    head_shells: frozenset[str]
    product_family: str | None
    product_variant: str | None
    requires_morph: bool


SLOT_CONTRACTS = (
    SlotContract(
        key="fire",
        filename="01-fire.dess",
        display_name="DESS FIRE",
        control_ids=FIRE_CONTROLS,
        head_shells=frozenset({"fire-chrome-v1"}),
        product_family=None,
        product_variant="",
        requires_morph=False,
    ),
    SlotContract(
        key="forge",
        filename="02-forge.dess",
        display_name="DESS FORGE",
        control_ids=FIRE_CONTROLS,
        head_shells=frozenset({"studio-black-v1", "forge-streamlined-v1"}),
        product_family=None,
        product_variant="",
        requires_morph=True,
    ),
    SlotContract(
        key="redline_us",
        filename="03-redline-us.dess",
        display_name="DESS REDLINE",
        control_ids=("gain_index",),
        head_shells=frozenset({"dual-red-modern-v1"}),
        product_family="redline",
        product_variant="us",
        requires_morph=False,
    ),
    SlotContract(
        key="redline_brit",
        filename="04-redline-brit.dess",
        display_name="DESS REDLINE",
        control_ids=("gain_index",),
        head_shells=frozenset({"dual-red-modern-v1"}),
        product_family="redline",
        product_variant="brit",
        requires_morph=False,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_sha256(value: str, label: str) -> str:
    digest = value.strip().lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return digest


def _checked_external_file(value: Path, label: str, suffix: str | None = None) -> Path:
    unresolved = value.expanduser()
    if unresolved.is_symlink():
        raise RuntimeError(f"{label} must not be a symbolic link")
    path = unresolved.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {path}")
    if suffix is not None and path.suffix.lower() != suffix:
        raise ValueError(f"{label} must use the {suffix} extension")
    if path == REPOSITORY_ROOT or path.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError(f"{label} must remain outside the repository")
    return path


def _checked_output_directory(value: Path) -> Path:
    unresolved = value.expanduser()
    if unresolved.exists() or unresolved.is_symlink():
        raise RuntimeError(f"Refusing to replace an existing catalogue path: {unresolved}")
    path = unresolved.resolve()
    if path == REPOSITORY_ROOT or path.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError("Private catalogue output must remain outside the repository")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError("Private catalogue output parent must be one existing real directory")
    return path


def _read_bounded_json(path: Path, label: str) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_EVIDENCE_JSON_BYTES:
        raise RuntimeError(f"{label} is empty or exceeds the 4 MiB evidence limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object")
    return value


def _controls(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    controls = manifest.get("controls")
    if not isinstance(controls, list) or any(not isinstance(control, Mapping) for control in controls):
        raise RuntimeError("Package controls must be an array of objects")
    identifiers: list[str] = []
    indices: list[int] = []
    for control in controls:
        control_id = control.get("id")
        model_index = control.get("model_parameter_index")
        if not isinstance(control_id, str) or not control_id:
            raise RuntimeError("Package control IDs must be non-empty strings")
        if isinstance(model_index, bool) or not isinstance(model_index, int):
            raise RuntimeError("Package model parameter indices must be integers")
        identifiers.append(control_id)
        indices.append(model_index)
    if indices != list(range(len(indices))):
        raise RuntimeError("Package model parameter indices do not preserve manifest order")
    return tuple(identifiers)


def _fixed_values(manifest: Mapping[str, Any]) -> dict[str, float]:
    raw = manifest.get("fixed_controls", [])
    if not isinstance(raw, list) or any(not isinstance(control, Mapping) for control in raw):
        raise RuntimeError("Package fixed_controls must be an array of objects")
    values: dict[str, float] = {}
    for control in raw:
        control_id = control.get("id")
        value = control.get("value")
        if not isinstance(control_id, str) or not control_id:
            raise RuntimeError("Package fixed control IDs must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("Package fixed control values must be numeric")
        if control_id in values:
            raise RuntimeError(f"Package repeats fixed control {control_id!r}")
        values[control_id] = float(value)
    return values


def _validate_manifest(contract: SlotContract, manifest: Mapping[str, Any]) -> dict[str, Any]:
    name = manifest.get("name")
    if not isinstance(name, str) or not name.strip().upper().startswith(contract.display_name):
        raise RuntimeError(f"{contract.key} package is not named {contract.display_name}")
    if _controls(manifest) != contract.control_ids:
        raise RuntimeError(f"{contract.key} package control order does not match the catalogue contract")

    model = manifest.get("model")
    if not isinstance(model, Mapping) or model.get("global_condition_size") != len(contract.control_ids):
        raise RuntimeError(f"{contract.key} package global condition size is invalid")

    presentation = manifest.get("presentation")
    if not isinstance(presentation, Mapping):
        raise RuntimeError(f"{contract.key} package has no presentation contract")
    head_shell = presentation.get("head_shell")
    if head_shell not in contract.head_shells:
        raise RuntimeError(f"{contract.key} package head shell is invalid")
    product_family = presentation.get("product_family", "")
    product_variant = presentation.get("product_variant", "")
    if not isinstance(product_family, str) or not isinstance(product_variant, str):
        raise RuntimeError(f"{contract.key} product family/variant must be strings")
    if contract.product_family is None:
        allowed_families = {"", contract.key}
        if contract.key == "redline_us" or contract.key == "redline_brit":
            allowed_families.add("redline")
        if product_family not in allowed_families:
            raise RuntimeError(f"{contract.key} product family is invalid")
    elif product_family != contract.product_family:
        raise RuntimeError(f"{contract.key} product family is invalid")
    if product_variant != contract.product_variant:
        raise RuntimeError(f"{contract.key} product variant is invalid")

    morph = manifest.get("morph")
    if contract.requires_morph:
        if not isinstance(morph, Mapping) or morph.get("mode") != "dual_model_linear_v1":
            raise RuntimeError("Forge package must contain the accepted dual-model Morph contract")
        morph_control = morph.get("control")
        if not isinstance(morph_control, Mapping) or morph_control.get("id") != "morph":
            raise RuntimeError("Forge package Morph control is invalid")
    elif morph is not None:
        raise RuntimeError(f"{contract.key} package must not contain a Morph model")

    fixed = _fixed_values(manifest)
    if contract.key in {"fire", "forge"}:
        for control_id, expected in (("bright", 1.0), ("bottom", 0.0), ("mid_boost", 0.0)):
            if fixed.get(control_id) != expected:
                raise RuntimeError(f"{contract.key} package fixed {control_id} value is invalid")
    if contract.key == "fire" and fixed.get("lead_volume") != 3.5:
        raise RuntimeError("Fire package must freeze Lead Volume at 3.5")

    return {
        "name": name,
        "control_ids": list(contract.control_ids),
        "presentation": {
            "head_shell": head_shell,
            "product_family": product_family,
            "product_variant": product_variant,
        },
        "fixed_controls": fixed,
        "morph": contract.requires_morph,
    }


def _validate_with_core(loader: Path, package: Path, contract: SlotContract) -> dict[str, Any]:
    result = subprocess.run(
        [str(loader), str(package)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "current Core rejected the package")[-4_000:]
        raise RuntimeError(f"{contract.key} current-Core validation failed: {detail}")
    try:
        output = json.loads(result.stdout.strip())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{contract.key} current-Core output is not JSON") from error
    if not isinstance(output, Mapping) or output.get("status") != "valid":
        raise RuntimeError(f"{contract.key} current-Core result is not valid")
    expected_parameter_count = len(contract.control_ids) + (1 if contract.requires_morph else 0)
    if output.get("parameter_count") != expected_parameter_count:
        raise RuntimeError(f"{contract.key} current-Core parameter count is invalid")
    if bool(output.get("morph")) != contract.requires_morph:
        raise RuntimeError(f"{contract.key} current-Core Morph state is invalid")
    return {
        "status": "valid",
        "parameter_count": output.get("parameter_count"),
        "morph": bool(output.get("morph")),
        "slimmable": bool(output.get("slimmable")),
        "selected_size": output.get("selected_size"),
        "size_breakpoints": output.get("size_breakpoints"),
    }


def _validate_fire_report(report_path: Path, report_sha256: str, fire_sha256: str) -> dict[str, Any]:
    expected = _checked_sha256(report_sha256, "Fire package report SHA-256")
    actual = sha256_file(report_path)
    if actual != expected:
        raise RuntimeError("Fire package report SHA-256 does not match")
    report = _read_bounded_json(report_path, "Fire package report")
    package = report.get("package")
    evidence = report.get("external_evidence")
    presentation = report.get("presentation")
    if report.get("status") != "accepted":
        raise RuntimeError("Fire package report is not externally accepted")
    if not isinstance(evidence, Mapping) or evidence.get("passed") is not True:
        raise RuntimeError("Fire package report has no passed external gate")
    if not isinstance(package, Mapping) or package.get("sha256") != fire_sha256:
        raise RuntimeError("Fire package report does not bind the selected package hash")
    if not isinstance(presentation, Mapping) or presentation.get("head_shell") != "fire-chrome-v1":
        raise RuntimeError("Fire package report does not bind the DESS FIRE shell")
    return {
        "status": "accepted",
        "sha256": actual,
        "selected_member": report.get("selected_member"),
        "external_quality_tier": evidence.get("quality_tier"),
    }


def _validate_fire_calibration_report(
    report_path: Path,
    report_sha256: str,
    calibrated_fire_sha256: str,
) -> dict[str, Any]:
    expected = _checked_sha256(report_sha256, "Fire calibration report SHA-256")
    actual = sha256_file(report_path)
    if actual != expected:
        raise RuntimeError("Fire calibration report SHA-256 does not match")
    report = _read_bounded_json(report_path, "Fire calibration report")
    source = report.get("source_package")
    calibrated = report.get("calibrated_package")
    core = report.get("current_core")
    input_level_dbu = report.get("input_level_dbu")
    if report.get("status") != "metadata-only-input-calibration-repackage":
        raise RuntimeError("Fire calibration report has the wrong status")
    if report.get("network_payload_unchanged") is not True:
        raise RuntimeError("Fire calibration report does not preserve the model network")
    if not isinstance(source, Mapping) or SHA256_PATTERN.fullmatch(str(source.get("sha256", ""))) is None:
        raise RuntimeError("Fire calibration report has no valid source package hash")
    if not isinstance(calibrated, Mapping) or calibrated.get("sha256") != calibrated_fire_sha256:
        raise RuntimeError("Fire calibration report does not bind the calibrated Fire package")
    if (
        isinstance(input_level_dbu, bool)
        or not isinstance(input_level_dbu, (int, float))
        or not math.isfinite(float(input_level_dbu))
    ):
        raise RuntimeError("Fire calibration report has no finite input_level_dbu")
    if not isinstance(core, Mapping) or core.get("status") != "valid":
        raise RuntimeError("Fire calibration report has no current-Core validation")
    return {
        "status": "metadata-only-input-calibration-repackage",
        "sha256": actual,
        "source_package_sha256": source["sha256"],
        "calibrated_package_sha256": calibrated_fire_sha256,
        "input_level_dbu": float(input_level_dbu),
        "network_payload_sha256": report.get("network_payload_sha256"),
        "network_payload_unchanged": True,
    }


def assemble_catalogue(
    *,
    packages: Mapping[str, Path],
    package_hashes: Mapping[str, str],
    fire_report: Path,
    fire_report_sha256: str,
    core_loader: Path,
    output_dir: Path,
    fire_calibration_report: Path | None = None,
    fire_calibration_report_sha256: str | None = None,
) -> dict[str, Any]:
    output = _checked_output_directory(output_dir)
    loader = _checked_external_file(core_loader, "current Core manifest loader")
    expected_hashes = {
        contract.key: _checked_sha256(package_hashes.get(contract.key, ""), f"{contract.key} package SHA-256")
        for contract in SLOT_CONTRACTS
    }
    sources = {
        contract.key: _checked_external_file(packages[contract.key], f"{contract.key} package", ".dess")
        for contract in SLOT_CONTRACTS
    }
    report_path = _checked_external_file(fire_report, "Fire package report", ".json")

    actual_hashes: dict[str, str] = {}
    slot_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="dessamp-catalogue-package-cache.") as cache:
        cache_dir = Path(cache)
        for contract in SLOT_CONTRACTS:
            source = sources[contract.key]
            actual = sha256_file(source)
            if actual != expected_hashes[contract.key]:
                raise RuntimeError(f"{contract.key} package SHA-256 does not match")
            package = read_dess_package(source, cache_dir=cache_dir)
            manifest_record = _validate_manifest(contract, package.manifest)
            core_record = _validate_with_core(loader, source, contract)
            actual_hashes[contract.key] = actual
            slot_records.append(
                {
                    "slot": contract.key,
                    "filename": contract.filename,
                    "sha256": actual,
                    "package_format_version": package.format_version,
                    "manifest_sha256": package.manifest_sha256,
                    "model_sha256": package.model_sha256,
                    "morph_model_sha256": package.morph_model_sha256,
                    "manifest": manifest_record,
                    "current_core": core_record,
                }
            )

    fire_calibration = None
    accepted_fire_sha256 = actual_hashes["fire"]
    if (fire_calibration_report is None) != (fire_calibration_report_sha256 is None):
        raise ValueError("Fire calibration report path and SHA-256 must be supplied together")
    if fire_calibration_report is not None and fire_calibration_report_sha256 is not None:
        calibration_path = _checked_external_file(
            fire_calibration_report, "Fire calibration report", ".json"
        )
        fire_calibration = _validate_fire_calibration_report(
            calibration_path,
            fire_calibration_report_sha256,
            actual_hashes["fire"],
        )
        accepted_fire_sha256 = fire_calibration["source_package_sha256"]

    fire_acceptance = _validate_fire_report(
        report_path,
        fire_report_sha256,
        accepted_fire_sha256,
    )
    loader_sha256 = sha256_file(loader)
    lock = {
        "schema_version": 1,
        "classification": "private_owner_listening_workflow_validation_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalogue_product_order": ["fire", "forge", "redline"],
        "catalogue_product_count": 3,
        "slot_count": 4,
        "redline_variants_share_product_root": True,
        "fire_acceptance": fire_acceptance,
        "fire_calibration": fire_calibration,
        "current_core_loader_sha256": loader_sha256,
        "slots": slot_records,
        "gates": {
            "owner_listening_passed": False,
            "signing_approved": False,
            "installation_approved": False,
            "redistribution_approved": False,
        },
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for contract in SLOT_CONTRACTS:
            destination = staging / contract.filename
            shutil.copyfile(sources[contract.key], destination)
            if sha256_file(destination) != actual_hashes[contract.key]:
                raise RuntimeError(f"{contract.key} copied package failed SHA-256 verification")
        lock_path = staging / "catalogue-lock.json"
        with lock_path.open("w", encoding="utf-8") as destination:
            json.dump(lock, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_lock = output / "catalogue-lock.json"
    return {
        "status": "assembled-private-owner-listening-catalogue",
        "output_dir": str(output),
        "catalogue_lock": str(final_lock),
        "catalogue_lock_sha256": sha256_file(final_lock),
        "slot_hashes": actual_hashes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for contract in SLOT_CONTRACTS:
        option = contract.key.replace("_", "-")
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(f"--{option}-sha256", required=True)
    parser.add_argument("--fire-package-report", type=Path, required=True)
    parser.add_argument("--fire-package-report-sha256", required=True)
    parser.add_argument("--fire-calibration-report", type=Path)
    parser.add_argument("--fire-calibration-report-sha256")
    parser.add_argument("--core-loader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    packages = {contract.key: getattr(args, contract.key) for contract in SLOT_CONTRACTS}
    hashes = {contract.key: getattr(args, f"{contract.key}_sha256") for contract in SLOT_CONTRACTS}
    result = assemble_catalogue(
        packages=packages,
        package_hashes=hashes,
        fire_report=args.fire_package_report,
        fire_report_sha256=args.fire_package_report_sha256,
        core_loader=args.core_loader,
        output_dir=args.output_dir,
        fire_calibration_report=args.fire_calibration_report,
        fire_calibration_report_sha256=args.fire_calibration_report_sha256,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
