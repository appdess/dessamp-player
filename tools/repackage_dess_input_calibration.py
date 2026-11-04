#!/usr/bin/env python3
"""Add measured input calibration metadata to one immutable .dess package.

The model network, weights, architecture, sample rate, controls, and player
manifest contract remain unchanged.  Only ``metadata.input_level_dbu`` inside
the embedded NAM JSON is added, and the resulting package is validated by an
explicit current-Core loader before it is materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SOURCE = REPOSITORY_ROOT / "capture_companion" / "src"
sys.path.insert(0, str(CAPTURE_SOURCE))

from dessmetal_capture.dess_package import create_dess_package, read_dess_package  # noqa: E402


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _network_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(model)
    payload.pop("metadata", None)
    return payload


def _checked_source(path_value: Path, expected_sha256: str) -> Path:
    unresolved = path_value.expanduser()
    if unresolved.is_symlink():
        raise RuntimeError("source package must not be a symbolic link")
    path = unresolved.resolve()
    if not path.is_file() or path.suffix.lower() != ".dess":
        raise FileNotFoundError(f"source .dess package is unavailable: {path}")
    if SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("expected source SHA-256 must be one lowercase digest")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("source package SHA-256 does not match")
    return path


def _checked_new_path(path_value: Path, suffix: str, label: str) -> Path:
    unresolved = path_value.expanduser()
    if unresolved.exists() or unresolved.is_symlink():
        raise RuntimeError(f"Refusing to replace existing {label}: {unresolved}")
    path = unresolved.resolve()
    if path.suffix.lower() != suffix:
        raise ValueError(f"{label} must use the {suffix} extension")
    if path == REPOSITORY_ROOT or path.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError(f"{label} must remain outside the repository")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise RuntimeError(f"{label} parent must be one existing real directory")
    return path


def _checked_loader(path_value: Path) -> Path:
    unresolved = path_value.expanduser()
    if unresolved.is_symlink():
        raise RuntimeError("current Core loader must not be a symbolic link")
    path = unresolved.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"current Core loader is unavailable: {path}")
    return path


def _validate_with_core(loader: Path, package_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(loader), str(package_path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "current Core rejected package")[-4_000:]
        raise RuntimeError(f"current-Core validation failed: {detail}")
    try:
        output = json.loads(result.stdout.strip())
    except json.JSONDecodeError as error:
        raise RuntimeError("current-Core output is not JSON") from error
    if not isinstance(output, Mapping) or output.get("status") != "valid":
        raise RuntimeError("current-Core result is not valid")
    return {
        "status": "valid",
        "parameter_count": output.get("parameter_count"),
        "morph": bool(output.get("morph")),
        "slimmable": bool(output.get("slimmable")),
        "selected_size": output.get("selected_size"),
    }


def repackage_input_calibration(
    *,
    source_package: Path,
    expected_source_sha256: str,
    input_level_dbu: float,
    core_loader: Path,
    output_package: Path,
    report_path: Path,
) -> dict[str, Any]:
    if isinstance(input_level_dbu, bool) or not math.isfinite(float(input_level_dbu)):
        raise ValueError("input_level_dbu must be finite")
    source = _checked_source(source_package, expected_source_sha256)
    output = _checked_new_path(output_package, ".dess", "calibrated package")
    report = _checked_new_path(report_path, ".json", "calibration report")
    loader = _checked_loader(core_loader)

    with tempfile.TemporaryDirectory(
        prefix=f".{output.stem}-input-calibration.", dir=output.parent
    ) as temporary:
        root = Path(temporary)
        package = read_dess_package(source, cache_dir=root / "cache")
        if package.format_version != 1 or package.morph_model_path is not None:
            raise RuntimeError("metadata-only calibration currently accepts one-model format-v1 packages")
        try:
            model = json.loads(package.model_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("embedded NAM payload is not UTF-8 JSON") from error
        if not isinstance(model, dict):
            raise RuntimeError("embedded NAM payload root must be an object")
        before_network_sha256 = canonical_sha256(_network_payload(model))
        metadata = model.get("metadata")
        if metadata is None:
            metadata = {}
            model["metadata"] = metadata
        if not isinstance(metadata, dict):
            raise RuntimeError("embedded NAM metadata must be an object")
        existing = metadata.get("input_level_dbu")
        if existing is not None and (
            isinstance(existing, bool)
            or not isinstance(existing, (int, float))
            or not math.isclose(float(existing), float(input_level_dbu), abs_tol=1.0e-9)
        ):
            raise RuntimeError("embedded NAM payload has a conflicting input_level_dbu")
        metadata["input_level_dbu"] = float(input_level_dbu)
        after_network_sha256 = canonical_sha256(_network_payload(model))
        if before_network_sha256 != after_network_sha256:
            raise RuntimeError("metadata edit changed the NAM network payload")

        model_path = root / "model.nam"
        model_path.write_text(
            json.dumps(model, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        model_sha256 = sha256_file(model_path)
        manifest = dict(package.manifest)
        manifest_model = manifest.get("model")
        if not isinstance(manifest_model, dict):
            raise RuntimeError("package manifest model must be an object")
        manifest_model = dict(manifest_model)
        manifest_model["path"] = "model.nam"
        manifest_model["sha256"] = model_sha256
        manifest["model"] = manifest_model
        manifest_path = root / "player-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_package = root / output.name
        package_record = create_dess_package(manifest_path, temporary_package)
        core_record = _validate_with_core(loader, temporary_package)
        calibrated = read_dess_package(temporary_package, cache_dir=root / "verification-cache")
        calibrated_model = json.loads(calibrated.model_path.read_text(encoding="utf-8"))
        calibrated_metadata = calibrated_model.get("metadata")
        if not isinstance(calibrated_metadata, Mapping) or not math.isclose(
            float(calibrated_metadata.get("input_level_dbu")),
            float(input_level_dbu),
            abs_tol=1.0e-9,
        ):
            raise RuntimeError("repackaged model did not retain input calibration metadata")
        if canonical_sha256(_network_payload(calibrated_model)) != before_network_sha256:
            raise RuntimeError("repackaged model network payload changed")

        report_value = {
            "schema_version": 1,
            "status": "metadata-only-input-calibration-repackage",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_level_dbu": float(input_level_dbu),
            "source_package": {
                "sha256": expected_source_sha256,
                "model_sha256": package.model_sha256,
                "manifest_sha256": package.manifest_sha256,
            },
            "calibrated_package": {
                "sha256": package_record["sha256"],
                "model_sha256": package_record["model_sha256"],
                "manifest_sha256": package_record["manifest_sha256"],
            },
            "network_payload_sha256": before_network_sha256,
            "network_payload_unchanged": True,
            "current_core_loader_sha256": sha256_file(loader),
            "current_core": core_record,
            "gates": {
                "sound_acceptance_changed": False,
                "installation_approved": False,
                "signing_approved": False,
                "redistribution_approved": False,
            },
        }
        temporary_report = root / report.name
        temporary_report.write_text(
            json.dumps(report_value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_package, output)
        os.replace(temporary_report, report)

    return {
        "status": "repackaged-input-calibration",
        "output_package": str(output),
        "output_package_sha256": sha256_file(output),
        "report": str(report),
        "report_sha256": sha256_file(report),
        "input_level_dbu": float(input_level_dbu),
        "network_payload_unchanged": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--input-level-dbu", type=float, required=True)
    parser.add_argument("--core-loader", type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = repackage_input_calibration(
        source_package=args.source_package,
        expected_source_sha256=args.expected_source_sha256,
        input_level_dbu=args.input_level_dbu,
        core_loader=args.core_loader,
        output_package=args.output_package,
        report_path=args.report,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
