from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping

from .domain import DomainError
from .dess_package import create_dess_package
from .store import CaptureStore, utc_now


MAX_MODEL_BYTES = 256 * 1024 * 1024
PLAYER_MANIFEST_SCHEMA_VERSION = 1

_FIXED_SWITCH_CONTROLS = (
    ("bright", "Bright"),
    ("bottom", "Bottom"),
    ("mid_boost", "Mid Boost"),
)

_FIXED_LINEAR_CONTROLS = (
    ("lead_volume", "Lead Volume", 0.0, 10.0, ""),
)


def _fixed_switch_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        return parsed if parsed in (0.0, 1.0) else None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"on", "enabled", "true", "1"}:
            return 1.0
        if normalized in {"off", "disabled", "false", "0"}:
            return 0.0
    return None


def fixed_controls_from_project(project: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = project.get("metadata")
    frozen = metadata.get("frozen_controls") if isinstance(metadata, Mapping) else None
    if not isinstance(frozen, Mapping):
        return []
    learned_ids = {str(control.get("id")) for control in project.get("controls", [])}
    fixed_controls: list[dict[str, Any]] = []
    for control_id, label in _FIXED_SWITCH_CONTROLS:
        if control_id not in frozen:
            continue
        value = _fixed_switch_value(frozen[control_id])
        if value is None:
            continue
        if control_id in learned_ids:
            raise DomainError(f"Control {control_id!r} cannot be both learned and fixed")
        fixed_controls.append(
            {
                "id": control_id,
                "label": label,
                "taper": "switch",
                "value": value,
            }
        )
    for control_id, label, minimum, maximum, unit in _FIXED_LINEAR_CONTROLS:
        if control_id not in frozen:
            continue
        raw_value = frozen[control_id]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value < minimum or value > maximum:
            continue
        if control_id in learned_ids:
            raise DomainError(f"Control {control_id!r} cannot be both learned and fixed")
        fixed_controls.append(
            {
                "id": control_id,
                "label": label,
                "taper": "linear",
                "value": value,
                "minimum": minimum,
                "maximum": maximum,
                "unit": unit,
            }
        )
    return fixed_controls


def import_model_artifact(
    store: CaptureStore,
    *,
    project_id: str,
    source_model_path: Path,
    name: str,
    provenance: Mapping[str, Any],
    loader_path: Path | None,
) -> dict[str, Any]:
    project = store.get_project(project_id)
    clean_name = name.strip()
    if not clean_name or len(clean_name) > 120:
        raise DomainError("Model name must contain 1 to 120 characters")
    metadata = inspect_nam_model(source_model_path)
    artifact_id = str(uuid.uuid4())
    artifact_dir = store.model_dir / project_id / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    packaged_model = artifact_dir / "model.nam"
    shutil.copyfile(source_model_path, packaged_model)
    digest = sha256_file(packaged_model)
    captures = store.list_captures(project_id)
    global_condition_size = metadata["global_condition_size"]
    if global_condition_size > len(project["controls"]):
        shutil.rmtree(artifact_dir)
        raise DomainError(
            f"Model expects {global_condition_size} global controls, but the project defines only "
            f"{len(project['controls'])}"
        )
    parameter_mode = "global_condition" if global_condition_size else "snapshot"
    controls = []
    for index, control in enumerate(project["controls"]):
        entry = dict(control)
        entry["model_parameter_index"] = index if index < global_condition_size else None
        entry["default_value"] = (float(control["minimum"]) + float(control["maximum"])) / 2.0
        controls.append(entry)
    validation_status, validation_message = validate_with_loader(packaged_model, loader_path)
    manifest = {
        "schema_version": PLAYER_MANIFEST_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "created_at": utc_now(),
        "name": clean_name,
        "capture_project": {
            "id": project_id,
            "name": project["name"],
            "amp_name": project["amp_name"],
        },
        "model": {
            "path": "model.nam",
            "format": "nam",
            "sha256": digest,
            "format_version": metadata["version"],
            "architecture": metadata["architecture"],
            "sample_rate": metadata["sample_rate"],
            "parameter_mode": parameter_mode,
            "global_condition_size": global_condition_size,
            "runtime_quality": metadata["runtime_quality"],
        },
        "controls": controls,
        "fixed_controls": fixed_controls_from_project(project),
        "capture_evidence": {
            "capture_ids": [capture["id"] for capture in captures],
            "capture_count": len(captures),
        },
        "provenance": dict(provenance),
        "runtime_validation": {
            "status": validation_status,
            "message": validation_message,
        },
    }
    manifest_path = artifact_dir / "player-manifest.json"
    temporary_manifest = artifact_dir / ".player-manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    package = create_dess_package(manifest_path, artifact_dir / "model.dess")
    artifact = store.add_model_artifact(
        artifact_id=artifact_id,
        project_id=project_id,
        name=clean_name,
        model_path=packaged_model,
        manifest_path=manifest_path,
        model_sha256=digest,
        model_format_version=metadata["version"],
        architecture=metadata["architecture"],
        sample_rate=metadata["sample_rate"],
        validation_status=validation_status,
        validation_message=validation_message,
        provenance=provenance,
    )
    artifact["package_path"] = package["path"]
    artifact["package_sha256"] = package["sha256"]
    artifact["package_format_version"] = package["format_version"]
    return artifact


def inspect_nam_model(path: Path | str) -> dict[str, Any]:
    model_path = Path(path)
    if not model_path.is_file():
        raise DomainError(f"NAM model does not exist: {model_path}")
    size = model_path.stat().st_size
    if size <= 0 or size > MAX_MODEL_BYTES:
        raise DomainError("NAM model is empty or exceeds the 256 MiB application limit")
    try:
        with model_path.open("r", encoding="utf-8") as source:
            model = json.load(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DomainError(f"NAM model is not valid UTF-8 JSON: {error}") from error
    if not isinstance(model, dict):
        raise DomainError("NAM model root must be an object")
    metadata = _inspect_nam_object(model)
    metadata["size_bytes"] = size
    return metadata


def _inspect_nam_object(model: Mapping[str, Any]) -> dict[str, Any]:
    version = model.get("version")
    architecture = model.get("architecture")
    sample_rate = model.get("sample_rate")
    config = model.get("config")
    weights = model.get("weights")
    if not isinstance(version, str) or not version:
        raise DomainError("NAM model is missing its format version")
    if not isinstance(architecture, str) or not architecture:
        raise DomainError("NAM model is missing its architecture")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float))
        or not math.isfinite(float(sample_rate))
        or not 8_000 <= float(sample_rate) <= 384_000
    ):
        raise DomainError("NAM model sample rate is invalid")
    if not isinstance(config, dict):
        raise DomainError("NAM model config must be an object")
    if not isinstance(weights, list):
        raise DomainError("NAM model weights must be an array")
    global_condition_size = config.get("global_condition_size", 0)
    if not isinstance(global_condition_size, int) or not 0 <= global_condition_size <= 64:
        raise DomainError("NAM model global condition size is invalid")
    runtime_quality: dict[str, Any] = {"mode": "fixed", "default_size": 1.0, "breakpoints": []}
    if architecture == "SlimmableContainer":
        if weights:
            raise DomainError("SlimmableContainer top-level weights must be empty")
        submodels = config.get("submodels")
        if not isinstance(submodels, list) or not 1 <= len(submodels) <= 16:
            raise DomainError("SlimmableContainer must define 1 to 16 submodels")
        thresholds: list[float] = []
        child_metadata: list[dict[str, Any]] = []
        for entry in submodels:
            if not isinstance(entry, dict):
                raise DomainError("SlimmableContainer submodel entries must be objects")
            threshold = entry.get("max_value")
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
                or float(threshold) <= 0.0
            ):
                raise DomainError("SlimmableContainer thresholds must be finite and positive")
            if thresholds and float(threshold) <= thresholds[-1]:
                raise DomainError("SlimmableContainer thresholds must be strictly increasing")
            child = entry.get("model")
            if not isinstance(child, dict):
                raise DomainError("SlimmableContainer submodel is missing its model")
            thresholds.append(float(threshold))
            child_metadata.append(_inspect_nam_object(child))
        if thresholds[-1] < 1.0:
            raise DomainError("SlimmableContainer final threshold must cover 1.0")
        if any(child["sample_rate"] != int(round(float(sample_rate))) for child in child_metadata):
            raise DomainError("SlimmableContainer submodels must share the container sample rate")
        child_conditions = {child["global_condition_size"] for child in child_metadata}
        if len(child_conditions) != 1:
            raise DomainError("SlimmableContainer submodels must share a global condition size")
        global_condition_size = child_conditions.pop()
        channels = [_a2_channels(entry["model"]) for entry in submodels]
        runtime_quality = {
            "mode": "slimmable",
            "family": "a2" if channels == [3, 8] else "generic",
            "default_size": 1.0,
            "breakpoints": thresholds[:-1],
            "variants": [
                {
                    "maximum_size": threshold,
                    "architecture": child["architecture"],
                    "format_version": child["version"],
                    "channels": channel,
                }
                for threshold, child, channel in zip(thresholds, child_metadata, channels)
            ],
        }

    return {
        "version": version,
        "architecture": architecture,
        "sample_rate": int(round(float(sample_rate))),
        "global_condition_size": global_condition_size,
        "runtime_quality": runtime_quality,
    }


def _a2_channels(model: Mapping[str, Any]) -> int | None:
    if model.get("architecture") != "WaveNet":
        return None
    config = model.get("config")
    if not isinstance(config, dict):
        return None
    layers = config.get("layers")
    if not isinstance(layers, list) or len(layers) != 1 or not isinstance(layers[0], dict):
        return None
    layer = layers[0]
    channels = layer.get("channels")
    kernel_sizes = layer.get("kernel_sizes")
    if channels not in (3, 8) or not isinstance(kernel_sizes, list) or len(kernel_sizes) != 23:
        return None
    return channels


def validate_with_loader(model_path: Path, loader_path: Path | None) -> tuple[str, str]:
    if loader_path is None:
        return "unverified", "No NAM Core loader executable was configured"
    loader = loader_path.resolve()
    if not loader.is_file():
        return "unverified", f"NAM Core loader executable is unavailable: {loader}"
    try:
        result = subprocess.run(
            [str(loader), str(model_path.resolve())],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return "invalid", f"NAM Core loader could not complete: {error}"
    message = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    message = message[-4_000:]
    return ("valid", message or "NAM Core loaded the model") if result.returncode == 0 else (
        "invalid",
        message or f"NAM Core loader exited with status {result.returncode}",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
