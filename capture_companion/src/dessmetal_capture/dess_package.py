from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .domain import DomainError


DESS_MAGIC = b"DESSAMP\0"
DESS_FORMAT_VERSION = 2
DESS_HEADER_V1 = struct.Struct("<8sIIQQ32s32s")
DESS_HEADER_V2 = struct.Struct("<8sIIQQQ32s32s32s")
# Kept as a source-compatible alias for callers that inspect the original
# single-model package header.
DESS_HEADER = DESS_HEADER_V1
MAX_DESS_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_DESS_MANIFEST_BYTES = 1024 * 1024
MAX_DESS_MODEL_BYTES = MAX_DESS_ARTIFACT_BYTES - DESS_HEADER_V2.size - 2


@dataclass(frozen=True)
class DessPackage:
    path: Path
    manifest: dict[str, Any]
    model_path: Path
    manifest_sha256: str
    model_sha256: str
    format_version: int = 1
    morph_model_path: Path | None = None
    morph_model_sha256: str | None = None


def create_dess_package(
    manifest_path: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Create one deterministic, self-contained amp artifact.

    The package embeds a runtime-only projection of the player manifest and the
    byte-exact NAM payload.  The source manifest remains the private evidence
    record; capture identifiers, validation logs, training provenance, and
    local paths must not cross the distributable package boundary.
    """

    source_manifest = Path(manifest_path).expanduser().resolve()
    manifest = _read_manifest(source_manifest)
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise DomainError("Player manifest model must be an object")
    relative_model = model.get("path")
    if relative_model != "model.nam":
        raise DomainError("A .dess package requires the canonical model.nam payload path")
    model_path = (source_manifest.parent / relative_model).resolve()
    if not model_path.is_relative_to(source_manifest.parent) or not model_path.is_file():
        raise DomainError("Player manifest model path escapes or is missing from the artifact")
    model_bytes = model_path.read_bytes()
    if not model_bytes or len(model_bytes) > MAX_DESS_MODEL_BYTES:
        raise DomainError("NAM payload is empty or exceeds the .dess package limit")
    model_digest = hashlib.sha256(model_bytes).digest()
    if model.get("sha256") != model_digest.hex():
        raise DomainError("Player manifest model SHA-256 does not match model.nam")

    morph_model_bytes = b""
    morph_model_digest: bytes | None = None
    morph = manifest.get("morph")
    format_version = 1
    if morph is not None:
        if not isinstance(morph, dict) or not isinstance(morph.get("model"), dict):
            raise DomainError("Player manifest morph model must be an object")
        morph_model = morph["model"]
        relative_morph_model = morph_model.get("path")
        if relative_morph_model != "morph-model.nam":
            raise DomainError("A morph .dess package requires the canonical morph-model.nam payload path")
        morph_model_path = (source_manifest.parent / relative_morph_model).resolve()
        if not morph_model_path.is_relative_to(source_manifest.parent) or not morph_model_path.is_file():
            raise DomainError("Player manifest morph model path escapes or is missing from the artifact")
        morph_model_bytes = morph_model_path.read_bytes()
        if not morph_model_bytes or len(morph_model_bytes) > MAX_DESS_MODEL_BYTES:
            raise DomainError("Morph NAM payload is empty or exceeds the .dess package limit")
        morph_model_digest = hashlib.sha256(morph_model_bytes).digest()
        if morph_model.get("sha256") != morph_model_digest.hex():
            raise DomainError("Player manifest morph model SHA-256 does not match morph-model.nam")
        format_version = DESS_FORMAT_VERSION

    portable_manifest = _portable_manifest(manifest)
    manifest_bytes = json.dumps(
        portable_manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not manifest_bytes or len(manifest_bytes) > MAX_DESS_MANIFEST_BYTES:
        raise DomainError("Player manifest is empty or exceeds the .dess package limit")
    manifest_digest = hashlib.sha256(manifest_bytes).digest()
    if format_version == 1:
        header = DESS_HEADER_V1.pack(
            DESS_MAGIC,
            format_version,
            0,
            len(manifest_bytes),
            len(model_bytes),
            manifest_digest,
            model_digest,
        )
    else:
        assert morph_model_digest is not None
        header = DESS_HEADER_V2.pack(
            DESS_MAGIC,
            format_version,
            0,
            len(manifest_bytes),
            len(model_bytes),
            len(morph_model_bytes),
            manifest_digest,
            model_digest,
            morph_model_digest,
        )
    if len(header) + len(manifest_bytes) + len(model_bytes) + len(morph_model_bytes) > MAX_DESS_ARTIFACT_BYTES:
        raise DomainError("DessAmp package exceeds the 64 MiB artifact limit")
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source_manifest.with_suffix(".dess")
    )
    if destination.suffix.lower() != ".dess":
        raise DomainError("DessMetal amp packages must use the .dess extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(header)
        temporary.write(manifest_bytes)
        temporary.write(model_bytes)
        temporary.write(morph_model_bytes)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "format_version": format_version,
        "manifest_sha256": manifest_digest.hex(),
        "model_sha256": model_digest.hex(),
        "morph_model_sha256": morph_model_digest.hex() if morph_model_digest is not None else None,
    }


def read_dess_package(
    package_path: Path | str,
    *,
    cache_dir: Path | str | None = None,
) -> DessPackage:
    source = Path(package_path).expanduser().resolve()
    if not source.is_file():
        raise DomainError(f"DessMetal amp package is unavailable: {source}")
    size = source.stat().st_size
    if size <= DESS_HEADER_V1.size or size > MAX_DESS_ARTIFACT_BYTES:
        raise DomainError("DessMetal amp package is empty or exceeds 64 MiB")
    with source.open("rb") as package:
        prefix = package.read(16)
        if len(prefix) != 16:
            raise DomainError("DessMetal amp package header is truncated")
        magic, version, flags = struct.unpack("<8sII", prefix)
        if magic != DESS_MAGIC:
            raise DomainError("DessMetal amp package magic is invalid")
        if version not in (1, DESS_FORMAT_VERSION) or flags != 0:
            raise DomainError("Unsupported DessMetal amp package version or flags")
        header_struct = DESS_HEADER_V1 if version == 1 else DESS_HEADER_V2
        package.seek(0)
        header = package.read(header_struct.size)
        if len(header) != header_struct.size:
            raise DomainError("DessMetal amp package header is truncated")
        if version == 1:
            _, _, _, manifest_size, model_size, manifest_digest, model_digest = header_struct.unpack(header)
            morph_model_size = 0
            morph_model_digest = None
        else:
            (
                _,
                _,
                _,
                manifest_size,
                model_size,
                morph_model_size,
                manifest_digest,
                model_digest,
                morph_model_digest,
            ) = header_struct.unpack(header)
        if not 0 < manifest_size <= MAX_DESS_MANIFEST_BYTES:
            raise DomainError("DessMetal amp package manifest size is invalid")
        if not 0 < model_size <= MAX_DESS_MODEL_BYTES:
            raise DomainError("DessMetal amp package model size is invalid")
        if version == DESS_FORMAT_VERSION and not 0 < morph_model_size <= MAX_DESS_MODEL_BYTES:
            raise DomainError("DessAmp morph package model size is invalid")
        if header_struct.size + manifest_size + model_size + morph_model_size != size:
            raise DomainError("DessMetal amp package length or trailing data is invalid")
        manifest_bytes = package.read(manifest_size)
        model_bytes = package.read(model_size)
        morph_model_bytes = package.read(morph_model_size)
    if hashlib.sha256(manifest_bytes).digest() != manifest_digest:
        raise DomainError("DessMetal amp package manifest hash is invalid")
    if hashlib.sha256(model_bytes).digest() != model_digest:
        raise DomainError("DessMetal amp package model hash is invalid")
    if morph_model_digest is not None and hashlib.sha256(morph_model_bytes).digest() != morph_model_digest:
        raise DomainError("DessAmp morph package model hash is invalid")
    try:
        raw_manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DomainError(f"DessMetal amp package manifest is not valid UTF-8 JSON: {error}") from error
    if not isinstance(raw_manifest, dict):
        raise DomainError("DessMetal amp package manifest root must be an object")
    model = raw_manifest.get("model")
    if not isinstance(model, dict) or model.get("path") != "model.nam":
        raise DomainError("DessMetal amp package requires a canonical model.nam manifest entry")
    if model.get("sha256") != model_digest.hex():
        raise DomainError("DessMetal amp package manifest disagrees with its model payload hash")
    raw_morph = raw_manifest.get("morph")
    if version == 1 and raw_morph is not None:
        raise DomainError("A version 1 DessMetal package cannot embed a morph model")
    if version == DESS_FORMAT_VERSION:
        if not isinstance(raw_morph, dict) or not isinstance(raw_morph.get("model"), dict):
            raise DomainError("DessAmp morph package manifest is missing its morph model")
        if raw_morph["model"].get("path") != "morph-model.nam":
            raise DomainError("DessAmp morph package requires a canonical morph-model.nam manifest entry")
        assert morph_model_digest is not None
        if raw_morph["model"].get("sha256") != morph_model_digest.hex():
            raise DomainError("DessAmp morph package manifest disagrees with its morph model payload hash")

    cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else Path(tempfile.gettempdir()).resolve() / f"dessamp-player-models-v{version}"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    extracted_model = _extract_verified_model(cache_root, model_bytes, model_digest, "model")
    extracted_morph_model = (
        _extract_verified_model(cache_root, morph_model_bytes, morph_model_digest, "morph model")
        if morph_model_digest is not None
        else None
    )
    return DessPackage(
        path=source,
        manifest=dict(raw_manifest),
        model_path=extracted_model,
        manifest_sha256=manifest_digest.hex(),
        model_sha256=model_digest.hex(),
        format_version=version,
        morph_model_path=extracted_morph_model,
        morph_model_sha256=morph_model_digest.hex() if morph_model_digest is not None else None,
    )


def _extract_verified_model(cache_root: Path, payload: bytes, digest: bytes, label: str) -> Path:
    digest_hex = digest.hex()
    destination = cache_root / f"{digest_hex}.nam"
    if destination.is_file() and _sha256_file(destination) == digest_hex:
        return destination
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{digest_hex}.",
        suffix=".tmp",
        dir=cache_root,
        delete=False,
    ) as temporary:
        temporary_model = Path(temporary.name)
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        temporary_model.replace(destination)
    except Exception:
        temporary_model.unlink(missing_ok=True)
        raise
    if _sha256_file(destination) != digest_hex:
        destination.unlink(missing_ok=True)
        raise DomainError(f"Extracted DessAmp {label} failed SHA-256 verification")
    return destination


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DomainError(f"Player manifest is unavailable: {path}")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_DESS_MANIFEST_BYTES:
        raise DomainError("Player manifest is empty or exceeds 1 MiB")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DomainError(f"Player manifest is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise DomainError("Player manifest root must be an object")
    return value


def _portable_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project the private evidence manifest onto the public runtime schema.

    This is intentionally an allowlist, not a recursive redactor.  New capture
    or training evidence therefore stays private by default until the package
    contract explicitly adopts a bounded runtime field.
    """

    portable: dict[str, Any] = {
        "schema_version": manifest.get("schema_version"),
        "artifact_id": manifest.get("artifact_id"),
        "name": manifest.get("name"),
        "model": _portable_model(manifest.get("model")),
        "controls": [
            _copy_fields(control, _CONTROL_FIELDS)
            for control in _require_object_array(manifest.get("controls"), "controls")
        ],
    }
    presentation = manifest.get("presentation")
    if presentation is not None:
        portable["presentation"] = _copy_fields(
            presentation,
            ("head_shell", "product_family", "product_variant"),
        )
    fixed_controls = manifest.get("fixed_controls")
    if fixed_controls is not None:
        portable["fixed_controls"] = [
            _copy_fields(control, _FIXED_CONTROL_FIELDS)
            for control in _require_object_array(fixed_controls, "fixed_controls")
        ]
    morph = manifest.get("morph")
    if morph is not None:
        if not isinstance(morph, Mapping):
            raise DomainError("Player manifest morph must be an object")
        portable["morph"] = {
            "mode": morph.get("mode"),
            "from_label": morph.get("from_label"),
            "to_label": morph.get("to_label"),
            "model": _portable_model(morph.get("model")),
            "alignment": _copy_fields(
                morph.get("alignment"),
                ("delay_samples", "polarity", "level_scale", "reference_sample_rate"),
            ),
            "blend": _copy_fields(
                morph.get("blend"),
                ("law", "center_gain", "smoothing_ms"),
            ),
            "control": _copy_fields(morph.get("control"), _CONTROL_FIELDS),
        }
    return portable


_CONTROL_FIELDS = (
    "id",
    "label",
    "minimum",
    "maximum",
    "step",
    "unit",
    "taper",
    "display_minimum",
    "display_maximum",
    "default_value",
    "model_parameter_index",
)

_FIXED_CONTROL_FIELDS = (
    "id",
    "label",
    "taper",
    "value",
    "minimum",
    "maximum",
    "unit",
)


def _copy_fields(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DomainError("Player manifest runtime field must be an object")
    return {field: value[field] for field in fields if field in value}


def _portable_model(value: Any) -> dict[str, Any]:
    model = _copy_fields(
        value,
        (
            "path",
            "format",
            "sha256",
            "format_version",
            "architecture",
            "sample_rate",
            "parameter_mode",
            "global_condition_size",
        ),
    )
    assert isinstance(value, Mapping)
    runtime_quality = value.get("runtime_quality")
    if runtime_quality is not None:
        model["runtime_quality"] = _copy_fields(
            runtime_quality,
            ("mode", "default_size", "breakpoints"),
        )
    return model


def _require_object_array(value: Any, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(entry, Mapping) for entry in value):
        raise DomainError(f"Player manifest {field} must be an array of objects")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
