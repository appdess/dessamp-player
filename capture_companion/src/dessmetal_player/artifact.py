from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from dessmetal_capture.domain import ControlDefinition, DomainError
from dessmetal_capture.dess_package import MAX_DESS_ARTIFACT_BYTES, read_dess_package
from dessmetal_capture.manifests import inspect_nam_model
from dessmetal_capture.wav_io import inspect_wav


MAX_MANIFEST_BYTES = 1024 * 1024
MAX_RENDER_INPUT_BYTES = 16 * 1024 * 1024


class PlayerArtifact:
    """Validated legacy manifest or self-contained .dess amp artifact."""

    def __init__(self, manifest_path: Path | str, renderer_path: Path | str):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.renderer_path = Path(renderer_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise DomainError(f"Player manifest is unavailable: {self.manifest_path}")
        if not self.renderer_path.is_file() or not os.access(self.renderer_path, os.X_OK):
            raise DomainError(f"NAM Core renderer is not executable: {self.renderer_path}")
        self.package_path: Path | None = None
        self._packaged_model_path: Path | None = None
        self._packaged_morph_model_path: Path | None = None
        if self.manifest_path.suffix.lower() == ".dess":
            if self.manifest_path.stat().st_size > MAX_DESS_ARTIFACT_BYTES:
                raise DomainError("DessMetal amp package exceeds 64 MiB")
            package = read_dess_package(self.manifest_path)
            self.package_path = package.path
            self._packaged_model_path = package.model_path
            self._packaged_morph_model_path = package.morph_model_path
            raw = package.manifest
        else:
            size = self.manifest_path.stat().st_size
            if size <= 0 or size > MAX_MANIFEST_BYTES:
                raise DomainError("Player manifest is empty or exceeds 1 MiB")
            try:
                raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise DomainError(f"Player manifest is not valid UTF-8 JSON: {error}") from error
        self.manifest = self._validate_manifest(raw)

    def _validate_manifest(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise DomainError("Unsupported player manifest schema_version")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise DomainError("Player manifest name must contain 1 to 120 characters")
        model = raw.get("model")
        if not isinstance(model, dict) or model.get("format") != "nam":
            raise DomainError("Player manifest model must use NAM format")
        relative_model = model.get("path")
        if not isinstance(relative_model, str) or not relative_model or len(relative_model) > 512:
            raise DomainError("Player manifest model path is invalid")
        relative_path = Path(relative_model)
        if relative_path.is_absolute():
            raise DomainError("Player manifest model path must be relative")
        if self._packaged_model_path is not None:
            if relative_model != "model.nam":
                raise DomainError("DessMetal amp package model path must be model.nam")
            model_path = self._packaged_model_path
        else:
            artifact_root = self.manifest_path.parent.resolve()
            model_path = (artifact_root / relative_path).resolve()
            if not model_path.is_relative_to(artifact_root) or not model_path.is_file():
                raise DomainError("Player manifest model path escapes or is missing from the artifact")
        digest = _sha256_file(model_path)
        if model.get("sha256") != digest:
            raise DomainError("Player manifest model SHA-256 does not match model.nam")
        metadata = inspect_nam_model(model_path)
        for field, expected in (
            ("format_version", metadata["version"]),
            ("architecture", metadata["architecture"]),
            ("sample_rate", metadata["sample_rate"]),
            ("global_condition_size", metadata["global_condition_size"]),
        ):
            if model.get(field) != expected:
                raise DomainError(f"Player manifest {field} disagrees with model.nam")
        condition_size = metadata["global_condition_size"]
        expected_mode = "global_condition" if condition_size else "snapshot"
        if model.get("parameter_mode") != expected_mode:
            raise DomainError("Player manifest parameter_mode disagrees with model.nam")
        runtime_quality = model.get("runtime_quality")
        if not isinstance(runtime_quality, dict):
            raise DomainError("Player manifest runtime_quality must be an object")
        actual_quality = metadata["runtime_quality"]
        if runtime_quality.get("mode") != actual_quality["mode"]:
            raise DomainError("Player manifest runtime quality mode disagrees with model.nam")
        if runtime_quality.get("breakpoints", []) != actual_quality["breakpoints"]:
            raise DomainError("Player manifest runtime quality breakpoints disagree with model.nam")
        default_quality = _finite_float(runtime_quality.get("default_size", 1.0), "runtime default_size")
        if not 0.0 <= default_quality <= 1.0:
            raise DomainError("Player manifest runtime default_size must be in [0, 1]")

        raw_presentation = raw.get("presentation", {})
        if not isinstance(raw_presentation, dict):
            raise DomainError("Player manifest presentation must be an object")
        head_shell = raw_presentation.get("head_shell", "studio-black-v1")
        if not isinstance(head_shell, str) or re.fullmatch(r"[a-z0-9-]{1,64}", head_shell) is None:
            raise DomainError("Player manifest presentation head_shell token is invalid")
        product_family = raw_presentation.get("product_family", "")
        product_variant = raw_presentation.get("product_variant", "")
        if not isinstance(product_family, str) or (
            product_family and re.fullmatch(r"[a-z0-9-]{1,64}", product_family) is None
        ):
            raise DomainError("Player manifest presentation product_family token is invalid")
        if not isinstance(product_variant, str) or (
            product_variant and re.fullmatch(r"[a-z0-9-]{1,64}", product_variant) is None
        ):
            raise DomainError("Player manifest presentation product_variant token is invalid")
        if product_variant and not product_family:
            raise DomainError("Player manifest presentation product_variant requires product_family")
        presentation = {
            **raw_presentation,
            "head_shell": head_shell,
            "product_family": product_family,
            "product_variant": product_variant,
        }

        raw_controls = raw.get("controls")
        if not isinstance(raw_controls, list):
            raise DomainError("Player manifest controls must be an array")
        controls: list[dict[str, Any]] = []
        assigned: set[int] = set()
        identifiers: set[str] = set()
        for entry in raw_controls:
            if not isinstance(entry, dict):
                raise DomainError("Player manifest controls must contain objects")
            control = ControlDefinition.from_mapping(entry)
            if control.id in identifiers:
                raise DomainError("Player manifest control ids must be unique")
            identifiers.add(control.id)
            default_value = _finite_float(entry.get("default_value"), f"{control.id} default_value")
            if not control.minimum <= default_value <= control.maximum:
                raise DomainError(f"Control {control.id!r} default lies outside its range")
            parameter_index = entry.get("model_parameter_index")
            if parameter_index is not None:
                if isinstance(parameter_index, bool) or not isinstance(parameter_index, int):
                    raise DomainError("Control model_parameter_index must be an integer or null")
                if not 0 <= parameter_index < condition_size or parameter_index in assigned:
                    raise DomainError("Control model_parameter_index is outside or duplicated")
                assigned.add(parameter_index)
            controls.append(
                {
                    **control.as_dict(),
                    "default_value": default_value,
                    "model_parameter_index": parameter_index,
                }
            )
        if assigned != set(range(condition_size)):
            raise DomainError("Player manifest must map every model parameter exactly once")
        validated_morph = self._validate_morph(
            raw.get("morph"),
            primary_sample_rate=int(metadata["sample_rate"]),
            model_parameter_index=condition_size,
            identifiers=identifiers,
        )
        if validated_morph is not None:
            controls.append(validated_morph["control"])
        raw_fixed_controls = raw.get("fixed_controls", [])
        if not isinstance(raw_fixed_controls, list) or len(raw_fixed_controls) > 16:
            raise DomainError("Player manifest fixed_controls must be a bounded array")
        fixed_controls: list[dict[str, Any]] = []
        for entry in raw_fixed_controls:
            if not isinstance(entry, dict):
                raise DomainError("Player manifest fixed_controls must contain objects")
            control_id = entry.get("id")
            label = entry.get("label")
            taper = entry.get("taper")
            if not isinstance(control_id, str) or not control_id or len(control_id) > 64:
                raise DomainError("Fixed control id must be a short string")
            if not isinstance(label, str) or not label or len(label) > 64:
                raise DomainError("Fixed control label must be a short string")
            if taper not in {"switch", "linear"}:
                raise DomainError("Fixed controls support switch or linear taper only")
            value = _finite_float(entry.get("value"), f"{control_id} fixed value")
            if control_id in identifiers:
                raise DomainError("Player manifest control ids must be unique across mapped and fixed controls")
            identifiers.add(control_id)
            fixed_control = {"id": control_id, "label": label, "taper": taper, "value": value}
            if taper == "switch":
                if value not in (0.0, 1.0):
                    raise DomainError("Fixed switch value must be 0 or 1")
            else:
                minimum = _finite_float(entry.get("minimum"), f"{control_id} fixed minimum")
                maximum = _finite_float(entry.get("maximum"), f"{control_id} fixed maximum")
                unit = entry.get("unit", "")
                if not minimum < maximum:
                    raise DomainError("Fixed linear control minimum must be less than maximum")
                if not minimum <= value <= maximum:
                    raise DomainError("Fixed linear control value is outside its range")
                if not isinstance(unit, str) or len(unit) > 16:
                    raise DomainError("Fixed linear control unit must be a short string")
                fixed_control.update({"minimum": minimum, "maximum": maximum, "unit": unit})
            fixed_controls.append(fixed_control)
        validated = dict(raw)
        validated["model"] = {**model, "resolved_path": str(model_path), "runtime_quality": runtime_quality}
        validated["presentation"] = presentation
        validated["controls"] = controls
        validated["fixed_controls"] = fixed_controls
        if validated_morph is not None:
            validated["morph"] = validated_morph
        return validated

    def _validate_morph(
        self,
        raw_morph: Any,
        *,
        primary_sample_rate: int,
        model_parameter_index: int,
        identifiers: set[str],
    ) -> dict[str, Any] | None:
        if raw_morph is None:
            if self._packaged_morph_model_path is not None:
                raise DomainError("DessAmp package embeds an unreferenced morph model")
            return None
        if not isinstance(raw_morph, dict) or raw_morph.get("mode") != "dual_model_linear_v1":
            raise DomainError("Player manifest morph mode is unsupported")
        model = raw_morph.get("model")
        if not isinstance(model, dict) or model.get("format") != "nam":
            raise DomainError("Player manifest morph model must use NAM format")
        relative_model = model.get("path")
        if not isinstance(relative_model, str) or not relative_model or len(relative_model) > 512:
            raise DomainError("Player manifest morph model path is invalid")
        relative_path = Path(relative_model)
        if relative_path.is_absolute():
            raise DomainError("Player manifest morph model path must be relative")
        if self._packaged_morph_model_path is not None:
            if relative_model != "morph-model.nam":
                raise DomainError("DessAmp morph package model path must be morph-model.nam")
            model_path = self._packaged_morph_model_path
        else:
            artifact_root = self.manifest_path.parent.resolve()
            model_path = (artifact_root / relative_path).resolve()
            if not model_path.is_relative_to(artifact_root) or not model_path.is_file():
                raise DomainError("Player manifest morph model path escapes or is missing from the artifact")
        digest = _sha256_file(model_path)
        if model.get("sha256") != digest:
            raise DomainError("Player manifest morph model SHA-256 does not match morph-model.nam")
        metadata = inspect_nam_model(model_path)
        for field, expected in (
            ("format_version", metadata["version"]),
            ("architecture", metadata["architecture"]),
            ("sample_rate", metadata["sample_rate"]),
            ("global_condition_size", 0),
        ):
            if model.get(field) != expected:
                raise DomainError(f"Player manifest morph model {field} disagrees with morph-model.nam")
        if model.get("parameter_mode") != "snapshot":
            raise DomainError("Player manifest morph model must be a snapshot")
        if int(metadata["sample_rate"]) != primary_sample_rate:
            raise DomainError("Player manifest morph model sample rate must match the primary model")
        runtime_quality = model.get("runtime_quality")
        if not isinstance(runtime_quality, dict):
            raise DomainError("Player manifest morph runtime_quality must be an object")
        actual_quality = metadata["runtime_quality"]
        if runtime_quality.get("mode") != actual_quality["mode"]:
            raise DomainError("Player manifest morph runtime quality mode disagrees with morph-model.nam")
        if runtime_quality.get("breakpoints", []) != actual_quality["breakpoints"]:
            raise DomainError("Player manifest morph runtime quality breakpoints disagree with morph-model.nam")
        default_quality = _finite_float(runtime_quality.get("default_size", 1.0), "morph runtime default_size")
        if not 0.0 <= default_quality <= 1.0:
            raise DomainError("Player manifest morph runtime default_size must be in [0, 1]")

        alignment = raw_morph.get("alignment")
        if not isinstance(alignment, dict):
            raise DomainError("Player manifest morph alignment must be an object")
        delay_samples = alignment.get("delay_samples")
        polarity = alignment.get("polarity")
        if isinstance(delay_samples, bool) or not isinstance(delay_samples, int) or not 0 <= delay_samples <= 4096:
            raise DomainError("Player manifest morph delay_samples must be an integer in [0, 4096]")
        if isinstance(polarity, bool) or polarity not in (-1, 1):
            raise DomainError("Player manifest morph polarity must be -1 or 1")
        level_scale = _finite_float(alignment.get("level_scale"), "morph level_scale")
        reference_sample_rate = _finite_float(
            alignment.get("reference_sample_rate"), "morph reference_sample_rate"
        )
        if not 0.0 < level_scale <= 100.0:
            raise DomainError("Player manifest morph level_scale is outside the supported range")
        if reference_sample_rate != primary_sample_rate:
            raise DomainError("Player manifest morph reference sample rate must match the models")

        blend = raw_morph.get("blend")
        if not isinstance(blend, dict) or blend.get("law") != "linear":
            raise DomainError("Player manifest morph blend law is unsupported")
        center_gain = _finite_float(blend.get("center_gain"), "morph center_gain")
        smoothing_ms = _finite_float(blend.get("smoothing_ms"), "morph smoothing_ms")
        if not 0.25 <= center_gain <= 4.0:
            raise DomainError("Player manifest morph center_gain is outside the supported range")
        if not 0.0 <= smoothing_ms <= 1000.0:
            raise DomainError("Player manifest morph smoothing_ms is outside the supported range")

        from_label = raw_morph.get("from_label")
        to_label = raw_morph.get("to_label")
        if not isinstance(from_label, str) or not from_label or len(from_label) > 16:
            raise DomainError("Player manifest morph from_label must be a short string")
        if not isinstance(to_label, str) or not to_label or len(to_label) > 16:
            raise DomainError("Player manifest morph to_label must be a short string")
        raw_control = raw_morph.get("control")
        if not isinstance(raw_control, dict):
            raise DomainError("Player manifest morph control must be an object")
        control = ControlDefinition.from_mapping(raw_control)
        if control.id in identifiers:
            raise DomainError("Player manifest control ids must be unique")
        if control.id != "morph" or control.taper == "switch" or control.minimum != 0.0 or control.maximum != 1.0:
            raise DomainError("Player manifest morph control must be continuous and span [0, 1]")
        default_value = _finite_float(raw_control.get("default_value"), "morph default_value")
        if not control.minimum <= default_value <= control.maximum:
            raise DomainError("Player manifest morph default lies outside its range")
        identifiers.add(control.id)
        validated_control = {
            **control.as_dict(),
            "default_value": default_value,
            "model_parameter_index": model_parameter_index,
        }
        return {
            **raw_morph,
            "model": {**model, "resolved_path": str(model_path), "runtime_quality": runtime_quality},
            "alignment": {
                **alignment,
                "delay_samples": delay_samples,
                "polarity": polarity,
                "level_scale": level_scale,
                "reference_sample_rate": reference_sample_rate,
            },
            "blend": {**blend, "center_gain": center_gain, "smoothing_ms": smoothing_ms},
            "control": validated_control,
        }

    @property
    def model_path(self) -> Path:
        return Path(self.manifest["model"]["resolved_path"])

    @property
    def sample_rate(self) -> int:
        return int(self.manifest["model"]["sample_rate"])

    def describe(self) -> dict[str, Any]:
        model = dict(self.manifest["model"])
        model.pop("resolved_path", None)
        description = {
            "schema_version": 1,
            "artifact_format": "dess" if self.package_path is not None else "legacy-manifest",
            "artifact_id": self.manifest.get("artifact_id"),
            "name": self.manifest["name"],
            "capture_project": self.manifest.get("capture_project"),
            "model": model,
            "controls": self.manifest["controls"],
            "fixed_controls": self.manifest["fixed_controls"],
            "presentation": self.manifest["presentation"],
            "runtime_validation": self.manifest.get("runtime_validation"),
            "provenance": self.manifest.get("provenance", {}),
        }
        morph = self.manifest.get("morph")
        if isinstance(morph, dict):
            described_morph = dict(morph)
            morph_model = dict(described_morph["model"])
            morph_model.pop("resolved_path", None)
            described_morph["model"] = morph_model
            description["morph"] = described_morph
        return description

    def validate_input(self, input_path: Path | str) -> dict[str, Any]:
        path = Path(input_path)
        if path.stat().st_size > MAX_RENDER_INPUT_BYTES:
            raise DomainError("Player input exceeds the current 16 MiB Core render limit")
        info = inspect_wav(path)
        if info["channels"] != 1:
            raise DomainError("Player input must be mono")
        if info["sample_rate"] != self.sample_rate:
            raise DomainError(
                f"Player input is {info['sample_rate']} Hz; model requires {self.sample_rate} Hz"
            )
        return info

    def render(
        self,
        *,
        input_path: Path | str,
        output_path: Path | str,
        settings: Mapping[str, Any],
        quality: float | None = None,
    ) -> dict[str, Any]:
        input_file = Path(input_path).resolve()
        output_file = Path(output_path).resolve()
        self.validate_input(input_file)
        controls = self.manifest["controls"]
        expected_ids = {control["id"] for control in controls if control["model_parameter_index"] is not None}
        if set(settings) != expected_ids:
            raise DomainError(f"Player settings must contain exactly these model controls: {sorted(expected_ids)}")
        parameters = [0.0] * sum(
            control["model_parameter_index"] is not None for control in controls
        )
        for control in controls:
            index = control["model_parameter_index"]
            if index is None:
                continue
            value = _finite_float(settings[control["id"]], control["id"])
            if value < control["minimum"] or value > control["maximum"]:
                raise DomainError(f"Control {control['id']!r} lies outside its model range")
            parameters[index] = value
        quality_mode = self.manifest["model"]["runtime_quality"]["mode"]
        selected_quality = (
            _finite_float(
                self.manifest["model"]["runtime_quality"].get("default_size", 1.0)
                if quality is None
                else quality,
                "quality",
            )
        )
        if not 0.0 <= selected_quality <= 1.0:
            raise DomainError("Player quality must be in [0, 1]")
        render_artifact = self.package_path if "morph" in self.manifest else self.model_path
        if render_artifact is None:
            raise DomainError("Morph rendering requires a self-contained .dess package")
        command = [str(self.renderer_path), str(render_artifact), str(input_file), str(output_file)]
        for parameter in parameters:
            command.extend(("--param", format(parameter, ".17g")))
        if quality_mode == "slimmable":
            command.extend(("--quality", format(selected_quality, ".17g")))
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DomainError(f"NAM Core renderer could not complete: {error}") from error
        if result.returncode != 0:
            output_file.unlink(missing_ok=True)
            message = (result.stderr or result.stdout or "Renderer failed").strip()[-4_000:]
            raise DomainError(message)
        try:
            evidence = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            output_file.unlink(missing_ok=True)
            raise DomainError("NAM Core renderer returned invalid evidence") from error
        output_info = inspect_wav(output_file)
        return {
            "settings": {key: float(value) for key, value in settings.items()},
            "quality": selected_quality,
            "renderer": evidence,
            "output": output_info,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise DomainError(f"{label} must be numeric") from error
    if not math.isfinite(parsed):
        raise DomainError(f"{label} must be finite")
    return parsed
