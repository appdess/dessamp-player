"""Render a deterministic two-control teacher-morph training corpus.

The two endpoint teachers are already-selected, externally audited conditioned
``.dess`` packages.  This helper renders both endpoints on one immutable
training DI and one byte-disjoint internal-validation DI, then materializes a
single synthetic surface with controls ordered exactly as ``gain,morph``.

Morph is a parallel-head audio blend, not weight or latent interpolation.  The
blend preserves both endpoints exactly and uses one corpus-wide centre
compensation measured only from the training DI.  No per-setting alignment,
gain normalization, or target-dependent scaling is allowed.

The frozen external DI is intentionally not accepted by this command.  Its
targets must be rendered only after internal-only member selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dessmetal_capture.dess_package import read_dess_package
from dessmetal_capture.wav_io import inspect_wav, load_mono_float32, write_pcm24_mono


RF_SAMPLES = 6_347
MAX_JSON_BYTES = 4 * 1024 * 1024


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


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace immutable evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if not 0 < len(encoded) <= MAX_JSON_BYTES:
        raise ValueError("JSON evidence is empty or exceeds the 4 MiB limit")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(encoded)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def checked_grid(values: Sequence[float], label: str) -> list[float]:
    normalized = [float(value) for value in values]
    if not normalized:
        raise ValueError(f"{label} grid must not be empty")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized):
        raise ValueError(f"{label} grid values must be finite and normalized to [0,1]")
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} grid must be strictly increasing and unique")
    return normalized


def parse_grid(raw: str, label: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError(f"{label} grid contains a non-numeric value") from error
    return checked_grid(values, label)


def checked_input(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    info = inspect_wav(resolved)
    if (
        info["sample_rate"] != 48_000
        or info["channels"] != 1
        or info["sample_width_bits"] != 24
        or info["frames"] <= RF_SAMPLES
    ):
        raise ValueError(f"{label} must be RF-long 48 kHz mono PCM24")
    peak = float(info["peak"])
    if not math.isfinite(peak) or not 0.0 < peak < 1.0:
        raise ValueError(f"{label} is silent, non-finite, or clipped")
    return {
        "path": str(resolved),
        "sha256": str(info["sha256"]),
        "frames": int(info["frames"]),
        "sample_rate": 48_000,
        "channels": 1,
        "sample_width_bits": 24,
        "peak": peak,
    }


def checked_teacher(path: Path, role: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    package = read_dess_package(resolved)
    manifest = package.manifest
    model = manifest.get("model")
    controls = manifest.get("controls")
    if not isinstance(model, Mapping) or model.get("global_condition_size") != 1:
        raise ValueError(f"{role} teacher must contain one conditioned model parameter")
    if model.get("parameter_mode") != "global_condition":
        raise ValueError(f"{role} teacher must use global_condition parameter mode")
    if not isinstance(controls, list) or len(controls) != 1:
        raise ValueError(f"{role} teacher must expose exactly one control")
    control = controls[0]
    if (
        not isinstance(control, Mapping)
        or control.get("model_parameter_index") != 0
        or float(control.get("minimum", math.nan)) != 0.0
        or float(control.get("maximum", math.nan)) != 1.0
    ):
        raise ValueError(f"{role} teacher control must map normalized [0,1] to model index 0")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("workflow_validation_only") is not True:
        raise ValueError(f"{role} teacher must remain workflow_validation_only")
    return {
        "role": role,
        "path": str(resolved),
        "package_sha256": sha256_file(resolved),
        "manifest_sha256": package.manifest_sha256,
        "model_sha256": package.model_sha256,
        "artifact_id": manifest.get("artifact_id"),
        "name": manifest.get("name"),
        "control": dict(control),
        "source_provenance": provenance.get("source_provenance"),
    }


def blend_audio(
    us_audio: np.ndarray,
    brit_audio: np.ndarray,
    morph: float,
    center_gain: float,
) -> np.ndarray:
    if us_audio.shape != brit_audio.shape:
        raise ValueError("US and BRIT endpoint renders must have identical shapes")
    if not 0.0 <= morph <= 1.0:
        raise ValueError("morph must be normalized to [0,1]")
    if not math.isfinite(center_gain) or not 0.25 <= center_gain <= 4.0:
        raise ValueError("center_gain is outside the supported range [0.25,4]")
    compensation = 1.0 + 4.0 * (center_gain - 1.0) * morph * (1.0 - morph)
    return np.asarray(
        compensation * ((1.0 - morph) * us_audio.astype(np.float64) + morph * brit_audio.astype(np.float64)),
        dtype=np.float32,
    )


def calculate_center_gain(
    endpoint_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    receptive_field: int = RF_SAMPLES,
) -> tuple[float, list[float]]:
    if not endpoint_pairs:
        raise ValueError("at least one training endpoint pair is required")
    ratios: list[float] = []
    for us_audio, brit_audio in endpoint_pairs:
        if us_audio.shape != brit_audio.shape or us_audio.size <= receptive_field:
            raise ValueError("training endpoint pair is mismatched or shorter than the receptive field")
        us = us_audio[receptive_field:].astype(np.float64)
        brit = brit_audio[receptive_field:].astype(np.float64)
        us_rms = float(np.sqrt(np.mean(np.square(us))))
        brit_rms = float(np.sqrt(np.mean(np.square(brit))))
        center_rms = float(np.sqrt(np.mean(np.square(0.5 * (us + brit)))))
        if min(us_rms, brit_rms, center_rms) <= 1.0e-9:
            raise ValueError("training endpoint render is effectively silent")
        desired_center_rms = math.sqrt(0.5 * (us_rms * us_rms + brit_rms * brit_rms))
        ratio = desired_center_rms / center_rms
        if not math.isfinite(ratio) or not 0.25 <= ratio <= 4.0:
            raise ValueError("measured centre compensation is outside the supported range")
        ratios.append(ratio)
    return float(statistics.median(ratios)), ratios


def _tag(value: float) -> str:
    return f"{int(round(value * 10_000.0)):04d}"


def _render_endpoint(
    *,
    renderer: Path,
    teacher: Path,
    input_path: Path,
    output_path: Path,
    gain: float,
    expected_input: Mapping[str, Any],
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace endpoint render: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(renderer),
            str(teacher),
            str(input_path),
            str(output_path),
            "--param",
            format(gain, ".10f"),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "renderer failed")[-4_000:]
        raise RuntimeError(detail)
    info = inspect_wav(output_path)
    if (
        info["sample_rate"] != expected_input["sample_rate"]
        or info["channels"] != 1
        or info["sample_width_bits"] != 24
        or info["frames"] != expected_input["frames"]
    ):
        raise ValueError("endpoint render does not exactly match its immutable input")
    peak = float(info["peak"])
    if not math.isfinite(peak) or not 0.0 < peak < 0.999:
        raise ValueError("endpoint render is silent, non-finite, or clipped")
    return {
        "path": str(output_path.resolve()),
        "sha256": str(info["sha256"]),
        "frames": int(info["frames"]),
        "peak": peak,
        "gain": gain,
    }


def _write_target(
    *,
    output_path: Path,
    us_audio: np.ndarray,
    brit_audio: np.ndarray,
    gain: float,
    morph: float,
    center_gain: float,
    sample_rate: int,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace morph target: {output_path}")
    target = blend_audio(us_audio, brit_audio, morph, center_gain)
    peak = float(np.max(np.abs(target)))
    if not math.isfinite(peak) or not 0.0 < peak < 0.999:
        raise ValueError(
            f"morph target g={gain:.6f}, m={morph:.6f} is silent, non-finite, or clipped ({peak})"
        )
    write_pcm24_mono(output_path, target, sample_rate)
    info = inspect_wav(output_path)
    return {
        "path": str(output_path.resolve()),
        "sha256": str(info["sha256"]),
        "frames": int(info["frames"]),
        "peak": float(info["peak"]),
        "g_vector": [gain, morph],
    }


def make_data_config(
    *,
    root: Path,
    training_input: Mapping[str, Any],
    validation_input: Mapping[str, Any],
    train_targets: Sequence[Mapping[str, Any]],
    validation_targets: Sequence[Mapping[str, Any]],
    teachers: Sequence[Mapping[str, Any]],
    renderer: Mapping[str, Any],
    center_gain: float,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    controls = [
        {
            "id": "gain",
            "label": "Gain",
            "minimum": 0.0,
            "maximum": 1.0,
            "display_minimum": 1.0,
            "display_maximum": 10.0,
            "default_value": 4.0 / 9.0,
            "step": 0.01,
            "taper": "linear",
            "unit": "",
        },
        {
            "id": "morph",
            "label": "Morph",
            "minimum": 0.0,
            "maximum": 1.0,
            "display_minimum": 0.0,
            "display_maximum": 10.0,
            "default_value": 0.0,
            "step": 0.01,
            "taper": "linear",
            "unit": "",
        },
    ]
    return {
        "schema_version": 1,
        "type": "shared_in_multi_out",
        "metadata": {
            "project_name": "DESS REDLINE US-BRIT trained morph",
            "amp_name": "DESS REDLINE",
            "controls": controls,
            "control_order": ["gain", "morph"],
            "physical_amplifier": False,
            "source_kind": "second_stage_teacher_model_distillation",
            "source_provenance": dict(provenance),
            "workflow_validation_only": True,
            "training_input_sha256": training_input["sha256"],
            "validation_input_sha256": validation_input["sha256"],
            "validation_strategy": "byte-disjoint DI with interleaved unseen two-dimensional vectors",
            "train_capture_count": len(train_targets),
            "validation_reference_count": len(validation_targets),
            "teacher_packages": list(teachers),
            "renderer": dict(renderer),
            "blend": {
                "law": "linear_with_parabolic_center_compensation_v1",
                "direction": "US_to_BRIT",
                "center_gain": center_gain,
                "delay_samples": 0,
                "polarity": 1,
                "level_scale": 1.0,
                "per_setting_alignment": False,
                "per_setting_normalization": False,
                "formula": "(1+4*(center_gain-1)*m*(1-m))*((1-m)*US+m*BRIT)",
            },
            "external_audit_materialized": False,
            "root": str(root),
        },
        "common": {
            "allow_unequal_lengths": False,
            "delay": 0,
            "input_gain": 0.0,
            "x_path": training_input["path"],
        },
        "train": {
            "x_path": training_input["path"],
            "outputs": [
                {"g_vector": list(target["g_vector"]), "y_path": str(target["path"])}
                for target in train_targets
            ],
        },
        "validation": {
            "x_path": validation_input["path"],
            "outputs": [
                {"g_vector": list(target["g_vector"]), "y_path": str(target["path"])}
                for target in validation_targets
            ],
        },
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    renderer = args.renderer.expanduser().resolve()
    if not renderer.is_file() or not os.access(renderer, os.X_OK):
        raise ValueError(f"renderer is unavailable or not executable: {renderer}")
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to replace immutable corpus root: {output_root}")
    output_root.mkdir(parents=True)

    us_teacher = checked_teacher(args.us_teacher, "US")
    brit_teacher = checked_teacher(args.brit_teacher, "BRIT")
    if us_teacher["package_sha256"] == brit_teacher["package_sha256"]:
        raise ValueError("US and BRIT endpoint packages must be byte-distinct")
    training_input = checked_input(args.train_input, "training input")
    validation_input = checked_input(args.validation_input, "internal-validation input")
    if training_input["sha256"] == validation_input["sha256"]:
        raise ValueError("training and internal-validation inputs must be byte-distinct")

    train_gains = parse_grid(args.train_gains, "training gain")
    train_morphs = parse_grid(args.train_morphs, "training morph")
    validation_gains = parse_grid(args.validation_gains, "validation gain")
    validation_morphs = parse_grid(args.validation_morphs, "validation morph")
    train_vectors = {(gain, morph) for gain in train_gains for morph in train_morphs}
    validation_vectors = {(gain, morph) for gain in validation_gains for morph in validation_morphs}
    if train_vectors & validation_vectors:
        raise ValueError("training and internal-validation vectors must be disjoint")
    if 0.0 not in train_morphs or 1.0 not in train_morphs:
        raise ValueError("training morph grid must contain both exact endpoints")

    endpoint_records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    endpoint_audio: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {
        "train": {},
        "validation": {},
    }
    for split, input_record, gains in (
        ("train", training_input, train_gains),
        ("validation", validation_input, validation_gains),
    ):
        input_path = Path(str(input_record["path"]))
        for gain in gains:
            us_path = output_root / "endpoint-renders" / split / f"{split}-us-g{_tag(gain)}.wav"
            brit_path = output_root / "endpoint-renders" / split / f"{split}-brit-g{_tag(gain)}.wav"
            us_record = _render_endpoint(
                renderer=renderer,
                teacher=Path(us_teacher["path"]),
                input_path=input_path,
                output_path=us_path,
                gain=gain,
                expected_input=input_record,
            )
            brit_record = _render_endpoint(
                renderer=renderer,
                teacher=Path(brit_teacher["path"]),
                input_path=input_path,
                output_path=brit_path,
                gain=gain,
                expected_input=input_record,
            )
            us_audio, us_rate = load_mono_float32(us_path)
            brit_audio, brit_rate = load_mono_float32(brit_path)
            if us_rate != 48_000 or brit_rate != 48_000 or us_audio.shape != brit_audio.shape:
                raise ValueError("endpoint renders disagree on rate or length")
            endpoint_audio[split][gain] = (us_audio, brit_audio)
            endpoint_records[split].append({"gain": gain, "us": us_record, "brit": brit_record})

    center_gain, per_gain_center_gains = calculate_center_gain(
        list(endpoint_audio["train"].values())
    )
    train_targets: list[dict[str, Any]] = []
    validation_targets: list[dict[str, Any]] = []
    for split, gains, morphs, target_records in (
        ("train", train_gains, train_morphs, train_targets),
        ("validation", validation_gains, validation_morphs, validation_targets),
    ):
        for gain in gains:
            us_audio, brit_audio = endpoint_audio[split][gain]
            for morph in morphs:
                target_path = (
                    output_root
                    / "targets"
                    / split
                    / f"{split}-target-g{_tag(gain)}-m{_tag(morph)}.wav"
                )
                target_records.append(
                    _write_target(
                        output_path=target_path,
                        us_audio=us_audio,
                        brit_audio=brit_audio,
                        gain=gain,
                        morph=morph,
                        center_gain=center_gain,
                        sample_rate=48_000,
                    )
                )

    try:
        provenance = json.loads(args.provenance_json)
    except json.JSONDecodeError as error:
        raise ValueError("--provenance-json must be valid JSON") from error
    if not isinstance(provenance, dict):
        raise ValueError("--provenance-json root must be an object")
    provenance = {
        **provenance,
        "purpose": "private DESS REDLINE trained US-to-BRIT morph POC",
        "redistribution_decision": "private workflow-validation only; no source or distilled package redistribution approved",
        "endpoint_teacher_roles": ["US", "BRIT"],
    }
    renderer_record = {"path": str(renderer), "sha256": sha256_file(renderer)}
    data_config = make_data_config(
        root=output_root,
        training_input=training_input,
        validation_input=validation_input,
        train_targets=train_targets,
        validation_targets=validation_targets,
        teachers=[us_teacher, brit_teacher],
        renderer=renderer_record,
        center_gain=center_gain,
        provenance=provenance,
    )
    data_config_path = output_root / "data-config.json"
    evidence_path = output_root / "corpus-evidence.json"
    evidence = {
        "schema_version": 1,
        "status": "materialized_internal_selection_only_external_audit_unopened",
        "root": str(output_root),
        "data_config_path": str(data_config_path),
        "data_config_canonical_sha256": canonical_json_sha256(data_config),
        "control_order": ["gain", "morph"],
        "direction": "US_to_BRIT",
        "teachers": [us_teacher, brit_teacher],
        "renderer": renderer_record,
        "training_input": training_input,
        "validation_input": validation_input,
        "train_gains": train_gains,
        "train_morphs": train_morphs,
        "validation_gains": validation_gains,
        "validation_morphs": validation_morphs,
        "train_vectors_disjoint_from_validation_vectors": True,
        "center_gain": center_gain,
        "per_training_gain_center_gains": per_gain_center_gains,
        "blend_formula": "(1+4*(center_gain-1)*m*(1-m))*((1-m)*US+m*BRIT)",
        "alignment": {"delay_samples": 0, "polarity": 1, "level_scale": 1.0},
        "per_setting_alignment": False,
        "per_setting_normalization": False,
        "endpoint_renders": endpoint_records,
        "train_targets": train_targets,
        "validation_targets": validation_targets,
        "external_audit_materialized": False,
        "provenance": provenance,
    }
    atomic_write_json(data_config_path, data_config)
    atomic_write_json(evidence_path, evidence)
    return {
        "status": "materialized",
        "data_config": str(data_config_path),
        "data_config_sha256": sha256_file(data_config_path),
        "corpus_evidence": str(evidence_path),
        "corpus_evidence_sha256": sha256_file(evidence_path),
        "train_capture_count": len(train_targets),
        "validation_reference_count": len(validation_targets),
        "center_gain": center_gain,
        "external_audit_materialized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a leakage-safe two-control US-to-BRIT teacher-morph corpus."
    )
    parser.add_argument("--us-teacher", type=Path, required=True)
    parser.add_argument("--brit-teacher", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-gains", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--train-morphs", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--validation-gains", default="0.125,0.375,0.625,0.875")
    parser.add_argument("--validation-morphs", default="0.125,0.375,0.625,0.875")
    parser.add_argument("--provenance-json", default="{}")
    return parser


def main() -> None:
    result = materialize(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
