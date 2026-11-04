"""Capture and validate a seven-control Fortin Nameless PANAMA model.

This is an evidence-producing development workflow, not a release trainer.  It
uses the persistent DessMetal capture domain for every recommendation and
capture, preserves unity-gain float renders, derives one measured uniform
safety trim, trains a sequential PANAMA disagreement ensemble, imports its next
point into the Trainer queue, retrains, and packages the result for the
standalone Player.

The frozen discrete mode is High input, Pull Gain off, Grind on at 35%, with
the cabinet and time effects disabled.  The conditioned controls are MVC,
Presence, Bass, Middle, Treble, Gain I, and Gain II, in that exact order.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf

from local_active_learner import (
    CandidateScore,
    evaluate_ensemble_esr,
    score_candidate_grid,
    select_diverse_batch,
    train_ensemble,
    write_new_points,
)


SAMPLE_RATE = 48_000
EXPECTED_CAPTURE_FRAMES = 9_120_000
CONTROL_ORDER = (
    "amp_mvc",
    "amp_presence",
    "amp_bass",
    "amp_middle",
    "amp_treble",
    "amp_gain_1",
    "amp_gain_2",
)
CONTROL_DEFINITIONS = (
    {"id": "amp_mvc", "label": "Master", "minimum": 0.35, "maximum": 0.70, "step": 0.05},
    {"id": "amp_presence", "label": "Presence", "minimum": 0.10, "maximum": 0.70, "step": 0.05},
    {"id": "amp_bass", "label": "Bass", "minimum": 0.30, "maximum": 0.75, "step": 0.05},
    {"id": "amp_middle", "label": "Middle", "minimum": 0.25, "maximum": 0.75, "step": 0.05},
    {"id": "amp_treble", "label": "Treble", "minimum": 0.30, "maximum": 0.75, "step": 0.05},
    {"id": "amp_gain_1", "label": "Gain I", "minimum": 0.35, "maximum": 0.80, "step": 0.05},
    {"id": "amp_gain_2", "label": "Gain II", "minimum": 0.55, "maximum": 1.00, "step": 0.05},
)
HALTON_BASES = (2, 3, 5, 7, 11, 13, 17)


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


DEFAULT_BASE_RUN = _environment_path("DESSAMP_FORTIN_BASE_RUN")
DEFAULT_RUN_ROOT = _environment_path("DESSAMP_FORTIN_RUN_ROOT")
DEFAULT_STRIFE_INPUT = _environment_path("DESSAMP_FORTIN_STRIFE_INPUT")
DEFAULT_LOGIC_PROJECT = _environment_path("DESSAMP_FORTIN_LOGIC_PROJECT")
BASE_MANIFEST_RELATIVE = Path(
    "01_PRESETS/manifests/heads/neuraldsp-fortin-nameless-x-user-default-grind.json"
)
INPUT_RELATIVE = Path("00_WORKFLOW/input.wav")
RENDERER_RELATIVE = Path("00_WORKFLOW/render_fortin_au")
RENDERER_SOURCE_RELATIVE = Path("00_WORKFLOW/render_fortin_au.swift")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _halton(index: int, base: int) -> float:
    value = 0.0
    fraction = 1.0
    while index:
        index, digit = divmod(index, base)
        fraction /= base
        value += digit * fraction
    return value


def _quantized_value(control: Mapping[str, Any], unit_value: float) -> float:
    minimum = float(control["minimum"])
    maximum = float(control["maximum"])
    step = float(control["step"])
    raw = minimum + unit_value * (maximum - minimum)
    steps = round((raw - minimum) / step)
    return round(min(maximum, max(minimum, minimum + steps * step)), 10)


def make_candidate_points(base_settings: Mapping[str, float], count: int) -> list[dict[str, float]]:
    if count < 16 or count > 4096:
        raise ValueError("candidate count must be between 16 and 4096")
    controls = {entry["id"]: entry for entry in CONTROL_DEFINITIONS}
    center = {
        control_id: _quantized_value(controls[control_id], 0.5)
        for control_id in CONTROL_ORDER
    }
    lower = {control_id: float(controls[control_id]["minimum"]) for control_id in CONTROL_ORDER}
    upper = {control_id: float(controls[control_id]["maximum"]) for control_id in CONTROL_ORDER}
    base = {control_id: round(float(base_settings[control_id]), 10) for control_id in CONTROL_ORDER}
    points: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()

    def append(point: Mapping[str, float]) -> None:
        normalized = {control_id: round(float(point[control_id]), 10) for control_id in CONTROL_ORDER}
        key = tuple(normalized[control_id] for control_id in CONTROL_ORDER)
        if key not in seen:
            seen.add(key)
            points.append(normalized)

    append(center)
    append(base)
    append(lower)
    append(upper)
    # Axis endpoints make every individual control boundary observable without
    # requiring the full Cartesian product.
    for control_id in CONTROL_ORDER:
        low_axis = dict(center)
        high_axis = dict(center)
        low_axis[control_id] = lower[control_id]
        high_axis[control_id] = upper[control_id]
        append(low_axis)
        append(high_axis)
    index = 1
    while len(points) < count:
        append(
            {
                control_id: _quantized_value(controls[control_id], _halton(index, base_value))
                for control_id, base_value in zip(CONTROL_ORDER, HALTON_BASES)
            }
        )
        index += 1
    return points[:count]


def select_coverage_holdouts(
    candidate_vectors: Sequence[Sequence[float]],
    captured_vectors: Sequence[Sequence[float]],
    *,
    maximum: int,
    minimum_distance: float,
) -> list[CandidateScore]:
    """Freeze model-independent audit points far from captured coverage."""

    if maximum < 1:
        raise ValueError("maximum must be positive")
    captured = [tuple(float(value) for value in vector) for vector in captured_vectors]
    if not captured:
        raise ValueError("captured_vectors must not be empty")
    dimensions = len(captured[0])
    if dimensions < 1 or any(len(vector) != dimensions for vector in captured):
        raise ValueError("captured vectors must have a consistent non-zero dimension")
    captured_keys = {tuple(round(value, 10) for value in vector) for vector in captured}
    ranked: list[CandidateScore] = []
    for index, raw_vector in enumerate(candidate_vectors):
        vector = tuple(float(value) for value in raw_vector)
        if len(vector) != dimensions or any(not math.isfinite(value) for value in vector):
            raise ValueError("candidate vectors must match the finite captured dimensions")
        if tuple(round(value, 10) for value in vector) in captured_keys:
            continue
        ranked.append(
            CandidateScore(
                source_index=index,
                g_vector=vector,
                disagreement_score=min(math.dist(vector, existing) for existing in captured),
            )
        )
    ranked.sort(key=lambda item: (-item.disagreement_score, item.source_index))
    selected = select_diverse_batch(
        ranked,
        maximum=maximum,
        minimum_distance=minimum_distance,
    )
    if len(selected) != maximum:
        raise RuntimeError(
            f"only {len(selected)} model-independent holdouts satisfy distance {minimum_distance}"
        )
    return selected


def vector_from_settings(settings: Mapping[str, Any]) -> list[float]:
    return [float(settings[control_id]) for control_id in CONTROL_ORDER]


def settings_from_vector(vector: Sequence[float]) -> dict[str, float]:
    if len(vector) != len(CONTROL_ORDER):
        raise ValueError("condition vector has the wrong dimension")
    return {control_id: float(value) for control_id, value in zip(CONTROL_ORDER, vector)}


def make_manifest(
    base_manifest: Mapping[str, Any],
    *,
    settings: Mapping[str, float],
    capture_id: str,
    source_role: str,
    safety_trim_db: float | None,
) -> dict[str, Any]:
    manifest = json.loads(json.dumps(base_manifest))
    recipe = manifest["head_capture_recipe"]
    for control_id in CONTROL_ORDER:
        recipe[control_id] = float(settings[control_id])
    manifest["id"] = capture_id
    manifest["schema"] = "dess.nam.capture.conditioned.v1"
    manifest["capture_type"] = "head-only-conditioned"
    manifest["conditioning_contract"] = {
        "control_order": list(CONTROL_ORDER),
        "g_vector": vector_from_settings(settings),
        "source_role": source_role,
        "continuous_controls": [dict(entry) for entry in CONTROL_DEFINITIONS],
        "discrete_mode": {
            "amp_input_high": True,
            "amp_pull_gain": False,
            "grind_active": True,
            "grind_boost": 0.35,
            "cabinet": "off",
            "graphic_eq": "flat",
        },
    }
    manifest["capture_calibration"] = {
        "unity_float_master_preserved": True,
        "uniform_post_amp_safety_trim_db": safety_trim_db,
        "pcm24_derivation_gain_db": 0.0,
    }
    manifest["training_contract"] = {
        "architecture": "PANAMA conditioned WaveNet",
        "global_condition_size": len(CONTROL_ORDER),
        "purpose": "functional active-learning and standalone-player validation",
        "release_model": False,
    }
    return manifest


def inspect_float_wav(path: Path, *, expected_frames: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or audio.shape[1] != 1:
        raise RuntimeError(f"{path} is not mono {SAMPLE_RATE} Hz")
    mono = audio[:, 0]
    if expected_frames is not None and mono.size != expected_frames:
        raise RuntimeError(f"{path} has {mono.size} frames; expected {expected_frames}")
    if not np.all(np.isfinite(mono)):
        raise RuntimeError(f"{path} contains non-finite samples")
    peak = float(np.max(np.abs(mono)))
    return mono, {
        "sample_rate": rate,
        "channels": 1,
        "frames": int(mono.size),
        "duration_seconds": mono.size / rate,
        "peak": peak,
        "peak_dbfs": 20.0 * math.log10(peak) if peak else None,
        "sha256": sha256_file(path),
    }


def write_pcm24_exact(path: Path, audio: np.ndarray) -> float:
    import wave

    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(mono)) or float(np.max(np.abs(mono))) >= 1.0:
        raise RuntimeError("PCM24 source must be finite and strictly below full scale")
    integers = np.rint(mono.astype(np.float64) * 8_388_608.0).astype("<i4")
    integers = np.clip(integers, -8_388_608, 8_388_607).astype("<i4", copy=False)
    pcm24 = integers.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(3)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm24)
    decoded, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE or decoded.shape != mono.shape:
        raise RuntimeError("PCM24 round trip changed the audio contract")
    delta = float(np.max(np.abs(decoded - mono)))
    if delta > 1.0 / 8_388_608.0:
        raise RuntimeError(f"PCM24 round-trip delta {delta} exceeds one 24-bit LSB")
    return delta


def render_plugin(
    renderer: Path,
    *,
    manifest_path: Path,
    input_path: Path,
    output_path: Path,
    expected_frames: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(renderer),
            "--mode",
            "head",
            "--manifest",
            str(manifest_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Fortin renderer failed")[-8_000:])
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Fortin renderer returned invalid evidence") from error
    _, wav_info = inspect_float_wav(output_path, expected_frames=expected_frames)
    report["unity_float"] = wav_info
    return report


def validate_control_sensitivity(
    *,
    run_root: Path,
    renderer: Path,
    input_path: Path,
    base_manifest: Mapping[str, Any],
    base_settings: Mapping[str, float],
) -> dict[str, Any]:
    """Require every declared conditioned control to affect true-High audio."""

    probe_root = run_root / "05_VALIDATION" / "control-sensitivity"
    rows: list[dict[str, Any]] = []
    for control in CONTROL_DEFINITIONS:
        control_id = str(control["id"])
        rendered: list[tuple[Path, np.ndarray, dict[str, Any]]] = []
        for label, value in (("low", control["minimum"]), ("high", control["maximum"])):
            settings = {key: float(item) for key, item in base_settings.items()}
            settings[control_id] = float(value)
            manifest = make_manifest(
                base_manifest,
                settings=settings,
                capture_id=f"sensitivity-{control_id}-{label}",
                source_role="pre-training-control-sensitivity",
                safety_trim_db=None,
            )
            manifest_path = probe_root / "manifests" / f"{control_id}-{label}.json"
            output_path = probe_root / "audio" / f"{control_id}-{label}.wav"
            write_json(manifest_path, manifest)
            render_plugin(
                renderer,
                manifest_path=manifest_path,
                input_path=input_path,
                output_path=output_path,
                expected_frames=int(sf.info(input_path).frames),
            )
            audio, info = inspect_float_wav(output_path)
            rendered.append((output_path, audio, info))
        low_path, low_audio, low_info = rendered[0]
        high_path, high_audio, high_info = rendered[1]
        delta = low_audio.astype(np.float64) - high_audio.astype(np.float64)
        delta_rms = float(np.sqrt(np.mean(delta * delta)))
        reference_power = float(np.mean(high_audio.astype(np.float64) ** 2))
        row = {
            "control": control_id,
            "low": float(control["minimum"]),
            "high": float(control["maximum"]),
            "low_path": str(low_path),
            "high_path": str(high_path),
            "low_sha256": low_info["sha256"],
            "high_sha256": high_info["sha256"],
            "byte_identical": low_info["sha256"] == high_info["sha256"],
            "delta_rms": delta_rms,
            "esr_low_vs_high": float(np.mean(delta * delta) / reference_power),
        }
        if row["byte_identical"] or delta_rms <= 1.0e-6:
            raise RuntimeError(
                f"conditioned control {control_id!r} is audio-inactive in the frozen mode"
            )
        rows.append(row)
    report = {
        "status": "pass",
        "frozen_amp_input": "High",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "all_declared_controls_audio_sensitive": True,
        "controls": rows,
    }
    write_json(probe_root / "report.json", report)
    return report


def derive_safety_trim_db(peak: float, *, target_dbfs: float = -1.0) -> float:
    if peak <= 0.0:
        raise RuntimeError("safety pilot is silent")
    peak_dbfs = 20.0 * math.log10(peak)
    if peak_dbfs <= target_dbfs:
        return 0.0
    required = target_dbfs - peak_dbfs
    # Round downward to the next half dB so this is a fixed safety margin, not
    # per-capture normalization.
    return math.floor(required * 2.0) / 2.0


def validate_official(input_path: Path, output_path: Path) -> tuple[dict[str, Any], str]:
    from nam.train.core import validate_data

    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        validation = validate_data(input_path, output_path, None)
    if hasattr(validation, "model_dump"):
        evidence = validation.model_dump(mode="json")
    else:
        evidence = json.loads(validation.json())
    if not evidence.get("passed_critical"):
        raise RuntimeError(f"Official NAM validation failed critically for {output_path}")
    return evidence, log.getvalue()


def render_capture(
    *,
    run_root: Path,
    renderer: Path,
    input_path: Path,
    base_manifest: Mapping[str, Any],
    settings: Mapping[str, float],
    capture_id: str,
    source_role: str,
    safety_trim_db: float,
    official_validation: bool,
) -> dict[str, Any]:
    from dessmetal_capture.wav_io import inspect_wav, validate_capture_wav

    manifest = make_manifest(
        base_manifest,
        settings=settings,
        capture_id=capture_id,
        source_role=source_role,
        safety_trim_db=safety_trim_db,
    )
    manifest_path = run_root / "01_MANIFESTS" / f"{capture_id}.json"
    raw_path = run_root / "02_CAPTURES" / "unity-float" / f"{capture_id}__unity-float.wav"
    pcm_path = run_root / "02_CAPTURES" / "training-pcm24" / f"{capture_id}__train-pcm24.wav"
    report_path = run_root / "05_VALIDATION" / "render-reports" / f"{capture_id}.json"
    write_json(manifest_path, manifest)
    render_report = render_plugin(
        renderer,
        manifest_path=manifest_path,
        input_path=input_path,
        output_path=raw_path,
        expected_frames=EXPECTED_CAPTURE_FRAMES if official_validation else int(sf.info(input_path).frames),
    )
    raw, raw_info = inspect_float_wav(raw_path)
    scale = 10.0 ** (safety_trim_db / 20.0)
    trimmed = raw * scale
    trimmed_peak = float(np.max(np.abs(trimmed)))
    if trimmed_peak >= 1.0:
        raise RuntimeError(
            f"Uniform safety trim {safety_trim_db:.1f} dB is insufficient for {capture_id}: peak={trimmed_peak}"
        )
    quantization_delta = write_pcm24_exact(pcm_path, trimmed)
    wav_info = inspect_wav(pcm_path)
    status, capture_validation = validate_capture_wav(
        wav_info,
        expected_sample_rate=SAMPLE_RATE,
        expected_frames=int(sf.info(input_path).frames),
    )
    official = None
    official_log = ""
    if official_validation:
        official, official_log = validate_official(input_path, pcm_path)
    if status == "failed":
        raise RuntimeError(f"Capture-domain validation failed for {capture_id}: {capture_validation['failures']}")
    evidence = {
        "capture_id": capture_id,
        "source_role": source_role,
        "settings": {key: float(value) for key, value in settings.items()},
        "g_vector": vector_from_settings(settings),
        "manifest_path": str(manifest_path),
        "unity_float_path": str(raw_path),
        "pcm24_path": str(pcm_path),
        "unity_float": raw_info,
        "uniform_safety_trim_db": safety_trim_db,
        "trimmed_peak": trimmed_peak,
        "trimmed_peak_dbfs": 20.0 * math.log10(trimmed_peak) if trimmed_peak else None,
        "pcm24_roundtrip_max_delta": quantization_delta,
        "wav_info": wav_info,
        "capture_validation": capture_validation,
        "official_nam_validation": official,
        "official_nam_validator_log": official_log,
        "renderer": render_report,
    }
    write_json(report_path, evidence)
    return evidence


def model_config_for_dimensions(template: Mapping[str, Any], dimensions: int) -> dict[str, Any]:
    config = json.loads(json.dumps(template))
    config["net"]["global_condition_size"] = dimensions
    for layer in config["net"]["config"]["layers_configs"]:
        layer["global_condition_size"] = dimensions
    return config


def make_data_config(
    input_path: Path,
    captures: Sequence[Mapping[str, Any]],
    *,
    delay_samples: int = 0,
) -> dict[str, Any]:
    if len(captures) < 2:
        raise ValueError("conditioned training requires at least two captures")
    if isinstance(delay_samples, bool) or not isinstance(delay_samples, int):
        raise ValueError("delay_samples must be an integer")
    outputs = [
        {"y_path": str(Path(capture["pcm24_path"]).resolve()), "g_vector": list(capture["g_vector"])}
        for capture in captures
    ]
    return {
        "type": "shared_in_multi_out",
        "common": {
            "x_path": str(input_path.resolve()),
            "input_gain": 0.0,
            "delay": delay_samples,
            "allow_unequal_lengths": False,
        },
        # Keep the center capture out of gradient updates so checkpoint choice
        # sees one complete, real held-out knob setting.
        "train": {"outputs": outputs[1:]},
        "validation": {"outputs": [outputs[0]]},
    }


def make_functional_training_view(
    *,
    output_dir: Path,
    input_path: Path,
    captures: Sequence[Mapping[str, Any]],
    start_seconds: float,
    duration_seconds: float,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    """Create a bounded real-audio view while retaining full validated masters."""

    if start_seconds < 0.0 or duration_seconds <= 0.0:
        raise ValueError("functional training start must be non-negative and duration must be positive")
    start_frame = int(round(start_seconds * SAMPLE_RATE))
    frame_count = int(round(duration_seconds * SAMPLE_RATE))
    stop_frame = start_frame + frame_count
    input_audio, input_rate = sf.read(input_path, dtype="float32", always_2d=True)
    if input_rate != SAMPLE_RATE or input_audio.shape[1] != 1 or stop_frame > input_audio.shape[0]:
        raise RuntimeError("functional training view lies outside the standardized mono 48 kHz input")
    output_dir.mkdir(parents=True, exist_ok=False)
    view_input = output_dir / "input.wav"
    write_pcm24_exact(view_input, input_audio[start_frame:stop_frame, 0])
    view_captures: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {"input": sha256_file(input_path)}
    for index, capture in enumerate(captures):
        source = Path(capture["pcm24_path"])
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE or audio.shape[1] != 1 or stop_frame > audio.shape[0]:
            raise RuntimeError(f"functional training view lies outside capture {source}")
        destination = output_dir / f"capture-{index:02d}.wav"
        write_pcm24_exact(destination, audio[start_frame:stop_frame, 0])
        view_capture = dict(capture)
        view_capture["pcm24_path"] = str(destination)
        view_captures.append(view_capture)
        source_hashes[f"capture-{index:02d}"] = sha256_file(source)
    evidence = {
        "purpose": "bounded real-audio functional training while the production MPS queue is occupied",
        "start_seconds": start_seconds,
        "duration_seconds": duration_seconds,
        "start_frame": start_frame,
        "frames": frame_count,
        "input_path": str(view_input),
        "input_sha256": sha256_file(view_input),
        "source_hashes": source_hashes,
        "full_validated_masters_retained": True,
        "release_quality_claim": False,
    }
    write_json(output_dir / "training-view-evidence.json", evidence)
    return view_input, view_captures, evidence


def _copy_required(source: Path, destination: Path, *, executable: bool = False) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if executable:
        destination.chmod(destination.stat().st_mode | 0o111)


def prepare(args: argparse.Namespace) -> None:
    from dessmetal_capture.store import CaptureStore
    from dessmetal_capture.wav_io import inspect_wav

    run_root = args.run_root.expanduser().resolve()
    if run_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing run root: {run_root}")
    workflow = run_root / "00_WORKFLOW"
    workflow.mkdir(parents=True)
    base_run = args.base_run.expanduser().resolve()
    if (args.renderer_source is None) != (args.renderer_binary is None):
        raise RuntimeError("--renderer-source and --renderer-binary must be supplied together")
    input_path = workflow / "input.wav"
    renderer = workflow / "render_fortin_au"
    base_manifest_path = workflow / "base-fortin-manifest.json"
    _copy_required(base_run / INPUT_RELATIVE, input_path)
    renderer_source = (
        args.renderer_source.expanduser().resolve()
        if args.renderer_source is not None
        else base_run / RENDERER_SOURCE_RELATIVE
    )
    renderer_binary = (
        args.renderer_binary.expanduser().resolve()
        if args.renderer_binary is not None
        else base_run / RENDERER_RELATIVE
    )
    copied_renderer_source = workflow / "render_fortin_au.swift"
    _copy_required(renderer_binary, renderer, executable=True)
    _copy_required(renderer_source, copied_renderer_source)
    _copy_required(base_run / BASE_MANIFEST_RELATIVE, base_manifest_path)
    _copy_required(Path(__file__).resolve(), workflow / Path(__file__).name)
    if sha256_file(input_path) != sha256_file(base_run / INPUT_RELATIVE):
        raise RuntimeError("standardized capture input changed during copy")

    base_manifest = read_json(base_manifest_path)
    base_settings = {
        control_id: float(base_manifest["head_capture_recipe"][control_id])
        for control_id in CONTROL_ORDER
    }
    candidates = make_candidate_points(base_settings, args.candidate_count)
    write_json(workflow / "candidate-points.json", candidates)

    strife_source = args.strife_input.expanduser().resolve()
    strife, strife_rate = sf.read(strife_source, dtype="float32", always_2d=True)
    if strife_rate != SAMPLE_RATE or strife.shape[1] != 1 or not np.all(np.isfinite(strife)):
        raise RuntimeError("Strife audition input must be finite mono 48 kHz")
    strife_input = workflow / "strife-di-pcm24.wav"
    write_pcm24_exact(strife_input, strife[:, 0])

    sensitivity_report = validate_control_sensitivity(
        run_root=run_root,
        renderer=renderer,
        input_path=strife_input,
        base_manifest=base_manifest,
        base_settings=base_settings,
    )

    store = CaptureStore(run_root / "04_TRAINER_DATA")
    project = store.create_project(
        project_id="fortin-nameless-seven-control",
        name="Fortin Nameless seven-control active capture",
        amp_name="Neural DSP Fortin Nameless Suite X",
        description=(
            "Functional real-plugin PANAMA validation: all seven continuous amp controls; High input, "
            "Pull Gain off, Grind 35%, flat post EQ, cabinet and time effects off."
        ),
        controls=[
            {
                **entry,
                "unit": "%",
                "display_minimum": float(entry["minimum"]) * 100.0,
                "display_maximum": float(entry["maximum"]) * 100.0,
            }
            for entry in CONTROL_DEFINITIONS
        ],
        candidate_points=candidates,
        acquisition_strategy="coverage-v1",
        metadata={
            "purpose": "functional-active-learning-validation",
            "plugin": "Neural DSP Fortin Nameless Suite X 1.0.1",
            "control_order": list(CONTROL_ORDER),
            "logic_project_copy": str(args.logic_project.expanduser().resolve()),
        },
    )
    store.set_input_signal(project["id"], input_path, inspect_wav(input_path))

    upper = {entry["id"]: float(entry["maximum"]) for entry in CONTROL_DEFINITIONS}
    pilot_manifest = make_manifest(
        base_manifest,
        settings=upper,
        capture_id="safety-pilot-upper-envelope",
        source_role="safety-pilot-not-training",
        safety_trim_db=None,
    )
    pilot_manifest_path = run_root / "01_MANIFESTS" / "safety-pilot-upper-envelope.json"
    pilot_path = run_root / "02_CAPTURES" / "rejected-or-pilot" / "safety-pilot-upper-envelope__unity-float.wav"
    write_json(pilot_manifest_path, pilot_manifest)
    pilot_report = render_plugin(
        renderer,
        manifest_path=pilot_manifest_path,
        input_path=input_path,
        output_path=pilot_path,
        expected_frames=EXPECTED_CAPTURE_FRAMES,
    )
    safety_trim_db = derive_safety_trim_db(float(pilot_report["unity_float"]["peak"]))
    pilot_report["selected_uniform_safety_trim_db"] = safety_trim_db
    pilot_report["policy"] = "one uniform post-amp trim from the upper-envelope pilot; no per-capture normalization"
    write_json(run_root / "05_VALIDATION" / "safety-pilot.json", pilot_report)

    starter_captures: list[dict[str, Any]] = []
    recommendation_evidence: list[dict[str, Any]] = []
    for starter_index in range(args.starter_count):
        recommendation = store.create_recommendation(project["id"])
        capture_id = f"starter-{starter_index:02d}"
        evidence = render_capture(
            run_root=run_root,
            renderer=renderer,
            input_path=input_path,
            base_manifest=base_manifest,
            settings=recommendation["settings"],
            capture_id=capture_id,
            source_role="coverage-starter",
            safety_trim_db=safety_trim_db,
            official_validation=True,
        )
        capture = store.add_capture(
            project_id=project["id"],
            recommendation_id=recommendation["id"],
            source="offline-fortin-au",
            wav_path=Path(evidence["pcm24_path"]),
            wav_info=evidence["wav_info"],
            quality_status=evidence["capture_validation"]["status"],
            validation={
                "capture": evidence["capture_validation"],
                "official_nam": evidence["official_nam_validation"],
                "uniform_safety_trim_db": safety_trim_db,
            },
            notes="Autonomous deterministic AU render from the persistent coverage recommendation.",
        )
        starter_captures.append(evidence)
        recommendation_evidence.append(
            {
                "recommendation_id": recommendation["id"],
                "round_index": recommendation["round_index"],
                "strategy": recommendation["strategy"],
                "rationale": recommendation["rationale"],
                "score": recommendation["score"],
                "settings": recommendation["settings"],
                "capture_id": capture["id"],
            }
        )

    trainer_root = Path(__file__).resolve().parent
    model_config = model_config_for_dimensions(
        read_json(trainer_root / "parametric_wavenet.json"), len(CONTROL_ORDER)
    )
    learning_config = read_json(trainer_root / "default_config_files" / "learning" / "default.json")
    data_config = make_data_config(
        input_path,
        starter_captures,
        delay_samples=args.delay_samples,
    )
    write_json(workflow / "starter-data-config.json", data_config)
    write_json(workflow / "model-config-7d.json", model_config)
    write_json(workflow / "learning-config.json", learning_config)
    write_json(
        workflow / "candidate-config.json",
        {
            "control_order": list(CONTROL_ORDER),
            "candidate_vectors": [vector_from_settings(point) for point in candidates],
            "captured_vectors": [capture["g_vector"] for capture in starter_captures],
        },
    )
    state = {
        "schema_version": 1,
        "status": "prepared",
        "run_root": str(run_root),
        "project_id": project["id"],
        "control_order": list(CONTROL_ORDER),
        "base_settings": base_settings,
        "candidate_count": len(candidates),
        "starter_count": len(starter_captures),
        "training_delay_samples": args.delay_samples,
        "uniform_safety_trim_db": safety_trim_db,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "strife_input_path": str(strife_input),
        "strife_input_sha256": sha256_file(strife_input),
        "renderer_path": str(renderer),
        "renderer_sha256": sha256_file(renderer),
        "renderer_source_path": str(copied_renderer_source),
        "renderer_source_sha256": sha256_file(copied_renderer_source),
        "control_sensitivity_report": str(
            run_root / "05_VALIDATION" / "control-sensitivity" / "report.json"
        ),
        "all_declared_controls_audio_sensitive": sensitivity_report[
            "all_declared_controls_audio_sensitive"
        ],
        "base_manifest_path": str(base_manifest_path),
        "base_manifest_sha256": sha256_file(base_manifest_path),
        "starter_captures": starter_captures,
        "recommendations": recommendation_evidence,
        "logic_project_copy": str(args.logic_project.expanduser().resolve()),
        "claims": [
            "real Fortin AU capture and persistent workflow evidence",
            "not yet trained",
            "not a tonal or release-quality model claim",
        ],
    }
    write_json(workflow / "run-state.json", state)
    write_json(run_root / "05_VALIDATION" / "prepare-report.json", state)
    print(json.dumps({"status": "prepared", "run_root": str(run_root), "captures": len(starter_captures)}))


def export_member(trainer_root: Path, member_directory: Path, export_directory: Path) -> Path:
    result = subprocess.run(
        [sys.executable, str(trainer_root / "export_model.py"), str(member_directory), str(export_directory)],
        cwd=trainer_root.parent,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "conditioned model export failed")[-8_000:])
    model_path = export_directory / "model.nam"
    if not model_path.is_file():
        raise RuntimeError("conditioned model export produced no model.nam")
    return model_path


def train(args: argparse.Namespace) -> None:
    from dessmetal_capture.domain import ControlDefinition
    from dessmetal_capture.manifests import import_model_artifact
    from dessmetal_capture.panama import parse_new_points, source_digest
    from dessmetal_capture.store import CaptureStore
    from dessmetal_player.artifact import PlayerArtifact

    run_root = args.run_root.expanduser().resolve()
    workflow = run_root / "00_WORKFLOW"
    state_path = workflow / "run-state.json"
    state = read_json(state_path)
    if state["status"] != "prepared":
        raise RuntimeError(f"run state must be prepared, got {state['status']!r}")
    training_root = run_root / "03_TRAINING"
    if training_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing training output: {training_root}")
    _copy_required(Path(__file__).resolve(), workflow / "training-orchestrator.py")
    input_path = Path(state["input_path"])
    renderer = Path(state["renderer_path"])
    base_manifest = read_json(Path(state["base_manifest_path"]))
    candidate_config = read_json(workflow / "candidate-config.json")
    model_config = read_json(workflow / "model-config-7d.json")
    learning_config = read_json(workflow / "learning-config.json")
    starter_captures = list(state["starter_captures"])
    training_delay_samples = int(state.get("training_delay_samples", 0))

    training_input, training_starters, training_view = make_functional_training_view(
        output_dir=training_root / "functional-view-initial",
        input_path=input_path,
        captures=starter_captures,
        start_seconds=args.training_start_seconds,
        duration_seconds=args.training_duration_seconds,
    )
    starter_data = make_data_config(
        training_input,
        training_starters,
        delay_samples=training_delay_samples,
    )
    write_json(workflow / "functional-starter-data-config.json", starter_data)

    initial_checkpoints = train_ensemble(
        data_config=starter_data,
        model_config=model_config,
        learning_config=learning_config,
        output_dir=training_root / "initial-ensemble",
        backend=args.backend,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        seed=args.seed,
    )
    ranking = score_candidate_grid(
        checkpoint_paths=initial_checkpoints,
        model_config=model_config,
        input_path=training_input,
        candidate_vectors=candidate_config["candidate_vectors"],
        captured_vectors=candidate_config["captured_vectors"],
        backend=args.backend,
        chunk_size=args.score_chunk_size,
        batch_size=args.score_batch_size,
        max_chunks=args.max_score_chunks,
    )
    selected_points = write_new_points(workflow / "new_points.json", ranking, maximum=1)

    store = CaptureStore(run_root / "04_TRAINER_DATA")
    project = store.get_project(state["project_id"])
    controls = [ControlDefinition.from_mapping(control) for control in project["controls"]]
    suggestions = parse_new_points(selected_points, controls=controls, control_order=CONTROL_ORDER)
    queued = store.enqueue_panama_suggestions(
        project["id"],
        round_index=0,
        source_sha256=source_digest(selected_points),
        control_order=CONTROL_ORDER,
        suggestions=suggestions,
    )
    recommendation = store.create_recommendation(project["id"])
    if recommendation["strategy"] != "panama-ensemble-v1":
        raise RuntimeError("Trainer did not activate the PANAMA disagreement recommendation")
    selected_capture = render_capture(
        run_root=run_root,
        renderer=renderer,
        input_path=input_path,
        base_manifest=base_manifest,
        settings=recommendation["settings"],
        capture_id="active-round-00",
        source_role="panama-ensemble-disagreement",
        safety_trim_db=float(state["uniform_safety_trim_db"]),
        official_validation=True,
    )
    stored_selected = store.add_capture(
        project_id=project["id"],
        recommendation_id=recommendation["id"],
        source="offline-fortin-au",
        wav_path=Path(selected_capture["pcm24_path"]),
        wav_info=selected_capture["wav_info"],
        quality_status=selected_capture["capture_validation"]["status"],
        validation={
            "capture": selected_capture["capture_validation"],
            "official_nam": selected_capture["official_nam_validation"],
            "selection": selected_points[0],
        },
        notes="Rendered only after the initial ensemble selected this physical knob vector.",
    )
    selected_vector = list(selected_capture["g_vector"])
    if selected_vector not in candidate_config["captured_vectors"]:
        candidate_config["captured_vectors"].append(selected_vector)
    write_json(workflow / "candidate-config.json", candidate_config)

    all_captures = [*starter_captures, selected_capture]
    enriched_training_input, enriched_training_captures, enriched_training_view = make_functional_training_view(
        output_dir=training_root / "functional-view-enriched",
        input_path=input_path,
        captures=all_captures,
        start_seconds=args.training_start_seconds,
        duration_seconds=args.training_duration_seconds,
    )
    enriched_data = make_data_config(
        enriched_training_input,
        enriched_training_captures,
        delay_samples=training_delay_samples,
    )
    write_json(workflow / "enriched-data-config.json", enriched_data)
    enriched_checkpoints = train_ensemble(
        data_config=enriched_data,
        model_config=model_config,
        learning_config=learning_config,
        output_dir=training_root / "enriched-ensemble",
        backend=args.backend,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        seed=args.seed,
    )

    captured_keys = {tuple(capture["g_vector"]) for capture in all_captures}
    selected_vector = list(selected_points[0]["g_vector"])
    evaluation_vectors: list[list[float]] = [selected_vector]
    for item in ranking[1:]:
        vector = list(item.g_vector)
        if tuple(vector) not in captured_keys:
            evaluation_vectors.append(vector)
        if len(evaluation_vectors) == args.hidden_count:
            break
    if len(evaluation_vectors) < args.hidden_count:
        raise RuntimeError("not enough uncaptured candidates for hidden evaluation")

    hidden_references: list[tuple[list[float], Path]] = []
    hidden_evidence: list[dict[str, Any]] = []
    strife_input = Path(state["strife_input_path"])
    evaluation_roles: list[str] = []
    for index, vector in enumerate(evaluation_vectors):
        role = "active-selected-on-unseen-di" if index == 0 else "uncaptured-setting-on-unseen-di"
        evidence = render_capture(
            run_root=run_root,
            renderer=renderer,
            input_path=strife_input,
            base_manifest=base_manifest,
            settings=settings_from_vector(vector),
            capture_id=f"evaluation-strife-{index:02d}",
            source_role=role,
            safety_trim_db=float(state["uniform_safety_trim_db"]),
            official_validation=False,
        )
        hidden_references.append((vector, Path(evidence["pcm24_path"])))
        hidden_evidence.append(evidence)
        evaluation_roles.append(role)

    candidate_config["reserved_evaluation_vectors"] = evaluation_vectors[1:]
    candidate_config["reserved_policy"] = (
        "Never enqueue or capture these vectors during this validation run; "
        "they remain hidden Strife-DI audit settings."
    )
    write_json(workflow / "candidate-config.json", candidate_config)

    initial_metrics = evaluate_ensemble_esr(
        checkpoint_paths=initial_checkpoints,
        model_config=model_config,
        input_path=strife_input,
        references=hidden_references,
        backend=args.backend,
        chunk_size=args.score_chunk_size,
        delay_samples=training_delay_samples,
    )
    enriched_metrics = evaluate_ensemble_esr(
        checkpoint_paths=enriched_checkpoints,
        model_config=model_config,
        input_path=strife_input,
        references=hidden_references,
        backend=args.backend,
        chunk_size=args.score_chunk_size,
        delay_samples=training_delay_samples,
    )

    trainer_root = Path(__file__).resolve().parent
    exported_model = export_member(
        trainer_root,
        training_root / "enriched-ensemble" / "ensemble-member-00",
        training_root / "exported-model",
    )
    artifact = import_model_artifact(
        store,
        project_id=project["id"],
        source_model_path=exported_model,
        name="Fortin Nameless seven-control conditioned WaveNet",
        provenance={
            "purpose": "real-plugin-active-learning-validation",
            "workflow_validation_only": True,
            "control_order": list(CONTROL_ORDER),
            "starter_g_vectors": [capture["g_vector"] for capture in starter_captures],
            "selected_point": selected_points[0],
            "selection_method": "panama-ensemble-disagreement-finite-grid",
            "uniform_safety_trim_db": state["uniform_safety_trim_db"],
            "backend": args.backend,
            "ensemble_size": args.ensemble_size,
            "epochs": args.epochs,
        },
        loader_path=args.loader.expanduser().resolve(),
    )
    if artifact["validation_status"] != "valid":
        raise RuntimeError(f"native NAM loader rejected or did not verify the artifact: {artifact}")

    manifest_path = Path(artifact["manifest_path"])
    player_manifest = read_json(manifest_path)
    for control in player_manifest["controls"]:
        control["default_value"] = float(state["base_settings"][control["id"]])
    write_json(manifest_path, player_manifest)
    manifest_loader = subprocess.run(
        [str(args.manifest_loader.expanduser().resolve()), str(manifest_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if manifest_loader.returncode != 0:
        raise RuntimeError((manifest_loader.stderr or manifest_loader.stdout or "manifest loader failed")[-8_000:])

    player = PlayerArtifact(manifest_path, args.renderer.expanduser().resolve())
    selected_settings = recommendation["settings"]
    alternate_settings = settings_from_vector(evaluation_vectors[1])
    render_settings = [
        ("base", state["base_settings"]),
        ("active-selected", selected_settings),
        ("alternate", alternate_settings),
    ]
    player_renders: list[dict[str, Any]] = []
    for label, settings in render_settings:
        output_path = run_root / "06_PLAYER" / "renders" / f"{label}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evidence = player.render(input_path=strife_input, output_path=output_path, settings=settings)
        player_renders.append(
            {
                "label": label,
                "settings": settings,
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "peak_dbfs": evidence["output"]["peak_dbfs"],
                "frames": evidence["output"]["frames"],
            }
        )

    initial_mean = float(initial_metrics["mean_esr"])
    enriched_mean = float(enriched_metrics["mean_esr"])
    report = {
        "status": "passed" if enriched_mean < initial_mean else "completed-without-measured-improvement",
        "claim": "real Fortin plug-in functional evidence; human tonal acceptance still required",
        "control_order": list(CONTROL_ORDER),
        "control_count": len(CONTROL_ORDER),
        "control_ranges": [dict(entry) for entry in CONTROL_DEFINITIONS],
        "frozen_mode": {
            "amp_input": "High",
            "pull_gain": "off",
            "grind": 0.35,
            "graphic_eq": "flat",
            "cabinet_and_time_effects": "off",
        },
        "candidate_count": state["candidate_count"],
        "starter_count": len(starter_captures),
        "selected_point": selected_points[0],
        "trainer_queue": {
            "queued_strategy": queued[0]["strategy"],
            "recommendation_id": recommendation["id"],
            "recommendation_strategy": recommendation["strategy"],
            "capture_id": stored_selected["id"],
            "capture_settings": stored_selected["settings"],
        },
        "backend": args.backend,
        "ensemble_size": args.ensemble_size,
        "epochs": args.epochs,
        "training_view": training_view,
        "enriched_training_view": enriched_training_view,
        "evaluation_settings": [
            {"role": role, "g_vector": vector}
            for role, vector in zip(evaluation_roles, evaluation_vectors)
        ],
        "initial_hidden_metrics": initial_metrics,
        "enriched_hidden_metrics": enriched_metrics,
        "mean_esr_ratio_enriched_over_initial": enriched_mean / initial_mean,
        "model_path": artifact["model_path"],
        "model_sha256": artifact["model_sha256"],
        "manifest_path": artifact["manifest_path"],
        "native_model_loader": artifact["validation_message"],
        "native_manifest_loader": (manifest_loader.stdout + manifest_loader.stderr).strip(),
        "player_renders": player_renders,
        "uniform_safety_trim_db": state["uniform_safety_trim_db"],
        "training_delay_samples": training_delay_samples,
        "source_hashes": {
            "local_active_learner.py": sha256_file(trainer_root / "local_active_learner.py"),
            "workflow.py": sha256_file(Path(__file__).resolve()),
            "input.wav": sha256_file(input_path),
            "base-manifest.json": sha256_file(Path(state["base_manifest_path"])),
        },
        "unproven": [
            "human tone and feel",
            "production-quality interpolation over the complete seven-dimensional domain",
            "real-time Player callback and signed app packaging",
            "physical-amplifier capture",
        ],
    }
    write_json(run_root / "05_VALIDATION" / "validation-report.json", report)
    state.update(
        {
            "status": report["status"],
            "selected_capture": selected_capture,
            "selected_point": selected_points[0],
            "reserved_evaluation_vectors": evaluation_vectors[1:],
            "reserved_evaluation_policy": candidate_config["reserved_policy"],
            "model_manifest_path": artifact["manifest_path"],
            "validation_report_path": str(run_root / "05_VALIDATION" / "validation-report.json"),
        }
    )
    write_json(state_path, state)
    print(json.dumps(report, sort_keys=True))


def reserve_evaluation(args: argparse.Namespace) -> None:
    """Freeze and render an external-DI audit split before cloud training."""

    run_root = args.run_root.expanduser().resolve()
    workflow = run_root / "00_WORKFLOW"
    state_path = workflow / "run-state.json"
    candidate_config_path = workflow / "candidate-config.json"
    report_path = run_root / "05_VALIDATION" / "hidden-evaluation-reservation.json"
    state = read_json(state_path)
    candidate_config = read_json(candidate_config_path)
    if state.get("status") != "prepared":
        raise RuntimeError(f"run state must be prepared, got {state.get('status')!r}")
    if candidate_config.get("reserved_evaluation_vectors") or report_path.exists():
        raise RuntimeError("external evaluation settings are already reserved")

    selected = select_coverage_holdouts(
        candidate_config["candidate_vectors"],
        candidate_config["captured_vectors"],
        maximum=args.hidden_count,
        minimum_distance=args.minimum_distance,
    )
    vectors = [list(item.g_vector) for item in selected]
    renderer = Path(state["renderer_path"])
    base_manifest = read_json(Path(state["base_manifest_path"]))
    strife_input = Path(state["strife_input_path"])
    references: list[dict[str, Any]] = []
    for index, (item, vector) in enumerate(zip(selected, vectors)):
        evidence = render_capture(
            run_root=run_root,
            renderer=renderer,
            input_path=strife_input,
            base_manifest=base_manifest,
            settings=settings_from_vector(vector),
            capture_id=f"evaluation-strife-{index:02d}",
            source_role="frozen-coverage-holdout-on-unseen-di",
            safety_trim_db=float(state["uniform_safety_trim_db"]),
            official_validation=False,
        )
        references.append(
            {
                "candidate_index": item.source_index,
                "coverage_distance_to_starter": item.disagreement_score,
                "g_vector": vector,
                "path": evidence["pcm24_path"],
                "sha256": evidence["wav_info"]["sha256"],
                "peak_dbfs": evidence["wav_info"]["peak_dbfs"],
            }
        )

    candidate_config["reserved_evaluation_vectors"] = vectors
    candidate_config["reserved_policy"] = (
        "Frozen before cloud training; never render these settings on the training input or "
        "enqueue them as captures during this validation run."
    )
    write_json(candidate_config_path, candidate_config)
    report = {
        "status": "reserved",
        "method": "model-independent maximum distance from starter coverage plus greedy separation",
        "minimum_distance_between_holdouts": args.minimum_distance,
        "input_path": str(strife_input),
        "input_sha256": sha256_file(strife_input),
        "target_audio_rendered_before_training": True,
        "target_audio_available_to_acquisition": False,
        "references": references,
        "candidate_config_path": str(candidate_config_path),
        "candidate_config_sha256": sha256_file(candidate_config_path),
    }
    write_json(report_path, report)
    state["status"] = "evaluation-reserved"
    state["reserved_evaluation_vectors"] = vectors
    state["reserved_evaluation_policy"] = candidate_config["reserved_policy"]
    state["hidden_evaluation_report"] = str(report_path)
    state["hidden_evaluation_report_sha256"] = sha256_file(report_path)
    write_json(state_path, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "count": len(references),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


def _backup_capture_database(database_path: Path, backup_path: Path) -> None:
    """Create a transactionally consistent SQLite backup, including WAL state."""

    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    if backup_path.exists():
        raise RuntimeError(f"Refusing to replace existing database backup: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"{database_path.as_uri()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True, timeout=10.0) as source:
        with sqlite3.connect(backup_path, timeout=10.0) as destination:
            source.backup(destination)
            integrity = destination.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise RuntimeError(f"SQLite backup integrity check failed: {integrity!r}")


def capture_batch(args: argparse.Namespace) -> None:
    """Capture one audit-safe, disagreement-ranked batch through the Trainer queue."""

    from dessmetal_capture.domain import ControlDefinition
    from dessmetal_capture.panama import parse_new_points, source_digest
    from dessmetal_capture.store import CaptureStore

    run_root = args.run_root.expanduser().resolve()
    workflow = run_root / "00_WORKFLOW"
    state_path = workflow / "run-state.json"
    candidate_config_path = workflow / "candidate-config.json"
    state = read_json(state_path)
    candidate_config = read_json(candidate_config_path)
    selection_report_path = args.selection_report.expanduser().resolve()
    selection_report = read_json(selection_report_path)
    protocol = selection_report.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError("Selection report has no audit protocol")
    required_protocol = {
        "captured_vectors_excluded_before_render": True,
        "reserved_vectors_excluded_before_render": True,
        "target_audio_read": False,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) is not expected:
            raise RuntimeError(f"Selection report violates {key}={expected!r}")
    expected_candidate_hash = selection_report.get("candidate_config_sha256")
    actual_candidate_hash = sha256_file(candidate_config_path)
    if expected_candidate_hash != actual_candidate_hash:
        raise RuntimeError(
            "Selection report is stale: candidate-config.json no longer matches its bound SHA-256"
        )
    ranking_payload = selection_report.get("ranking")
    if not isinstance(ranking_payload, list) or not ranking_payload:
        raise RuntimeError("Selection report contains no disagreement ranking")
    ranking = [
        CandidateScore(
            source_index=int(item["index"]),
            g_vector=tuple(float(value) for value in item["g_vector"]),
            disagreement_score=float(item["disagreement_score"]),
        )
        for item in ranking_payload
    ]
    selected = select_diverse_batch(
        ranking,
        maximum=args.maximum,
        minimum_distance=args.minimum_distance,
    )
    if not selected:
        raise RuntimeError("Diversity selection produced no capture points")
    selected_points = [item.as_dict() for item in selected]
    captured = {tuple(float(value) for value in vector) for vector in candidate_config["captured_vectors"]}
    reserved = {
        tuple(float(value) for value in vector)
        for vector in candidate_config.get("reserved_evaluation_vectors", [])
    }
    for point in selected_points:
        vector = tuple(point["g_vector"])
        if vector in captured or vector in reserved:
            raise RuntimeError(f"Selected point is captured or reserved: {vector}")

    round_label = f"{args.round_index:02d}"
    new_points_path = workflow / f"new_points-round-{round_label}.json"
    if new_points_path.exists():
        raise RuntimeError(f"Refusing to replace existing selection: {new_points_path}")
    target_ids = [f"active-round-{round_label}-{index:02d}" for index in range(len(selected_points))]
    for capture_id in target_ids:
        targets = (
            run_root / "01_MANIFESTS" / f"{capture_id}.json",
            run_root / "02_CAPTURES" / "unity-float" / f"{capture_id}__unity-float.wav",
            run_root / "02_CAPTURES" / "training-pcm24" / f"{capture_id}__train-pcm24.wav",
            run_root / "05_VALIDATION" / "render-reports" / f"{capture_id}.json",
        )
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise RuntimeError(f"Refusing to replace capture evidence: {existing}")
    existing_rounds = state.get("cloud_active_rounds", [])
    if any(int(entry.get("round_index", -1)) == args.round_index for entry in existing_rounds):
        raise RuntimeError(f"Cloud active round {args.round_index} is already recorded")

    store = CaptureStore(run_root / "04_TRAINER_DATA")
    pending = [
        item for item in store.list_recommendations(state["project_id"])
        if item["status"] == "pending"
    ]
    if pending:
        raise RuntimeError(f"Project already has a pending recommendation: {pending[0]['id']}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = (
        run_root
        / "04_TRAINER_DATA"
        / "backups"
        / f"capture-before-active-round-{round_label}-{timestamp}.sqlite3"
    )
    _backup_capture_database(store.database_path, backup_path)
    write_new_points(new_points_path, selected, maximum=len(selected))

    project = store.get_project(state["project_id"])
    controls = [ControlDefinition.from_mapping(control) for control in project["controls"]]
    suggestions = parse_new_points(
        selected_points,
        controls=controls,
        control_order=CONTROL_ORDER,
    )
    queued = store.enqueue_panama_suggestions(
        project["id"],
        round_index=args.round_index,
        source_sha256=source_digest(selected_points),
        control_order=CONTROL_ORDER,
        suggestions=suggestions,
    )
    if len(queued) != len(selected_points):
        raise RuntimeError("Trainer queue did not persist the complete selected batch")

    renderer = Path(state["renderer_path"])
    input_path = Path(state["input_path"])
    base_manifest = read_json(Path(state["base_manifest_path"]))
    capture_records: list[dict[str, Any]] = []
    selection_report_sha256 = sha256_file(selection_report_path)
    batch_record = {
        "round_index": args.round_index,
        "selection_report": str(selection_report_path),
        "selection_report_sha256": selection_report_sha256,
        "new_points_path": str(new_points_path),
        "new_points_sha256": sha256_file(new_points_path),
        "minimum_distance": args.minimum_distance,
        "requested_maximum": args.maximum,
        "planned_count": len(selected_points),
        "selected_count": 0,
        "database_backup": str(backup_path),
        "database_backup_sha256": sha256_file(backup_path),
        "reserved_filter_applied": True,
        "target_audio_read": False,
        "captures": capture_records,
    }
    state.setdefault("cloud_active_rounds", []).append(batch_record)
    state["status"] = f"cloud-active-round-{round_label}-batch-in-progress"
    write_json(state_path, state)
    for position, (capture_id, point) in enumerate(zip(target_ids, selected_points)):
        recommendation = store.create_recommendation(project["id"])
        if recommendation["strategy"] != "panama-ensemble-v1":
            raise RuntimeError("Trainer did not activate the PANAMA disagreement recommendation")
        recommended_vector = vector_from_settings(recommendation["settings"])
        if recommended_vector != point["g_vector"]:
            raise RuntimeError(
                f"Trainer queue order changed at batch position {position}: "
                f"expected {point['g_vector']}, got {recommended_vector}"
            )
        evidence = render_capture(
            run_root=run_root,
            renderer=renderer,
            input_path=input_path,
            base_manifest=base_manifest,
            settings=recommendation["settings"],
            capture_id=capture_id,
            source_role="panama-ensemble-disagreement-batch",
            safety_trim_db=float(state["uniform_safety_trim_db"]),
            official_validation=True,
        )
        stored = store.add_capture(
            project_id=project["id"],
            recommendation_id=recommendation["id"],
            source="offline-fortin-au",
            wav_path=Path(evidence["pcm24_path"]),
            wav_info=evidence["wav_info"],
            quality_status=evidence["capture_validation"]["status"],
            validation={
                "capture": evidence["capture_validation"],
                "official_nam": evidence["official_nam_validation"],
                "selection": point,
                "selection_report_sha256": selection_report_sha256,
            },
            notes="Audit-safe disagreement batch rendered through the persistent Trainer queue.",
        )
        capture_records.append(
            {
                "batch_position": position,
                "capture_id": capture_id,
                "stored_capture_id": stored["id"],
                "recommendation_id": recommendation["id"],
                "selection": point,
                "capture_path": evidence["pcm24_path"],
                "capture_sha256": sha256_file(Path(evidence["pcm24_path"])),
                "official_nam_passed_critical": evidence["official_nam_validation"]["passed_critical"],
                "capture_validation": evidence["capture_validation"],
            }
        )
        vector = list(point["g_vector"])
        if vector not in candidate_config["captured_vectors"]:
            candidate_config["captured_vectors"].append(vector)
        batch_record["selected_count"] = len(capture_records)
        write_json(candidate_config_path, candidate_config)
        write_json(state_path, state)

    state["status"] = f"cloud-active-round-{round_label}-batch-captured"
    write_json(state_path, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "selected_count": len(selected_points),
                "capture_ids": target_ids,
                "database_backup": str(backup_path),
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        required=DEFAULT_RUN_ROOT is None,
        help="Persistent run root (or set DESSAMP_FORTIN_RUN_ROOT)",
    )

    prepare_parser = subparsers.add_parser("prepare", parents=[common])
    prepare_parser.add_argument(
        "--base-run",
        type=Path,
        default=DEFAULT_BASE_RUN,
        required=DEFAULT_BASE_RUN is None,
        help="Existing evidence run (or set DESSAMP_FORTIN_BASE_RUN)",
    )
    prepare_parser.add_argument(
        "--renderer-binary",
        type=Path,
        help="Corrected Fortin AU renderer to persist in this run instead of the base-run binary.",
    )
    prepare_parser.add_argument(
        "--renderer-source",
        type=Path,
        help="Source matching --renderer-binary; copied into the run for provenance.",
    )
    prepare_parser.add_argument(
        "--strife-input",
        type=Path,
        default=DEFAULT_STRIFE_INPUT,
        required=DEFAULT_STRIFE_INPUT is None,
        help="Reference DI input (or set DESSAMP_FORTIN_STRIFE_INPUT)",
    )
    prepare_parser.add_argument(
        "--logic-project",
        type=Path,
        default=DEFAULT_LOGIC_PROJECT,
        required=DEFAULT_LOGIC_PROJECT is None,
        help="Logic capture project (or set DESSAMP_FORTIN_LOGIC_PROJECT)",
    )
    prepare_parser.add_argument("--candidate-count", type=int, default=96)
    prepare_parser.add_argument("--starter-count", type=int, default=10)
    prepare_parser.add_argument(
        "--delay-samples",
        type=int,
        default=0,
        help=(
            "Measured capture-path delay using NAM Dataset convention. "
            "The Fortin plug-in corpus uses zero; measure a physical send/return path explicitly."
        ),
    )
    prepare_parser.set_defaults(function=prepare)

    reserve_parser = subparsers.add_parser("reserve-evaluation", parents=[common])
    reserve_parser.add_argument("--hidden-count", type=int, default=8)
    reserve_parser.add_argument("--minimum-distance", type=float, default=0.35)
    reserve_parser.set_defaults(function=reserve_evaluation)

    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--backend", choices=("mps", "cpu"), default="mps")
    train_parser.add_argument("--ensemble-size", type=int, default=3)
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--ny", type=int, default=2048)
    train_parser.add_argument("--score-chunk-size", type=int, default=8192)
    train_parser.add_argument("--score-batch-size", type=int, default=8)
    train_parser.add_argument("--max-score-chunks", type=int, default=24)
    train_parser.add_argument("--hidden-count", type=int, default=8)
    train_parser.add_argument("--training-start-seconds", type=float, default=45.0)
    train_parser.add_argument("--training-duration-seconds", type=float, default=15.0)
    train_parser.add_argument("--seed", type=int, default=260818)
    train_parser.add_argument("--loader", type=Path, required=True)
    train_parser.add_argument("--manifest-loader", type=Path, required=True)
    train_parser.add_argument("--renderer", type=Path, required=True)
    train_parser.set_defaults(function=train)

    capture_parser = subparsers.add_parser("capture-batch", parents=[common])
    capture_parser.add_argument("--selection-report", type=Path, required=True)
    capture_parser.add_argument("--round-index", type=int, required=True)
    capture_parser.add_argument("--maximum", type=int, default=8)
    capture_parser.add_argument("--minimum-distance", type=float, default=0.35)
    capture_parser.set_defaults(function=capture_batch)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
