#!/usr/bin/env python3
"""Prepare provenance-bound teacher/student listening copies without boosting."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from evaluate_conditioned_exports import read_mono, sha256_file


def _audio_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _checked_path(raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} path must be a non-empty string")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _active_rms_pair(
    left: np.ndarray,
    right: np.ndarray,
    *,
    trim_start: int,
    active_threshold: float,
) -> tuple[float, float, int]:
    if left.shape != right.shape:
        raise ValueError("teacher and student WAV lengths differ")
    if trim_start < 0 or trim_start >= len(left):
        raise ValueError("receptive-field trim leaves no listening audio")
    left_tail = left[trim_start:]
    right_tail = right[trim_start:]
    active = np.maximum(np.abs(left_tail), np.abs(right_tail)) > active_threshold
    active_frames = int(np.count_nonzero(active))
    if active_frames == 0:
        raise ValueError("teacher/student pair has no active signal")
    left_rms = float(np.sqrt(np.mean(np.square(left_tail[active], dtype=np.float64))))
    right_rms = float(np.sqrt(np.mean(np.square(right_tail[active], dtype=np.float64))))
    if not math.isfinite(left_rms) or not math.isfinite(right_rms):
        raise ValueError("teacher/student pair has non-finite RMS")
    if left_rms <= 0.0 or right_rms <= 0.0:
        raise ValueError("teacher/student pair has zero RMS")
    return left_rms, right_rms, active_frames


def _attenuation_only_match(
    teacher: np.ndarray,
    student: np.ndarray,
    *,
    trim_start: int,
    active_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    teacher_rms, student_rms, active_frames = _active_rms_pair(
        teacher,
        student,
        trim_start=trim_start,
        active_threshold=active_threshold,
    )
    target_rms = min(teacher_rms, student_rms)
    teacher_gain = min(1.0, target_rms / teacher_rms)
    student_gain = min(1.0, target_rms / student_rms)
    teacher_copy = np.asarray(teacher, dtype=np.float64) * teacher_gain
    student_copy = np.asarray(student, dtype=np.float64) * student_gain
    peak_before_safety = max(
        float(np.max(np.abs(teacher_copy))), float(np.max(np.abs(student_copy)))
    )
    if not math.isfinite(peak_before_safety):
        raise ValueError("listening copy has a non-finite peak")
    # Cabinet convolution can make both RMS-matched copies exceed full scale.
    # Apply one identical attenuation to the pair so their relative level is
    # unchanged and the listening helper never clips or boosts either side.
    safety_peak = 0.98
    common_safety_gain = min(1.0, safety_peak / peak_before_safety)
    teacher_copy *= common_safety_gain
    student_copy *= common_safety_gain
    peak = max(float(np.max(np.abs(teacher_copy))), float(np.max(np.abs(student_copy))))
    if not math.isfinite(peak) or peak >= 1.0:
        raise ValueError("listening copy would clip PCM24 output")
    matched_teacher_rms, matched_student_rms, _ = _active_rms_pair(
        teacher_copy,
        student_copy,
        trim_start=trim_start,
        active_threshold=active_threshold,
    )
    return teacher_copy, student_copy, {
        "active_threshold": active_threshold,
        "active_frames": active_frames,
        "teacher_original_rms": teacher_rms,
        "student_original_rms": student_rms,
        "target_rms": target_rms,
        "teacher_gain_linear": teacher_gain,
        "student_gain_linear": student_gain,
        "teacher_gain_db": 20.0 * math.log10(teacher_gain),
        "student_gain_db": 20.0 * math.log10(student_gain),
        "peak_before_common_safety_attenuation": peak_before_safety,
        "common_safety_gain_linear": common_safety_gain,
        "common_safety_gain_db": 20.0 * math.log10(common_safety_gain),
        "teacher_matched_rms": matched_teacher_rms,
        "student_matched_rms": matched_student_rms,
        "peak_after_matching": peak,
        "boost_applied": teacher_gain > 1.0 or student_gain > 1.0,
    }


def _write_pcm24(path: Path, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    sf.write(path, audio, sample_rate, subtype="PCM_24")
    info = sf.info(path)
    return {
        **_audio_record(path),
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "subtype": info.subtype,
    }


def prepare(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    spec_path = spec_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to replace existing listening directory: {output_dir}")
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("listening spec must be a JSON object")
    settings = payload.get("settings")
    if not isinstance(settings, list) or not settings:
        raise ValueError("listening spec requires at least one setting")
    receptive_field = int(payload.get("receptive_field", 6347))
    if receptive_field < 1:
        raise ValueError("receptive_field must be positive")
    trim_start = receptive_field - 1
    active_threshold = float(payload.get("active_threshold", 1.0e-5))
    if not math.isfinite(active_threshold) or active_threshold <= 0.0:
        raise ValueError("active_threshold must be finite and positive")
    input_path = _checked_path(payload.get("input"), label="input")
    ir_path = (
        _checked_path(payload.get("cabinet_ir"), label="cabinet_ir")
        if payload.get("cabinet_ir") is not None
        else None
    )

    resolved_settings: list[tuple[str, list[float], Path, Path]] = []
    seen_labels: set[str] = set()
    for index, raw in enumerate(settings):
        if not isinstance(raw, Mapping):
            raise ValueError(f"setting {index} must be an object")
        label = str(raw.get("label", f"setting-{index:02d}")).strip()
        if not label or label in seen_labels or Path(label).name != label:
            raise ValueError(f"setting {index} has an invalid or duplicate label")
        seen_labels.add(label)
        g_vector = [float(value) for value in raw.get("g_vector", [])]
        if not g_vector or not all(math.isfinite(value) for value in g_vector):
            raise ValueError(f"setting {index} g_vector must be finite and non-empty")
        resolved_settings.append(
            (
                label,
                g_vector,
                _checked_path(raw.get("teacher"), label=f"setting {index} teacher"),
                _checked_path(raw.get("student"), label=f"setting {index} student"),
            )
        )

    output_dir.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    ir_audio: np.ndarray | None = None
    ir_rate: int | None = None
    if ir_path is not None:
        ir_audio, ir_rate = read_mono(ir_path)
    for label, g_vector, teacher_path, student_path in resolved_settings:
        teacher, sample_rate = read_mono(teacher_path)
        student, _ = read_mono(student_path, expected_rate=sample_rate)
        if len(teacher) != len(student):
            raise ValueError(f"{label}: teacher and student WAV lengths differ")
        setting_dir = output_dir / label
        setting_dir.mkdir()
        teacher_head, student_head, head_match = _attenuation_only_match(
            teacher,
            student,
            trim_start=trim_start,
            active_threshold=active_threshold,
        )
        result: dict[str, Any] = {
            "label": label,
            "g_vector": g_vector,
            "raw_teacher": _audio_record(teacher_path),
            "raw_student": _audio_record(student_path),
            "head_only": {
                "matching": head_match,
                "teacher": _write_pcm24(setting_dir / "teacher-head.wav", teacher_head, sample_rate),
                "student": _write_pcm24(setting_dir / "student-head.wav", student_head, sample_rate),
            },
        }
        if ir_audio is not None:
            if ir_rate != sample_rate:
                raise ValueError(f"{label}: cabinet IR sample rate differs from render rate")
            teacher_cab = fftconvolve(teacher, ir_audio, mode="full")
            student_cab = fftconvolve(student, ir_audio, mode="full")
            teacher_cab, student_cab, cab_match = _attenuation_only_match(
                teacher_cab,
                student_cab,
                trim_start=trim_start,
                active_threshold=active_threshold,
            )
            result["cabinet"] = {
                "matching": cab_match,
                "teacher": _write_pcm24(
                    setting_dir / "teacher-cabinet.wav", teacher_cab, sample_rate
                ),
                "student": _write_pcm24(
                    setting_dir / "student-cabinet.wav", student_cab, sample_rate
                ),
            }
        results.append(result)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec": _audio_record(spec_path),
        "input": _audio_record(input_path),
        "cabinet_ir": _audio_record(ir_path) if ir_path is not None else None,
        "output_dir": str(output_dir),
        "policy": {
            "numerical_evaluation_audio_modified": False,
            "matching": "attenuate only the louder teacher/student copy to the quieter active-signal RMS, then apply identical safety attenuation to both copies only when required to prevent clipping",
            "boost_allowed": False,
            "eq_or_time_alignment_applied": False,
            "receptive_field": receptive_field,
            "trim_start_frames": trim_start,
            "active_threshold": active_threshold,
            "cabinet_convolution": "identical raw mono IR, scipy.signal.fftconvolve full mode; any common post-convolution safety attenuation is recorded per pair",
        },
        "settings": results,
        "human_listening_passed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_path = args.report.expanduser().resolve()
    if report_path.exists():
        raise RuntimeError(f"Refusing to replace existing listening report: {report_path}")
    report = prepare(args.spec, args.output_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "settings": len(report["settings"]),
                "human_listening_passed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
