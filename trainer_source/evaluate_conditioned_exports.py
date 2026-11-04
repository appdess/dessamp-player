"""Evaluate native conditioned NAM exports on a fixed DessAmp audit split.

This is deliberately separate from the trainer. It renders every exported
model through the current native NAM Core, computes individual and arithmetic-
ensemble ESR in float64, and refuses to replace an existing report or render
directory. The capture project's ``validation-report.json`` remains the source
of truth for the permanently frozen control vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf


DEFAULT_MEAN_ESR_GATE = 0.035
DEFAULT_MAX_ESR_GATE = 0.035
DEFAULT_TARGET_MEAN_ESR_GATE = 0.01
DEFAULT_TARGET_MAX_ESR_GATE = 0.02
DEFAULT_PREFERRED_MEAN_ESR_GATE = 0.005
LEGACY_WAVENET_RECEPTIVE_FIELD = 4093
DEFAULT_CHUNK_SIZE = 8192
DEFAULT_BLOCK_SIZE = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("metric values must be a non-empty finite sequence")
    array = np.asarray(values, dtype=np.float64)
    return {
        "min_esr": float(np.min(array)),
        "mean_esr": float(np.mean(array)),
        "median_esr": float(np.median(array)),
        "p95_esr": float(np.percentile(array, 95)),
        "max_esr": float(np.max(array)),
    }


def quality_assessment(
    summary: Mapping[str, Any],
    *,
    mean_gate: float,
    max_gate: float,
) -> dict[str, Any]:
    """Evaluate one metric summary against a mean-and-worst-setting contract."""

    mean_esr = float(summary["mean_esr"])
    max_esr = float(summary["max_esr"])
    values = (mean_esr, max_esr, mean_gate, max_gate)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("quality metrics and gates must be finite and non-negative")
    return {
        "passed": mean_esr <= mean_gate and max_esr <= max_gate,
        "required_mean_esr_lte": mean_gate,
        "required_max_esr_lte": max_gate,
    }


def aligned_esr(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("prediction and target must be equal-length mono arrays")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("prediction and target must contain only finite samples")
    target_energy = float(np.sum(target * target, dtype=np.float64))
    if target_energy <= 0.0:
        raise ValueError("target energy must be positive")
    error = prediction - target
    return float(np.sum(error * error, dtype=np.float64) / target_energy)


def normalized_member_variance(predictions: np.ndarray, target: np.ndarray) -> float:
    if predictions.ndim != 2 or predictions.shape[0] < 2:
        raise ValueError("at least two aligned member predictions are required")
    if predictions.shape[1:] != target.shape:
        raise ValueError("member predictions and target must be aligned")
    target_energy = float(np.sum(target * target, dtype=np.float64))
    if target_energy <= 0.0:
        raise ValueError("target energy must be positive")
    sample_variance = np.var(predictions, axis=0, ddof=1, dtype=np.float64)
    return float(np.sum(sample_variance, dtype=np.float64) / target_energy)


def evaluated_frame_count(
    input_frames: int,
    *,
    receptive_field: int = LEGACY_WAVENET_RECEPTIVE_FIELD,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    if receptive_field < 1 or chunk_size <= receptive_field:
        raise ValueError("chunk size must exceed a positive receptive field")
    if input_frames < chunk_size:
        raise ValueError("evaluation input is shorter than one chunk")
    step = chunk_size - receptive_field + 1
    return ((input_frames - chunk_size) // step + 1) * step


def read_mono(path: Path, *, expected_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    if audio.ndim != 1:
        raise ValueError(f"expected mono WAV: {path}")
    if expected_rate is not None and sample_rate != expected_rate:
        raise ValueError(
            f"sample-rate mismatch for {path}: expected {expected_rate}, got {sample_rate}"
        )
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"non-finite audio samples in {path}")
    return audio, int(sample_rate)


def parse_members(
    raw_members: Sequence[str], *, minimum_members: int = 2
) -> dict[str, Path]:
    if minimum_members < 1:
        raise ValueError("minimum member count must be positive")
    members: dict[str, Path] = {}
    for raw in raw_members:
        label, separator, raw_path = raw.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("each --member must use LABEL=/absolute/model.nam")
        if label in members:
            raise ValueError(f"duplicate member label: {label}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"member model is not a file: {path}")
        members[label] = path
    if len(members) < minimum_members:
        noun = "export" if minimum_members == 1 else "exports"
        raise ValueError(
            f"at least {minimum_members} --member {noun} are required"
        )
    return members


def _load_stats(model_path: Path) -> Mapping[str, Any] | None:
    stats_path = model_path.with_name("model_stats.json")
    if not stats_path.is_file():
        return None
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"model stats must be a JSON object: {stats_path}")
    return payload


def model_receptive_field(model_path: Path) -> int | None:
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    value = metadata.get("receptive_field") if isinstance(metadata, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def resolve_receptive_field(
    members: Mapping[str, Path], explicit: int | None
) -> int:
    if explicit is not None:
        if explicit < 1:
            raise ValueError("receptive field must be positive")
        return explicit
    inferred = {model_receptive_field(path) for path in members.values()}
    if None in inferred or len(inferred) != 1:
        raise ValueError(
            "unable to infer one receptive field from all exports; pass --receptive-field "
            "(conditioned A2 Full/Lite uses 6347)"
        )
    return int(next(iter(inferred)))


def load_evaluation_reservation(
    reservation_path: Path, *, containment_root: Path | None = None
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    source = json.loads(reservation_path.read_text(encoding="utf-8"))
    references = source.get("references")
    if (
        source.get("target_audio_available_to_acquisition") is not False
        or not isinstance(references, list)
        or not references
    ):
        raise ValueError("hidden evaluation reservation violates the acquisition boundary")
    settings: list[dict[str, Any]] = []
    normalized_references: list[dict[str, Any]] = []
    evaluation_scope = source.get("evaluation_scope")
    evaluation_role = (
        evaluation_scope.get("role")
        if isinstance(evaluation_scope, Mapping)
        and isinstance(evaluation_scope.get("role"), str)
        and evaluation_scope.get("role")
        else "frozen-coverage-holdout-on-unseen-di"
    )
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise ValueError(f"hidden reference {index} must be an object")
        vector = reference.get("g_vector")
        raw_path = reference.get("path")
        expected_sha256 = reference.get("sha256")
        if not isinstance(vector, list) or not vector or not isinstance(raw_path, str):
            raise ValueError(f"hidden reference {index} is incomplete")
        reference_path = Path(raw_path).expanduser().resolve()
        if not reference_path.is_file():
            raise ValueError(f"hidden reference is missing: {reference_path}")
        if containment_root is not None and not reference_path.is_relative_to(containment_root):
            raise ValueError(f"hidden reference escapes the run root: {reference_path}")
        actual_sha256 = sha256_file(reference_path)
        if expected_sha256 != actual_sha256:
            raise ValueError(f"hidden reference hash mismatch: {reference_path}")
        settings.append(
            {
                "role": evaluation_role,
                "setting_id": reference.get("setting_id"),
                "g_vector": list(vector),
            }
        )
        normalized_references.append(
            {
                "path": str(reference_path),
                "sha256": actual_sha256,
                "setting_id": reference.get("setting_id"),
            }
        )
    return reservation_path, settings, normalized_references


def load_fixed_evaluation(run_root: Path) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    """Load either the pre-training reservation or the legacy validation report."""

    reservation_path = run_root / "05_VALIDATION" / "hidden-evaluation-reservation.json"
    if reservation_path.is_file():
        return load_evaluation_reservation(reservation_path, containment_root=run_root)

    source_report_path = run_root / "05_VALIDATION" / "validation-report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    settings = source_report.get("evaluation_settings")
    if not isinstance(settings, list) or not settings:
        raise ValueError("validation-report.json has no fixed evaluation_settings")
    references = [
        {
            "path": str(
                run_root
                / "02_CAPTURES"
                / "training-pcm24"
                / f"evaluation-strife-{index:02d}__train-pcm24.wav"
            )
        }
        for index in range(len(settings))
    ]
    return source_report_path, settings, references


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.block_size < 1 or args.render_timeout_seconds < 1:
        raise ValueError("block size and render timeout must be positive")
    gates = (
        args.mean_esr_gate,
        args.max_esr_gate,
        args.target_mean_esr_gate,
        args.target_max_esr_gate,
        args.preferred_mean_esr_gate,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in gates):
        raise ValueError("ESR gates must be finite and non-negative")
    if args.run_root is not None:
        run_root = args.run_root.expanduser().resolve()
        source_report_path, settings, fixed_references = load_fixed_evaluation(run_root)
        input_path = (
            args.input.expanduser().resolve()
            if args.input is not None
            else run_root / "00_WORKFLOW" / "strife-di-pcm24.wav"
        )
    else:
        reservation_path = args.evaluation_reservation.expanduser().resolve()
        source_report_path, settings, fixed_references = load_evaluation_reservation(
            reservation_path
        )
        reservation_payload = json.loads(reservation_path.read_text(encoding="utf-8"))
        input_record = reservation_payload.get("evaluation_input")
        if args.input is not None:
            input_path = args.input.expanduser().resolve()
        elif isinstance(input_record, Mapping) and isinstance(input_record.get("path"), str):
            input_path = Path(input_record["path"]).expanduser().resolve()
        else:
            raise ValueError(
                "generic reservation has no evaluation_input; pass --input explicitly"
            )
        if isinstance(input_record, Mapping) and input_record.get("sha256") != sha256_file(input_path):
            raise ValueError("evaluation input does not match the reservation SHA-256")

    renderer = args.renderer.expanduser().resolve()
    if not renderer.is_file():
        raise ValueError(f"native renderer is not a file: {renderer}")
    # A release gate may evaluate one independently deployable child. Active-
    # learning disagreement still calls parse_members() with its default of two.
    members = parse_members(args.member, minimum_members=1)
    receptive_field = resolve_receptive_field(members, args.receptive_field)
    input_audio, sample_rate = read_mono(input_path)
    covered = evaluated_frame_count(
        len(input_audio),
        receptive_field=receptive_field,
        chunk_size=args.chunk_size,
    )
    trim_start = receptive_field - 1

    if args.render_dir is None:
        render_dir = Path(tempfile.mkdtemp(prefix="dessmetal-conditioned-audit-"))
    else:
        render_dir = args.render_dir.expanduser().resolve()
        render_dir.mkdir(parents=True, exist_ok=False)

    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise ValueError(f"refusing to replace existing report: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_records: dict[str, Any] = {}
    individual: dict[str, Any] = {
        label: {"settings": []} for label in members
    }
    ensemble_settings: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []

    for label, model_path in members.items():
        model_records[label] = {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "trainer_stats": _load_stats(model_path),
        }

    for setting_index, entry in enumerate(settings):
        vector = entry.get("g_vector")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"evaluation setting {setting_index} has no g_vector")
        values = [float(value) for value in vector]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"evaluation setting {setting_index} is non-finite")
        reference_path = Path(fixed_references[setting_index]["path"])
        reference_audio, _ = read_mono(reference_path, expected_rate=sample_rate)
        target = reference_audio[trim_start : trim_start + covered]
        if len(target) != covered:
            raise ValueError(f"reference is too short after alignment: {reference_path}")
        references.append({"path": str(reference_path), "sha256": sha256_file(reference_path)})

        member_predictions: list[np.ndarray] = []
        for label, model_path in members.items():
            render_path = render_dir / f"member-{label}-setting-{setting_index:02d}.wav"
            command = [str(renderer), str(model_path), str(input_path), str(render_path)]
            for value in values:
                command.extend(["--param", str(value)])
            command.extend(["--block-size", str(args.block_size)])
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.render_timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "native render failed")[-8_000:]
                raise RuntimeError(f"member {label}, setting {setting_index}: {detail}")
            prediction_audio, _ = read_mono(render_path, expected_rate=sample_rate)
            prediction = prediction_audio[trim_start : trim_start + covered]
            if len(prediction) != covered:
                raise ValueError(f"native render is too short after alignment: {render_path}")
            member_predictions.append(prediction)
            individual[label]["settings"].append(
                {"setting_index": setting_index, "esr": aligned_esr(prediction, target)}
            )

        prediction_matrix = np.stack(member_predictions, axis=0)
        ensemble_prediction = np.mean(prediction_matrix, axis=0, dtype=np.float64)
        ensemble_settings.append(
            {
                "setting_index": setting_index,
                "setting_id": entry.get("setting_id"),
                "role": entry.get("role"),
                "g_vector": values,
                "esr": aligned_esr(ensemble_prediction, target),
                "normalized_member_variance": (
                    normalized_member_variance(prediction_matrix, target)
                    if prediction_matrix.shape[0] > 1
                    else None
                ),
            }
        )

    for record in individual.values():
        record["summary"] = metric_summary([item["esr"] for item in record["settings"]])
    ensemble_summary = metric_summary([item["esr"] for item in ensemble_settings])
    acceptance = quality_assessment(
        ensemble_summary,
        mean_gate=args.mean_esr_gate,
        max_gate=args.max_esr_gate,
    )
    acceptance.update(
        {
            "tier": "nam-normal",
            "description": "Every held-out setting remains within NAM's normal Not bad ESR band.",
        }
    )
    quality_target = quality_assessment(
        ensemble_summary,
        mean_gate=args.target_mean_esr_gate,
        max_gate=args.target_max_esr_gate,
    )
    quality_target.update(
        {
            "tier": "project-target",
            "description": "Stricter conditioned-model target retained for further acquisition and training.",
        }
    )
    preferred = quality_assessment(
        ensemble_summary,
        mean_gate=args.preferred_mean_esr_gate,
        max_gate=args.target_max_esr_gate,
    )
    preferred.update(
        {
            "tier": "preferred",
            "description": "Preferred mean ESR without relaxing the project-target worst-setting bound.",
        }
    )
    passed = bool(acceptance["passed"])
    report = {
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "acceptance": acceptance,
        "quality_target": quality_target,
        "preferred": preferred,
        "method": {
            "renderer": str(renderer),
            "block_size": args.block_size,
            "delay_samples": 0,
            "receptive_field": receptive_field,
            "trim_start_frames": trim_start,
            "chunk_size_for_coverage": args.chunk_size,
            "evaluated_frames_per_setting": covered,
            "metric": "sum((prediction-target)^2)/sum(target^2), float64; PCM24 native render",
            "ensemble": (
                "identity for one independently gated model"
                if len(members) == 1
                else "arithmetic mean of native rendered outputs"
            ),
        },
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "frames": len(input_audio),
            "sample_rate": sample_rate,
        },
        "fixed_evaluation_source": {
            "path": str(source_report_path),
            "sha256": sha256_file(source_report_path),
            "evaluation_scope": (
                json.loads(source_report_path.read_text(encoding="utf-8")).get(
                    "evaluation_scope"
                )
                if args.evaluation_reservation is not None
                else None
            ),
        },
        "references": references,
        "models": model_records,
        "individual": individual,
        "ensemble": {"settings": ensemble_settings, "summary": ensemble_summary},
        "render_directory": str(render_dir),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path, help="Legacy regression-harness root")
    source.add_argument(
        "--evaluation-reservation",
        type=Path,
        help="Hash-bound generic hidden-evaluation reservation JSON",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Override unseen evaluation DI; bytes must match the generic reservation when declared",
    )
    parser.add_argument("--renderer", required=True, type=Path)
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument(
        "--receptive-field",
        type=int,
        help="Required when exports do not declare metadata.receptive_field",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--render-timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--mean-esr-gate",
        type=float,
        default=DEFAULT_MEAN_ESR_GATE,
        help="NAM-normal acceptance mean (default: 0.035)",
    )
    parser.add_argument(
        "--max-esr-gate",
        type=float,
        default=DEFAULT_MAX_ESR_GATE,
        help="NAM-normal acceptance worst setting (default: 0.035)",
    )
    parser.add_argument(
        "--target-mean-esr-gate",
        type=float,
        default=DEFAULT_TARGET_MEAN_ESR_GATE,
        help="stricter project target mean (default: 0.01)",
    )
    parser.add_argument(
        "--target-max-esr-gate",
        type=float,
        default=DEFAULT_TARGET_MAX_ESR_GATE,
        help="stricter project target worst setting (default: 0.02)",
    )
    parser.add_argument(
        "--preferred-mean-esr-gate",
        type=float,
        default=DEFAULT_PREFERRED_MEAN_ESR_GATE,
        help="preferred mean while retaining the target worst-setting bound (default: 0.005)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate(args)
    print(json.dumps({"status": report["status"], "summary": report["ensemble"]["summary"]}))


if __name__ == "__main__":
    main()
