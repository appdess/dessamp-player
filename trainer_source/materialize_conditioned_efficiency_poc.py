#!/usr/bin/env python3
"""Freeze a disjoint existing-corpus POC for conditioned-A2 data efficiency."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import wave
from pathlib import Path
from typing import Any, Sequence


RECEPTIVE_FIELD = 6347
DEFAULT_NY = 8192
DEFAULT_SAMPLE_RATE = 48_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"Source config must not be a symlink: {expanded}")
    path = expanded.resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= 2 * 1024 * 1024:
        raise ValueError(f"Source config is missing, empty, or oversized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Source config root must be an object")
    return value


def _atomic_json(path: Path, value: Any, *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def uniform_stratified_indices(available: int, selected: int) -> tuple[int, ...]:
    if (
        isinstance(available, bool)
        or isinstance(selected, bool)
        or not isinstance(available, int)
        or not isinstance(selected, int)
        or not 1 <= selected <= available
    ):
        raise ValueError("Selected chunk count must be between one and available")
    indices = tuple(
        ((2 * index + 1) * available) // (2 * selected)
        for index in range(selected)
    )
    if len(set(indices)) != selected:
        raise RuntimeError("Uniform stratification produced duplicate chunk indices")
    return indices


def _indices_sha256(indices: Sequence[int]) -> str:
    encoded = json.dumps(tuple(indices), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _audio_geometry(config: dict[str, Any], *, ny: int) -> tuple[int, int, int]:
    common = config.get("common")
    if not isinstance(common, dict) or not isinstance(common.get("x_path"), str):
        raise ValueError("Source config requires common.x_path")
    raw_input_path = Path(common["x_path"]).expanduser()
    if raw_input_path.is_symlink():
        raise ValueError(f"Input WAV must not be a symlink: {raw_input_path}")
    input_path = raw_input_path.resolve()
    if not input_path.is_file():
        raise ValueError(f"Input WAV is unavailable: {input_path}")
    with wave.open(str(input_path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        frames = source.getnframes()
    if channels != 1 or sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError("Efficiency POC requires mono 48 kHz aligned input audio")
    available = (frames - RECEPTIVE_FIELD + 1) // ny
    if available < 1:
        raise ValueError("Aligned input is too short for conditioned-A2 training")
    return frames, sample_rate, available


def materialize_efficiency_poc(
    *,
    source_config_path: Path,
    holdout_basenames: Sequence[str],
    run_id: str,
    chunks_per_output: int,
    ny: int = DEFAULT_NY,
    epochs: int = 60,
    baseline_seed: int = 110,
    replicate_seed: int = 111,
    hard_cost_ceiling_usd: float = 14.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = source_config_path.expanduser().resolve()
    config = copy.deepcopy(_read_json(source_path))
    if not holdout_basenames or len(set(holdout_basenames)) != len(holdout_basenames):
        raise ValueError("Holdout basenames must be non-empty and unique")
    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_id
    ):
        raise ValueError("Run ID must use lowercase letters, numbers, and hyphens")
    if not 256 <= ny <= 65_536:
        raise ValueError("Training chunk length is outside the supported range")
    if not 1 <= epochs <= 500 or baseline_seed == replicate_seed:
        raise ValueError("Epochs or matched-seed plan is invalid")
    if not math.isfinite(hard_cost_ceiling_usd) or hard_cost_ceiling_usd <= 0:
        raise ValueError("Hard cost ceiling must be positive")

    train = config.get("train")
    validation = config.get("validation")
    train_outputs = train.get("outputs") if isinstance(train, dict) else None
    validation_outputs = (
        validation.get("outputs") if isinstance(validation, dict) else None
    )
    if not isinstance(train_outputs, list) or not train_outputs:
        raise ValueError("Source config requires non-empty train.outputs")
    if not isinstance(validation_outputs, list) or not validation_outputs:
        raise ValueError("Source config requires non-empty validation.outputs")

    requested = set(holdout_basenames)
    selected_holdouts: list[dict[str, Any]] = []
    retained_train: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, output in enumerate(train_outputs):
        if not isinstance(output, dict) or not isinstance(output.get("y_path"), str):
            raise ValueError(f"train.outputs[{index}] is malformed")
        basename = Path(output["y_path"]).name
        if basename in seen_names:
            raise ValueError(f"Duplicate training output basename: {basename}")
        seen_names.add(basename)
        if basename in requested:
            selected_holdouts.append(copy.deepcopy(output))
        else:
            retained_train.append(copy.deepcopy(output))
    missing = requested - {Path(value["y_path"]).name for value in selected_holdouts}
    if missing:
        raise ValueError(f"Requested holdouts are not in train.outputs: {sorted(missing)}")
    if not retained_train:
        raise ValueError("POC holdouts would leave no training captures")

    existing_validation_paths: set[Path] = set()
    for index, value in enumerate(validation_outputs):
        if not isinstance(value, dict) or not isinstance(value.get("y_path"), str):
            raise ValueError(f"validation.outputs[{index}] is malformed")
        existing_validation_paths.add(Path(value["y_path"]).expanduser().resolve())
    holdout_paths = {
        Path(value["y_path"]).expanduser().resolve() for value in selected_holdouts
    }
    retained_paths = {
        Path(value["y_path"]).expanduser().resolve() for value in retained_train
    }
    if (
        existing_validation_paths & holdout_paths
        or existing_validation_paths & retained_paths
        or holdout_paths & retained_paths
    ):
        raise ValueError("Training and POC validation audio must be disjoint")

    frames, sample_rate, available_chunks = _audio_geometry(config, ny=ny)
    selected_indices = uniform_stratified_indices(
        available_chunks, chunks_per_output
    )
    metadata = config.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Source config metadata must be an object")
    control_order = metadata.get("control_order")
    if not isinstance(control_order, list) or not control_order:
        raise ValueError("Source config metadata.control_order must be a non-empty list")

    holdout_evidence = [
        {
            "source_train_index": train_outputs.index(
                next(
                    original
                    for original in train_outputs
                    if Path(original["y_path"]).name == Path(output["y_path"]).name
                )
            ),
            "basename": Path(output["y_path"]).name,
            "g_vector": output.get("g_vector"),
        }
        for output in selected_holdouts
    ]
    metadata["efficiency_poc"] = {
        "schema_version": 1,
        "run_id": run_id,
        "artifact_purpose": "workflow_validation_only",
        "source_config_path": str(source_path),
        "source_config_sha256": _sha256(source_path),
        "release_hidden_used": False,
        "selection_panel_role": "poc_selection_and_comparison_only",
        "holdout_policy": (
            "predeclared stratified control-space panel; moved from training before "
            "either arm was trained"
        ),
        "holdouts": holdout_evidence,
        "original_training_captures": len(train_outputs),
        "poc_training_captures": len(retained_train),
        "poc_validation_captures": len(validation_outputs)
        + len(selected_holdouts),
    }
    train["outputs"] = retained_train
    validation["outputs"] = copy.deepcopy(validation_outputs) + selected_holdouts

    common_arm = {
        "child": "full",
        "epochs": epochs,
        "batch_size": 16,
        "ny": ny,
        "validation_ny": 262_144,
        "expected_train_captures": len(retained_train),
        "artifact_purpose": "workflow_validation_only",
    }
    plan = {
        "schema_version": 1,
        "status": "frozen_local_preflight_cloud_submission_pending",
        "run_id": run_id,
        "objective": (
            "Measure whether deterministic half-corpus temporal exposure preserves "
            "conditioned-A2 quality while materially reducing cost."
        ),
        "hard_cost_ceiling_usd": hard_cost_ceiling_usd,
        "release_hidden_used": False,
        "audio": {
            "frames_per_capture": frames,
            "sample_rate": sample_rate,
            "receptive_field": RECEPTIVE_FIELD,
            "available_chunks_per_output": available_chunks,
            "full_audio_seconds_per_output": available_chunks * ny / sample_rate,
        },
        "dataset": {
            "source_config_path": str(source_path),
            "source_config_sha256": _sha256(source_path),
            "training_captures": len(retained_train),
            "validation_captures": len(validation_outputs)
            + len(selected_holdouts),
            "control_order": control_order,
            "holdouts": holdout_evidence,
        },
        "stage_1_matched_seed": [
            {
                **common_arm,
                "member_variant": "full190",
                "seed": baseline_seed,
                "train_chunks_per_output": None,
                "step_indices_sha256": _indices_sha256(
                    tuple(range(available_chunks))
                ),
            },
            {
                **common_arm,
                "member_variant": "stratified95",
                "seed": baseline_seed,
                "train_chunks_per_output": chunks_per_output,
                "selected_audio_seconds_per_output": chunks_per_output
                * ny
                / sample_rate,
                "step_indices_sha256": _indices_sha256(selected_indices),
            },
        ],
        "stage_2_conditional_replication": [
            {
                **common_arm,
                "member_variant": "full190",
                "seed": replicate_seed,
                "train_chunks_per_output": None,
            },
            {
                **common_arm,
                "member_variant": "stratified95",
                "seed": replicate_seed,
                "train_chunks_per_output": chunks_per_output,
            },
        ],
        "adoption_gates": {
            "minimum_independent_seeds": 2,
            "mean_and_p95_esr_relative_regression_max": 0.10,
            "individual_setting_esr_absolute_regression_max": 0.003,
            "new_crossing_above_esr_0_035_allowed": False,
            "measured_cost_ratio_max": 0.60,
            "current_core_render_required": True,
            "matched_listening_required_before_capture_policy_change": True,
        },
    }
    return config, plan


def build_poc_evaluation_reservation(
    config: dict[str, Any], *, materialized_config_path: Path
) -> dict[str, Any]:
    common = config.get("common")
    validation = config.get("validation")
    metadata = config.get("metadata")
    outputs = validation.get("outputs") if isinstance(validation, dict) else None
    control_order = metadata.get("control_order") if isinstance(metadata, dict) else None
    if (
        not isinstance(common, dict)
        or not isinstance(common.get("x_path"), str)
        or not isinstance(outputs, list)
        or not outputs
        or not isinstance(control_order, list)
        or not control_order
    ):
        raise ValueError("Materialized config is incomplete for POC evaluation")
    input_path = Path(common["x_path"]).expanduser().resolve()
    references = []
    for index, output in enumerate(outputs):
        if (
            not isinstance(output, dict)
            or not isinstance(output.get("y_path"), str)
            or not isinstance(output.get("g_vector"), list)
            or len(output["g_vector"]) != len(control_order)
        ):
            raise ValueError(f"validation.outputs[{index}] is malformed")
        path = Path(output["y_path"]).expanduser().resolve()
        capture_match = re.search(r"TRN-\d{3}", path.name)
        setting_id = (
            capture_match.group(0)
            if capture_match is not None
            else f"POC-VAL-{index:02d}"
        )
        references.append(
            {
                "setting_id": setting_id,
                "g_vector": list(output["g_vector"]),
                "path": str(path),
                "sha256": _sha256(path),
                "source_basename": path.name,
            }
        )
    config_path = materialized_config_path.expanduser().resolve()
    return {
        "schema_version": 1,
        "artifact_purpose": "workflow_validation_only",
        "target_audio_available_to_acquisition": False,
        "release_hidden_reservation_used": False,
        "evaluation_scope": {
            "role": "poc-comparison-holdout",
            "selection_use": "full-versus-reduced existing-corpus comparison only",
        },
        "control_order": list(control_order),
        "materialized_data_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "evaluation_input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
        },
        "references": references,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-plan", required=True, type=Path)
    parser.add_argument("--output-evaluation-reservation", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--holdout-basename", action="append", required=True)
    parser.add_argument("--chunks-per-output", type=int, default=556)
    parser.add_argument("--ny", type=int, default=DEFAULT_NY)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--baseline-seed", type=int, default=110)
    parser.add_argument("--replicate-seed", type=int, default=111)
    parser.add_argument("--hard-cost-ceiling-usd", type=float, default=14.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = (
        args.output_config.expanduser().resolve(),
        args.output_plan.expanduser().resolve(),
        args.output_evaluation_reservation.expanduser().resolve(),
    )
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    config, plan = materialize_efficiency_poc(
        source_config_path=args.source_config,
        holdout_basenames=args.holdout_basename,
        run_id=args.run_id,
        chunks_per_output=args.chunks_per_output,
        ny=args.ny,
        epochs=args.epochs,
        baseline_seed=args.baseline_seed,
        replicate_seed=args.replicate_seed,
        hard_cost_ceiling_usd=args.hard_cost_ceiling_usd,
    )
    _atomic_json(args.output_config, config, overwrite=args.overwrite)
    evaluation_reservation = build_poc_evaluation_reservation(
        config, materialized_config_path=args.output_config
    )
    _atomic_json(
        args.output_evaluation_reservation,
        evaluation_reservation,
        overwrite=args.overwrite,
    )
    plan["materialized_data_config_path"] = str(
        args.output_config.expanduser().resolve()
    )
    plan["materialized_data_config_sha256"] = _sha256(
        args.output_config.expanduser().resolve()
    )
    plan["evaluation_reservation_path"] = str(
        args.output_evaluation_reservation.expanduser().resolve()
    )
    plan["evaluation_reservation_sha256"] = _sha256(
        args.output_evaluation_reservation.expanduser().resolve()
    )
    _atomic_json(args.output_plan, plan, overwrite=args.overwrite)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
