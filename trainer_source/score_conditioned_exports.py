"""Rank capture candidates by disagreement between native conditioned exports.

Reserved audit vectors and already captured vectors are removed before any
native render is launched. This makes the resulting report suitable for a
subsequent active-capture round without exposing the audit targets to the
acquisition pass.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from evaluate_conditioned_exports import (
    parse_members,
    read_mono,
    resolve_receptive_field,
    sha256_file,
)


DEFAULT_BLOCK_SIZE = 64


def vector_key(vector: Iterable[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 10) for value in vector)


def available_candidates(config: dict[str, Any]) -> list[tuple[int, list[float]]]:
    raw_candidates = config.get("candidate_vectors")
    raw_captured = config.get("captured_vectors")
    raw_reserved = config.get("reserved_evaluation_vectors")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate config requires candidate_vectors")
    if not isinstance(raw_captured, list) or not isinstance(raw_reserved, list):
        raise ValueError(
            "candidate config requires captured_vectors and reserved_evaluation_vectors"
        )
    captured = {vector_key(vector) for vector in raw_captured}
    reserved = {vector_key(vector) for vector in raw_reserved}
    if captured & reserved:
        raise ValueError("captured and reserved vector sets overlap")
    available: list[tuple[int, list[float]]] = []
    seen: set[tuple[float, ...]] = set()
    dimensions: int | None = None
    for index, raw_vector in enumerate(raw_candidates):
        vector = [float(value) for value in raw_vector]
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError("candidate vectors must be finite and non-empty")
        if dimensions is None:
            dimensions = len(vector)
        elif len(vector) != dimensions:
            raise ValueError("candidate vectors must have a consistent dimension")
        key = vector_key(vector)
        if key in seen:
            raise ValueError("candidate vectors contain a duplicate")
        seen.add(key)
        if key not in captured and key not in reserved:
            available.append((index, vector))
    if not available:
        raise ValueError("no audit-safe uncaptured candidates remain")
    return available


def score(args: argparse.Namespace) -> dict[str, Any]:
    if args.block_size < 1 or args.render_timeout_seconds < 1:
        raise ValueError("block size and render timeout must be positive")
    candidate_config_path = (
        args.candidate_config.expanduser().resolve()
        if args.candidate_config is not None
        else args.run_root.expanduser().resolve() / "00_WORKFLOW" / "candidate-config.json"
    )
    candidate_config = json.loads(candidate_config_path.read_text(encoding="utf-8"))
    candidates = available_candidates(candidate_config)
    members = parse_members(args.member)
    receptive_field = resolve_receptive_field(members, args.receptive_field)
    renderer = args.renderer.expanduser().resolve()
    if not renderer.is_file():
        raise ValueError(f"native renderer is not a file: {renderer}")
    input_path = args.input.expanduser().resolve()
    input_audio, sample_rate = read_mono(input_path)
    trim_start = receptive_field - 1
    if len(input_audio) <= trim_start:
        raise ValueError("scoring input is shorter than the model receptive field")
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise ValueError(f"refusing to replace existing report: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    explicit_render_dir = args.render_dir is not None
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if explicit_render_dir:
        render_dir = args.render_dir.expanduser().resolve()
        render_dir.mkdir(parents=True, exist_ok=False)
    else:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="dessmetal-conditioned-disagreement-"
        )
        render_dir = Path(temporary_directory.name)

    ranking: list[dict[str, Any]] = []
    try:
        for source_index, vector in candidates:
            predictions: list[np.ndarray] = []
            member_peak = 0.0
            for label, model_path in members.items():
                render_path = render_dir / f"candidate-{source_index:04d}-member-{label}.wav"
                command = [str(renderer), str(model_path), str(input_path), str(render_path)]
                for value in vector:
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
                    raise RuntimeError(f"candidate {source_index}, member {label}: {detail}")
                prediction, _ = read_mono(render_path, expected_rate=sample_rate)
                if len(prediction) != len(input_audio):
                    raise ValueError(f"native render length mismatch: {render_path}")
                member_peak = max(member_peak, float(np.max(np.abs(prediction))))
                predictions.append(prediction[trim_start:])
            matrix = np.stack(predictions, axis=0)
            variance = np.var(matrix, axis=0, ddof=1, dtype=np.float64)
            ranking.append(
                {
                    "index": source_index,
                    "g_vector": vector,
                    "disagreement_score": float(np.mean(variance, dtype=np.float64)),
                    "member_peak": member_peak,
                }
            )
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

    ranking.sort(key=lambda item: (-item["disagreement_score"], item["index"]))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "mean unbiased sample variance across native NAM renders after receptive-field trim",
        "protocol": {
            "reserved_vectors_excluded_before_render": True,
            "captured_vectors_excluded_before_render": True,
            "target_audio_read": False,
            "receptive_field": receptive_field,
            "trim_start_frames": trim_start,
            "block_size": args.block_size,
        },
        "candidate_config_path": str(candidate_config_path),
        "candidate_config_sha256": sha256_file(candidate_config_path),
        "candidate_count": len(candidate_config["candidate_vectors"]),
        "captured_count": len(candidate_config["captured_vectors"]),
        "reserved_count": len(candidate_config["reserved_evaluation_vectors"]),
        "available_count": len(candidates),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "models": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in members.items()
        },
        "ranking": ranking,
        "render_directory": str(render_dir) if explicit_render_dir else None,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path, help="Legacy regression-harness root")
    source.add_argument(
        "--candidate-config",
        type=Path,
        help="Hash-bound generic candidate/reservation config",
    )
    parser.add_argument("--renderer", required=True, type=Path)
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument(
        "--receptive-field",
        type=int,
        help="Required when exports do not declare metadata.receptive_field",
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--render-timeout-seconds", type=int, default=300)
    return parser


def main() -> None:
    report = score(build_parser().parse_args())
    print(
        json.dumps(
            {
                "available_count": report["available_count"],
                "top": report["ranking"][0],
            }
        )
    )


if __name__ == "__main__":
    main()
