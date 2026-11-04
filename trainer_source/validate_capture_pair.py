"""Validate repeatability or low/high control sensitivity from two aligned capture WAVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_aligned(path: Path, *, trim_start: int, trim_end: int) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    if audio.ndim != 1 or not np.all(np.isfinite(audio)):
        raise ValueError(f"capture must be finite mono audio: {path}")
    stop = len(audio) - trim_end if trim_end else len(audio)
    if trim_start < 0 or trim_end < 0 or stop <= trim_start:
        raise ValueError("capture trim removes the complete signal")
    return audio[trim_start:stop], int(sample_rate)


def error_to_signal(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("capture pair must have equal aligned lengths")
    energy = float(np.sum(reference * reference, dtype=np.float64))
    if energy <= 0.0:
        raise ValueError("reference capture has no energy")
    error = candidate - reference
    return float(np.sum(error * error, dtype=np.float64) / energy)


def validate(args: argparse.Namespace) -> dict[str, object]:
    reference_path = args.reference.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    reference, sample_rate = read_aligned(
        reference_path, trim_start=args.trim_start_frames, trim_end=args.trim_end_frames
    )
    candidate, candidate_rate = read_aligned(
        candidate_path, trim_start=args.trim_start_frames, trim_end=args.trim_end_frames
    )
    if sample_rate != candidate_rate:
        raise ValueError("capture pair sample rates differ")
    metric = error_to_signal(reference, candidate)
    if args.mode == "repeatability":
        threshold = args.maximum_esr
        passed = metric <= threshold
        criterion = "esr_lte"
    else:
        threshold = args.minimum_delta_esr
        passed = metric >= threshold
        criterion = "delta_esr_gte"
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    return {
        "status": "passed" if passed else "failed",
        "mode": args.mode,
        "metric": "sum((candidate-reference)^2)/sum(reference^2)",
        "esr": metric,
        "criterion": criterion,
        "threshold": threshold,
        "sample_rate": sample_rate,
        "frames": len(reference),
        "trim_start_frames": args.trim_start_frames,
        "trim_end_frames": args.trim_end_frames,
        "reference": {"path": str(reference_path), "sha256": sha256_file(reference_path)},
        "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("repeatability", "sensitivity"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trim-start-frames", type=int, default=0)
    parser.add_argument("--trim-end-frames", type=int, default=0)
    parser.add_argument("--maximum-esr", type=float, default=0.01)
    parser.add_argument("--minimum-delta-esr", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate(args)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise RuntimeError(f"Refusing to replace existing validation report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
