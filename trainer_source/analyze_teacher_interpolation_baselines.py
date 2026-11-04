#!/usr/bin/env python3
"""Measure simple public-snapshot baselines on frozen teacher holdouts."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from evaluate_conditioned_exports import (
    aligned_esr,
    evaluated_frame_count,
    metric_summary,
    read_mono,
    sha256_file,
)


def _audio_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def analyze(spec_path: Path) -> dict[str, Any]:
    spec_path = spec_path.expanduser().resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("baseline spec must be a JSON object")
    settings = payload.get("settings")
    if not isinstance(settings, list) or not settings:
        raise ValueError("baseline spec requires at least one setting")
    receptive_field = int(payload.get("receptive_field", 6347))
    chunk_size = int(payload.get("chunk_size", 8192))
    trim_start = receptive_field - 1

    results: list[dict[str, Any]] = []
    for index, raw in enumerate(settings):
        if not isinstance(raw, Mapping):
            raise ValueError(f"setting {index} must be an object")
        paths: dict[str, Path] = {}
        for key in ("target", "lower", "upper"):
            value = raw.get(key)
            if not isinstance(value, str):
                raise ValueError(f"setting {index} has no {key} path")
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            paths[key] = path
        alpha = float(raw.get("alpha", 0.5))
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"setting {index} alpha must be within [0,1]")

        target_audio, sample_rate = read_mono(paths["target"])
        lower_audio, _ = read_mono(paths["lower"], expected_rate=sample_rate)
        upper_audio, _ = read_mono(paths["upper"], expected_rate=sample_rate)
        if not (len(target_audio) == len(lower_audio) == len(upper_audio)):
            raise ValueError(f"setting {index} WAV lengths differ")
        covered = evaluated_frame_count(
            len(target_audio), receptive_field=receptive_field, chunk_size=chunk_size
        )
        stop = trim_start + covered
        target = target_audio[trim_start:stop]
        lower = lower_audio[trim_start:stop]
        upper = upper_audio[trim_start:stop]
        blended = (1.0 - alpha) * lower + alpha * upper
        lower_esr = aligned_esr(lower, target)
        upper_esr = aligned_esr(upper, target)
        results.append(
            {
                "setting_index": index,
                "label": str(raw.get("label", f"setting-{index:02d}")),
                "g_vector": [float(value) for value in raw.get("g_vector", [])],
                "alpha": alpha,
                "sample_rate": sample_rate,
                "evaluated_frames": covered,
                "target": _audio_record(paths["target"]),
                "lower": _audio_record(paths["lower"]),
                "upper": _audio_record(paths["upper"]),
                "metrics": {
                    "lower_adjacent_snapshot_esr": lower_esr,
                    "upper_adjacent_snapshot_esr": upper_esr,
                    "lower_tie_single_snapshot_esr": lower_esr,
                    "optimistic_best_adjacent_snapshot_esr": min(lower_esr, upper_esr),
                    "linear_output_blend_esr": aligned_esr(blended, target),
                },
            }
        )

    metric_names = tuple(results[0]["metrics"])
    summary = {
        name: metric_summary([float(item["metrics"][name]) for item in results])
        for name in metric_names
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec": _audio_record(spec_path),
        "method": {
            "metric": "sum((prediction-target)^2)/sum(target^2), float64",
            "receptive_field": receptive_field,
            "trim_start_frames": trim_start,
            "chunk_size_for_coverage": chunk_size,
            "single_snapshot_tie_break": "lower",
            "optimistic_baseline": "minimum ESR of the two equidistant adjacent snapshots",
            "linear_output_blend": "sample-wise interpolation of adjacent native teacher renders",
        },
        "settings": results,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace existing report: {output}")
    report = analyze(args.spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
