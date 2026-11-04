"""Fit a diagnostic capture-count curve from fixed-protocol audit reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_PREFERRED_MEAN_ESR = 0.005


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_metric(value: Any, label: str) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(metric) or metric <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return metric


def fit_power_law(capture_counts: Sequence[int], values: Sequence[float]) -> dict[str, float]:
    if len(capture_counts) != len(values) or len(values) < 2:
        raise ValueError("power-law fit requires at least two paired points")
    if len(set(capture_counts)) != len(capture_counts) or any(count < 1 for count in capture_counts):
        raise ValueError("capture counts must be unique positive integers")
    x = [math.log(float(count)) for count in capture_counts]
    y = [math.log(finite_metric(value, "curve value")) for value in values]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0.0:
        raise ValueError("capture counts do not span a fit range")
    slope = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x, y)) / denominator
    intercept = y_mean - slope * x_mean
    predictions = [intercept + slope * value for value in x]
    total = sum((value - y_mean) ** 2 for value in y)
    residual = sum((value - prediction) ** 2 for value, prediction in zip(y, predictions))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return {
        "coefficient": math.exp(intercept),
        "exponent": -slope,
        "r_squared_log_space": r_squared,
    }


def projected_count(fit: Mapping[str, float], target: float) -> int | None:
    target = finite_metric(target, "projection target")
    exponent = float(fit["exponent"])
    coefficient = float(fit["coefficient"])
    if not math.isfinite(exponent) or exponent <= 0.0:
        return None
    return int(math.ceil((coefficient / target) ** (1.0 / exponent)))


def parse_point(raw: str) -> tuple[int, Path]:
    raw_count, separator, raw_path = raw.partition("=")
    if not separator or not raw_count or not raw_path:
        raise ValueError("each --point must use CAPTURE_COUNT=/absolute/report.json")
    try:
        count = int(raw_count)
    except ValueError as error:
        raise ValueError("capture count must be an integer") from error
    path = Path(raw_path).expanduser().resolve()
    if count < 1 or not path.is_file():
        raise ValueError(f"invalid curve point: {raw!r}")
    return count, path


def _summary(record: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} record is missing")
    summary = record.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"{label} summary is missing")
    return summary


def analyze(
    raw_points: Sequence[str],
    *,
    preferred_mean_esr: float = DEFAULT_PREFERRED_MEAN_ESR,
) -> dict[str, Any]:
    preferred_mean_esr = finite_metric(preferred_mean_esr, "preferred mean ESR")
    parsed = sorted((parse_point(raw) for raw in raw_points), key=lambda item: item[0])
    if len(parsed) < 2 or len({count for count, _ in parsed}) != len(parsed):
        raise ValueError("at least two unique capture-count points are required")
    points: list[dict[str, Any]] = []
    acceptance_contract: tuple[float, float] | None = None
    quality_target_contract: tuple[float, float] | None = None
    for capture_count, path in parsed:
        report = json.loads(path.read_text(encoding="utf-8"))
        ensemble = _summary(report.get("ensemble"), "ensemble")
        acceptance = report.get("acceptance")
        individual = report.get("individual")
        if not isinstance(acceptance, Mapping) or not isinstance(individual, Mapping) or not individual:
            raise ValueError(f"report is missing acceptance or individual records: {path}")
        mean_gate = finite_metric(acceptance.get("required_mean_esr_lte"), "mean ESR gate")
        max_gate = finite_metric(acceptance.get("required_max_esr_lte"), "max ESR gate")
        contract = (mean_gate, max_gate)
        if acceptance_contract is None:
            acceptance_contract = contract
        elif acceptance_contract != contract:
            raise ValueError("curve reports do not share one acceptance contract")
        quality_target = report.get("quality_target")
        if not isinstance(quality_target, Mapping):
            quality_target = acceptance
        target_contract = (
            finite_metric(
                quality_target.get("required_mean_esr_lte"),
                "quality-target mean ESR gate",
            ),
            finite_metric(
                quality_target.get("required_max_esr_lte"),
                "quality-target max ESR gate",
            ),
        )
        if quality_target_contract is None:
            quality_target_contract = target_contract
        elif quality_target_contract != target_contract:
            raise ValueError("curve reports do not share one quality-target contract")
        members: list[tuple[float, float, str, Mapping[str, Any]]] = []
        for label, record in individual.items():
            summary = _summary(record, f"member {label}")
            members.append(
                (
                    finite_metric(summary.get("mean_esr"), f"member {label} mean ESR"),
                    finite_metric(summary.get("max_esr"), f"member {label} max ESR"),
                    str(label),
                    summary,
                )
            )
        _, _, best_label, best_summary = min(members)
        points.append(
            {
                "capture_count": capture_count,
                "report_path": str(path),
                "report_sha256": sha256_file(path),
                "ensemble_mean_esr": finite_metric(ensemble.get("mean_esr"), "ensemble mean ESR"),
                "ensemble_max_esr": finite_metric(ensemble.get("max_esr"), "ensemble max ESR"),
                "best_single_member": best_label,
                "best_single_mean_esr": finite_metric(best_summary.get("mean_esr"), "single mean ESR"),
                "best_single_max_esr": finite_metric(best_summary.get("max_esr"), "single max ESR"),
            }
        )
    assert acceptance_contract is not None
    assert quality_target_contract is not None
    count_values = [point["capture_count"] for point in points]
    series = {
        "ensemble_mean_esr": [point["ensemble_mean_esr"] for point in points],
        "ensemble_max_esr": [point["ensemble_max_esr"] for point in points],
        "best_single_mean_esr": [point["best_single_mean_esr"] for point in points],
        "best_single_max_esr": [point["best_single_max_esr"] for point in points],
    }
    fits: dict[str, Any] = {}
    for label, values in series.items():
        all_points = fit_power_law(count_values, values)
        latest_segment = fit_power_law(count_values[-2:], values[-2:])
        acceptable_target = acceptance_contract[0] if "mean" in label else acceptance_contract[1]
        target = quality_target_contract[0] if "mean" in label else quality_target_contract[1]
        acceptable_projections = [
            value
            for value in (
                projected_count(all_points, acceptable_target),
                projected_count(latest_segment, acceptable_target),
            )
            if value is not None
        ]
        projections = [
            value
            for value in (
                projected_count(all_points, target),
                projected_count(latest_segment, target),
            )
            if value is not None
        ]
        fit_record = {
            "all_points": all_points,
            "latest_segment": latest_segment,
            "acceptable_target": acceptable_target,
            "acceptable_projected_capture_count_range": (
                [min(acceptable_projections), max(acceptable_projections)]
                if acceptable_projections
                else None
            ),
            "gate_target": target,
            "projected_capture_count_range": (
                [min(projections), max(projections)] if projections else None
            ),
        }
        if "mean" in label:
            preferred_projections = [
                value
                for value in (
                    projected_count(all_points, preferred_mean_esr),
                    projected_count(latest_segment, preferred_mean_esr),
                )
                if value is not None
            ]
            fit_record["preferred_target"] = preferred_mean_esr
            fit_record["preferred_projected_capture_count_range"] = (
                [min(preferred_projections), max(preferred_projections)]
                if preferred_projections
                else None
            )
        fits[label] = fit_record
    latest = points[-1]
    previous = points[-2]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "diagnostic log-log power-law fit; not a physical-amp guarantee",
        "point_count": len(points),
        "points": points,
        "acceptance": {
            "required_mean_esr_lte": acceptance_contract[0],
            "required_max_esr_lte": acceptance_contract[1],
            "quality_target": {
                "required_mean_esr_lte": quality_target_contract[0],
                "required_max_esr_lte": quality_target_contract[1],
            },
            "preferred_mean_esr_lte": preferred_mean_esr,
        },
        "latest_change": {
            "from_capture_count": previous["capture_count"],
            "to_capture_count": latest["capture_count"],
            "ensemble_mean_reduction_fraction": 1.0
            - latest["ensemble_mean_esr"] / previous["ensemble_mean_esr"],
            "ensemble_max_reduction_fraction": 1.0
            - latest["ensemble_max_esr"] / previous["ensemble_max_esr"],
        },
        "fits": fits,
        "limitations": [
            "The reports must share architecture, epochs, alignment, metric code, hidden DI, and hidden settings.",
            "Three or fewer curve points give a planning range, not a reliable asymptote.",
            "An amp-sim curve excludes physical routing noise, drift, latency, calibration, and knob-setting error.",
            "The real-time player requires one single member; ensemble projections are acquisition diagnostics.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point",
        action="append",
        required=True,
        help="CAPTURE_COUNT=/absolute/path/to/fixed-protocol-report.json; repeat",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preferred-mean-esr", type=float, default=DEFAULT_PREFERRED_MEAN_ESR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace existing curve report: {output}")
    result = analyze(args.point, preferred_mean_esr=args.preferred_mean_esr)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"point_count": result["point_count"], "fits": result["fits"]}, sort_keys=True))


if __name__ == "__main__":
    main()
