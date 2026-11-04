from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


CONTROL_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
MAX_CONTROLS = 12
MAX_CANDIDATE_POINTS = 4096


class DomainError(ValueError):
    """Raised when a persistent capture-domain invariant is violated."""


@dataclass(frozen=True)
class ControlDefinition:
    id: str
    label: str
    minimum: float
    maximum: float
    step: float
    unit: str = ""
    taper: str = "linear"
    display_minimum: float | None = None
    display_maximum: float | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ControlDefinition":
        control = cls(
            id=str(raw.get("id", "")).strip(),
            label=str(raw.get("label", "")).strip(),
            minimum=_finite_float(raw.get("minimum"), "minimum"),
            maximum=_finite_float(raw.get("maximum"), "maximum"),
            step=_finite_float(raw.get("step"), "step"),
            unit=str(raw.get("unit", "")).strip(),
            taper=str(raw.get("taper", "linear")).strip().lower(),
            display_minimum=_optional_finite_float(raw.get("display_minimum"), "display_minimum"),
            display_maximum=_optional_finite_float(raw.get("display_maximum"), "display_maximum"),
        )
        control.validate()
        return control

    def validate(self) -> None:
        if not CONTROL_ID_RE.fullmatch(self.id):
            raise DomainError(
                f"Control id {self.id!r} must start with a lowercase letter and contain only lowercase letters, "
                "digits, underscores, or hyphens"
            )
        if not self.label or len(self.label) > 80:
            raise DomainError("Control label must contain 1 to 80 characters")
        if self.minimum >= self.maximum:
            raise DomainError(f"Control {self.id!r} minimum must be less than maximum")
        if self.step <= 0 or self.step > self.maximum - self.minimum:
            raise DomainError(f"Control {self.id!r} step must be positive and no larger than its range")
        if self.taper not in {"linear", "log", "switch"}:
            raise DomainError(f"Unsupported taper {self.taper!r}")
        if (self.display_minimum is None) != (self.display_maximum is None):
            raise DomainError("Display minimum and maximum must either both be set or both be omitted")
        if self.display_minimum is not None and self.display_minimum >= self.display_maximum:
            raise DomainError("Display minimum must be less than display maximum")

    def grid_values(self) -> list[float]:
        span = self.maximum - self.minimum
        steps = int(math.floor(span / self.step + 1e-9))
        values = [round(self.minimum + index * self.step, 10) for index in range(steps + 1)]
        if not math.isclose(values[-1], self.maximum, rel_tol=0.0, abs_tol=1e-8):
            values.append(round(self.maximum, 10))
        return values

    def normalize(self, value: float) -> float:
        return (value - self.minimum) / (self.maximum - self.minimum)

    def display_value(self, value: float) -> float:
        if self.display_minimum is None or self.display_maximum is None:
            return value
        normalized = self.normalize(value)
        return self.display_minimum + normalized * (self.display_maximum - self.display_minimum)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "unit": self.unit,
            "taper": self.taper,
            "display_minimum": self.display_minimum,
            "display_maximum": self.display_maximum,
        }


def validate_controls(raw_controls: Sequence[Mapping[str, Any]]) -> list[ControlDefinition]:
    if not raw_controls or len(raw_controls) > MAX_CONTROLS:
        raise DomainError(f"A project requires between 1 and {MAX_CONTROLS} controls")
    controls = [ControlDefinition.from_mapping(raw) for raw in raw_controls]
    ids = [control.id for control in controls]
    if len(ids) != len(set(ids)):
        raise DomainError("Control ids must be unique")
    return controls


def generate_candidate_points(controls: Sequence[ControlDefinition]) -> list[dict[str, float]]:
    grids = [control.grid_values() for control in controls]
    total = math.prod(len(grid) for grid in grids)
    if total > MAX_CANDIDATE_POINTS:
        raise DomainError(
            f"The control grid contains {total} points; provide an explicit candidate set with at most "
            f"{MAX_CANDIDATE_POINTS} points"
        )
    return [
        {control.id: float(value) for control, value in zip(controls, values)}
        for values in itertools.product(*grids)
    ]


def validate_candidate_points(
    raw_points: Sequence[Mapping[str, Any]], controls: Sequence[ControlDefinition]
) -> list[dict[str, float]]:
    if not raw_points or len(raw_points) > MAX_CANDIDATE_POINTS:
        raise DomainError(f"Candidate set requires between 1 and {MAX_CANDIDATE_POINTS} points")
    expected = {control.id for control in controls}
    validated: list[dict[str, float]] = []
    seen: set[str] = set()
    for raw in raw_points:
        if set(raw) != expected:
            raise DomainError(f"Every candidate must contain exactly these controls: {sorted(expected)}")
        point: dict[str, float] = {}
        for control in controls:
            value = _finite_float(raw[control.id], control.id)
            if value < control.minimum - 1e-9 or value > control.maximum + 1e-9:
                raise DomainError(f"Candidate value {control.id}={value} lies outside the control range")
            point[control.id] = round(value, 10)
        key = settings_key(point)
        if key in seen:
            raise DomainError(f"Duplicate candidate point {point}")
        seen.add(key)
        validated.append(point)
    return validated


def settings_key(settings: Mapping[str, Any]) -> str:
    normalized = {str(key): round(_finite_float(value, str(key)), 10) for key, value in settings.items()}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def settings_display(
    settings: Mapping[str, float], controls: Sequence[ControlDefinition]
) -> list[dict[str, Any]]:
    return [
        {
            "id": control.id,
            "label": control.label,
            "model_value": settings[control.id],
            "display_value": control.display_value(settings[control.id]),
            "unit": control.unit,
        }
        for control in controls
    ]


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise DomainError(f"{name} must be a number") from error
    if not math.isfinite(parsed):
        raise DomainError(f"{name} must be finite")
    return parsed


def _optional_finite_float(value: Any, name: str) -> float | None:
    if value is None or value == "":
        return None
    return _finite_float(value, name)
