from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .domain import ControlDefinition, DomainError, settings_key


class AcquisitionStrategy(Protocol):
    name: str

    def suggest(
        self,
        controls: Sequence[ControlDefinition],
        candidate_points: Sequence[Mapping[str, float]],
        captured_points: Sequence[Mapping[str, float]],
        pending_points: Sequence[Mapping[str, float]],
    ) -> "AcquisitionSuggestion": ...


@dataclass(frozen=True)
class AcquisitionSuggestion:
    settings: dict[str, float]
    rationale: str
    score: float
    strategy: str


class CoverageAcquisitionStrategy:
    """Deterministic farthest-point baseline for workflow and hardware validation.

    This is intentionally not presented as PANAMA. It gives the persistent app
    a useful acquisition loop before a trained ensemble is available, and it
    exercises the same strategy boundary that a PANAMA adapter will implement.
    """

    name = "coverage-v1"

    def suggest(
        self,
        controls: Sequence[ControlDefinition],
        candidate_points: Sequence[Mapping[str, float]],
        captured_points: Sequence[Mapping[str, float]],
        pending_points: Sequence[Mapping[str, float]],
    ) -> AcquisitionSuggestion:
        excluded = {settings_key(point) for point in [*captured_points, *pending_points]}
        available = [dict(point) for point in candidate_points if settings_key(point) not in excluded]
        if not available:
            raise DomainError("All candidate settings have already been captured or recommended")

        if not captured_points:
            center = {control.id: (control.minimum + control.maximum) / 2.0 for control in controls}
            selected = min(
                available,
                key=lambda point: (self._distance(point, center, controls), settings_key(point)),
            )
            return AcquisitionSuggestion(
                settings=selected,
                rationale="Start near the center of the defined control space before exploring its boundaries.",
                score=0.0,
                strategy=self.name,
            )

        ranked: list[tuple[float, str, dict[str, float]]] = []
        for point in available:
            nearest = min(self._distance(point, captured, controls) for captured in captured_points)
            ranked.append((nearest, settings_key(point), point))
        score, _, selected = max(ranked, key=lambda item: (item[0], item[1]))
        return AcquisitionSuggestion(
            settings=selected,
            rationale=(
                "Choose the setting furthest from the captured settings in normalized control space. "
                "This is a coverage baseline; PANAMA will replace the score with ensemble disagreement."
            ),
            score=score,
            strategy=self.name,
        )

    @staticmethod
    def _distance(
        left: Mapping[str, float], right: Mapping[str, float], controls: Sequence[ControlDefinition]
    ) -> float:
        return math.sqrt(
            sum((control.normalize(left[control.id]) - control.normalize(right[control.id])) ** 2 for control in controls)
        )


def get_strategy(name: str) -> AcquisitionStrategy:
    if name == CoverageAcquisitionStrategy.name:
        return CoverageAcquisitionStrategy()
    if name == "panama":
        raise DomainError(
            "PANAMA acquisition is not configured yet. Complete captures remain compatible with the future adapter."
        )
    raise DomainError(f"Unknown acquisition strategy {name!r}")
