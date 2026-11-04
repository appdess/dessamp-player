from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .domain import ControlDefinition, DomainError, validate_candidate_points


MAX_PANAMA_POINTS = 64


@dataclass(frozen=True)
class PanamaSuggestion:
    source_index: int
    settings: dict[str, float]
    score: float
    rationale: str


def parse_new_points(
    raw_points: Sequence[Mapping[str, Any]],
    *,
    controls: Sequence[ControlDefinition],
    control_order: Sequence[str],
) -> list[PanamaSuggestion]:
    """Translate PANAMA's new_points.json vectors into named capture settings.

    PANAMA serializes each point as ``{"index": N, "g_vector": [...]}`` and
    does not carry control names in that file. Requiring an explicit order at
    the boundary prevents a valid-looking Gain/Bass/Treble permutation from
    silently becoming capture instructions.
    """

    if not raw_points or len(raw_points) > MAX_PANAMA_POINTS:
        raise DomainError(f"PANAMA import requires between 1 and {MAX_PANAMA_POINTS} points")

    expected_order = [control.id for control in controls]
    supplied_order = [str(value).strip() for value in control_order]
    if len(supplied_order) != len(set(supplied_order)):
        raise DomainError("PANAMA control_order must not contain duplicates")
    if set(supplied_order) != set(expected_order) or len(supplied_order) != len(expected_order):
        raise DomainError(f"PANAMA control_order must contain exactly these controls: {expected_order}")

    mapped_points: list[dict[str, float]] = []
    source_indices: list[int] = []
    scores: list[float] = []
    seen_indices: set[int] = set()
    for position, raw in enumerate(raw_points):
        if not isinstance(raw, Mapping):
            raise DomainError("PANAMA points must be JSON objects")
        source_index = raw.get("index")
        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            raise DomainError("Every PANAMA point requires a non-negative integer index")
        if source_index in seen_indices:
            raise DomainError(f"Duplicate PANAMA point index {source_index}")
        seen_indices.add(source_index)

        vector = raw.get("g_vector")
        if not isinstance(vector, list) or len(vector) != len(supplied_order):
            raise DomainError(
                f"PANAMA point {source_index} must contain {len(supplied_order)} g_vector values"
            )
        settings: dict[str, float] = {}
        for control_id, raw_value in zip(supplied_order, vector):
            if isinstance(raw_value, bool):
                raise DomainError(f"PANAMA point {source_index} contains a non-numeric value")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise DomainError(f"PANAMA point {source_index} contains a non-numeric value") from error
            if not math.isfinite(value):
                raise DomainError(f"PANAMA point {source_index} contains a non-finite value")
            settings[control_id] = value
        mapped_points.append(settings)
        source_indices.append(source_index)

        raw_score = raw.get("disagreement_score", raw.get("score", 0.0))
        if isinstance(raw_score, bool):
            raise DomainError(f"PANAMA point {source_index} contains an invalid disagreement score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise DomainError(f"PANAMA point {source_index} contains an invalid disagreement score") from error
        if not math.isfinite(score):
            raise DomainError(f"PANAMA point {source_index} contains a non-finite disagreement score")
        scores.append(score)

    # This validates exact control membership, finite values, ranges, and
    # duplicate settings while intentionally allowing continuous off-grid
    # points proposed by PANAMA's gradient optimization.
    validated = validate_candidate_points(mapped_points, controls)
    suggestions: list[PanamaSuggestion] = []
    for position, (source_index, settings, score) in enumerate(zip(source_indices, validated, scores)):
        score_note = (
            f" Exported disagreement score: {score:.6g}." if score != 0.0 else " The upstream file did not export its disagreement score."
        )
        suggestions.append(
            PanamaSuggestion(
                source_index=source_index,
                settings=settings,
                score=score,
                rationale=(
                    f"PANAMA ensemble-disagreement suggestion {position + 1} of {len(validated)} "
                    f"(source index {source_index}).{score_note}"
                ),
            )
        )
    return suggestions


def source_digest(raw_points: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(raw_points, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
