"""Create a hash-bound hidden-evaluation reservation before model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from dessmetal_capture.wav_io import inspect_wav


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_reservation(spec: Mapping[str, Any]) -> dict[str, Any]:
    control_order = spec.get("control_order")
    references = spec.get("references")
    input_raw = spec.get("evaluation_input")
    if (
        not isinstance(control_order, list)
        or not control_order
        or len(control_order) != len(set(control_order))
        or any(not isinstance(value, str) or not value for value in control_order)
    ):
        raise ValueError("spec control_order must contain unique non-empty ids")
    if not isinstance(references, list) or not references:
        raise ValueError("spec requires at least one hidden reference")
    if not isinstance(input_raw, str):
        raise ValueError("spec requires one unseen evaluation_input path")
    input_path = Path(input_raw).expanduser().resolve()
    input_info = inspect_wav(input_path)
    if (
        input_info["sample_rate"] != 48_000
        or input_info["channels"] != 1
        or input_info["sample_width_bits"] != 24
    ):
        raise ValueError("evaluation input must be 48 kHz mono PCM24")
    raw_scope = spec.get("evaluation_scope")
    if raw_scope is None:
        evaluation_scope = {"role": "frozen-coverage-holdout-on-unseen-di"}
    elif not isinstance(raw_scope, Mapping):
        raise ValueError("evaluation_scope must be an object")
    else:
        raw_role = raw_scope.get("role")
        if not isinstance(raw_role, str) or not raw_role.strip():
            raise ValueError("evaluation_scope.role must be a non-empty string")
        evaluation_scope = dict(raw_scope)
        evaluation_scope["role"] = raw_role.strip()
    normalized_references = []
    seen: set[tuple[float, ...]] = set()
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise ValueError(f"reference {index} must be an object")
        vector = reference.get("g_vector")
        raw_path = reference.get("path")
        if not isinstance(vector, list) or len(vector) != len(control_order):
            raise ValueError(f"reference {index} does not match control_order")
        values = tuple(float(value) for value in vector)
        if any(not math.isfinite(value) for value in values) or values in seen:
            raise ValueError(f"reference {index} is non-finite or duplicated")
        seen.add(values)
        if not isinstance(raw_path, str):
            raise ValueError(f"reference {index} has no WAV path")
        path = Path(raw_path).expanduser().resolve()
        info = inspect_wav(path)
        if (
            info["sample_rate"] != 48_000
            or info["channels"] != 1
            or info["sample_width_bits"] != 24
        ):
            raise ValueError(f"hidden reference {index} must be 48 kHz mono PCM24")
        normalized_reference = {
                "g_vector": list(values),
                "path": str(path),
                "sha256": info["sha256"],
                "frames": info["frames"],
                "sample_rate": info["sample_rate"],
        }
        raw_setting_id = reference.get("setting_id")
        if raw_setting_id is not None:
            if not isinstance(raw_setting_id, str) or not raw_setting_id.strip():
                raise ValueError(f"reference {index} setting_id must be a non-empty string")
            normalized_reference["setting_id"] = raw_setting_id.strip()
        raw_teacher_sha256 = reference.get("teacher_sha256")
        if raw_teacher_sha256 is not None:
            if (
                not isinstance(raw_teacher_sha256, str)
                or len(raw_teacher_sha256) != 64
                or any(character not in "0123456789abcdef" for character in raw_teacher_sha256)
            ):
                raise ValueError(
                    f"reference {index} teacher_sha256 must be one lowercase SHA-256 digest"
                )
            normalized_reference["teacher_sha256"] = raw_teacher_sha256
        normalized_references.append(normalized_reference)
    raw_provenance = spec.get("provenance")
    if raw_provenance is not None and not isinstance(raw_provenance, Mapping):
        raise ValueError("provenance must be an object")
    return {
        "schema_version": 1,
        "control_order": control_order,
        "target_audio_available_to_acquisition": False,
        "evaluation_scope": evaluation_scope,
        "evaluation_input": {
            "path": str(input_path),
            "sha256": input_info["sha256"],
            "frames": input_info["frames"],
            "sample_rate": input_info["sample_rate"],
        },
        "references": normalized_references,
        "provenance": dict(raw_provenance or {}),
        "notes": str(spec.get("notes", "")),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec_path = args.spec.expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("reservation spec root must be an object")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace existing reservation: {output}")
    reservation = build_reservation(spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(reservation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "reference_count": len(reservation["references"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
