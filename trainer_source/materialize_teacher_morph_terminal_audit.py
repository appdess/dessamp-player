"""Open a frozen two-dimensional teacher-morph audit after member selection.

The command renders the frozen US/BRIT endpoint packages on the untouched DI,
applies the training corpus's immutable blend law, and writes one 25-point
surface reservation. It also derives two endpoint reservations from the
original teacher audits by appending the fixed Morph endpoint to each existing
Gain vector. No target is rendered or inspected before the hash-bound internal
member-selection report exists.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from dessmetal_capture.wav_io import load_mono_float32

from materialize_teacher_morph_data_config import (
    _render_endpoint,
    _tag,
    _write_target,
    atomic_write_json,
    checked_input,
    checked_teacher,
    sha256_file,
)


def read_object(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def require_hash(path: Path, expected: Any, label: str) -> str:
    resolved = path.expanduser().resolve()
    actual = sha256_file(resolved)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return actual


def _finite_unit(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be finite and normalized to [0,1]")
    return parsed


def derive_endpoint_reservation(
    *,
    source_path: Path,
    expected_source_sha256: str,
    expected_count: int,
    morph: float,
    role: str,
    evaluation_input: Mapping[str, Any],
    plan_record: Mapping[str, Any],
    selection_record: Mapping[str, Any],
) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve()
    source_sha256 = require_hash(source_path, expected_source_sha256, f"{role} source reservation")
    source = read_object(source_path, f"{role} source reservation")
    if source.get("target_audio_available_to_acquisition") is not False:
        raise ValueError(f"{role} source reservation violates the acquisition boundary")
    if source.get("control_order") != ["gain_index"]:
        raise ValueError(f"{role} source reservation control order is unsupported")
    source_input = source.get("evaluation_input")
    if not isinstance(source_input, Mapping) or source_input.get("sha256") != evaluation_input.get("sha256"):
        raise ValueError(f"{role} source reservation uses a different external DI")
    references = source.get("references")
    if not isinstance(references, list) or len(references) != expected_count:
        raise ValueError(f"{role} source reservation count disagrees with the frozen plan")

    derived: list[dict[str, Any]] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise ValueError(f"{role} reference {index} must be an object")
        vector = reference.get("g_vector")
        path_value = reference.get("path")
        if not isinstance(vector, list) or len(vector) != 1 or not isinstance(path_value, str):
            raise ValueError(f"{role} reference {index} is incomplete")
        gain = _finite_unit(vector[0], f"{role} Gain vector")
        target_path = Path(path_value).expanduser().resolve()
        target_sha256 = require_hash(target_path, reference.get("sha256"), f"{role} reference {index}")
        derived.append(
            {
                "setting_id": reference.get("setting_id"),
                "g_vector": [gain, morph],
                "path": str(target_path),
                "sha256": target_sha256,
                "frames": int(reference.get("frames", evaluation_input["frames"])),
                "sample_rate": int(reference.get("sample_rate", evaluation_input["sample_rate"])),
                "source_teacher_sha256": reference.get("teacher_sha256"),
            }
        )

    return {
        "schema_version": 1,
        "control_order": ["gain", "morph"],
        "evaluation_input": dict(evaluation_input),
        "evaluation_scope": {
            "role": f"original-{role.lower()}-snapshot-endpoint-on-unseen-di",
            "release_gate": True,
            "morph_endpoint": morph,
            "source_reference_count": expected_count,
        },
        "references": derived,
        "provenance": {
            "frozen_plan": dict(plan_record),
            "internal_member_selection": dict(selection_record),
            "source_reservation": {
                "path": str(source_path),
                "sha256": source_sha256,
            },
            "vector_derivation": "append fixed Morph endpoint to the source Gain vector",
            "target_audio_re_rendered": False,
        },
        "target_audio_available_to_acquisition": False,
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.expanduser().resolve()
    selection_path = args.selection_report.expanduser().resolve()
    corpus_path = args.corpus_evidence.expanduser().resolve()
    selected_model = args.selected_model.expanduser().resolve()
    renderer = args.renderer.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to replace immutable audit root: {output_root}")
    if not renderer.is_file() or not os.access(renderer, os.X_OK):
        raise ValueError(f"renderer is unavailable or not executable: {renderer}")

    plan = read_object(plan_path, "frozen terminal audit plan")
    selection = read_object(selection_path, "internal member selection")
    corpus = read_object(corpus_path, "morph corpus evidence")
    plan_sha256 = sha256_file(plan_path)
    selection_sha256 = sha256_file(selection_path)
    corpus_sha256 = sha256_file(corpus_path)
    if plan.get("control_order") != ["gain", "morph"] or plan.get("direction") != "US_to_BRIT":
        raise ValueError("frozen plan does not describe the Gain/Morph US-to-BRIT contract")
    if selection.get("criterion") != "minimum best_internal_validation_esr":
        raise ValueError("member selection used an unsupported criterion")
    if selection.get("selection_opened_hidden_evidence") is not False:
        raise ValueError("member selection was not internal-only")
    if selection.get("selected_model_sha256") != sha256_file(selected_model):
        raise ValueError("selected model does not match the internal-only selection report")
    if corpus.get("external_audit_materialized") is not False:
        raise ValueError("corpus evidence says the external audit was already opened")
    if corpus.get("blend_formula") != "(1+4*(center_gain-1)*m*(1-m))*((1-m)*US+m*BRIT)":
        raise ValueError("corpus blend formula is unsupported")
    center_gain = float(corpus.get("center_gain", math.nan))
    if not math.isfinite(center_gain) or not 0.25 <= center_gain <= 4.0:
        raise ValueError("corpus center gain is invalid")

    evaluation_input_plan = plan.get("evaluation_input")
    if not isinstance(evaluation_input_plan, Mapping) or not isinstance(evaluation_input_plan.get("path"), str):
        raise ValueError("frozen plan has no evaluation input")
    evaluation_input = checked_input(Path(evaluation_input_plan["path"]), "frozen evaluation input")
    if evaluation_input["sha256"] != evaluation_input_plan.get("sha256"):
        raise ValueError("frozen evaluation input SHA-256 mismatch")
    if evaluation_input["sha256"] in {
        corpus.get("training_input", {}).get("sha256"),
        corpus.get("validation_input", {}).get("sha256"),
    }:
        raise ValueError("external DI overlaps the training or internal-validation DI")

    endpoint_plan = plan.get("endpoint_packages")
    if not isinstance(endpoint_plan, Mapping):
        raise ValueError("frozen plan has no endpoint package records")
    us_teacher = checked_teacher(Path(endpoint_plan["us"]["path"]), "US")
    brit_teacher = checked_teacher(Path(endpoint_plan["brit"]["path"]), "BRIT")
    if us_teacher["package_sha256"] != endpoint_plan["us"].get("sha256"):
        raise ValueError("US endpoint package SHA-256 mismatch")
    if brit_teacher["package_sha256"] != endpoint_plan["brit"].get("sha256"):
        raise ValueError("BRIT endpoint package SHA-256 mismatch")
    corpus_teacher_hashes = {entry.get("role"): entry.get("package_sha256") for entry in corpus.get("teachers", [])}
    if corpus_teacher_hashes != {"US": us_teacher["package_sha256"], "BRIT": brit_teacher["package_sha256"]}:
        raise ValueError("frozen endpoint packages disagree with the training corpus")

    surface = plan.get("surface_audit")
    if not isinstance(surface, Mapping):
        raise ValueError("frozen plan has no surface audit")
    gains = [_finite_unit(value, "surface Gain") for value in surface.get("gain_values", [])]
    morphs = [_finite_unit(value, "surface Morph") for value in surface.get("morph_values", [])]
    if len(gains) * len(morphs) != int(surface.get("reference_count", -1)):
        raise ValueError("frozen surface grid count is inconsistent")

    output_root.mkdir(parents=True)
    endpoint_records: list[dict[str, Any]] = []
    surface_references: list[dict[str, Any]] = []
    input_path = Path(evaluation_input["path"])
    for gain in gains:
        us_path = output_root / "surface-endpoints" / f"audit-us-g{_tag(gain)}.wav"
        brit_path = output_root / "surface-endpoints" / f"audit-brit-g{_tag(gain)}.wav"
        us_record = _render_endpoint(
            renderer=renderer,
            teacher=Path(us_teacher["path"]),
            input_path=input_path,
            output_path=us_path,
            gain=gain,
            expected_input=evaluation_input,
        )
        brit_record = _render_endpoint(
            renderer=renderer,
            teacher=Path(brit_teacher["path"]),
            input_path=input_path,
            output_path=brit_path,
            gain=gain,
            expected_input=evaluation_input,
        )
        us_audio, us_rate = load_mono_float32(us_path)
        brit_audio, brit_rate = load_mono_float32(brit_path)
        if us_rate != 48_000 or brit_rate != 48_000 or us_audio.shape != brit_audio.shape:
            raise ValueError("surface endpoint renders disagree on rate or length")
        endpoint_records.append({"gain": gain, "us": us_record, "brit": brit_record})
        for morph in morphs:
            target_path = output_root / "surface-targets" / f"audit-g{_tag(gain)}-m{_tag(morph)}.wav"
            target = _write_target(
                output_path=target_path,
                us_audio=us_audio,
                brit_audio=brit_audio,
                gain=gain,
                morph=morph,
                center_gain=center_gain,
                sample_rate=48_000,
            )
            surface_references.append(
                {
                    "setting_id": f"REDLINE-SURFACE-G{_tag(gain)}-M{_tag(morph)}",
                    "g_vector": target["g_vector"],
                    "path": target["path"],
                    "sha256": target["sha256"],
                    "frames": target["frames"],
                    "sample_rate": 48_000,
                }
            )

    frozen_plan_record = {"path": str(plan_path), "sha256": plan_sha256}
    selection_record = {
        "path": str(selection_path),
        "sha256": selection_sha256,
        "selected_member": selection["selected_member"],
        "selected_model_sha256": selection["selected_model_sha256"],
    }
    surface_reservation = {
        "schema_version": 1,
        "control_order": ["gain", "morph"],
        "evaluation_input": evaluation_input,
        "evaluation_scope": {
            "role": "unseen-two-dimensional-teacher-surface-on-unseen-di",
            "release_gate": True,
            "reference_count": len(surface_references),
        },
        "references": surface_references,
        "provenance": {
            "frozen_plan": frozen_plan_record,
            "internal_member_selection": selection_record,
            "corpus_evidence": {"path": str(corpus_path), "sha256": corpus_sha256},
            "renderer": {"path": str(renderer), "sha256": sha256_file(renderer)},
            "endpoint_packages": {"us": us_teacher, "brit": brit_teacher},
            "blend_formula": corpus["blend_formula"],
            "center_gain": center_gain,
            "per_setting_alignment": False,
            "per_setting_normalization": False,
        },
        "target_audio_available_to_acquisition": False,
    }

    source_endpoint = plan.get("source_endpoint_audit")
    if not isinstance(source_endpoint, Mapping):
        raise ValueError("frozen plan has no source endpoint audit")
    us_reservation = derive_endpoint_reservation(
        source_path=Path(source_endpoint["us"]["source_reservation"]),
        expected_source_sha256=source_endpoint["us"]["source_reservation_sha256"],
        expected_count=int(source_endpoint["us"]["reference_count"]),
        morph=float(source_endpoint["us"]["morph"]),
        role="US",
        evaluation_input=evaluation_input,
        plan_record=frozen_plan_record,
        selection_record=selection_record,
    )
    brit_reservation = derive_endpoint_reservation(
        source_path=Path(source_endpoint["brit"]["source_reservation"]),
        expected_source_sha256=source_endpoint["brit"]["source_reservation_sha256"],
        expected_count=int(source_endpoint["brit"]["reference_count"]),
        morph=float(source_endpoint["brit"]["morph"]),
        role="BRIT",
        evaluation_input=evaluation_input,
        plan_record=frozen_plan_record,
        selection_record=selection_record,
    )

    surface_path = output_root / "surface-reservation.json"
    us_path = output_root / "us-endpoint-reservation.json"
    brit_path = output_root / "brit-endpoint-reservation.json"
    atomic_write_json(surface_path, surface_reservation)
    atomic_write_json(us_path, us_reservation)
    atomic_write_json(brit_path, brit_reservation)
    evidence = {
        "schema_version": 1,
        "status": "materialized_after_internal_member_selection",
        "frozen_plan": frozen_plan_record,
        "internal_member_selection": selection_record,
        "selected_model": {"path": str(selected_model), "sha256": sha256_file(selected_model)},
        "corpus_evidence": {"path": str(corpus_path), "sha256": corpus_sha256},
        "surface_endpoint_renders": endpoint_records,
        "reservations": {
            "surface": {"path": str(surface_path), "sha256": sha256_file(surface_path), "count": len(surface_references)},
            "us_endpoint": {"path": str(us_path), "sha256": sha256_file(us_path), "count": len(us_reservation["references"])},
            "brit_endpoint": {"path": str(brit_path), "sha256": sha256_file(brit_path), "count": len(brit_reservation["references"])},
        },
        "target_audio_available_to_training_or_selection": False,
    }
    evidence_path = output_root / "materialization-evidence.json"
    atomic_write_json(evidence_path, evidence)
    return {
        "status": evidence["status"],
        "surface_reservation": str(surface_path),
        "us_endpoint_reservation": str(us_path),
        "brit_endpoint_reservation": str(brit_path),
        "materialization_evidence": str(evidence_path),
        "surface_count": len(surface_references),
        "us_count": len(us_reservation["references"]),
        "brit_count": len(brit_reservation["references"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--selected-model", type=Path, required=True)
    parser.add_argument("--corpus-evidence", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(materialize(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
