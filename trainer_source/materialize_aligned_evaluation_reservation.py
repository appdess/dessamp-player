"""Materialize aligned hidden references and a hash-bound reservation.

This consumes the persisted evaluation plan plus the authoritative aligned
training config.  It never edits capture masters: each accepted hidden target
is cropped with the same integer-delay convention as the training bundle, and
the resulting input/reference bytes are bound into a new evaluation
reservation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any, Mapping

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _wav_signature(path: Path) -> tuple[int, int, int, int, str]:
    with wave.open(str(path), "rb") as source:
        return (
            source.getnchannels(),
            source.getsampwidth(),
            source.getframerate(),
            source.getnframes(),
            source.getcomptype(),
        )


def _inspect_wav(path: Path) -> dict[str, Any]:
    channels, sample_width, sample_rate, frames, compression = _wav_signature(path)
    if compression != "NONE":
        raise ValueError(f"compressed WAV is not supported: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "channels": channels,
        "sample_width_bits": sample_width * 8,
        "sample_rate": sample_rate,
        "frames": frames,
    }


def _copy_wav_frames(source_path: Path, output_path: Path, *, start: int, stop: int) -> None:
    if output_path.exists():
        raise RuntimeError(f"Refusing to replace aligned hidden WAV: {output_path}")
    with wave.open(str(source_path), "rb") as source:
        frame_count = source.getnframes()
        if source.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is not supported: {source_path}")
        if not (0 <= start < stop <= frame_count):
            raise ValueError(
                f"invalid frame interval [{start}, {stop}) for {source_path} "
                f"({frame_count} frames)"
            )
        params = source.getparams()
        source.setpos(start)
        frames = source.readframes(stop - start)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as output:
        output.setparams(params)
        output.writeframes(frames)


def _normalized_vector(
    settings: Mapping[str, Any], control_order: list[str], *, context: str
) -> list[float]:
    if set(settings) != set(control_order):
        raise ValueError(f"{context} controls do not match authoritative control_order")
    vector = [float(settings[control_id]) for control_id in control_order]
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{context} contains a non-finite control value")
    return vector


def materialize_aligned_reservation(
    *,
    evaluation_plan_path: Path,
    training_data_config_path: Path,
    output_reservation_path: Path,
    audio_dir: Path,
) -> dict[str, Any]:
    if output_reservation_path.exists():
        raise RuntimeError(f"Refusing to replace evaluation reservation: {output_reservation_path}")
    if audio_dir.exists():
        raise RuntimeError(f"Refusing to reuse aligned hidden audio directory: {audio_dir}")

    evaluation_plan = _load_object(evaluation_plan_path)
    training_config = _load_object(training_data_config_path)
    metadata = training_config.get("metadata")
    common = training_config.get("common")
    if not isinstance(metadata, Mapping) or not isinstance(common, Mapping):
        raise ValueError("training data config is missing common/metadata")
    control_order_raw = metadata.get("control_order")
    if (
        not isinstance(control_order_raw, list)
        or not control_order_raw
        or any(not isinstance(value, str) or not value for value in control_order_raw)
    ):
        raise ValueError("training data config has no valid control_order")
    control_order = list(control_order_raw)

    alignment = metadata.get("alignment_materialization")
    if not isinstance(alignment, Mapping):
        raise ValueError("training data config has no alignment materialization evidence")
    source_frames = alignment.get("source_frames")
    aligned_frames = alignment.get("aligned_frames")
    if (
        isinstance(source_frames, bool)
        or not isinstance(source_frames, int)
        or isinstance(aligned_frames, bool)
        or not isinstance(aligned_frames, int)
    ):
        raise ValueError("training alignment frame/delay contract is invalid")
    alignment_schema = alignment.get("schema_version", 1)
    source_delay = alignment.get("source_delay_samples")
    source_input_record = alignment.get("input")
    if not isinstance(source_input_record, Mapping):
        raise ValueError("aligned/source training input evidence is missing")
    if alignment_schema == 1:
        if (
            isinstance(source_delay, bool)
            or not isinstance(source_delay, int)
            or aligned_frames != source_frames - abs(source_delay)
        ):
            raise ValueError("training alignment frame/delay contract is invalid")
        input_crop_start = max(-source_delay, 0)
        input_crop_stop = source_frames - max(source_delay, 0)
    elif alignment_schema == 2:
        input_crop_start = source_input_record.get("crop_start")
        input_crop_stop = source_input_record.get("crop_stop")
        if (
            isinstance(input_crop_start, bool)
            or not isinstance(input_crop_start, int)
            or isinstance(input_crop_stop, bool)
            or not isinstance(input_crop_stop, int)
            or not (0 <= input_crop_start < input_crop_stop <= source_frames)
            or input_crop_stop - input_crop_start != aligned_frames
        ):
            raise ValueError("epoch-aligned training input crop contract is invalid")
    else:
        raise ValueError(f"unsupported alignment materialization schema: {alignment_schema!r}")
    if common.get("delay") != 0 or common.get("allow_unequal_lengths") is not False:
        raise ValueError("authoritative training config must already be equal-length delay=0")

    aligned_input = Path(str(common.get("x_path", ""))).expanduser().resolve()
    if not aligned_input.is_file():
        raise ValueError("aligned/source training input evidence is missing")
    source_input = Path(str(source_input_record.get("source_path", ""))).expanduser().resolve()
    if not source_input.is_file():
        raise FileNotFoundError(source_input)
    if source_input_record.get("source_sha256") != sha256_file(source_input):
        raise ValueError("source training input hash mismatch")
    if source_input_record.get("aligned_sha256") != sha256_file(aligned_input):
        raise ValueError("aligned training input hash mismatch")

    source_signature = _wav_signature(source_input)
    aligned_signature = _wav_signature(aligned_input)
    if source_signature != (1, 3, 48_000, source_frames, "NONE"):
        raise ValueError("source evaluation stimulus must be mono PCM24/48k with expected frames")
    if aligned_signature != (1, 3, 48_000, aligned_frames, "NONE"):
        raise ValueError("aligned evaluation stimulus has an unexpected format or length")

    scope = evaluation_plan.get("current_hidden_evaluation_scope")
    if scope is None:
        scope = evaluation_plan.get("initial_hidden_audit_scope")
    hidden_settings = evaluation_plan.get("hidden_settings")
    if not isinstance(scope, Mapping) or not isinstance(hidden_settings, list):
        raise ValueError("evaluation plan has no hidden-audit scope/settings")
    accepted_ids = scope.get("accepted_hidden_ids")
    if not isinstance(accepted_ids, list) or not accepted_ids:
        raise ValueError("evaluation plan has no accepted hidden ids")
    hidden_by_id = {
        entry.get("id"): entry
        for entry in hidden_settings
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
    }
    if len(hidden_by_id) != len(hidden_settings):
        raise ValueError("evaluation plan has invalid or duplicate hidden ids")

    references: list[dict[str, Any]] = []
    derivations: list[dict[str, Any]] = []
    seen_vectors: set[tuple[float, ...]] = set()
    for hidden_id in accepted_ids:
        if not isinstance(hidden_id, str) or hidden_id not in hidden_by_id:
            raise ValueError(f"accepted hidden id is missing from plan: {hidden_id!r}")
        plan_entry = hidden_by_id[hidden_id]
        if not str(plan_entry.get("capture_status", "")).startswith("accepted_hidden_audit_only"):
            raise ValueError(f"{hidden_id} is not an accepted hidden capture")
        normalized = plan_entry.get("normalized")
        evidence_raw = plan_entry.get("evidence")
        if not isinstance(normalized, Mapping) or not isinstance(evidence_raw, str):
            raise ValueError(f"{hidden_id} plan entry is incomplete")
        vector = _normalized_vector(normalized, control_order, context=f"{hidden_id} plan")
        vector_key = tuple(vector)
        if vector_key in seen_vectors:
            raise ValueError(f"duplicate accepted hidden vector: {hidden_id}")
        seen_vectors.add(vector_key)

        evidence_path = (evaluation_plan_path.parent / evidence_raw).resolve()
        evidence = _load_object(evidence_path)
        if evidence.get("id") != hidden_id or evidence.get("quality_status") != "pass":
            raise ValueError(f"{hidden_id} evidence identity/quality is invalid")
        if evidence.get("role") != "hidden_audit_reference":
            raise ValueError(f"{hidden_id} evidence role is not hidden-audit-only")
        exclusion = evidence.get("training_exclusion")
        if not isinstance(exclusion, Mapping) or any(
            exclusion.get(key) is not False
            for key in ("enters_training_sqlite", "enters_recommendations", "enters_training_corpus")
        ):
            raise ValueError(f"{hidden_id} training separation evidence is invalid")
        evidence_settings = evidence.get("settings_normalized")
        if not isinstance(evidence_settings, Mapping):
            raise ValueError(f"{hidden_id} evidence has no normalized settings")
        evidence_vector = _normalized_vector(
            evidence_settings, control_order, context=f"{hidden_id} evidence"
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(vector, evidence_vector)
        ):
            raise ValueError(f"{hidden_id} plan/evidence vector mismatch")
        validation = evidence.get("validation")
        reference_delay = validation.get("latency_samples") if isinstance(validation, Mapping) else None
        if isinstance(reference_delay, bool) or not isinstance(reference_delay, int):
            raise ValueError(f"{hidden_id} has no valid latency evidence")
        if alignment_schema == 1 and reference_delay != source_delay:
            raise ValueError(f"{hidden_id} latency differs from the training alignment contract")

        y_start = input_crop_start + reference_delay
        y_stop = input_crop_stop + reference_delay
        if not (0 <= y_start < y_stop <= source_frames):
            raise ValueError(
                f"{hidden_id} latency cannot be aligned to the authoritative input interval"
            )

        reference_record = evidence.get("reference_audio")
        if not isinstance(reference_record, Mapping):
            raise ValueError(f"{hidden_id} evidence has no reference audio")
        reference_source = Path(str(reference_record.get("path", ""))).expanduser().resolve()
        if not reference_source.is_file():
            raise FileNotFoundError(reference_source)
        if reference_record.get("sha256") != sha256_file(reference_source):
            raise ValueError(f"{hidden_id} reference hash mismatch")
        if _wav_signature(reference_source) != source_signature:
            raise ValueError(f"{hidden_id} reference format/length differs from source stimulus")

        aligned_output = audio_dir / f"{hidden_id.lower()}__aligned-pcm24.wav"
        _copy_wav_frames(reference_source, aligned_output, start=y_start, stop=y_stop)
        aligned_info = _inspect_wav(aligned_output)
        if _wav_signature(aligned_output) != aligned_signature:
            raise ValueError(f"{hidden_id} aligned reference does not match aligned input")
        references.append({"g_vector": vector, "path": str(aligned_output.resolve())})
        derivations.append(
            {
                "hidden_id": hidden_id,
                "evidence_path": str(evidence_path),
                "evidence_sha256": sha256_file(evidence_path),
                "source_path": str(reference_source),
                "source_sha256": sha256_file(reference_source),
                "aligned_path": str(aligned_output.resolve()),
                "aligned_sha256": aligned_info["sha256"],
                "crop_start": y_start,
                "crop_stop": y_stop,
                "source_delay_samples": reference_delay,
                "g_vector": vector,
            }
        )

    # Build the same core reservation schema as create_evaluation_reservation.py,
    # then retain explicit provenance that this first audit uses the standardized
    # training stimulus.  It is a hidden-setting audit, not an unseen-DI gate.
    input_info = _inspect_wav(aligned_input)
    reservation = {
        "schema_version": 1,
        "control_order": control_order,
        "target_audio_available_to_acquisition": False,
        "evaluation_input": {
            "path": str(aligned_input),
            "sha256": input_info["sha256"],
            "frames": input_info["frames"],
            "sample_rate": input_info["sample_rate"],
        },
        "references": [
            {
                "g_vector": item["g_vector"],
                "path": item["path"],
                "sha256": sha256_file(Path(item["path"])),
                "frames": aligned_frames,
                "sample_rate": 48_000,
            }
            for item in references
        ],
        "evaluation_scope": {
            "role": "frozen_hidden_settings_on_standardized_training_stimulus",
            "unseen_control_vectors": True,
            "unseen_dry_performance": False,
            "release_gate_interpretation": "provisional_hidden_setting_interpolation_audit",
        },
        "alignment_materialization": {
            "schema_version": 1,
            "method": "nam.data.Dataset._apply_delay_int integer crop; no padding",
            "source_delay_samples": source_delay if alignment_schema == 1 else None,
            "source_delay_samples_by_hidden_id": {
                item["hidden_id"]: item["source_delay_samples"] for item in derivations
            },
            "input_crop_start": input_crop_start,
            "input_crop_stop": input_crop_stop,
            "source_frames": source_frames,
            "aligned_frames": aligned_frames,
            "input_reused_from_authoritative_training_bundle": True,
            "training_data_config_path": str(training_data_config_path),
            "training_data_config_sha256": sha256_file(training_data_config_path),
            "evaluation_plan_path": str(evaluation_plan_path),
            "evaluation_plan_sha256": sha256_file(evaluation_plan_path),
            "references": derivations,
        },
        "notes": (
            "Accepted physical hidden-setting references, aligned byte-exactly with the shared "
            "training delay. The dry stimulus is the same standardized NAM v3 signal used for "
            "training, so this does not satisfy the separate unseen-performance-DI release gate."
        ),
    }
    output_reservation_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_reservation_path.with_name(f".{output_reservation_path.name}.tmp")
    temporary.write_text(json.dumps(reservation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_reservation_path)
    return reservation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-plan", required=True, type=Path)
    parser.add_argument("--training-data-config", required=True, type=Path)
    parser.add_argument("--output-reservation", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    args = parser.parse_args()
    output_path = args.output_reservation.expanduser().resolve()
    reservation = materialize_aligned_reservation(
        evaluation_plan_path=args.evaluation_plan.expanduser().resolve(),
        training_data_config_path=args.training_data_config.expanduser().resolve(),
        output_reservation_path=output_path,
        audio_dir=args.audio_dir.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "output_reservation": str(output_path),
                "sha256": sha256_file(output_path),
                "reference_count": len(reservation["references"]),
                "aligned_frames": reservation["evaluation_input"]["frames"],
                "evaluation_role": reservation["evaluation_scope"]["role"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
