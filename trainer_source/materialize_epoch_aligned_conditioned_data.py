"""Materialize a delay-zero conditioned-A2 corpus across capture epochs.

One physical capture project can cross a measured interface-latency boundary
without changing the amplifier itself.  Conditioned A2 still needs one shared
input and equal-length outputs, so this tool intersects the valid input range
for every declared integer delay and crops each output at the matching offset.
It never pads, resamples, or requantizes PCM data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import wave
from pathlib import Path
from typing import Any, Mapping


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_info(path: Path) -> tuple[int, int, int, int, str]:
    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is not supported: {path}")
        return (
            source.getnchannels(),
            source.getsampwidth(),
            source.getframerate(),
            source.getnframes(),
            source.getcomptype(),
        )


def _copy_wav_frames(source_path: Path, output_path: Path, *, start: int, stop: int) -> None:
    if output_path.exists():
        raise RuntimeError(f"Refusing to replace aligned WAV: {output_path}")
    with wave.open(str(source_path), "rb") as source:
        frame_count = source.getnframes()
        if not (0 <= start < stop <= frame_count):
            raise ValueError(
                f"invalid frame interval [{start}, {stop}) for {source_path} ({frame_count} frames)"
            )
        params = source.getparams()
        source.setpos(start)
        frames = source.readframes(stop - start)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as output:
        output.setparams(params)
        output.writeframes(frames)


def _load_delay_plan(path: Path, *, source_config_sha256: str) -> dict[str, dict[str, Any]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1:
        raise ValueError("alignment plan schema_version must be 1")
    if plan.get("source_config_sha256") != source_config_sha256:
        raise ValueError("alignment plan source_config_sha256 mismatch")
    records = plan.get("outputs")
    if not isinstance(records, list) or not records:
        raise ValueError("alignment plan outputs must be a non-empty list")
    by_hash: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"alignment plan outputs[{index}] must be an object")
        source_sha256 = record.get("source_sha256")
        delay = record.get("delay_samples")
        epoch_id = record.get("epoch_id")
        if not isinstance(source_sha256, str) or _SHA256_RE.fullmatch(source_sha256) is None:
            raise ValueError(f"alignment plan outputs[{index}] has an invalid source_sha256")
        if isinstance(delay, bool) or not isinstance(delay, int):
            raise ValueError(f"alignment plan outputs[{index}].delay_samples must be an integer")
        if not isinstance(epoch_id, str) or not epoch_id.strip():
            raise ValueError(f"alignment plan outputs[{index}].epoch_id must be non-empty")
        if source_sha256 in by_hash:
            raise ValueError(f"duplicate alignment-plan source hash: {source_sha256}")
        by_hash[source_sha256] = record
    return by_hash


def materialize_epoch_aligned_config(
    *,
    source_config_path: Path,
    alignment_plan_path: Path,
    output_config_path: Path,
    audio_dir: Path,
) -> dict[str, Any]:
    if output_config_path.exists():
        raise RuntimeError(f"Refusing to replace aligned data config: {output_config_path}")
    if audio_dir.exists():
        raise RuntimeError(f"Refusing to reuse aligned audio directory: {audio_dir}")

    source_config_sha256 = sha256_file(source_config_path)
    delay_plan = _load_delay_plan(
        alignment_plan_path, source_config_sha256=source_config_sha256
    )
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(source_config)
    common = config.get("common")
    if not isinstance(common, dict):
        raise ValueError("source data config has no common object")
    if common.get("delay", 0) != 0:
        raise ValueError("epoch alignment requires an unapplied source config with common.delay=0")
    if common.get("allow_unequal_lengths", False):
        raise ValueError("epoch alignment requires equal source lengths")

    x_source = Path(str(common.get("x_path", ""))).expanduser().resolve()
    if not x_source.is_file():
        raise FileNotFoundError(x_source)
    x_info = _wav_info(x_source)
    frame_count = x_info[3]

    outputs: list[tuple[str, int, dict[str, Any], Path, str, dict[str, Any]]] = []
    seen_source_hashes: set[str] = set()
    for split in ("train", "validation"):
        split_config = config.get(split)
        if not isinstance(split_config, dict) or not isinstance(split_config.get("outputs"), list):
            raise ValueError(f"source data config has no {split}.outputs list")
        for index, item in enumerate(split_config["outputs"]):
            if not isinstance(item, dict):
                raise ValueError(f"{split}.outputs[{index}] must be an object")
            y_source = Path(str(item.get("y_path", ""))).expanduser().resolve()
            if not y_source.is_file():
                raise FileNotFoundError(y_source)
            if _wav_info(y_source) != x_info:
                raise ValueError(f"output format/length does not match shared input: {y_source}")
            source_sha256 = sha256_file(y_source)
            if source_sha256 in seen_source_hashes:
                raise ValueError(f"duplicate source output hash in data config: {source_sha256}")
            seen_source_hashes.add(source_sha256)
            record = delay_plan.get(source_sha256)
            if record is None:
                raise ValueError(f"alignment plan has no record for output hash {source_sha256}")
            outputs.append((split, index, item, y_source, source_sha256, record))

    extra_hashes = set(delay_plan) - seen_source_hashes
    if extra_hashes:
        raise ValueError("alignment plan contains outputs absent from the source config")

    delays = [int(record["delay_samples"]) for *_, record in outputs]
    x_start = max(max(-delay, 0) for delay in delays)
    x_stop = min(frame_count - max(delay, 0) for delay in delays)
    if x_start >= x_stop:
        raise ValueError("capture delays have no common aligned input interval")
    aligned_frames = x_stop - x_start

    x_output = audio_dir / "input__aligned.wav"
    _copy_wav_frames(x_source, x_output, start=x_start, stop=x_stop)
    manifest_outputs: list[dict[str, Any]] = []
    for split, index, item, y_source, source_sha256, record in outputs:
        delay = int(record["delay_samples"])
        y_start = x_start + delay
        y_stop = x_stop + delay
        y_output = audio_dir / f"{split}-{index:03d}__{y_source.name}"
        _copy_wav_frames(y_source, y_output, start=y_start, stop=y_stop)
        item["y_path"] = str(y_output.resolve())
        manifest_outputs.append(
            {
                "aligned_path": str(y_output.resolve()),
                "aligned_sha256": sha256_file(y_output),
                "source_path": str(y_source),
                "source_sha256": source_sha256,
                "source_delay_samples": delay,
                "epoch_id": record["epoch_id"],
                "basis": record.get("basis"),
                "crop_start": y_start,
                "crop_stop": y_stop,
                "split": split,
                "index": index,
            }
        )

    common["x_path"] = str(x_output.resolve())
    common["delay"] = 0
    metadata = config.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    metadata["alignment_materialization"] = {
        "schema_version": 2,
        "source_config_path": str(source_config_path.resolve()),
        "source_config_sha256": source_config_sha256,
        "alignment_plan_path": str(alignment_plan_path.resolve()),
        "alignment_plan_sha256": sha256_file(alignment_plan_path),
        "method": (
            "intersection of nam.data.Dataset integer-delay input intervals; "
            "per-epoch output crop; no padding, resampling, or requantization"
        ),
        "source_frames": frame_count,
        "aligned_frames": aligned_frames,
        "input": {
            "source_path": str(x_source),
            "source_sha256": sha256_file(x_source),
            "aligned_path": str(x_output.resolve()),
            "aligned_sha256": sha256_file(x_output),
            "crop_start": x_start,
            "crop_stop": x_stop,
        },
        "outputs": manifest_outputs,
    }
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--alignment-plan", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    args = parser.parse_args()
    config = materialize_epoch_aligned_config(
        source_config_path=args.source_config.expanduser().resolve(),
        alignment_plan_path=args.alignment_plan.expanduser().resolve(),
        output_config_path=args.output_config.expanduser().resolve(),
        audio_dir=args.audio_dir.expanduser().resolve(),
    )
    materialization = config["metadata"]["alignment_materialization"]
    print(
        json.dumps(
            {
                "output_config": str(args.output_config.expanduser().resolve()),
                "delay": config["common"]["delay"],
                "aligned_frames": materialization["aligned_frames"],
                "sha256": sha256_file(args.output_config.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
