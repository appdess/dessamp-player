"""Materialize sample-aligned conditioned-A2 WAVs from a data config.

The conditioned-A2 loader deliberately accepts only ``delay=0``.  This tool
applies the integer delay convention used by ``nam.data.Dataset`` to the shared
input and every output, writes new PCM WAVs without padding, and emits a new
hash-bound data config whose delay is zero.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import wave
from pathlib import Path
from typing import Any


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


def materialize_aligned_config(
    *, source_config_path: Path, output_config_path: Path, audio_dir: Path
) -> dict[str, Any]:
    if output_config_path.exists():
        raise RuntimeError(f"Refusing to replace aligned data config: {output_config_path}")
    if audio_dir.exists():
        raise RuntimeError(f"Refusing to reuse aligned audio directory: {audio_dir}")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(source_config)
    common = config.get("common")
    if not isinstance(common, dict):
        raise ValueError("source data config has no common object")
    delay = common.get("delay", 0)
    if isinstance(delay, bool) or not isinstance(delay, int):
        raise ValueError("common.delay must be an integer")
    if delay == 0:
        raise ValueError("source data config is already delay=0")
    if common.get("allow_unequal_lengths", False):
        raise ValueError("alignment materialization requires equal source lengths")

    x_source = Path(str(common.get("x_path", ""))).expanduser().resolve()
    if not x_source.is_file():
        raise FileNotFoundError(x_source)
    x_info = _wav_info(x_source)
    frame_count = x_info[3]
    if abs(delay) >= frame_count:
        raise ValueError("delay removes the complete capture")

    outputs: list[tuple[str, int, dict[str, Any], Path]] = []
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
            outputs.append((split, index, item, y_source))

    # Match nam.data.Dataset._apply_delay_int exactly.
    x_start = max(-delay, 0)
    x_stop = frame_count - max(delay, 0)
    y_start = max(delay, 0)
    y_stop = frame_count - max(-delay, 0)
    aligned_frames = frame_count - abs(delay)

    x_output = audio_dir / "input__aligned.wav"
    _copy_wav_frames(x_source, x_output, start=x_start, stop=x_stop)
    manifest_outputs: list[dict[str, Any]] = []
    for split, index, item, y_source in outputs:
        y_output = audio_dir / f"{split}-{index:03d}__{y_source.name}"
        _copy_wav_frames(y_source, y_output, start=y_start, stop=y_stop)
        item["y_path"] = str(y_output.resolve())
        manifest_outputs.append(
            {
                "aligned_sha256": sha256_file(y_output),
                "aligned_path": str(y_output.resolve()),
                "source_sha256": sha256_file(y_source),
                "source_path": str(y_source),
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
        "schema_version": 1,
        "source_config_path": str(source_config_path.resolve()),
        "source_config_sha256": sha256_file(source_config_path),
        "source_delay_samples": delay,
        "method": "nam.data.Dataset._apply_delay_int integer crop; no padding",
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
    output_config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    args = parser.parse_args()
    config = materialize_aligned_config(
        source_config_path=args.source_config.expanduser().resolve(),
        output_config_path=args.output_config.expanduser().resolve(),
        audio_dir=args.audio_dir.expanduser().resolve(),
    )
    print(json.dumps({
        "output_config": str(args.output_config.expanduser().resolve()),
        "delay": config["common"]["delay"],
        "aligned_frames": config["metadata"]["alignment_materialization"]["aligned_frames"],
        "sha256": sha256_file(args.output_config.expanduser().resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
