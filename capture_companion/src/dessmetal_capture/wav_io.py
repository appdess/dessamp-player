from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .domain import DomainError


MAX_WAV_BYTES = 1024 * 1024 * 1024


def inspect_wav(path: Path | str) -> dict[str, Any]:
    wav_path = Path(path)
    if not wav_path.is_file():
        raise DomainError(f"WAV file does not exist: {wav_path}")
    size = wav_path.stat().st_size
    if size <= 0 or size > MAX_WAV_BYTES:
        raise DomainError("WAV file is empty or exceeds the 1 GiB application limit")
    digest = hashlib.sha256()
    peak = 0.0
    try:
        with wav_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        with wave.open(str(wav_path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise DomainError("Only uncompressed PCM WAV files are supported")
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.getnframes()
            if channels <= 0 or channels > 32:
                raise DomainError("WAV channel count is invalid")
            if sample_rate < 8_000 or sample_rate > 384_000:
                raise DomainError("WAV sample rate is outside the supported range")
            if sample_width not in {1, 2, 3, 4}:
                raise DomainError("Only 8-, 16-, 24-, and 32-bit PCM WAV files are supported")
            remaining = frames
            while remaining:
                count = min(65_536, remaining)
                raw = wav_file.readframes(count)
                if not raw:
                    break
                decoded = _decode_pcm(raw, sample_width)
                if decoded.size % channels:
                    raise DomainError("WAV audio data ends in a partial frame")
                if decoded.size:
                    peak = max(peak, float(np.max(np.abs(decoded))))
                remaining -= decoded.size // channels
            if remaining:
                raise DomainError("WAV audio data is shorter than its declared frame count")
    except (wave.Error, EOFError, ValueError) as error:
        raise DomainError(f"Invalid WAV file: {error}") from error
    if frames <= 0:
        raise DomainError("WAV file contains no audio frames")
    return {
        "path": str(wav_path.resolve()),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bits": sample_width * 8,
        "frames": frames,
        "duration_seconds": frames / sample_rate,
        "peak": peak,
        "peak_dbfs": 20.0 * math.log10(peak) if peak > 0.0 else None,
    }


def load_mono_float32(path: Path | str) -> tuple[np.ndarray, int]:
    wav_path = Path(path)
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise DomainError("Only uncompressed PCM WAV files are supported")
            if wav_file.getnchannels() != 1:
                raise DomainError("The standardized capture input must be mono")
            sample_width = wav_file.getsampwidth()
            if sample_width not in {1, 2, 3, 4}:
                raise DomainError("Unsupported PCM sample width")
            sample_rate = wav_file.getframerate()
            raw = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, EOFError) as error:
        raise DomainError(f"Invalid WAV file: {error}") from error
    return _decode_pcm(raw, sample_width).astype(np.float32, copy=False), sample_rate


def write_pcm24_mono(path: Path | str, audio: np.ndarray, sample_rate: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(mono)):
        raise DomainError("Captured audio contains non-finite samples")
    clipped = np.clip(mono, -1.0, 1.0)
    integers = np.rint(clipped * 8_388_607.0).astype("<i4")
    bytes_4 = integers.view(np.uint8).reshape(-1, 4)
    pcm24 = bytes_4[:, :3].tobytes()
    with wave.open(str(output), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(3)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm24)


def validate_capture_wav(
    wav_info: Mapping[str, Any],
    *,
    expected_sample_rate: int,
    expected_frames: int | None,
) -> tuple[str, dict[str, Any]]:
    failures: list[str] = []
    warnings: list[str] = []
    if int(wav_info["channels"]) != 1:
        failures.append("Capture must be mono")
    if int(wav_info["sample_rate"]) != int(expected_sample_rate):
        failures.append(
            f"Capture sample rate is {wav_info['sample_rate']} Hz; project requires {expected_sample_rate} Hz"
        )
    if int(wav_info["sample_width_bits"]) != 24:
        warnings.append(f"Capture is {wav_info['sample_width_bits']}-bit PCM; 24-bit PCM is recommended")
    peak = float(wav_info["peak"])
    if peak >= 0.999:
        warnings.append("Capture reaches or nearly reaches 0 dBFS; inspect it for clipping")
    if peak < 1e-5:
        failures.append("Capture is effectively silent")
    elif peak < 1e-3:
        warnings.append("Capture level is very low")
    length_delta_frames = None
    length_delta_seconds = None
    if expected_frames is not None:
        length_delta_frames = int(wav_info["frames"]) - int(expected_frames)
        length_delta_seconds = length_delta_frames / int(expected_sample_rate)
        if length_delta_frames != 0:
            if abs(length_delta_seconds) > 1.0:
                failures.append(
                    f"Capture length differs from the standardized input by {length_delta_seconds:.6f} seconds"
                )
            else:
                warnings.append(
                    f"Capture length differs from the standardized input by {length_delta_seconds:.6f} seconds; "
                    "alignment must be resolved before training"
                )
    status = "failed" if failures else "warning" if warnings else "pass"
    return status, {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "length_delta_frames": length_delta_frames,
        "length_delta_seconds": length_delta_seconds,
    }


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32_768.0
    if sample_width == 3:
        bytes_3 = np.frombuffer(raw, dtype=np.uint8)
        if bytes_3.size % 3:
            raise DomainError("24-bit WAV data is truncated")
        values = bytes_3.reshape(-1, 3).astype(np.int32)
        integers = values[:, 0] | (values[:, 1] << 8) | (values[:, 2] << 16)
        integers = np.where(integers & 0x800000, integers | ~0xFFFFFF, integers)
        return integers.astype(np.float32) / 8_388_608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    raise DomainError("Unsupported PCM sample width")
