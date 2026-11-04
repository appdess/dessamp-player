from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .domain import DomainError
from .wav_io import inspect_wav, load_mono_float32, write_pcm24_mono


class AudioCaptureService:
    """PortAudio-backed direct reamp capture.

    The service is kept behind this small boundary so a future CoreAudio,
    embedded Linux, or dedicated-hardware implementation can replace it without
    changing projects, recommendations, captures, or player manifests.
    """

    def list_devices(self) -> list[dict[str, Any]]:
        sounddevice = _sounddevice()
        try:
            raw_devices = sounddevice.query_devices()
            default_input, default_output = sounddevice.default.device
        except Exception as error:
            raise DomainError(f"Unable to enumerate audio devices: {error}") from error
        devices: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_devices):
            max_input = int(raw["max_input_channels"])
            max_output = int(raw["max_output_channels"])
            if max_input == 0 and max_output == 0:
                continue
            devices.append(
                {
                    "id": index,
                    "name": str(raw["name"]),
                    "hostapi": int(raw["hostapi"]),
                    "max_input_channels": max_input,
                    "max_output_channels": max_output,
                    "default_sample_rate": float(raw["default_samplerate"]),
                    "default_input": index == default_input,
                    "default_output": index == default_output,
                }
            )
        return devices

    def capture_duplex(
        self,
        *,
        input_signal_path: Path,
        output_path: Path,
        input_device: int,
        output_device: int,
    ) -> dict[str, Any]:
        sounddevice = _sounddevice()
        signal, sample_rate = load_mono_float32(input_signal_path)
        try:
            sounddevice.check_input_settings(device=input_device, channels=1, samplerate=sample_rate)
            sounddevice.check_output_settings(device=output_device, channels=1, samplerate=sample_rate)
            recording = sounddevice.playrec(
                signal.reshape(-1, 1),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=(input_device, output_device),
                blocking=True,
            )
        except Exception as error:
            raise DomainError(f"Direct audio capture failed: {error}") from error
        if recording.shape != (signal.shape[0], 1):
            raise DomainError(f"Audio backend returned an unexpected capture shape {recording.shape}")
        write_pcm24_mono(output_path, np.asarray(recording[:, 0]), sample_rate)
        info = inspect_wav(output_path)
        info["input_device"] = int(input_device)
        info["output_device"] = int(output_device)
        return info


def _sounddevice():
    try:
        import sounddevice
    except ImportError as error:
        raise DomainError(
            "Direct interface capture requires the sounddevice package and a working PortAudio installation"
        ) from error
    return sounddevice
