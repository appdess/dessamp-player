#!/usr/bin/env python3
"""Recreate Version C from the original masters and verify the committed float samples."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy import signal
import soundfile as sf

from dessamp_factory_defaults import BANK, CABINET_HASHES

RATE = 48_000
TAPS = 3072
MASTERS = (
    ('DZL.wav', 'Desfire.wav', 140, '315f0ae92c4092c304c9b23d2b8a4e8cbe87170050463eb742551628f82a34ce'),
    ('ENGL-SHFLD.wav', 'SHFLD-Mix.wav', 250, '12686f94fddc7f030dcd74e3902265162d2b4c545aeb316cb649a37ce5f3892e'),
    ('Gleichrichter.wav', 'Gleichrichter.wav', 170, 'c8daedb530279d898777e88a50975048067a4473105da841f3acd834e40fc28b'),
    ('green.wav', 'Green.wav', 175, 'b7289ff17cf955484baa87e16b8f4e97992bf3253acdfaeed09709da4c74a375'),
)


def version_c(samples, center):
    peak = np.argmax(np.abs(samples))
    time_ms = (np.arange(TAPS) - peak) * 1000 / RATE
    fade = 0.5 + 0.5 * np.cos(np.pi * np.clip((time_ms - 4) / 14, 0, 1))
    padded = np.pad(samples, (0, TAPS - len(samples))) * fade
    cuts = np.vstack((signal.butter(2, 80, "highpass", fs=RATE, output="sos"),
                      signal.butter(2, 9000, "lowpass", fs=RATE, output="sos")))
    omega = 2 * np.pi * center / RATE
    gain = 10 ** (-3.5 / 40)
    alpha = np.sin(omega) / (2 * 1.1)
    b = np.array([1 + alpha * gain, -2 * np.cos(omega), 1 - alpha * gain])
    a = np.array([1 + alpha / gain, -2 * np.cos(omega), 1 - alpha / gain])
    result = signal.lfilter(b / a[0], a / a[0], signal.sosfilt(cuts, padded))
    result[-64:] *= np.linspace(1, 0, 64)
    return result.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("masters", type=Path)
    args = parser.parse_args()
    for raw_name, name, center, expected in MASTERS:
        raw = args.masters / raw_name
        if hashlib.sha256(raw.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unrecognized master: {raw_name}")
        source, rate = sf.read(raw)
        if rate != RATE or source.shape != (2048,):
            raise ValueError(f"Invalid format: {raw_name}")
        cabinet = BANK / name
        if hashlib.sha256(cabinet.read_bytes()).hexdigest() != CABINET_HASHES[name]:
            raise ValueError(f"Wrong factory WAV: {name}")
        stored, rate = sf.read(cabinet, dtype="float32")
        if rate != RATE or not np.array_equal(stored, version_c(source, center)):
            raise ValueError(f"Version C samples differ: {name}")
        print(f"{name}: verified all {TAPS} samples")


if __name__ == "__main__":
    main()
