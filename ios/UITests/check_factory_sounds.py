"""Verify the actual named presets written by testModelDefaultRigs on an iPhone simulator."""
import argparse
import math
from pathlib import Path
import struct


def read_sound(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<8sII", data)
    assert magic == b"DESSPRST" and version == 1 and length == len(data) - 16, path
    offset = 16
    strings = []
    for _ in range(4):
        (size,) = struct.unpack_from("<i", data, offset)
        assert 0 <= size <= 4096
        offset += 4
        strings.append(data[offset:offset + size].decode())
        offset += size
    assert strings[:2] == ["###NeuralAmpModeler###", "0.2.0"]
    return Path(strings[2]).name, Path(strings[3]).name, struct.unpack("<33d", data[offset:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app_data", type=Path, help="Path from simctl get_app_container <device> com.dessdynamics.dessamp data")
    parser.add_argument("run", help="Factory QA run ID printed by the XCTest")
    args = parser.parse_args()
    bank = args.app_data / "Library/Application Support/DessAmp/Presets"
    for cab, amp, wav, on, slot in (
        ("SHFLD Mix", "amp-01.dess", "SHFLD-Mix.wav", 0, 2),
        ("DZL", "amp-02.dess", "Desfire.wav", 1, 2),
        ("Green", "amp-03.dess", "Green.wav", 1, 3),
    ):
        path, = bank.glob("*/Factory QA " + args.run + " " + cab + ".desspreset")
        model, ir, p = read_sound(path)
        assert (model, ir, p[11], p[14]) == (amp, wav, on, slot), (cab, model, ir, p[11], p[14])
        if cab == "SHFLD Mix":
            assert math.isclose(2 + p[21] * 6 / 100, 6.1, abs_tol=0.001), p[21]
            assert math.isclose(1 + p[25] * 6 / 100, 3.9, abs_tol=0.001), p[25]
        if cab == "DZL":
            assert math.isclose(p[8], 7.8, abs_tol=0.00001), p[8]  # AU host parameter is float32.
        if cab == "Green":
            assert math.isclose(1 + p[21] * 9 / 100, 7, abs_tol=0.001) and p[29:] == (0, 0, 0, 7), p
        print(cab, model, "pedal", slot, "on" if on else "off", "tone", p[8], "Nocturne trim", p[32])
    edited, = bank.glob("*/Factory QA " + args.run + " Verdant Edited.desspreset")
    model, ir, p = read_sound(edited)
    assert (model, ir, p[11], p[14]) == ("amp-02.dess", "Desfire.wav", 1, 2)
    assert p[8] > 7.85, p[8]
    print("Verdant touch edit:", p[8])


if __name__ == "__main__":
    main()
