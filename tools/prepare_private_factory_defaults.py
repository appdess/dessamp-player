#!/usr/bin/env python3
"""Materialize an owner-selected FIRE preset plus unchanged other amp defaults.

Run with PYTHONPATH=capture_companion/src. Output must be outside the repository.
This creates private build inputs, never changes a model or grants distribution.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

from dessmetal_capture.dess_package import read_dess_package


def decode_preset(path):
    data = path.read_bytes()
    if len(data) > 4 * 1024 * 1024 or data[:8] != b"DESSPRST":
        raise ValueError("Invalid preset envelope")
    version, size = struct.unpack_from("<II", data, 8)
    if version != 1 or size != len(data) - 16:
        raise ValueError("Invalid preset size/version")
    position, strings = 16, []
    for _ in range(4):
        size = struct.unpack_from("<i", data, position)[0]
        position += 4
        if not 0 <= size <= 4096 or position + size > len(data):
            raise ValueError("Invalid preset string")
        strings.append(data[position:position + size].decode("utf-8"))
        position += size
    if strings[:2] != ["###NeuralAmpModeler###", "0.2.0"] or len(data) - position != 29 * 8:
        raise ValueError("Expected the current 29-parameter preset")
    values = list(struct.unpack_from("<29d", data, position))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite preset parameter")
    return strings[2], Path(strings[3]), values


def prepare(preset, packages, output):
    output = output.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]) or output.exists():
        raise ValueError("Output must be a new directory outside the repository")
    original_model, ir, fire_values = decode_preset(preset)
    original = read_dess_package(original_model).manifest
    manifests = [read_dess_package(path).manifest for path in packages]
    if len(manifests) != 4 or original["artifact_id"] != manifests[0]["artifact_id"]:
        raise ValueError("Preset must belong to the first (FIRE) catalogue model")
    if original["model"]["sha256"] != manifests[0]["model"]["sha256"]:
        raise ValueError("FIRE preset model weights do not match the build")
    if [c["id"] for c in manifests[0]["controls"]] != ["lead_gain", "bass", "middle", "treble", "presence"]:
        raise ValueError("Unexpected FIRE model-vector order")
    if not ir.is_file() or ir.is_symlink():
        raise ValueError("The saved preset cabinet must be a regular file")
    bank = []
    for index, manifest in enumerate(manifests):
        # Existing fresh-instance host defaults; manifest controls stay exact.
        values = [0., 5., -80., 5., 5., 5., 0., 0., 5., 1., 0., 0., 5., 1., 0., 0., 12., 0., 0., 1., 1.] + [50.] * 8
        controls = list(manifest["controls"])
        if "morph" in manifest:
            controls.append(manifest["morph"]["control"])
        for slot, control in enumerate(controls):
            values[21 + slot] = 100 * (control["default_value"] - control["minimum"]) / (control["maximum"] - control["minimum"])
        if index == 0:
            values = fire_values
        bank.append({"artifact_id": manifest["artifact_id"], "model_sha256": manifest["model"]["sha256"], "parameters": values})
    document = {"schema_version": 1, "ir_sha256": hashlib.sha256(ir.read_bytes()).hexdigest(), "amplifiers": bank}
    output.mkdir(parents=True)
    shutil.copyfile(ir, output / "Desfire.wav")
    (output / "FactoryDefaults.json").write_text(json.dumps(document, indent=2) + "\n")
    evidence = {"preset": str(preset), "preset_sha256": hashlib.sha256(preset.read_bytes()).hexdigest(),
                "original_ir": str(ir), "ir_sha256": document["ir_sha256"],
                "package_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in packages],
                "redistribution_authorized": False, "model_weights_changed": False}
    (output / "provenance-private.json").write_text(json.dumps(evidence, indent=2) + "\n")
    shutil.copyfile(preset, output / "source-bh4.desspreset")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--package", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.preset, args.package, args.output), indent=2))
