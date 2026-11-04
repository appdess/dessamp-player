from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "repackage_dess_input_calibration.py"
SPEC = importlib.util.spec_from_file_location("dessamp_repackage_calibration", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from dessmetal_capture.dess_package import create_dess_package, read_dess_package  # noqa: E402


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_package(root: pathlib.Path) -> pathlib.Path:
    model = {
        "version": "0.7.0",
        "architecture": "WaveNet",
        "config": {"global_condition_size": 1, "layers": []},
        "weights": [1.0, 2.0, 3.0],
        "sample_rate": 48000,
        "metadata": {"name": "fixture"},
    }
    model_path = root / "model.nam"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_id": "fixture",
        "name": "DESS FIRE fixture",
        "model": {
            "path": "model.nam",
            "format": "nam",
            "sha256": _sha256(model_path),
            "format_version": "0.7.0",
            "architecture": "WaveNet",
            "sample_rate": 48000,
            "parameter_mode": "global_condition",
            "global_condition_size": 1,
        },
        "controls": [
            {
                "id": "gain",
                "label": "Gain",
                "minimum": 0.0,
                "maximum": 1.0,
                "step": 0.1,
                "unit": "",
                "taper": "linear",
                "default_value": 0.5,
                "model_parameter_index": 0,
            }
        ],
    }
    manifest_path = root / "player-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return pathlib.Path(create_dess_package(manifest_path, root / "source.dess")["path"])


def _loader(root: pathlib.Path) -> pathlib.Path:
    path = root / "loadmodel_manifest"
    path.write_text(
        "#!/bin/sh\necho '{\"status\":\"valid\",\"parameter_count\":1,\"morph\":false}'\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_metadata_only_repackage_preserves_network_and_adds_input_reference(tmp_path):
    source = _source_package(tmp_path)
    output = tmp_path / "calibrated.dess"
    report = tmp_path / "calibration-report.json"
    result = MODULE.repackage_input_calibration(
        source_package=source,
        expected_source_sha256=_sha256(source),
        input_level_dbu=-0.54,
        core_loader=_loader(tmp_path),
        output_package=output,
        report_path=report,
    )

    package = read_dess_package(output, cache_dir=tmp_path / "cache")
    model = json.loads(package.model_path.read_text(encoding="utf-8"))
    evidence = json.loads(report.read_text(encoding="utf-8"))
    assert model["metadata"]["input_level_dbu"] == -0.54
    assert evidence["network_payload_unchanged"] is True
    assert evidence["source_package"]["sha256"] == _sha256(source)
    assert evidence["calibrated_package"]["sha256"] == _sha256(output)
    assert result["report_sha256"] == _sha256(report)


def test_repackage_rejects_conflicting_existing_reference(tmp_path):
    source = _source_package(tmp_path)
    package = read_dess_package(source, cache_dir=tmp_path / "source-cache")
    model = json.loads(package.model_path.read_text(encoding="utf-8"))
    model["metadata"]["input_level_dbu"] = 12.0
    conflict_root = tmp_path / "conflict-root"
    conflict_root.mkdir()
    conflict_model = conflict_root / "model.nam"
    conflict_model.write_text(json.dumps(model), encoding="utf-8")
    manifest = package.manifest
    manifest["model"]["sha256"] = _sha256(conflict_model)
    manifest_path = conflict_root / "player-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    conflict = pathlib.Path(create_dess_package(manifest_path, tmp_path / "conflict.dess")["path"])

    with pytest.raises(RuntimeError, match="conflicting input_level_dbu"):
        MODULE.repackage_input_calibration(
            source_package=conflict,
            expected_source_sha256=_sha256(conflict),
            input_level_dbu=-0.54,
            core_loader=_loader(tmp_path),
            output_package=tmp_path / "never.dess",
            report_path=tmp_path / "never.json",
        )
