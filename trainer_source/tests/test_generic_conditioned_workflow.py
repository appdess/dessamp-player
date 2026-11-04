import argparse
import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CANDIDATES = _load(
    "materialize_conditioned_candidate_config_test",
    "materialize_conditioned_candidate_config.py",
)
IMPORTER = _load(
    "import_conditioned_suggestions_test",
    "import_conditioned_suggestions.py",
)
INITIALIZER = _load(
    "init_conditioned_project_test",
    "init_conditioned_project.py",
)
PAIR = _load(
    "validate_capture_pair_test",
    "validate_capture_pair.py",
)
RESERVATION = _load(
    "create_evaluation_reservation_test",
    "create_evaluation_reservation.py",
)


def _project():
    controls = [
        {
            "id": "gain",
            "label": "Gain",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.5,
            "unit": "%",
            "taper": "linear",
            "display_minimum": 0.0,
            "display_maximum": 100.0,
        },
        {
            "id": "bright",
            "label": "Bright",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 1.0,
            "unit": "",
            "taper": "switch",
            "display_minimum": None,
            "display_maximum": None,
        },
    ]
    return {
        "id": "project-1",
        "name": "Lead channel",
        "controls": controls,
        "candidate_points": [
            {"gain": 0.0, "bright": 0.0},
            {"gain": 0.5, "bright": 0.0},
            {"gain": 0.5, "bright": 1.0},
            {"gain": 1.0, "bright": 1.0},
        ],
    }


def test_candidate_materialization_excludes_target_paths(tmp_path):
    reservation_path = tmp_path / "reservation.json"
    reservation = {
        "control_order": ["gain", "bright"],
        "target_audio_available_to_acquisition": False,
        "references": [
            {"g_vector": [0.25, 1.0], "path": "/secret/audit.wav", "sha256": "a" * 64}
        ],
    }
    reservation_path.write_text(json.dumps(reservation))
    config = CANDIDATES.build_candidate_config(
        project=_project(),
        captures=[
            {
                "quality_status": "pass",
                "settings": {"gain": 0.0, "bright": 0.0},
            }
        ],
        reservation=reservation,
        reservation_path=reservation_path,
    )
    assert config["reserved_evaluation_vectors"] == [[0.25, 1.0]]
    assert "/secret/audit.wav" not in json.dumps(config)
    assert config["reservation"]["target_audio_paths_included"] is False


def test_candidate_materialization_rejects_leaked_audit_vector(tmp_path):
    reservation_path = tmp_path / "reservation.json"
    reservation = {
        "target_audio_available_to_acquisition": False,
        "references": [{"g_vector": [0.25, 1.0]}],
    }
    reservation_path.write_text(json.dumps(reservation))
    with pytest.raises(ValueError, match="already captured"):
        CANDIDATES.build_candidate_config(
            project=_project(),
            captures=[
                {
                    "quality_status": "pass",
                    "settings": {"gain": 0.25, "bright": 1.0},
                }
            ],
            reservation=reservation,
            reservation_path=reservation_path,
        )


def test_candidate_materialization_rejects_acquirable_audit_vector(tmp_path):
    reservation_path = tmp_path / "reservation.json"
    reservation = {
        "target_audio_available_to_acquisition": False,
        "references": [{"g_vector": [0.5, 1.0]}],
    }
    reservation_path.write_text(json.dumps(reservation))
    with pytest.raises(ValueError, match="acquirable candidate"):
        CANDIDATES.build_candidate_config(
            project=_project(),
            captures=[],
            reservation=reservation,
            reservation_path=reservation_path,
        )


def test_disagreement_batch_enforces_spatial_separation():
    controls = [
        IMPORTER.ControlDefinition.from_mapping(control)
        for control in _project()["controls"]
    ]
    ranking = [
        {"index": 1, "g_vector": [0.5, 0.0], "disagreement_score": 3.0},
        {"index": 2, "g_vector": [0.5, 1.0], "disagreement_score": 2.0},
        {"index": 3, "g_vector": [1.0, 0.0], "disagreement_score": 1.0},
    ]
    selected = IMPORTER.select_batch(
        ranking,
        controls=controls,
        occupied=[[0.0, 0.0]],
        maximum=2,
        minimum_distance=0.6,
    )
    assert [point["index"] for point in selected] == [2, 3]


def test_project_initializer_rejects_unmeasured_physical_calibration(tmp_path):
    config = {
        "name": "Physical proof",
        "amp_name": "Owner amp",
        "controls": _project()["controls"],
        "candidate_points": _project()["candidate_points"],
        "metadata": {
            "physical_amplifier": True,
            "control_order": ["gain", "bright"],
            "base_settings": {"gain": 0.5, "bright": 0.0},
            "frozen_controls": {"channel": "lead"},
            "training_delay_samples": 21,
            "calibration": {"status": "REPLACE BEFORE USE"},
        },
    }
    path = tmp_path / "project.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="calibration placeholder"):
        INITIALIZER.load_project_config(path)

    config["metadata"]["calibration"] = {
        "status": "measured",
        "input_level_dbu": -0.54,
        "uniform_safety_trim_db": -3.0,
    }
    path.write_text(json.dumps(config))
    assert INITIALIZER.load_project_config(path)["metadata"]["training_delay_samples"] == 21

    config["metadata"]["calibration"].pop("input_level_dbu")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="finite input_level_dbu"):
        INITIALIZER.load_project_config(path)


def test_initializer_rejects_silent_input_before_creating_project(tmp_path):
    template = json.loads((ROOT.parent / "docs/examples/physical-lead-two-control-project.json").read_text())
    template["metadata"]["training_delay_samples"] = 0
    template["metadata"]["calibration"] = {"status": "measured", "input_level_dbu": 0.0}
    config = tmp_path / "project.json"
    config.write_text(json.dumps(template))
    dry = tmp_path / "silent.wav"
    with wave.open(str(dry), "wb") as output:
        output.setparams((1, 3, 48000, 0, "NONE", "not compressed"))
        output.writeframes(bytes(3 * 48000))
    data = tmp_path / "data"
    with pytest.raises(ValueError, match="input is silent"):
        INITIALIZER.create_project(argparse.Namespace(project_config=config, input=dry, data_dir=data))
    assert not data.exists()


def test_three_control_lead_templates_are_reviewed_and_audit_disjoint():
    project = json.loads(
        (ROOT.parent / "docs/examples/physical-lead-three-control-project.json").read_text()
    )
    reservation = json.loads(
        (
            ROOT.parent
            / "docs/examples/evaluation-reservation-spec-lead-three-control.json"
        ).read_text()
    )
    order = ["lead_gain", "bright", "bottom"]
    assert [control["id"] for control in project["controls"]] == order
    assert reservation["control_order"] == order
    candidates = {
        tuple(float(point[control_id]) for control_id in order)
        for point in project["candidate_points"]
    }
    hidden = {
        tuple(float(value) for value in reference["g_vector"])
        for reference in reservation["references"]
    }
    assert len(candidates) == 24
    assert len(hidden) == 8
    assert candidates.isdisjoint(hidden)
    assert {(point[1], point[2]) for point in candidates} == {
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    }
    assert "bright" not in project["metadata"]["frozen_controls"]
    assert "bottom" not in project["metadata"]["frozen_controls"]


def test_capture_pair_metric_distinguishes_repeat_from_axis_change():
    reference = np.array([1.0, -1.0], dtype=np.float64)
    assert PAIR.error_to_signal(reference, reference.copy()) == 0.0
    assert PAIR.error_to_signal(reference, reference * 0.5) == 0.25


def _write_pcm24(path: Path):
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(3)
        destination.setframerate(48_000)
        destination.writeframes(b"\x64\x00\x00" * 256)


def test_evaluation_reservation_binds_input_and_hidden_reference_hashes(tmp_path):
    input_path = tmp_path / "unseen.wav"
    target_path = tmp_path / "target.wav"
    _write_pcm24(input_path)
    _write_pcm24(target_path)
    reservation = RESERVATION.build_reservation(
        {
            "control_order": ["gain", "bright"],
            "evaluation_input": str(input_path),
            "references": [{"g_vector": [0.5, 1.0], "path": str(target_path)}],
        }
    )
    assert len(reservation["evaluation_input"]["sha256"]) == 64
    assert len(reservation["references"][0]["sha256"]) == 64
    assert reservation["target_audio_available_to_acquisition"] is False


def test_evaluation_reservation_preserves_scope_and_setting_id(tmp_path):
    input_path = tmp_path / "input.wav"
    reference_path = tmp_path / "reference.wav"
    _write_pcm24(input_path)
    _write_pcm24(reference_path)
    reservation = RESERVATION.build_reservation(
        {
            "control_order": ["gain"],
            "evaluation_input": str(input_path),
            "evaluation_scope": {
                "role": "internal-validation-on-unseen-di-diagnostic",
                "release_gate": False,
            },
            "references": [
                {
                    "setting_id": "GAIN-05-DIAGNOSTIC",
                    "g_vector": [0.5],
                    "path": str(reference_path),
                }
            ],
        }
    )
    assert reservation["evaluation_scope"]["role"] == (
        "internal-validation-on-unseen-di-diagnostic"
    )
    assert reservation["evaluation_scope"]["release_gate"] is False
    assert reservation["references"][0]["setting_id"] == "GAIN-05-DIAGNOSTIC"
