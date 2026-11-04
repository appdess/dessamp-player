import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load(
    "materialize_teacher_distillation_data_config_test",
    "materialize_teacher_distillation_data_config.py",
)
RESERVATION = _load(
    "create_teacher_validation_reservation_test",
    "create_evaluation_reservation.py",
)

from dessmetal_capture.store import CaptureStore
from dessmetal_capture.wav_io import inspect_wav


def _write_pcm24(path: Path, value: int) -> None:
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(3)
        destination.setframerate(48_000)
        destination.writeframes(bytes((value, 0, 0)) * 512)


def _fixture(tmp_path: Path):
    training_input = tmp_path / "training-input.wav"
    validation_input = tmp_path / "validation-input.wav"
    _write_pcm24(training_input, 20)
    _write_pcm24(validation_input, 21)
    store = CaptureStore(tmp_path / "trainer-data")
    project = store.create_project(
        name="Teacher All Train",
        amp_name="Teacher Amp",
        description="test",
        controls=[
            {
                "id": "gain",
                "label": "Gain",
                "minimum": 0.0,
                "maximum": 1.0,
                "step": 1.0,
                "unit": "",
                "taper": "linear",
                "display_minimum": 1.0,
                "display_maximum": 2.0,
            }
        ],
        candidate_points=[{"gain": 0.0}, {"gain": 1.0}],
        acquisition_strategy="coverage-v1",
        sample_rate=48_000,
        metadata={
            "physical_amplifier": False,
            "source_kind": "teacher_model_distillation",
            "workflow_validation_only": True,
            "control_order": ["gain"],
            "source": {"platform": "test"},
        },
    )
    project = store.set_input_signal(project["id"], training_input, inspect_wav(training_input))

    references = []
    teacher_hashes = ("a" * 64, "b" * 64)
    for index, gain in enumerate((0.0, 1.0), 1):
        recommendation = store.create_recommendation(project["id"])
        training_output = tmp_path / f"training-output-{index}.wav"
        validation_output = tmp_path / f"validation-output-{index}.wav"
        _write_pcm24(training_output, 30 + index)
        _write_pcm24(validation_output, 40 + index)
        capture_id = f"TEACH-{index}"
        store.add_capture(
            project_id=project["id"],
            recommendation_id=recommendation["id"],
            source="teacher_model_distillation",
            wav_path=training_output,
            wav_info=inspect_wav(training_output),
            quality_status="pass",
            validation={
                "status": "passed",
                "source_kind": "teacher_model_distillation",
                "role": "train",
                "shared_input_sha256": project["input_signal_info"]["sha256"],
                "teacher_model": {"sha256": teacher_hashes[index - 1]},
            },
            notes="train teacher",
            capture_id=capture_id,
        )
        references.append(
            {
                "setting_id": capture_id,
                "g_vector": [gain],
                "path": str(validation_output),
                "teacher_sha256": teacher_hashes[index - 1],
            }
        )
    reservation = RESERVATION.build_reservation(
        {
            "control_order": ["gain"],
            "evaluation_input": str(validation_input),
            "evaluation_scope": {"role": "internal-validation-on-disjoint-di"},
            "provenance": {
                "source_kind": "teacher_model_distillation",
                "project_id": project["id"],
            },
            "references": references,
        }
    )
    reservation_path = tmp_path / "validation-reservation.json"
    reservation_path.write_text(json.dumps(reservation))
    return store, project, reservation, reservation_path


def test_builds_all_train_and_same_vector_disjoint_di_validation(tmp_path):
    store, project, reservation, reservation_path = _fixture(tmp_path)
    config = MODULE.build_teacher_data_config(
        project=store.get_project(project["id"]),
        captures=store.list_captures(project["id"]),
        validation_reservation=reservation,
        validation_reservation_path=reservation_path,
    )
    assert len(config["train"]["outputs"]) == 2
    assert len(config["validation"]["outputs"]) == 2
    assert config["train"]["outputs"][0]["g_vector"] == [0.0]
    assert config["validation"]["outputs"][0]["g_vector"] == [0.0]
    assert config["train"]["x_path"] != config["validation"]["x_path"]
    assert config["metadata"]["validation_strategy"] == "separate_disjoint_di_reservation"


def test_rejects_validation_reference_from_a_different_teacher(tmp_path):
    store, project, reservation, reservation_path = _fixture(tmp_path)
    reservation["references"][0]["teacher_sha256"] = "c" * 64
    reservation_path.write_text(json.dumps(reservation))
    with pytest.raises(ValueError, match="matching teacher"):
        MODULE.build_teacher_data_config(
            project=store.get_project(project["id"]),
            captures=store.list_captures(project["id"]),
            validation_reservation=reservation,
            validation_reservation_path=reservation_path,
        )


def test_rejects_training_input_reused_as_validation_input(tmp_path):
    store, project, reservation, reservation_path = _fixture(tmp_path)
    reservation["evaluation_input"] = {
        "path": project["input_signal_path"],
        "sha256": project["input_signal_info"]["sha256"],
        "frames": project["input_signal_info"]["frames"],
        "sample_rate": 48_000,
    }
    reservation_path.write_text(json.dumps(reservation))
    with pytest.raises(ValueError, match="byte-distinct"):
        MODULE.build_teacher_data_config(
            project=store.get_project(project["id"]),
            captures=store.list_captures(project["id"]),
            validation_reservation=reservation,
            validation_reservation_path=reservation_path,
        )
