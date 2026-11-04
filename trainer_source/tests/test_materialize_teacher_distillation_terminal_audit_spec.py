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
    "materialize_teacher_distillation_terminal_audit_spec_test",
    "materialize_teacher_distillation_terminal_audit_spec.py",
)
RESERVATION = _load(
    "create_terminal_teacher_audit_reservation_test",
    "create_evaluation_reservation.py",
)


def _write_pcm24(path: Path, value: int, frames: int = 512) -> None:
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(3)
        destination.setframerate(48_000)
        destination.writeframes(bytes((value, 0, 0)) * frames)


def _fixture(tmp_path: Path) -> dict[str, object]:
    paths = {}
    for label, value in (
        ("training_input", 10),
        ("internal_input", 11),
        ("external_input", 12),
        ("internal_target_1", 20),
        ("internal_target_2", 21),
        ("terminal_target_1", 30),
        ("terminal_target_2", 31),
    ):
        path = tmp_path / f"{label.replace('_', '-')}.wav"
        _write_pcm24(path, value)
        paths[label] = path

    source_spec = {
        "schema_version": 1,
        "control_order": ["gain_index"],
        "target_audio_available_to_acquisition": False,
        "evaluation_scope": {"role": "internal-validation-on-disjoint-di"},
        "evaluation_input": str(paths["internal_input"]),
        "references": [
            {
                "g_vector": [0.0],
                "path": str(paths["internal_target_1"]),
                "setting_id": "TEACH-01",
                "teacher_sha256": "a" * 64,
            },
            {
                "g_vector": [1.0],
                "path": str(paths["internal_target_2"]),
                "setting_id": "TEACH-02",
                "teacher_sha256": "b" * 64,
            },
        ],
        "provenance": {
            "project_id": "teacher-project",
            "source_kind": "teacher_model_distillation",
        },
    }
    source_spec_path = tmp_path / "internal-validation-reservation-spec.json"
    source_spec_path.write_text(json.dumps(source_spec), encoding="utf-8")

    selection_report = {
        "schema_version": 1,
        "selection_opened_hidden_evidence": False,
        "criterion": "minimum best_internal_validation_esr",
        "members": [
            {"label": "101", "model_sha256": "c" * 64},
            {"label": "102", "model_sha256": "d" * 64},
        ],
        "selected_member": "102",
        "selected_model_sha256": "d" * 64,
    }
    selection_report_path = tmp_path / "internal-member-selection.json"
    selection_report_path.write_text(json.dumps(selection_report), encoding="utf-8")

    renderer_path = tmp_path / "render_model"
    renderer_path.write_bytes(b"current native renderer test fixture")
    renderer_path.chmod(renderer_path.stat().st_mode | 0o111)

    return {
        **paths,
        "source_spec": source_spec,
        "source_spec_path": source_spec_path,
        "selection_report_path": selection_report_path,
        "renderer_path": renderer_path,
        "target_paths": [paths["terminal_target_1"], paths["terminal_target_2"]],
    }


def _build(fixture: dict[str, object], **overrides):
    kwargs = {
        "source_spec_path": fixture["source_spec_path"],
        "selection_report_path": fixture["selection_report_path"],
        "training_input_paths": [fixture["training_input"]],
        "evaluation_input_path": fixture["external_input"],
        "target_paths": fixture["target_paths"],
        "renderer_path": fixture["renderer_path"],
        "evaluation_scope_overrides": {
            "role": "external-known-settings-on-unseen-di",
            "release_gate": True,
        },
        "provenance_overrides": {
            "purpose": "terminal-teacher-distillation-audit"
        },
    }
    kwargs.update(overrides)
    return MODULE.build_terminal_audit_spec(**kwargs)


def test_builds_hash_bound_spec_consumable_by_reservation_builder(tmp_path):
    fixture = _fixture(tmp_path)
    spec = _build(fixture)

    source_spec = fixture["source_spec"]
    assert spec["control_order"] == source_spec["control_order"]
    assert [item["g_vector"] for item in spec["references"]] == [
        item["g_vector"] for item in source_spec["references"]
    ]
    assert [item["setting_id"] for item in spec["references"]] == [
        item["setting_id"] for item in source_spec["references"]
    ]
    assert [item["teacher_sha256"] for item in spec["references"]] == [
        item["teacher_sha256"] for item in source_spec["references"]
    ]
    assert spec["target_audio_available_to_acquisition"] is False
    assert spec["provenance"]["source_kind"] == "teacher_model_distillation"
    assert (
        spec["provenance"]["target_generation_available_to_acquisition"] is False
    )
    assert spec["provenance"]["member_selection_report"]["sha256"] == (
        MODULE.sha256_file(fixture["selection_report_path"])
    )
    assert spec["provenance"]["renderer"]["sha256"] == MODULE.sha256_file(
        fixture["renderer_path"]
    )

    reservation = RESERVATION.build_reservation(spec)
    assert reservation["target_audio_available_to_acquisition"] is False
    assert [item["setting_id"] for item in reservation["references"]] == [
        "TEACH-01",
        "TEACH-02",
    ]
    assert reservation["evaluation_scope"]["release_gate"] is True


def test_rejects_missing_selection_report(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="selected-member report"):
        _build(
            fixture,
            selection_report_path=tmp_path / "missing-member-selection.json",
        )


def test_rejects_external_di_that_overlaps_training_input(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="overlaps train/internal input"):
        _build(fixture, evaluation_input_path=fixture["training_input"])


def test_rejects_terminal_target_count_mismatch(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="target count"):
        _build(fixture, target_paths=[fixture["terminal_target_1"]])


def test_refuses_to_overwrite_existing_spec(tmp_path):
    fixture = _fixture(tmp_path)
    spec = _build(fixture)
    output = tmp_path / "terminal-evaluation-spec.json"
    output.write_text("owner evidence\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to replace"):
        MODULE.write_terminal_audit_spec(spec, output)
    assert output.read_text(encoding="utf-8") == "owner evidence\n"
