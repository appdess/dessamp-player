import argparse
import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "import_teacher_distillation_plan.py"
SPEC = importlib.util.spec_from_file_location("import_teacher_distillation_plan_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from dessmetal_capture.store import CaptureStore
from dessmetal_capture.wav_io import inspect_wav


def _write_pcm24(path: Path, value: int) -> None:
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(3)
        destination.setframerate(48_000)
        destination.writeframes(bytes((value, 0, 0)) * 512)


def _write_snapshot(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "0.7.0",
                "architecture": "Linear",
                "sample_rate": 48_000,
                "config": {},
                "weights": [],
            }
        )
    )


def _fixture(tmp_path: Path):
    root = tmp_path / "experiment"
    root.mkdir()
    input_path = root / "input.wav"
    _write_pcm24(input_path, 20)
    store = CaptureStore(root / "trainer-data")
    project = store.create_project(
        name="Teacher POC",
        amp_name="Teacher",
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
        },
    )
    project = store.set_input_signal(project["id"], input_path, inspect_wav(input_path))
    entries = []
    for index, (gain, role) in enumerate(((0.0, "train"), (1.0, "validation")), 1):
        model = root / f"teacher-{index}.nam"
        render = root / f"render-{index}.wav"
        _write_snapshot(model)
        _write_pcm24(render, 20 + index)
        entries.append(
            {
                "capture_id": f"TEACH-{index}",
                "gain_label": str(index),
                "gain": gain,
                "role": role,
                "teacher_relative": model.name,
                "teacher_sha256": MODULE.sha256_file(model),
                "render_relative": render.name,
                "render_sha256": MODULE.sha256_file(render),
            }
        )
    plan = {
        "schema_version": 1,
        "project_id": project["id"],
        "source_kind": "teacher_model_distillation",
        "experiment_root": str(root),
        "shared_input_relative": input_path.name,
        "shared_input_sha256": MODULE.sha256_file(input_path),
        "entries": entries,
    }
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    return root, store, project, plan, plan_path


def test_imports_hash_bound_train_and_validation_without_audit(tmp_path):
    root, store, project, _plan, plan_path = _fixture(tmp_path)
    output = root / "result.json"
    result = MODULE.import_plan(
        argparse.Namespace(data_dir=store.data_dir, plan=plan_path, output=output)
    )
    assert result["capture_count"] == 2
    assert result["train_count"] == 1
    assert result["validation_capture_id"] == "TEACH-2"
    assert result["frozen_audit_imported"] is False
    assert len(store.list_captures(project["id"])) == 2
    assert output.is_file()


def test_rejects_frozen_audit_role_from_persistent_training_project(tmp_path):
    _root, _store, project, plan, _plan_path = _fixture(tmp_path)
    plan["entries"][1]["role"] = "frozen_audit"
    with pytest.raises(ValueError, match="frozen audit targets stay separate"):
        MODULE.validate_plan_entries(plan, project=project)


def test_imports_all_train_when_validation_is_a_separate_disjoint_di(tmp_path):
    root, store, project, plan, plan_path = _fixture(tmp_path)
    plan["validation_strategy"] = "separate_disjoint_di_reservation"
    for entry in plan["entries"]:
        entry["role"] = "train"
    plan_path.write_text(json.dumps(plan))
    output = root / "all-train-result.json"
    result = MODULE.import_plan(
        argparse.Namespace(data_dir=store.data_dir, plan=plan_path, output=output)
    )
    assert result["capture_count"] == 2
    assert result["train_count"] == 2
    assert result["validation_capture_id"] is None
    assert result["validation_strategy"] == "separate_disjoint_di_reservation"


def test_separate_reservation_strategy_rejects_an_imported_validation(tmp_path):
    _root, _store, project, plan, _plan_path = _fixture(tmp_path)
    plan["validation_strategy"] = "separate_disjoint_di_reservation"
    with pytest.raises(ValueError, match="every imported teacher capture to be train"):
        MODULE.validate_plan_entries(plan, project=project)
