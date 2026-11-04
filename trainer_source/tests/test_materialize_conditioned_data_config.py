import importlib.util
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "materialize_conditioned_data_config.py"
SPEC = importlib.util.spec_from_file_location("materialize_conditioned_data_config_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MaterializeDataConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "input.wav"
        self.input_path.write_bytes(b"input")
        self.validation_path = self.root / MODULE.DEFAULT_VALIDATION_BASENAME
        self.validation_path.write_bytes(b"validation")
        self.training_path = self.root / "active-round-01__train-pcm24.wav"
        self.training_path.write_bytes(b"training")
        self.captures = [
            {
                "id": "training-capture",
                "created_at": "2026-01-02T00:00:00+00:00",
                "quality_status": "pass",
                "settings": {"gain": 0.7, "presence": 0.4},
                "wav_path": str(self.training_path),
                "wav_sha256": hashlib.sha256(b"training").hexdigest(),
            },
            {
                "id": "validation-capture",
                "created_at": "2026-01-01T00:00:00+00:00",
                "quality_status": "pass",
                "settings": {"gain": 0.5, "presence": 0.3},
                "wav_path": str(self.validation_path),
                "wav_sha256": hashlib.sha256(b"validation").hexdigest(),
            },
        ]

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_fixed_validation_capture_is_separated_from_training(self):
        config = MODULE.build_data_config(
            input_path=self.input_path,
            control_order=["gain", "presence"],
            captures=self.captures,
        )
        self.assertEqual(config["train"]["outputs"][0]["g_vector"], [0.7, 0.4])
        self.assertEqual(config["validation"]["outputs"][0]["g_vector"], [0.5, 0.3])
        self.assertEqual(config["common"]["delay"], 0)
        self.assertEqual(config["metadata"]["control_order"], ["gain", "presence"])

    def test_generic_project_selects_validation_by_stable_capture_id(self):
        config = MODULE.build_data_config(
            input_path=self.input_path,
            control_order=["gain", "presence"],
            captures=self.captures,
            validation_capture_id="validation-capture",
            validation_basename=None,
            metadata={"project_id": "amp-project", "physical_amplifier": True},
        )
        self.assertEqual(config["validation"]["outputs"][0]["g_vector"], [0.5, 0.3])
        self.assertEqual(config["metadata"]["project_id"], "amp-project")
        self.assertTrue(config["metadata"]["physical_amplifier"])
        self.assertEqual(config["metadata"]["validation_capture_id"], "validation-capture")

    def test_measured_input_calibration_is_preserved(self):
        calibration = {"status": "measured", "input_level_dbu": -0.54}
        config = MODULE.build_data_config(
            input_path=self.input_path,
            control_order=["gain", "presence"],
            captures=self.captures,
            validation_capture_id="validation-capture",
            validation_basename=None,
            metadata={
                "physical_amplifier": True,
                "calibration": calibration,
            },
        )
        self.assertEqual(config["metadata"]["calibration"], calibration)

    def test_teacher_distillation_provenance_is_preserved(self):
        config = MODULE.build_data_config(
            input_path=self.input_path,
            control_order=["gain", "presence"],
            captures=self.captures,
            validation_capture_id="validation-capture",
            validation_basename=None,
            metadata={
                "source": "persistent-capture-project",
                "source_kind": "teacher_model_distillation",
                "source_provenance": {"platform": "TONE3000", "tone_id": 70207},
                "workflow_validation_only": True,
            },
        )
        self.assertEqual(
            config["metadata"]["source_kind"], "teacher_model_distillation"
        )
        self.assertEqual(config["metadata"]["source_provenance"]["tone_id"], 70207)
        self.assertTrue(config["metadata"]["workflow_validation_only"])

    def test_metadata_cannot_change_vector_order(self):
        with self.assertRaisesRegex(ValueError, "metadata control_order"):
            MODULE.build_data_config(
                input_path=self.input_path,
                control_order=["gain", "presence"],
                captures=self.captures,
                metadata={"control_order": ["presence", "gain"]},
            )

    def test_duplicate_accepted_vector_is_rejected(self):
        duplicate_path = self.root / "duplicate.wav"
        duplicate_path.write_bytes(b"duplicate")
        self.captures.append(
            {
                "created_at": "2026-01-03T00:00:00+00:00",
                "quality_status": "pass",
                "settings": {"gain": 0.7, "presence": 0.4},
                "wav_path": str(duplicate_path),
                "wav_sha256": hashlib.sha256(b"duplicate").hexdigest(),
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.build_data_config(
                input_path=self.input_path,
                control_order=["gain", "presence"],
                captures=self.captures,
            )

    def test_failed_capture_is_not_materialized(self):
        failed_path = self.root / "failed.wav"
        failed_path.write_bytes(b"failed")
        self.captures.append(
            {
                "created_at": "2026-01-03T00:00:00+00:00",
                "quality_status": "failed",
                "settings": {"gain": 0.9, "presence": 0.6},
                "wav_path": str(failed_path),
                "wav_sha256": hashlib.sha256(b"failed").hexdigest(),
            }
        )
        config = MODULE.build_data_config(
            input_path=self.input_path,
            control_order=["gain", "presence"],
            captures=self.captures,
        )
        self.assertEqual(len(config["train"]["outputs"]), 1)

    def test_accepted_capture_requires_a_well_formed_sqlite_hash(self):
        self.captures[0].pop("wav_sha256")
        with self.assertRaisesRegex(ValueError, "valid SQLite wav_sha256"):
            MODULE.build_data_config(
                input_path=self.input_path,
                control_order=["gain", "presence"],
                captures=self.captures,
            )

        self.captures[0]["wav_sha256"] = "not-a-sha256"
        with self.assertRaisesRegex(ValueError, "valid SQLite wav_sha256"):
            MODULE.build_data_config(
                input_path=self.input_path,
                control_order=["gain", "presence"],
                captures=self.captures,
            )

    def test_accepted_capture_rejects_wav_changed_after_acceptance(self):
        self.training_path.write_bytes(b"changed-after-acceptance")
        with self.assertRaisesRegex(ValueError, "no longer matches SQLite"):
            MODULE.build_data_config(
                input_path=self.input_path,
                control_order=["gain", "presence"],
                captures=self.captures,
            )

    def test_warning_capture_requires_explicit_admission(self):
        self.captures[0]["quality_status"] = "warning"
        with self.assertRaisesRegex(ValueError, "allow-capture-warnings"):
            MODULE.build_data_config(
                input_path=self.input_path,
                control_order=["gain", "presence"],
                captures=self.captures,
            )
        config = MODULE.build_data_config(
            input_path=self.input_path,
            control_order=["gain", "presence"],
            captures=self.captures,
            allow_capture_warnings=True,
        )
        self.assertEqual(len(config["train"]["outputs"]), 1)


if __name__ == "__main__":
    unittest.main()
