import hashlib
import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "conditioned_a2_contract.py"
SPEC = importlib.util.spec_from_file_location("conditioned_a2_contract_test", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ConditionedA2ContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_dir = self.root / "configs"
        self.data_dir = self.root / "data"
        self.config_dir.mkdir()
        self.data_dir.mkdir()
        self.data_config = self.config_dir / "data.json"
        self.capture = self.data_dir / "capture.wav"
        self.data_config.write_bytes(b'{"schema_version":1}')
        self.capture.write_bytes(b"RIFF-capture")
        self.manifest = {
            "schema_version": 1,
            "project": "dessamp-training",
            "bucket": "dessamp-training-data",
            "run_id": "physical-r01",
            "control_order": ["gain", "bright"],
            "expected_train_captures": 40,
            "objects": [
                self._object("jobs/physical-r01/configs/data.json", self.data_config),
                self._object("jobs/physical-r01/data/capture.wav", self.capture),
            ],
            "total_bytes": self.data_config.stat().st_size + self.capture.stat().st_size,
        }

    @staticmethod
    def _object(name, path):
        return {
            "name": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    def test_manifest_identity_and_downloaded_bytes_are_verified(self):
        MODULE.validate_input_manifest(
            self.manifest,
            project="dessamp-training",
            bucket="dessamp-training-data",
            run_id="physical-r01",
            control_order=("gain", "bright"),
            expected_train_captures=40,
        )
        MODULE.verify_downloaded_input_bundle(
            self.manifest,
            source_prefix="jobs/physical-r01",
            config_dir=self.config_dir,
            data_dir=self.data_dir,
        )

    def test_downloaded_byte_change_is_rejected_before_resume(self):
        self.capture.write_bytes(b"RIFF-changed")
        with self.assertRaisesRegex(ValueError, "differs from manifest"):
            MODULE.verify_downloaded_input_bundle(
                self.manifest,
                source_prefix="jobs/physical-r01",
                config_dir=self.config_dir,
                data_dir=self.data_dir,
            )

    def test_manifest_cannot_change_control_order_or_count(self):
        with self.assertRaisesRegex(ValueError, "control_order"):
            MODULE.validate_input_manifest(
                self.manifest,
                project="dessamp-training",
                bucket="dessamp-training-data",
                run_id="physical-r01",
                control_order=("bright", "gain"),
                expected_train_captures=40,
            )
        with self.assertRaisesRegex(ValueError, "expected_train_captures"):
            MODULE.validate_input_manifest(
                self.manifest,
                project="dessamp-training",
                bucket="dessamp-training-data",
                run_id="physical-r01",
                control_order=("gain", "bright"),
                expected_train_captures=48,
            )

    def test_manifest_rejects_unlisted_runtime_paths_and_oversized_objects(self):
        for name in (
            "jobs/physical-r01/configs/extra.json",
            "jobs/physical-r01/data/../../private-key.json",
            "jobs/physical-r01/data/not-a-wave.txt",
            "jobs/physical-r01/results/output.nam",
        ):
            with self.subTest(name=name):
                manifest = copy.deepcopy(self.manifest)
                manifest["objects"][1]["name"] = name
                with self.assertRaisesRegex(ValueError, "runtime allowlist"):
                    MODULE.validate_input_manifest(
                        manifest,
                        project="dessamp-training",
                        bucket="dessamp-training-data",
                        run_id="physical-r01",
                        control_order=("gain", "bright"),
                        expected_train_captures=40,
                    )

        manifest = copy.deepcopy(self.manifest)
        manifest["objects"][1]["size_bytes"] = MODULE.MAX_INPUT_OBJECT_BYTES + 1
        manifest["total_bytes"] = (
            manifest["objects"][0]["size_bytes"] + manifest["objects"][1]["size_bytes"]
        )
        with self.assertRaisesRegex(ValueError, "malformed"):
            MODULE.validate_input_manifest(
                manifest,
                project="dessamp-training",
                bucket="dessamp-training-data",
                run_id="physical-r01",
                control_order=("gain", "bright"),
                expected_train_captures=40,
            )

    def test_conditioned_image_copies_the_shared_contract_module(self):
        dockerfile = (SCRIPT_PATH.parent / "Dockerfile.conditioned-a2").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "COPY cloud/conditioned_a2_contract.py "
            "/app/trainer_source/cloud/conditioned_a2_contract.py",
            dockerfile,
        )

    def test_efficiency_recipe_is_hash_bound_and_default_contract_stays_legacy_compatible(self):
        common = {
            "project": "dessamp-training",
            "bucket": "dessamp-training-data",
            "run_id": "physical-r01",
            "result_prefix": "jobs/physical-r01/results/full-seed-110",
            "input_manifest_sha256": "c" * 64,
            "image_uri": "registry.example/a2@sha256:" + "a" * 64,
            "child": "full",
            "seed": 110,
            "control_order": ("gain", "bass"),
            "epochs": 60,
            "batch_size": 16,
            "ny": 8192,
            "validation_ny": 262144,
            "num_workers": 4,
            "expected_train_captures": 40,
            "machine_type": "runpod-pod",
            "accelerator_type": "NVIDIA A40",
            "accelerator_count": 1,
        }
        legacy = MODULE.build_member_contract(**common)
        self.assertNotIn("chunk_selection", legacy["training_recipe"])
        self.assertNotIn("intent", legacy)

        efficient = MODULE.build_member_contract(
            **common,
            train_chunks_per_output=556,
            artifact_purpose="workflow_validation_only",
        )
        self.assertEqual(
            efficient["training_recipe"]["chunk_selection"],
            {"mode": "uniform-stratified-v1", "chunks_per_output": 556},
        )
        self.assertEqual(
            efficient["intent"],
            {"artifact_purpose": "workflow_validation_only"},
        )
        self.assertNotEqual(
            MODULE.canonical_json_sha256(legacy),
            MODULE.canonical_json_sha256(efficient),
        )


if __name__ == "__main__":
    unittest.main()
