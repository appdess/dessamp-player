import importlib.util
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "conditioned_vertex_job.py"
SPEC = importlib.util.spec_from_file_location("conditioned_vertex_job", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
DIGEST_IMAGE_URI = (
    "europe-west3-docker.pkg.dev/dessamp-training-test/"
    "nam-training/trainer@sha256:" + "a" * 64
)


class CostGuardTests(unittest.TestCase):
    def test_four_parallel_t4_workers_fit_under_explicit_ceiling(self):
        result = MODULE.bounded_cost(
            hourly_rate_eur=Decimal("1.0265"),
            timeout_minutes=120,
            worker_count=4,
            overhead_reserve_eur=Decimal("2.00"),
            authorized_max_eur=Decimal("20.00"),
        )
        self.assertEqual(result["training_ceiling_eur"], "8.22")
        self.assertEqual(result["total_ceiling_eur"], "10.22")

    def test_rejects_more_than_repository_hard_ceiling(self):
        with self.assertRaisesRegex(MODULE.PreflightError, "hard ceiling"):
            MODULE.bounded_cost(
                hourly_rate_eur=Decimal("1.0265"),
                timeout_minutes=60,
                worker_count=1,
                overhead_reserve_eur=Decimal("2"),
                authorized_max_eur=Decimal("50.01"),
            )

    def test_requires_exact_project_confirmation(self):
        with self.assertRaisesRegex(MODULE.PreflightError, "exactly match"):
            MODULE.require_project_confirmation("dessamp-training-test", "flowcoach-prod")

    def test_five_hour_job_is_allowed_but_longer_job_is_rejected(self):
        result = MODULE.bounded_cost(
            hourly_rate_eur=Decimal("1.0265"),
            timeout_minutes=300,
            worker_count=1,
            overhead_reserve_eur=Decimal("0.50"),
            authorized_max_eur=Decimal("5.64"),
        )
        self.assertEqual(result["total_ceiling_eur"], "5.64")
        with self.assertRaisesRegex(MODULE.PreflightError, "300 minutes"):
            MODULE.bounded_cost(
                hourly_rate_eur=Decimal("1.0265"),
                timeout_minutes=301,
                worker_count=1,
                overhead_reserve_eur=Decimal("0.50"),
                authorized_max_eur=Decimal("50.00"),
            )


class BundlePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "input.wav"
        self.train_path = self.root / "train.wav"
        self.validation_path = self.root / "validation.wav"
        for path in (self.input_path, self.train_path, self.validation_path):
            path.write_bytes(b"RIFF\x04\x00\x00\x00WAVEtest")
        self.data_config = self.root / "data.json"
        self.data_config.write_text(
            json.dumps(
                {
                    "common": {"x_path": str(self.input_path)},
                    "train": {"outputs": [{"y_path": str(self.train_path), "g_vector": [0.2, 0.8]}]},
                    "validation": {
                        "outputs": [{"y_path": str(self.validation_path), "g_vector": [0.5, 0.5]}]
                    },
                }
            ),
            encoding="utf-8",
        )
        self.model_config = self.root / "model.json"
        self.learning_config = self.root / "learning.json"
        self.model_config.write_text("{}", encoding="utf-8")
        self.learning_config.write_text("{}", encoding="utf-8")

    def test_upload_plan_is_hash_bound_and_portable(self):
        uploads = MODULE.plan_uploads(
            run_id="fortin-smoke",
            data_config=self.data_config,
            model_config=self.model_config,
            learning_config=self.learning_config,
        )
        self.assertEqual(len(uploads), 6)
        self.assertEqual(
            {item.object_name for item in uploads},
            {
                "jobs/fortin-smoke/configs/data.json",
                "jobs/fortin-smoke/configs/model.json",
                "jobs/fortin-smoke/configs/learning.json",
                "jobs/fortin-smoke/data/input.wav",
                "jobs/fortin-smoke/data/train.wav",
                "jobs/fortin-smoke/data/validation.wav",
            },
        )
        self.assertTrue(all(len(item.sha256) == 64 for item in uploads))

    def test_job_spec_has_server_side_timeout_and_conditioned_args(self):
        spec = MODULE.build_job_spec(
            project="dessamp-training-test",
            bucket="dessmetal-bucket",
            run_id="fortin-smoke",
            image_uri=DIGEST_IMAGE_URI,
            service_account=(
                "trainer@dessamp-training-test.iam.gserviceaccount.com"
            ),
            model_name="fortin-smoke",
            timeout_minutes=45,
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            epochs=1,
            seed=2,
            batch_size=32,
            ny=2048,
            threshold_esr=0.005,
            input_level=12.0,
            output_level=12.0,
            checkpoint_name_value="member-21.ckpt",
        )
        self.assertEqual(spec["scheduling"]["timeout"], "2700s")
        self.assertEqual(
            spec["serviceAccount"],
            "trainer@dessamp-training-test.iam.gserviceaccount.com",
        )
        args = spec["workerPoolSpecs"][0]["containerSpec"]["args"]
        self.assertIn("--source-prefix", args)
        self.assertIn("jobs/fortin-smoke", args)
        self.assertIn("--seed", args)
        self.assertIn("2", args)
        self.assertIn("--checkpoint-name", args)
        self.assertIn("member-21.ckpt", args)

    def test_cpu_proof_omits_accelerator_fields(self):
        spec = MODULE.build_job_spec(
            project="dessamp-training-test",
            bucket="dessmetal-bucket",
            run_id="fortin-cpu-smoke",
            image_uri=DIGEST_IMAGE_URI,
            service_account=(
                "trainer@dessamp-training-test.iam.gserviceaccount.com"
            ),
            model_name="fortin-cpu-smoke",
            timeout_minutes=45,
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=0,
            epochs=1,
            seed=0,
            batch_size=32,
            ny=2048,
            threshold_esr=0.005,
            input_level=None,
            output_level=None,
            checkpoint_name_value=None,
        )
        machine = spec["workerPoolSpecs"][0]["machineSpec"]
        self.assertEqual(machine, {"machineType": "n1-standard-8"})
        args = spec["workerPoolSpecs"][0]["containerSpec"]["args"]
        self.assertNotIn("--input-level", args)

    def test_upload_plan_hash_binds_member_checkpoints(self):
        checkpoint = self.root / "source.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        uploads = MODULE.plan_uploads(
            run_id="fortin-finetune",
            data_config=self.data_config,
            model_config=self.model_config,
            learning_config=self.learning_config,
            checkpoints=[("member-21", checkpoint)],
        )
        checkpoint_upload = next(
            item for item in uploads if item.object_name.endswith("member-21.ckpt")
        )
        self.assertEqual(
            checkpoint_upload.object_name,
            "jobs/fortin-finetune/checkpoints/member-21.ckpt",
        )
        self.assertEqual(checkpoint_upload.size_bytes, len(b"checkpoint"))

    def test_job_spec_rejects_mutable_or_misbound_image_references(self):
        invalid_images = (
            "europe-west3-docker.pkg.dev/dessamp-training-test/nam/trainer:latest",
            "europe-west3-docker.pkg.dev/dessamp-training-test/nam/trainer:tag@sha256:"
            + "a" * 64,
            "europe-west3-docker.pkg.dev/other-project/nam/trainer@sha256:"
            + "a" * 64,
            "europe-west3-docker.pkg.dev/dessamp-training-test/nam/trainer@sha256:"
            + "A" * 64,
            DIGEST_IMAGE_URI + "trailing",
        )
        for image_uri in invalid_images:
            with self.subTest(image_uri=image_uri):
                with self.assertRaisesRegex(MODULE.PreflightError, "digest-pinned"):
                    MODULE.build_job_spec(
                        project="dessamp-training-test",
                        bucket="dessmetal-bucket",
                        run_id="fortin-smoke",
                        image_uri=image_uri,
                        service_account=(
                            "trainer@dessamp-training-test.iam.gserviceaccount.com"
                        ),
                        model_name="fortin-smoke",
                        timeout_minutes=45,
                        machine_type="n1-standard-8",
                        accelerator_type="NVIDIA_TESLA_T4",
                        accelerator_count=1,
                        epochs=1,
                        seed=2,
                        batch_size=32,
                        ny=2048,
                        threshold_esr=0.005,
                        input_level=None,
                        output_level=None,
                    )

    def test_legacy_mutations_fail_before_reading_arguments_or_running_commands(self):
        for mutation in (MODULE.upload, MODULE.submit):
            with self.subTest(mutation=mutation.__name__):
                with self.assertRaisesRegex(MODULE.PreflightError, "disabled"):
                    mutation(None)


if __name__ == "__main__":
    unittest.main()
