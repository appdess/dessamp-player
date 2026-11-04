import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "conditioned_a2_vertex_job.py"
SPEC = importlib.util.spec_from_file_location("conditioned_a2_vertex_job", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fake_wav(label: str) -> bytes:
    payload = label.encode("utf-8")
    return b"RIFF" + len(payload).to_bytes(4, "little") + b"WAVE" + payload


class ConditionedA2VertexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        x = root / "input.wav"
        validation_x = root / "validation-input.wav"
        validation = root / "validation.wav"
        x.write_bytes(fake_wav("input"))
        validation_x.write_bytes(fake_wav("validation-input"))
        validation.write_bytes(fake_wav("validation"))
        outputs = []
        for index in range(40):
            path = root / f"capture-{index:02d}.wav"
            path.write_bytes(fake_wav(str(index)))
            outputs.append({"y_path": str(path), "g_vector": [0.5] * 7})
        self.config = root / "data.json"
        self.config.write_text(
            json.dumps(
                {
                    "metadata": {"control_order": list(MODULE.DEFAULT_CONTROL_ORDER)},
                    "common": {"x_path": str(x)},
                    "train": {"outputs": outputs},
                    "validation": {
                        "x_path": str(validation_x),
                        "outputs": [
                            {"y_path": str(validation), "g_vector": [0.55] * 7}
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_bundle_requires_declared_count_by_7_contract(self):
        uploads = MODULE.plan_a2_uploads(
            run_id="fortin-a2-40c-r01", data_config=self.config
        )
        self.assertEqual(len(uploads), 44)
        self.assertTrue(all(len(item.sha256) == 64 for item in uploads))

        config = json.loads(self.config.read_text())
        config["train"]["outputs"][0]["g_vector"] = [0.5] * 6
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.PreflightError, "7 normalized"):
            MODULE.plan_a2_uploads(run_id="fortin-a2-40c-r01", data_config=self.config)

    def test_submit_requires_an_explicit_current_hourly_rate(self):
        base_args = [
            "submit",
            "--project=dessamp-training",
            "--confirm-project=dessamp-training",
            "--bucket=dessamp-training-data",
            "--run-id=physical-amp-r01",
            "--image-uri=europe-west3-docker.pkg.dev/p/r/a@sha256:" + "a" * 64,
            "--service-account=trainer@dessamp-training.iam.gserviceaccount.com",
            "--child=full",
            "--seed=1",
            "--max-cost-eur=8.00",
        ]
        with self.assertRaises(SystemExit):
            MODULE.build_parser().parse_args(base_args)

        parsed = MODULE.build_parser().parse_args(
            base_args + ["--hourly-rate-eur=1.0265"]
        )
        self.assertEqual(parsed.hourly_rate_eur, MODULE.Decimal("1.0265"))

    def test_bundle_and_job_accept_an_arbitrary_declared_control_order(self):
        config = json.loads(self.config.read_text())
        config["metadata"]["control_order"] = ["gain", "bright"]
        for split in ("train", "validation"):
            for output in config[split]["outputs"]:
                output["g_vector"] = [0.25, 1.0]
        self.config.write_text(json.dumps(config), encoding="utf-8")

        uploads = MODULE.plan_a2_uploads(
            run_id="lead-two-control-r01",
            data_config=self.config,
            control_order="gain,bright",
        )
        self.assertEqual(len(uploads), 44)
        with self.assertRaisesRegex(MODULE.PreflightError, "does not match"):
            MODULE.plan_a2_uploads(
                run_id="lead-two-control-r01",
                data_config=self.config,
                control_order="gain,bottom",
            )

    def test_bundle_accepts_a_declared_48_capture_active_round(self):
        config = json.loads(self.config.read_text())
        root = self.config.parent
        for index in range(40, 48):
            path = root / f"capture-{index:02d}.wav"
            path.write_bytes(fake_wav(str(index)))
            config["train"]["outputs"].append(
                {"y_path": str(path), "g_vector": [0.5] * 7}
            )
        self.config.write_text(json.dumps(config), encoding="utf-8")

        uploads = MODULE.plan_a2_uploads(
            run_id="fortin-a2-48c-r02",
            data_config=self.config,
            expected_train_captures=48,
        )
        self.assertEqual(len(uploads), 52)

        with self.assertRaisesRegex(MODULE.PreflightError, "exactly 47"):
            MODULE.plan_a2_uploads(
                run_id="fortin-a2-48c-r02",
                data_config=self.config,
                expected_train_captures=47,
            )

    def test_bundle_rejects_config_paths_outside_the_authorized_data_root(self):
        outside_directory = tempfile.TemporaryDirectory()
        self.addCleanup(outside_directory.cleanup)
        outside = Path(outside_directory.name) / "outside.wav"
        outside.write_bytes(fake_wav("outside"))
        config = json.loads(self.config.read_text())
        config["train"]["outputs"][0]["y_path"] = str(outside)
        self.config.write_text(json.dumps(config), encoding="utf-8")

        with self.assertRaisesRegex(MODULE.PreflightError, "escapes the authorized data root"):
            MODULE.plan_a2_uploads(
                run_id="fortin-a2-40c-r01",
                data_config=self.config,
            )

    def test_bundle_accepts_an_explicit_bounded_parent_data_root(self):
        config_directory = self.config.parent / "configs"
        config_directory.mkdir()
        moved_config = config_directory / "data.json"
        moved_config.write_bytes(self.config.read_bytes())

        uploads = MODULE.plan_a2_uploads(
            run_id="fortin-a2-40c-r01",
            data_config=moved_config,
            data_root=self.config.parent,
        )
        self.assertEqual(len(uploads), 44)

    def test_bundle_rejects_non_wav_content_and_symbolic_links(self):
        config = json.loads(self.config.read_text())
        invalid = self.config.parent / "invalid.wav"
        invalid.write_bytes(b"not audio")
        config["train"]["outputs"][0]["y_path"] = str(invalid)
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.PreflightError, "WAV container header"):
            MODULE.plan_a2_uploads(
                run_id="fortin-a2-40c-r01",
                data_config=self.config,
            )

        target = self.config.parent / "valid-target.wav"
        target.write_bytes(fake_wav("target"))
        linked = self.config.parent / "linked.wav"
        linked.symlink_to(target)
        config["train"]["outputs"][0]["y_path"] = str(linked)
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.PreflightError, "symbolic link"):
            MODULE.plan_a2_uploads(
                run_id="fortin-a2-40c-r01",
                data_config=self.config,
            )

    def test_spot_job_has_restartable_child_contract(self):
        spec = MODULE.build_job_spec(
            project="dessamp-training-test",
            bucket="dessamp-training-test-data",
            run_id="fortin-a2-40c-r01",
            image_uri=(
                "europe-west3-docker.pkg.dev/dessamp-training-test/"
                "nam-training/conditioned-a2@sha256:" + "a" * 64
            ),
            service_account=(
                "trainer@dessamp-training-test.iam.gserviceaccount.com"
            ),
            child="lite",
            seed=81,
            timeout_minutes=300,
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            epochs=60,
            batch_size=16,
            ny=8192,
            validation_ny=262144,
            num_workers=4,
            expected_train_captures=48,
            scheduling_strategy="spot",
            member_config_sha256="b" * 64,
            control_order=("gain", "bright"),
        )
        self.assertEqual(spec["scheduling"]["strategy"], "SPOT")
        self.assertEqual(spec["scheduling"]["timeout"], "18000s")
        self.assertTrue(spec["scheduling"]["restartJobOnWorkerRestart"])
        args = spec["workerPoolSpecs"][0]["containerSpec"]["args"]
        self.assertIn("--child", args)
        self.assertIn("lite", args)
        self.assertIn("--expected-train-captures", args)
        expected_index = args.index("--expected-train-captures")
        self.assertEqual(args[expected_index + 1], "48")
        control_index = args.index("--control-order")
        self.assertEqual(args[control_index + 1], "gain,bright")
        contract_index = args.index("--member-config-sha256")
        self.assertEqual(args[contract_index + 1], "b" * 64)

    def test_job_rejects_a_mutable_image_tag(self):
        with self.assertRaisesRegex(MODULE.PreflightError, "digest-pinned"):
            MODULE.build_job_spec(
                project="dessamp-training",
                bucket="dessamp-training-data",
                run_id="physical-amp-r01",
                image_uri="europe-west3-docker.pkg.dev/dessamp-training/nam/a2:latest",
                service_account="trainer@dessamp-training.iam.gserviceaccount.com",
                child="full",
                seed=1,
                timeout_minutes=60,
                machine_type="n1-standard-8",
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1,
                epochs=1,
                batch_size=16,
                ny=8192,
                validation_ny=262144,
                num_workers=0,
                expected_train_captures=40,
                scheduling_strategy="spot",
                member_config_sha256="b" * 64,
            )

    def _member_contract(self, *, epochs=60, image_digest="a"):
        return MODULE.build_member_contract(
            project="dessamp-training",
            bucket="dessamp-training-data",
            run_id="physical-amp-r01",
            result_prefix="jobs/physical-amp-r01/results/full-seed-1",
            input_manifest_sha256="c" * 64,
            image_uri=(
                "europe-west3-docker.pkg.dev/dessamp-training/nam/a2@sha256:"
                + image_digest * 64
            ),
            child="full",
            seed=1,
            control_order=("gain", "bright"),
            epochs=epochs,
            batch_size=16,
            ny=8192,
            validation_ny=262144,
            num_workers=4,
            expected_train_captures=40,
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
        )

    def test_member_contract_binds_absolute_target_and_training_recipe(self):
        contract = self._member_contract()
        self.assertEqual(
            contract["training_recipe"]["epoch_contract"],
            {"mode": "absolute_target", "target_epochs": 60},
        )
        self.assertEqual(contract["architecture"]["channels"], 8)
        self.assertEqual(contract["architecture"]["receptive_field"], 6347)
        self.assertEqual(len(MODULE.canonical_json_sha256(contract)), 64)

        changed_epochs = self._member_contract(epochs=72)
        with self.assertRaisesRegex(ValueError, "cannot reuse this checkpoint"):
            MODULE.require_exact_member_contract(contract, changed_epochs)

        changed_image = self._member_contract(image_digest="d")
        with self.assertRaisesRegex(ValueError, "cannot reuse this checkpoint"):
            MODULE.require_exact_member_contract(contract, changed_image)

    def test_member_contract_creation_uses_generation_zero(self):
        completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with mock.patch.object(MODULE, "run_checked", return_value=completed) as runner:
            result = MODULE.register_member_contract(
                project="dessamp-training",
                bucket="dessamp-training-data",
                result_prefix="jobs/physical-amp-r01/results/full-seed-1",
                contract=self._member_contract(),
                resume_existing_member=False,
            )
        self.assertEqual(result["state"], "immutable-contract-created")
        self.assertIn("--if-generation-match=0", runner.call_args.args[0])

    def test_existing_member_requires_explicit_exact_resume(self):
        contract = self._member_contract()
        create_failure = subprocess.CalledProcessError(1, ["gcloud", "storage", "cp"])
        existing = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(contract), stderr=""
        )
        with mock.patch.object(
            MODULE, "run_checked", side_effect=[create_failure, existing]
        ):
            with self.assertRaisesRegex(MODULE.PreflightError, "resume-existing-member"):
                MODULE.register_member_contract(
                    project="dessamp-training",
                    bucket="dessamp-training-data",
                    result_prefix="jobs/physical-amp-r01/results/full-seed-1",
                    contract=contract,
                    resume_existing_member=False,
                )

        with mock.patch.object(
            MODULE, "run_checked", side_effect=[create_failure, existing]
        ):
            result = MODULE.register_member_contract(
                project="dessamp-training",
                bucket="dessamp-training-data",
                result_prefix="jobs/physical-amp-r01/results/full-seed-1",
                contract=contract,
                resume_existing_member=True,
            )
        self.assertEqual(result["state"], "exact-contract-reused")

    def test_changed_existing_member_contract_is_rejected(self):
        create_failure = subprocess.CalledProcessError(1, ["gcloud", "storage", "cp"])
        existing = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(self._member_contract(epochs=72)), stderr=""
        )
        with mock.patch.object(
            MODULE, "run_checked", side_effect=[create_failure, existing]
        ):
            with self.assertRaisesRegex(MODULE.PreflightError, "cannot reuse"):
                MODULE.register_member_contract(
                    project="dessamp-training",
                    bucket="dessamp-training-data",
                    result_prefix="jobs/physical-amp-r01/results/full-seed-1",
                    contract=self._member_contract(),
                    resume_existing_member=True,
                )

    def _writer_lease_existing(self, *, expires_at):
        contract_sha256 = MODULE.canonical_json_sha256(self._member_contract())
        payload = {
            "schema_version": 1,
            "provider": "runpod",
            "member_contract_sha256": contract_sha256,
            "attempt_nonce": "1" * 32,
            "pod_name": "dessamp-physical-amp-r01-s1-" + "1" * 32,
            "created_at": "2026-08-25T08:00:00Z",
            "expires_at": expires_at,
        }
        raw = MODULE.canonical_json_bytes(payload).decode("utf-8")
        metadata = json.dumps({"generation": "123", "size": len(raw.encode())})
        return contract_sha256, raw, metadata

    def test_writer_lease_creation_is_generation_zero_and_contract_bound(self):
        now = datetime(2026, 8, 25, 9, tzinfo=timezone.utc)
        contract_sha256 = MODULE.canonical_json_sha256(self._member_contract())
        completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with mock.patch.object(MODULE, "run_checked", return_value=completed) as runner:
            result = MODULE.acquire_member_writer_lease(
                project="dessamp-training",
                bucket="dessamp-training-data",
                result_prefix="jobs/physical-amp-r01/results/full-seed-1",
                member_contract_sha256=contract_sha256,
                attempt_nonce="2" * 32,
                pod_name="dessamp-physical-amp-r01-s1-" + "2" * 32,
                now=now,
                expires_at=now + timedelta(hours=2),
            )
        command = runner.call_args.args[0]
        self.assertIn("--if-generation-match=0", command)
        self.assertEqual(result["member_contract_sha256"], contract_sha256)
        self.assertEqual(result["attempt_nonce"], "2" * 32)

    def test_unexpired_writer_lease_blocks_duplicate_submission(self):
        now = datetime(2026, 8, 25, 9, tzinfo=timezone.utc)
        contract_sha256, raw, metadata = self._writer_lease_existing(
            expires_at="2026-08-25T10:00:00Z"
        )
        create_failure = subprocess.CalledProcessError(1, ["gcloud", "storage", "cp"])
        responses = [
            create_failure,
            subprocess.CompletedProcess([], 0, stdout=metadata, stderr=""),
            subprocess.CompletedProcess([], 0, stdout=raw, stderr=""),
            subprocess.CompletedProcess([], 0, stdout=metadata, stderr=""),
        ]
        with mock.patch.object(MODULE, "run_checked", side_effect=responses) as runner:
            with self.assertRaisesRegex(MODULE.PreflightError, "still active"):
                MODULE.acquire_member_writer_lease(
                    project="dessamp-training",
                    bucket="dessamp-training-data",
                    result_prefix="jobs/physical-amp-r01/results/full-seed-1",
                    member_contract_sha256=contract_sha256,
                    attempt_nonce="2" * 32,
                    pod_name="dessamp-physical-amp-r01-s1-" + "2" * 32,
                    now=now,
                    expires_at=now + timedelta(hours=2),
                )
        self.assertEqual(len(runner.call_args_list), 4)
        self.assertIn(
            f"--range=0-{len(raw.encode()) - 1}",
            runner.call_args_list[2].args[0],
        )

    def test_expired_exact_contract_writer_lease_uses_generation_cas(self):
        now = datetime(2026, 8, 25, 9, tzinfo=timezone.utc)
        contract_sha256, raw, metadata = self._writer_lease_existing(
            expires_at="2026-08-25T08:59:59Z"
        )
        create_failure = subprocess.CalledProcessError(1, ["gcloud", "storage", "cp"])
        completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        responses = [
            create_failure,
            subprocess.CompletedProcess([], 0, stdout=metadata, stderr=""),
            subprocess.CompletedProcess([], 0, stdout=raw, stderr=""),
            subprocess.CompletedProcess([], 0, stdout=metadata, stderr=""),
            completed,
        ]
        with mock.patch.object(MODULE, "run_checked", side_effect=responses) as runner:
            result = MODULE.acquire_member_writer_lease(
                project="dessamp-training",
                bucket="dessamp-training-data",
                result_prefix="jobs/physical-amp-r01/results/full-seed-1",
                member_contract_sha256=contract_sha256,
                attempt_nonce="2" * 32,
                pod_name="dessamp-physical-amp-r01-s1-" + "2" * 32,
                now=now,
                expires_at=now + timedelta(hours=2),
            )
        self.assertEqual(result["state"], "expired-lease-reclaimed")
        self.assertIn("--if-generation-match=123", runner.call_args_list[-1].args[0])

    def test_exact_failed_submission_writer_lease_release_uses_generation_cas(self):
        contract_sha256, raw, metadata = self._writer_lease_existing(
            expires_at="2026-08-25T10:00:00Z"
        )
        completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        responses = [
            subprocess.CompletedProcess([], 0, stdout=metadata, stderr=""),
            subprocess.CompletedProcess([], 0, stdout=raw, stderr=""),
            subprocess.CompletedProcess([], 0, stdout=metadata, stderr=""),
            completed,
        ]
        with mock.patch.object(MODULE, "run_checked", side_effect=responses) as runner:
            result = MODULE.release_member_writer_lease(
                project="dessamp-training",
                bucket="dessamp-training-data",
                result_prefix="jobs/physical-amp-r01/results/full-seed-1",
                member_contract_sha256=contract_sha256,
                attempt_nonce="1" * 32,
                pod_name="dessamp-physical-amp-r01-s1-" + "1" * 32,
            )
        self.assertEqual(result["state"], "failed-submission-lease-released")
        self.assertIn("--if-generation-match=123", runner.call_args_list[-1].args[0])

    def test_failed_submission_writer_lease_release_rejects_identity_mismatch(self):
        contract_sha256, raw, metadata = self._writer_lease_existing(
            expires_at="2026-08-25T10:00:00Z"
        )
        responses = [
            subprocess.CompletedProcess([], 0, stdout=metadata, stderr=""),
            subprocess.CompletedProcess([], 0, stdout=raw, stderr=""),
            subprocess.CompletedProcess([], 0, stdout=metadata, stderr=""),
        ]
        with mock.patch.object(MODULE, "run_checked", side_effect=responses) as runner:
            with self.assertRaisesRegex(MODULE.PreflightError, "identity does not match"):
                MODULE.release_member_writer_lease(
                    project="dessamp-training",
                    bucket="dessamp-training-data",
                    result_prefix="jobs/physical-amp-r01/results/full-seed-1",
                    member_contract_sha256=contract_sha256,
                    attempt_nonce="2" * 32,
                    pod_name="dessamp-physical-amp-r01-s1-" + "2" * 32,
                )
        self.assertEqual(len(runner.call_args_list), 3)


if __name__ == "__main__":
    unittest.main()
