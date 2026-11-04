from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from dessmetal_capture.domain import DomainError
from dessmetal_capture.store import CaptureStore
from dessmetal_capture.training import TrainingJobManager, build_training_job
from dessmetal_capture.wav_io import inspect_wav, write_pcm24_mono


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINER_ROOT = REPOSITORY_ROOT / "trainer_source"


class TrainingJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = CaptureStore(self.root / "trainer-data")
        self.project = self.store.create_project(
            name="Conditioned gain",
            amp_name="Test amp",
            controls=[
                {
                    "id": "gain",
                    "label": "Gain",
                    "minimum": 0.2,
                    "maximum": 0.8,
                    "step": 0.3,
                    "display_minimum": 2.0,
                    "display_maximum": 8.0,
                }
            ],
        )
        self.input_path = self.root / "input.wav"
        audio = np.sin(np.linspace(0.0, 200.0, 96_000, dtype=np.float32)) * 0.1
        write_pcm24_mono(self.input_path, audio, 48_000)
        self.store.set_input_signal(self.project["id"], self.input_path, inspect_wav(self.input_path))
        self._add_capture(0.5, "pass")
        self._add_capture(0.8, "pass")
        self._add_capture(0.2, "pass")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _add_capture(self, expected_gain: float, quality_status: str) -> None:
        recommendation = self.store.create_recommendation(self.project["id"])
        self.assertAlmostEqual(recommendation["settings"]["gain"], expected_gain)
        capture_path = self.root / f"capture-{expected_gain:.1f}.wav"
        signal = np.sin(np.linspace(0.0, 200.0, 96_000, dtype=np.float32)) * (0.1 + expected_gain / 10.0)
        write_pcm24_mono(capture_path, signal, 48_000)
        info = inspect_wav(capture_path)
        self.store.add_capture(
            project_id=self.project["id"],
            recommendation_id=recommendation["id"],
            source="test",
            wav_path=capture_path,
            wav_info=info,
            quality_status=quality_status,
            validation={"status": quality_status, "warnings": [], "failures": []},
        )

    def test_builder_freezes_condition_order_hashes_and_panama_configs(self) -> None:
        job = build_training_job(
            self.store,
            project_id=self.project["id"],
            trainer_root=TRAINER_ROOT,
            backend="local-cpu",
            epochs=1,
            batch_size=2,
            ny=1024,
            allow_capture_warnings=False,
            alignment_policy="strict",
            workflow_validation_only=False,
        )
        self.assertEqual(job["status"], "ready")
        self.assertEqual(job["control_order"], ["gain"])
        data = json.loads(Path(job["data_config_path"]).read_text(encoding="utf-8"))
        self.assertEqual([entry["g_vector"] for entry in data["train"]["outputs"]], [[0.2], [0.8]])
        self.assertEqual(data["validation"]["outputs"][0]["g_vector"], [0.5])
        model = json.loads(Path(job["model_config_path"]).read_text(encoding="utf-8"))
        self.assertEqual(model["net"]["global_condition_size"], 1)
        self.assertTrue(
            all(layer["global_condition_size"] == 1 for layer in model["net"]["config"]["layers_configs"])
        )
        learning = json.loads(Path(job["learning_config_path"]).read_text(encoding="utf-8"))
        self.assertEqual(learning["trainer"]["accelerator"], "cpu")
        self.assertEqual(learning["train_dataloader"]["num_workers"], 0)
        self.assertEqual(job["options"]["receptive_field"], 4093)
        self.assertEqual(job["options"]["validation_mode"], "held-out-control-setting")
        frozen = json.loads((Path(job["job_dir"]) / "training-job.json").read_text(encoding="utf-8"))
        self.assertEqual(frozen["source_evidence"]["input"]["sha256"], inspect_wav(self.input_path)["sha256"])
        reopened = CaptureStore(self.root / "trainer-data")
        self.assertEqual(reopened.get_training_job(job["id"])["status"], "ready")

    def test_clipped_workflow_job_records_non_quality_evidence(self) -> None:
        job = build_training_job(
            self.store,
            project_id=self.project["id"],
            trainer_root=TRAINER_ROOT,
            backend="local-mps",
            epochs=1,
            batch_size=2,
            ny=1024,
            allow_capture_warnings=True,
            alignment_policy="truncate-to-shortest",
            workflow_validation_only=True,
            clip_start_seconds=0.25,
            clip_duration_seconds=1.0,
        )
        data = json.loads(Path(job["data_config_path"]).read_text(encoding="utf-8"))
        self.assertEqual(data["common"]["start_seconds"], 0.25)
        self.assertEqual(data["common"]["stop_seconds"], 1.25)
        self.assertTrue(job["options"]["workflow_validation_only"])
        with self.assertRaisesRegex(DomainError, "workflow_validation_only"):
            build_training_job(
                self.store,
                project_id=self.project["id"],
                trainer_root=TRAINER_ROOT,
                backend="local-cpu",
                epochs=1,
                batch_size=2,
                ny=1024,
                allow_capture_warnings=False,
                alignment_policy="strict",
                workflow_validation_only=False,
                clip_start_seconds=0.0,
                clip_duration_seconds=1.0,
            )

    def test_background_worker_exports_a_player_artifact(self) -> None:
        fake_trainer = self._make_fake_trainer()
        job = build_training_job(
            self.store,
            project_id=self.project["id"],
            trainer_root=fake_trainer,
            backend="local-cpu",
            epochs=1,
            batch_size=2,
            ny=1024,
            allow_capture_warnings=False,
            alignment_policy="strict",
            workflow_validation_only=True,
        )
        manager = TrainingJobManager(
            self.store,
            trainer_root=fake_trainer,
            loader_path=None,
            python_executable=sys.executable,
        )
        try:
            manager.start(job["id"])
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                result = manager.get(job["id"])
                if result["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.05)
            self.assertEqual(
                result["status"], "completed", result.get("log_tail") or result.get("error_message")
            )
            manifest_path = Path(result["result"]["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model"]["global_condition_size"], 1)
            self.assertEqual(manifest["controls"][0]["model_parameter_index"], 0)
            self.assertEqual(manifest["provenance"]["training_job_id"], job["id"])
            self.assertEqual(result["result"]["training_metrics"]["esr"], 0.125)
            self.assertEqual(manifest["provenance"]["training_metrics"]["mse"], 0.01)
        finally:
            manager.close()

    def test_closing_trainer_terminates_and_persists_active_job(self) -> None:
        fake_trainer = self._make_fake_trainer()
        (fake_trainer / "custom_train_full.py").write_text(
            "import time\nprint('long fixture started', flush=True)\ntime.sleep(30)\n",
            encoding="utf-8",
        )
        job = build_training_job(
            self.store,
            project_id=self.project["id"],
            trainer_root=fake_trainer,
            backend="local-cpu",
            epochs=1,
            batch_size=2,
            ny=1024,
            allow_capture_warnings=False,
            alignment_policy="strict",
            workflow_validation_only=True,
        )
        manager = TrainingJobManager(
            self.store,
            trainer_root=fake_trainer,
            loader_path=None,
            python_executable=sys.executable,
        )
        manager.start(job["id"])
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and manager.get(job["id"])["status"] != "running":
            time.sleep(0.01)
        manager.close()
        stopped = self.store.get_training_job(job["id"])
        self.assertEqual(stopped["status"], "failed")
        self.assertRegex(stopped["error_message"], "status|stopped")

    def _make_fake_trainer(self) -> Path:
        trainer = self.root / "fake-trainer"
        (trainer / "default_config_files" / "learning").mkdir(parents=True)
        shutil.copyfile(TRAINER_ROOT / "LICENSE", trainer / "LICENSE")
        shutil.copyfile(TRAINER_ROOT / "parametric_wavenet.json", trainer / "parametric_wavenet.json")
        shutil.copyfile(
            TRAINER_ROOT / "default_config_files" / "learning" / "default.json",
            trainer / "default_config_files" / "learning" / "default.json",
        )
        (trainer / "custom_train_full.py").write_text(
            "import argparse, json\n"
            "from pathlib import Path\n"
            "p=argparse.ArgumentParser()\n"
            "p.add_argument('--base-outdir'); p.add_argument('--outdir-suffix')\n"
            "p.add_argument('--model-config'); p.add_argument('--data-config')\n"
            "p.add_argument('--learning-config'); p.add_argument('--batch-size')\n"
            "p.add_argument('--check-val-every-n-epoch'); p.add_argument('--epochs'); p.add_argument('--ny')\n"
            "a=p.parse_args(); out=Path(a.base_outdir)/('run'+a.outdir_suffix); out.mkdir(parents=True)\n"
            "(out/'config_model.json').write_text(Path(a.model_config).read_text())\n"
            "(out/'model.ckpt').write_text('fixture')\n",
            encoding="utf-8",
        )
        source_fixture = REPOSITORY_ROOT / "NeuralAmpModelerCore/example_models/wavenet.nam"
        fixture = trainer / "conditioned-test-fixture.nam"
        conditioned_model = json.loads(source_fixture.read_text(encoding="utf-8"))
        conditioned_model["config"]["global_condition_size"] = 1
        fixture.write_text(json.dumps(conditioned_model), encoding="utf-8")
        (trainer / "export_model.py").write_text(
            "import shutil, sys\n"
            "from pathlib import Path\n"
            f"source=Path({str(fixture)!r})\n"
            "out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)\n"
            "shutil.copyfile(source, out/'model.nam')\n"
            "(out/'model_stats.json').write_text('{\"ESR\":0.125,\"MSE\":0.01,\"Checkpoint\":\"fixture.ckpt\"}')\n",
            encoding="utf-8",
        )
        return trainer


if __name__ == "__main__":
    unittest.main()
