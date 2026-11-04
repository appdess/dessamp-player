from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dessmetal_capture.demo import TRAINING_PATH_PREFIX, bootstrap_dessblock_green_demo
from dessmetal_capture.api import create_app
from dessmetal_capture.domain import DomainError
from dessmetal_capture.manifests import fixed_controls_from_project, import_model_artifact, inspect_nam_model
from dessmetal_capture.store import CaptureStore
from dessmetal_capture.wav_io import inspect_wav, validate_capture_wav, write_pcm24_mono


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DEMO_AVAILABLE = all(
    path.is_file()
    for path in (
        REPOSITORY_ROOT / "trainer_source" / "training_data" / "input.wav",
        REPOSITORY_ROOT
        / "NeuralAmpModeler"
        / "resources"
        / "models"
        / "DessBlock-green"
        / "model.nam",
    )
)


class DemoAndManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.capture_root = self.root / "captures"
        config = json.loads(
            (REPOSITORY_ROOT / "trainer_source" / "parametric_data_dessblock_green.json").read_text(encoding="utf-8")
        )
        for index, output in enumerate(config["train"]["outputs"]):
            relative = Path(str(output["y_path"]).removeprefix(TRAINING_PATH_PREFIX))
            write_pcm24_mono(
                self.capture_root / relative,
                np.full(4_800, 0.05 + index * 0.01, dtype=np.float32),
                48_000,
            )
        self.store = CaptureStore(self.root / "data")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @unittest.skipUnless(
        LEGACY_DEMO_AVAILABLE,
        "legacy DessMetal replay assets are intentionally excluded from DessAmp",
    )
    def test_replay_workflow_reaches_player_manifest(self) -> None:
        demo = bootstrap_dessblock_green_demo(
            self.store,
            repository_root=REPOSITORY_ROOT,
            capture_root=self.capture_root,
        )
        project = demo["project"]
        self.assertEqual(demo["replay_asset_count"], 7)
        short_input = self.root / "short-input.wav"
        write_pcm24_mono(short_input, np.full(4_800, 0.1, dtype=np.float32), 48_000)
        project = self.store.set_input_signal(project["id"], short_input, inspect_wav(short_input))
        input_frames = project["input_signal_info"]["frames"]
        for _ in range(7):
            recommendation = self.store.create_recommendation(project["id"])
            wav_path = self.store.get_replay_asset(project["id"], recommendation["settings"])
            info = inspect_wav(wav_path)
            quality, validation = validate_capture_wav(
                info,
                expected_sample_rate=48_000,
                expected_frames=input_frames,
            )
            self.store.add_capture(
                project_id=project["id"],
                recommendation_id=recommendation["id"],
                source="replay-dataset",
                wav_path=wav_path,
                wav_info=info,
                quality_status=quality,
                validation=validation,
            )
        with self.assertRaises(DomainError):
            self.store.create_recommendation(project["id"])

        artifact = import_model_artifact(
            self.store,
            project_id=project["id"],
            source_model_path=Path(demo["fixture_model_path"]),
            name="Replay model",
            provenance={"training_backend": "pretrained-fixture", "workflow_validation_only": True},
            loader_path=None,
        )
        self.assertEqual(artifact["validation_status"], "unverified")
        manifest = json.loads(Path(artifact["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["model"]["parameter_mode"], "global_condition")
        self.assertEqual(manifest["model"]["global_condition_size"], 1)
        self.assertEqual(manifest["model"]["runtime_quality"]["mode"], "fixed")
        self.assertEqual(manifest["capture_evidence"]["capture_count"], 7)
        self.assertTrue((Path(artifact["manifest_path"]).parent / manifest["model"]["path"]).is_file())
        self.assertTrue(Path(artifact["package_path"]).is_file())
        self.assertEqual(Path(artifact["package_path"]).suffix, ".dess")

        app = create_app(data_dir=self.root / "data", access_token="test-token")
        package_route = next(
            route
            for route in app.routes
            if getattr(route, "path", None) == "/api/models/{artifact_id}/package"
        )
        response = asyncio.run(package_route.endpoint(artifact["id"]))
        self.assertEqual(Path(response.path), Path(artifact["package_path"]))
        self.assertEqual(response.media_type, "application/vnd.dessmetal.amp-model")

    def test_a2_container_metadata_is_separate_from_amp_controls(self) -> None:
        kernels = [6] * 23
        child = lambda channels: {
            "version": "0.7.0",
            "architecture": "WaveNet",
            "sample_rate": 48_000.0,
            "config": {"layers": [{"channels": channels, "kernel_sizes": kernels}]},
            "weights": [],
        }
        model_path = self.root / "a2.nam"
        model_path.write_text(
            json.dumps(
                {
                    "version": "0.7.0",
                    "architecture": "SlimmableContainer",
                    "sample_rate": 48_000.0,
                    "config": {
                        "submodels": [
                            {"max_value": 0.5, "model": child(3)},
                            {"max_value": 1.0, "model": child(8)},
                        ]
                    },
                    "weights": [],
                }
            ),
            encoding="utf-8",
        )
        metadata = inspect_nam_model(model_path)
        self.assertEqual(metadata["sample_rate"], 48_000)
        self.assertEqual(metadata["global_condition_size"], 0)
        self.assertEqual(metadata["runtime_quality"]["mode"], "slimmable")
        self.assertEqual(metadata["runtime_quality"]["family"], "a2")
        self.assertEqual(metadata["runtime_quality"]["breakpoints"], [0.5])

    def test_fixed_switches_come_from_project_metadata_without_consuming_model_controls(self) -> None:
        project = {
            "controls": [{"id": "lead_gain"}],
            "metadata": {
                "frozen_controls": {
                    "bright": "on",
                    "bottom": "off",
                    "mid_boost": "off",
                    "lead_volume": 3.5,
                }
            },
        }
        self.assertEqual(
            fixed_controls_from_project(project),
            [
                {"id": "bright", "label": "Bright", "taper": "switch", "value": 1.0},
                {"id": "bottom", "label": "Bottom", "taper": "switch", "value": 0.0},
                {"id": "mid_boost", "label": "Mid Boost", "taper": "switch", "value": 0.0},
                {
                    "id": "lead_volume",
                    "label": "Lead Volume",
                    "taper": "linear",
                    "value": 3.5,
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "unit": "",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
