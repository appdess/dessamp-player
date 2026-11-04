from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dessmetal_capture.dess_package import create_dess_package, read_dess_package
from dessmetal_capture.domain import DomainError
from dessmetal_capture.wav_io import write_pcm24_mono
from dessmetal_player.artifact import PlayerArtifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class DessPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _make_morph_manifest(self) -> Path:
        primary = self.root / "model.nam"
        secondary = self.root / "morph-model.nam"
        shutil.copyfile(
            REPOSITORY_ROOT / "NeuralAmpModelerCore/example_models/lstm.nam",
            primary,
        )
        shutil.copyfile(
            REPOSITORY_ROOT / "NeuralAmpModelerCore/example_models/wavenet.nam",
            secondary,
        )
        primary_sha256 = hashlib.sha256(primary.read_bytes()).hexdigest()
        secondary_sha256 = hashlib.sha256(secondary.read_bytes()).hexdigest()
        runtime_quality = {
            "mode": "fixed",
            "default_size": 1.0,
            "breakpoints": [],
            "private_profile_path": "/Users/private/runtime-profile.json",
        }
        manifest = {
            "schema_version": 1,
            "artifact_id": "dess-forge-package-test",
            "name": "DESS FORGE",
            "model": {
                "path": "model.nam",
                "format": "nam",
                "sha256": primary_sha256,
                "format_version": "0.5.4",
                "architecture": "LSTM",
                "sample_rate": 48_000.0,
                "parameter_mode": "snapshot",
                "global_condition_size": 0,
                "runtime_quality": runtime_quality,
            },
            "controls": [],
            "presentation": {
                "head_shell": "forge-streamlined-v1",
                "product_family": "forge",
                "private_asset_path": "/Users/private/faceplate.png",
            },
            "capture_project": {"id": "private-project-id"},
            "capture_evidence": {"capture_ids": ["private-capture-id"]},
            "provenance": {
                "training_input": "/Users/private/training/input.wav",
                "service_account": "private@example.invalid",
            },
            "runtime_validation": {"message": "/private/build/loadmodel"},
            "morph": {
                "mode": "dual_model_linear_v1",
                "from_label": "GER",
                "to_label": "BRIT",
                "model": {
                    "path": "morph-model.nam",
                    "format": "nam",
                    "sha256": secondary_sha256,
                    "format_version": "0.5.4",
                    "architecture": "WaveNet",
                    "sample_rate": 48_000.0,
                    "parameter_mode": "snapshot",
                    "global_condition_size": 0,
                    "runtime_quality": runtime_quality,
                },
                "alignment": {
                    "delay_samples": 31,
                    "polarity": -1,
                    "level_scale": 0.6074090625600933,
                    "reference_sample_rate": 48_000.0,
                },
                "blend": {
                    "law": "linear",
                    "center_gain": 1.1137418941545114,
                    "smoothing_ms": 30.0,
                },
                "control": {
                    "id": "morph",
                    "label": "Morph",
                    "unit": "%",
                    "taper": "linear",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "step": 0.01,
                    "default_value": 0.5,
                    "display_minimum": 0.0,
                    "display_maximum": 100.0,
                },
            },
        }
        path = self.root / "player-manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_v2_morph_package_embeds_and_verifies_both_models(self) -> None:
        manifest_path = self._make_morph_manifest()
        package_path = self.root / "forge.dess"
        result = create_dess_package(manifest_path, package_path)
        self.assertEqual(result["format_version"], 2)
        self.assertIsNotNone(result["morph_model_sha256"])

        (self.root / "model.nam").unlink()
        (self.root / "morph-model.nam").unlink()
        package = read_dess_package(package_path, cache_dir=self.root / "cache")
        self.assertEqual(package.format_version, 2)
        self.assertTrue(package.model_path.is_file())
        self.assertIsNotNone(package.morph_model_path)
        assert package.morph_model_path is not None
        self.assertTrue(package.morph_model_path.is_file())
        self.assertEqual(package.model_sha256, result["model_sha256"])
        self.assertEqual(package.morph_model_sha256, result["morph_model_sha256"])
        self.assertEqual(package.manifest["presentation"]["head_shell"], "forge-streamlined-v1")
        self.assertEqual(package.manifest["presentation"]["product_family"], "forge")
        self.assertNotIn("private_asset_path", package.manifest["presentation"])
        self.assertNotIn("capture_project", package.manifest)
        self.assertNotIn("capture_evidence", package.manifest)
        self.assertNotIn("provenance", package.manifest)
        self.assertNotIn("runtime_validation", package.manifest)
        self.assertNotIn("private_profile_path", package.manifest["model"]["runtime_quality"])
        self.assertNotIn(
            "private_profile_path",
            package.manifest["morph"]["model"]["runtime_quality"],
        )
        self.assertNotIn("/Users/private", json.dumps(package.manifest, sort_keys=True))

        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertIn("capture_evidence", source_manifest)
        self.assertIn("provenance", source_manifest)

    def test_v2_morph_package_rejects_secondary_payload_corruption(self) -> None:
        package_path = self.root / "forge.dess"
        create_dess_package(self._make_morph_manifest(), package_path)
        damaged = bytearray(package_path.read_bytes())
        damaged[-1] ^= 0x01
        damaged_path = self.root / "forge-damaged.dess"
        damaged_path.write_bytes(damaged)
        with self.assertRaisesRegex(DomainError, "morph package model hash"):
            read_dess_package(damaged_path, cache_dir=self.root / "cache")

    def test_diagnostic_player_routes_morph_through_the_complete_package(self) -> None:
        package_path = self.root / "forge.dess"
        create_dess_package(self._make_morph_manifest(), package_path)
        renderer_path = self.root / "fake-renderer"
        renderer_path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, shutil, sys\n"
            "args = sys.argv[1:]\n"
            "shutil.copyfile(args[1], args[2])\n"
            "parameters = [float(args[index + 1]) for index, value in enumerate(args) if value == '--param']\n"
            "print(json.dumps({'artifact': args[0], 'parameters': parameters}))\n",
            encoding="utf-8",
        )
        renderer_path.chmod(0o700)
        input_path = self.root / "input.wav"
        output_path = self.root / "output.wav"
        write_pcm24_mono(
            input_path,
            np.sin(np.linspace(0.0, 20.0, 4_800, dtype=np.float32)) * 0.1,
            48_000,
        )

        artifact = PlayerArtifact(package_path, renderer_path)
        description = artifact.describe()
        self.assertEqual(description["morph"]["from_label"], "GER")
        self.assertEqual(description["morph"]["to_label"], "BRIT")
        self.assertEqual(description["controls"][0]["id"], "morph")
        self.assertEqual(description["controls"][0]["model_parameter_index"], 0)
        evidence = artifact.render(
            input_path=input_path,
            output_path=output_path,
            settings={"morph": 0.5},
        )
        self.assertEqual(Path(evidence["renderer"]["artifact"]), package_path.resolve())
        self.assertEqual(evidence["renderer"]["parameters"], [0.5])
        self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
