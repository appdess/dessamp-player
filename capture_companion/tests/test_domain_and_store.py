from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dessmetal_capture.domain import DomainError
from dessmetal_capture.store import CaptureStore
from dessmetal_capture.wav_io import inspect_wav, validate_capture_wav, write_pcm24_mono


class DomainAndStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = CaptureStore(self.root / "data")
        self.project = self.store.create_project(
            name="Gain stages",
            amp_name="Test amp",
            controls=[
                {
                    "id": "gain",
                    "label": "Gain",
                    "minimum": 0.2,
                    "maximum": 0.8,
                    "step": 0.1,
                    "display_minimum": 2,
                    "display_maximum": 8,
                }
            ],
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_recommendations_are_persistent_and_cover_uncaptured_points(self) -> None:
        first = self.store.create_recommendation(self.project["id"])
        self.assertEqual(first["settings"], {"gain": 0.5})
        self.assertEqual(self.store.create_recommendation(self.project["id"])["id"], first["id"])

        wav_path = self.root / "capture.wav"
        write_pcm24_mono(wav_path, np.linspace(-0.2, 0.2, 480, dtype=np.float32), 48_000)
        info = inspect_wav(wav_path)
        capture = self.store.add_capture(
            project_id=self.project["id"],
            recommendation_id=first["id"],
            source="test",
            wav_path=wav_path,
            wav_info=info,
            quality_status="pass",
            validation={"status": "pass", "warnings": [], "failures": []},
        )
        self.assertEqual(capture["settings"], {"gain": 0.5})
        second = self.store.create_recommendation(self.project["id"])
        self.assertEqual(second["settings"], {"gain": 0.8})

        reopened = CaptureStore(self.root / "data")
        self.assertEqual(reopened.get_project(self.project["id"])["capture_count"], 1)
        self.assertEqual(reopened.create_recommendation(self.project["id"])["id"], second["id"])

    def test_invalid_grid_is_rejected_before_persistence(self) -> None:
        with self.assertRaises(DomainError):
            self.store.create_project(
                name="Too big",
                amp_name="Amp",
                controls=[
                    {"id": "a", "label": "A", "minimum": 0, "maximum": 1, "step": 0.01},
                    {"id": "b", "label": "B", "minimum": 0, "maximum": 1, "step": 0.01},
                ],
            )

    def test_capture_validation_preserves_alignment_warning(self) -> None:
        wav_path = self.root / "long.wav"
        write_pcm24_mono(wav_path, np.full(49_000, 0.1, dtype=np.float32), 48_000)
        info = inspect_wav(wav_path)
        status, validation = validate_capture_wav(
            info,
            expected_sample_rate=48_000,
            expected_frames=48_000,
        )
        self.assertEqual(status, "warning")
        self.assertAlmostEqual(validation["length_delta_seconds"], 1_000 / 48_000)
        self.assertTrue(any("alignment" in warning for warning in validation["warnings"]))

    def test_truncated_wav_is_rejected(self) -> None:
        wav_path = self.root / "truncated.wav"
        write_pcm24_mono(wav_path, np.full(4_800, 0.1, dtype=np.float32), 48_000)
        with wav_path.open("r+b") as wav_file:
            wav_file.truncate(wav_path.stat().st_size - 300)
        with self.assertRaises(DomainError):
            inspect_wav(wav_path)

    def test_database_backup_is_integrity_checked_and_non_overwriting(self) -> None:
        destination = self.root / "backups" / "capture.sqlite3"
        self.assertEqual(self.store.backup(destination), destination.resolve())
        reopened = CaptureStore(self.root / "backup-open")
        reopened.database_path = destination.resolve()
        self.assertEqual(reopened.get_project(self.project["id"])["name"], "Gain stages")
        with self.assertRaisesRegex(DomainError, "Refusing to replace"):
            self.store.backup(destination)


if __name__ == "__main__":
    unittest.main()
