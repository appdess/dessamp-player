import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_fortin_conditioned_active_learning.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("fortin_conditioned_workflow_config", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FortinDataConfigTests(unittest.TestCase):
    def setUp(self):
        self.input_path = Path("input.wav")
        self.captures = [
            {"pcm24_path": "center.wav", "g_vector": [0.5, 0.4]},
            {"pcm24_path": "edge.wav", "g_vector": [0.7, 0.6]},
        ]

    def test_zero_delay_is_the_plugin_default(self):
        config = MODULE.make_data_config(self.input_path, self.captures)
        self.assertEqual(config["common"]["delay"], 0)
        self.assertEqual(config["validation"]["outputs"][0]["g_vector"], [0.5, 0.4])
        self.assertEqual(config["train"]["outputs"][0]["g_vector"], [0.7, 0.6])

    def test_explicit_measured_delay_is_preserved(self):
        config = MODULE.make_data_config(
            self.input_path,
            self.captures,
            delay_samples=17,
        )
        self.assertEqual(config["common"]["delay"], 17)

    def test_boolean_delay_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            MODULE.make_data_config(
                self.input_path,
                self.captures,
                delay_samples=True,
            )


class CoverageHoldoutTests(unittest.TestCase):
    def test_selects_uncaptured_points_far_from_starters_and_each_other(self):
        selected = MODULE.select_coverage_holdouts(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.1, 0.1]],
            [[0.0, 0.0]],
            maximum=2,
            minimum_distance=1.0,
        )
        self.assertEqual([item.source_index for item in selected], [3, 1])
        self.assertEqual([list(item.g_vector) for item in selected], [[1.0, 1.0], [1.0, 0.0]])

    def test_refuses_to_weaken_the_holdout_distance_contract(self):
        with self.assertRaisesRegex(RuntimeError, "only 1"):
            MODULE.select_coverage_holdouts(
                [[0.0], [0.2], [0.4]],
                [[0.0]],
                maximum=2,
                minimum_distance=0.3,
            )


class CaptureDatabaseBackupTests(unittest.TestCase):
    def test_backup_is_consistent_with_live_wal_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "capture.sqlite3"
            backup_path = root / "backups" / "capture.sqlite3"
            with sqlite3.connect(source_path) as source:
                source.execute("PRAGMA journal_mode = WAL")
                source.execute("CREATE TABLE captures(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
                source.execute("INSERT INTO captures(value) VALUES ('persisted')")
                source.commit()
                MODULE._backup_capture_database(source_path, backup_path)
            with sqlite3.connect(backup_path) as backup:
                self.assertEqual(
                    backup.execute("SELECT value FROM captures").fetchall(),
                    [("persisted",)],
                )
                self.assertEqual(backup.execute("PRAGMA integrity_check").fetchone(), ("ok",))

    def test_backup_refuses_to_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "capture.sqlite3"
            backup_path = root / "backup.sqlite3"
            with sqlite3.connect(source_path) as source:
                source.execute("CREATE TABLE captures(id INTEGER PRIMARY KEY)")
            backup_path.touch()
            with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
                MODULE._backup_capture_database(source_path, backup_path)


if __name__ == "__main__":
    unittest.main()
