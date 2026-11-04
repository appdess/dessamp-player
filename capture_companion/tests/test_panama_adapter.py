from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dessmetal_capture.domain import ControlDefinition, DomainError
from dessmetal_capture.panama import parse_new_points, source_digest
from dessmetal_capture.store import CaptureStore


class PanamaAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = CaptureStore(self.root / "data")
        self.project = self.store.create_project(
            name="Two knob active capture",
            amp_name="Test amp",
            controls=[
                {"id": "gain", "label": "Gain", "minimum": 0.0, "maximum": 1.0, "step": 0.5},
                {"id": "treble", "label": "Treble", "minimum": 0.0, "maximum": 1.0, "step": 0.5},
            ],
        )
        self.controls = [ControlDefinition.from_mapping(control) for control in self.project["controls"]]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_panama_batch_is_persistent_and_drives_recommendations_in_order(self) -> None:
        points = [
            {"index": 8, "g_vector": [0.75, 0.25], "disagreement_score": 0.42},
            {"index": 9, "g_vector": [0.1, 0.9]},
        ]
        suggestions = parse_new_points(
            points,
            controls=self.controls,
            control_order=["treble", "gain"],
        )
        digest = source_digest(points)
        queued = self.store.enqueue_panama_suggestions(
            self.project["id"],
            round_index=2,
            source_sha256=digest,
            control_order=["treble", "gain"],
            suggestions=suggestions,
        )
        self.assertEqual([item["status"] for item in queued], ["queued", "queued"])

        first = self.store.create_recommendation(self.project["id"])
        self.assertEqual(first["strategy"], "panama-ensemble-v1")
        self.assertEqual(first["settings"], {"gain": 0.25, "treble": 0.75})
        self.store.dismiss_recommendation(first["id"])

        second = self.store.create_recommendation(self.project["id"])
        self.assertEqual(second["settings"], {"gain": 0.9, "treble": 0.1})
        reopened = CaptureStore(self.root / "data")
        self.assertEqual(reopened.create_recommendation(self.project["id"])["id"], second["id"])
        statuses = [item["status"] for item in reopened.list_external_suggestions(self.project["id"])]
        self.assertEqual(statuses, ["dismissed", "activated"])

        # The batch identity is content-derived, so importing the same trainer
        # result again is idempotent even after a point was consumed.
        repeated = reopened.enqueue_panama_suggestions(
            self.project["id"],
            round_index=2,
            source_sha256=digest,
            control_order=["treble", "gain"],
            suggestions=suggestions,
        )
        self.assertEqual(len(repeated), 2)

    def test_panama_import_requires_explicit_safe_mapping(self) -> None:
        with self.assertRaises(DomainError):
            parse_new_points(
                [{"index": 0, "g_vector": [0.2, 0.8]}],
                controls=self.controls,
                control_order=["gain", "gain"],
            )
        with self.assertRaises(DomainError):
            parse_new_points(
                [{"index": 0, "g_vector": [0.2, 1.2]}],
                controls=self.controls,
                control_order=["gain", "treble"],
            )


if __name__ == "__main__":
    unittest.main()
