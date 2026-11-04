from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from dessmetal_capture.domain import ControlDefinition
from dessmetal_capture.panama import parse_new_points


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINER_ROOT = REPOSITORY_ROOT / "trainer_source"
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from local_active_learner import CandidateScore, _validated_vectors, write_new_points


class LocalActiveLearnerContractTests(unittest.TestCase):
    def test_new_points_are_accepted_by_the_persistent_trainer_boundary(self) -> None:
        ranked = [
            CandidateScore(source_index=8, g_vector=(0.6,), disagreement_score=0.0039),
            CandidateScore(source_index=7, g_vector=(0.55,), disagreement_score=0.0036),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "new_points.json"
            points = write_new_points(path, ranked, maximum=1)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), points)
        controls = [
            ControlDefinition.from_mapping(
                {
                    "id": "gain",
                    "label": "Gain",
                    "minimum": 0.2,
                    "maximum": 0.8,
                    "step": 0.05,
                }
            )
        ]
        imported = parse_new_points(points, controls=controls, control_order=["gain"])
        self.assertEqual(imported[0].settings, {"gain": 0.6})
        self.assertAlmostEqual(imported[0].score, 0.0039)

    def test_vector_validation_rejects_ambiguous_ensemble_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _validated_vectors([[0.2], [0.2]], "candidate_vectors")
        with self.assertRaisesRegex(ValueError, "consistent dimension"):
            _validated_vectors([[0.2], [0.3, 0.4]], "candidate_vectors")


if __name__ == "__main__":
    unittest.main()
