import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "local_active_learner.py"
SPEC = importlib.util.spec_from_file_location("local_active_learner_diversity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidate(index, vector, score):
    return MODULE.CandidateScore(index, tuple(vector), score)


class DiverseBatchTests(unittest.TestCase):
    def test_preserves_rank_and_skips_near_duplicates(self):
        ranked = [
            candidate(1, [0.0, 0.0], 4.0),
            candidate(2, [0.1, 0.0], 3.0),
            candidate(3, [0.8, 0.0], 2.0),
            candidate(4, [0.0, 0.8], 1.0),
        ]
        selected = MODULE.select_diverse_batch(
            ranked,
            maximum=3,
            minimum_distance=0.5,
        )
        self.assertEqual([item.source_index for item in selected], [1, 3, 4])

    def test_returns_fewer_points_instead_of_diluting_distance(self):
        ranked = [
            candidate(1, [0.0], 2.0),
            candidate(2, [0.1], 1.0),
        ]
        selected = MODULE.select_diverse_batch(
            ranked,
            maximum=2,
            minimum_distance=0.5,
        )
        self.assertEqual([item.source_index for item in selected], [1])

    def test_rejects_duplicate_or_inconsistent_vectors(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.select_diverse_batch(
                [candidate(1, [0.0], 2.0), candidate(2, [0.0], 1.0)],
                maximum=2,
                minimum_distance=0.0,
            )
        with self.assertRaisesRegex(ValueError, "consistent"):
            MODULE.select_diverse_batch(
                [candidate(1, [0.0], 2.0), candidate(2, [0.5, 0.5], 1.0)],
                maximum=2,
                minimum_distance=0.0,
            )


if __name__ == "__main__":
    unittest.main()
