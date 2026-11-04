import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "score_conditioned_exports.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("score_conditioned_exports", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AuditSafeCandidateTests(unittest.TestCase):
    def test_excludes_captured_and_reserved_before_scoring(self):
        config = {
            "candidate_vectors": [[0.0], [0.25], [0.5], [0.75]],
            "captured_vectors": [[0.0]],
            "reserved_evaluation_vectors": [[0.5]],
        }
        self.assertEqual(
            MODULE.available_candidates(config),
            [(1, [0.25]), (3, [0.75])],
        )

    def test_rejects_a_reserved_vector_that_was_already_captured(self):
        config = {
            "candidate_vectors": [[0.0], [0.5]],
            "captured_vectors": [[0.0]],
            "reserved_evaluation_vectors": [[0.0]],
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            MODULE.available_candidates(config)

    def test_vector_key_tolerates_json_float_noise(self):
        self.assertEqual(
            MODULE.vector_key([0.1 + 0.2]),
            MODULE.vector_key([0.3]),
        )


if __name__ == "__main__":
    unittest.main()
