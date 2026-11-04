import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "local_active_learner.py"
SPEC = importlib.util.spec_from_file_location("local_active_learner_alignment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EvaluationAlignmentTests(unittest.TestCase):
    def test_positive_delay_drops_input_tail_and_target_head(self):
        input_slice, target_start = MODULE.evaluation_alignment(100, 7)
        self.assertEqual(input_slice, slice(0, 93))
        self.assertEqual(target_start, 7)

    def test_negative_delay_drops_input_head(self):
        input_slice, target_start = MODULE.evaluation_alignment(100, -7)
        self.assertEqual(input_slice, slice(7, 100))
        self.assertEqual(target_start, 0)

    def test_zero_delay_preserves_complete_input(self):
        input_slice, target_start = MODULE.evaluation_alignment(100, 0)
        self.assertEqual(input_slice, slice(0, 100))
        self.assertEqual(target_start, 0)

    def test_rejects_bool_and_complete_removal(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            MODULE.evaluation_alignment(100, True)
        with self.assertRaisesRegex(ValueError, "complete"):
            MODULE.evaluation_alignment(100, 100)


if __name__ == "__main__":
    unittest.main()
