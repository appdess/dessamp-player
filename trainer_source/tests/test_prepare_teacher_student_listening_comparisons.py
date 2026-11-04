import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "prepare_teacher_student_listening_comparisons.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "prepare_teacher_student_listening_comparisons_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TeacherStudentListeningComparisonTests(unittest.TestCase):
    def test_applies_identical_safety_attenuation_when_matched_pair_would_clip(self):
        teacher = np.full(4096, 2.0, dtype=np.float64)
        student = np.full(4096, 1.5, dtype=np.float64)
        teacher_copy, student_copy, matching = MODULE._attenuation_only_match(
            teacher,
            student,
            trim_start=0,
            active_threshold=1.0e-5,
        )

        self.assertAlmostEqual(matching["teacher_gain_linear"], 0.75, places=7)
        self.assertEqual(matching["student_gain_linear"], 1.0)
        self.assertLess(matching["common_safety_gain_linear"], 1.0)
        self.assertAlmostEqual(matching["peak_after_matching"], 0.98, places=7)
        self.assertLess(np.max(np.abs(teacher_copy)), 1.0)
        self.assertLess(np.max(np.abs(student_copy)), 1.0)
        self.assertLess(np.max(np.abs(teacher_copy - student_copy)), 1.0e-12)
        self.assertFalse(matching["boost_applied"])

    def test_attenuates_only_louder_copy_and_writes_cabinet_pair(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.wav"
            teacher_path = root / "teacher.wav"
            student_path = root / "student.wav"
            ir_path = root / "ir.wav"
            signal = 0.1 * np.sin(2.0 * np.pi * np.arange(4096) / 64.0)
            sf.write(input_path, signal, 48_000, subtype="PCM_24")
            sf.write(teacher_path, 2.0 * signal, 48_000, subtype="PCM_24")
            sf.write(student_path, signal, 48_000, subtype="PCM_24")
            sf.write(ir_path, np.array([0.5, 0.25]), 48_000, subtype="PCM_24")
            spec = root / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "receptive_field": 3,
                        "input": str(input_path),
                        "cabinet_ir": str(ir_path),
                        "settings": [
                            {
                                "label": "gain-05",
                                "g_vector": [4.0 / 9.0],
                                "teacher": str(teacher_path),
                                "student": str(student_path),
                            }
                        ],
                    }
                )
            )
            output_dir = root / "listening"
            report = MODULE.prepare(spec, output_dir)
            setting = report["settings"][0]
            head = setting["head_only"]
            self.assertAlmostEqual(head["matching"]["teacher_gain_linear"], 0.5, places=5)
            self.assertEqual(head["matching"]["student_gain_linear"], 1.0)
            self.assertFalse(head["matching"]["boost_applied"])
            self.assertIn("cabinet", setting)
            for section in ("head_only", "cabinet"):
                teacher_copy, rate = sf.read(setting[section]["teacher"]["path"])
                student_copy, student_rate = sf.read(setting[section]["student"]["path"])
                self.assertEqual(rate, 48_000)
                self.assertEqual(student_rate, rate)
                self.assertEqual(len(teacher_copy), len(student_copy))
                self.assertLess(np.max(np.abs(teacher_copy - student_copy)), 2.0e-6)
            self.assertFalse(report["human_listening_passed"])

    def test_refuses_to_replace_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec = root / "spec.json"
            spec.write_text("{}")
            output_dir = root / "listening"
            output_dir.mkdir()
            with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
                MODULE.prepare(spec, output_dir)


if __name__ == "__main__":
    unittest.main()
