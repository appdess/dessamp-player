import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "analyze_teacher_interpolation_baselines.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "analyze_teacher_interpolation_baselines_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TeacherInterpolationBaselineTests(unittest.TestCase):
    def test_midpoint_blend_can_reconstruct_midpoint_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lower = np.linspace(-0.5, 0.5, 64, dtype=np.float64)
            upper = -lower
            target = 0.5 * lower + 0.5 * upper
            paths = {}
            for name, audio in (("lower", lower), ("upper", upper), ("target", target)):
                path = root / f"{name}.wav"
                sf.write(path, audio, 48_000, subtype="PCM_24")
                paths[name] = str(path)
            # Keep positive target energy while retaining exact midpoint reconstruction.
            target_audio = np.full(64, 0.125, dtype=np.float64)
            lower_audio = target_audio - 0.25
            upper_audio = target_audio + 0.25
            for name, audio in (
                ("lower", lower_audio),
                ("upper", upper_audio),
                ("target", target_audio),
            ):
                sf.write(paths[name], audio, 48_000, subtype="PCM_24")
            spec = root / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "receptive_field": 3,
                        "chunk_size": 8,
                        "settings": [{**paths, "alpha": 0.5, "g_vector": [0.5]}],
                    }
                )
            )
            report = MODULE.analyze(spec)
            metrics = report["settings"][0]["metrics"]
            self.assertEqual(metrics["linear_output_blend_esr"], 0.0)
            self.assertGreater(metrics["lower_adjacent_snapshot_esr"], 0.0)
            self.assertGreater(metrics["upper_adjacent_snapshot_esr"], 0.0)

    def test_rejects_out_of_range_alpha(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audio = root / "audio.wav"
            sf.write(audio, np.ones(32), 48_000, subtype="PCM_24")
            spec = root / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "receptive_field": 3,
                        "chunk_size": 8,
                        "settings": [
                            {
                                "target": str(audio),
                                "lower": str(audio),
                                "upper": str(audio),
                                "alpha": 1.1,
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "alpha"):
                MODULE.analyze(spec)


if __name__ == "__main__":
    unittest.main()
