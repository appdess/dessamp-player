import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze_conditioned_learning_curve.py"
SPEC = importlib.util.spec_from_file_location("conditioned_learning_curve_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PowerLawFitTests(unittest.TestCase):
    def test_exact_inverse_square_curve_is_recovered(self):
        fit = MODULE.fit_power_law([10, 20, 40], [0.08, 0.02, 0.005])
        self.assertAlmostEqual(fit["coefficient"], 8.0, places=10)
        self.assertAlmostEqual(fit["exponent"], 2.0, places=10)
        self.assertAlmostEqual(fit["r_squared_log_space"], 1.0, places=10)
        self.assertEqual(MODULE.projected_count(fit, 0.01), 29)

    def test_flat_or_worsening_curve_has_no_projection(self):
        fit = MODULE.fit_power_law([10, 20], [0.02, 0.03])
        self.assertLess(fit["exponent"], 0.0)
        self.assertIsNone(MODULE.projected_count(fit, 0.01))

    def test_invalid_metrics_are_rejected(self):
        for value in (0.0, -1.0, math.inf, math.nan, "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MODULE.finite_metric(value, "metric")

    def test_curve_projects_normal_acceptance_and_stricter_target_separately(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            points = []
            for count, mean, maximum in ((10, 0.08, 0.10), (40, 0.02, 0.03)):
                report = {
                    "acceptance": {
                        "required_mean_esr_lte": 0.035,
                        "required_max_esr_lte": 0.035,
                    },
                    "quality_target": {
                        "required_mean_esr_lte": 0.01,
                        "required_max_esr_lte": 0.02,
                    },
                    "ensemble": {"summary": {"mean_esr": mean, "max_esr": maximum}},
                    "individual": {
                        "72": {"summary": {"mean_esr": mean * 1.2, "max_esr": maximum * 1.2}}
                    },
                }
                path = root / f"{count}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                points.append(f"{count}={path}")
            result = MODULE.analyze(points)
            fit = result["fits"]["ensemble_mean_esr"]
            self.assertEqual(fit["acceptable_target"], 0.035)
            self.assertEqual(fit["gate_target"], 0.01)
            self.assertEqual(
                result["acceptance"]["quality_target"]["required_max_esr_lte"],
                0.02,
            )


if __name__ == "__main__":
    unittest.main()
