import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluate_conditioned_exports.py"
SPEC = importlib.util.spec_from_file_location("evaluate_conditioned_exports", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ConditionedExportMetricTests(unittest.TestCase):
    def test_esr_uses_target_energy(self):
        target = np.array([1.0, -1.0], dtype=np.float64)
        prediction = np.array([0.5, -0.5], dtype=np.float64)
        self.assertEqual(MODULE.aligned_esr(prediction, target), 0.25)

    def test_normalized_variance_uses_sample_variance(self):
        target = np.array([1.0, 1.0], dtype=np.float64)
        predictions = np.array([[0.0, 1.0], [2.0, 1.0]], dtype=np.float64)
        self.assertEqual(MODULE.normalized_member_variance(predictions, target), 1.0)

    def test_fixed_audit_coverage(self):
        self.assertEqual(MODULE.evaluated_frame_count(730_141), 725_700)

    def test_metric_summary_matches_round_zero_percentile_convention(self):
        summary = MODULE.metric_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["median_esr"], 2.5)
        self.assertAlmostEqual(summary["p95_esr"], 3.85)

    def test_default_contract_accepts_nam_normal_but_retains_stricter_target(self):
        summary = {"mean_esr": 0.024256, "max_esr": 0.033319}
        acceptable = MODULE.quality_assessment(
            summary,
            mean_gate=MODULE.DEFAULT_MEAN_ESR_GATE,
            max_gate=MODULE.DEFAULT_MAX_ESR_GATE,
        )
        target = MODULE.quality_assessment(
            summary,
            mean_gate=MODULE.DEFAULT_TARGET_MEAN_ESR_GATE,
            max_gate=MODULE.DEFAULT_TARGET_MAX_ESR_GATE,
        )
        self.assertTrue(acceptable["passed"])
        self.assertFalse(target["passed"])

    def test_normal_gate_rejects_one_bad_hidden_setting(self):
        assessment = MODULE.quality_assessment(
            {"mean_esr": 0.02, "max_esr": 0.0351},
            mean_gate=MODULE.DEFAULT_MEAN_ESR_GATE,
            max_gate=MODULE.DEFAULT_MAX_ESR_GATE,
        )
        self.assertFalse(assessment["passed"])

    def test_member_parser_rejects_duplicate_labels(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.parse_members([f"x={MODULE_PATH}", f"x={MODULE_PATH}"])

    def test_member_parser_accepts_one_for_an_independent_release_gate(self):
        members = MODULE.parse_members(
            [f"full={MODULE_PATH}"], minimum_members=1
        )
        self.assertEqual(members, {"full": MODULE_PATH.resolve()})

    def test_member_parser_still_requires_two_for_disagreement(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            MODULE.parse_members([f"full={MODULE_PATH}"])

    def test_pretraining_reservation_is_hash_bound_and_preferred(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            validation = root / "05_VALIDATION"
            capture = root / "02_CAPTURES" / "training-pcm24" / "audit.wav"
            validation.mkdir(parents=True)
            capture.parent.mkdir(parents=True)
            capture.write_bytes(b"RIFF-audit")
            digest = hashlib.sha256(capture.read_bytes()).hexdigest()
            reservation = validation / "hidden-evaluation-reservation.json"
            reservation.write_text(
                json.dumps(
                    {
                        "target_audio_available_to_acquisition": False,
                        "references": [
                            {"g_vector": [0.2, 0.8], "path": str(capture), "sha256": digest}
                        ],
                    }
                )
            )
            source, settings, references = MODULE.load_fixed_evaluation(root.resolve())
            self.assertEqual(source, reservation.resolve())
            self.assertEqual(settings[0]["g_vector"], [0.2, 0.8])
            self.assertEqual(references[0]["sha256"], digest)

    def test_reservation_preserves_explicit_evaluation_role(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "audit.wav"
            capture.write_bytes(b"RIFF-audit")
            digest = hashlib.sha256(capture.read_bytes()).hexdigest()
            reservation = root / "hidden-evaluation-reservation.json"
            reservation.write_text(
                json.dumps(
                    {
                        "target_audio_available_to_acquisition": False,
                        "evaluation_scope": {"role": "same-stimulus-hidden-setting-audit"},
                        "references": [
                            {"g_vector": [0.2, 0.8], "path": str(capture), "sha256": digest}
                        ],
                    }
                )
            )
            _, settings, _ = MODULE.load_evaluation_reservation(reservation.resolve())
            self.assertEqual(settings[0]["role"], "same-stimulus-hidden-setting-audit")

    def test_reservation_preserves_setting_id(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "audit.wav"
            capture.write_bytes(b"RIFF-audit")
            digest = hashlib.sha256(capture.read_bytes()).hexdigest()
            reservation = root / "hidden-evaluation-reservation.json"
            reservation.write_text(
                json.dumps(
                    {
                        "target_audio_available_to_acquisition": False,
                        "references": [
                            {
                                "setting_id": "HID-03",
                                "g_vector": [0.2],
                                "path": str(capture),
                                "sha256": digest,
                            }
                        ],
                    }
                )
            )
            _, settings, references = MODULE.load_evaluation_reservation(
                reservation.resolve()
            )
            self.assertEqual(settings[0]["setting_id"], "HID-03")
            self.assertEqual(references[0]["setting_id"], "HID-03")

    def test_pretraining_reservation_requires_acquisition_isolation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            validation = root / "05_VALIDATION"
            validation.mkdir(parents=True)
            (validation / "hidden-evaluation-reservation.json").write_text(
                json.dumps({"target_audio_available_to_acquisition": True, "references": []})
            )
            with self.assertRaisesRegex(ValueError, "acquisition boundary"):
                MODULE.load_fixed_evaluation(root.resolve())


if __name__ == "__main__":
    unittest.main()
