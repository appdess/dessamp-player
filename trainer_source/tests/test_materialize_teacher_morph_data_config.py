from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "materialize_teacher_morph_data_config.py"
SPEC = importlib.util.spec_from_file_location("materialize_teacher_morph_data_config", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TeacherMorphDataConfigTests(unittest.TestCase):
    def test_blend_preserves_both_endpoints_exactly(self) -> None:
        us = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        brit = np.asarray([-0.4, 0.5, -0.6], dtype=np.float32)
        np.testing.assert_array_equal(MODULE.blend_audio(us, brit, 0.0, 1.3), us)
        np.testing.assert_array_equal(MODULE.blend_audio(us, brit, 1.0, 1.3), brit)

    def test_center_gain_is_one_for_identical_endpoints(self) -> None:
        endpoint = np.tile(np.asarray([0.2, -0.3, 0.4, -0.1], dtype=np.float32), 2_000)
        gain, ratios = MODULE.calculate_center_gain([(endpoint, endpoint)])
        self.assertAlmostEqual(gain, 1.0, places=7)
        self.assertEqual(len(ratios), 1)

    def test_center_gain_uses_constant_energy_target(self) -> None:
        prefix = np.zeros(MODULE.RF_SAMPLES, dtype=np.float32)
        us = np.concatenate(
            [prefix, np.tile(np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float32), 2_000)]
        )
        brit = np.concatenate(
            [prefix, np.tile(np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float32), 2_000)]
        )
        gain, _ = MODULE.calculate_center_gain([(us, brit)])
        self.assertAlmostEqual(gain, np.sqrt(2.0), places=6)

    def test_grids_must_be_strictly_increasing_unique_and_normalized(self) -> None:
        self.assertEqual(MODULE.parse_grid("0,0.5,1", "test"), [0.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            MODULE.parse_grid("0,0.5,0.5", "test")
        with self.assertRaisesRegex(ValueError, "normalized"):
            MODULE.parse_grid("0,1.1", "test")

    def test_data_config_keeps_gain_then_morph_order_and_separate_inputs(self) -> None:
        config = MODULE.make_data_config(
            root=Path("/tmp/evidence"),
            training_input={"path": "/tmp/train.wav", "sha256": "a" * 64},
            validation_input={"path": "/tmp/validation.wav", "sha256": "b" * 64},
            train_targets=[{"g_vector": [0.0, 0.0], "path": "/tmp/train-target.wav"}],
            validation_targets=[{"g_vector": [0.5, 0.5], "path": "/tmp/validation-target.wav"}],
            teachers=[],
            renderer={"path": "/tmp/render_model", "sha256": "c" * 64},
            center_gain=1.2,
            provenance={},
        )
        self.assertEqual(config["metadata"]["control_order"], ["gain", "morph"])
        self.assertEqual(config["train"]["x_path"], "/tmp/train.wav")
        self.assertEqual(config["validation"]["x_path"], "/tmp/validation.wav")
        self.assertEqual(config["metadata"]["controls"][1]["default_value"], 0.0)
        self.assertFalse(config["metadata"]["external_audit_materialized"])


if __name__ == "__main__":
    unittest.main()
