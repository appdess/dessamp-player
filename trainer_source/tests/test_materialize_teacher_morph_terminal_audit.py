from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "materialize_teacher_morph_terminal_audit_test",
    ROOT / "materialize_teacher_morph_terminal_audit.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from dessmetal_capture.wav_io import inspect_wav, write_pcm24_mono


class EndpointReservationTests(unittest.TestCase):
    def test_existing_gain_reference_reuses_audio_and_appends_morph_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            target = root / "target.wav"
            write_pcm24_mono(
                target,
                np.tile(np.asarray([0.2, -0.1], dtype=np.float32), 4_000),
                48_000,
            )
            info = inspect_wav(target)
            source = {
                "schema_version": 1,
                "control_order": ["gain_index"],
                "evaluation_input": {"path": "/tmp/input.wav", "sha256": "a" * 64},
                "references": [
                    {
                        "setting_id": "TEACH-01",
                        "g_vector": [0.25],
                        "path": str(target),
                        "sha256": info["sha256"],
                        "frames": info["frames"],
                        "sample_rate": 48_000,
                        "teacher_sha256": "b" * 64,
                    }
                ],
                "target_audio_available_to_acquisition": False,
            }
            source_path = root / "source.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            derived = MODULE.derive_endpoint_reservation(
                source_path=source_path,
                expected_source_sha256=MODULE.sha256_file(source_path),
                expected_count=1,
                morph=1.0,
                role="BRIT",
                evaluation_input={
                    "path": "/tmp/input.wav",
                    "sha256": "a" * 64,
                    "frames": info["frames"],
                    "sample_rate": 48_000,
                },
                plan_record={"path": "/tmp/plan.json", "sha256": "c" * 64},
                selection_record={"selected_member": "122"},
            )
            self.assertEqual(derived["control_order"], ["gain", "morph"])
            self.assertEqual(derived["references"][0]["g_vector"], [0.25, 1.0])
            self.assertEqual(derived["references"][0]["path"], str(target.resolve()))
            self.assertFalse(derived["target_audio_available_to_acquisition"])

    def test_source_with_visible_targets_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            source_path = Path(raw_root) / "source.json"
            source_path.write_text(
                json.dumps(
                    {
                        "control_order": ["gain_index"],
                        "references": [{}],
                        "target_audio_available_to_acquisition": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "acquisition boundary"):
                MODULE.derive_endpoint_reservation(
                    source_path=source_path,
                    expected_source_sha256=MODULE.sha256_file(source_path),
                    expected_count=1,
                    morph=0.0,
                    role="US",
                    evaluation_input={"sha256": "a" * 64},
                    plan_record={},
                    selection_record={},
                )


if __name__ == "__main__":
    unittest.main()
