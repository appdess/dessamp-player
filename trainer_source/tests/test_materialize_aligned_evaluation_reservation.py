import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "materialize_aligned_evaluation_reservation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_aligned_evaluation_reservation_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_pcm24(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    for sample in samples:
        payload.extend(int(sample).to_bytes(3, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(3)
        output.setframerate(48_000)
        output.writeframes(bytes(payload))


def read_pcm24(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as source:
        payload = source.readframes(source.getnframes())
    return [
        int.from_bytes(payload[index : index + 3], byteorder="little", signed=True)
        for index in range(0, len(payload), 3)
    ]


class MaterializeAlignedEvaluationReservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.workflow = self.root / "00_WORKFLOW"
        self.evidence_dir = self.root / "03_EVIDENCE" / "R2" / "HIDDEN" / "HID-01"
        self.workflow.mkdir(parents=True)
        self.evidence_dir.mkdir(parents=True)
        self.source_input = self.root / "input.wav"
        self.aligned_input = self.root / "input__aligned.wav"
        self.reference = self.root / "hidden.wav"
        write_pcm24(self.source_input, list(range(10)))
        write_pcm24(self.aligned_input, list(range(2, 10)))
        write_pcm24(self.reference, list(range(100, 110)))

        self.evidence = self.evidence_dir / "validation.json"
        self.evidence.write_text(
            json.dumps(
                {
                    "id": "HID-01",
                    "role": "hidden_audit_reference",
                    "quality_status": "pass",
                    "training_exclusion": {
                        "enters_training_sqlite": False,
                        "enters_recommendations": False,
                        "enters_training_corpus": False,
                    },
                    "settings_normalized": {"gain": 0.25},
                    "reference_audio": {
                        "path": str(self.reference),
                        "sha256": MODULE.sha256_file(self.reference),
                    },
                    "validation": {"latency_samples": -2},
                }
            )
        )
        self.plan = self.workflow / "evaluation-plan.json"
        self.plan.write_text(
            json.dumps(
                {
                    "initial_hidden_audit_scope": {"accepted_hidden_ids": ["HID-01"]},
                    "hidden_settings": [
                        {
                            "id": "HID-01",
                            "normalized": {"gain": 0.25},
                            "capture_status": "accepted_hidden_audit_only",
                            "evidence": "../03_EVIDENCE/R2/HIDDEN/HID-01/validation.json",
                        }
                    ],
                }
            )
        )
        self.config = self.root / "data-config.json"
        self.config.write_text(
            json.dumps(
                {
                    "common": {
                        "x_path": str(self.aligned_input),
                        "delay": 0,
                        "allow_unequal_lengths": False,
                    },
                    "metadata": {
                        "control_order": ["gain"],
                        "alignment_materialization": {
                            "source_delay_samples": -2,
                            "source_frames": 10,
                            "aligned_frames": 8,
                            "input": {
                                "source_path": str(self.source_input),
                                "source_sha256": MODULE.sha256_file(self.source_input),
                                "aligned_sha256": MODULE.sha256_file(self.aligned_input),
                            },
                        },
                    },
                }
            )
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def materialize(self):
        output = self.root / "hidden-evaluation-reservation.json"
        audio_dir = self.root / "aligned-hidden"
        reservation = MODULE.materialize_aligned_reservation(
            evaluation_plan_path=self.plan,
            training_data_config_path=self.config,
            output_reservation_path=output,
            audio_dir=audio_dir,
        )
        return reservation, output, audio_dir

    def test_negative_delay_crops_hidden_tail_and_binds_provenance(self):
        reservation, output, audio_dir = self.materialize()
        aligned_hidden = audio_dir / "hid-01__aligned-pcm24.wav"
        self.assertEqual(read_pcm24(aligned_hidden), list(range(100, 108)))
        self.assertEqual(reservation["evaluation_input"]["frames"], 8)
        self.assertEqual(reservation["references"][0]["g_vector"], [0.25])
        self.assertEqual(
            reservation["evaluation_scope"]["role"],
            "frozen_hidden_settings_on_standardized_training_stimulus",
        )
        self.assertFalse(reservation["evaluation_scope"]["unseen_dry_performance"])
        self.assertTrue(output.is_file())

    def test_refuses_plan_and_evidence_vector_mismatch(self):
        payload = json.loads(self.evidence.read_text())
        payload["settings_normalized"]["gain"] = 0.5
        self.evidence.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "plan/evidence vector mismatch"):
            self.materialize()

    def test_refuses_reuse(self):
        self.materialize()
        with self.assertRaisesRegex(RuntimeError, "replace evaluation reservation"):
            self.materialize()

    def test_epoch_aligned_config_supports_per_reference_delays(self):
        second_evidence_dir = self.root / "03_EVIDENCE" / "R2" / "HIDDEN" / "HID-02"
        second_evidence_dir.mkdir(parents=True)
        second_reference = self.root / "hidden-02.wav"
        write_pcm24(second_reference, list(range(200, 210)))
        second_evidence = second_evidence_dir / "validation.json"
        second_evidence.write_text(
            json.dumps(
                {
                    "id": "HID-02",
                    "role": "hidden_audit_reference",
                    "quality_status": "pass",
                    "training_exclusion": {
                        "enters_training_sqlite": False,
                        "enters_recommendations": False,
                        "enters_training_corpus": False,
                    },
                    "settings_normalized": {"gain": 0.75},
                    "reference_audio": {
                        "path": str(second_reference),
                        "sha256": MODULE.sha256_file(second_reference),
                    },
                    "validation": {"latency_samples": -1},
                }
            )
        )
        plan = json.loads(self.plan.read_text())
        plan["current_hidden_evaluation_scope"] = {
            "accepted_hidden_ids": ["HID-01", "HID-02"]
        }
        plan["hidden_settings"].append(
            {
                "id": "HID-02",
                "normalized": {"gain": 0.75},
                "capture_status": "accepted_hidden_audit_only",
                "evidence": "../03_EVIDENCE/R2/HIDDEN/HID-02/validation.json",
            }
        )
        self.plan.write_text(json.dumps(plan))

        write_pcm24(self.aligned_input, list(range(2, 9)))
        config = json.loads(self.config.read_text())
        config["metadata"]["alignment_materialization"] = {
            "schema_version": 2,
            "source_frames": 10,
            "aligned_frames": 7,
            "input": {
                "source_path": str(self.source_input),
                "source_sha256": MODULE.sha256_file(self.source_input),
                "aligned_sha256": MODULE.sha256_file(self.aligned_input),
                "crop_start": 2,
                "crop_stop": 9,
            },
        }
        self.config.write_text(json.dumps(config))

        reservation, _, audio_dir = self.materialize()
        self.assertEqual(read_pcm24(audio_dir / "hid-01__aligned-pcm24.wav"), list(range(100, 107)))
        self.assertEqual(read_pcm24(audio_dir / "hid-02__aligned-pcm24.wav"), list(range(201, 208)))
        self.assertEqual(
            reservation["alignment_materialization"]["source_delay_samples_by_hidden_id"],
            {"HID-01": -2, "HID-02": -1},
        )


if __name__ == "__main__":
    unittest.main()
