import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "materialize_epoch_aligned_conditioned_data.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_epoch_aligned_conditioned_data_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_pcm24(path: Path, values: list[int]) -> None:
    frames = b"".join(int(value).to_bytes(3, "little", signed=True) for value in values)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 3, 48_000, 0, "NONE", "not compressed"))
        output.writeframes(frames)


def read_pcm24(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as source:
        data = source.readframes(source.getnframes())
    return [int.from_bytes(data[i : i + 3], "little", signed=True) for i in range(0, len(data), 3)]


class EpochAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.x = self.root / "x.wav"
        self.y_old = self.root / "y-old.wav"
        self.y_new = self.root / "y-new.wav"
        write_pcm24(self.x, list(range(10)))
        write_pcm24(self.y_old, list(range(100, 110)))
        write_pcm24(self.y_new, list(range(200, 210)))
        self.source_config = self.root / "source.json"
        self.source_config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metadata": {},
                    "type": "shared_in_multi_out",
                    "common": {
                        "allow_unequal_lengths": False,
                        "delay": 0,
                        "input_gain": 0.0,
                        "x_path": str(self.x),
                    },
                    "train": {"outputs": [{"g_vector": [0.0], "y_path": str(self.y_old)}]},
                    "validation": {
                        "outputs": [{"g_vector": [1.0], "y_path": str(self.y_new)}]
                    },
                }
            ),
            encoding="utf-8",
        )
        self.plan = self.root / "alignment-plan.json"
        self.plan.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_config_sha256": MODULE.sha256_file(self.source_config),
                    "outputs": [
                        {
                            "source_sha256": MODULE.sha256_file(self.y_old),
                            "delay_samples": -2,
                            "epoch_id": "old",
                        },
                        {
                            "source_sha256": MODULE.sha256_file(self.y_new),
                            "delay_samples": -4,
                            "epoch_id": "new",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_materializes_common_input_intersection_and_epoch_output_offsets(self) -> None:
        output_config = self.root / "aligned.json"
        audio_dir = self.root / "audio"
        config = MODULE.materialize_epoch_aligned_config(
            source_config_path=self.source_config,
            alignment_plan_path=self.plan,
            output_config_path=output_config,
            audio_dir=audio_dir,
        )
        self.assertEqual(config["common"]["delay"], 0)
        self.assertEqual(read_pcm24(Path(config["common"]["x_path"])), list(range(4, 10)))
        self.assertEqual(
            read_pcm24(Path(config["train"]["outputs"][0]["y_path"])),
            list(range(102, 108)),
        )
        self.assertEqual(
            read_pcm24(Path(config["validation"]["outputs"][0]["y_path"])),
            list(range(200, 206)),
        )
        materialization = config["metadata"]["alignment_materialization"]
        self.assertEqual(materialization["aligned_frames"], 6)
        self.assertEqual(materialization["input"]["crop_start"], 4)
        self.assertEqual(
            [item["source_delay_samples"] for item in materialization["outputs"]],
            [-2, -4],
        )

    def test_rejects_plan_not_hash_bound_to_source_config(self) -> None:
        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        plan["source_config_sha256"] = "0" * 64
        self.plan.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source_config_sha256 mismatch"):
            MODULE.materialize_epoch_aligned_config(
                source_config_path=self.source_config,
                alignment_plan_path=self.plan,
                output_config_path=self.root / "aligned.json",
                audio_dir=self.root / "audio",
            )

    def test_rejects_missing_or_extra_output_records(self) -> None:
        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        plan["outputs"] = plan["outputs"][:1]
        self.plan.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no record for output hash"):
            MODULE.materialize_epoch_aligned_config(
                source_config_path=self.source_config,
                alignment_plan_path=self.plan,
                output_config_path=self.root / "aligned.json",
                audio_dir=self.root / "audio",
            )


if __name__ == "__main__":
    unittest.main()
