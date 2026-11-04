import importlib.util
import json
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "materialize_aligned_conditioned_data.py"
SPEC = importlib.util.spec_from_file_location("materialize_aligned_conditioned_data_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_pcm16(path: Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def read_pcm16(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as source:
        frames = source.readframes(source.getnframes())
    return list(struct.unpack(f"<{len(frames) // 2}h", frames))


class MaterializeAlignedConditionedDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.x = self.root / "x.wav"
        self.y_train = self.root / "train.wav"
        self.y_validation = self.root / "validation.wav"
        write_pcm16(self.x, list(range(10)))
        write_pcm16(self.y_train, list(range(100, 110)))
        write_pcm16(self.y_validation, list(range(200, 210)))

    def tearDown(self):
        self.temporary_directory.cleanup()

    def materialize(self, delay: int):
        source = self.root / f"source-{delay}.json"
        output = self.root / f"output-{delay}.json"
        audio_dir = self.root / f"audio-{delay}"
        source.write_text(json.dumps({
            "metadata": {"control_order": ["gain"]},
            "common": {"x_path": str(self.x), "delay": delay, "allow_unequal_lengths": False},
            "train": {"outputs": [{"y_path": str(self.y_train), "g_vector": [0.0]}]},
            "validation": {"outputs": [{"y_path": str(self.y_validation), "g_vector": [1.0]}]},
        }))
        config = MODULE.materialize_aligned_config(
            source_config_path=source, output_config_path=output, audio_dir=audio_dir
        )
        return config, output

    def test_negative_delay_crops_input_head_and_output_tail(self):
        config, output = self.materialize(-2)
        self.assertEqual(config["common"]["delay"], 0)
        self.assertEqual(read_pcm16(Path(config["common"]["x_path"])), list(range(2, 10)))
        self.assertEqual(read_pcm16(Path(config["train"]["outputs"][0]["y_path"])), list(range(100, 108)))
        self.assertEqual(config["metadata"]["alignment_materialization"]["aligned_frames"], 8)
        self.assertTrue(output.is_file())

    def test_positive_delay_crops_input_tail_and_output_head(self):
        config, _ = self.materialize(2)
        self.assertEqual(read_pcm16(Path(config["common"]["x_path"])), list(range(8)))
        self.assertEqual(read_pcm16(Path(config["validation"]["outputs"][0]["y_path"])), list(range(202, 210)))

    def test_refuses_zero_delay_and_reuse(self):
        with self.assertRaisesRegex(ValueError, "already delay=0"):
            self.materialize(0)
        self.materialize(-1)
        with self.assertRaisesRegex(RuntimeError, "replace aligned data config"):
            self.materialize(-1)


if __name__ == "__main__":
    unittest.main()
