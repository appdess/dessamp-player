import importlib.util
from pathlib import Path
import tempfile
import unittest


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "conditioned_a2" / "checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "conditioned_a2_checkpoint_safety", CHECKPOINT_PATH
)
CHECKPOINT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKPOINT)


class FakeTensor:
    pass


class FakeTorch:
    __version__ = "2.6.0"

    def __init__(self, result):
        self.result = result
        self.calls = []

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    def load(self, path, *, map_location=None, weights_only=None):
        self.calls.append(
            (path, {"map_location": map_location, "weights_only": weights_only})
        )
        return self.result


def _write_sentinel(path):
    Path(path).write_text("unsafe pickle executed", encoding="utf-8")


class MaliciousPayload:
    def __init__(self, sentinel_path):
        self.sentinel_path = sentinel_path

    def __reduce__(self):
        return _write_sentinel, (self.sentinel_path,)


class ConditionedCheckpointSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.checkpoint_path = Path(self.temporary_directory.name) / "model.ckpt"
        self.checkpoint_path.write_bytes(b"checkpoint fixture")

    def test_forces_restricted_cpu_loading(self):
        fake_torch = FakeTorch(
            {"state_dict": {"net.weight": FakeTensor()}, "sample_rate": 48000}
        )

        result = CHECKPOINT.load_weights_only_checkpoint(
            self.checkpoint_path, torch_module=fake_torch
        )

        self.assertEqual(result["sample_rate"], 48000)
        self.assertEqual(
            fake_torch.calls[0][1],
            {"map_location": "cpu", "weights_only": True},
        )

    def test_rejects_symlinked_checkpoint(self):
        target = Path(self.temporary_directory.name) / "target.ckpt"
        target.write_bytes(b"checkpoint fixture")
        link = Path(self.temporary_directory.name) / "link.ckpt"
        link.symlink_to(target)

        with self.assertRaises(FileNotFoundError):
            CHECKPOINT.load_weights_only_checkpoint(
                link,
                torch_module=FakeTorch(
                    {
                        "state_dict": {"net.weight": FakeTensor()},
                        "sample_rate": 48000,
                    }
                ),
            )

    def test_real_loader_rejects_pickle_payload(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        sentinel = Path(self.temporary_directory.name) / "pickle-ran"
        malicious = Path(self.temporary_directory.name) / "malicious.ckpt"
        torch.save(
            {
                "state_dict": {"net.weight": torch.zeros(1)},
                "sample_rate": 48000,
                "payload": MaliciousPayload(str(sentinel)),
            },
            malicious,
        )

        with self.assertRaises(Exception):
            CHECKPOINT.load_weights_only_checkpoint(malicious)
        self.assertFalse(sentinel.exists())

    def test_fixed_checkpoint_is_compatible(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")

        checkpoint = CHECKPOINT.load_weights_only_checkpoint(
            Path(__file__).resolve().parents[2] / "demo_ckpt.ckpt"
        )
        self.assertEqual(checkpoint["sample_rate"], 48000)
        self.assertTrue(checkpoint["optimizer_states"])
        self.assertTrue(checkpoint["lr_schedulers"])


if __name__ == "__main__":
    unittest.main()
