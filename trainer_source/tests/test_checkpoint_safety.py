import importlib.util
from pathlib import Path
import tempfile
import unittest


TRAINER_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_MODULE_PATH = TRAINER_ROOT / "nam" / "train" / "checkpoint.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_safety", CHECKPOINT_MODULE_PATH)
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
        kwargs = {"map_location": map_location, "weights_only": weights_only}
        self.calls.append((path, kwargs))
        return self.result


class LegacyFakeTorch:
    __version__ = "2.6.0"

    @staticmethod
    def load(path, *, map_location=None):
        return {"state_dict": {"net.weight": FakeTensor()}, "sample_rate": 48000}


def _write_sentinel(path):
    Path(path).write_text("unsafe pickle executed", encoding="utf-8")


class MaliciousPayload:
    def __init__(self, sentinel_path):
        self.sentinel_path = sentinel_path

    def __reduce__(self):
        return _write_sentinel, (self.sentinel_path,)


class CheckpointSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.checkpoint_path = Path(self.temporary_directory.name) / "model.ckpt"
        self.checkpoint_path.write_bytes(b"checkpoint fixture")

    def test_forces_weights_only_and_cpu_mapping(self):
        fake_torch = FakeTorch(
            {"state_dict": {"net.weight": FakeTensor()}, "sample_rate": 48000}
        )

        result = CHECKPOINT.load_weights_only_checkpoint(
            self.checkpoint_path,
            torch_module=fake_torch,
        )

        self.assertEqual(result["sample_rate"], 48000)
        self.assertEqual(len(fake_torch.calls), 1)
        _, kwargs = fake_torch.calls[0]
        self.assertIs(kwargs["weights_only"], True)
        self.assertEqual(kwargs["map_location"], "cpu")

    def test_rejects_a_loader_without_explicit_weights_only_support(self):
        with self.assertRaises(RuntimeError):
            CHECKPOINT.load_weights_only_checkpoint(
                self.checkpoint_path,
                torch_module=LegacyFakeTorch(),
            )

    def test_rejects_vulnerable_or_unverifiable_pytorch_versions(self):
        for version in ("2.5.1", "2.6.0rc1", "nightly", ""):
            with self.subTest(version=version):
                fake_torch = FakeTorch(
                    {"state_dict": {"net.weight": FakeTensor()}, "sample_rate": 48000}
                )
                fake_torch.__version__ = version
                with self.assertRaises(RuntimeError):
                    CHECKPOINT.load_weights_only_checkpoint(
                        self.checkpoint_path,
                        torch_module=fake_torch,
                    )
                self.assertEqual(fake_torch.calls, [])

    def test_rejects_invalid_checkpoint_structure(self):
        invalid_checkpoints = (
            [],
            {},
            {"state_dict": {}, "sample_rate": 48000},
            {"state_dict": {1: FakeTensor()}, "sample_rate": 48000},
            {"state_dict": {"net.weight": object()}, "sample_rate": 48000},
            {"state_dict": {"net.weight": FakeTensor()}, "sample_rate": 0},
            {"state_dict": {"net.weight": FakeTensor()}, "sample_rate": float("nan")},
            {"state_dict": {"net.weight": FakeTensor()}, "sample_rate": float("inf")},
        )
        for checkpoint in invalid_checkpoints:
            with self.subTest(checkpoint=checkpoint):
                with self.assertRaises(ValueError):
                    CHECKPOINT.load_weights_only_checkpoint(
                        self.checkpoint_path,
                        torch_module=FakeTorch(checkpoint),
                    )

    def test_rejects_pickle_checkpoint_options(self):
        for option in ("ckpt_path", "resume_from_checkpoint"):
            with self.subTest(option=option):
                with self.assertRaises(ValueError):
                    CHECKPOINT.validated_trainer_options(
                        {option: "untrusted.ckpt"},
                        context="test options",
                    )

        original = {"max_epochs": 10}
        validated = CHECKPOINT.validated_trainer_options(
            original,
            context="test options",
        )
        self.assertEqual(validated, original)
        self.assertIsNot(validated, original)

    def test_real_weights_only_loader_rejects_a_pickle_payload(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        sentinel = Path(self.temporary_directory.name) / "pickle-ran"
        malicious_checkpoint = Path(self.temporary_directory.name) / "malicious.ckpt"
        torch.save(
            {
                "state_dict": {"net.weight": torch.zeros(1)},
                "sample_rate": 48000,
                "payload": MaliciousPayload(str(sentinel)),
            },
            malicious_checkpoint,
        )

        with self.assertRaises(Exception):
            CHECKPOINT.load_weights_only_checkpoint(malicious_checkpoint)
        self.assertFalse(sentinel.exists())

    def test_tracked_demo_checkpoint_deserializes_in_restricted_mode(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")

        checkpoint = CHECKPOINT.load_weights_only_checkpoint(
            TRAINER_ROOT / "demo_ckpt.ckpt"
        )
        self.assertEqual(checkpoint["sample_rate"], 48000)
        self.assertGreater(len(checkpoint["state_dict"]), 0)

    def test_no_training_utility_calls_the_pickle_capable_loader(self):
        offenders = []
        for source_path in TRAINER_ROOT.rglob("*.py"):
            if source_path == Path(__file__).resolve():
                continue
            if ".load_from_checkpoint(" in source_path.read_text(encoding="utf-8"):
                offenders.append(str(source_path.relative_to(TRAINER_ROOT)))
        self.assertEqual(offenders, [])

    def test_trainer_fit_kwargs_are_validated_before_expansion(self):
        for relative_path in (
            "nam/train/full.py",
            "active_learner_multi_gpu.py",
        ):
            source = (TRAINER_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("_validated_trainer_options(", source)
            self.assertNotIn(
                '**learning_config.get("trainer_fit_kwargs", {})',
                source,
            )

    def test_conditioned_a2_resume_uses_restricted_checkpoint_loading(self):
        source = (
            TRAINER_ROOT / "conditioned_a2" / "train.py"
        ).read_text(encoding="utf-8")

        self.assertIn("load_weights_only_checkpoint(checkpoint_path", source)
        self.assertIn("weights_only=True,", source)
        self.assertNotIn("weights_only=False", source)


if __name__ == "__main__":
    unittest.main()
