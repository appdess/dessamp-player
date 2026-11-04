import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "conditioned_a2_entrypoint.py"
SPEC = importlib.util.spec_from_file_location("conditioned_a2_entrypoint_under_test", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QuiesceReached(RuntimeError):
    pass


class FakeCapabilityClient:
    def __init__(self):
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ConditionedA2EntrypointTests(unittest.TestCase):
    def test_vertex_execution_keeps_native_identity_path(self):
        args = SimpleNamespace(runpod_quiesce_on_success=False)
        with mock.patch.object(
            MODULE.SignedUrlStorageClient, "from_environment", return_value=None
        ), mock.patch.object(MODULE, "run_training") as run_training:
            MODULE.execute(args)

        run_training.assert_called_once_with(args, storage_client=None)

    def test_runpod_execution_requires_scoped_capability_bundle(self):
        args = SimpleNamespace(runpod_quiesce_on_success=True)
        with mock.patch.object(
            MODULE.SignedUrlStorageClient, "from_environment", return_value=None
        ), mock.patch.object(MODULE, "run_training") as run_training:
            with self.assertRaisesRegex(
                RuntimeError, "requires a validated scoped GCS capability bundle"
            ):
                MODULE.execute(args)

        run_training.assert_not_called()

    def test_training_failure_clears_capability_client(self):
        client = FakeCapabilityClient()
        args = SimpleNamespace(runpod_quiesce_on_success=False)
        with mock.patch.object(
            MODULE.SignedUrlStorageClient, "from_environment", return_value=client
        ), mock.patch.object(
            MODULE,
            "run_training",
            side_effect=RuntimeError("training failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                MODULE.execute(args)

        self.assertTrue(client.closed)

    def test_success_clears_capabilities_before_quiescing(self):
        client = FakeCapabilityClient()
        args = SimpleNamespace(runpod_quiesce_on_success=True)
        sleep_calls: list[float] = []

        def assert_capability_open(_args, *, storage_client) -> None:
            self.assertIs(storage_client, client)
            self.assertFalse(client.closed)

        def stop_at_first_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            self.assertTrue(client.closed)
            raise QuiesceReached

        output = io.StringIO()
        with mock.patch.object(
            MODULE.SignedUrlStorageClient, "from_environment", return_value=client
        ), mock.patch.object(
            MODULE,
            "run_training",
            side_effect=assert_capability_open,
        ), redirect_stdout(output):
            with self.assertRaises(QuiesceReached):
                MODULE.execute(args, sleep=stop_at_first_sleep)

        self.assertEqual(sleep_calls, [3600])
        self.assertIn(
            "DESSAMP_TRAINING_COMPLETE_WAITING_FOR_POD_TERMINATION",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
