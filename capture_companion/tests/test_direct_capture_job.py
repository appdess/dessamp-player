from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from dessmetal_capture.api import CaptureJobManager
from dessmetal_capture.store import CaptureStore
from dessmetal_capture.wav_io import inspect_wav, write_pcm24_mono


class FakeAudioCaptureService:
    def capture_duplex(self, *, input_signal_path, output_path, input_device, output_device):
        input_info = inspect_wav(input_signal_path)
        write_pcm24_mono(
            output_path,
            np.full(input_info["frames"], 0.1, dtype=np.float32),
            input_info["sample_rate"],
        )
        return inspect_wav(output_path)


class DirectCaptureJobTests(unittest.TestCase):
    def test_background_capture_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = CaptureStore(root / "data")
            project = store.create_project(
                name="Direct",
                amp_name="Amp",
                controls=[{"id": "gain", "label": "Gain", "minimum": 0, "maximum": 1, "step": 0.5}],
            )
            input_path = root / "input.wav"
            write_pcm24_mono(input_path, np.full(4_800, 0.1, dtype=np.float32), 48_000)
            store.set_input_signal(project["id"], input_path, inspect_wav(input_path))
            recommendation = store.create_recommendation(project["id"])
            jobs = CaptureJobManager(store, FakeAudioCaptureService())
            job = jobs.start(
                project_id=project["id"],
                recommendation_id=recommendation["id"],
                input_device=1,
                output_device=2,
                notes="test",
            )
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                current = jobs.get(job["id"])
                if current["status"] not in {"queued", "recording"}:
                    break
                time.sleep(0.01)
            self.assertEqual(current["status"], "complete", current)
            captures = store.list_captures(project["id"])
            self.assertEqual(len(captures), 1)
            self.assertEqual(captures[0]["source"], "direct-interface")
            self.assertEqual(captures[0]["quality_status"], "pass")


if __name__ == "__main__":
    unittest.main()
