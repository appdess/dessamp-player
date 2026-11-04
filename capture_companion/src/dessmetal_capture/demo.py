from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain import DomainError
from .store import CaptureStore
from .wav_io import inspect_wav


TRAINING_PATH_PREFIX = "NeuralAmpModeler/resources/models/parametric-training-files/"


def bootstrap_dessblock_green_demo(
    store: CaptureStore,
    *,
    repository_root: Path,
    capture_root: Path,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    capture_root = capture_root.expanduser().resolve()
    config_path = repository_root / "trainer_source" / "parametric_data_dessblock_green.json"
    input_path = repository_root / "trainer_source" / "training_data" / "input.wav"
    model_path = repository_root / "NeuralAmpModeler" / "resources" / "models" / "DessBlock-green" / "model.nam"
    if not config_path.is_file() or not input_path.is_file() or not model_path.is_file():
        raise DomainError("The DessBlock Green demo requires the checked-in trainer input, config, and model")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        outputs = config["train"]["outputs"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DomainError(f"Unable to read the DessBlock Green data config: {error}") from error
    candidate_points: list[dict[str, float]] = []
    assets: list[tuple[dict[str, float], Path]] = []
    for output in outputs:
        settings = {"gain": float(output["g_vector"][0])}
        configured_path = str(output["y_path"])
        if not configured_path.startswith(TRAINING_PATH_PREFIX):
            raise DomainError(f"Unexpected replay path in demo config: {configured_path}")
        relative_path = Path(configured_path.removeprefix(TRAINING_PATH_PREFIX))
        wav_path = (capture_root / relative_path).resolve()
        if not wav_path.is_relative_to(capture_root):
            raise DomainError("Replay path escapes the configured capture root")
        if not wav_path.is_file():
            raise DomainError(f"Demo replay WAV is unavailable: {wav_path}")
        candidate_points.append(settings)
        assets.append((settings, wav_path))
    project = store.create_project(
        name="DessBlock Green — Gain replay",
        amp_name="DessBlock Green",
        description=(
            "Development project that replays the seven existing Gain-stage captures. "
            "It validates persistence, recommendations, capture evidence, and player handoff without audio hardware."
        ),
        metadata={
            "demo_profile": "dessblock-green-gain-v1",
            "fixture_model_path": str(model_path.resolve()),
            "capture_root": str(capture_root),
        },
        controls=[
            {
                "id": "gain",
                "label": "Gain",
                "minimum": 0.2,
                "maximum": 0.8,
                "step": 0.1,
                "display_minimum": 2.0,
                "display_maximum": 8.0,
                "unit": "",
                "taper": "linear",
            }
        ],
        candidate_points=candidate_points,
        acquisition_strategy="coverage-v1",
        sample_rate=48_000,
    )
    store.set_input_signal(project["id"], input_path, inspect_wav(input_path))
    for settings, wav_path in assets:
        store.register_replay_asset(project["id"], settings, wav_path)
    return {
        "project": store.get_project(project["id"]),
        "fixture_model_path": str(model_path.resolve()),
        "replay_asset_count": len(assets),
    }
