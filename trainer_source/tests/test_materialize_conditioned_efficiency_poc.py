import json
import wave
from pathlib import Path

from trainer_source.materialize_conditioned_efficiency_poc import (
    RECEPTIVE_FIELD,
    build_poc_evaluation_reservation,
    materialize_efficiency_poc,
    uniform_stratified_indices,
)


def _write_silent_wav(path: Path, frames: int) -> None:
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(48_000)
        destination.writeframes(b"\0\0" * frames)


def test_materializer_freezes_disjoint_panel_and_matched_seed_arms(tmp_path):
    frames = RECEPTIVE_FIELD - 1 + 8 * 8192
    input_path = tmp_path / "input.wav"
    _write_silent_wav(input_path, frames)
    outputs = []
    for index in range(5):
        path = tmp_path / f"output-{index}.wav"
        _write_silent_wav(path, frames)
        outputs.append(path)

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "metadata": {"control_order": ["gain", "bass"]},
                "common": {
                    "x_path": str(input_path),
                    "delay": 0,
                    "allow_unequal_lengths": False,
                },
                "train": {
                    "outputs": [
                        {
                            "y_path": str(outputs[index]),
                            "g_vector": [index / 3.0, (3 - index) / 3.0],
                        }
                        for index in range(4)
                    ]
                },
                "validation": {
                    "outputs": [
                        {"y_path": str(outputs[4]), "g_vector": [0.5, 0.5]}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    config, plan = materialize_efficiency_poc(
        source_config_path=source,
        holdout_basenames=("output-0.wav", "output-3.wav"),
        run_id="amp-efficiency-poc",
        chunks_per_output=4,
    )

    assert len(config["train"]["outputs"]) == 2
    assert len(config["validation"]["outputs"]) == 3
    train_paths = {item["y_path"] for item in config["train"]["outputs"]}
    validation_paths = {
        item["y_path"] for item in config["validation"]["outputs"]
    }
    assert train_paths.isdisjoint(validation_paths)
    assert not config["metadata"]["efficiency_poc"]["release_hidden_used"]
    assert [arm["seed"] for arm in plan["stage_1_matched_seed"]] == [110, 110]
    assert plan["stage_1_matched_seed"][1]["train_chunks_per_output"] == 4
    assert plan["audio"]["available_chunks_per_output"] == 8
    assert uniform_stratified_indices(8, 4) == (1, 3, 5, 7)

    materialized = tmp_path / "materialized.json"
    materialized.write_text(json.dumps(config), encoding="utf-8")
    reservation = build_poc_evaluation_reservation(
        config, materialized_config_path=materialized
    )
    assert reservation["target_audio_available_to_acquisition"] is False
    assert reservation["release_hidden_reservation_used"] is False
    assert len(reservation["references"]) == 3
    assert reservation["references"][0]["setting_id"] == "POC-VAL-00"


def test_materializer_rejects_unknown_or_overlapping_holdouts(tmp_path):
    frames = RECEPTIVE_FIELD - 1 + 2 * 8192
    input_path = tmp_path / "input.wav"
    train_path = tmp_path / "train.wav"
    validation_path = tmp_path / "validation.wav"
    for path in (input_path, train_path, validation_path):
        _write_silent_wav(path, frames)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "metadata": {"control_order": ["gain"]},
                "common": {"x_path": str(input_path)},
                "train": {
                    "outputs": [{"y_path": str(train_path), "g_vector": [0.5]}]
                },
                "validation": {
                    "outputs": [
                        {"y_path": str(validation_path), "g_vector": [0.5]}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        materialize_efficiency_poc(
            source_config_path=source,
            holdout_basenames=("missing.wav",),
            run_id="amp-efficiency-poc",
            chunks_per_output=1,
        )
    except ValueError as error:
        assert "not in train.outputs" in str(error)
    else:
        raise AssertionError("Unknown POC holdout was accepted")
