import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from trainer_source.conditioned_a2.dataset import (
    ConditionedCaptureDataset,
    load_capture_dataset,
)
from trainer_source.conditioned_a2.model import (
    build_conditioned_a2,
    export_child_dict,
    pack_slimmable_container,
)
from trainer_source.conditioned_a2.train import (
    DEFAULT_CONTROL_ORDER,
    RemoteCheckpointStore,
    train_child,
)


def _write_wav(path: Path, samples: np.ndarray, rate: int = 48_000) -> None:
    pcm = np.clip(samples, -0.999, 0.999)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(rate)
        destination.writeframes(pcm.tobytes())


def test_remote_checkpoint_store_uses_injected_scoped_storage_client(tmp_path):
    operations = []

    class Blob:
        def __init__(self, name):
            self.name = name

        def download_if_present(self, filename):
            Path(filename).write_bytes(b"resume")
            operations.append(("download", self.name))
            return True

        def upload_from_filename(self, filename):
            operations.append(("upload", self.name, Path(filename).read_bytes()))

    bucket = type("Bucket", (), {"blob": lambda _self, name: Blob(name)})()
    client = type("Client", (), {"bucket": lambda _self, _name: bucket})()
    store = RemoteCheckpointStore.from_uri(
        "gs://dessamp-evidence/jobs/run/results/member/durable/hash",
        storage_client=client,
    )
    assert store is not None
    restored = tmp_path / "resume.ckpt"
    assert store.download_if_present("resume.ckpt", restored)
    assert restored.read_bytes() == b"resume"
    store.upload(restored, "resume.ckpt")
    assert operations == [
        ("download", "jobs/run/results/member/durable/hash/resume.ckpt"),
        (
            "upload",
            "jobs/run/results/member/durable/hash/resume.ckpt",
            b"resume",
        ),
    ]


@pytest.mark.parametrize("child,channels", (("lite", 3), ("full", 8)))
def test_conditioned_a2_forward_and_export(child, channels):
    model = build_conditioned_a2(child, 7, sample_rate=48_000)
    x = torch.linspace(-0.1, 0.1, model.receptive_field + 31).repeat(2, 1)
    controls = torch.tensor(
        [[0.0] * 7, [1.0] * 7],
        dtype=torch.float32,
    )
    output = model(x, controls, pad_start=False)
    assert output.shape == (2, 32)
    assert torch.isfinite(output).all()
    assert not torch.equal(output[0], output[1])

    exported = export_child_dict(model, DEFAULT_CONTROL_ORDER)
    assert exported["architecture"] == "WaveNet"
    assert exported["config"]["global_condition_size"] == 7
    assert exported["config"]["layers"][0]["condition_size"] == 1
    assert exported["config"]["layers"][0]["global_condition_size"] == 7
    assert exported["metadata"]["a2_channels"] == channels
    assert exported["metadata"]["receptive_field"] == model.receptive_field


def test_uniform_stratified_chunks_cover_the_full_stimulus_deterministically():
    x = torch.arange(18, dtype=torch.float32)
    output = torch.arange(100, 118, dtype=torch.float32)
    dataset = ConditionedCaptureDataset(
        x=x,
        outputs=[output],
        conditions=[torch.tensor([0.5])],
        nx=3,
        ny=2,
        sample_rate=8,
        split="train",
        chunks_per_output=4,
    )

    assert len(dataset) == 4
    assert [int(dataset[index][0][0]) for index in range(len(dataset))] == [2, 6, 10, 14]
    assert dataset.chunk_selection == {
        "mode": "uniform-stratified-v1",
        "available_chunks_per_output": 8,
        "selected_chunks_per_output": 4,
        "selected_fraction": 0.5,
        "selected_audio_seconds_per_output": 1.0,
        "first_step_index": 1,
        "last_step_index": 7,
        "step_indices_sha256": "23040c3e2adb5784b0dc91327d1b128ddf23cc342020f50c75b1a107bcd9af02",
    }
    dataset.teardown()


def test_uniform_stratified_chunk_count_cannot_exceed_available_chunks():
    with pytest.raises(ValueError, match="available chunk count"):
        ConditionedCaptureDataset(
            x=torch.arange(18, dtype=torch.float32),
            outputs=[torch.arange(18, dtype=torch.float32)],
            conditions=[torch.tensor([0.5])],
            nx=3,
            ny=2,
            sample_rate=8,
            split="train",
            chunks_per_output=9,
        )


def test_conditioned_a2_container_requires_matching_children():
    lite = export_child_dict(
        build_conditioned_a2("lite", 7, sample_rate=48_000), DEFAULT_CONTROL_ORDER
    )
    full = export_child_dict(
        build_conditioned_a2("full", 7, sample_rate=48_000), DEFAULT_CONTROL_ORDER
    )
    container = pack_slimmable_container(lite, full)
    assert container["architecture"] == "SlimmableContainer"
    assert [entry["max_value"] for entry in container["config"]["submodels"]] == [
        0.5,
        1.0,
    ]

    full["metadata"]["control_order"] = list(reversed(DEFAULT_CONTROL_ORDER))
    with pytest.raises(ValueError, match="different control orders"):
        pack_slimmable_container(lite, full)


def test_conditioned_a2_dataset_and_fast_dev_training(tmp_path):
    receptive_field = build_conditioned_a2("lite", 7).receptive_field
    sample_count = receptive_field + 1024
    phase = np.arange(sample_count, dtype=np.float32) / 48_000.0
    x = 0.05 * np.sin(2.0 * np.pi * 197.0 * phase)
    y_train = np.tanh(2.0 * x).astype(np.float32)
    y_validation = np.tanh(2.1 * x).astype(np.float32)
    x_path = tmp_path / "input.wav"
    train_path = tmp_path / "train.wav"
    validation_path = tmp_path / "validation.wav"
    _write_wav(x_path, x)
    _write_wav(train_path, y_train)
    _write_wav(validation_path, y_validation)

    data_config = {
        "common": {
            "x_path": str(x_path),
            "delay": 0,
            "input_gain": 0.0,
            "allow_unequal_lengths": False,
        },
        "train": {
            "outputs": [
                {"y_path": str(train_path), "g_vector": [0.5] * 7},
            ]
        },
        "validation": {
            "outputs": [
                {"y_path": str(validation_path), "g_vector": [0.55] * 7},
            ]
        },
    }
    dataset = load_capture_dataset(
        data_config,
        split="train",
        nx=receptive_field,
        ny=256,
    )
    x_chunk, condition, y_chunk = dataset[0]
    assert x_chunk.shape == (receptive_field + 255,)
    assert condition.shape == (7,)
    assert y_chunk.shape == (256,)
    dataset.teardown()

    config_path = tmp_path / "data.json"
    config_path.write_text(json.dumps(data_config), encoding="utf-8")
    outdir = tmp_path / "output"
    manifest = train_child(
        data_config_path=config_path,
        outdir=outdir,
        child="lite",
        control_order=DEFAULT_CONTROL_ORDER,
        epochs=1,
        batch_size=1,
        ny=256,
        validation_ny=256,
        seed=123,
        accelerator="cpu",
        num_workers=0,
        mrstft_weight=0.0,
        expected_train_captures=1,
        train_chunks_per_output=1,
        artifact_purpose="workflow_validation_only",
        fast_dev_run=True,
    )
    assert manifest["status"] == "workflow_validation_only_external_evaluation_pending"
    assert manifest["artifact_purpose"] == "workflow_validation_only"
    assert manifest["training_chunk_selection"]["mode"] == "uniform-stratified-v1"
    assert manifest["training_chunk_selection"]["selected_chunks_per_output"] == 1
    model_path = outdir / "conditioned-a2-lite.nam"
    assert model_path.is_file()
    exported = json.loads(model_path.read_text(encoding="utf-8"))
    assert exported["config"]["global_condition_size"] == 7
    assert exported["metadata"]["artifact_purpose"] == "workflow_validation_only"


def test_conditioned_a2_split_specific_inputs_can_reuse_control_vectors(tmp_path):
    receptive_field = build_conditioned_a2("lite", 1).receptive_field
    sample_count = receptive_field + 512
    phase = np.arange(sample_count, dtype=np.float32) / 48_000.0
    train_x = 0.04 * np.sin(2.0 * np.pi * 197.0 * phase)
    validation_x = 0.03 * np.sin(2.0 * np.pi * 311.0 * phase)
    train_y = np.tanh(2.0 * train_x).astype(np.float32)
    validation_y = np.tanh(2.0 * validation_x).astype(np.float32)
    train_x_path = tmp_path / "train-input.wav"
    validation_x_path = tmp_path / "validation-input.wav"
    train_y_path = tmp_path / "train-output.wav"
    validation_y_path = tmp_path / "validation-output.wav"
    for path, samples in (
        (train_x_path, train_x),
        (validation_x_path, validation_x),
        (train_y_path, train_y),
        (validation_y_path, validation_y),
    ):
        _write_wav(path, samples)

    config = {
        "common": {
            "x_path": str(train_x_path),
            "delay": 0,
            "input_gain": 0.0,
            "allow_unequal_lengths": False,
        },
        "train": {
            "x_path": str(train_x_path),
            "outputs": [{"y_path": str(train_y_path), "g_vector": [0.5]}],
        },
        "validation": {
            "x_path": str(validation_x_path),
            "outputs": [
                {"y_path": str(validation_y_path), "g_vector": [0.5]}
            ],
        },
    }
    train = load_capture_dataset(
        config, split="train", nx=receptive_field, ny=256, expected_condition_size=1
    )
    validation = load_capture_dataset(
        config,
        split="validation",
        nx=receptive_field,
        ny=256,
        expected_condition_size=1,
    )
    train_x_chunk, train_condition, _ = train[0]
    validation_x_chunk, validation_condition, _ = validation[0]
    assert not torch.equal(train_x_chunk, validation_x_chunk)
    assert torch.equal(train_condition, validation_condition)
    train.teardown()
    validation.teardown()
