#!/usr/bin/env python3
"""Train one independently gated conditioned A2 child."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from trainer_source.calibration_metadata import calibration_model_metadata
from nam.train.lightning_module import LightningModule, LossConfig
from .dataset import load_capture_dataset
from .checkpoint import load_weights_only_checkpoint
from .model import (
    A2_CHANNELS,
    ConditionedA2Net,
    build_conditioned_a2,
    export_child_dict,
    write_model_dict,
)


DEFAULT_CONTROL_ORDER = (
    "amp_mvc",
    "amp_presence",
    "amp_bass",
    "amp_middle",
    "amp_treble",
    "amp_gain_1",
    "amp_gain_2",
)
_GCS_URI = re.compile(r"gs://([a-z0-9][a-z0-9._-]{1,221})/(.+)\Z")
ARTIFACT_PURPOSES = {"production_candidate", "workflow_validation_only"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class RemoteCheckpointStore:
    bucket_name: str
    prefix: str
    storage_client: Any | None = None

    @classmethod
    def from_uri(
        cls, uri: str | None, *, storage_client: Any | None = None
    ) -> "RemoteCheckpointStore | None":
        if uri is None:
            return None
        match = _GCS_URI.fullmatch(uri.rstrip("/"))
        if match is None or any(part in {"", ".", ".."} for part in match.group(2).split("/")):
            raise ValueError("Remote checkpoint URI must be a safe gs://bucket/prefix URI")
        return cls(
            bucket_name=match.group(1),
            prefix=match.group(2),
            storage_client=storage_client,
        )

    def _blob(self, name: str):
        storage_client = self.storage_client
        if storage_client is None:
            from google.cloud import storage

            storage_client = storage.Client()
        return storage_client.bucket(self.bucket_name).blob(f"{self.prefix}/{name}")

    def upload(self, local_path: Path, name: str) -> None:
        if local_path.is_symlink() or not local_path.is_file():
            raise ValueError(f"Refusing to upload invalid checkpoint: {local_path}")
        self._blob(name).upload_from_filename(str(local_path))
        print(f"Mirrored checkpoint: {local_path} -> gs://{self.bucket_name}/{self.prefix}/{name}")

    def download_if_present(self, name: str, local_path: Path) -> bool:
        blob = self._blob(name)
        capability_download = getattr(blob, "download_if_present", None)
        if callable(capability_download):
            local_path.parent.mkdir(parents=True, exist_ok=True)
            restored = capability_download(str(local_path))
            if restored:
                print(
                    f"Restored checkpoint: gs://{self.bucket_name}/{self.prefix}/{name} "
                    f"-> {local_path}"
                )
            return bool(restored)
        if not blob.exists():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        print(f"Restored checkpoint: gs://{self.bucket_name}/{self.prefix}/{name} -> {local_path}")
        return True


class DurableCheckpoint(Callback):
    """Keep restart and best checkpoints locally and, for Spot, in GCS."""

    def __init__(self, directory: Path, remote: RemoteCheckpointStore | None = None):
        super().__init__()
        self.directory = directory
        self.remote = remote
        self.best_score = math.inf
        self.best_epoch = -1

    @property
    def best_path(self) -> Path:
        return self.directory / "best.ckpt"

    @property
    def resume_path(self) -> Path:
        return self.directory / "resume.ckpt"

    @property
    def state_key(self) -> str:
        return "conditioned_a2.DurableCheckpoint"

    def state_dict(self) -> dict[str, Any]:
        return {"best_score": self.best_score, "best_epoch": self.best_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_score = float(state_dict.get("best_score", math.inf))
        self.best_epoch = int(state_dict.get("best_epoch", -1))

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get("val_loss")
        if metric is None:
            raise RuntimeError("Conditioned A2 validation did not report val_loss")
        score = float(metric.detach().cpu()) if isinstance(metric, torch.Tensor) else float(metric)
        if not math.isfinite(score):
            raise RuntimeError(f"Conditioned A2 validation produced non-finite loss {score}")

        self.directory.mkdir(parents=True, exist_ok=True)
        if score < self.best_score:
            self.best_score = score
            self.best_epoch = int(trainer.current_epoch)
            trainer.save_checkpoint(self.best_path)
            if self.remote is not None:
                self.remote.upload(self.best_path, "best.ckpt")
        trainer.save_checkpoint(self.resume_path)
        if self.remote is not None:
            self.remote.upload(self.resume_path, "resume.ckpt")


def _module(model: ConditionedA2Net, mrstft_weight: float) -> LightningModule:
    loss_config: dict[str, Any] = {"val_loss": "esr"}
    if mrstft_weight > 0.0:
        loss_config["mrstft_weight"] = mrstft_weight
    return LightningModule(
        net=model,
        optimizer_config={"lr": 0.004, "weight_decay": 3.17e-7},
        scheduler_config={
            "class": "ExponentialLR",
            "kwargs": {"gamma": 0.994},
        },
        loss_config=LossConfig.init_from_config(loss_config),
    )


def _load_best_module(
    checkpoint_path: Path,
    *,
    child: str,
    condition_size: int,
    sample_rate: int,
    mrstft_weight: float,
) -> LightningModule:
    model = build_conditioned_a2(child, condition_size, sample_rate=sample_rate)
    module = _module(model, mrstft_weight)
    checkpoint = load_weights_only_checkpoint(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no state_dict: {checkpoint_path}")
    module.load_state_dict(state, strict=True)
    module.cpu().eval()
    return module


def train_child(
    *,
    data_config_path: Path,
    outdir: Path,
    child: str,
    control_order: Sequence[str],
    epochs: int,
    batch_size: int,
    ny: int,
    validation_ny: int,
    seed: int,
    accelerator: str,
    num_workers: int,
    mrstft_weight: float = 0.0005,
    remote_checkpoint_uri: str | None = None,
    resume_checkpoint: Path | None = None,
    expected_train_captures: int | None = None,
    fast_dev_run: bool | int = False,
    remote_storage_client: Any | None = None,
    train_chunks_per_output: int | None = None,
    artifact_purpose: str = "production_candidate",
) -> dict[str, Any]:
    if child not in A2_CHANNELS:
        raise ValueError(f"child must be one of {sorted(A2_CHANNELS)}")
    if not 1 <= epochs <= 500 or not 1 <= batch_size <= 512:
        raise ValueError("epochs or batch size is outside the supported range")
    if not 256 <= ny <= 65536 or not 256 <= validation_ny <= 1048576:
        raise ValueError("training or validation chunk length is outside the supported range")
    if not 0.0 <= mrstft_weight <= 1.0 or not math.isfinite(mrstft_weight):
        raise ValueError("MRSTFT weight must be finite and between zero and one")
    if len(control_order) < 1 or len(set(control_order)) != len(control_order):
        raise ValueError("Control order must contain unique names")
    if artifact_purpose not in ARTIFACT_PURPOSES:
        raise ValueError(
            f"artifact_purpose must be one of {sorted(ARTIFACT_PURPOSES)}"
        )

    data_config_path = data_config_path.expanduser().resolve()
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with data_config_path.open("r", encoding="utf-8") as source:
        data_config = json.load(source)
    train_outputs = data_config.get("train", {}).get("outputs", [])
    if expected_train_captures is not None and len(train_outputs) != expected_train_captures:
        raise ValueError(
            f"Expected {expected_train_captures} training captures, found {len(train_outputs)}"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    model = build_conditioned_a2(child, len(control_order))
    train_dataset = load_capture_dataset(
        data_config,
        split="train",
        nx=model.receptive_field,
        ny=ny,
        expected_condition_size=len(control_order),
        chunks_per_output=train_chunks_per_output,
    )
    validation_dataset = load_capture_dataset(
        data_config,
        split="validation",
        nx=model.receptive_field,
        ny=validation_ny,
        expected_condition_size=len(control_order),
    )
    if train_dataset.sample_rate != validation_dataset.sample_rate:
        raise ValueError("Train and validation sample rates differ")
    model.sample_rate = train_dataset.sample_rate
    module = _module(model, mrstft_weight)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": accelerator == "gpu",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    remote = RemoteCheckpointStore.from_uri(
        remote_checkpoint_uri, storage_client=remote_storage_client
    )
    checkpoint_callback = DurableCheckpoint(outdir / "checkpoints", remote=remote)
    resume_path = None
    if resume_checkpoint is not None:
        resume_path = resume_checkpoint.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
    elif remote is not None:
        candidate = checkpoint_callback.resume_path
        if remote.download_if_present("resume.ckpt", candidate):
            resume_path = candidate
        remote.download_if_present("best.ckpt", checkpoint_callback.best_path)

    training_chunk_selection = train_dataset.chunk_selection
    learning_config = {
        "child": child,
        "channels": A2_CHANNELS[child],
        "global_condition_size": len(control_order),
        "control_order": list(control_order),
        "optimizer": {"lr": 0.004, "weight_decay": 3.17e-7},
        "lr_scheduler": {"class": "ExponentialLR", "gamma": 0.994},
        "loss": {"val_loss": "esr", "mrstft_weight": mrstft_weight},
        "epochs": epochs,
        "batch_size": batch_size,
        "ny": ny,
        "validation_ny": validation_ny,
        "seed": seed,
        "accelerator": accelerator,
        "num_workers": num_workers,
        "deterministic_mode": "warn",
        "upstream_recipe": "nam-0.13.0-config_model_packed-child",
        "artifact_purpose": artifact_purpose,
        "training_chunk_selection": training_chunk_selection,
        "training_examples_per_epoch": len(train_dataset),
        "optimizer_steps_per_epoch": len(train_loader),
    }
    _atomic_json(outdir / "config_data.json", data_config)
    _atomic_json(outdir / "config_learning.json", learning_config)
    _atomic_json(outdir / "config_model_export.json", model.export_config())

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=outdir,
        # MRSTFT uses reflection_pad1d_backward, for which CUDA has no strict
        # deterministic implementation. Preserve fixed seeds/data order while
        # allowing PyTorch to run that operation and emit an explicit warning.
        deterministic="warn",
        log_every_n_steps=20,
        enable_checkpointing=False,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=str(resume_path) if resume_path is not None else None,
        weights_only=True,
    )

    if fast_dev_run:
        # The fast-dev loop suppresses normal validation-end checkpointing.
        checkpoint_callback.directory.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint_callback.best_path)
        checkpoint_callback.best_score = float(trainer.callback_metrics.get("val_loss", math.nan))
        checkpoint_callback.best_epoch = int(trainer.current_epoch)
    if not checkpoint_callback.best_path.is_file():
        raise RuntimeError("Training completed without a durable best checkpoint")

    best = _load_best_module(
        checkpoint_callback.best_path,
        child=child,
        condition_size=len(control_order),
        sample_rate=train_dataset.sample_rate,
        mrstft_weight=mrstft_weight,
    )
    external_gate = (
        {"threshold": 0.035, "passed": False, "status": "pending"}
        if artifact_purpose == "production_candidate"
        else {
            "threshold": 0.035,
            "passed": False,
            "status": "not_used_for_workflow_validation",
        }
    )
    child_dict = export_child_dict(
        best.net,
        control_order,
        metadata={
            **calibration_model_metadata(data_config),
            "training_seed": seed,
            "training_captures": len(train_outputs),
            "training_chunk_selection": training_chunk_selection,
            "artifact_purpose": artifact_purpose,
            "best_internal_validation_esr": checkpoint_callback.best_score,
            "best_epoch": checkpoint_callback.best_epoch,
            "external_hidden_gate": external_gate,
        },
    )
    model_path = outdir / f"conditioned-a2-{child}.nam"
    write_model_dict(child_dict, model_path)

    manifest_status = (
        "trained_hidden_gate_pending"
        if artifact_purpose == "production_candidate"
        else "workflow_validation_only_external_evaluation_pending"
    )
    manifest = {
        "schema_version": 1,
        "status": manifest_status,
        "artifact_purpose": artifact_purpose,
        "release_eligible": False,
        "child": child,
        "channels": A2_CHANNELS[child],
        "control_order": list(control_order),
        "training_captures": len(train_outputs),
        "validation_captures": len(data_config.get("validation", {}).get("outputs", [])),
        "receptive_field": model.receptive_field,
        "sample_rate": train_dataset.sample_rate,
        "training_chunk_selection": training_chunk_selection,
        "training_examples_per_epoch": len(train_dataset),
        "optimizer_steps_per_epoch": len(train_loader),
        "best_internal_validation_esr": checkpoint_callback.best_score,
        "best_epoch": checkpoint_callback.best_epoch,
        "hidden_gate": {
            "all_settings_below": 0.035,
            "status": (
                "pending"
                if artifact_purpose == "production_candidate"
                else "not_used_for_workflow_validation"
            ),
        },
        "artifacts": {
            "model": {"path": model_path.name, "sha256": _sha256(model_path)},
            "best_checkpoint": {
                "path": str(checkpoint_callback.best_path.relative_to(outdir)),
                "sha256": _sha256(checkpoint_callback.best_path),
            },
            "data_config": {
                "path": "config_data.json",
                "sha256": _sha256(outdir / "config_data.json"),
            },
        },
    }
    _atomic_json(outdir / "training-manifest.json", manifest)
    if remote is not None:
        remote.upload(model_path, model_path.name)
        remote.upload(outdir / "training-manifest.json", "training-manifest.json")

    train_dataset.teardown()
    validation_dataset.teardown()
    return manifest


def _control_order(value: str) -> tuple[str, ...]:
    controls = tuple(item.strip() for item in value.split(","))
    if not controls or any(not item for item in controls) or len(set(controls)) != len(controls):
        raise argparse.ArgumentTypeError("Control order must be unique comma-separated names")
    return controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--child", required=True, choices=sorted(A2_CHANNELS))
    parser.add_argument("--control-order", type=_control_order, default=DEFAULT_CONTROL_ORDER)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ny", type=int, default=8192)
    parser.add_argument("--validation-ny", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mrstft-weight", type=float, default=0.0005)
    parser.add_argument("--remote-checkpoint-uri")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--expected-train-captures", type=int)
    parser.add_argument("--train-chunks-per-output", type=int)
    parser.add_argument(
        "--artifact-purpose",
        choices=sorted(ARTIFACT_PURPOSES),
        default="production_candidate",
    )
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    manifest = train_child(
        data_config_path=args.data_config,
        outdir=args.outdir,
        child=args.child,
        control_order=args.control_order,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        validation_ny=args.validation_ny,
        seed=args.seed,
        accelerator=args.accelerator,
        num_workers=args.num_workers,
        mrstft_weight=args.mrstft_weight,
        remote_checkpoint_uri=args.remote_checkpoint_uri,
        resume_checkpoint=args.resume_checkpoint,
        expected_train_captures=args.expected_train_captures,
        train_chunks_per_output=args.train_chunks_per_output,
        artifact_purpose=args.artifact_purpose,
        fast_dev_run=args.fast_dev_run,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
