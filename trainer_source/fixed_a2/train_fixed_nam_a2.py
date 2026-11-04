"""Train a fixed-setting A2 reference for comparison with a conditioned amp."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import wave
from importlib.metadata import version
from pathlib import Path

from nam.models.metadata import GearType, ToneType, UserMetadata
from nam.train import core, metadata


TARGET_TIER = "channels_8"
MIN_EPOCH = 200  # The packed checkpoint records the zero-based epoch index.
MAX_EPOCHS = 500
MAX_ESR = 0.005


def qualifies(row: dict, minimum_epoch: int, threshold: float) -> bool:
    return (
        row["submodel_name"] == TARGET_TIER
        and row["epoch"] >= minimum_epoch
        and 0.0 <= row["best_metric"] <= threshold
    )


def train(args: argparse.Namespace, user: UserMetadata):
    # NAM 0.13 builds this callback inside core.train and restores the best
    # checkpoint for each packed tier before returning the exportable model.
    upstream_callback = core._PackedBestCheckpoint

    class QualifiedPackedCheckpoint(upstream_callback):
        def on_validation_end(self, trainer, pl_module) -> None:
            if trainer.current_epoch < args.minimum_epochs:
                return
            super().on_validation_end(trainer, pl_module)
            if any(qualifies(row, args.minimum_epochs, args.threshold_esr) for row in self.best.values()):
                trainer.should_stop = True

    core._PackedBestCheckpoint = QualifiedPackedCheckpoint
    try:
        return core.train(
            str(args.input), str(args.output), str(args.work_dir),
            epochs=args.epochs, threshold_esr=None, save_plot=True, silent=True,
            modelname=args.name, local=False, user_metadata=user,
        )
    finally:
        core._PackedBestCheckpoint = upstream_callback


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def capture_info(path: Path) -> tuple[int, str]:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (48000, 1, 3):
            raise ValueError(f"Expected a mono 48 kHz PCM24 capture: {path}")
        frames = wav.getnframes()
    return frames, sha256(path)


def run(args: argparse.Namespace) -> Path:
    if version("neural-amp-modeler") != "0.13.0":
        raise ValueError("Use the locked official NAM 0.13.0 environment")
    if args.minimum_epochs < MIN_EPOCH or not args.minimum_epochs < args.epochs <= MAX_EPOCHS:
        raise ValueError("Use a minimum checkpoint epoch of at least 200 and a higher epoch count up to 500")
    if not 0 < args.threshold_esr <= MAX_ESR or not math.isfinite(args.output_trim_db):
        raise ValueError("Use a finite trim and a positive ESR threshold no higher than 0.005")
    if not args.name.strip() or args.name != Path(args.name).name or args.name in {".", ".."}:
        raise ValueError("Use a model name that is a single filename")
    if args.export_dir.exists():
        raise FileExistsError(f"Choose a new export directory: {args.export_dir}")
    input_frames, input_hash = capture_info(args.input)
    output_frames, output_hash = capture_info(args.output)
    if not input_frames or input_frames != output_frames:
        raise ValueError("The dry and return recordings must have the same nonzero frame count")
    calibration_hash = sha256(args.capture_metadata) if args.capture_metadata else None
    user = UserMetadata(
        name=args.name, modeled_by=args.modeled_by, gear_make=args.gear_make,
        gear_model=args.gear_model, gear_type=GearType(args.gear_type),
        tone_type=ToneType(args.tone_type),
    )

    # A fresh work directory precludes stale checkpoints and concurrent writers.
    args.work_dir.mkdir(parents=True, exist_ok=False)
    result = train(args, user)
    if result is None or result.model is None:
        raise RuntimeError("NAM refused the capture or returned no model")
    packed = json.loads((args.work_dir / "packed_best.json").read_text())
    high = next((row for row in packed["submodels"] if row["submodel_name"] == TARGET_TIER), None)
    if high is None or not qualifies(high, args.minimum_epochs, args.threshold_esr):
        raise RuntimeError(f"Fixed-setting reference missed the checkpoint gate: {high}")
    if (sha256(args.input), sha256(args.output)) != (input_hash, output_hash):
        raise RuntimeError("The capture files changed during training")

    capture = {
        "capture_scope": "head_only",
        "signal_format": "mono",
        "sample_rate": 48000,
        "input_wav_sha256": input_hash,
        "output_wav_sha256": output_hash,
        "output_trim_db": args.output_trim_db,
        "calibration_manifest_sha256": calibration_hash,
        "minimum_epochs": args.minimum_epochs,
        "max_epochs": args.epochs,
        "release_validation_esr_ceiling": args.threshold_esr,
    }
    args.export_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.export_dir.parent) as scratch:
        package = Path(scratch) / "export"
        package.mkdir()
        result.model.net.export(
            package,
            basename=args.name,
            user_metadata=user,
            other_metadata={
                metadata.TRAINING_KEY: result.metadata.model_dump(),
                "capture": capture,
                "target_tier": TARGET_TIER,
                "high_tier_epoch": high["epoch"],
                "high_tier_validation_esr": high["best_metric"],
            },
        )
        model_name = f"{args.name}.nam"
        summary = {
            "name": args.name,
            "architecture": "A2 SlimmableContainer",
            "model": str(args.export_dir / model_name),
            "model_sha256": sha256(package / model_name),
            "qualified": True,
            "target_tier": TARGET_TIER,
            "high_tier_epoch": high["epoch"],
            "high_tier_validation_esr": high["best_metric"],
            "aggregate_validation_esr": result.metadata.validation_esr,
            "capture": capture,
            "a2_submodel_validation": packed,
        }
        (package / f"{args.name}__training-result.json").write_text(json.dumps(summary, indent=2) + "\n")
        package.rename(args.export_dir)
    return args.export_dir / model_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Standardized dry NAM input")
    parser.add_argument("--output", required=True, type=Path, help="Validated return at one fixed setting")
    parser.add_argument("--work-dir", required=True, type=Path, help="New directory for this run")
    parser.add_argument("--export-dir", required=True, type=Path, help="New directory for the qualified export")
    parser.add_argument("--name", required=True)
    parser.add_argument("--gear-make", required=True)
    parser.add_argument("--gear-model", required=True)
    parser.add_argument("--modeled-by")
    parser.add_argument("--gear-type", choices=("amp", "pedal_amp"), default="amp")
    parser.add_argument("--tone-type", choices=[tone.value for tone in ToneType], default="hi_gain")
    parser.add_argument("--capture-metadata", type=Path)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--threshold-esr", type=float, default=MAX_ESR)
    parser.add_argument("--minimum-epochs", type=int, default=MIN_EPOCH)
    parser.add_argument("--output-trim-db", type=float, default=0.0)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
