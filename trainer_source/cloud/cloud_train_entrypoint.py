#!/usr/bin/env python3
"""Provider-neutral GCS worker helpers used by conditioned cloud training.

RunPod is the production compute plane. Google Cloud Storage remains the
durable input, checkpoint, and result plane; retained Vertex compatibility code
imports these same bounded transfer helpers.
"""

import os
import sys
import json
import argparse
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

# Suppress urllib3 warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")


_MODEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_CONFIG_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
MAX_GCS_DOWNLOAD_OBJECTS = 2048
MAX_GCS_OBJECT_BYTES = 512 * 1024 * 1024
MAX_GCS_DOWNLOAD_BYTES = 64 * 1024 * 1024 * 1024


def validated_model_name(value: str) -> str:
    """Accept model names that are safe as local and remote path components."""
    if not _MODEL_NAME_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "model name must contain only letters, numbers, '_' or '-', "
            "start with a letter or number, and be at most 128 characters"
        )
    return value


def normalized_gcs_prefix(source_prefix: str) -> str:
    """Return a safe GCS prefix without its optional trailing slash."""
    prefix = source_prefix.rstrip('/')
    prefix_parts = prefix.split('/')
    if (
        not prefix
        or prefix.startswith('/')
        or '\\' in prefix
        or any(part in ('', '.', '..') for part in prefix_parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in prefix)
    ):
        raise ValueError(f"Unsafe GCS source prefix: {source_prefix!r}")
    return prefix


def validated_config_name(value: str) -> str:
    """Accept a single safe filename, never a path."""
    if not _CONFIG_NAME_RE.fullmatch(value) or Path(value).name != value:
        raise argparse.ArgumentTypeError(
            "config name must be one filename using letters, numbers, '.', '_' or '-'"
        )
    return value


def rewrite_data_config_paths(config: dict, data_dir: Path) -> dict:
    """Map a portable config's input/output basenames into the downloaded bundle."""
    common = config.get("common")
    if not isinstance(common, dict) or not isinstance(common.get("x_path"), str):
        raise ValueError("Data config requires common.x_path")
    common["x_path"] = str(data_dir / Path(common["x_path"]).name)
    for split in ("train", "validation"):
        split_config = config.get(split)
        if not isinstance(split_config, dict) or not isinstance(split_config.get("outputs"), list):
            raise ValueError(f"Data config requires {split}.outputs")
        split_x_path = split_config.get("x_path")
        if split_x_path is not None:
            if not isinstance(split_x_path, str) or not split_x_path:
                raise ValueError(f"Data config {split}.x_path must be a path")
            split_config["x_path"] = str(data_dir / Path(split_x_path).name)
        for index, output in enumerate(split_config["outputs"]):
            if not isinstance(output, dict) or not isinstance(output.get("y_path"), str):
                raise ValueError(f"Data config {split}.outputs[{index}] requires y_path")
            output["y_path"] = str(data_dir / Path(output["y_path"]).name)
    return config


def referenced_audio_basenames(config: Mapping[str, Any], *, include_common: bool = True) -> list[str]:
    """Return the exact portable WAV filenames named by a training config."""

    raw_paths: list[str] = []
    common = config.get("common")
    if not isinstance(common, Mapping) or not isinstance(common.get("x_path"), str):
        raise ValueError("Data config requires common.x_path")
    if include_common:
        raw_paths.append(common["x_path"])
    for split in ("train", "validation"):
        split_config = config.get(split)
        if not isinstance(split_config, Mapping) or not isinstance(split_config.get("outputs"), list):
            raise ValueError(f"Data config requires {split}.outputs")
        split_x_path = split_config.get("x_path")
        if split_x_path is not None:
            if not isinstance(split_x_path, str) or not split_x_path:
                raise ValueError(f"Data config {split}.x_path must be a path")
            raw_paths.append(split_x_path)
        for index, output in enumerate(split_config["outputs"]):
            if not isinstance(output, Mapping) or not isinstance(output.get("y_path"), str):
                raise ValueError(f"Data config {split}.outputs[{index}] requires y_path")
            raw_paths.append(output["y_path"])
    names: list[str] = []
    for raw_path in raw_paths:
        name = Path(raw_path).name
        if (
            not _CONFIG_NAME_RE.fullmatch(name)
            or not name.lower().endswith(".wav")
            or "\\" in raw_path
        ):
            raise ValueError(f"Data config references an unsafe WAV name: {raw_path!r}")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Data config references duplicate WAV basenames")
    return names


def safe_gcs_download_path(blob_name: str, source_prefix: str, dest_dir: Path):
    """Map one object below an exact GCS prefix to a contained local path."""
    prefix = normalized_gcs_prefix(source_prefix)

    # GCS may contain objects ending in '/' that are used as directory markers.
    if blob_name == prefix or blob_name == f"{prefix}/":
        return None

    exact_prefix = f"{prefix}/"
    if not blob_name.startswith(exact_prefix):
        raise ValueError(
            f"GCS object {blob_name!r} is outside exact prefix {exact_prefix!r}"
        )

    relative_name = blob_name[len(exact_prefix):]
    is_directory_marker = relative_name.endswith('/')
    path_name = relative_name[:-1] if is_directory_marker else relative_name
    relative_parts = path_name.split('/')
    if (
        '\\' in path_name
        or any(part in ('', '.', '..') for part in relative_parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in path_name)
    ):
        raise ValueError(f"Unsafe GCS object path: {blob_name!r}")
    if is_directory_marker:
        return None

    if dest_dir.is_symlink():
        raise ValueError(f"Download directory must not be a symlink: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if dest_dir.is_symlink():
        raise ValueError(f"Download directory must not be a symlink: {dest_dir}")
    destination_root = dest_dir.resolve()
    local_path = destination_root.joinpath(*relative_parts)
    resolved_path = local_path.resolve(strict=False)

    try:
        contained = (
            os.path.commonpath((str(destination_root), str(resolved_path)))
            == str(destination_root)
        )
    except ValueError:
        contained = False
    if not contained or resolved_path != local_path:
        raise ValueError(f"GCS object resolves outside its download directory: {blob_name!r}")

    current_path = destination_root
    for index, component in enumerate(relative_parts):
        current_path /= component
        if current_path.is_symlink():
            raise ValueError(f"Download path contains a symlink: {blob_name!r}")
        if (
            index < len(relative_parts) - 1
            and current_path.exists()
            and not current_path.is_dir()
        ):
            raise ValueError(f"Download path has a file as an ancestor: {blob_name!r}")
    if local_path.exists() and local_path.is_dir():
        raise ValueError(f"Download target is an existing directory: {blob_name!r}")

    return local_path


def plan_gcs_downloads(
    bucket,
    object_specs: Sequence[Mapping[str, Any]],
    source_prefix: str,
    dest_dir: Path,
    *,
    max_object_bytes: int,
    max_total_bytes: int,
):
    """Resolve an explicit object allowlist without listing a remote prefix."""

    if not 1 <= len(object_specs) <= MAX_GCS_DOWNLOAD_OBJECTS:
        raise ValueError("GCS download allowlist is empty or exceeds the object-count limit")
    plan = []
    planned_paths = set()
    total_bytes = 0
    seen_objects: set[str] = set()
    for index, spec in enumerate(object_specs):
        if not isinstance(spec, Mapping) or not isinstance(spec.get("name"), str):
            raise ValueError(f"GCS object specification {index} is malformed")
        object_name = spec["name"]
        if object_name in seen_objects:
            raise ValueError(f"GCS object allowlist contains a duplicate: {object_name!r}")
        seen_objects.add(object_name)
        local_path = safe_gcs_download_path(object_name, source_prefix, dest_dir)
        if local_path is None:
            raise ValueError(f"GCS object allowlist contains a directory marker: {object_name!r}")
        if local_path in planned_paths:
            raise ValueError(f"Multiple GCS objects map to the same path: {object_name!r}")
        if any(
            local_path in existing_path.parents or existing_path in local_path.parents
            for existing_path in planned_paths
        ):
            raise ValueError(f"GCS objects have a file/directory collision: {object_name!r}")
        expected_size = spec.get("size_bytes")
        if expected_size is not None and (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or not 0 < expected_size <= max_object_bytes
        ):
            raise ValueError(f"GCS object expected size is invalid: {object_name!r}")
        expected_sha256 = spec.get("sha256")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise ValueError(f"GCS object expected SHA-256 is invalid: {object_name!r}")
        blob = bucket.blob(object_name)
        blob.reload()
        observed_size = blob.size
        declared_max_bytes = getattr(blob, "declared_max_bytes", None)
        if observed_size is None:
            if (
                isinstance(declared_max_bytes, bool)
                or not isinstance(declared_max_bytes, int)
                or not 0 < declared_max_bytes <= max_object_bytes
            ):
                raise ValueError(
                    f"GCS object has no bounded size capability: {object_name!r}"
                )
            planned_size = expected_size or declared_max_bytes
            if planned_size > declared_max_bytes:
                raise ValueError(
                    f"GCS object expected size exceeds its capability: {object_name!r}"
                )
        elif (
            isinstance(observed_size, bool)
            or not isinstance(observed_size, int)
            or not 0 < observed_size <= max_object_bytes
        ):
            raise ValueError(f"GCS object is missing, empty, or oversized: {object_name!r}")
        else:
            planned_size = observed_size
        if expected_size is not None and observed_size is not None and observed_size != expected_size:
            raise ValueError(f"GCS object size differs from its manifest: {object_name!r}")
        total_bytes += planned_size
        if total_bytes > max_total_bytes:
            raise ValueError("GCS download allowlist exceeds the aggregate byte limit")
        planned_paths.add(local_path)
        plan.append((blob, local_path, observed_size, expected_sha256))
    return plan


def download_from_gcs(
    bucket_name: str,
    source_prefix: str,
    dest_dir: Path,
    *,
    object_specs: Sequence[Mapping[str, Any]],
    max_object_bytes: int = MAX_GCS_OBJECT_BYTES,
    max_total_bytes: int = MAX_GCS_DOWNLOAD_BYTES,
    storage_client=None,
):
    """Download only explicitly allowed, bounded objects from GCS."""
    if storage_client is None:
        from google.cloud import storage

        storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    plan = plan_gcs_downloads(
        bucket,
        object_specs,
        source_prefix,
        dest_dir,
        max_object_bytes=max_object_bytes,
        max_total_bytes=max_total_bytes,
    )

    # Create and recheck every destination before the first remote write. A
    # failed listing therefore cannot leave a partially refreshed dataset.
    for blob, local_path, _, _ in plan:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if safe_gcs_download_path(blob.name, source_prefix, dest_dir) != local_path:
            raise ValueError(f"Download target changed during preflight: {blob.name!r}")

    for blob, local_path, observed_size, expected_sha256 in plan:
        print(f"Downloading: {blob.name} -> {local_path}")
        with tempfile.NamedTemporaryFile(
            prefix=f".{local_path.name}.",
            suffix=".tmp",
            dir=local_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            blob.download_to_filename(str(temporary_path))
            downloaded_size = temporary_path.stat().st_size
            if observed_size is not None and downloaded_size != observed_size:
                raise ValueError(f"Downloaded GCS object size changed: {blob.name!r}")
            if not 0 < downloaded_size <= max_object_bytes:
                raise ValueError(f"Downloaded GCS object is empty or oversized: {blob.name!r}")
            if expected_sha256 is not None and _sha256_file(temporary_path) != expected_sha256:
                raise ValueError(f"Downloaded GCS object hash differs from its manifest: {blob.name!r}")
            temporary_path.replace(local_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upload_to_gcs(
    bucket_name: str,
    source_dir: Path,
    dest_prefix: str,
    storage_client=None,
    *,
    relative_paths: Sequence[str | Path] | None = None,
):
    """Upload bounded directory contents to GCS.

    Conditioned-A2 callers provide an exact relative-path allowlist. The
    legacy trainer may omit it and retains its historical directory walk.
    """
    if storage_client is None:
        from google.cloud import storage

        storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    dest_prefix = normalized_gcs_prefix(dest_prefix)
    source_dir = source_dir.expanduser().resolve()
    if relative_paths is None:
        local_paths = sorted(source_dir.rglob("*"))
    else:
        if not 1 <= len(relative_paths) <= MAX_GCS_DOWNLOAD_OBJECTS:
            raise ValueError("GCS upload allowlist is empty or exceeds its object-count limit")
        local_paths = []
        seen: set[Path] = set()
        for value in relative_paths:
            raw_relative = str(value)
            relative = Path(value)
            if (
                relative.is_absolute()
                or not relative.parts
                or "\\" in raw_relative
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in raw_relative
                )
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ValueError(f"Unsafe GCS upload relative path: {value!r}")
            candidate_path = source_dir.joinpath(relative)
            if candidate_path.is_symlink():
                raise ValueError(f"Refusing to upload a symlink: {candidate_path}")
            local_path = candidate_path.resolve()
            if os.path.commonpath((str(source_dir), str(local_path))) != str(source_dir):
                raise ValueError(f"GCS upload path escapes its source directory: {value!r}")
            if local_path in seen:
                raise ValueError(f"Duplicate GCS upload relative path: {value!r}")
            seen.add(local_path)
            local_paths.append(local_path)

    for local_path in local_paths:
        if local_path.is_symlink():
            raise ValueError(f"Refusing to upload a symlink: {local_path}")
        if local_path.is_file():
            rel_path = local_path.relative_to(source_dir)
            blob_name = f"{dest_prefix}/{rel_path}"
            
            print(f"Uploading: {local_path} -> gs://{bucket_name}/{blob_name}")
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))
        elif relative_paths is not None:
            raise FileNotFoundError(f"Required GCS upload is missing: {local_path}")


def main():
    parser = argparse.ArgumentParser(description="Cloud training entrypoint")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument(
        "--model-name",
        required=True,
        type=validated_model_name,
        help="Model name (e.g., DessBlock-green)",
    )
    parser.add_argument("--epochs", type=int, default=800, help="Max epochs")
    parser.add_argument("--threshold-esr", type=float, default=0.009, help="Early stopping ESR threshold")
    parser.add_argument("--input-level", type=float, help="Measured input dBu level")
    parser.add_argument("--output-level", type=float, help="Measured output dBu level")
    parser.add_argument(
        "--source-prefix",
        help="Portable bundle prefix containing data/ and configs/ (for conditioned jobs)",
    )
    parser.add_argument("--result-prefix", help="GCS prefix for checkpoints and exports")
    parser.add_argument("--data-config-name", type=validated_config_name)
    parser.add_argument("--model-config-name", type=validated_config_name)
    parser.add_argument("--learning-config-name", type=validated_config_name)
    parser.add_argument("--checkpoint-name", type=validated_config_name)
    parser.add_argument("--seed", type=int, default=0, help="Independent ensemble seed")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ny", type=int, default=8192)
    args = parser.parse_args()

    if not 1 <= args.epochs <= 500:
        parser.error("--epochs must be between 1 and 500")
    if not 1 <= args.batch_size <= 512:
        parser.error("--batch-size must be between 1 and 512")
    if not 256 <= args.ny <= 65536:
        parser.error("--ny must be between 256 and 65536")
    if args.source_prefix and not all(
        (args.data_config_name, args.model_config_name, args.learning_config_name, args.result_prefix)
    ):
        parser.error(
            "conditioned bundle mode requires config names and --result-prefix"
        )
    if args.checkpoint_name and not args.source_prefix:
        parser.error("--checkpoint-name is only valid in conditioned bundle mode")
    if args.checkpoint_name and not args.checkpoint_name.endswith(".ckpt"):
        parser.error("--checkpoint-name must end in .ckpt")
    if (args.input_level is None) != (args.output_level is None):
        parser.error("--input-level and --output-level must be supplied together")
    
    # Setup directories
    work_dir = Path("/app/work")
    data_dir = work_dir / "data"
    output_dir = work_dir / "output"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== NAM Cloud Training: {args.model_name} ===")
    print(f"Bucket: {args.bucket}")
    print(f"Epochs: {args.epochs}, ESR threshold: {args.threshold_esr}")
    
    config_dir = work_dir / "configs"
    if args.source_prefix:
        source_prefix = normalized_gcs_prefix(args.source_prefix)
        print("\n--- Downloading portable conditioned job bundle ---")
        config_prefix = f"{source_prefix}/configs"
        download_from_gcs(
            args.bucket,
            config_prefix,
            config_dir,
            object_specs=[
                {"name": f"{config_prefix}/{name}"}
                for name in (
                    args.data_config_name,
                    args.model_config_name,
                    args.learning_config_name,
                )
            ],
            max_object_bytes=2 * 1024 * 1024,
            max_total_bytes=6 * 1024 * 1024,
        )
        data_config_path = config_dir / args.data_config_name
        model_config_path = config_dir / args.model_config_name
        learning_config_path = config_dir / args.learning_config_name
        with data_config_path.open("r", encoding="utf-8") as source:
            portable_config = json.load(source)
        data_prefix = f"{source_prefix}/data"
        download_from_gcs(
            args.bucket,
            data_prefix,
            data_dir,
            object_specs=[
                {"name": f"{data_prefix}/{name}"}
                for name in referenced_audio_basenames(portable_config)
            ],
        )
        checkpoint_path = None
        if args.checkpoint_name:
            checkpoint_dir = work_dir / "checkpoints"
            checkpoint_prefix = f"{source_prefix}/checkpoints"
            download_from_gcs(
                args.bucket,
                checkpoint_prefix,
                checkpoint_dir,
                object_specs=[
                    {"name": f"{checkpoint_prefix}/{args.checkpoint_name}"}
                ],
                max_object_bytes=256 * 1024 * 1024,
                max_total_bytes=256 * 1024 * 1024,
            )
            checkpoint_path = checkpoint_dir / args.checkpoint_name
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Required downloaded checkpoint is missing: {checkpoint_path}"
                )
    else:
        print("\n--- Downloading legacy fixed-model training data ---")
        model_data_dir = data_dir / args.model_name
        data_config_name = f"parametric_data_{args.model_name.lower().replace('-', '_')}.json"
        config_names = (data_config_name, "parametric_wavenet.json", "default.json")
        download_from_gcs(
            args.bucket,
            "configs",
            config_dir,
            object_specs=[{"name": f"configs/{name}"} for name in config_names],
            max_object_bytes=2 * 1024 * 1024,
            max_total_bytes=6 * 1024 * 1024,
        )
        data_config_path = config_dir / data_config_name
        model_config_path = config_dir / "parametric_wavenet.json"
        learning_config_path = config_dir / "default.json"
        with data_config_path.open("r", encoding="utf-8") as source:
            legacy_config = json.load(source)
        model_data_prefix = f"training-data/{args.model_name}"
        download_from_gcs(
            args.bucket,
            model_data_prefix,
            model_data_dir,
            object_specs=[
                {"name": f"{model_data_prefix}/{name}"}
                for name in referenced_audio_basenames(legacy_config, include_common=False)
            ],
        )
        download_from_gcs(
            args.bucket,
            "training-data",
            data_dir,
            object_specs=[{"name": "training-data/input.wav"}],
            max_object_bytes=MAX_GCS_OBJECT_BYTES,
            max_total_bytes=MAX_GCS_OBJECT_BYTES,
        )

    for required_path in (data_config_path, model_config_path, learning_config_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required downloaded config is missing: {required_path}")
    with data_config_path.open("r", encoding="utf-8") as source:
        config = json.load(source)
    rewrite_data_config_paths(config, data_dir if args.source_prefix else model_data_dir)
    if not args.source_prefix:
        config["common"]["x_path"] = str(data_dir / "input.wav")
        for split in ("train", "validation"):
            config[split].pop("x_path", None)
    with data_config_path.open("w", encoding="utf-8") as destination:
        json.dump(config, destination, indent=2, sort_keys=True)
    print(f"Updated config: {data_config_path}")
    
    # Run training
    print("\n--- Starting training ---")
    train_cmd = [
        "python", "-u", "/app/trainer_source/custom_train_full.py",
        "--base-outdir", str(output_dir / args.model_name),
        "--data-config", str(data_config_path),
        "--model-config", str(model_config_path),
        "--learning-config", str(learning_config_path),
        "--epochs", str(args.epochs),
        "--threshold-esr", str(args.threshold_esr),
        "--check-val-every-n-epoch", "1",
        "--batch-size", str(args.batch_size),
        "--ny", str(args.ny),
        "--seed", str(args.seed),
    ]
    if args.checkpoint_name:
        train_cmd.extend(["--checkpoint-path", str(checkpoint_path)])
    
    print(f"Command: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, cwd="/app")
    
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        sys.exit(1)
    
    # Export model
    print("\n--- Exporting model ---")
    # Find the training output directory
    training_dirs = list((output_dir / args.model_name).glob("20*"))
    if training_dirs:
        latest_training = sorted(training_dirs)[-1]
        
        export_cmd = [
            "python", "-u", "/app/trainer_source/export_model.py",
            str(latest_training),
            str(output_dir / "exported" / args.model_name)
        ]
        if args.input_level is not None:
            export_cmd[3:3] = [
                "--input-level", str(args.input_level),
                "--output-level", str(args.output_level),
            ]
        
        print(f"Export command: {' '.join(export_cmd)}")
        export_result = subprocess.run(export_cmd, cwd="/app")
        if export_result.returncode != 0:
            print(f"Export failed with return code {export_result.returncode}")
            sys.exit(1)
    
    # Upload results to GCS
    print("\n--- Uploading results ---")
    upload_to_gcs(
        args.bucket,
        output_dir,
        args.result_prefix or f"outputs/{args.model_name}"
    )
    
    print("\n=== Training complete! ===")


if __name__ == "__main__":
    main()
