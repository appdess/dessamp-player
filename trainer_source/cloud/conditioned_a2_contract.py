"""Immutable cloud-member contract for restartable conditioned-A2 training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


A2_ARCHITECTURE_REVISION = "conditioned-a2-global-mixer-v1"
A2_CHANNELS = {"lite": 3, "full": 8}
A2_RECEPTIVE_FIELD = 6347
UPSTREAM_RECIPE = "nam-0.13.0-config_model_packed-child"
MAX_INPUT_OBJECTS = 2048
MAX_INPUT_OBJECT_BYTES = 512 * 1024 * 1024
MAX_INPUT_BUNDLE_BYTES = 64 * 1024 * 1024 * 1024
MAX_INPUT_CONFIG_BYTES = 2 * 1024 * 1024
ARTIFACT_PURPOSES = {"production_candidate", "workflow_validation_only"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_input_manifest(
    manifest: Mapping[str, Any],
    *,
    project: str,
    bucket: str,
    run_id: str,
    control_order: Sequence[str],
    expected_train_captures: int,
) -> None:
    expected_identity = {
        "schema_version": 1,
        "project": project,
        "bucket": bucket,
        "run_id": run_id,
        "control_order": list(control_order),
        "expected_train_captures": expected_train_captures,
    }
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"Remote input manifest {field!r} does not match this submission"
            )
    objects = manifest.get("objects")
    if (
        not isinstance(objects, list)
        or not objects
        or len(objects) > MAX_INPUT_OBJECTS
    ):
        raise ValueError("Remote input manifest has no objects")
    expected_prefix = f"jobs/{run_id}/"
    seen_names: set[str] = set()
    summed_size = 0
    for index, value in enumerate(objects):
        if not isinstance(value, Mapping):
            raise ValueError(f"Remote input manifest object {index} is not an object")
        name = value.get("name")
        sha256 = value.get("sha256")
        size_bytes = value.get("size_bytes")
        if (
            not isinstance(name, str)
            or not name.startswith(expected_prefix)
            or name in seen_names
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or size_bytes > MAX_INPUT_OBJECT_BYTES
        ):
            raise ValueError(f"Remote input manifest object {index} is malformed")
        relative_name = name[len(expected_prefix) :]
        if relative_name == "configs/data.json":
            if size_bytes > MAX_INPUT_CONFIG_BYTES:
                raise ValueError("Remote data config exceeds the 2 MiB limit")
        else:
            data_name = relative_name.removeprefix("data/")
            if (
                not relative_name.startswith("data/")
                or not data_name
                or not data_name.lower().endswith(".wav")
                or Path(data_name).name != data_name
                or "\\" in data_name
                or any(ord(character) < 32 or ord(character) == 127 for character in data_name)
            ):
                raise ValueError(
                    f"Remote input manifest object is outside the runtime allowlist: {name!r}"
                )
        seen_names.add(name)
        summed_size += size_bytes
        if summed_size > MAX_INPUT_BUNDLE_BYTES:
            raise ValueError("Remote input manifest exceeds the 64 GiB bundle limit")
    required_config = f"jobs/{run_id}/configs/data.json"
    if required_config not in seen_names:
        raise ValueError("Remote input manifest does not bind configs/data.json")
    if manifest.get("total_bytes") != summed_size:
        raise ValueError("Remote input manifest total_bytes does not match its objects")


def verify_downloaded_input_bundle(
    manifest: Mapping[str, Any],
    *,
    source_prefix: str,
    config_dir: Path,
    data_dir: Path,
) -> None:
    """Verify every downloaded config/audio byte against the prepared manifest."""

    objects = manifest.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Input manifest objects must be a list")
    config_prefix = f"{source_prefix}/configs/"
    data_prefix = f"{source_prefix}/data/"
    checked = 0
    for index, value in enumerate(objects):
        if not isinstance(value, Mapping):
            raise ValueError(f"Input manifest object {index} is malformed")
        name = value.get("name")
        if not isinstance(name, str):
            raise ValueError(f"Input manifest object {index} has no name")
        if name.startswith(config_prefix):
            relative = name[len(config_prefix) :]
            root = config_dir
        elif name.startswith(data_prefix):
            relative = name[len(data_prefix) :]
            root = data_dir
        else:
            raise ValueError(f"Input manifest object is outside the prepared bundle: {name!r}")
        if not relative or Path(relative).name != relative:
            raise ValueError(f"Input manifest object must use one filename: {name!r}")
        local_path = root / relative
        if not local_path.is_file():
            raise FileNotFoundError(f"Downloaded input is missing: {local_path}")
        if local_path.stat().st_size != value.get("size_bytes"):
            raise ValueError(f"Downloaded input size differs from manifest: {local_path}")
        if file_sha256(local_path) != value.get("sha256"):
            raise ValueError(f"Downloaded input hash differs from manifest: {local_path}")
        checked += 1
    if checked == 0:
        raise ValueError("Input manifest did not verify any downloaded objects")


def build_member_contract(
    *,
    project: str,
    bucket: str,
    run_id: str,
    result_prefix: str,
    input_manifest_sha256: str,
    image_uri: str,
    child: str,
    seed: int,
    control_order: Sequence[str],
    epochs: int,
    batch_size: int,
    ny: int,
    validation_ny: int,
    num_workers: int,
    expected_train_captures: int,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    mrstft_weight: float = 0.0005,
    train_chunks_per_output: int | None = None,
    artifact_purpose: str = "production_candidate",
) -> dict[str, Any]:
    if child not in A2_CHANNELS:
        raise ValueError("child must be lite or full")
    controls = list(control_order)
    if not controls or len(controls) != len(set(controls)):
        raise ValueError("control order must contain unique names")
    if (
        train_chunks_per_output is not None
        and (
            isinstance(train_chunks_per_output, bool)
            or not isinstance(train_chunks_per_output, int)
            or not 1 <= train_chunks_per_output <= 1_000_000
        )
    ):
        raise ValueError("train_chunks_per_output must be a positive integer")
    if artifact_purpose not in ARTIFACT_PURPOSES:
        raise ValueError("artifact purpose is unsupported")
    contract = {
        "schema_version": 1,
        "identity": {
            "project": project,
            "bucket": bucket,
            "run_id": run_id,
            "result_prefix": result_prefix,
            "child": child,
            "seed": seed,
        },
        "input_bundle": {
            "source_prefix": f"jobs/{run_id}",
            "manifest_object": f"jobs/{run_id}/configs/input-manifest.json",
            "manifest_sha256": input_manifest_sha256,
            "expected_train_captures": expected_train_captures,
            "control_order": controls,
        },
        "runtime": {
            "image_uri": image_uri,
            "machine_type": machine_type,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
        },
        "architecture": {
            "revision": A2_ARCHITECTURE_REVISION,
            "child": child,
            "channels": A2_CHANNELS[child],
            "receptive_field": A2_RECEPTIVE_FIELD,
            "global_condition_size": len(controls),
        },
        "training_recipe": {
            "optimizer": {
                "class": "torch.optim.Adam",
                "lr": 0.004,
                "weight_decay": 3.17e-7,
            },
            "lr_scheduler": {"class": "ExponentialLR", "gamma": 0.994},
            "loss": {"val_loss": "esr", "mrstft_weight": mrstft_weight},
            "batch_size": batch_size,
            "ny": ny,
            "validation_ny": validation_ny,
            "num_workers": num_workers,
            "deterministic_mode": "warn",
            "upstream_recipe": UPSTREAM_RECIPE,
            "epoch_contract": {
                "mode": "absolute_target",
                "target_epochs": epochs,
            },
        },
    }
    # Omitting default-only additions preserves byte-for-byte compatibility
    # with immutable member contracts created before efficiency experiments.
    if train_chunks_per_output is not None:
        contract["training_recipe"]["chunk_selection"] = {
            "mode": "uniform-stratified-v1",
            "chunks_per_output": train_chunks_per_output,
        }
    if artifact_purpose != "production_candidate":
        contract["intent"] = {"artifact_purpose": artifact_purpose}
    return contract


def require_exact_member_contract(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise ValueError(
            "Existing member contract differs; changed data, image, architecture, seed, "
            "target epochs, optimizer, or training arguments cannot reuse this checkpoint"
        )
