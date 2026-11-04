#!/usr/bin/env python3
"""Cloud entrypoint for one restartable conditioned A2 child."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Callable

from trainer_source.cloud.conditioned_a2_contract import (
    A2_CHANNELS as CONTRACT_A2_CHANNELS,
    A2_RECEPTIVE_FIELD,
    build_member_contract,
    canonical_json_sha256,
    require_exact_member_contract,
    validate_input_manifest,
    verify_downloaded_input_bundle,
)
from trainer_source.cloud.cloud_train_entrypoint import (
    download_from_gcs,
    normalized_gcs_prefix,
    rewrite_data_config_paths,
    upload_to_gcs,
    validated_model_name,
)
from trainer_source.cloud.gcs_capabilities import SignedUrlStorageClient
from trainer_source.conditioned_a2.model import A2_CHANNELS, build_conditioned_a2
from trainer_source.conditioned_a2.train import ARTIFACT_PURPOSES, train_child


def parse_control_order(raw: str) -> tuple[str, ...]:
    controls = tuple(value.strip() for value in raw.split(","))
    if (
        not 1 <= len(controls) <= 12
        or len(controls) != len(set(controls))
        or any(not value for value in controls)
    ):
        raise ValueError("--control-order must contain 1 to 12 unique ids")
    return controls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--result-prefix", required=True)
    parser.add_argument("--model-name", required=True, type=validated_model_name)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--machine-type", required=True)
    parser.add_argument("--accelerator-type", required=True)
    parser.add_argument("--accelerator-count", required=True, type=int)
    parser.add_argument("--member-config-sha256", required=True)
    parser.add_argument("--control-order", required=True)
    parser.add_argument("--child", required=True, choices=sorted(A2_CHANNELS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ny", type=int, default=8192)
    parser.add_argument("--validation-ny", type=int, default=262144)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-train-captures", type=int, default=40)
    parser.add_argument("--train-chunks-per-output", type=int)
    parser.add_argument(
        "--artifact-purpose",
        choices=sorted(ARTIFACT_PURPOSES),
        default="production_candidate",
    )
    parser.add_argument("--runpod-quiesce-on-success", action="store_true")
    return parser


def run_training(args: argparse.Namespace, *, storage_client=None) -> None:
    control_order = parse_control_order(args.control_order)
    if dict(A2_CHANNELS) != CONTRACT_A2_CHANNELS:
        raise RuntimeError("Member-contract A2 channel map differs from the trainer implementation")
    architecture_probe = build_conditioned_a2(args.child, len(control_order))
    if architecture_probe.receptive_field != A2_RECEPTIVE_FIELD:
        raise RuntimeError(
            "Member-contract receptive field differs from the trainer implementation"
        )

    source_prefix = normalized_gcs_prefix(args.source_prefix)
    result_prefix = normalized_gcs_prefix(args.result_prefix)
    work_dir = Path("/app/work")
    data_dir = work_dir / "data"
    config_dir = work_dir / "configs"
    contract_dir = work_dir / "contract"
    output_dir = work_dir / "output" / args.child
    for directory in (data_dir, config_dir, contract_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"=== Conditioned A2 {args.child}: {args.model_name} ===")
    print(f"Source: gs://{args.bucket}/{source_prefix}")
    print(f"Results: gs://{args.bucket}/{result_prefix}")
    if re.fullmatch(r"[0-9a-f]{64}", args.member_config_sha256) is None:
        raise ValueError("--member-config-sha256 must be 64 lowercase hex characters")
    input_manifest_path = config_dir / "input-manifest.json"
    member_contract_path = contract_dir / "member-config.json"
    config_prefix = f"{source_prefix}/configs"
    contract_prefix = f"{result_prefix}/contract"
    download_from_gcs(
        args.bucket,
        config_prefix,
        config_dir,
        object_specs=[{"name": f"{config_prefix}/input-manifest.json"}],
        max_object_bytes=2 * 1024 * 1024,
        max_total_bytes=2 * 1024 * 1024,
        storage_client=storage_client,
    )
    download_from_gcs(
        args.bucket,
        contract_prefix,
        contract_dir,
        object_specs=[{"name": f"{contract_prefix}/member-config.json"}],
        max_object_bytes=2 * 1024 * 1024,
        max_total_bytes=2 * 1024 * 1024,
        storage_client=storage_client,
    )
    with input_manifest_path.open("r", encoding="utf-8") as source:
        input_manifest = json.load(source)
    with member_contract_path.open("r", encoding="utf-8") as source:
        member_contract = json.load(source)
    if not isinstance(input_manifest, dict) or not isinstance(member_contract, dict):
        raise ValueError("Input manifest and member contract roots must be objects")
    validate_input_manifest(
        input_manifest,
        project=args.project,
        bucket=args.bucket,
        run_id=source_prefix.removeprefix("jobs/"),
        control_order=control_order,
        expected_train_captures=args.expected_train_captures,
    )
    input_manifest_sha256 = canonical_json_sha256(input_manifest)
    expected_contract = build_member_contract(
        project=args.project,
        bucket=args.bucket,
        run_id=source_prefix.removeprefix("jobs/"),
        result_prefix=result_prefix,
        input_manifest_sha256=input_manifest_sha256,
        image_uri=args.image_uri,
        child=args.child,
        seed=args.seed,
        control_order=control_order,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        validation_ny=args.validation_ny,
        num_workers=args.num_workers,
        expected_train_captures=args.expected_train_captures,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        train_chunks_per_output=args.train_chunks_per_output,
        artifact_purpose=args.artifact_purpose,
    )
    require_exact_member_contract(member_contract, expected_contract)
    if canonical_json_sha256(member_contract) != args.member_config_sha256:
        raise ValueError("Downloaded member contract does not match the job-bound SHA-256")

    config_objects = []
    data_objects = []
    for value in input_manifest["objects"]:
        name = value["name"]
        if name.startswith(f"{config_prefix}/"):
            config_objects.append(value)
        elif name.startswith(f"{source_prefix}/data/"):
            data_objects.append(value)
        else:
            raise ValueError(f"Input object is outside the prepared runtime bundle: {name!r}")
    download_from_gcs(
        args.bucket,
        config_prefix,
        config_dir,
        object_specs=config_objects,
        max_object_bytes=2 * 1024 * 1024,
        max_total_bytes=2 * 1024 * 1024,
        storage_client=storage_client,
    )
    download_from_gcs(
        args.bucket,
        f"{source_prefix}/data",
        data_dir,
        object_specs=data_objects,
        storage_client=storage_client,
    )
    verify_downloaded_input_bundle(
        input_manifest,
        source_prefix=source_prefix,
        config_dir=config_dir,
        data_dir=data_dir,
    )

    data_config_path = config_dir / "data.json"
    if not data_config_path.is_file():
        raise FileNotFoundError(f"Downloaded bundle has no {data_config_path}")
    with data_config_path.open("r", encoding="utf-8") as source:
        data_config = json.load(source)
    metadata = data_config.get("metadata")
    declared_order = metadata.get("control_order") if isinstance(metadata, dict) else None
    if declared_order is not None and list(control_order) != declared_order:
        raise ValueError(
            "Job control order does not match the hash-bound data config metadata"
        )
    for split in ("train", "validation"):
        for index, output in enumerate(data_config.get(split, {}).get("outputs", [])):
            vector = output.get("g_vector") if isinstance(output, dict) else None
            if not isinstance(vector, list) or len(vector) != len(control_order):
                raise ValueError(
                    f"{split}.outputs[{index}] does not match --control-order"
                )
    rewrite_data_config_paths(data_config, data_dir)
    with data_config_path.open("w", encoding="utf-8") as destination:
        json.dump(data_config, destination, indent=2, sort_keys=True)

    manifest = train_child(
        data_config_path=data_config_path,
        outdir=output_dir,
        child=args.child,
        control_order=control_order,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        validation_ny=args.validation_ny,
        seed=args.seed,
        accelerator="gpu",
        num_workers=args.num_workers,
        remote_checkpoint_uri=(
            f"gs://{args.bucket}/{result_prefix}/durable/{args.member_config_sha256}"
        ),
        expected_train_captures=args.expected_train_captures,
        remote_storage_client=storage_client,
        train_chunks_per_output=args.train_chunks_per_output,
        artifact_purpose=args.artifact_purpose,
    )
    upload_to_gcs(
        args.bucket,
        output_dir,
        result_prefix,
        storage_client=storage_client,
        relative_paths=(
            "config_data.json",
            "config_learning.json",
            "config_model_export.json",
            "checkpoints/best.ckpt",
            "checkpoints/resume.ckpt",
            f"conditioned-a2-{args.child}.nam",
            "training-manifest.json",
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("=== Conditioned A2 training complete; hidden gate remains pending ===")


def quiesce_until_termination(*, sleep: Callable[[float], None] = time.sleep) -> None:
    """Keep a completed container inert until the Pod is explicitly deleted."""
    print("DESSAMP_TRAINING_COMPLETE_WAITING_FOR_POD_TERMINATION", flush=True)
    while True:
        sleep(3600)


def execute(
    args: argparse.Namespace,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    storage_client = SignedUrlStorageClient.from_environment()
    if args.runpod_quiesce_on_success and storage_client is None:
        raise RuntimeError(
            "RunPod execution requires a validated scoped GCS capability bundle"
        )
    try:
        run_training(args, storage_client=storage_client)
    finally:
        # Expiring bearer capabilities leave the environment before parsing;
        # clear their in-memory allowlists before intentional quiescence too.
        if storage_client is not None:
            storage_client.close()
    if args.runpod_quiesce_on_success:
        quiesce_until_termination(sleep=sleep)


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
