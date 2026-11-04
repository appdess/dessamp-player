#!/usr/bin/env python3
"""Prepare and submit budget-bounded conditioned-A2 Vertex jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from trainer_source.cloud.conditioned_a2_contract import (
        ARTIFACT_PURPOSES,
        build_member_contract,
        canonical_json_bytes,
        canonical_json_sha256,
        require_exact_member_contract,
        validate_input_manifest,
    )
    from trainer_source.cloud.conditioned_vertex_job import (
        InputObject,
        PreflightError,
        bounded_cost,
        decimal_eur,
        load_training_inputs,
        require_project_confirmation,
        run_checked,
    )
except ModuleNotFoundError:
    from conditioned_a2_contract import (  # type: ignore[no-redef]
        ARTIFACT_PURPOSES,
        build_member_contract,
        canonical_json_bytes,
        canonical_json_sha256,
        require_exact_member_contract,
        validate_input_manifest,
    )
    from conditioned_vertex_job import (  # type: ignore[no-redef]
        InputObject,
        PreflightError,
        bounded_cost,
        decimal_eur,
        load_training_inputs,
        require_project_confirmation,
        run_checked,
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
MAX_WRITER_LEASE_BYTES = 32 * 1024
MAX_WRITER_LEASE_DURATION = timedelta(hours=25)
WRITER_LEASE_NONCE_RE = re.compile(r"\A[0-9a-f]{32}\Z", re.ASCII)
WRITER_LEASE_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z", re.ASCII)


def parse_control_order(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_CONTROL_ORDER
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    controls = tuple(str(value).strip() for value in values)
    if (
        not 1 <= len(controls) <= 12
        or len(controls) != len(set(controls))
        or any(
            not control
            or len(control) > 32
            or not control[0].islower()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in control)
            for control in controls
        )
    ):
        raise PreflightError(
            "Control order must contain 1 to 12 unique lowercase control ids"
        )
    return controls


def control_order_from_config(
    config: dict[str, Any], requested: str | Sequence[str] | None = None
) -> tuple[str, ...]:
    metadata = config.get("metadata")
    declared = metadata.get("control_order") if isinstance(metadata, dict) else None
    if declared is not None and not isinstance(declared, list):
        raise PreflightError("Data config metadata.control_order must be a list")
    declared_order = parse_control_order(declared) if declared is not None else None
    requested_order = parse_control_order(requested) if requested is not None else None
    if declared_order is not None and requested_order is not None and declared_order != requested_order:
        raise PreflightError("Requested control order does not match data config metadata")
    return declared_order or requested_order or DEFAULT_CONTROL_ORDER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or not 0 < path.stat().st_size <= 2 * 1024 * 1024:
        raise PreflightError(f"Config is missing, empty or oversized: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"Invalid JSON config {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError("Data config root must be an object")
    return value


def plan_a2_uploads(
    *,
    run_id: str,
    data_config: Path,
    expected_train_captures: int = 40,
    control_order: str | Sequence[str] | None = None,
    data_root: Path | None = None,
) -> list[InputObject]:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_id):
        raise PreflightError("Run ID must use lowercase letters, numbers and hyphens")
    if not 1 <= expected_train_captures <= 512:
        raise PreflightError("Expected training capture count must be between 1 and 512")
    config_path = data_config.expanduser().resolve()
    config = _read_json(config_path)
    controls = control_order_from_config(config, control_order)
    train_outputs = config.get("train", {}).get("outputs")
    validation_outputs = config.get("validation", {}).get("outputs")
    if not isinstance(train_outputs, list) or len(train_outputs) != expected_train_captures:
        raise PreflightError(
            f"Expected exactly {expected_train_captures} training captures"
        )
    if not isinstance(validation_outputs, list) or not validation_outputs:
        raise PreflightError("A2 bundle requires at least one internal validation capture")
    for split, outputs in (("train", train_outputs), ("validation", validation_outputs)):
        for index, output in enumerate(outputs):
            vector = output.get("g_vector") if isinstance(output, dict) else None
            if (
                not isinstance(vector, list)
                or len(vector) != len(controls)
                or any(not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0 for value in vector)
            ):
                raise PreflightError(
                    f"{split}.outputs[{index}] does not contain {len(controls)} normalized controls"
                )

    entries = [(config_path, f"jobs/{run_id}/configs/data.json")]
    entries.extend(
        (path, f"jobs/{run_id}/data/{path.name}")
        for path in load_training_inputs(config_path, data_root=data_root)
    )
    return [
        InputObject(
            local_path=path,
            object_name=object_name,
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
        )
        for path, object_name in entries
    ]


def build_job_spec(
    *,
    project: str,
    bucket: str,
    run_id: str,
    image_uri: str,
    service_account: str,
    child: str,
    seed: int,
    timeout_minutes: int,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    epochs: int,
    batch_size: int,
    ny: int,
    validation_ny: int,
    num_workers: int,
    expected_train_captures: int,
    scheduling_strategy: str,
    member_config_sha256: str,
    control_order: Sequence[str] = DEFAULT_CONTROL_ORDER,
    train_chunks_per_output: int | None = None,
    artifact_purpose: str = "production_candidate",
) -> dict[str, Any]:
    if child not in {"lite", "full"}:
        raise PreflightError("Child must be lite or full")
    if scheduling_strategy not in {"standard", "spot"}:
        raise PreflightError("Scheduling strategy must be standard or spot")
    image_pattern = re.compile(
        rf"^[a-z0-9-]+-docker\.pkg\.dev/{re.escape(project)}/[^@]+@sha256:[0-9a-f]{{64}}$"
    )
    if image_pattern.fullmatch(image_uri) is None:
        raise PreflightError(
            "Image must be a digest-pinned artifact in the confirmed project's registry"
        )
    if not service_account.endswith(f"@{project}.iam.gserviceaccount.com"):
        raise PreflightError("Service account must belong to the confirmed project")
    if not 1 <= epochs <= 500 or not 1 <= batch_size <= 512:
        raise PreflightError("Epoch or batch count is outside the supported range")
    if not 256 <= ny <= 65536 or not 256 <= validation_ny <= 1048576:
        raise PreflightError("Training or validation chunk size is outside the supported range")
    if not 0 <= num_workers <= 16 or not 1 <= accelerator_count <= 4:
        raise PreflightError("Worker or accelerator count is outside the supported range")
    if not 1 <= expected_train_captures <= 512:
        raise PreflightError("Expected training capture count must be between 1 and 512")
    controls = parse_control_order(control_order)
    if re.fullmatch(r"[0-9a-f]{64}", member_config_sha256) is None:
        raise PreflightError("Member config SHA-256 must be 64 lowercase hex characters")

    source_prefix = f"jobs/{run_id}"
    result_prefix = f"{source_prefix}/results/{child}-seed-{seed}"
    container_args = [
        "--bucket",
        bucket,
        "--source-prefix",
        source_prefix,
        "--result-prefix",
        result_prefix,
        "--model-name",
        f"dessamp-a2-{child}",
        "--project",
        project,
        "--image-uri",
        image_uri,
        "--machine-type",
        machine_type,
        "--accelerator-type",
        accelerator_type,
        "--accelerator-count",
        str(accelerator_count),
        "--member-config-sha256",
        member_config_sha256,
        "--control-order",
        ",".join(controls),
        "--child",
        child,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--ny",
        str(ny),
        "--validation-ny",
        str(validation_ny),
        "--num-workers",
        str(num_workers),
        "--expected-train-captures",
        str(expected_train_captures),
    ]
    if train_chunks_per_output is not None:
        container_args.extend(
            ["--train-chunks-per-output", str(train_chunks_per_output)]
        )
    if artifact_purpose != "production_candidate":
        container_args.extend(["--artifact-purpose", artifact_purpose])
    scheduling: dict[str, Any] = {
        "timeout": f"{timeout_minutes * 60}s",
        # Vertex can reprovision a preempted Spot worker under the same Custom
        # Job. The entrypoint then restores the exact durable member checkpoint.
        # Keep standard jobs single-attempt so this retry policy cannot broaden
        # their execution envelope unexpectedly.
        "restartJobOnWorkerRestart": scheduling_strategy == "spot",
    }
    if scheduling_strategy == "spot":
        scheduling["strategy"] = "SPOT"
    return {
        "serviceAccount": service_account,
        "workerPoolSpecs": [
            {
                "machineSpec": {
                    "machineType": machine_type,
                    "acceleratorType": accelerator_type,
                    "acceleratorCount": accelerator_count,
                },
                "replicaCount": 1,
                "containerSpec": {"imageUri": image_uri, "args": container_args},
            }
        ],
        "scheduling": scheduling,
        "baseOutputDirectory": {
            "outputUriPrefix": f"gs://{bucket}/{result_prefix}/vertex-output"
        },
    }


def read_remote_input_manifest(*, project: str, bucket: str, run_id: str) -> dict[str, Any]:
    object_uri = f"gs://{bucket}/jobs/{run_id}/configs/input-manifest.json"
    response = run_checked(
        [
            "gcloud",
            "storage",
            "cat",
            object_uri,
            f"--project={project}",
        ]
    )
    if not 0 < len(response.stdout.encode("utf-8")) <= 2 * 1024 * 1024:
        raise PreflightError(f"Remote input manifest is empty or oversized: {object_uri}")
    try:
        value = json.loads(response.stdout)
    except json.JSONDecodeError as error:
        raise PreflightError(f"Remote input manifest is invalid JSON: {object_uri}") from error
    if not isinstance(value, dict):
        raise PreflightError("Remote input manifest root must be an object")
    return value


def register_member_contract(
    *,
    project: str,
    bucket: str,
    result_prefix: str,
    contract: dict[str, Any],
    resume_existing_member: bool,
) -> dict[str, str]:
    """Create one immutable member contract or explicitly reuse an exact match."""

    object_uri = f"gs://{bucket}/{result_prefix}/contract/member-config.json"
    payload = canonical_json_bytes(contract)
    contract_sha256 = canonical_json_sha256(contract)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".json") as config_file:
        config_file.write(payload)
        config_file.flush()
        try:
            run_checked(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    config_file.name,
                    object_uri,
                    "--if-generation-match=0",
                    f"--project={project}",
                    "--quiet",
                ]
            )
        except subprocess.CalledProcessError as create_error:
            try:
                response = run_checked(
                    [
                        "gcloud",
                        "storage",
                        "cat",
                        object_uri,
                        f"--project={project}",
                    ]
                )
                existing = json.loads(response.stdout)
                if not isinstance(existing, dict):
                    raise ValueError("member contract root is not an object")
                require_exact_member_contract(existing, contract)
            except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as read_error:
                if isinstance(read_error, ValueError) and "Existing member contract differs" in str(read_error):
                    raise PreflightError(str(read_error)) from read_error
                raise PreflightError(
                    f"Could not create or verify immutable member contract at {object_uri}"
                ) from create_error
            if not resume_existing_member:
                raise PreflightError(
                    "This exact member contract already exists. Pass --resume-existing-member "
                    "only after confirming no job is currently writing that member prefix."
                )
            return {
                "state": "exact-contract-reused",
                "uri": object_uri,
                "sha256": contract_sha256,
            }
    return {
        "state": "immutable-contract-created",
        "uri": object_uri,
        "sha256": contract_sha256,
    }


def _writer_lease_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise PreflightError("Writer lease timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _parse_writer_lease_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not 1 <= len(value) <= 40:
        raise PreflightError("Existing writer lease has an invalid expiry")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise PreflightError("Existing writer lease has an invalid expiry") from None
    if parsed.tzinfo is None:
        raise PreflightError("Existing writer lease has an invalid expiry")
    return parsed.astimezone(timezone.utc)


def _describe_writer_lease(
    *, project: str, object_uri: str
) -> tuple[str, int]:
    response = run_checked(
        [
            "gcloud",
            "storage",
            "objects",
            "describe",
            object_uri,
            f"--project={project}",
            "--format=json",
            "--quiet",
        ]
    )
    try:
        metadata = json.loads(response.stdout)
        generation = str(metadata["generation"])
        size_bytes = int(metadata["size"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise PreflightError("Existing writer lease metadata is invalid") from None
    if not generation.isascii() or not generation.isdigit() or len(generation) > 32:
        raise PreflightError("Existing writer lease generation is invalid")
    if not 0 < size_bytes <= MAX_WRITER_LEASE_BYTES:
        raise PreflightError("Existing writer lease is empty or oversized")
    return generation, size_bytes


def _read_stable_writer_lease(
    *, project: str, object_uri: str
) -> tuple[dict[str, Any], str]:
    """Read one exact GCS generation with a bounded, double-checked snapshot."""

    before_generation, before_size = _describe_writer_lease(
        project=project, object_uri=object_uri
    )
    response = run_checked(
        [
            "gcloud",
            "storage",
            "cat",
            object_uri,
            f"--range=0-{before_size - 1}",
            f"--project={project}",
        ]
    )
    after_generation, after_size = _describe_writer_lease(
        project=project, object_uri=object_uri
    )
    encoded = response.stdout.encode("utf-8")
    if (
        before_generation != after_generation
        or before_size != after_size
        or len(encoded) != before_size
    ):
        raise PreflightError("Writer lease changed while it was being read")
    try:
        value = json.loads(response.stdout)
    except json.JSONDecodeError:
        raise PreflightError("Existing writer lease is invalid JSON") from None
    if not isinstance(value, dict):
        raise PreflightError("Existing writer lease root must be an object")
    return value, before_generation


def acquire_member_writer_lease(
    *,
    project: str,
    bucket: str,
    result_prefix: str,
    member_contract_sha256: str,
    attempt_nonce: str,
    pod_name: str,
    now: datetime,
    expires_at: datetime,
) -> dict[str, str]:
    """Atomically serialize one bounded RunPod writer for a member prefix.

    A fixed object name plus a generation precondition prevents two helpers
    from launching the same member concurrently. An expired exact-contract
    lease is reclaimed only with a compare-and-swap against the generation
    that was read. The lease intentionally remains active after submission;
    its bounded expiry is after the provider's deletion deadline.
    """

    if WRITER_LEASE_SHA256_RE.fullmatch(member_contract_sha256) is None:
        raise PreflightError("Writer lease requires a valid member contract hash")
    if WRITER_LEASE_NONCE_RE.fullmatch(attempt_nonce) is None:
        raise PreflightError("Writer lease requires a 32-character attempt nonce")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,190}", pod_name):
        raise PreflightError("Writer lease Pod name contains unsupported characters")
    now_utc = now.astimezone(timezone.utc) if now.tzinfo is not None else None
    expires_utc = (
        expires_at.astimezone(timezone.utc) if expires_at.tzinfo is not None else None
    )
    if (
        now_utc is None
        or expires_utc is None
        or expires_utc <= now_utc
        or expires_utc - now_utc > MAX_WRITER_LEASE_DURATION
    ):
        raise PreflightError("Writer lease expiry is invalid or exceeds 25 hours")

    object_uri = f"gs://{bucket}/{result_prefix}/control/active-writer-lease.json"
    payload = {
        "schema_version": 1,
        "provider": "runpod",
        "member_contract_sha256": member_contract_sha256,
        "attempt_nonce": attempt_nonce,
        "pod_name": pod_name,
        "created_at": _writer_lease_timestamp(now_utc),
        "expires_at": _writer_lease_timestamp(expires_utc),
    }
    encoded = canonical_json_bytes(payload)
    if len(encoded) > MAX_WRITER_LEASE_BYTES:
        raise PreflightError("Writer lease payload is oversized")

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".json") as lease_file:
        lease_file.write(encoded)
        lease_file.flush()
        create_command = [
            "gcloud",
            "storage",
            "cp",
            lease_file.name,
            object_uri,
            "--if-generation-match=0",
            f"--project={project}",
            "--quiet",
        ]
        try:
            run_checked(create_command)
        except subprocess.CalledProcessError as create_error:
            try:
                existing, generation = _read_stable_writer_lease(
                    project=project, object_uri=object_uri
                )
                if (
                    existing.get("schema_version") != 1
                    or existing.get("provider") != "runpod"
                    or existing.get("member_contract_sha256")
                    != member_contract_sha256
                ):
                    raise PreflightError(
                        "Existing writer lease is not bound to this exact member contract"
                    )
                existing_expiry = _parse_writer_lease_timestamp(
                    existing.get("expires_at")
                )
            except (subprocess.CalledProcessError, PreflightError):
                raise PreflightError(
                    "Could not create or safely verify the member writer lease"
                ) from create_error
            if existing_expiry > now_utc:
                raise PreflightError(
                    "An exact-contract writer lease is still active; no RunPod Pod "
                    f"was submitted. Lease expiry: {_writer_lease_timestamp(existing_expiry)}"
                ) from None
            try:
                run_checked(
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        lease_file.name,
                        object_uri,
                        f"--if-generation-match={generation}",
                        f"--project={project}",
                        "--quiet",
                    ]
                )
            except subprocess.CalledProcessError:
                raise PreflightError(
                    "Writer lease changed during atomic reclaim; no RunPod Pod was submitted"
                ) from None
            state = "expired-lease-reclaimed"
            replaced_generation = generation
        else:
            state = "writer-lease-created"
            replaced_generation = ""

    result = {
        "state": state,
        "uri": object_uri,
        "member_contract_sha256": member_contract_sha256,
        "attempt_nonce": attempt_nonce,
        "pod_name": pod_name,
        "expires_at": payload["expires_at"],
    }
    if replaced_generation:
        result["replaced_generation"] = replaced_generation
    return result


def release_member_writer_lease(
    *,
    project: str,
    bucket: str,
    result_prefix: str,
    member_contract_sha256: str,
    attempt_nonce: str,
    pod_name: str,
) -> dict[str, str]:
    """Atomically remove only the exact lease for a failed Pod submission."""

    if WRITER_LEASE_SHA256_RE.fullmatch(member_contract_sha256) is None:
        raise PreflightError("Writer lease release requires a valid member contract hash")
    if WRITER_LEASE_NONCE_RE.fullmatch(attempt_nonce) is None:
        raise PreflightError("Writer lease release requires a 32-character attempt nonce")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,190}", pod_name):
        raise PreflightError("Writer lease release Pod name contains unsupported characters")

    object_uri = f"gs://{bucket}/{result_prefix}/control/active-writer-lease.json"
    existing, generation = _read_stable_writer_lease(
        project=project, object_uri=object_uri
    )
    expected = {
        "schema_version": 1,
        "provider": "runpod",
        "member_contract_sha256": member_contract_sha256,
        "attempt_nonce": attempt_nonce,
        "pod_name": pod_name,
    }
    if any(existing.get(key) != value for key, value in expected.items()):
        raise PreflightError(
            "Writer lease release identity does not match the exact failed submission"
        )
    try:
        run_checked(
            [
                "gcloud",
                "storage",
                "rm",
                object_uri,
                f"--if-generation-match={generation}",
                f"--project={project}",
                "--quiet",
            ]
        )
    except subprocess.CalledProcessError:
        raise PreflightError(
            "Writer lease changed during atomic release; lease was preserved"
        ) from None
    return {
        "state": "failed-submission-lease-released",
        "uri": object_uri,
        "generation": generation,
        "member_contract_sha256": member_contract_sha256,
        "attempt_nonce": attempt_nonce,
        "pod_name": pod_name,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    require_project_confirmation(args.project, args.confirm_project)
    config = _read_json(args.data_config.expanduser().resolve())
    control_order = control_order_from_config(config, args.control_order)
    uploads = plan_a2_uploads(
        run_id=args.run_id,
        data_config=args.data_config,
        expected_train_captures=args.expected_train_captures,
        control_order=control_order,
        data_root=args.data_root,
    )
    manifest = {
        "schema_version": 1,
        "project": args.project,
        "bucket": args.bucket,
        "run_id": args.run_id,
        "control_order": list(control_order),
        "expected_train_captures": args.expected_train_captures,
        "objects": [
            {"name": item.object_name, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in uploads
        ],
        "total_bytes": sum(item.size_bytes for item in uploads),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.flush()
        upload_pairs = [(item.local_path, item.object_name) for item in uploads]
        upload_pairs.append(
            (Path(manifest_file.name), f"jobs/{args.run_id}/configs/input-manifest.json")
        )
        for local_path, object_name in upload_pairs:
            run_checked(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    str(local_path),
                    f"gs://{args.bucket}/{object_name}",
                    "--if-generation-match=0",
                    f"--project={args.project}",
                    "--quiet",
                ]
            )
    return manifest


def submit(args: argparse.Namespace) -> dict[str, Any]:
    require_project_confirmation(args.project, args.confirm_project)
    control_order = parse_control_order(args.control_order)
    cost = bounded_cost(
        hourly_rate_eur=args.hourly_rate_eur,
        timeout_minutes=args.timeout_minutes,
        worker_count=1,
        overhead_reserve_eur=args.overhead_reserve_eur,
        authorized_max_eur=args.max_cost_eur,
    )
    input_manifest = read_remote_input_manifest(
        project=args.project,
        bucket=args.bucket,
        run_id=args.run_id,
    )
    try:
        validate_input_manifest(
            input_manifest,
            project=args.project,
            bucket=args.bucket,
            run_id=args.run_id,
            control_order=control_order,
            expected_train_captures=args.expected_train_captures,
        )
    except ValueError as error:
        raise PreflightError(str(error)) from error
    result_prefix = f"jobs/{args.run_id}/results/{args.child}-seed-{args.seed}"
    member_contract = build_member_contract(
        project=args.project,
        bucket=args.bucket,
        run_id=args.run_id,
        result_prefix=result_prefix,
        input_manifest_sha256=canonical_json_sha256(input_manifest),
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
    member_config_sha256 = canonical_json_sha256(member_contract)
    spec = build_job_spec(
        project=args.project,
        bucket=args.bucket,
        run_id=args.run_id,
        image_uri=args.image_uri,
        service_account=args.service_account,
        child=args.child,
        seed=args.seed,
        timeout_minutes=args.timeout_minutes,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        validation_ny=args.validation_ny,
        num_workers=args.num_workers,
        expected_train_captures=args.expected_train_captures,
        scheduling_strategy=args.scheduling_strategy,
        member_config_sha256=member_config_sha256,
        control_order=control_order,
        train_chunks_per_output=args.train_chunks_per_output,
        artifact_purpose=args.artifact_purpose,
    )
    contract_registration = register_member_contract(
        project=args.project,
        bucket=args.bucket,
        result_prefix=result_prefix,
        contract=member_contract,
        resume_existing_member=args.resume_existing_member,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as config:
        json.dump(spec, config, separators=(",", ":"), sort_keys=True)
        config.flush()
        response = run_checked(
            [
                "gcloud",
                "ai",
                "custom-jobs",
                "create",
                f"--project={args.project}",
                f"--region={args.region}",
                f"--display-name=dessamp-{args.run_id}-{args.child}-s{args.seed}",
                f"--config={config.name}",
                "--labels=application=dessamp,purpose=conditioned-a2",
                "--format=json",
                "--quiet",
            ]
        )
    return {
        "job": json.loads(response.stdout),
        "cost_guard": cost,
        "member_contract": contract_registration,
        "job_spec": spec,
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", required=True)
    parser.add_argument("--confirm-project", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    _common(prepare_parser)
    prepare_parser.add_argument("--data-config", required=True, type=Path)
    prepare_parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Authorized root for every WAV referenced by the data config; "
            "defaults to the data-config directory"
        ),
    )
    prepare_parser.add_argument("--expected-train-captures", type=int, default=40)
    prepare_parser.add_argument(
        "--control-order",
        help="Comma-separated model-vector order; must match data metadata when declared",
    )

    submit_parser = commands.add_parser("submit")
    _common(submit_parser)
    submit_parser.add_argument("--region", default="europe-west3")
    submit_parser.add_argument("--image-uri", required=True)
    submit_parser.add_argument("--service-account", required=True)
    submit_parser.add_argument("--child", required=True, choices=("lite", "full"))
    submit_parser.add_argument("--seed", type=int, required=True)
    submit_parser.add_argument("--machine-type", default="n1-standard-8")
    submit_parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_T4")
    submit_parser.add_argument("--accelerator-count", type=int, default=1)
    submit_parser.add_argument("--epochs", type=int, default=60)
    submit_parser.add_argument("--batch-size", type=int, default=16)
    submit_parser.add_argument("--ny", type=int, default=8192)
    submit_parser.add_argument("--validation-ny", type=int, default=262144)
    submit_parser.add_argument("--num-workers", type=int, default=4)
    submit_parser.add_argument("--expected-train-captures", type=int, default=40)
    submit_parser.add_argument("--train-chunks-per-output", type=int)
    submit_parser.add_argument(
        "--artifact-purpose",
        choices=sorted(ARTIFACT_PURPOSES),
        default="production_candidate",
    )
    submit_parser.add_argument(
        "--control-order",
        default=",".join(DEFAULT_CONTROL_ORDER),
        help="Comma-separated model-vector order; preserve it on every continuation",
    )
    submit_parser.add_argument(
        "--resume-existing-member",
        action="store_true",
        help=(
            "Explicitly continue only an exact immutable member contract with the same "
            "absolute target epoch count; also confirm no concurrent writer exists"
        ),
    )
    submit_parser.add_argument("--scheduling-strategy", choices=("spot", "standard"), default="spot")
    submit_parser.add_argument("--timeout-minutes", type=int, default=300)
    submit_parser.add_argument(
        "--hourly-rate-eur",
        type=decimal_eur,
        required=True,
        help=(
            "Explicit currently verified conservative all-in hourly rate; record "
            "the source, region, and timestamp before every submission"
        ),
    )
    submit_parser.add_argument("--overhead-reserve-eur", type=decimal_eur, default=Decimal("2.00"))
    submit_parser.add_argument("--max-cost-eur", type=decimal_eur, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = prepare(args) if args.command == "prepare" else submit(args)
    except (PreflightError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            raise SystemExit(error.stderr.strip()) from error
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
