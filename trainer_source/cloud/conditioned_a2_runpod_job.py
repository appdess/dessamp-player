#!/usr/bin/env python3
"""Prepare GCS evidence and run exact-contract conditioned-A2 members on RunPod.

RunPod supplies only the GPU process. The immutable input bundle, member
contract, durable epoch checkpoints, and final artifacts remain in GCS so a
terminated Pod can be replaced without changing the member identity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import stat
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_UP
from pathlib import Path
from typing import Any, NoReturn, Sequence

try:
    from trainer_source.cloud.conditioned_a2_contract import (
        ARTIFACT_PURPOSES,
        build_member_contract,
        canonical_json_bytes,
        canonical_json_sha256,
        validate_input_manifest,
    )
    from trainer_source.cloud.conditioned_a2_vertex_job import (
        PreflightError,
        acquire_member_writer_lease,
        parse_control_order,
        prepare as prepare_bundle,
        read_remote_input_manifest,
        register_member_contract,
        release_member_writer_lease,
    )
    from trainer_source.cloud.conditioned_vertex_job import require_project_confirmation
    from trainer_source.cloud.gcs_capabilities import (
        CAPABILITY_ENVIRONMENT_VARIABLE,
        encode_capability_bundle,
    )
except ModuleNotFoundError:
    from conditioned_a2_contract import (  # type: ignore[no-redef]
        ARTIFACT_PURPOSES,
        build_member_contract,
        canonical_json_bytes,
        canonical_json_sha256,
        validate_input_manifest,
    )
    from conditioned_a2_vertex_job import (  # type: ignore[no-redef]
        PreflightError,
        acquire_member_writer_lease,
        parse_control_order,
        prepare as prepare_bundle,
        read_remote_input_manifest,
        register_member_contract,
        release_member_writer_lease,
    )
    from conditioned_vertex_job import require_project_confirmation  # type: ignore[no-redef]
    from gcs_capabilities import (  # type: ignore[no-redef]
        CAPABILITY_ENVIRONMENT_VARIABLE,
        encode_capability_bundle,
    )


RUNPOD_API = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL_API = "https://api.runpod.io/graphql"
RUNPOD_USER_AGENT = "DessAmp-Conditioned-A2/1.0"
MAX_RUNPOD_MEMBER_COST_USD = Decimal("50.00")
MAX_SIGNING_CREDENTIAL_BYTES = 64 * 1024
MAX_JSON_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_MODEL_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_RUNPOD_RESPONSE_BYTES = 256 * 1024
MAX_RUNPOD_ERROR_BYTES = 8 * 1024
CAPABILITY_EXPIRATION_RESERVE = timedelta(minutes=15)
GAR_DIGEST_RE = re.compile(
    r"^[a-z0-9-]+-docker\.pkg\.dev/[a-z0-9-]+/[^@]+@sha256:[0-9a-f]{64}$"
)
# RunPod Pod IDs are opaque 14-character lowercase ASCII-alphanumeric values.
# Keep this grammar deliberately narrower than a generic URL path segment and
# fail closed if the provider changes formats: an ID is interpolated into both
# the fresh-identity GET and destructive DELETE paths.
RUNPOD_POD_ID_RE = re.compile(r"\A[a-z0-9]{14}\Z", re.ASCII)
RUNPOD_PROVIDER_DETAIL_REDACTED = "<provider detail withheld>"
MEMBER_VARIANT_RE = re.compile(r"\A[a-z0-9](?:[a-z0-9-]{0,31})\Z", re.ASCII)


class RunPodIdentityProofError(PreflightError):
    """A created Pod could not be freshly and exactly re-identified."""


class RunPodDeleteError(PreflightError):
    """A DELETE failed after the exact created Pod was freshly identified."""


class RejectRunPodRedirects(urllib.request.HTTPRedirectHandler):
    """Reject every redirect before urllib can forward a bearer credential."""

    def redirect_request(
        self, request, file_pointer, code, message, headers, new_url
    ):
        del new_url
        raise urllib.error.HTTPError(
            request.full_url,
            code,
            "RunPod redirect refused",
            headers,
            file_pointer,
        )


RUNPOD_URL_OPENER = urllib.request.build_opener(RejectRunPodRedirects())


def _open_runpod_url(request: urllib.request.Request):
    return RUNPOD_URL_OPENER.open(request, timeout=60)


def load_gcs_signing_credentials(path: Path, *, project: str):
    """Load a local-only service-account signer without following the leaf path."""

    expanded = path.expanduser()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(expanded, flags)
    except OSError as error:
        raise PreflightError(f"Could not open local GCS signing credential: {expanded}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PreflightError("GCS signing credential must be a regular file")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise PreflightError("GCS signing credential must not be group/world accessible")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            raw = source.read(MAX_SIGNING_CREDENTIAL_BYTES + 1)
    finally:
        os.close(descriptor)
    if not 0 < len(raw) <= MAX_SIGNING_CREDENTIAL_BYTES:
        raise PreflightError("GCS signing credential is empty or oversized")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError("GCS signing credential is not valid JSON") from error
    expected_suffix = f"@{project}.iam.gserviceaccount.com"
    if (
        not isinstance(payload, dict)
        or payload.get("type") != "service_account"
        or payload.get("project_id") != project
        or not str(payload.get("client_email", "")).endswith(expected_suffix)
    ):
        raise PreflightError("GCS signing credential does not belong to the confirmed project")
    try:
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_info(payload)
    except (ImportError, ValueError) as error:
        raise PreflightError("Could not load the local GCS signing credential") from error
    if credentials.service_account_email != payload["client_email"]:
        raise PreflightError("Loaded GCS signer identity differs from its credential metadata")
    return credentials


def generate_gcs_capability_bundle(
    *,
    project: str,
    bucket: str,
    run_id: str,
    result_prefix: str,
    child: str,
    member_config_sha256: str,
    member_contract: dict[str, Any],
    input_manifest: dict[str, Any],
    signing_credentials_path: Path,
    expires_at: datetime,
) -> tuple[str, dict[str, Any]]:
    """Mint exact, expiring GCS reads and server-size-bounded writes."""

    if expires_at.tzinfo is None:
        raise PreflightError("GCS capability expiration must include a timezone")
    expires_at = expires_at.astimezone(timezone.utc).replace(microsecond=0)

    credentials = load_gcs_signing_credentials(
        signing_credentials_path, project=project
    )
    try:
        from google.cloud import storage
    except ImportError as error:
        raise PreflightError("google-cloud-storage is required to mint GCS capabilities") from error
    storage_client = storage.Client(project=project, credentials=credentials)
    storage_bucket = storage_client.bucket(bucket)

    def signed_entry(
        name: str,
        *,
        method: str,
        max_bytes: int,
        size_bytes: int | None = None,
        sha256: str | None = None,
        optional: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "version": "v4",
            "expiration": expires_at,
            "method": method,
            "credentials": credentials,
        }
        signed_headers: dict[str, str] | None = None
        if method == "PUT":
            kwargs["content_type"] = "application/octet-stream"
            signed_headers = {
                "x-goog-content-length-range": f"1,{max_bytes}",
            }
            kwargs["headers"] = signed_headers
        url = storage_bucket.blob(name).generate_signed_url(**kwargs)
        entry: dict[str, Any] = {
            "name": name,
            "url": url,
            "max_bytes": max_bytes,
        }
        if method == "PUT":
            entry["headers"] = {
                "content-type": "application/octet-stream",
                "x-goog-content-length-range": signed_headers[
                    "x-goog-content-length-range"
                ],
            }
        if size_bytes is not None:
            entry["size_bytes"] = size_bytes
        if sha256 is not None:
            entry["sha256"] = sha256
        if optional:
            entry["optional"] = True
        return entry

    source_prefix = f"jobs/{run_id}"
    contract_bytes = canonical_json_bytes(member_contract)
    read_objects = [
        signed_entry(
            f"{source_prefix}/configs/input-manifest.json",
            method="GET",
            max_bytes=MAX_JSON_ARTIFACT_BYTES,
        ),
        signed_entry(
            f"{result_prefix}/contract/member-config.json",
            method="GET",
            max_bytes=len(contract_bytes),
            size_bytes=len(contract_bytes),
            sha256=member_config_sha256,
        ),
    ]
    for value in input_manifest["objects"]:
        read_objects.append(
            signed_entry(
                value["name"],
                method="GET",
                max_bytes=value["size_bytes"],
                size_bytes=value["size_bytes"],
                sha256=value["sha256"],
            )
        )

    durable_prefix = f"{result_prefix}/durable/{member_config_sha256}"
    for filename in ("best.ckpt", "resume.ckpt"):
        read_objects.append(
            signed_entry(
                f"{durable_prefix}/{filename}",
                method="GET",
                max_bytes=MAX_CHECKPOINT_BYTES,
                optional=True,
            )
        )

    write_limits = {
        "config_data.json": MAX_JSON_ARTIFACT_BYTES,
        "config_learning.json": MAX_JSON_ARTIFACT_BYTES,
        "config_model_export.json": MAX_JSON_ARTIFACT_BYTES,
        "checkpoints/best.ckpt": MAX_CHECKPOINT_BYTES,
        "checkpoints/resume.ckpt": MAX_CHECKPOINT_BYTES,
        f"conditioned-a2-{child}.nam": MAX_MODEL_ARTIFACT_BYTES,
        "training-manifest.json": MAX_JSON_ARTIFACT_BYTES,
    }
    write_objects = [
        signed_entry(
            f"{result_prefix}/{relative}",
            method="PUT",
            max_bytes=limit,
        )
        for relative, limit in write_limits.items()
    ]
    write_objects.extend(
        signed_entry(
            f"{durable_prefix}/{filename}",
            method="PUT",
            max_bytes=limit,
        )
        for filename, limit in (
            ("best.ckpt", MAX_CHECKPOINT_BYTES),
            ("resume.ckpt", MAX_CHECKPOINT_BYTES),
            (f"conditioned-a2-{child}.nam", MAX_MODEL_ARTIFACT_BYTES),
            ("training-manifest.json", MAX_JSON_ARTIFACT_BYTES),
        )
    )
    expires_text = expires_at.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    payload = {
        "schema_version": 2,
        "bucket": bucket,
        "expires_at": expires_text,
        "read_objects": read_objects,
        "write_objects": write_objects,
    }
    encoded = encode_capability_bundle(payload)
    safe_scope = {
        "expires_at": expires_text,
        "read_object_count": len(read_objects),
        "write_object_count": len(write_objects),
        "scope_sha256": canonical_json_sha256(
            {
                "bucket": bucket,
                "expires_at": expires_text,
                "read_objects": [
                    {
                        key: value[key]
                        for key in (
                            "name",
                            "max_bytes",
                            "size_bytes",
                            "sha256",
                            "optional",
                        )
                        if key in value
                    }
                    for value in read_objects
                ],
                "write_objects": [
                    {
                        "name": value["name"],
                        "max_bytes": value["max_bytes"],
                        "headers": value["headers"],
                    }
                    for value in write_objects
                ],
            }
        ),
    }
    close = getattr(storage_client, "close", None)
    if callable(close):
        close()
    return encoded, safe_scope


def conditioned_container_args(
    *,
    project: str,
    bucket: str,
    run_id: str,
    image_uri: str,
    child: str,
    seed: int,
    member_config_sha256: str,
    control_order: Sequence[str],
    epochs: int,
    batch_size: int,
    ny: int,
    validation_ny: int,
    num_workers: int,
    expected_train_captures: int,
    machine_type: str,
    accelerator_type: str,
    result_prefix: str | None = None,
    train_chunks_per_output: int | None = None,
    artifact_purpose: str = "production_candidate",
) -> list[str]:
    result_prefix = result_prefix or f"jobs/{run_id}/results/{child}-seed-{seed}"
    arguments = [
        "--bucket", bucket,
        "--source-prefix", f"jobs/{run_id}",
        "--result-prefix", result_prefix,
        "--model-name", f"dessamp-a2-{child}",
        "--project", project,
        "--image-uri", image_uri,
        "--machine-type", machine_type,
        "--accelerator-type", accelerator_type,
        "--accelerator-count", "1",
        "--member-config-sha256", member_config_sha256,
        "--control-order", ",".join(control_order),
        "--child", child,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--batch-size", str(batch_size),
        "--ny", str(ny),
        "--validation-ny", str(validation_ny),
        "--num-workers", str(num_workers),
        "--expected-train-captures", str(expected_train_captures),
        "--runpod-quiesce-on-success",
    ]
    if train_chunks_per_output is not None:
        arguments.extend(
            ["--train-chunks-per-output", str(train_chunks_per_output)]
        )
    if artifact_purpose != "production_candidate":
        arguments.extend(["--artifact-purpose", artifact_purpose])
    return arguments


def decimal_usd(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("expected a decimal USD amount") from error
    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError("USD amounts must be finite and positive")
    return parsed


def runpod_billing_decimal(value: Any, *, field: str) -> Decimal:
    """Parse an untrusted RunPod billing field into a finite, non-negative value."""
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise PreflightError(f"RunPod returned invalid {field}") from error
    if not parsed.is_finite() or parsed < 0:
        raise PreflightError(f"RunPod returned invalid {field}")
    return parsed


def bounded_runpod_cost(
    *, hourly_rate_usd: Decimal, timeout_minutes: int, authorized_max_usd: Decimal
) -> dict[str, str | int]:
    if not 1 <= timeout_minutes <= 24 * 60:
        raise PreflightError("RunPod timeout must be between 1 and 1440 minutes")
    if authorized_max_usd > MAX_RUNPOD_MEMBER_COST_USD:
        raise PreflightError(
            f"per-member authorization exceeds repository hard maximum USD {MAX_RUNPOD_MEMBER_COST_USD}"
        )
    compute = hourly_rate_usd * Decimal(timeout_minutes) / Decimal(60)
    overhead = Decimal("0.10")
    ceiling = compute + overhead
    if ceiling > authorized_max_usd:
        raise PreflightError(
            f"worst-case RunPod ceiling USD {ceiling:.4f} exceeds authorization USD {authorized_max_usd:.2f}"
        )
    return {
        "hourly_rate_usd": str(hourly_rate_usd),
        "timeout_minutes": timeout_minutes,
        "compute_ceiling_usd": str(compute.quantize(Decimal("0.0001"), rounding=ROUND_UP)),
        "overhead_reserve_usd": str(overhead),
        "total_ceiling_usd": str(ceiling.quantize(Decimal("0.0001"), rounding=ROUND_UP)),
        "authorized_max_usd": str(authorized_max_usd),
    }


def build_pod_spec(
    *,
    name: str,
    image_uri: str,
    registry_auth_id: str,
    gcs_capabilities_b64: str,
    gpu_type: str,
    cloud_type: str,
    interruptible: bool,
    container_args: Sequence[str],
) -> dict[str, Any]:
    if GAR_DIGEST_RE.fullmatch(image_uri) is None:
        raise PreflightError("RunPod image must be a digest-pinned GAR artifact")
    if cloud_type not in {"COMMUNITY", "SECURE"}:
        raise PreflightError("RunPod cloud type must be COMMUNITY or SECURE")
    if not registry_auth_id or not gcs_capabilities_b64:
        raise PreflightError("RunPod registry auth and scoped GCS capabilities are required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,190}", name):
        raise PreflightError("RunPod Pod name contains unsupported characters")

    entrypoint = "/app/trainer_source/cloud/conditioned_a2_entrypoint.py"
    training_command = " ".join(
        [
            "python -u",
            shlex.quote(entrypoint),
            shlex.join(list(container_args)),
        ]
    )
    # A RunPod Pod whose desired state remains RUNNING may restart a container
    # after its command exits.  A completed trainer would then republish its
    # final model and manifest on every restart, so keep the successful
    # container alive and quiescent until the external monitor deliberately
    # stops the Pod.  Failures still exit normally for recovery handling.
    command = " ".join(
        [
            "set -euo pipefail;",
            "umask 077;",
            f"if {training_command}; then",
            f"unset {CAPABILITY_ENVIRONMENT_VARIABLE};",
            "printf '%s\\n' DESSAMP_TRAINING_COMPLETE_WAITING_FOR_POD_STOP;",
            "exec tail -f /dev/null;",
            "else",
            "training_status=$?;",
            f"unset {CAPABILITY_ENVIRONMENT_VARIABLE};",
            "exit \"$training_status\";",
            "fi",
        ]
    )
    return {
        "name": name,
        "imageName": image_uri,
        "containerRegistryAuthId": registry_auth_id,
        "cloudType": cloud_type,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": [gpu_type],
        "gpuTypePriority": "availability",
        "interruptible": interruptible,
        "allowedCudaVersions": [
            "13.0", "12.9", "12.8", "12.7", "12.6", "12.5", "12.4"
        ],
        "containerDiskInGb": 30,
        "volumeInGb": 10,
        "volumeMountPath": "/workspace",
        "minRAMPerGPU": 32,
        "minVCPUPerGPU": 8,
        "supportPublicIp": False,
        "ports": [],
        "locked": False,
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": [command],
        "env": {
            CAPABILITY_ENVIRONMENT_VARIABLE: gcs_capabilities_b64,
            "PYTHONUNBUFFERED": "1",
        },
    }


def build_on_demand_graphql_input(
    *,
    name: str,
    image_uri: str,
    registry_auth_id: str,
    gcs_capabilities_b64: str,
    gpu_type: str,
    cloud_type: str,
    container_args: Sequence[str],
    hourly_rate_usd: Decimal,
    terminate_after: str,
) -> dict[str, Any]:
    if GAR_DIGEST_RE.fullmatch(image_uri) is None:
        raise PreflightError("RunPod image must be a digest-pinned GAR artifact")
    if cloud_type not in {"COMMUNITY", "SECURE"}:
        raise PreflightError("RunPod cloud type must be COMMUNITY or SECURE")
    if not registry_auth_id or not gcs_capabilities_b64:
        raise PreflightError("RunPod registry auth and scoped GCS capabilities are required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,190}", name):
        raise PreflightError("RunPod Pod name contains unsupported characters")
    try:
        parsed_termination = datetime.fromisoformat(terminate_after.replace("Z", "+00:00"))
    except ValueError as error:
        raise PreflightError("RunPod terminateAfter must be an ISO-8601 timestamp") from error
    if parsed_termination.tzinfo is None:
        raise PreflightError("RunPod terminateAfter must include a timezone")
    return {
        "name": name,
        "imageName": image_uri,
        "containerRegistryAuthId": registry_auth_id,
        "cloudType": cloud_type,
        "gpuCount": 1,
        "gpuTypeId": gpu_type,
        "deployCost": float(hourly_rate_usd),
        "terminateAfter": terminate_after,
        "containerDiskInGb": 30,
        "volumeInGb": 10,
        "volumeMountPath": "/workspace",
        "minMemoryInGb": 32,
        "minVcpuCount": 8,
        "supportPublicIp": False,
        "startSsh": False,
        "dockerArgs": shlex.join(list(container_args)),
        "env": [
            {"key": CAPABILITY_ENVIRONMENT_VARIABLE, "value": gcs_capabilities_b64},
            {"key": "PYTHONUNBUFFERED", "value": "1"},
        ],
    }


def redacted_graphql_input(value: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(value))
    for item in redacted.get("env", []):
        if item.get("key") == CAPABILITY_ENVIRONMENT_VARIABLE:
            item["value"] = "<redacted>"
    return redacted


def redacted_pod_spec(spec: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(spec))
    if "env" in value and CAPABILITY_ENVIRONMENT_VARIABLE in value["env"]:
        value["env"][CAPABILITY_ENVIRONMENT_VARIABLE] = "<redacted>"
    return value


def redacted_provider_detail(detail: str, payload: Any) -> str:
    """Return a fixed replacement for untrusted provider-controlled text.

    Exact string replacement is insufficient here: RunPod sees the API key in
    the Authorization header and the temporary GCS capability in the request
    body, and an error can echo either value in a transformed representation.
    No provider-controlled diagnostic text is therefore safe to render.
    """

    del detail, payload
    return RUNPOD_PROVIDER_DETAIL_REDACTED


def _read_bounded_runpod_response(response: Any, *, operation: str) -> bytes:
    """Read at most one small control-plane JSON response."""

    try:
        raw = response.read(MAX_RUNPOD_RESPONSE_BYTES + 1)
    except Exception:
        raise PreflightError(f"{operation} response could not be read") from None
    if not isinstance(raw, (bytes, bytearray)):
        raise PreflightError(f"{operation} returned a non-byte response")
    if len(raw) > MAX_RUNPOD_RESPONSE_BYTES:
        raise PreflightError(f"{operation} response exceeded the size limit")
    return bytes(raw)


def _read_bounded_runpod_json(response: Any, *, operation: str) -> Any:
    """Parse bounded provider JSON without reflecting malformed input."""

    raw = _read_bounded_runpod_response(response, operation=operation)
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise PreflightError(f"{operation} returned invalid JSON") from None


def _discard_bounded_runpod_error(error: urllib.error.HTTPError) -> None:
    """Bound error-body consumption while deliberately retaining no content."""

    try:
        error.read(MAX_RUNPOD_ERROR_BYTES + 1)
    except Exception:
        pass


def runpod_request(
    *, api_key: str, method: str, path: str, payload: dict[str, Any] | None = None
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{RUNPOD_API}{path}",
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": RUNPOD_USER_AGENT,
        },
    )
    try:
        with _open_runpod_url(request) as response:
            return _read_bounded_runpod_json(
                response, operation=f"RunPod API {method} {path}"
            )
    except urllib.error.HTTPError as error:
        _discard_bounded_runpod_error(error)
        detail = redacted_provider_detail("", payload)
        raise PreflightError(
            f"RunPod API {method} {path} failed ({error.code}): {detail}"
        ) from None


def runpod_graphql_request(
    *, api_key: str, query: str, variables: dict[str, Any]
) -> dict[str, Any]:
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = urllib.request.Request(
        RUNPOD_GRAPHQL_API,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": RUNPOD_USER_AGENT,
        },
    )
    try:
        with _open_runpod_url(request) as response:
            payload = _read_bounded_runpod_json(
                response, operation="RunPod GraphQL"
            )
    except urllib.error.HTTPError as error:
        _discard_bounded_runpod_error(error)
        detail = redacted_provider_detail("", variables)
        raise PreflightError(
            f"RunPod GraphQL failed ({error.code}): {detail}"
        ) from None
    if not isinstance(payload, dict):
        raise PreflightError("RunPod GraphQL returned a non-object response")
    errors = payload.get("errors")
    if errors:
        raise PreflightError(
            f"RunPod GraphQL failed: {RUNPOD_PROVIDER_DETAIL_REDACTED}"
        )
    data = payload.get("data")
    if not isinstance(data, dict):
        raise PreflightError("RunPod GraphQL response has no data object")
    return data


def runpod_account_snapshot(*, api_key: str) -> dict[str, Decimal]:
    data = runpod_graphql_request(
        api_key=api_key,
        query=(
            "query DessAmpAccountPreflight { myself { "
            "clientBalance currentSpendPerHr spendLimit } }"
        ),
        variables={},
    )
    myself = data.get("myself")
    if not isinstance(myself, dict):
        raise PreflightError("RunPod account preflight returned no account")
    try:
        return {
            "client_balance_usd": runpod_billing_decimal(
                myself["clientBalance"], field="clientBalance"
            ),
            "current_spend_per_hour_usd": runpod_billing_decimal(
                myself["currentSpendPerHr"], field="currentSpendPerHr"
            ),
            "spend_limit_per_hour_usd": runpod_billing_decimal(
                myself["spendLimit"], field="spendLimit"
            ),
        }
    except KeyError as error:
        raise PreflightError("RunPod account preflight returned invalid billing fields") from error


CREATE_ON_DEMAND_POD_MUTATION = """
mutation DessAmpCreatePod($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    desiredStatus
    costPerHr
    imageName
  }
}
"""

ALL_PODS_IDENTITY_QUERY = """
query DessAmpAllPodIdentities {
  myself {
    pods {
      id
      name
      desiredStatus
    }
  }
}
"""


def prove_runpod_pod_name_absent(*, api_key: str, expected_name: str) -> None:
    """Fail closed unless the account-wide Pod identity list proves absence."""

    data = runpod_graphql_request(
        api_key=api_key,
        query=ALL_PODS_IDENTITY_QUERY,
        variables={},
    )
    myself = data.get("myself")
    pods = myself.get("pods") if isinstance(myself, dict) else None
    if not isinstance(pods, list):
        raise PreflightError("RunPod Pod-absence proof returned no Pod list")
    for pod in pods:
        if not isinstance(pod, dict):
            raise PreflightError("RunPod Pod-absence proof returned an invalid Pod")
        name = pod.get("name")
        if not isinstance(name, str):
            raise PreflightError("RunPod Pod-absence proof returned an invalid Pod name")
        if name == expected_name:
            raise PreflightError(
                "RunPod Pod-absence proof found the exact attempted Pod name"
            )


def release_failed_submission_writer_lease(
    *,
    api_key: str,
    project: str,
    bucket: str,
    result_prefix: str,
    member_contract_sha256: str,
    attempt_nonce: str,
    pod_name: str,
) -> dict[str, str]:
    """Release a lease only after provider-wide exact-name absence proof."""

    prove_runpod_pod_name_absent(api_key=api_key, expected_name=pod_name)
    return release_member_writer_lease(
        project=project,
        bucket=bucket,
        result_prefix=result_prefix,
        member_contract_sha256=member_contract_sha256,
        attempt_nonce=attempt_nonce,
        pod_name=pod_name,
    )

FRESH_POD_RUNTIME_QUERY = """
query DessAmpFreshPodRuntime($input: PodFilter) {
  pod(input: $input) {
    id
    name
    desiredStatus
    costPerHr
    imageName
    containerRegistryAuthId
    gpuCount
    podType
    machine {
      gpuTypeId
      secureCloud
      dataCenterId
    }
  }
}
"""


def validated_runpod_pod_id(value: Any) -> str:
    """Validate RunPod's opaque ID before using it as a REST path segment."""

    if not isinstance(value, str) or RUNPOD_POD_ID_RE.fullmatch(value) is None:
        raise PreflightError(
            "RunPod Pod ID must be exactly 14 lowercase ASCII-alphanumeric characters"
        )
    return value


def validate_created_pod_identity(
    pod: Any,
    *,
    expected_name: str,
    expected_image_uri: str,
    expected_registry_auth_id: str,
    expected_pod_id: str | None = None,
    require_registry_identity: bool = False,
    required_image_field: str | None = None,
) -> str:
    """Bind an untrusted create/read response to the Pod we just requested."""

    if GAR_DIGEST_RE.fullmatch(expected_image_uri) is None:
        raise PreflightError("Expected RunPod image identity is not digest-pinned GAR")
    if not expected_registry_auth_id:
        raise PreflightError("Expected RunPod registry identity is empty")
    if not isinstance(pod, dict):
        raise PreflightError("RunPod create response has no Pod identity")
    try:
        pod_id = validated_runpod_pod_id(pod.get("id"))
        if expected_pod_id is not None:
            expected_pod_id = validated_runpod_pod_id(expected_pod_id)
    except PreflightError as error:
        raise PreflightError("RunPod create response has an invalid Pod identity") from error
    if expected_pod_id is not None and pod_id != expected_pod_id:
        raise PreflightError("RunPod create response has an unexpected Pod identity")
    image_aliases = {
        field: pod[field]
        for field in ("imageName", "image")
        if field in pod
    }
    if not image_aliases or any(
        value != expected_image_uri for value in image_aliases.values()
    ):
        raise PreflightError("RunPod create response has an unexpected image identity")
    if required_image_field is not None and pod.get(required_image_field) != expected_image_uri:
        raise PreflightError(
            "RunPod create response is missing its required native image identity"
        )
    if (
        pod.get("name") != expected_name
        or pod.get("desiredStatus") != "RUNNING"
    ):
        raise PreflightError("RunPod create response does not match the requested Pod")
    registry_auth_id = pod.get("containerRegistryAuthId")
    if (
        require_registry_identity and registry_auth_id != expected_registry_auth_id
    ) or (
        registry_auth_id is not None and registry_auth_id != expected_registry_auth_id
    ):
        raise PreflightError("RunPod create response has an unexpected registry identity")
    return pod_id


def validate_fresh_pod_runtime(
    pod: Any,
    *,
    expected_gpu_type: str,
    expected_gpu_count: int,
    expected_cloud_type: str,
    expected_interruptible: bool,
    declared_hourly_rate_usd: Decimal,
) -> Decimal:
    """Bind the fresh REST allocation, tier, and rate to the request contract."""

    if not isinstance(pod, dict):
        raise PreflightError("Fresh RunPod Pod response is not an object")
    if not isinstance(expected_gpu_count, int) or isinstance(expected_gpu_count, bool) or expected_gpu_count <= 0:
        raise PreflightError("Expected RunPod GPU count is invalid")
    if expected_cloud_type not in {"COMMUNITY", "SECURE"}:
        raise PreflightError("Expected RunPod cloud type is invalid")
    if not isinstance(expected_interruptible, bool):
        raise PreflightError("Expected RunPod interruptibility is invalid")

    observed_interruptible = pod.get("interruptible")
    observed_pod_type = pod.get("podType")
    gpu = pod.get("gpu")
    machine = pod.get("machine")
    expected_secure_cloud = expected_cloud_type == "SECURE"
    observed_gpu_count = pod.get("gpuCount")
    if observed_gpu_count is None and isinstance(gpu, dict):
        observed_gpu_count = gpu.get("count")
    observed_gpu_display = gpu.get("displayName") if isinstance(gpu, dict) else None
    if not isinstance(observed_interruptible, bool):
        if observed_pod_type == "RESERVED":
            observed_interruptible = False
        elif observed_pod_type in {"INTERRUPTABLE", "BID"}:
            observed_interruptible = True
    if (
        not isinstance(observed_interruptible, bool)
        or observed_interruptible is not expected_interruptible
        or not isinstance(observed_gpu_count, int)
        or isinstance(observed_gpu_count, bool)
        or observed_gpu_count != expected_gpu_count
        or (
            observed_gpu_display is not None
            and observed_gpu_display != expected_gpu_type
        )
        or not isinstance(machine, dict)
        or machine.get("gpuTypeId") != expected_gpu_type
        or not isinstance(machine.get("secureCloud"), bool)
        or machine.get("secureCloud") is not expected_secure_cloud
    ):
        raise PreflightError(
            "Fresh RunPod allocation does not match the requested GPU, tier, or interruptibility"
        )

    actual_rate = runpod_billing_decimal(pod.get("costPerHr"), field="costPerHr")
    if actual_rate > declared_hourly_rate_usd:
        raise PreflightError("Fresh RunPod allocation exceeds the declared hourly-rate ceiling")
    return actual_rate


def fetch_verified_created_pod(
    *,
    api_key: str,
    pod_id: str,
    expected_name: str,
    expected_image_uri: str,
    expected_registry_auth_id: str,
) -> dict[str, Any]:
    """Fetch and prove one exact created Pod through the fresh REST boundary."""

    try:
        pod_id = validated_runpod_pod_id(pod_id)
    except PreflightError:
        raise RunPodIdentityProofError(
            "RunPod created-Pod identity is unsafe for automatic deletion"
        ) from None
    try:
        observed = runpod_request(
            api_key=api_key,
            method="GET",
            path=f"/pods/{pod_id}?includeMachine=true",
        )
        validate_created_pod_identity(
            observed,
            expected_name=expected_name,
            expected_image_uri=expected_image_uri,
            expected_registry_auth_id=expected_registry_auth_id,
            expected_pod_id=pod_id,
            require_registry_identity=True,
        )
    except (OSError, PreflightError):
        raise RunPodIdentityProofError(
            "RunPod created-Pod identity could not be freshly proven"
        ) from None
    return observed


def fetch_verified_created_pod_runtime(
    *,
    api_key: str,
    pod_id: str,
    expected_name: str,
    expected_image_uri: str,
    expected_registry_auth_id: str,
) -> dict[str, Any]:
    """Cross-check the fresh REST identity with a field-allowlisted GraphQL read."""

    try:
        pod_id = validated_runpod_pod_id(pod_id)
        data = runpod_graphql_request(
            api_key=api_key,
            query=FRESH_POD_RUNTIME_QUERY,
            variables={"input": {"podId": pod_id}},
        )
        observed = data.get("pod")
        validate_created_pod_identity(
            observed,
            expected_name=expected_name,
            expected_image_uri=expected_image_uri,
            expected_registry_auth_id=expected_registry_auth_id,
            expected_pod_id=pod_id,
            require_registry_identity=True,
            required_image_field="imageName",
        )
    except (OSError, PreflightError):
        raise RunPodIdentityProofError(
            "RunPod created-Pod runtime identity could not be freshly proven"
        ) from None
    return observed


def terminate_verified_created_pod(
    *,
    api_key: str,
    pod_id: str,
    expected_name: str,
    expected_image_uri: str,
    expected_registry_auth_id: str,
) -> None:
    """Delete only after a fresh REST read proves the exact created-Pod identity."""

    fetch_verified_created_pod(
        api_key=api_key,
        pod_id=pod_id,
        expected_name=expected_name,
        expected_image_uri=expected_image_uri,
        expected_registry_auth_id=expected_registry_auth_id,
    )
    try:
        runpod_request(api_key=api_key, method="DELETE", path=f"/pods/{pod_id}")
    except (OSError, PreflightError):
        raise RunPodDeleteError(
            "RunPod DELETE failed after exact created-Pod identity proof"
        ) from None


def _raise_cost_failure_after_cleanup(
    *,
    reason: str,
    api_key: str,
    pod_id: str,
    expected_name: str,
    expected_image_uri: str,
    expected_registry_auth_id: str,
) -> NoReturn:
    """Attempt identity-bound cleanup, then always report the cost failure."""

    try:
        terminate_verified_created_pod(
            api_key=api_key,
            pod_id=pod_id,
            expected_name=expected_name,
            expected_image_uri=expected_image_uri,
            expected_registry_auth_id=expected_registry_auth_id,
        )
    except RunPodIdentityProofError:
        raise PreflightError(
            f"{reason}; possible orphaned RunPod Pod {pod_id} requires manual "
            "review; automatic DELETE was not issued because exact identity "
            "could not be proven"
        ) from None
    except RunPodDeleteError:
        raise PreflightError(
            f"{reason}; verified RunPod Pod {pod_id} may be orphaned because "
            "DELETE failed; manual review is required"
        ) from None
    raise PreflightError(
        f"{reason}; verified RunPod Pod {pod_id} DELETE requested"
    )


def validate_created_pod_cost(
    *,
    api_key: str,
    created_pod: Any,
    expected_name: str,
    expected_image_uri: str,
    expected_registry_auth_id: str,
    declared_hourly_rate_usd: Decimal,
) -> Decimal:
    """Validate cost and delete only a freshly re-identified created Pod."""

    try:
        pod_id = validate_created_pod_identity(
            created_pod,
            expected_name=expected_name,
            expected_image_uri=expected_image_uri,
            expected_registry_auth_id=expected_registry_auth_id,
        )
    except PreflightError as error:
        raise PreflightError(
            f"{error}; possible orphaned RunPod Pod requires manual review; "
            "automatic DELETE was not attempted"
        ) from None
    try:
        actual_rate = runpod_billing_decimal(
            created_pod.get("costPerHr"), field="costPerHr"
        )
    except PreflightError as error:
        _raise_cost_failure_after_cleanup(
            reason=str(error),
            api_key=api_key,
            pod_id=pod_id,
            expected_name=expected_name,
            expected_image_uri=expected_image_uri,
            expected_registry_auth_id=expected_registry_auth_id,
        )
    if actual_rate > declared_hourly_rate_usd:
        _raise_cost_failure_after_cleanup(
            reason=(
                f"RunPod allocated USD {actual_rate}/h above the declared USD "
                f"{declared_hourly_rate_usd}/h ceiling"
            ),
            api_key=api_key,
            pod_id=pod_id,
            expected_name=expected_name,
            expected_image_uri=expected_image_uri,
            expected_registry_auth_id=expected_registry_auth_id,
        )
    return actual_rate


def allowlisted_created_pod_evidence(
    pod: Any,
    *,
    expected_name: str,
    expected_image_uri: str,
    expected_registry_auth_id: str,
    attempt_nonce: str,
    expected_gpu_type: str,
    expected_gpu_count: int,
    expected_cloud_type: str,
    expected_interruptible: bool,
    declared_hourly_rate_usd: Decimal,
) -> dict[str, Any]:
    """Retain only validated, non-secret fields from the create response."""

    if not re.fullmatch(r"[0-9a-f]{32}", attempt_nonce):
        raise PreflightError("RunPod attempt nonce is invalid")
    if not expected_name.endswith(f"-{attempt_nonce}"):
        raise PreflightError("RunPod Pod name is not bound to the attempt nonce")
    pod_id = validate_created_pod_identity(
        pod,
        expected_name=expected_name,
        expected_image_uri=expected_image_uri,
        expected_registry_auth_id=expected_registry_auth_id,
        require_registry_identity=True,
    )
    actual_rate = validate_fresh_pod_runtime(
        pod,
        expected_gpu_type=expected_gpu_type,
        expected_gpu_count=expected_gpu_count,
        expected_cloud_type=expected_cloud_type,
        expected_interruptible=expected_interruptible,
        declared_hourly_rate_usd=declared_hourly_rate_usd,
    )
    return {
        "id": pod_id,
        "name": expected_name,
        "desiredStatus": "RUNNING",
        "costPerHr": format(actual_rate, "f"),
        "imageName": expected_image_uri,
        "containerRegistryAuthId": expected_registry_auth_id,
        "attemptNonce": attempt_nonce,
        "gpuType": expected_gpu_type,
        "gpuCount": expected_gpu_count,
        "cloudType": expected_cloud_type,
        "interruptible": expected_interruptible,
    }


def verified_created_pod_evidence(
    *,
    api_key: str,
    created_pod: Any,
    expected_name: str,
    expected_image_uri: str,
    expected_registry_auth_id: str,
    attempt_nonce: str,
    expected_gpu_type: str,
    expected_gpu_count: int,
    expected_cloud_type: str,
    expected_interruptible: bool,
    declared_hourly_rate_usd: Decimal,
) -> dict[str, Any]:
    """Freshly prove a normal successful Pod before returning ledger evidence."""

    pod_id = validate_created_pod_identity(
        created_pod,
        expected_name=expected_name,
        expected_image_uri=expected_image_uri,
        expected_registry_auth_id=expected_registry_auth_id,
    )
    try:
        observed = fetch_verified_created_pod(
            api_key=api_key,
            pod_id=pod_id,
            expected_name=expected_name,
            expected_image_uri=expected_image_uri,
            expected_registry_auth_id=expected_registry_auth_id,
        )
        observed = fetch_verified_created_pod_runtime(
            api_key=api_key,
            pod_id=pod_id,
            expected_name=expected_name,
            expected_image_uri=expected_image_uri,
            expected_registry_auth_id=expected_registry_auth_id,
        )
    except RunPodIdentityProofError:
        raise PreflightError(
            f"RunPod Pod {pod_id} is a possible orphan requiring manual review; "
            "normal submission evidence was not persisted because fresh exact "
            "identity proof failed"
        ) from None
    try:
        return allowlisted_created_pod_evidence(
            observed,
            expected_name=expected_name,
            expected_image_uri=expected_image_uri,
            expected_registry_auth_id=expected_registry_auth_id,
            attempt_nonce=attempt_nonce,
            expected_gpu_type=expected_gpu_type,
            expected_gpu_count=expected_gpu_count,
            expected_cloud_type=expected_cloud_type,
            expected_interruptible=expected_interruptible,
            declared_hourly_rate_usd=declared_hourly_rate_usd,
        )
    except PreflightError:
        try:
            runpod_request(api_key=api_key, method="DELETE", path=f"/pods/{pod_id}")
        except (OSError, PreflightError):
            raise PreflightError(
                f"Fresh RunPod Pod {pod_id} runtime proof failed and its exact "
                "identity may remain billable because DELETE failed; manual review is required"
            ) from None
        raise PreflightError(
            f"Fresh RunPod Pod {pod_id} runtime proof failed; exact Pod DELETE requested "
            "and submission evidence was withheld"
        ) from None


def submit(args: argparse.Namespace) -> dict[str, Any]:
    require_project_confirmation(args.project, args.confirm_project)
    if args.interruptible:
        raise PreflightError(
            "cost-capped RunPod submissions currently require --no-interruptible so the provider can bind terminateAfter"
        )
    cost_guard = bounded_runpod_cost(
        hourly_rate_usd=args.hourly_rate_usd,
        timeout_minutes=args.timeout_minutes,
        authorized_max_usd=args.max_cost_usd,
    )
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key and args.runpod_keychain_service:
        api_key = subprocess.check_output(
            [
                "security",
                "find-generic-password",
                "-a",
                args.runpod_keychain_account,
                "-s",
                args.runpod_keychain_service,
                "-w",
            ],
            text=True,
        ).strip()
    if not api_key:
        raise PreflightError("RUNPOD_API_KEY is required in the environment")
    account_snapshot = runpod_account_snapshot(api_key=api_key)
    if account_snapshot["client_balance_usd"] < args.max_cost_usd:
        raise PreflightError(
            "RunPod funded balance is below this member's authorized maximum"
        )
    if account_snapshot["current_spend_per_hour_usd"] + args.hourly_rate_usd > account_snapshot["spend_limit_per_hour_usd"]:
        raise PreflightError("RunPod hourly spend limit would be exceeded")
    controls = parse_control_order(args.control_order)
    input_manifest = read_remote_input_manifest(
        project=args.project, bucket=args.bucket, run_id=args.run_id
    )
    try:
        validate_input_manifest(
            input_manifest,
            project=args.project,
            bucket=args.bucket,
            run_id=args.run_id,
            control_order=controls,
            expected_train_captures=args.expected_train_captures,
        )
    except ValueError as error:
        raise PreflightError(str(error)) from error

    member_variant = args.member_variant
    if member_variant is not None and MEMBER_VARIANT_RE.fullmatch(member_variant) is None:
        raise PreflightError(
            "RunPod member variant must use 1 to 32 lowercase letters, numbers, or hyphens"
        )
    member_name = f"{args.child}-seed-{args.seed}"
    if member_variant is not None:
        member_name = f"{member_name}-{member_variant}"
    result_prefix = f"jobs/{args.run_id}/results/{member_name}"
    machine_type = "runpod-pod"
    member_contract = build_member_contract(
        project=args.project,
        bucket=args.bucket,
        run_id=args.run_id,
        result_prefix=result_prefix,
        input_manifest_sha256=canonical_json_sha256(input_manifest),
        image_uri=args.image_uri,
        child=args.child,
        seed=args.seed,
        control_order=controls,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        validation_ny=args.validation_ny,
        num_workers=args.num_workers,
        expected_train_captures=args.expected_train_captures,
        machine_type=machine_type,
        accelerator_type=args.gpu_type,
        accelerator_count=1,
        train_chunks_per_output=args.train_chunks_per_output,
        artifact_purpose=args.artifact_purpose,
    )
    member_sha256 = canonical_json_sha256(member_contract)
    now = datetime.now(timezone.utc)
    terminate_at = now + timedelta(minutes=args.timeout_minutes)
    capability_expires_at = terminate_at + CAPABILITY_EXPIRATION_RESERVE
    attempt_nonce = secrets.token_hex(16)
    pod_member = f"s{args.seed}"
    if member_variant is not None:
        pod_member = f"{pod_member}-{member_variant}"
    pod_name = f"dessamp-{args.run_id}-{pod_member}-{attempt_nonce}"
    gcs_capabilities_b64, capability_scope = generate_gcs_capability_bundle(
        project=args.project,
        bucket=args.bucket,
        run_id=args.run_id,
        result_prefix=result_prefix,
        child=args.child,
        member_config_sha256=member_sha256,
        member_contract=member_contract,
        input_manifest=input_manifest,
        signing_credentials_path=args.gcs_signing_credentials,
        expires_at=capability_expires_at,
    )
    container_args = conditioned_container_args(
        project=args.project,
        bucket=args.bucket,
        run_id=args.run_id,
        image_uri=args.image_uri,
        child=args.child,
        seed=args.seed,
        member_config_sha256=member_sha256,
        control_order=controls,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        validation_ny=args.validation_ny,
        num_workers=args.num_workers,
        expected_train_captures=args.expected_train_captures,
        machine_type=machine_type,
        accelerator_type=args.gpu_type,
        result_prefix=result_prefix,
        train_chunks_per_output=args.train_chunks_per_output,
        artifact_purpose=args.artifact_purpose,
    )
    terminate_after = terminate_at.isoformat(timespec="seconds").replace("+00:00", "Z")
    pod_spec = build_on_demand_graphql_input(
        name=pod_name,
        image_uri=args.image_uri,
        registry_auth_id=args.registry_auth_id,
        gcs_capabilities_b64=gcs_capabilities_b64,
        gpu_type=args.gpu_type,
        cloud_type=args.cloud_type,
        container_args=container_args,
        hourly_rate_usd=args.hourly_rate_usd,
        terminate_after=terminate_after,
    )
    registration = register_member_contract(
        project=args.project,
        bucket=args.bucket,
        result_prefix=result_prefix,
        contract=member_contract,
        resume_existing_member=args.resume_existing_member,
    )
    writer_lease = acquire_member_writer_lease(
        project=args.project,
        bucket=args.bucket,
        result_prefix=result_prefix,
        member_contract_sha256=member_sha256,
        attempt_nonce=attempt_nonce,
        pod_name=pod_name,
        now=now,
        expires_at=capability_expires_at,
    )
    try:
        data = runpod_graphql_request(
            api_key=api_key,
            query=CREATE_ON_DEMAND_POD_MUTATION,
            variables={"input": pod_spec},
        )
    except (OSError, PreflightError) as submit_error:
        try:
            release_failed_submission_writer_lease(
                api_key=api_key,
                project=args.project,
                bucket=args.bucket,
                result_prefix=result_prefix,
                member_contract_sha256=member_sha256,
                attempt_nonce=attempt_nonce,
                pod_name=pod_name,
            )
        except (OSError, PreflightError):
            raise PreflightError(
                "RunPod submission failed and exact Pod absence plus atomic lease "
                "release could not be proven; the writer lease was preserved"
            ) from None
        raise PreflightError(
            "RunPod submission failed; exact Pod absence was proven and the "
            "writer lease was atomically released"
        ) from submit_error
    pod = data.get("podFindAndDeployOnDemand")
    validate_created_pod_cost(
        api_key=api_key,
        created_pod=pod,
        expected_name=pod_spec["name"],
        expected_image_uri=pod_spec["imageName"],
        expected_registry_auth_id=pod_spec["containerRegistryAuthId"],
        declared_hourly_rate_usd=args.hourly_rate_usd,
    )
    pod_evidence = verified_created_pod_evidence(
        api_key=api_key,
        created_pod=pod,
        expected_name=pod_spec["name"],
        expected_image_uri=pod_spec["imageName"],
        expected_registry_auth_id=pod_spec["containerRegistryAuthId"],
        attempt_nonce=attempt_nonce,
        expected_gpu_type=args.gpu_type,
        expected_gpu_count=1,
        expected_cloud_type=args.cloud_type,
        expected_interruptible=args.interruptible,
        declared_hourly_rate_usd=args.hourly_rate_usd,
    )
    return {
        "provider": "runpod",
        "pod": pod_evidence,
        "attempt_nonce": attempt_nonce,
        "writer_lease": writer_lease,
        "member_contract": registration,
        "member_config_sha256": member_sha256,
        "pod_spec": redacted_graphql_input(pod_spec),
        "cost_guard": cost_guard,
        "account_preflight": {key: str(value) for key, value in account_snapshot.items()},
        "gcs_capability_scope": capability_scope,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_gcs_identity(command: argparse.ArgumentParser) -> None:
        command.add_argument("--project", required=True)
        command.add_argument("--confirm-project", required=True)
        command.add_argument("--bucket", required=True)
        command.add_argument("--run-id", required=True)

    prepare_parser = commands.add_parser(
        "prepare", help="Upload one immutable hash-bound training bundle to GCS"
    )
    add_gcs_identity(prepare_parser)
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
        help="Comma-separated model-vector order; must match data metadata",
    )

    submit_parser = commands.add_parser(
        "submit", help="Start or continue one exact-contract RunPod member"
    )
    add_gcs_identity(submit_parser)
    submit_parser.add_argument("--image-uri", required=True)
    submit_parser.add_argument("--registry-auth-id", required=True)
    submit_parser.add_argument(
        "--gcs-signing-credentials",
        "--google-credentials",
        dest="gcs_signing_credentials",
        required=True,
        type=Path,
        help=(
            "Local-only service-account signer used to mint exact expiring GCS URLs; "
            "the credential itself is never sent to RunPod"
        ),
    )
    submit_parser.add_argument("--runpod-keychain-service")
    submit_parser.add_argument("--runpod-keychain-account", default="codex")
    submit_parser.add_argument("--child", choices=("lite", "full"), required=True)
    submit_parser.add_argument("--seed", required=True, type=int)
    submit_parser.add_argument("--gpu-type", default="NVIDIA RTX A5000")
    submit_parser.add_argument(
        "--cloud-type", choices=("COMMUNITY", "SECURE"), default="COMMUNITY"
    )
    submit_parser.add_argument(
        "--interruptible", action=argparse.BooleanOptionalAction, default=True
    )
    submit_parser.add_argument("--epochs", type=int, default=60)
    submit_parser.add_argument("--batch-size", type=int, default=16)
    submit_parser.add_argument("--ny", type=int, default=8192)
    submit_parser.add_argument("--validation-ny", type=int, default=262144)
    submit_parser.add_argument("--num-workers", type=int, default=4)
    submit_parser.add_argument("--expected-train-captures", type=int, default=40)
    submit_parser.add_argument("--member-variant")
    submit_parser.add_argument("--train-chunks-per-output", type=int)
    submit_parser.add_argument(
        "--artifact-purpose",
        choices=sorted(ARTIFACT_PURPOSES),
        default="production_candidate",
    )
    submit_parser.add_argument("--control-order", required=True)
    submit_parser.add_argument("--timeout-minutes", type=int, required=True)
    submit_parser.add_argument("--hourly-rate-usd", type=decimal_usd, required=True)
    submit_parser.add_argument("--max-cost-usd", type=decimal_usd, required=True)
    submit_parser.add_argument("--resume-existing-member", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = prepare_bundle(args) if args.command == "prepare" else submit(args)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        PreflightError,
    ) as error:
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            raise SystemExit(error.stderr.strip()) from error
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
