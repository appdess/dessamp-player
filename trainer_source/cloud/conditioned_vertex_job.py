#!/usr/bin/env python3
"""Shared validation helpers and disabled legacy generic Vertex entrypoints.

New production training uses the exact-contract conditioned-A2 RunPod/GCS
launcher.  The former generic ``prepare`` and ``submit`` mutations remain only
so historical callers fail with an explicit migration error; they perform no
cloud action.  Pure validation/cost helpers are retained for current launchers
and forensic replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
from dataclasses import dataclass
from decimal import Decimal, ROUND_UP
from pathlib import Path
from typing import Any, Iterable, NoReturn


HARD_MAX_COST_EUR = Decimal("50.00")
DEFAULT_T4_FRANKFURT_EUR_PER_HOUR = Decimal("1.0265")
MAX_TIMEOUT_MINUTES = 300
MAX_TRAINING_AUDIO_BYTES = 512 * 1024 * 1024
MAX_TRAINING_BUNDLE_BYTES = 64 * 1024 * 1024 * 1024
MAX_TRAINING_AUDIO_FILES = 2048
_ID_RE = re.compile(r"[a-z][a-z0-9-]{4,61}[a-z0-9]\Z")
_RUN_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")
_CHECKPOINT_LABEL_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")
_GAR_DIGEST_RE = re.compile(
    r"[a-z0-9-]+-docker\.pkg\.dev/[a-z0-9-]+/[^:@]+(?:/[^:@]+)*@sha256:[0-9a-f]{64}\Z"
)
LEGACY_MUTATION_ERROR = (
    "Legacy generic Vertex prepare/submit is disabled; use "
    "conditioned_a2_runpod_job.py with an immutable prepared bundle"
)


class PreflightError(ValueError):
    pass


@dataclass(frozen=True)
class InputObject:
    local_path: Path
    object_name: str
    sha256: str
    size_bytes: int


def decimal_eur(value: str) -> Decimal:
    try:
        amount = Decimal(value)
    except Exception as error:
        raise argparse.ArgumentTypeError(f"invalid EUR amount: {value!r}") from error
    if not amount.is_finite() or amount <= 0:
        raise argparse.ArgumentTypeError("EUR amount must be finite and positive")
    return amount


def checkpoint_binding(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not _CHECKPOINT_LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError(
            "checkpoint must use lowercase-label=/absolute/path.ckpt"
        )
    path = Path(raw_path).expanduser().resolve()
    if path.suffix.lower() != ".ckpt":
        raise argparse.ArgumentTypeError("checkpoint source must end in .ckpt")
    return label, path


def checkpoint_name(value: str) -> str:
    if (
        not value.endswith(".ckpt")
        or Path(value).name != value
        or not _CHECKPOINT_LABEL_RE.fullmatch(value[:-5])
    ):
        raise argparse.ArgumentTypeError(
            "checkpoint name must use lowercase letters, numbers and hyphens, ending in .ckpt"
        )
    return value


def require_project_confirmation(project: str, confirmation: str) -> None:
    if not _ID_RE.fullmatch(project):
        raise PreflightError(f"Invalid Google Cloud project ID: {project!r}")
    if project != confirmation:
        raise PreflightError(
            "--confirm-project must exactly match --project before a billable action"
        )


def bounded_cost(
    *,
    hourly_rate_eur: Decimal,
    timeout_minutes: int,
    worker_count: int,
    overhead_reserve_eur: Decimal,
    authorized_max_eur: Decimal,
) -> dict[str, str | int]:
    if not 1 <= timeout_minutes <= MAX_TIMEOUT_MINUTES:
        raise PreflightError(
            f"timeout must be between 1 and {MAX_TIMEOUT_MINUTES} minutes"
        )
    if not 1 <= worker_count <= 4:
        raise PreflightError("worker count must be between 1 and 4")
    if authorized_max_eur > HARD_MAX_COST_EUR:
        raise PreflightError("authorized maximum exceeds the repository hard ceiling of EUR 50")
    training = hourly_rate_eur * Decimal(timeout_minutes) / Decimal(60) * worker_count
    ceiling = training + overhead_reserve_eur
    if ceiling > authorized_max_eur:
        raise PreflightError(
            f"worst-case ceiling EUR {ceiling:.2f} exceeds authorization EUR {authorized_max_eur:.2f}"
        )
    return {
        "hourly_rate_eur_per_worker": str(hourly_rate_eur),
        "timeout_minutes": timeout_minutes,
        "worker_count": worker_count,
        "training_ceiling_eur": str(training.quantize(Decimal("0.01"), rounding=ROUND_UP)),
        "overhead_reserve_eur": str(overhead_reserve_eur),
        "total_ceiling_eur": str(ceiling.quantize(Decimal("0.01"), rounding=ROUND_UP)),
        "authorized_max_eur": str(authorized_max_eur),
    }


def load_training_inputs(
    data_config_path: Path,
    *,
    data_root: Path | None = None,
) -> list[Path]:
    config_path = data_config_path.expanduser().resolve()
    config = _read_json_object(config_path)
    root = (data_root or config_path.parent).expanduser().resolve()
    if not root.is_dir():
        raise PreflightError(f"training data root is not a directory: {root}")
    common = config.get("common")
    if not isinstance(common, dict) or not isinstance(common.get("x_path"), str):
        raise PreflightError("data config requires common.x_path")
    raw_paths = [common["x_path"]]
    for split in ("train", "validation"):
        split_config = config.get(split)
        if not isinstance(split_config, dict) or not isinstance(split_config.get("outputs"), list):
            raise PreflightError(f"data config requires {split}.outputs")
        split_x_path = split_config.get("x_path")
        if split_x_path is not None:
            if not isinstance(split_x_path, str) or not split_x_path:
                raise PreflightError(f"data config {split}.x_path must be a path")
            raw_paths.append(split_x_path)
        for index, output in enumerate(split_config["outputs"]):
            if not isinstance(output, dict) or not isinstance(output.get("y_path"), str):
                raise PreflightError(f"{split}.outputs[{index}] requires y_path")
            vector = output.get("g_vector")
            if not isinstance(vector, list) or not vector:
                raise PreflightError(f"{split}.outputs[{index}] requires a non-empty g_vector")
            raw_paths.append(output["y_path"])
    if len(raw_paths) > MAX_TRAINING_AUDIO_FILES:
        raise PreflightError(
            f"training config references more than {MAX_TRAINING_AUDIO_FILES} audio files"
        )
    paths = [
        _validated_training_audio_path(
            raw_path,
            config_directory=config_path.parent,
            data_root=root,
        )
        for raw_path in raw_paths
    ]
    seen: dict[str, Path] = {}
    total_bytes = 0
    for path in paths:
        total_bytes += path.stat().st_size
        if total_bytes > MAX_TRAINING_BUNDLE_BYTES:
            raise PreflightError("training audio exceeds the 64 GiB bundle limit")
        prior = seen.get(path.name)
        if prior is not None and prior != path:
            raise PreflightError(
                f"two training inputs share the basename {path.name!r}: {prior} and {path}"
            )
        seen[path.name] = path
    return list(dict.fromkeys(paths))


def _validated_training_audio_path(
    raw_path: str,
    *,
    config_directory: Path,
    data_root: Path,
) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = config_directory / candidate
    if candidate.is_symlink():
        raise PreflightError(f"training input must not be a symbolic link: {candidate}")
    path = candidate.resolve()
    if not path.is_relative_to(data_root):
        raise PreflightError(
            f"training input escapes the authorized data root {data_root}: {path}"
        )
    try:
        metadata = path.stat()
    except OSError as error:
        raise PreflightError(f"training input is unavailable: {path}") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > MAX_TRAINING_AUDIO_BYTES
        or path.suffix.lower() != ".wav"
    ):
        raise PreflightError(
            f"training input must be a non-empty regular .wav file no larger than 512 MiB: {path}"
        )
    try:
        with path.open("rb") as source:
            header = source.read(12)
    except OSError as error:
        raise PreflightError(f"training input cannot be read: {path}") from error
    if len(header) != 12 or header[:4] not in {b"RIFF", b"RF64"} or header[8:] != b"WAVE":
        raise PreflightError(f"training input does not have a WAV container header: {path}")
    return path


def plan_uploads(
    *,
    run_id: str,
    data_config: Path,
    model_config: Path,
    learning_config: Path,
    checkpoints: Iterable[tuple[str, Path]] = (),
    data_root: Path | None = None,
) -> list[InputObject]:
    if not _RUN_RE.fullmatch(run_id):
        raise PreflightError("run ID must use lowercase letters, numbers, and hyphens")
    config_paths = {
        "data.json": data_config.expanduser().resolve(),
        "model.json": model_config.expanduser().resolve(),
        "learning.json": learning_config.expanduser().resolve(),
    }
    for path in config_paths.values():
        _read_json_object(path)
    entries: list[tuple[Path, str]] = [
        (path, f"jobs/{run_id}/configs/{remote_name}")
        for remote_name, path in config_paths.items()
    ]
    entries.extend(
        (path, f"jobs/{run_id}/data/{path.name}")
        for path in load_training_inputs(config_paths["data.json"], data_root=data_root)
    )
    checkpoint_labels: set[str] = set()
    for label, raw_path in checkpoints:
        if not _CHECKPOINT_LABEL_RE.fullmatch(label) or label in checkpoint_labels:
            raise PreflightError(f"invalid or duplicate checkpoint label: {label!r}")
        path = raw_path.expanduser().resolve()
        if (
            path.suffix.lower() != ".ckpt"
            or not path.is_file()
            or path.stat().st_size <= 0
            or path.stat().st_size > 256 * 1024 * 1024
        ):
            raise PreflightError(f"checkpoint is missing, empty, oversized, or not .ckpt: {path}")
        checkpoint_labels.add(label)
        entries.append((path, f"jobs/{run_id}/checkpoints/{label}.ckpt"))
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
    model_name: str,
    timeout_minutes: int,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    epochs: int,
    seed: int,
    batch_size: int,
    ny: int,
    threshold_esr: float,
    input_level: float | None,
    output_level: float | None,
    checkpoint_name_value: str | None = None,
) -> dict[str, Any]:
    expected_image_prefix = f"europe-west3-docker.pkg.dev/{project}/"
    if (
        not image_uri.startswith(expected_image_prefix)
        or _GAR_DIGEST_RE.fullmatch(image_uri) is None
    ):
        raise PreflightError(
            "container image must be a digest-pinned image in the isolated "
            f"project's Frankfurt registry: {expected_image_prefix}...@sha256:<64 lowercase hex>"
        )
    expected_service_account_suffix = f"@{project}.iam.gserviceaccount.com"
    if not service_account.endswith(expected_service_account_suffix):
        raise PreflightError(
            "training service account must belong to the explicitly confirmed project"
        )
    if not 1 <= epochs <= 500:
        raise PreflightError("epochs must be between 1 and 500")
    if not 0 <= accelerator_count <= 4:
        raise PreflightError("accelerator count must be between 0 and 4")
    if not 1 <= batch_size <= 512 or not 256 <= ny <= 65536:
        raise PreflightError("batch size or training chunk length is outside the supported guardrail")
    source_prefix = f"jobs/{run_id}"
    args = [
        "--bucket", bucket,
        "--model-name", model_name,
        "--source-prefix", source_prefix,
        "--result-prefix", f"{source_prefix}/results/member-{seed:02d}",
        "--data-config-name", "data.json",
        "--model-config-name", "model.json",
        "--learning-config-name", "learning.json",
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--batch-size", str(batch_size),
        "--ny", str(ny),
        "--threshold-esr", str(threshold_esr),
    ]
    if (input_level is None) != (output_level is None):
        raise PreflightError("input and output dBu levels must be supplied together")
    if input_level is not None:
        args.extend(
            ["--input-level", str(input_level), "--output-level", str(output_level)]
        )
    if checkpoint_name_value is not None:
        try:
            validated_checkpoint_name = checkpoint_name(checkpoint_name_value)
        except argparse.ArgumentTypeError as error:
            raise PreflightError(str(error)) from error
        args.extend(["--checkpoint-name", validated_checkpoint_name])
    machine_spec: dict[str, Any] = {"machineType": machine_type}
    if accelerator_count:
        machine_spec.update(
            {
                "acceleratorType": accelerator_type,
                "acceleratorCount": accelerator_count,
            }
        )
    return {
        "serviceAccount": service_account,
        "workerPoolSpecs": [
            {
                "machineSpec": machine_spec,
                "replicaCount": 1,
                "containerSpec": {"imageUri": image_uri, "args": args},
            }
        ],
        "scheduling": {
            "timeout": f"{timeout_minutes * 60}s",
            "restartJobOnWorkerRestart": False,
        },
        "baseOutputDirectory": {
            "outputUriPrefix": f"gs://{bucket}/{source_prefix}/vertex-output"
        },
    }


def run_checked(command: Iterable[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), check=True, text=True, capture_output=True)


def _reject_legacy_mutation() -> None:
    raise PreflightError(LEGACY_MUTATION_ERROR)


def upload(args: argparse.Namespace) -> NoReturn:
    del args
    _reject_legacy_mutation()


def submit(args: argparse.Namespace) -> NoReturn:
    del args
    _reject_legacy_mutation()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0 or path.stat().st_size > 1024 * 1024:
        raise PreflightError(f"config is missing, empty, or over 1 MiB: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"invalid JSON config {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"config root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", required=True)
    parser.add_argument("--confirm-project", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="disabled legacy mutation; use conditioned_a2_runpod_job.py"
    )
    _common_parser(prepare)
    prepare.add_argument("--data-config", required=True, type=Path)
    prepare.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Authorized root for every WAV referenced by the data config; "
            "defaults to the data-config directory"
        ),
    )
    prepare.add_argument("--model-config", required=True, type=Path)
    prepare.add_argument("--learning-config", required=True, type=Path)
    prepare.add_argument(
        "--checkpoint",
        action="append",
        type=checkpoint_binding,
        default=[],
        help="optional LABEL=/absolute/path.ckpt initialization payload; repeat per member",
    )

    job = commands.add_parser(
        "submit", help="disabled legacy mutation; use conditioned_a2_runpod_job.py"
    )
    _common_parser(job)
    job.add_argument("--region", default="europe-west3")
    job.add_argument("--image-uri", required=True)
    job.add_argument("--service-account", required=True)
    job.add_argument("--model-name", required=True)
    job.add_argument("--machine-type", default="n1-standard-8")
    job.add_argument("--accelerator-type", default="NVIDIA_TESLA_T4")
    job.add_argument("--accelerator-count", type=int, default=1)
    job.add_argument("--epochs", type=int, default=1)
    job.add_argument("--seed", type=int, default=0)
    job.add_argument("--batch-size", type=int, default=32)
    job.add_argument("--ny", type=int, default=2048)
    job.add_argument("--threshold-esr", type=float, default=0.005)
    job.add_argument("--input-level", type=float)
    job.add_argument("--output-level", type=float)
    job.add_argument("--checkpoint-name", type=checkpoint_name)
    job.add_argument(
        "--purpose",
        choices=("training", "quality-training", "checkpoint-finetune", "pipeline-smoke"),
        default="training",
    )
    job.add_argument("--timeout-minutes", type=int, default=45)
    job.add_argument(
        "--hourly-rate-eur", type=decimal_eur,
        default=DEFAULT_T4_FRANKFURT_EUR_PER_HOUR,
    )
    job.add_argument("--overhead-reserve-eur", type=decimal_eur, default=Decimal("2.00"))
    job.add_argument("--max-cost-eur", type=decimal_eur, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = upload(args) if args.command == "prepare" else submit(args)
    except (PreflightError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            raise SystemExit(error.stderr.strip()) from error
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
