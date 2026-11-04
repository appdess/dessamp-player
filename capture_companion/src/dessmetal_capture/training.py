from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Optional

from .domain import ControlDefinition, DomainError, settings_key
from .manifests import import_model_artifact, sha256_file
from .store import CaptureStore


SUPPORTED_BACKENDS = {"local-mps", "local-cpu"}


def build_training_job(
    store: CaptureStore,
    *,
    project_id: str,
    trainer_root: Path | str,
    backend: str,
    epochs: int,
    batch_size: int,
    ny: int,
    allow_capture_warnings: bool,
    alignment_policy: str,
    workflow_validation_only: bool,
    clip_start_seconds: Optional[float] = None,
    clip_duration_seconds: Optional[float] = None,
) -> dict[str, Any]:
    trainer = Path(trainer_root).expanduser().resolve()
    _validate_trainer_root(trainer)
    if backend not in SUPPORTED_BACKENDS:
        raise DomainError(f"Training backend must be one of {sorted(SUPPORTED_BACKENDS)}")
    if isinstance(epochs, bool) or not 1 <= int(epochs) <= 10_000:
        raise DomainError("Training epochs must be between 1 and 10000")
    if isinstance(batch_size, bool) or not 1 <= int(batch_size) <= 512:
        raise DomainError("Training batch size must be between 1 and 512")
    if isinstance(ny, bool) or not 128 <= int(ny) <= 65_536:
        raise DomainError("Training chunk size must be between 128 and 65536 samples")
    if alignment_policy not in {"strict", "truncate-to-shortest"}:
        raise DomainError("Alignment policy must be strict or truncate-to-shortest")
    if (clip_start_seconds is None) != (clip_duration_seconds is None):
        raise DomainError("Training clip start and duration must be supplied together")
    if clip_start_seconds is not None:
        clip_start_seconds = _finite_float(clip_start_seconds, "clip_start_seconds")
        clip_duration_seconds = _finite_float(clip_duration_seconds, "clip_duration_seconds")
        if clip_start_seconds < 0.0 or clip_duration_seconds < 1.0:
            raise DomainError("Training clip start must be non-negative and duration at least one second")
        if not workflow_validation_only:
            raise DomainError("Clipped training jobs must be marked workflow_validation_only")

    project = store.get_project(project_id)
    controls = [ControlDefinition.from_mapping(control) for control in project["controls"]]
    control_order = [control.id for control in controls]
    input_path_raw = project.get("input_signal_path")
    input_info = project.get("input_signal_info")
    if not input_path_raw or not input_info:
        raise DomainError("A standardized input WAV is required before preparing training")
    input_path = Path(input_path_raw)
    if not input_path.is_file() or sha256_file(input_path) != input_info["sha256"]:
        raise DomainError("The standardized input WAV is missing or its hash changed")
    if clip_start_seconds is not None:
        clip_stop = clip_start_seconds + clip_duration_seconds
        if clip_stop > float(input_info["duration_seconds"]):
            raise DomainError("Workflow-validation clip extends beyond the standardized input")

    captures = sorted(store.list_captures(project_id), key=lambda capture: settings_key(capture["settings"]))
    usable: list[dict[str, Any]] = []
    seen_settings: set[str] = set()
    source_captures: list[dict[str, Any]] = []
    for capture in captures:
        if capture["quality_status"] == "failed":
            continue
        if capture["quality_status"] == "warning" and not allow_capture_warnings:
            raise DomainError(
                "Capture warnings must be resolved or explicitly admitted for a workflow-validation job"
            )
        wav_path = Path(capture["wav_path"])
        if not wav_path.is_file() or sha256_file(wav_path) != capture["wav_sha256"]:
            raise DomainError(f"Capture {capture['id']} is missing or its hash changed")
        frame_delta = int(capture["frames"]) - int(input_info["frames"])
        if frame_delta != 0 and alignment_policy == "strict":
            raise DomainError(
                f"Capture {capture['id']} differs from the input by {frame_delta} frames; choose and record an alignment policy"
            )
        if set(capture["settings"]) != set(control_order):
            raise DomainError(f"Capture {capture['id']} does not contain the project's exact control schema")
        key = settings_key(capture["settings"])
        if key in seen_settings:
            continue
        seen_settings.add(key)
        output = {
            "y_path": str(wav_path.resolve()),
            "g_vector": [float(capture["settings"][control_id]) for control_id in control_order],
        }
        usable.append(output)
        source_captures.append(
            {
                "id": capture["id"],
                "wav_sha256": capture["wav_sha256"],
                "frames": int(capture["frames"]),
                "frame_delta": frame_delta,
                "quality_status": capture["quality_status"],
                "settings": capture["settings"],
            }
        )
    if len(usable) < 2:
        raise DomainError("A conditioned training job requires at least two distinct captured settings")

    model_template_path = trainer / "parametric_wavenet.json"
    if not model_template_path.is_file():
        model_template_path = trainer / "default_config_files" / "models" / "wavenet.json"
    learning_template_path = trainer / "default_config_files" / "learning" / "default.json"
    try:
        model_config = json.loads(model_template_path.read_text(encoding="utf-8"))
        learning_config = json.loads(learning_template_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DomainError(f"Unable to read PANAMA trainer templates: {error}") from error
    model_config = _condition_model_config(model_config, len(controls))
    receptive_field = _wavenet_receptive_field(model_config)
    learning_config = _configure_learning(
        learning_config,
        backend=backend,
        epochs=int(epochs),
        batch_size=int(batch_size),
    )
    validation_index = min(
        range(len(usable)),
        key=lambda output_index: sum(
            (
                (usable[output_index]["g_vector"][index] - ((control.minimum + control.maximum) / 2.0))
                / (control.maximum - control.minimum)
            )
            ** 2
            for index, control in enumerate(controls)
        ),
    )
    validation_output = usable[validation_index]
    if len(usable) >= 3:
        training_outputs = [output for index, output in enumerate(usable) if index != validation_index]
        validation_mode = "held-out-control-setting"
    elif workflow_validation_only:
        training_outputs = usable
        validation_mode = "duplicated-functional-fixture"
    else:
        raise DomainError("A quality training job requires at least three settings so one can be held out")
    common: dict[str, Any] = {
        "nx": receptive_field,
        "input_gain": 0.0,
        "delay": 0,
        "allow_unequal_lengths": alignment_policy == "truncate-to-shortest",
        "x_path": str(input_path.resolve()),
    }
    if clip_start_seconds is not None:
        common.update(
            {
                "process_x": True,
                "check_y": True,
                "start_seconds": clip_start_seconds,
                "stop_seconds": clip_start_seconds + clip_duration_seconds,
                "require_input_pre_silence": None,
            }
        )
    all_frame_counts = [int(input_info["frames"]), *[int(capture["frames"]) for capture in source_captures]]
    if clip_duration_seconds is not None:
        sample_rate = int(input_info["sample_rate"])
        start_frame = int(math.floor(float(clip_start_seconds) * sample_rate))
        requested_frames = int(math.floor(float(clip_duration_seconds) * sample_rate))
        effective_frames = min(requested_frames, *[max(0, frames - start_frame) for frames in all_frame_counts])
    else:
        effective_frames = min(all_frame_counts)
    minimum_frames = receptive_field + int(ny) - 1
    if effective_frames < minimum_frames:
        raise DomainError(
            f"Training audio provides {effective_frames} effective frames; this WaveNet and chunk size require "
            f"at least {minimum_frames}"
        )
    chunks_per_capture = (effective_frames - receptive_field + 1) // int(ny)
    estimated_training_chunks = chunks_per_capture * len(training_outputs)
    if estimated_training_chunks < int(batch_size):
        raise DomainError(
            f"Training job would create only {estimated_training_chunks} chunks, fewer than batch size {batch_size}"
        )
    data_config = {
        "type": "shared_in_multi_out",
        "common": common,
        "train": {"outputs": training_outputs},
        "validation": {"outputs": [copy.deepcopy(validation_output)]},
    }

    job_id = str(uuid.uuid4())
    job_dir = store.training_dir / project_id / job_id
    output_dir = job_dir / "trainer-output"
    log_path = job_dir / "training.log"
    job_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    data_path = job_dir / "data-config.json"
    model_path = job_dir / "model-config.json"
    learning_path = job_dir / "learning-config.json"
    options = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "ny": int(ny),
        "allow_capture_warnings": bool(allow_capture_warnings),
        "alignment_policy": alignment_policy,
        "workflow_validation_only": bool(workflow_validation_only),
        "clip_start_seconds": clip_start_seconds,
        "clip_duration_seconds": clip_duration_seconds,
        "receptive_field": receptive_field,
        "effective_frames": effective_frames,
        "estimated_training_chunks": estimated_training_chunks,
        "validation_mode": validation_mode,
        "validation_capture_id": source_captures[validation_index]["id"],
        "validation_settings": source_captures[validation_index]["settings"],
    }
    source_evidence = {
        "input": {
            "path": str(input_path.resolve()),
            "sha256": input_info["sha256"],
            "frames": int(input_info["frames"]),
        },
        "captures": source_captures,
        "trainer": {
            "root": str(trainer),
            "license": str((trainer / "LICENSE").resolve()),
            "license_identifier": "MIT",
            "license_sha256": sha256_file(trainer / "LICENSE"),
            "implementation": "PANAMA conditioned WaveNet",
            "source_files": {
                "custom_train_full.py": sha256_file(trainer / "custom_train_full.py"),
                "export_model.py": sha256_file(trainer / "export_model.py"),
                model_template_path.name: sha256_file(model_template_path),
                "default_config_files/learning/default.json": sha256_file(learning_template_path),
            },
        },
        "runtime": {
            "implementation": "NeuralAmpModelerCore conditioned WaveNet",
            "a2_reuse_boundary": "runtime kernels, strict dispatch, prewarm, and optional container quality only",
        },
    }
    _write_json(data_path, data_config)
    _write_json(model_path, model_config)
    _write_json(learning_path, learning_config)
    _write_json(
        job_dir / "training-job.json",
        {
            "schema_version": 1,
            "id": job_id,
            "project_id": project_id,
            "project_name": project["name"],
            "control_order": control_order,
            "backend": backend,
            "options": options,
            "source_evidence": source_evidence,
            "configs": {
                "data": data_path.name,
                "model": model_path.name,
                "learning": learning_path.name,
            },
            "expected_output": "one self-contained .dess amp package (legacy manifest retained internally)",
        },
    )
    return store.add_training_job(
        project_id=project_id,
        backend=backend,
        job_dir=job_dir,
        data_config_path=data_path,
        model_config_path=model_path,
        learning_config_path=learning_path,
        output_dir=output_dir,
        log_path=log_path,
        control_order=control_order,
        options=options,
        source_evidence=source_evidence,
        job_id=job_id,
    )


class TrainingJobManager:
    def __init__(
        self,
        store: CaptureStore,
        *,
        trainer_root: Path | str,
        loader_path: Path | str | None,
        python_executable: Path | str = sys.executable,
    ):
        self.store = store
        self.trainer_root = Path(trainer_root).expanduser().resolve()
        self.loader_path = Path(loader_path).expanduser().resolve() if loader_path else None
        self.python_executable = Path(python_executable).expanduser().resolve()
        _validate_trainer_root(self.trainer_root)
        if not self.python_executable.is_file():
            raise DomainError(f"Training Python executable is unavailable: {self.python_executable}")
        self.store.fail_incomplete_training_jobs("Trainer stopped before the job completed")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dessmetal-trainer")
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._closing = False

    def start(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_training_job(job_id)
        if job["status"] != "ready":
            raise DomainError("Only a ready training job can be started")
        with self._lock:
            if self._closing:
                raise DomainError("The local trainer is shutting down")
            if self._active:
                raise DomainError("The local trainer is already running a job")
            self._active.add(job_id)
        queued = self.store.update_training_job(job_id, status="queued")
        self._executor.submit(self._run, job_id)
        return queued

    def get(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_training_job(job_id)
        log_path = Path(job["log_path"])
        if log_path.is_file():
            with log_path.open("rb") as source:
                source.seek(max(0, log_path.stat().st_size - 64 * 1024))
                job["log_tail"] = source.read().decode("utf-8", errors="replace")
        else:
            job["log_tail"] = ""
        return job

    def close(self) -> None:
        """Stop an active child process so quitting the standalone app cannot orphan training."""

        with self._lock:
            if self._closing:
                return
            self._closing = True
            processes = list(self._processes.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _run(self, job_id: str) -> None:
        try:
            job = self.store.update_training_job(job_id, status="running")
            log_path = Path(job["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(self.python_executable),
                str(self.trainer_root / "custom_train_full.py"),
                "--base-outdir",
                job["output_dir"],
                f"--outdir-suffix=-{job_id[:8]}",
                "--data-config",
                job["data_config_path"],
                "--model-config",
                job["model_config_path"],
                "--learning-config",
                job["learning_config_path"],
                "--batch-size",
                str(job["options"]["batch_size"]),
                "--check-val-every-n-epoch",
                "1",
                "--epochs",
                str(job["options"]["epochs"]),
                "--ny",
                str(job["options"]["ny"]),
            ]
            self._run_command(job_id, command, log_path, cwd=self.trainer_root.parent)
            output_root = Path(job["output_dir"])
            run_directories = [path for path in output_root.iterdir() if path.is_dir()]
            if not run_directories:
                raise DomainError("PANAMA trainer completed without creating a run directory")
            run_directory = max(run_directories, key=lambda path: path.stat().st_mtime_ns)
            exported_dir = Path(job["job_dir"]) / "exported-model"
            exported_dir.mkdir(parents=True, exist_ok=False)
            self._run_command(
                job_id,
                [
                    str(self.python_executable),
                    str(self.trainer_root / "export_model.py"),
                    str(run_directory),
                    str(exported_dir),
                ],
                log_path,
                cwd=self.trainer_root.parent,
            )
            exported_model = exported_dir / "model.nam"
            if not exported_model.is_file():
                raise DomainError("PANAMA export completed without model.nam")
            training_metrics = _read_training_metrics(exported_dir / "model_stats.json")
            project = self.store.get_project(job["project_id"])
            artifact = import_model_artifact(
                self.store,
                project_id=job["project_id"],
                source_model_path=exported_model,
                name=f"{project['amp_name']} conditioned {job_id[:8]}",
                provenance={
                    "training_backend": job["backend"],
                    "training_job_id": job_id,
                    "control_order": job["control_order"],
                    "workflow_validation_only": job["options"]["workflow_validation_only"],
                    "alignment_policy": job["options"]["alignment_policy"],
                    "training_metrics": training_metrics,
                    "source_evidence": job["source_evidence"],
                },
                loader_path=self.loader_path,
            )
            self.store.update_training_job(
                job_id,
                status="completed",
                result={
                    "model_artifact_id": artifact["id"],
                    "manifest_path": artifact["manifest_path"],
                    "model_sha256": artifact["model_sha256"],
                    "validation_status": artifact["validation_status"],
                    "training_metrics": training_metrics,
                },
            )
        except Exception as error:
            try:
                self.store.update_training_job(job_id, status="failed", error_message=str(error)[-4_000:])
            except Exception:
                pass
        finally:
            with self._lock:
                self._active.discard(job_id)

    def _run_command(self, job_id: str, command: list[str], log_path: Path, *, cwd: Path) -> None:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n$ " + " ".join(json.dumps(argument) for argument in command) + "\n")
            log.flush()
            with self._lock:
                if self._closing:
                    raise DomainError("Trainer stopped before the command could start")
                process = subprocess.Popen(
                    command,
                    cwd=str(cwd),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self._processes[job_id] = process
            try:
                return_code = process.wait()
            finally:
                with self._lock:
                    self._processes.pop(job_id, None)
            if return_code != 0:
                raise DomainError(f"Training command exited with status {return_code}; inspect training.log")


def _validate_trainer_root(trainer: Path) -> None:
    required = [
        trainer / "LICENSE",
        trainer / "custom_train_full.py",
        trainer / "export_model.py",
        trainer / "default_config_files" / "learning" / "default.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise DomainError(f"PANAMA trainer source is incomplete: {missing}")


def _condition_model_config(raw: Any, condition_size: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise DomainError("PANAMA model template root must be an object")
    config = copy.deepcopy(raw)
    try:
        net = config["net"]
        layers = net["config"]["layers_configs"]
    except (KeyError, TypeError) as error:
        raise DomainError("PANAMA model template is not a conditioned WaveNet config") from error
    if net.get("name") != "WaveNet" or not isinstance(layers, list) or not layers:
        raise DomainError("PANAMA model template must define WaveNet layers")
    net["global_condition_size"] = condition_size
    for layer in layers:
        if not isinstance(layer, dict):
            raise DomainError("PANAMA WaveNet layer config must be an object")
        layer["global_condition_size"] = condition_size
    return config


def _configure_learning(
    raw: Any, *, backend: str, epochs: int, batch_size: int
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise DomainError("PANAMA learning template root must be an object")
    config = copy.deepcopy(raw)
    try:
        dataloader = config["train_dataloader"]
        trainer = config["trainer"]
    except (KeyError, TypeError) as error:
        raise DomainError("PANAMA learning template is incomplete") from error
    dataloader.update(
        {
            "batch_size": batch_size,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": True,
        }
    )
    trainer.update(
        {
            "accelerator": "mps" if backend == "local-mps" else "cpu",
            "devices": 1,
            "max_epochs": epochs,
            "check_val_every_n_epoch": 1,
        }
    )
    return config


def _wavenet_receptive_field(config: Mapping[str, Any]) -> int:
    try:
        layers = config["net"]["config"]["layers_configs"]
    except (KeyError, TypeError) as error:
        raise DomainError("PANAMA WaveNet config is missing its layers") from error
    receptive_field = 1
    for layer in layers:
        kernel_size = layer.get("kernel_size") if isinstance(layer, dict) else None
        dilations = layer.get("dilations") if isinstance(layer, dict) else None
        if (
            isinstance(kernel_size, bool)
            or not isinstance(kernel_size, int)
            or kernel_size < 1
            or not isinstance(dilations, list)
            or not dilations
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in dilations)
        ):
            raise DomainError("PANAMA WaveNet layer has an invalid kernel or dilation schedule")
        receptive_field += (kernel_size - 1) * sum(dilations)
    return receptive_field


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_training_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0 or path.stat().st_size > 64 * 1024:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    metrics: dict[str, Any] = {}
    for key in ("ESR", "MSE"):
        if key in raw:
            try:
                value = float(raw[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                metrics[key.lower()] = value
    checkpoint = raw.get("Checkpoint")
    if isinstance(checkpoint, str) and 0 < len(checkpoint) <= 512:
        metrics["checkpoint"] = checkpoint
    return metrics


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise DomainError(f"{label} must be numeric") from error
    if not math.isfinite(parsed):
        raise DomainError(f"{label} must be finite")
    return parsed
