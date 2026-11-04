from __future__ import annotations

import json
import secrets
import shutil
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .audio import AudioCaptureService
from .demo import bootstrap_dessblock_green_demo
from .domain import ControlDefinition, DomainError
from .local_http import (
    api_path_requires_token,
    normalized_local_request_path,
    reject_untrusted_browser_boundary,
)
from .manifests import import_model_artifact
from .panama import MAX_PANAMA_POINTS, parse_new_points, source_digest
from .store import CaptureStore, utc_now
from .training import TrainingJobManager, build_training_job
from .wav_io import MAX_WAV_BYTES, inspect_wav, validate_capture_wav


MAX_NAM_UPLOAD_BYTES = 256 * 1024 * 1024


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ControlRequest(StrictModel):
    id: str
    label: str
    minimum: float
    maximum: float
    step: float
    unit: str = ""
    taper: str = "linear"
    display_minimum: Optional[float] = None
    display_maximum: Optional[float] = None


class ProjectRequest(StrictModel):
    name: str
    amp_name: str
    description: str = ""
    controls: list[ControlRequest]
    candidate_points: Optional[list[dict[str, float]]] = None
    acquisition_strategy: str = "coverage-v1"
    sample_rate: int = 48_000


class DirectCaptureRequest(StrictModel):
    recommendation_id: str
    input_device: int = Field(ge=0)
    output_device: int = Field(ge=0)
    notes: str = ""


class ModelImportRequest(StrictModel):
    name: str
    provenance: dict[str, Any] = Field(default_factory=dict)


class PanamaImportRequest(StrictModel):
    round_index: int = Field(ge=0)
    control_order: list[str] = Field(min_length=1, max_length=12)
    points: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_PANAMA_POINTS)


class TrainingJobRequest(StrictModel):
    backend: str = "local-mps"
    epochs: int = Field(default=50, ge=1, le=10_000)
    batch_size: int = Field(default=32, ge=1, le=512)
    ny: int = Field(default=8192, ge=128, le=65_536)
    allow_capture_warnings: bool = False
    alignment_policy: str = "strict"
    workflow_validation_only: bool = False
    clip_start_seconds: Optional[float] = Field(default=None, ge=0.0)
    clip_duration_seconds: Optional[float] = Field(default=None, ge=1.0)


class CaptureJobManager:
    def __init__(self, store: CaptureStore, audio_service: AudioCaptureService):
        self.store = store
        self.audio_service = audio_service
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dessmetal-capture")
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def start(
        self,
        *,
        project_id: str,
        recommendation_id: str,
        input_device: int,
        output_device: int,
        notes: str,
    ) -> dict[str, Any]:
        project = self.store.get_project(project_id)
        recommendation = self.store.get_recommendation(recommendation_id)
        if recommendation["project_id"] != project_id or recommendation["status"] != "pending":
            raise DomainError("A pending recommendation from this project is required")
        input_signal = project.get("input_signal_path")
        input_info = project.get("input_signal_info")
        if not input_signal or not input_info:
            raise DomainError("Upload a standardized input signal before starting direct capture")
        job_id = str(uuid.uuid4())
        capture_id = str(uuid.uuid4())
        output_path = self.store.capture_dir / project_id / f"{capture_id}.wav"
        job = {
            "id": job_id,
            "project_id": project_id,
            "recommendation_id": recommendation_id,
            "capture_id": capture_id,
            "status": "queued",
            "created_at": utc_now(),
            "message": "Waiting for the audio capture worker",
        }
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(
            self._run,
            job_id=job_id,
            project=project,
            recommendation_id=recommendation_id,
            input_device=input_device,
            output_device=output_device,
            output_path=output_path,
            capture_id=capture_id,
            notes=notes,
        )
        return dict(job)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise DomainError(f"Unknown capture job {job_id!r}")
            return dict(job)

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def _run(
        self,
        *,
        job_id: str,
        project: Mapping[str, Any],
        recommendation_id: str,
        input_device: int,
        output_device: int,
        output_path: Path,
        capture_id: str,
        notes: str,
    ) -> None:
        self._update(job_id, status="recording", message="Playing the input signal and recording the return")
        try:
            info = self.audio_service.capture_duplex(
                input_signal_path=Path(project["input_signal_path"]),
                output_path=output_path,
                input_device=input_device,
                output_device=output_device,
            )
            quality, validation = validate_capture_wav(
                info,
                expected_sample_rate=int(project["sample_rate"]),
                expected_frames=int(project["input_signal_info"]["frames"]),
            )
            capture = self.store.add_capture(
                project_id=project["id"],
                recommendation_id=recommendation_id,
                source="direct-interface",
                wav_path=output_path,
                wav_info=info,
                quality_status=quality,
                validation={
                    **validation,
                    "input_device": input_device,
                    "output_device": output_device,
                },
                notes=notes,
                capture_id=capture_id,
            )
            self._update(
                job_id,
                status="complete",
                message="Capture completed",
                completed_at=utc_now(),
                capture=capture,
            )
        except Exception as error:
            if output_path.exists():
                output_path.unlink()
            self._update(
                job_id,
                status="failed",
                message=str(error),
                completed_at=utc_now(),
            )


def create_app(
    *,
    data_dir: Path | str,
    repository_root: Path | str | None = None,
    demo_root: Path | str | None = None,
    loader_path: Path | str | None = None,
    trainer_root: Path | str | None = None,
    trainer_python: Path | str | None = None,
    access_token: str | None = None,
    audio_service: AudioCaptureService | None = None,
) -> FastAPI:
    package_dir = Path(__file__).resolve().parent
    inferred_repository_root = package_dir.parents[2]
    repository = Path(repository_root).resolve() if repository_root else inferred_repository_root
    configured_demo_root = Path(demo_root).expanduser().resolve() if demo_root else None
    configured_loader = Path(loader_path).expanduser().resolve() if loader_path else None
    configured_trainer = Path(trainer_root).expanduser().resolve() if trainer_root else None
    token = access_token or secrets.token_urlsafe(32)
    store = CaptureStore(data_dir)
    audio = audio_service or AudioCaptureService()
    jobs = CaptureJobManager(store, audio)
    training_jobs = (
        TrainingJobManager(
            store,
            trainer_root=configured_trainer,
            loader_path=configured_loader,
            python_executable=trainer_python or sys.executable,
        )
        if configured_trainer is not None
        else None
    )
    static_dir = package_dir / "static"

    app = FastAPI(title="DessMetal Trainer", docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.access_token = token
    app.state.repository_root = repository
    app.state.demo_root = configured_demo_root
    app.state.loader_path = configured_loader
    app.state.audio_service = audio
    app.state.jobs = jobs
    app.state.trainer_root = configured_trainer
    app.state.training_jobs = training_jobs
    if training_jobs is not None:
        app.add_event_handler("shutdown", training_jobs.close)

    @app.exception_handler(DomainError)
    async def handle_domain_error(_request: Request, error: DomainError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.middleware("http")
    async def protect_local_api(request: Request, call_next):
        response = reject_untrusted_browser_boundary(request)
        if response is None and api_path_requires_token(normalized_local_request_path(request)):
            if not secrets.compare_digest(request.headers.get("x-capture-token", ""), token):
                response = JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid local application token"},
                )
        if response is None:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("__CAPTURE_TOKEN__", json.dumps(token)))

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "dessmetal-trainer", "schema_version": 3}

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return {
            "data_dir": str(store.data_dir),
            "demo_available": configured_demo_root is not None,
            "loader_available": configured_loader is not None and configured_loader.is_file(),
            "trainer_available": training_jobs is not None,
            "trainer_root": str(configured_trainer) if configured_trainer else None,
            "acquisition_strategies": [
                {
                    "id": "coverage-v1",
                    "name": "Coverage baseline",
                    "production": False,
                    "description": "Farthest-point workflow baseline; not PANAMA disagreement.",
                },
                {
                    "id": "panama",
                    "name": "PANAMA active learner",
                    "production": False,
                    "available": True,
                    "trainer_available": False,
                    "local_worker_available": bool(
                        configured_trainer is not None
                        and (configured_trainer / "local_active_learner.py").is_file()
                    ),
                    "description": (
                        "Persistent new_points.json adapter with a sequential MPS/CPU development worker; "
                        "background launch and resume are not wired into the Trainer UI yet."
                    ),
                },
            ],
        }

    @app.get("/api/projects")
    async def list_projects() -> list[dict[str, Any]]:
        return store.list_projects()

    @app.post("/api/projects")
    async def create_project(payload: ProjectRequest) -> dict[str, Any]:
        return store.create_project(
            name=payload.name,
            amp_name=payload.amp_name,
            description=payload.description,
            controls=[control.model_dump() for control in payload.controls],
            candidate_points=payload.candidate_points,
            acquisition_strategy=payload.acquisition_strategy,
            sample_rate=payload.sample_rate,
        )

    @app.get("/api/projects/{project_id}")
    async def get_project(project_id: str) -> dict[str, Any]:
        return {
            "project": store.get_project(project_id),
            "recommendations": store.list_recommendations(project_id),
            "external_suggestions": store.list_external_suggestions(project_id),
            "captures": store.list_captures(project_id),
            "models": store.list_model_artifacts(project_id),
            "training_jobs": store.list_training_jobs(project_id),
        }

    @app.post("/api/projects/{project_id}/training/jobs")
    async def create_training_job(project_id: str, payload: TrainingJobRequest) -> dict[str, Any]:
        if configured_trainer is None:
            raise DomainError("Start DessMetal Trainer with --trainer-root to prepare training jobs")
        return build_training_job(
            store,
            project_id=project_id,
            trainer_root=configured_trainer,
            backend=payload.backend,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            ny=payload.ny,
            allow_capture_warnings=payload.allow_capture_warnings,
            alignment_policy=payload.alignment_policy,
            workflow_validation_only=payload.workflow_validation_only,
            clip_start_seconds=payload.clip_start_seconds,
            clip_duration_seconds=payload.clip_duration_seconds,
        )

    @app.post("/api/training/jobs/{job_id}/start")
    async def start_training_job(job_id: str) -> dict[str, Any]:
        if training_jobs is None:
            raise DomainError("The local training worker is unavailable")
        return training_jobs.start(job_id)

    @app.get("/api/training/jobs/{job_id}")
    async def get_training_job(job_id: str) -> dict[str, Any]:
        if training_jobs is None:
            return store.get_training_job(job_id)
        return training_jobs.get(job_id)

    @app.post("/api/projects/{project_id}/input-signal")
    async def upload_input_signal(project_id: str, request: Request) -> dict[str, Any]:
        project = store.get_project(project_id)
        destination = store.input_dir / project_id / "input.wav"
        await _stream_upload(request, destination, MAX_WAV_BYTES)
        try:
            info = inspect_wav(destination)
            if info["channels"] != 1:
                raise DomainError("The standardized input signal must be mono")
            if info["sample_rate"] != project["sample_rate"]:
                raise DomainError(
                    f"Input signal is {info['sample_rate']} Hz; project requires {project['sample_rate']} Hz"
                )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return store.set_input_signal(project_id, destination, info)

    @app.post("/api/projects/{project_id}/recommendations")
    async def recommend(project_id: str) -> dict[str, Any]:
        return store.create_recommendation(project_id)

    @app.post("/api/projects/{project_id}/recommendations/import-panama")
    async def import_panama_recommendations(
        project_id: str, payload: PanamaImportRequest
    ) -> dict[str, Any]:
        project = store.get_project(project_id)
        controls = [ControlDefinition.from_mapping(control) for control in project["controls"]]
        suggestions = parse_new_points(
            payload.points,
            controls=controls,
            control_order=payload.control_order,
        )
        digest = source_digest(payload.points)
        queued = store.enqueue_panama_suggestions(
            project_id,
            round_index=payload.round_index,
            source_sha256=digest,
            control_order=payload.control_order,
            suggestions=suggestions,
        )
        return {
            "source_sha256": digest,
            "round_index": payload.round_index,
            "queued_count": sum(item["status"] == "queued" for item in queued),
            "suggestions": queued,
        }

    @app.post("/api/recommendations/{recommendation_id}/dismiss")
    async def dismiss(recommendation_id: str) -> dict[str, Any]:
        return store.dismiss_recommendation(recommendation_id)

    @app.get("/api/audio/devices")
    async def audio_devices() -> list[dict[str, Any]]:
        return audio.list_devices()

    @app.post("/api/projects/{project_id}/captures/direct")
    async def direct_capture(project_id: str, payload: DirectCaptureRequest) -> dict[str, Any]:
        return jobs.start(
            project_id=project_id,
            recommendation_id=payload.recommendation_id,
            input_device=payload.input_device,
            output_device=payload.output_device,
            notes=payload.notes,
        )

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        return jobs.get(job_id)

    @app.post("/api/projects/{project_id}/captures/import")
    async def import_capture(
        project_id: str,
        request: Request,
        recommendation_id: str = Query(...),
        notes: str = Query(""),
    ) -> dict[str, Any]:
        project = store.get_project(project_id)
        recommendation = store.get_recommendation(recommendation_id)
        if recommendation["project_id"] != project_id or recommendation["status"] != "pending":
            raise DomainError("A pending recommendation from this project is required")
        capture_id = str(uuid.uuid4())
        destination = store.capture_dir / project_id / f"{capture_id}.wav"
        await _stream_upload(request, destination, MAX_WAV_BYTES)
        try:
            info = inspect_wav(destination)
            input_info = project.get("input_signal_info")
            quality, validation = validate_capture_wav(
                info,
                expected_sample_rate=project["sample_rate"],
                expected_frames=int(input_info["frames"]) if input_info else None,
            )
            return store.add_capture(
                project_id=project_id,
                recommendation_id=recommendation_id,
                source="logic-import",
                wav_path=destination,
                wav_info=info,
                quality_status=quality,
                validation=validation,
                notes=notes,
                capture_id=capture_id,
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    @app.post("/api/projects/{project_id}/captures/replay")
    async def replay_capture(project_id: str, recommendation_id: str = Query(...)) -> dict[str, Any]:
        project = store.get_project(project_id)
        recommendation = store.get_recommendation(recommendation_id)
        if recommendation["project_id"] != project_id or recommendation["status"] != "pending":
            raise DomainError("A pending recommendation from this project is required")
        wav_path = store.get_replay_asset(project_id, recommendation["settings"])
        info = inspect_wav(wav_path)
        input_info = project.get("input_signal_info")
        quality, validation = validate_capture_wav(
            info,
            expected_sample_rate=project["sample_rate"],
            expected_frames=int(input_info["frames"]) if input_info else None,
        )
        validation["replay_reference"] = True
        return store.add_capture(
            project_id=project_id,
            recommendation_id=recommendation_id,
            source="replay-dataset",
            wav_path=wav_path,
            wav_info=info,
            quality_status=quality,
            validation=validation,
            notes="Existing recording replayed for workflow validation; no audio was recorded.",
        )

    @app.post("/api/demo/dessblock-green")
    async def create_demo() -> dict[str, Any]:
        if configured_demo_root is None:
            raise DomainError("Start the app with --demo-root to enable the replay dataset")
        return bootstrap_dessblock_green_demo(
            store,
            repository_root=repository,
            capture_root=configured_demo_root,
        )

    @app.post("/api/projects/{project_id}/models/import")
    async def import_model(
        project_id: str,
        request: Request,
        name: str = Query(...),
        provenance_json: str = Query("{}"),
    ) -> dict[str, Any]:
        store.get_project(project_id)
        try:
            provenance = json.loads(provenance_json)
        except json.JSONDecodeError as error:
            raise DomainError("Model provenance must be valid JSON") from error
        if not isinstance(provenance, dict):
            raise DomainError("Model provenance must be a JSON object")
        incoming = store.data_dir / "incoming" / f"{uuid.uuid4()}.nam"
        await _stream_upload(request, incoming, MAX_NAM_UPLOAD_BYTES)
        try:
            return import_model_artifact(
                store,
                project_id=project_id,
                source_model_path=incoming,
                name=name,
                provenance=provenance,
                loader_path=configured_loader,
            )
        finally:
            incoming.unlink(missing_ok=True)

    @app.post("/api/projects/{project_id}/models/demo-fixture")
    async def import_demo_model(project_id: str) -> dict[str, Any]:
        project = store.get_project(project_id)
        fixture = project["metadata"].get("fixture_model_path")
        if not fixture:
            raise DomainError("This project does not define a demo model fixture")
        return import_model_artifact(
            store,
            project_id=project_id,
            source_model_path=Path(fixture),
            name=f"{project['amp_name']} replay model",
            provenance={
                "training_backend": "pretrained-fixture",
                "workflow_validation_only": True,
                "note": "The Trainer did not retrain this model; it packages the existing model for player handoff.",
            },
            loader_path=configured_loader,
        )

    @app.get("/api/models/{artifact_id}/manifest")
    async def download_manifest(artifact_id: str) -> FileResponse:
        """Legacy two-file handoff retained for development and migration."""
        artifact = store.get_model_artifact(artifact_id)
        path = Path(artifact["manifest_path"])
        return FileResponse(path, media_type="application/json", filename="player-manifest.json")

    @app.get("/api/models/{artifact_id}/package")
    async def download_package(artifact_id: str) -> FileResponse:
        """Download the self-contained, user-facing DessMetal amp model."""
        artifact = store.get_model_artifact(artifact_id)
        path = Path(artifact["manifest_path"]).parent / "model.dess"
        if not path.is_file():
            raise DomainError("The self-contained .dess package is unavailable")
        return FileResponse(
            path,
            media_type="application/vnd.dessmetal.amp-model",
            filename="DessMetal-Amp.dess",
        )

    return app


async def _stream_upload(request: Request, destination: Path, maximum_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4()}.upload")
    total = 0
    try:
        with temporary.open("wb") as output:
            async for chunk in request.stream():
                total += len(chunk)
                if total > maximum_bytes:
                    raise HTTPException(status_code=413, detail="Upload exceeds the application size limit")
                output.write(chunk)
        if total == 0:
            raise DomainError("Upload is empty")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
