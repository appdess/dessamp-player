from __future__ import annotations

import asyncio
import json
import secrets
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from dessmetal_capture.domain import DomainError
from dessmetal_capture.local_http import (
    api_path_requires_token,
    normalized_local_request_path,
    reject_untrusted_browser_boundary,
)
from dessmetal_capture.wav_io import MAX_WAV_BYTES, load_mono_float32, write_pcm24_mono

from .artifact import MAX_RENDER_INPUT_BYTES, PlayerArtifact


class RenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_id: str
    settings: dict[str, float]
    quality: Optional[float] = Field(default=None, ge=0.0, le=1.0)


def create_app(
    *,
    manifest_path: Path | str,
    renderer_path: Path | str,
    data_dir: Path | str,
    default_input_path: Path | str | None = None,
    access_token: str | None = None,
) -> FastAPI:
    artifact = PlayerArtifact(manifest_path, renderer_path)
    root = Path(data_dir).expanduser().resolve()
    input_dir = root / "inputs"
    render_dir = root / "renders"
    input_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    token = access_token or secrets.token_urlsafe(32)
    static_dir = Path(__file__).resolve().parent / "static"
    inputs: dict[str, dict[str, Any]] = {}
    renders: dict[str, dict[str, Any]] = {}

    if default_input_path is not None:
        identifier = str(uuid.uuid4())
        source = Path(default_input_path).expanduser().resolve()
        path = input_dir / f"{identifier}.wav"
        info = _canonicalize_input(source, path, artifact)
        inputs[identifier] = {"id": identifier, "path": str(path), "info": info, "source": "startup"}

    app = FastAPI(title="DessMetal Player", docs_url=None, redoc_url=None)
    app.state.artifact = artifact
    app.state.data_dir = root
    app.state.inputs = inputs
    app.state.renders = renders
    app.state.access_token = token

    @app.exception_handler(DomainError)
    async def handle_domain_error(_request: Request, error: DomainError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.middleware("http")
    async def protect_local_api(request: Request, call_next):
        response = reject_untrusted_browser_boundary(request)
        if response is None and api_path_requires_token(normalized_local_request_path(request)):
            if not secrets.compare_digest(request.headers.get("x-player-token", ""), token):
                response = JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid local player token"},
                )
        if response is None:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self'; "
            "media-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("__PLAYER_TOKEN__", json.dumps(token)))

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "dessmetal-player", "schema_version": 1}

    @app.get("/api/session")
    async def session() -> dict[str, Any]:
        default_input = next(iter(inputs.values()), None)
        return {
            "artifact": artifact.describe(),
            "input": _public_input(default_input) if default_input else None,
            "renderer": str(artifact.renderer_path),
            "mode": "offline-core-render",
        }

    @app.post("/api/input")
    async def upload_input(request: Request) -> dict[str, Any]:
        identifier = str(uuid.uuid4())
        destination = input_dir / f"{identifier}.wav"
        await _stream_upload(request, destination, min(MAX_WAV_BYTES, MAX_RENDER_INPUT_BYTES))
        try:
            info = _canonicalize_input(destination, destination, artifact)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        record = {"id": identifier, "path": str(destination), "info": info, "source": "upload"}
        inputs[identifier] = record
        return _public_input(record)

    @app.post("/api/render")
    async def render(payload: RenderRequest) -> dict[str, Any]:
        input_record = inputs.get(payload.input_id)
        if input_record is None:
            raise DomainError("Unknown player input")
        identifier = str(uuid.uuid4())
        output_path = render_dir / f"{identifier}.wav"
        evidence = await asyncio.to_thread(
            artifact.render,
            input_path=Path(input_record["path"]),
            output_path=output_path,
            settings=payload.settings,
            quality=payload.quality,
        )
        record = {
            "id": identifier,
            "path": str(output_path),
            "audio_url": f"/audio/{identifier}.wav",
            **evidence,
        }
        renders[identifier] = record
        _trim_renders(renders, keep=20)
        return {key: value for key, value in record.items() if key != "path"}

    @app.get("/audio/{render_id}.wav")
    async def audio(render_id: str) -> FileResponse:
        record = renders.get(render_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Unknown render")
        return FileResponse(record["path"], media_type="audio/wav", filename=f"{render_id}.wav")

    return app


def _public_input(record: dict[str, Any]) -> dict[str, Any]:
    return {"id": record["id"], "source": record["source"], "info": record["info"]}


def _trim_renders(renders: dict[str, dict[str, Any]], *, keep: int) -> None:
    while len(renders) > keep:
        identifier = next(iter(renders))
        record = renders.pop(identifier)
        Path(record["path"]).unlink(missing_ok=True)


async def _stream_upload(request: Request, destination: Path, maximum_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4()}.upload")
    total = 0
    try:
        with temporary.open("wb") as output:
            async for chunk in request.stream():
                total += len(chunk)
                if total > maximum_bytes:
                    raise HTTPException(status_code=413, detail="Player input exceeds the 16 MiB limit")
                output.write(chunk)
        if total == 0:
            raise DomainError("Player input upload is empty")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _canonicalize_input(source: Path, destination: Path, artifact: PlayerArtifact) -> dict[str, Any]:
    """Copy supported input into the exact mono PCM24 subset accepted by NAM Core."""

    artifact.validate_input(source)
    audio, sample_rate = load_mono_float32(source)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4()}.canonical")
    try:
        write_pcm24_mono(temporary, audio, sample_rate)
        info = artifact.validate_input(temporary)
        temporary.replace(destination)
        return {**info, "path": str(destination.resolve())}
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
