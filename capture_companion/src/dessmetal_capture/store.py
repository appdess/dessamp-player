from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .acquisition import get_strategy
from .domain import (
    ControlDefinition,
    DomainError,
    generate_candidate_points,
    settings_display,
    settings_key,
    validate_candidate_points,
    validate_controls,
)
from .panama import PanamaSuggestion


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CaptureStore:
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.capture_dir = self.data_dir / "captures"
        self.model_dir = self.data_dir / "models"
        self.input_dir = self.data_dir / "input-signals"
        self.training_dir = self.data_dir / "training-jobs"
        for directory in (self.capture_dir, self.model_dir, self.input_dir, self.training_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.database_path = self.data_dir / "capture.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def backup(self, destination_path: Path | str) -> Path:
        """Create a transactionally consistent, integrity-checked SQLite backup."""

        destination = Path(destination_path).expanduser().resolve()
        if destination.exists():
            raise DomainError(f"Refusing to replace existing database backup: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_uri = f"{self.database_path.as_uri()}?mode=ro"
        try:
            with sqlite3.connect(source_uri, uri=True, timeout=10.0) as source:
                with sqlite3.connect(destination, timeout=10.0) as target:
                    source.backup(target)
                    integrity = target.execute("PRAGMA integrity_check").fetchone()
                    if integrity is None or integrity[0] != "ok":
                        raise DomainError(
                            f"SQLite backup integrity check failed: {integrity!r}"
                        )
        except Exception:
            if destination.exists():
                destination.unlink()
            raise
        return destination

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    amp_name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    controls_json TEXT NOT NULL,
                    candidate_points_json TEXT NOT NULL,
                    acquisition_strategy TEXT NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    input_signal_path TEXT,
                    input_signal_info_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recommendations (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    round_index INTEGER NOT NULL,
                    settings_json TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    score REAL NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'captured', 'dismissed')),
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS one_pending_recommendation_per_project
                ON recommendations(project_id) WHERE status = 'pending';

                CREATE TABLE IF NOT EXISTS external_suggestions (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    batch_id TEXT NOT NULL,
                    batch_position INTEGER NOT NULL,
                    source_index INTEGER NOT NULL,
                    settings_json TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    score REAL NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    control_order_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued', 'activated', 'captured', 'dismissed')),
                    recommendation_id TEXT UNIQUE REFERENCES recommendations(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(project_id, batch_id, batch_position)
                );

                CREATE INDEX IF NOT EXISTS external_suggestion_queue
                ON external_suggestions(project_id, status, created_at, batch_position);

                CREATE TABLE IF NOT EXISTS captures (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    recommendation_id TEXT REFERENCES recommendations(id) ON DELETE SET NULL,
                    settings_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    wav_path TEXT NOT NULL,
                    wav_sha256 TEXT NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    sample_width_bits INTEGER NOT NULL,
                    frames INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    peak REAL NOT NULL,
                    quality_status TEXT NOT NULL CHECK(quality_status IN ('pass', 'warning', 'failed')),
                    validation_json TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS replay_assets (
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    settings_key TEXT NOT NULL,
                    wav_path TEXT NOT NULL,
                    PRIMARY KEY(project_id, settings_key)
                );

                CREATE TABLE IF NOT EXISTS model_artifacts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    model_format_version TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    validation_status TEXT NOT NULL CHECK(validation_status IN ('valid', 'invalid', 'unverified')),
                    validation_message TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_jobs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    status TEXT NOT NULL CHECK(status IN ('ready', 'queued', 'running', 'completed', 'failed')),
                    backend TEXT NOT NULL,
                    job_dir TEXT NOT NULL,
                    data_config_path TEXT NOT NULL,
                    model_config_path TEXT NOT NULL,
                    learning_config_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    control_order_json TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    source_evidence_json TEXT NOT NULL,
                    result_json TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS training_jobs_by_project
                ON training_jobs(project_id, created_at DESC);

                PRAGMA user_version = 3;
                """
            )

    def create_project(
        self,
        *,
        name: str,
        amp_name: str,
        controls: Sequence[Mapping[str, Any]],
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
        candidate_points: Sequence[Mapping[str, Any]] | None = None,
        acquisition_strategy: str = "coverage-v1",
        sample_rate: int = 48_000,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        clean_name = name.strip()
        clean_amp_name = amp_name.strip()
        if not clean_name or len(clean_name) > 120:
            raise DomainError("Project name must contain 1 to 120 characters")
        if not clean_amp_name or len(clean_amp_name) > 120:
            raise DomainError("Amp/model name must contain 1 to 120 characters")
        if not 8_000 <= int(sample_rate) <= 192_000:
            raise DomainError("Sample rate must be between 8 kHz and 192 kHz")
        get_strategy(acquisition_strategy)
        parsed_controls = validate_controls(controls)
        parsed_candidates = (
            validate_candidate_points(candidate_points, parsed_controls)
            if candidate_points is not None
            else generate_candidate_points(parsed_controls)
        )
        identifier = project_id or str(uuid.uuid4())
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO projects (
                    id, name, amp_name, description, metadata_json, controls_json, candidate_points_json,
                    acquisition_strategy, sample_rate, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    clean_name,
                    clean_amp_name,
                    description.strip(),
                    json.dumps(dict(metadata or {}), separators=(",", ":")),
                    json.dumps([control.as_dict() for control in parsed_controls], separators=(",", ":")),
                    json.dumps(parsed_candidates, separators=(",", ":")),
                    acquisition_strategy,
                    int(sample_rate),
                    timestamp,
                    timestamp,
                ),
            )
        return self.get_project(identifier)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.*,
                       (SELECT COUNT(*) FROM captures c WHERE c.project_id = p.id) AS capture_count,
                       (SELECT COUNT(*) FROM model_artifacts m WHERE m.project_id = p.id) AS model_count,
                       (SELECT COUNT(*) FROM training_jobs t WHERE t.project_id = p.id) AS training_job_count
                FROM projects p
                ORDER BY p.updated_at DESC, p.created_at DESC
                """
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT p.*,
                       (SELECT COUNT(*) FROM captures c WHERE c.project_id = p.id) AS capture_count,
                       (SELECT COUNT(*) FROM model_artifacts m WHERE m.project_id = p.id) AS model_count,
                       (SELECT COUNT(*) FROM training_jobs t WHERE t.project_id = p.id) AS training_job_count
                FROM projects p WHERE p.id = ?
                """,
                (project_id,),
            ).fetchone()
        if row is None:
            raise DomainError(f"Unknown project {project_id!r}")
        return self._project_from_row(row)

    def set_input_signal(self, project_id: str, path: Path, info: Mapping[str, Any]) -> dict[str, Any]:
        self.get_project(project_id)
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE projects SET input_signal_path = ?, input_signal_info_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(path.resolve()), json.dumps(dict(info), separators=(",", ":")), timestamp, project_id),
            )
        return self.get_project(project_id)

    def enqueue_panama_suggestions(
        self,
        project_id: str,
        *,
        round_index: int,
        source_sha256: str,
        control_order: Sequence[str],
        suggestions: Sequence[PanamaSuggestion],
    ) -> list[dict[str, Any]]:
        project = self.get_project(project_id)
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise DomainError("PANAMA round_index must be a non-negative integer")
        if len(source_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_sha256):
            raise DomainError("PANAMA source SHA-256 must be lowercase hexadecimal")
        if not suggestions:
            raise DomainError("PANAMA import contains no suggestions")
        project_control_ids = [control["id"] for control in project["controls"]]
        if set(control_order) != set(project_control_ids) or len(control_order) != len(project_control_ids):
            raise DomainError("PANAMA control order does not match this project")

        batch_id = f"panama-round-{round_index}-{source_sha256}"
        controls = [ControlDefinition.from_mapping(control) for control in project["controls"]]
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_batch = connection.execute(
                """
                SELECT * FROM external_suggestions
                WHERE project_id = ? AND batch_id = ?
                ORDER BY batch_position
                """,
                (project_id, batch_id),
            ).fetchall()
            if existing_batch:
                connection.commit()
                return [self._external_suggestion_from_row(row, controls) for row in existing_batch]

            occupied_rows = connection.execute(
                """
                SELECT settings_json FROM captures
                WHERE project_id = ? AND quality_status != 'failed'
                UNION
                SELECT settings_json FROM recommendations
                WHERE project_id = ? AND status = 'pending'
                UNION
                SELECT settings_json FROM external_suggestions
                WHERE project_id = ? AND status IN ('queued', 'activated')
                """,
                (project_id, project_id, project_id),
            ).fetchall()
            occupied = {row["settings_json"] for row in occupied_rows}
            for position, suggestion in enumerate(suggestions):
                settings_json = settings_key(suggestion.settings)
                if settings_json in occupied:
                    raise DomainError(
                        f"PANAMA suggestion {suggestion.source_index} duplicates a captured, pending, or queued setting"
                    )
                occupied.add(settings_json)
                connection.execute(
                    """
                    INSERT INTO external_suggestions (
                        id, project_id, batch_id, batch_position, source_index, settings_json,
                        strategy, rationale, score, source_sha256, control_order_json, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'panama-ensemble-v1', ?, ?, ?, ?, 'queued', ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        project_id,
                        batch_id,
                        position,
                        suggestion.source_index,
                        settings_json,
                        suggestion.rationale,
                        suggestion.score,
                        source_sha256,
                        json.dumps(list(control_order), separators=(",", ":")),
                        timestamp,
                    ),
                )
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (timestamp, project_id))
            rows = connection.execute(
                """
                SELECT * FROM external_suggestions
                WHERE project_id = ? AND batch_id = ?
                ORDER BY batch_position
                """,
                (project_id, batch_id),
            ).fetchall()
            connection.commit()
        return [self._external_suggestion_from_row(row, controls) for row in rows]

    def list_external_suggestions(self, project_id: str) -> list[dict[str, Any]]:
        project = self.get_project(project_id)
        controls = [ControlDefinition.from_mapping(control) for control in project["controls"]]
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM external_suggestions
                WHERE project_id = ?
                ORDER BY created_at DESC, batch_position
                """,
                (project_id,),
            ).fetchall()
        return [self._external_suggestion_from_row(row, controls) for row in rows]

    def create_recommendation(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        controls = [ControlDefinition.from_mapping(control) for control in project["controls"]]
        strategy = get_strategy(project["acquisition_strategy"])
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM recommendations WHERE project_id = ? AND status = 'pending'",
                (project_id,),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return self._recommendation_from_row(existing, controls)
            external = connection.execute(
                """
                SELECT * FROM external_suggestions
                WHERE project_id = ? AND status = 'queued'
                ORDER BY created_at, batch_id, batch_position
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()
            if external is None:
                captured_rows = connection.execute(
                    "SELECT settings_json FROM captures WHERE project_id = ? AND quality_status != 'failed'",
                    (project_id,),
                ).fetchall()
                pending_rows = connection.execute(
                    "SELECT settings_json FROM recommendations WHERE project_id = ? AND status = 'pending'",
                    (project_id,),
                ).fetchall()
                captured = [json.loads(row["settings_json"]) for row in captured_rows]
                pending = [json.loads(row["settings_json"]) for row in pending_rows]
                suggestion = strategy.suggest(controls, project["candidate_points"], captured, pending)
                settings_json = settings_key(suggestion.settings)
                suggestion_strategy = suggestion.strategy
                rationale = suggestion.rationale
                score = suggestion.score
            else:
                settings_json = external["settings_json"]
                suggestion_strategy = external["strategy"]
                rationale = external["rationale"]
                score = external["score"]
            round_index = connection.execute(
                "SELECT COALESCE(MAX(round_index), -1) + 1 FROM recommendations WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            identifier = str(uuid.uuid4())
            timestamp = utc_now()
            connection.execute(
                """
                INSERT INTO recommendations (
                    id, project_id, round_index, settings_json, strategy, rationale, score, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    identifier,
                    project_id,
                    int(round_index),
                    settings_json,
                    suggestion_strategy,
                    rationale,
                    score,
                    timestamp,
                ),
            )
            if external is not None:
                connection.execute(
                    """
                    UPDATE external_suggestions
                    SET status = 'activated', recommendation_id = ?
                    WHERE id = ?
                    """,
                    (identifier, external["id"]),
                )
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (timestamp, project_id))
            row = connection.execute("SELECT * FROM recommendations WHERE id = ?", (identifier,)).fetchone()
            connection.commit()
        return self._recommendation_from_row(row, controls)

    def dismiss_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
            if row is None:
                raise DomainError(f"Unknown recommendation {recommendation_id!r}")
            if row["status"] != "pending":
                raise DomainError("Only pending recommendations can be dismissed")
            timestamp = utc_now()
            connection.execute(
                "UPDATE recommendations SET status = 'dismissed', resolved_at = ? WHERE id = ?",
                (timestamp, recommendation_id),
            )
            connection.execute(
                "UPDATE external_suggestions SET status = 'dismissed' WHERE recommendation_id = ?",
                (recommendation_id,),
            )
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (timestamp, row["project_id"]))
        return self.get_recommendation(recommendation_id)

    def get_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
        if row is None:
            raise DomainError(f"Unknown recommendation {recommendation_id!r}")
        project = self.get_project(row["project_id"])
        controls = [ControlDefinition.from_mapping(control) for control in project["controls"]]
        return self._recommendation_from_row(row, controls)

    def list_recommendations(self, project_id: str) -> list[dict[str, Any]]:
        project = self.get_project(project_id)
        controls = [ControlDefinition.from_mapping(control) for control in project["controls"]]
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM recommendations WHERE project_id = ? ORDER BY round_index DESC", (project_id,)
            ).fetchall()
        return [self._recommendation_from_row(row, controls) for row in rows]

    def add_capture(
        self,
        *,
        project_id: str,
        recommendation_id: str,
        source: str,
        wav_path: Path,
        wav_info: Mapping[str, Any],
        quality_status: str,
        validation: Mapping[str, Any],
        notes: str = "",
        capture_id: str | None = None,
    ) -> dict[str, Any]:
        if quality_status not in {"pass", "warning", "failed"}:
            raise DomainError("Unsupported capture quality status")
        identifier = capture_id or str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            recommendation = connection.execute(
                "SELECT * FROM recommendations WHERE id = ? AND project_id = ?",
                (recommendation_id, project_id),
            ).fetchone()
            if recommendation is None:
                raise DomainError("Recommendation does not belong to this project")
            if recommendation["status"] != "pending":
                raise DomainError("The recommendation has already been resolved")
            timestamp = utc_now()
            connection.execute(
                """
                INSERT INTO captures (
                    id, project_id, recommendation_id, settings_json, source, wav_path, wav_sha256,
                    sample_rate, channels, sample_width_bits, frames, duration_seconds, peak,
                    quality_status, validation_json, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    project_id,
                    recommendation_id,
                    recommendation["settings_json"],
                    source,
                    str(wav_path.resolve()),
                    wav_info["sha256"],
                    int(wav_info["sample_rate"]),
                    int(wav_info["channels"]),
                    int(wav_info["sample_width_bits"]),
                    int(wav_info["frames"]),
                    float(wav_info["duration_seconds"]),
                    float(wav_info["peak"]),
                    quality_status,
                    json.dumps(dict(validation), separators=(",", ":")),
                    notes.strip(),
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE recommendations SET status = 'captured', resolved_at = ? WHERE id = ?",
                (timestamp, recommendation_id),
            )
            connection.execute(
                "UPDATE external_suggestions SET status = 'captured' WHERE recommendation_id = ?",
                (recommendation_id,),
            )
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (timestamp, project_id))
            connection.commit()
        return self.get_capture(identifier)

    def get_capture(self, capture_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
        if row is None:
            raise DomainError(f"Unknown capture {capture_id!r}")
        return self._capture_from_row(row)

    def list_captures(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM captures WHERE project_id = ? ORDER BY created_at DESC", (project_id,)
            ).fetchall()
        return [self._capture_from_row(row) for row in rows]

    def register_replay_asset(self, project_id: str, settings: Mapping[str, float], wav_path: Path) -> None:
        if not wav_path.is_file():
            raise DomainError(f"Replay WAV does not exist: {wav_path}")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO replay_assets(project_id, settings_key, wav_path) VALUES (?, ?, ?)
                ON CONFLICT(project_id, settings_key) DO UPDATE SET wav_path = excluded.wav_path
                """,
                (project_id, settings_key(settings), str(wav_path.resolve())),
            )

    def get_replay_asset(self, project_id: str, settings: Mapping[str, float]) -> Path:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT wav_path FROM replay_assets WHERE project_id = ? AND settings_key = ?",
                (project_id, settings_key(settings)),
            ).fetchone()
        if row is None:
            raise DomainError("No replay WAV is mapped to the recommended settings")
        path = Path(row["wav_path"])
        if not path.is_file():
            raise DomainError(f"Replay WAV is no longer available: {path}")
        return path

    def add_model_artifact(
        self,
        *,
        project_id: str,
        name: str,
        model_path: Path,
        manifest_path: Path,
        model_sha256: str,
        model_format_version: str,
        architecture: str,
        sample_rate: int,
        validation_status: str,
        validation_message: str,
        provenance: Mapping[str, Any],
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        if validation_status not in {"valid", "invalid", "unverified"}:
            raise DomainError("Unsupported model validation status")
        identifier = artifact_id or str(uuid.uuid4())
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_artifacts (
                    id, project_id, name, model_path, manifest_path, model_sha256,
                    model_format_version, architecture, sample_rate, validation_status,
                    validation_message, provenance_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    project_id,
                    name.strip(),
                    str(model_path.resolve()),
                    str(manifest_path.resolve()),
                    model_sha256,
                    model_format_version,
                    architecture,
                    int(sample_rate),
                    validation_status,
                    validation_message,
                    json.dumps(dict(provenance), separators=(",", ":")),
                    timestamp,
                ),
            )
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (timestamp, project_id))
        return self.get_model_artifact(identifier)

    def get_model_artifact(self, artifact_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM model_artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise DomainError(f"Unknown model artifact {artifact_id!r}")
        return self._model_from_row(row)

    def list_model_artifacts(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM model_artifacts WHERE project_id = ? ORDER BY created_at DESC", (project_id,)
            ).fetchall()
        return [self._model_from_row(row) for row in rows]

    def add_training_job(
        self,
        *,
        project_id: str,
        backend: str,
        job_dir: Path,
        data_config_path: Path,
        model_config_path: Path,
        learning_config_path: Path,
        output_dir: Path,
        log_path: Path,
        control_order: Sequence[str],
        options: Mapping[str, Any],
        source_evidence: Mapping[str, Any],
        job_id: str | None = None,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        identifier = job_id or str(uuid.uuid4())
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO training_jobs (
                    id, project_id, status, backend, job_dir, data_config_path, model_config_path,
                    learning_config_path, output_dir, log_path, control_order_json, options_json,
                    source_evidence_json, created_at
                ) VALUES (?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    project_id,
                    backend,
                    str(job_dir.resolve()),
                    str(data_config_path.resolve()),
                    str(model_config_path.resolve()),
                    str(learning_config_path.resolve()),
                    str(output_dir.resolve()),
                    str(log_path.resolve()),
                    json.dumps(list(control_order), separators=(",", ":")),
                    json.dumps(dict(options), separators=(",", ":")),
                    json.dumps(dict(source_evidence), separators=(",", ":")),
                    timestamp,
                ),
            )
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (timestamp, project_id))
        return self.get_training_job(identifier)

    def get_training_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise DomainError(f"Unknown training job {job_id!r}")
        return self._training_job_from_row(row)

    def list_training_jobs(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM training_jobs WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [self._training_job_from_row(row) for row in rows]

    def update_training_job(
        self,
        job_id: str,
        *,
        status: str,
        result: Mapping[str, Any] | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"ready", "queued", "running", "completed", "failed"}:
            raise DomainError("Unsupported training job status")
        current = self.get_training_job(job_id)
        allowed = {
            "ready": {"queued", "failed"},
            "queued": {"running", "failed"},
            "running": {"completed", "failed"},
            "completed": set(),
            "failed": set(),
        }
        if status != current["status"] and status not in allowed[current["status"]]:
            raise DomainError(f"Training job cannot move from {current['status']} to {status}")
        timestamp = utc_now()
        started_at = timestamp if status == "running" and not current["started_at"] else current["started_at"]
        completed_at = timestamp if status in {"completed", "failed"} else current["completed_at"]
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE training_jobs
                SET status = ?, result_json = ?, error_message = ?, started_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    json.dumps(dict(result), separators=(",", ":")) if result is not None else None,
                    error_message,
                    started_at,
                    completed_at,
                    job_id,
                ),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (timestamp, current["project_id"]),
            )
        return self.get_training_job(job_id)

    def fail_incomplete_training_jobs(self, message: str) -> int:
        timestamp = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE training_jobs
                SET status = 'failed', error_message = ?, completed_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (message, timestamp),
            )
        return int(cursor.rowcount)

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        result["controls"] = json.loads(result.pop("controls_json"))
        result["candidate_points"] = json.loads(result.pop("candidate_points_json"))
        info = result.pop("input_signal_info_json")
        result["input_signal_info"] = json.loads(info) if info else None
        result["capture_count"] = int(result.get("capture_count", 0))
        result["model_count"] = int(result.get("model_count", 0))
        result["training_job_count"] = int(result.get("training_job_count", 0))
        return result

    @staticmethod
    def _recommendation_from_row(
        row: sqlite3.Row, controls: Sequence[ControlDefinition]
    ) -> dict[str, Any]:
        result = dict(row)
        settings = json.loads(result.pop("settings_json"))
        result["settings"] = settings
        result["display_settings"] = settings_display(settings, controls)
        return result

    @staticmethod
    def _capture_from_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["settings"] = json.loads(result.pop("settings_json"))
        result["validation"] = json.loads(result.pop("validation_json"))
        return result

    @staticmethod
    def _external_suggestion_from_row(
        row: sqlite3.Row, controls: Sequence[ControlDefinition]
    ) -> dict[str, Any]:
        result = dict(row)
        settings = json.loads(result.pop("settings_json"))
        result["settings"] = settings
        result["display_settings"] = settings_display(settings, controls)
        result["control_order"] = json.loads(result.pop("control_order_json"))
        return result

    @staticmethod
    def _model_from_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["provenance"] = json.loads(result.pop("provenance_json"))
        return result

    @staticmethod
    def _training_job_from_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["control_order"] = json.loads(result.pop("control_order_json"))
        result["options"] = json.loads(result.pop("options_json"))
        result["source_evidence"] = json.loads(result.pop("source_evidence_json"))
        raw_result = result.pop("result_json")
        result["result"] = json.loads(raw_result) if raw_result else None
        return result
