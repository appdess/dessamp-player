"""End-to-end synthetic validation for DessMetal's PANAMA workflow.

This creates a deterministic hidden "amp", exposes only Gain 2 and Gain 7 to
the first ensemble, asks the local PANAMA adapter for the most-disagreed knob
point, synthesizes that capture, retrains from scratch, measures both ensembles
against never-trained settings, exports a NAM model, packages a Player
manifest, and renders several knob positions through native NAM Core.

The generated audio is test data, not an audible-quality proxy for a real amp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from local_active_learner import evaluate_ensemble_esr, score_candidate_grid, train_ensemble, write_new_points


SAMPLE_RATE = 48_000
STARTER_GAINS = (0.2, 0.7)
CANDIDATE_GAINS = tuple(round(0.2 + 0.05 * index, 2) for index in range(13))


def make_input(duration_seconds: float, seed: int) -> np.ndarray:
    frames = int(round(duration_seconds * SAMPLE_RATE))
    t = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    chirp_phase = 2.0 * np.pi * (65.0 * t + 0.5 * 1_350.0 * t * t / duration_seconds)
    carrier = (
        0.10 * np.sin(2.0 * np.pi * 83.0 * t)
        + 0.065 * np.sin(2.0 * np.pi * 173.0 * t + 0.4)
        + 0.04 * np.sin(chirp_phase)
    )
    noise = rng.normal(0.0, 1.0, frames + 16)
    noise = np.convolve(noise, np.ones(17, dtype=np.float64) / 17.0, mode="valid")
    gate = 0.28 + 0.72 * (np.sin(2.0 * np.pi * 1.7 * t) > -0.2)
    x = gate * (carrier + 0.018 * noise[:frames])
    fade = min(frames // 10, 1_024)
    x[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float64)
    x[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float64)
    return x.astype(np.float32)


def synthetic_amp(x: np.ndarray, gain: float) -> np.ndarray:
    """A deterministic nonlinear, dynamic target whose source is never trained."""

    g = float(gain)
    drive = 1.5 + 10.0 * g + 4.0 * math.sin(math.pi * g) ** 2
    bias = 0.12 * (g - 0.45)
    driven = np.tanh(drive * x.astype(np.float64) + bias) - math.tanh(bias)
    # A short gain-dependent FIR gives the WaveNet a small dynamic component.
    kernel = np.array(
        [0.68 - 0.20 * g, 0.20 + 0.08 * g, 0.08 + 0.07 * g, 0.04 + 0.05 * g],
        dtype=np.float64,
    )
    kernel /= np.sum(kernel)
    shaped = np.convolve(driven, kernel, mode="full")[: x.size]
    # Add an asymmetric second-order term so interpolation is not merely output gain.
    output = 0.52 * shaped + (0.07 + 0.12 * g) * shaped * np.abs(shaped)
    return np.clip(output, -0.92, 0.92).astype(np.float32)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_data_config(input_path: Path, captures: Sequence[tuple[float, Path]]) -> dict[str, Any]:
    outputs = [{"y_path": str(path.resolve()), "g_vector": [gain]} for gain, path in captures]
    return {
        "type": "shared_in_multi_out",
        "common": {
            "x_path": str(input_path.resolve()),
            "input_gain": 0.0,
            "delay": 0,
            "allow_unequal_lengths": False,
        },
        "train": {"outputs": outputs},
        # A duplicated starter is used only for checkpoint selection. Dense hidden
        # references below are never present in either training or validation.
        "validation": {"outputs": [dict(outputs[0])]},
    }


def export_member(trainer_root: Path, member_directory: Path, export_directory: Path) -> Path:
    result = subprocess.run(
        [sys.executable, str(trainer_root / "export_model.py"), str(member_directory), str(export_directory)],
        cwd=trainer_root.parent,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "PANAMA export failed")[-4_000:])
    model_path = export_directory / "model.nam"
    if not model_path.is_file():
        raise RuntimeError("PANAMA export produced no model.nam")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate sparse PANAMA acquisition with a synthetic amp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trainer-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--backend", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ny", type=int, default=2048)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=260813)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing validation directory: {output_dir}")
    output_dir.mkdir(parents=True)
    trainer_root = args.trainer_root.expanduser().resolve()
    model_config = json.loads((trainer_root / "parametric_wavenet.json").read_text(encoding="utf-8"))
    learning_config = json.loads(
        (trainer_root / "default_config_files" / "learning" / "default.json").read_text(encoding="utf-8")
    )

    # Import capture/player modules only after argument validation so this script
    # remains usable with an explicit PYTHONPATH in PANAMA development setups.
    from dessmetal_capture.domain import ControlDefinition
    from dessmetal_capture.manifests import import_model_artifact
    from dessmetal_capture.panama import parse_new_points, source_digest
    from dessmetal_capture.store import CaptureStore
    from dessmetal_capture.wav_io import inspect_wav, write_pcm24_mono
    from dessmetal_player.artifact import PlayerArtifact

    audio_dir = output_dir / "synthetic-audio"
    input_path = audio_dir / "input.wav"
    x = make_input(args.duration_seconds, args.seed)
    write_pcm24_mono(input_path, x, SAMPLE_RATE)
    references: dict[float, Path] = {}
    for gain in CANDIDATE_GAINS:
        path = audio_dir / f"gain-{gain:.2f}.wav"
        write_pcm24_mono(path, synthetic_amp(x, gain), SAMPLE_RATE)
        references[gain] = path

    starter_captures = [(gain, references[gain]) for gain in STARTER_GAINS]
    starter_data = make_data_config(input_path, starter_captures)
    write_json(output_dir / "starter-data-config.json", starter_data)
    candidate_config = {
        "control_order": ["gain"],
        "candidate_vectors": [[gain] for gain in CANDIDATE_GAINS],
        "captured_vectors": [[gain] for gain in STARTER_GAINS],
    }
    write_json(output_dir / "candidate-config.json", candidate_config)

    initial_checkpoints = train_ensemble(
        data_config=starter_data,
        model_config=model_config,
        learning_config=learning_config,
        output_dir=output_dir / "initial-ensemble",
        backend=args.backend,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        seed=args.seed,
    )
    ranking = score_candidate_grid(
        checkpoint_paths=initial_checkpoints,
        model_config=model_config,
        input_path=input_path,
        candidate_vectors=candidate_config["candidate_vectors"],
        captured_vectors=candidate_config["captured_vectors"],
        backend=args.backend,
        chunk_size=8192,
        batch_size=8,
        max_chunks=24,
    )
    selected_points = write_new_points(output_dir / "new_points.json", ranking, maximum=1)
    selected_gain = float(selected_points[0]["g_vector"][0])

    enriched_captures = [*starter_captures, (selected_gain, references[selected_gain])]
    enriched_data = make_data_config(input_path, enriched_captures)
    write_json(output_dir / "enriched-data-config.json", enriched_data)
    enriched_checkpoints = train_ensemble(
        data_config=enriched_data,
        model_config=model_config,
        learning_config=learning_config,
        output_dir=output_dir / "enriched-ensemble",
        backend=args.backend,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        seed=args.seed,
    )

    hidden_gains = [
        gain for gain in CANDIDATE_GAINS if gain not in {*STARTER_GAINS, selected_gain}
    ]
    hidden_references = [([gain], references[gain]) for gain in hidden_gains]
    initial_metrics = evaluate_ensemble_esr(
        checkpoint_paths=initial_checkpoints,
        model_config=model_config,
        input_path=input_path,
        references=hidden_references,
        backend=args.backend,
        chunk_size=8192,
    )
    enriched_metrics = evaluate_ensemble_esr(
        checkpoint_paths=enriched_checkpoints,
        model_config=model_config,
        input_path=input_path,
        references=hidden_references,
        backend=args.backend,
        chunk_size=8192,
    )

    exported_model = export_member(
        trainer_root,
        output_dir / "enriched-ensemble" / "ensemble-member-00",
        output_dir / "exported-model",
    )
    loader_result = subprocess.run(
        [str(args.loader.resolve()), str(exported_model.resolve())],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if loader_result.returncode != 0:
        raise RuntimeError((loader_result.stderr or loader_result.stdout or "NAM Core rejected export")[-4_000:])

    store = CaptureStore(output_dir / "player-data")
    project = store.create_project(
        name="Synthetic PANAMA active-learning validation",
        amp_name="Deterministic hidden amp",
        description="Generated evidence only; not a real amplifier capture.",
        controls=[
            {
                "id": "gain",
                "label": "Gain",
                "minimum": 0.2,
                "maximum": 0.8,
                "step": 0.05,
                "display_minimum": 2.0,
                "display_maximum": 8.0,
            }
        ],
    )
    store.set_input_signal(project["id"], input_path, inspect_wav(input_path))
    controls = [ControlDefinition.from_mapping(control) for control in project["controls"]]
    parsed_points = parse_new_points(selected_points, controls=controls, control_order=["gain"])
    queued_points = store.enqueue_panama_suggestions(
        project["id"],
        round_index=0,
        source_sha256=source_digest(selected_points),
        control_order=["gain"],
        suggestions=parsed_points,
    )
    recommendation = store.create_recommendation(project["id"])
    if recommendation["strategy"] != "panama-ensemble-v1" or not math.isclose(
        float(recommendation["settings"]["gain"]), selected_gain, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise RuntimeError("Trainer did not activate the selected PANAMA point")
    selected_capture_info = inspect_wav(references[selected_gain])
    selected_capture = store.add_capture(
        project_id=project["id"],
        recommendation_id=recommendation["id"],
        source="synthetic-hidden-amp",
        wav_path=references[selected_gain],
        wav_info=selected_capture_info,
        quality_status="pass",
        validation={"status": "pass", "failures": [], "warnings": [], "synthetic": True},
        notes="Generated only after the PANAMA ensemble selected this setting.",
    )
    artifact = import_model_artifact(
        store,
        project_id=project["id"],
        source_model_path=exported_model,
        name="Synthetic conditioned WaveNet",
        provenance={
            "purpose": "synthetic-active-learning-validation",
            "workflow_validation_only": True,
            "starter_g_vectors": [[gain] for gain in STARTER_GAINS],
            "selected_point": selected_points[0],
            "selection_method": "panama-ensemble-disagreement-finite-grid",
            "backend": args.backend,
            "ensemble_size": args.ensemble_size,
            "epochs": args.epochs,
        },
        loader_path=args.loader.resolve(),
    )
    player = PlayerArtifact(Path(artifact["manifest_path"]), args.renderer.resolve())
    render_evidence = []
    for gain in sorted({STARTER_GAINS[0], selected_gain, STARTER_GAINS[1]}):
        output_path = output_dir / "player-renders" / f"gain-{gain:.2f}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evidence = player.render(input_path=input_path, output_path=output_path, settings={"gain": gain})
        render_evidence.append(
            {
                "gain": gain,
                "wav_sha256": sha256_file(output_path),
                "peak_dbfs": evidence["output"]["peak_dbfs"],
                "frames": evidence["output"]["frames"],
            }
        )

    initial_mean = float(initial_metrics["mean_esr"])
    enriched_mean = float(enriched_metrics["mean_esr"])
    report = {
        "status": "passed" if enriched_mean < initial_mean else "completed-without-measured-improvement",
        "claim": "synthetic functional evidence only",
        "backend": args.backend,
        "ensemble_size": args.ensemble_size,
        "epochs": args.epochs,
        "starter_g_vectors": [[gain] for gain in STARTER_GAINS],
        "selected_point": selected_points[0],
        "trainer_queue": {
            "queued_strategy": queued_points[0]["strategy"],
            "recommendation_id": recommendation["id"],
            "recommendation_strategy": recommendation["strategy"],
            "capture_id": selected_capture["id"],
            "capture_settings": selected_capture["settings"],
        },
        "candidate_ranking": [item.as_dict() for item in ranking],
        "hidden_reference_gains": hidden_gains,
        "initial_hidden_metrics": initial_metrics,
        "enriched_hidden_metrics": enriched_metrics,
        "mean_esr_ratio_enriched_over_initial": enriched_mean / initial_mean,
        "model_path": str(Path(artifact["model_path"])),
        "model_sha256": artifact["model_sha256"],
        "manifest_path": artifact["manifest_path"],
        "loader_output": loader_result.stdout.strip(),
        "player_renders": render_evidence,
        "source_hashes": {
            "local_active_learner.py": sha256_file(trainer_root / "local_active_learner.py"),
            "parametric_wavenet.json": sha256_file(trainer_root / "parametric_wavenet.json"),
            "input.wav": sha256_file(input_path),
        },
    }
    write_json(output_dir / "validation-report.json", report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
