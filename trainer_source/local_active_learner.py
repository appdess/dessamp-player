"""Single-accelerator PANAMA ensemble learner for DessMetal capture projects.

The upstream PANAMA learner trains ensemble members in parallel on CUDA and
optimizes a continuous global-condition vector. A standalone capture appliance
usually has one Apple or embedded accelerator and physical knob increments.
This module keeps PANAMA's ensemble-disagreement acquisition objective while
training members sequentially and scoring a finite, explicitly supplied knob
grid. Its ``new_points.json`` output is accepted by the DessMetal Trainer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class CandidateScore:
    source_index: int
    g_vector: tuple[float, ...]
    disagreement_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.source_index,
            "g_vector": list(self.g_vector),
            "disagreement_score": self.disagreement_score,
        }


def train_ensemble(
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    learning_config: dict[str, Any],
    output_dir: Path,
    backend: str,
    ensemble_size: int,
    epochs: int,
    batch_size: int,
    ny: int,
    seed: int = 1729,
) -> list[Path]:
    """Train independent PANAMA ensemble members sequentially on MPS or CPU."""

    if backend not in {"mps", "cpu"}:
        raise ValueError("backend must be 'mps' or 'cpu'")
    if ensemble_size < 2 or ensemble_size > 8:
        raise ValueError("ensemble_size must be between 2 and 8")
    if epochs < 1 or batch_size < 1 or ny < 128:
        raise ValueError("epochs, batch_size, and ny must be positive")

    import pytorch_lightning as pl
    import torch
    from nam.train import full

    if backend == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is unavailable")

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints: list[Path] = []
    for member_index in range(ensemble_size):
        member_seed = seed + member_index
        pl.seed_everything(member_seed, workers=True, verbose=False)
        member_dir = output_dir / f"ensemble-member-{member_index:02d}"
        member_dir.mkdir(parents=False, exist_ok=False)

        member_data = copy.deepcopy(data_config)
        member_model = copy.deepcopy(model_config)
        member_model["checkpoint_path"] = None
        member_learning = copy.deepcopy(learning_config)
        member_learning.setdefault("train_dataloader", {})
        member_learning.setdefault("val_dataloader", {})
        member_learning.setdefault("trainer", {})
        member_learning["train_dataloader"].update(
            {
                "batch_size": int(batch_size),
                "shuffle": True,
                "drop_last": True,
                "num_workers": 0,
                "pin_memory": False,
            }
        )
        member_learning["val_dataloader"].update(
            {"batch_size": int(batch_size), "num_workers": 0, "pin_memory": False}
        )
        member_learning["trainer"].update(
            {
                "accelerator": backend,
                "devices": 1,
                "max_epochs": int(epochs),
                "check_val_every_n_epoch": 1,
                "enable_progress_bar": False,
                "logger": False,
            }
        )
        member_data.setdefault("train", {})["ny"] = int(ny)
        member_data.setdefault("validation", {})["ny"] = int(ny)

        checkpoint = full.main(
            member_data,
            member_model,
            member_learning,
            member_dir,
            no_show=True,
            make_plots=False,
        )
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise RuntimeError(f"PANAMA ensemble member {member_index} produced no checkpoint")
        checkpoints.append(checkpoint_path)
    return checkpoints


def score_candidate_grid(
    *,
    checkpoint_paths: Sequence[Path],
    model_config: dict[str, Any],
    input_path: Path,
    candidate_vectors: Sequence[Sequence[float]],
    captured_vectors: Sequence[Sequence[float]],
    backend: str,
    chunk_size: int = 8192,
    batch_size: int = 16,
    max_chunks: int = 32,
) -> list[CandidateScore]:
    """Rank uncaptured knob vectors by PANAMA ensemble output variance."""

    if backend not in {"mps", "cpu"}:
        raise ValueError("backend must be 'mps' or 'cpu'")
    if len(checkpoint_paths) < 2:
        raise ValueError("At least two checkpoints are required for ensemble disagreement")
    candidates = _validated_vectors(candidate_vectors, "candidate_vectors")
    captured = _validated_vectors(captured_vectors, "captured_vectors", allow_empty=True)
    dimensions = len(candidates[0])
    if captured and any(len(vector) != dimensions for vector in captured):
        raise ValueError("Captured and candidate vectors must have the same dimension")
    captured_keys = {_vector_key(vector) for vector in captured}
    available = [(index, vector) for index, vector in enumerate(candidates) if _vector_key(vector) not in captured_keys]
    if not available:
        raise ValueError("No uncaptured candidate vectors remain")

    import torch
    from nam.data import wav_to_tensor
    from nam.train.lightning_module import LightningModule

    if backend == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is unavailable")
    device = torch.device(backend)
    models = []
    for checkpoint_path in checkpoint_paths:
        model = LightningModule.load_from_safe_checkpoint(
            checkpoint_path,
            **LightningModule.parse_config(model_config),
        )
        model.eval()
        model.requires_grad_(False)
        model.to(device)
        models.append(model)

    condition_size = int(models[0].net.global_condition_size)
    if dimensions != condition_size:
        raise ValueError(
            f"Candidate vectors have {dimensions} values but the PANAMA model expects {condition_size}"
        )
    if any(int(model.net.global_condition_size) != condition_size for model in models):
        raise ValueError("PANAMA ensemble members disagree about global condition size")
    receptive_field = int(models[0].net.receptive_field)
    if chunk_size <= receptive_field:
        raise ValueError(
            f"score chunk_size must exceed the WaveNet receptive field ({receptive_field})"
        )
    if batch_size < 1 or max_chunks < 1:
        raise ValueError("batch_size and max_chunks must be positive")

    x = wav_to_tensor(str(input_path.resolve()))
    step = chunk_size - receptive_field + 1
    if x.numel() < chunk_size:
        raise ValueError(f"Scoring input has {x.numel()} frames but chunk_size is {chunk_size}")
    chunks = x.unfold(0, chunk_size, step).contiguous()
    if chunks.shape[0] > max_chunks:
        indices = torch.linspace(0, chunks.shape[0] - 1, steps=max_chunks).round().long()
        chunks = chunks.index_select(0, indices)

    ranked: list[CandidateScore] = []
    with torch.no_grad():
        for source_index, vector in available:
            variance_sum = 0.0
            variance_count = 0
            for start in range(0, chunks.shape[0], batch_size):
                batch = chunks[start : start + batch_size].to(device)
                g = torch.tensor(vector, dtype=batch.dtype, device=device).unsqueeze(0).expand(batch.shape[0], -1)
                outputs = torch.stack([model(batch, g) for model in models], dim=0)
                variance = torch.var(outputs, dim=0, unbiased=True)
                variance_sum += float(variance.sum().detach().cpu())
                variance_count += variance.numel()
            ranked.append(
                CandidateScore(
                    source_index=source_index,
                    g_vector=vector,
                    disagreement_score=variance_sum / variance_count,
                )
            )
    ranked.sort(key=lambda item: (-item.disagreement_score, item.source_index))
    return ranked


def evaluate_ensemble_esr(
    *,
    checkpoint_paths: Sequence[Path],
    model_config: dict[str, Any],
    input_path: Path,
    references: Sequence[tuple[Sequence[float], Path]],
    backend: str,
    chunk_size: int = 8192,
    delay_samples: int = 0,
) -> dict[str, Any]:
    """Measure ensemble-mean ESR against hidden, never-trained captures.

    ``delay_samples`` uses the same convention as ``nam.data.Dataset``: a
    positive delay means the recorded target trails the dry input, so the
    input tail and target head are removed before comparison. Keeping this
    alignment explicit prevents a capture-path latency from being counted as
    model error.
    """

    if not references:
        raise ValueError("At least one hidden reference is required")
    if isinstance(delay_samples, bool) or not isinstance(delay_samples, int):
        raise ValueError("delay_samples must be an integer")
    import torch
    from nam.data import wav_to_tensor
    from nam.train.lightning_module import LightningModule

    device = torch.device(backend)
    models = []
    for checkpoint_path in checkpoint_paths:
        model = LightningModule.load_from_safe_checkpoint(
            checkpoint_path,
            **LightningModule.parse_config(model_config),
        )
        model.eval()
        model.requires_grad_(False)
        model.to(device)
        models.append(model)
    raw_x = wav_to_tensor(str(input_path.resolve()))
    input_slice, target_start = evaluation_alignment(raw_x.numel(), delay_samples)
    x = raw_x[input_slice]
    receptive_field = int(models[0].net.receptive_field)
    if chunk_size <= receptive_field:
        raise ValueError("Evaluation chunk size must exceed the receptive field")
    step = chunk_size - receptive_field + 1
    chunks = x.unfold(0, chunk_size, step).contiguous()
    covered = chunks.shape[0] * step

    per_setting: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_vector, reference_path in references:
            vector = tuple(float(value) for value in raw_vector)
            raw_target = wav_to_tensor(str(reference_path.resolve()))
            target_offset = target_start + receptive_field - 1
            target = raw_target[target_offset : target_offset + covered]
            if target.numel() != covered:
                raise ValueError(
                    f"Aligned target {reference_path} has {target.numel()} frames; expected {covered}"
                )
            prediction_parts = []
            for chunk in chunks:
                batch = chunk.unsqueeze(0).to(device)
                g = torch.tensor(vector, dtype=batch.dtype, device=device).unsqueeze(0)
                member_outputs = [model(batch, g, pad_start=False).squeeze(0) for model in models]
                prediction_parts.append(torch.stack(member_outputs, dim=0).mean(dim=0).cpu())
            prediction = torch.cat(prediction_parts)[: target.numel()]
            error_energy = float(torch.sum((prediction - target) ** 2))
            target_energy = float(torch.sum(target**2))
            esr = error_energy / max(target_energy, sys.float_info.min)
            per_setting.append({"g_vector": list(vector), "esr": esr})
    return {
        "mean_esr": sum(item["esr"] for item in per_setting) / len(per_setting),
        "max_esr": max(item["esr"] for item in per_setting),
        "settings": per_setting,
        "evaluated_frames_per_setting": covered,
        "delay_samples": delay_samples,
    }


def evaluation_alignment(input_frames: int, delay_samples: int) -> tuple[slice, int]:
    """Return the NAM Dataset-compatible input slice and target start offset."""
    if isinstance(delay_samples, bool) or not isinstance(delay_samples, int):
        raise ValueError("delay_samples must be an integer")
    if input_frames <= 0 or abs(delay_samples) >= input_frames:
        raise ValueError("delay_samples removes the complete evaluation input")
    input_start = max(-delay_samples, 0)
    input_stop = input_frames - max(delay_samples, 0)
    return slice(input_start, input_stop), max(delay_samples, 0)


def write_new_points(path: Path, ranked: Sequence[CandidateScore], *, maximum: int) -> list[dict[str, Any]]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    points = [item.as_dict() for item in ranked[:maximum]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(points, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return points


def select_diverse_batch(
    ranked: Sequence[CandidateScore],
    *,
    maximum: int,
    minimum_distance: float,
) -> list[CandidateScore]:
    """Greedily retain disagreement rank while separating knob vectors.

    Upstream PANAMA clusters its continuous optimization restarts before asking
    for multiple captures. DessMetal uses a finite physical knob grid, so a
    deterministic Euclidean separation provides the equivalent batch guard.
    The function intentionally returns fewer than ``maximum`` points when the
    requested separation cannot be met; callers must not silently dilute the
    distance contract just to fill a batch.
    """

    if maximum < 1:
        raise ValueError("maximum must be positive")
    if not math.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError("minimum_distance must be finite and non-negative")
    selected: list[CandidateScore] = []
    dimensions: int | None = None
    seen: set[tuple[float, ...]] = set()
    for candidate in ranked:
        vector = tuple(float(value) for value in candidate.g_vector)
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError("ranked candidates must contain finite non-empty vectors")
        if dimensions is None:
            dimensions = len(vector)
        elif len(vector) != dimensions:
            raise ValueError("ranked candidate vectors must have a consistent dimension")
        key = _vector_key(vector)
        if key in seen:
            raise ValueError("ranked candidates contain a duplicate vector")
        seen.add(key)
        if all(math.dist(vector, existing.g_vector) >= minimum_distance for existing in selected):
            selected.append(candidate)
            if len(selected) == maximum:
                break
    return selected


def _validated_vectors(
    raw_vectors: Sequence[Sequence[float]], label: str, *, allow_empty: bool = False
) -> list[tuple[float, ...]]:
    if not raw_vectors:
        if allow_empty:
            return []
        raise ValueError(f"{label} must not be empty")
    vectors: list[tuple[float, ...]] = []
    dimensions: int | None = None
    for raw in raw_vectors:
        vector = tuple(float(value) for value in raw)
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError(f"{label} contains an empty or non-finite vector")
        if dimensions is None:
            dimensions = len(vector)
        elif len(vector) != dimensions:
            raise ValueError(f"{label} vectors must have a consistent dimension")
        vectors.append(vector)
    if len({_vector_key(vector) for vector in vectors}) != len(vectors):
        raise ValueError(f"{label} contains duplicate vectors")
    return vectors


def _vector_key(vector: Iterable[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 10) for value in vector)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one local PANAMA ensemble-disagreement round")
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--learning-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ny", type=int, default=2048)
    parser.add_argument("--score-chunk-size", type=int, default=8192)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--max-score-chunks", type=int, default=32)
    parser.add_argument("--max-points", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    candidates = _load_json(args.candidate_config, "candidate config")
    candidate_vectors = candidates.get("candidate_vectors")
    captured_vectors = candidates.get("captured_vectors")
    if not isinstance(candidate_vectors, list) or not isinstance(captured_vectors, list):
        raise ValueError("candidate config requires candidate_vectors and captured_vectors arrays")
    data_config = _load_json(args.data_config, "data config")
    model_config = _load_json(args.model_config, "model config")
    learning_config = _load_json(args.learning_config, "learning config")
    checkpoints = train_ensemble(
        data_config=data_config,
        model_config=model_config,
        learning_config=learning_config,
        output_dir=args.output_dir / "ensemble",
        backend=args.backend,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        seed=args.seed,
    )
    ranked = score_candidate_grid(
        checkpoint_paths=checkpoints,
        model_config=model_config,
        input_path=args.input,
        candidate_vectors=candidate_vectors,
        captured_vectors=captured_vectors,
        backend=args.backend,
        chunk_size=args.score_chunk_size,
        batch_size=args.score_batch_size,
        max_chunks=args.max_score_chunks,
    )
    points = write_new_points(args.output_dir / "new_points.json", ranked, maximum=args.max_points)
    evidence = {
        "backend": args.backend,
        "ensemble_size": args.ensemble_size,
        "epochs": args.epochs,
        "selection_method": "panama-ensemble-disagreement-finite-grid",
        "checkpoints": [str(path) for path in checkpoints],
        "ranking": [item.as_dict() for item in ranked],
        "selected": points,
    }
    (args.output_dir / "active-learning-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
