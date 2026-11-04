#!/usr/bin/env python3
"""Select a deployable member before opening frozen external evaluation."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate_conditioned_exports import sha256_file


def parse_member(raw: str) -> tuple[str, Path]:
    label, separator, path_raw = raw.partition("=")
    if not separator or not label or not path_raw:
        raise ValueError("each --member must use LABEL=/absolute/member-directory")
    directory = Path(path_raw).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"member directory is unavailable: {directory}")
    return label, directory


def load_member(label: str, directory: Path) -> dict[str, Any]:
    manifest_path = directory / "training-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"member {label} has no training-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"member {label} manifest must be an object")
    model_record = manifest.get("artifacts", {}).get("model")
    if not isinstance(model_record, Mapping):
        raise ValueError(f"member {label} manifest has no model artifact")
    relative_model = model_record.get("path")
    if not isinstance(relative_model, str) or Path(relative_model).is_absolute():
        raise ValueError(f"member {label} model path must be relative")
    model_path = (directory / relative_model).resolve()
    if not model_path.is_relative_to(directory) or not model_path.is_file():
        raise ValueError(f"member {label} model escapes or is missing")
    model_sha256 = sha256_file(model_path)
    if model_record.get("sha256") != model_sha256:
        raise ValueError(f"member {label} model hash does not match its manifest")
    esr = float(manifest.get("best_internal_validation_esr"))
    if not math.isfinite(esr) or esr < 0.0:
        raise ValueError(f"member {label} internal validation ESR is invalid")
    receptive_field = int(manifest.get("receptive_field", 0))
    if receptive_field != 6347:
        raise ValueError(f"member {label} does not declare A2 RF 6347")
    if manifest.get("child") != "full":
        raise ValueError(f"member {label} is not A2 Full")
    data_record = manifest.get("artifacts", {}).get("data_config")
    if not isinstance(data_record, Mapping) or not isinstance(data_record.get("sha256"), str):
        raise ValueError(f"member {label} has no bound data-config hash")
    return {
        "label": label,
        "directory": str(directory),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "best_internal_validation_esr": esr,
        "best_epoch": int(manifest.get("best_epoch", 0)),
        "child": "full",
        "receptive_field": receptive_field,
        "control_order": list(manifest.get("control_order", [])),
        "data_config_sha256": data_record["sha256"],
    }


def select(raw_members: Sequence[str]) -> dict[str, Any]:
    if len(raw_members) < 2:
        raise ValueError("at least two independent members are required for pre-audit selection")
    seen: set[str] = set()
    members: list[dict[str, Any]] = []
    for raw in raw_members:
        label, directory = parse_member(raw)
        if label in seen:
            raise ValueError(f"duplicate member label: {label}")
        seen.add(label)
        members.append(load_member(label, directory))
    contracts = {
        (
            tuple(item["control_order"]),
            item["data_config_sha256"],
            item["child"],
            item["receptive_field"],
        )
        for item in members
    }
    if len(contracts) != 1:
        raise ValueError("members do not share one data/control/architecture contract")
    # Python's stable min preserves CLI order as the exact-tie breaker.
    selected = min(members, key=lambda item: item["best_internal_validation_esr"])
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_opened_hidden_evidence": False,
        "criterion": "minimum best_internal_validation_esr",
        "exact_tie_breaker": "first --member argument",
        "members": members,
        "selected_member": selected["label"],
        "selected_model_path": selected["model_path"],
        "selected_model_sha256": selected["model_sha256"],
        "selected_internal_validation_esr": selected["best_internal_validation_esr"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace existing selection report: {output}")
    report = select(args.member)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_member": report["selected_member"],
                "selected_model_sha256": report["selected_model_sha256"],
                "selection_opened_hidden_evidence": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
