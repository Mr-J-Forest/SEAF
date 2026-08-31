#!/usr/bin/env python3
"""Combine coarse and interior LR calibration results before freezing rates."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


NAME_PATTERN = re.compile(r"^lr_(?P<family>.+)_(?P<label>[^_]+)$")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_refinement_rows(results_root: Path, stage: str) -> tuple[dict[str, list[dict]], set[str], list[str]]:
    stage_root = results_root.resolve() / stage
    families: dict[str, list[dict]] = {}
    source_hashes: set[str] = set()
    errors: list[str] = []
    if not stage_root.is_dir():
        return {}, set(), [f"missing refinement stage: {stage_root}"]
    for run_dir in sorted(stage_root.glob("*/seed_*")):
        match = NAME_PATTERN.fullmatch(run_dir.parent.name)
        if not match:
            continue
        required = [run_dir / "_SUCCESS", run_dir / "config.json", run_dir / "run_summary.json"]
        if not all(path.is_file() for path in required):
            errors.append(f"incomplete refinement run: {run_dir}")
            continue
        config = load_json(run_dir / "config.json")
        summary = load_json(run_dir / "run_summary.json")
        if summary.get("evaluation_scope") != "none":
            errors.append(f"refinement accessed an evaluation report: {run_dir}")
            continue
        losses = [float(value) for value in summary.get("validation_selection_losses", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            errors.append(f"missing/nonfinite validation loss history: {run_dir}")
            continue
        tail = losses[-5:]
        row = {
            "experiment": run_dir.parent.name,
            "learning_rate": float(config["learning_rate"]),
            "best_validation_loss": float(min(losses)),
            "tail_validation_loss_median": float(median(tail)),
            "best_epoch": summary.get("best_epoch"),
            "completed_epochs": summary.get("completed_epochs"),
            "run_dir": str(run_dir.resolve()),
            "source": "refinement",
        }
        families.setdefault(match.group("family"), []).append(row)
        if summary.get("training_source_hash"):
            source_hashes.add(summary["training_source_hash"])
    return families, source_hashes, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selection", required=True)
    parser.add_argument("--refinement-results-root", required=True)
    parser.add_argument("--stage", default="global_lr_refine")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_selection = load_json(Path(args.base_selection).resolve())
    refinement, refinement_hashes, errors = load_refinement_rows(
        Path(args.refinement_results_root), args.stage
    )
    selections = {}
    base_hashes = list(base_selection.get("training_source_hashes", []))

    for family, base_payload in sorted(base_selection.get("selections", {}).items()):
        base_rows = [dict(row, source="coarse") for row in base_payload.get("ranking", [])]
        if not base_rows:
            errors.append(f"missing coarse selection rows: {family}")
            continue
        rows = base_rows + refinement.get(family, [])
        if base_payload.get("requires_supplemental_calibration"):
            if not refinement.get(family):
                errors.append(f"missing refinement rows for boundary-selected family: {family}")
            rates = {row["learning_rate"] for row in rows}
            if len(rates) < 4:
                errors.append(f"{family}: combined LR grid must contain at least 4 rates")
        ranked = sorted(
            rows,
            key=lambda row: (
                row["tail_validation_loss_median"],
                row["best_validation_loss"],
                row["learning_rate"],
            ),
        )
        selected_rate = ranked[0]["learning_rate"]
        rates = {row["learning_rate"] for row in rows}
        interior = selected_rate not in {min(rates), max(rates)}
        if base_payload.get("requires_supplemental_calibration") and not interior:
            errors.append(
                f"{family}: combined selection {selected_rate:g} remains at the combined-grid boundary"
            )
        selections[family] = {
            "selected_learning_rate": selected_rate,
            "selection_rule": "minimum median validation-selection loss over final five validation epochs",
            "requires_supplemental_calibration": not interior,
            "combined_grid_is_interior": interior,
            "ranking": ranked,
        }

    payload = {
        "status": "completed" if not errors else "incomplete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_selection": str(Path(args.base_selection).resolve()),
        "refinement_results_root": str(Path(args.refinement_results_root).resolve()),
        "refinement_stage": args.stage,
        "base_training_source_hashes": sorted(base_hashes),
        "refinement_training_source_hashes": sorted(refinement_hashes),
        "selections": selections,
        "errors": errors,
        "note": "Coarse boundary candidates and interior refinement candidates are ranked together; no test split was accessed.",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(output)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
