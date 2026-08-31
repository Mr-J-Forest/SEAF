#!/usr/bin/env python3
"""Aggregate an evaluation-only manifest without replacing training metrics.

The confirmation run keeps its validation report in ``validation_results.json``.
After the checkpoint is frozen, ``evaluate_campaign_checkpoints.py`` writes a
separate held-out report.  This script summarizes that manifest explicitly so
the final test numbers cannot be confused with validation-selected numbers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any


NUMERIC_FIELDS = (
    "temp_rmse",
    "salt_rmse",
    "temp_skill_ap",
    "salt_skill_ap",
    "macro_skill_ap",
    "best_val_loss",
    "best_epoch",
    "completed_epochs",
    "mean_epoch_time_seconds",
    "wall_time_seconds",
    "parameter_count",
)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(payload: Any, *keys: str):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _load_row(record: dict, split: str) -> dict:
    result_dir = Path(str(record["result_dir"])).resolve()
    evaluation_path = Path(str(record["evaluation_path"])).resolve()
    config_path = Path(str(record["config"])).resolve()
    summary_path = result_dir / "run_summary.json"
    config_file = result_dir / "config.json"
    checkpoint_path = result_dir / "best_model.pth"
    for path in (evaluation_path, summary_path, config_file, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen-evaluation artifact: {path}")
    evaluation = _read_json(evaluation_path)
    config = _read_json(config_file)
    summary = _read_json(summary_path)
    comparison = _nested(
        evaluation, "baseline_comparison", "physical", "anomaly_persistence"
    ) or {}
    peak_bytes = summary.get("peak_cuda_memory_bytes")
    return {
        "stage": result_dir.parent.parent.name,
        "experiment": result_dir.parent.name,
        "seed": config.get("seed"),
        "split": split,
        "run_dir": str(result_dir),
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": _sha256(evaluation_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path) if config_path.is_file() else None,
        "model_type": config.get("model_type"),
        "parameter_count": summary.get("parameter_count"),
        "temp_rmse": _nested(evaluation, "physical_report", "by_variable", "TEMP", "rmse"),
        "salt_rmse": _nested(evaluation, "physical_report", "by_variable", "SALT", "rmse"),
        "temp_skill_ap": _nested(comparison, "by_variable", "TEMP", "mse_skill"),
        "salt_skill_ap": _nested(comparison, "by_variable", "SALT", "mse_skill"),
        "macro_skill_ap": _nested(comparison, "macro", "mse_skill", "mean"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_epoch": summary.get("best_epoch"),
        "completed_epochs": summary.get("completed_epochs"),
        "mean_epoch_time_seconds": summary.get("mean_epoch_time_seconds"),
        "wall_time_seconds": summary.get("wall_time_seconds"),
        "peak_cuda_memory_gib": (
            float(peak_bytes) / (1024 ** 3) if _finite(peak_bytes) else None
        ),
        "training_source_hash": summary.get("training_source_hash"),
        "training_config_fingerprint": summary.get("training_config_fingerprint"),
        "evaluation_provenance": evaluation.get("evaluation_provenance"),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    public = [
        {key: value for key, value in row.items() if key != "evaluation_provenance"}
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public[0]))
        writer.writeheader()
        writer.writerows(public)


def _summary(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["experiment"]), []).append(row)
    output = []
    for experiment, members in sorted(groups.items()):
        item = {"experiment": experiment, "n_seeds": len(members)}
        for field in NUMERIC_FIELDS:
            values = [float(row[field]) for row in members if _finite(row.get(field))]
            item[f"{field}_mean"] = mean(values) if values else None
            item[f"{field}_std"] = stdev(values) if len(values) > 1 else None
        output.append(item)
    return output


def _markdown(rows: list[dict]) -> str:
    headers = [
        "Experiment", "N", "TEMP RMSE", "SALT RMSE",
        "TEMP AP skill", "SALT AP skill", "Macro AP skill", "Params",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]

    def show(row: dict, field: str, digits: int = 4) -> str:
        avg = row.get(f"{field}_mean")
        std = row.get(f"{field}_std")
        if not _finite(avg):
            return ""
        if _finite(std):
            return f"{avg:.{digits}f} ± {std:.{digits}f}"
        return f"{avg:.{digits}f}"

    for row in rows:
        lines.append(
            "| " + " | ".join([
                str(row["experiment"]),
                str(row["n_seeds"]),
                show(row, "temp_rmse"),
                show(row, "salt_rmse"),
                show(row, "temp_skill_ap"),
                show(row, "salt_skill_ap"),
                show(row, "macro_skill_ap"),
                show(row, "parameter_count", 0),
            ]) + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise RuntimeError(f"frozen evaluation manifest is not complete: {manifest_path}")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"manifest has no records: {manifest_path}")
    split = str(manifest.get("split", "unknown"))
    rows = [_load_row(record, split) for record in records]
    output_dir = (
        Path(args.output).resolve()
        if args.output else manifest_path.parent / f"{split}_aggregate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _summary(rows)
    _write_csv(output_dir / "runs.csv", rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.md").write_text(_markdown(summary_rows), encoding="utf-8")
    audit = {
        "status": "completed",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "split": split,
        "run_count": len(rows),
        "experiments": sorted({row["experiment"] for row in rows}),
        "training_source_hashes": sorted({row.get("training_source_hash") for row in rows}),
        "checkpoint_sha256": {row["run_dir"]: row["checkpoint_sha256"] for row in rows},
        "evaluation_sha256": {row["run_dir"]: row["evaluation_sha256"] for row in rows},
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
