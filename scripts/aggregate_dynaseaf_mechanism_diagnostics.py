#!/usr/bin/env python3
"""Aggregate completed validation mechanism-diagnostic outputs.

This is a post-processing helper for the validation-only collector.  It reads
only the collector manifests and per-sample CSVs; it never opens a checkpoint
or evaluates the test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return fmean(values), stdev(values) if len(values) > 1 else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nblocked_missing_inputs\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    manifests = sorted(input_root.rglob("diagnostics_manifest.json"))
    records: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for manifest_path in manifests:
        try:
            manifest = load_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            blocked.append({"path": str(manifest_path), "reason": f"invalid_manifest:{exc.__class__.__name__}"})
            continue
        if manifest.get("status") != "completed":
            blocked.append({"path": str(manifest_path), "reason": manifest.get("status", "unknown")})
            continue
        sample_path = Path(manifest.get("sample_mechanism_metrics", {}).get("path", ""))
        if not sample_path.is_file():
            blocked.append({"path": str(manifest_path), "reason": "missing_sample_mechanism_metrics"})
            continue
        with sample_path.open("r", encoding="utf-8", newline="") as handle:
            records.extend(dict(row) for row in csv.DictReader(handle))

    metric_fields = (
        "final_rmse_normalized",
        "direct_rmse_normalized",
        "transport_rmse_normalized",
        "innovation_rmse_normalized",
        "gate_mean",
        "gate_p10",
        "gate_median",
        "gate_p90",
        "deformation_magnitude_mean",
        "deformation_magnitude_max",
        "predicted_dynamics_abs_mean",
    )
    grouped: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        key = (row.get("model", "unknown"), row.get("seed", "unknown"), row.get("variable", "unknown"), row.get("lead", "unknown"))
        for field in metric_fields:
            if finite(row.get(field)):
                grouped[key][field].append(float(row[field]))

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        model, seed, variable, lead = key
        row: dict[str, Any] = {
            "model": model,
            "seed": seed,
            "variable": variable,
            "lead": lead,
            "sample_count": len(grouped[key].get("final_rmse_normalized", [])),
            "status": "computed_validation_only",
        }
        for field in metric_fields:
            mean, sd = mean_sd(grouped[key].get(field, []))
            row[f"{field}_mean_per_sample"] = mean
            row[f"{field}_sample_sd"] = sd
        summary_rows.append(row)

    write_csv(output_dir / "mechanism_summary.csv", summary_rows)
    qualitative = []
    completed_manifests = []
    for manifest_path in manifests:
        try:
            manifest = load_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "completed":
            completed_manifests.append(str(manifest_path))
            qualitative.extend(manifest.get("qualitative_panels", []))

    payload = {
        "schema": "dynaseaf-mechanism-summary-v1",
        "status": "computed_validation_only" if summary_rows else "blocked_missing_inputs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "completed_run_manifests": completed_manifests,
        "blocked_inputs": blocked,
        "scope": {
            "split": "validation",
            "test_iteration": False,
            "retraining": False,
            "tensor_space": "normalized model/anomaly space",
            "aggregation_note": "Means/SDs summarize per-sample mechanism statistics; final forecast errors are also available in the raw chunk files.",
        },
        "outputs": {
            "summary_csv": str((output_dir / "mechanism_summary.csv").resolve()),
            "qualitative_panels": qualitative,
        },
        "row_count": len(summary_rows),
        "source_sample_row_count": len(records),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mechanism_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "qualitative_manifest.json").write_text(
        json.dumps({
            "schema": "dynaseaf-qualitative-panel-manifest-v1",
            "status": payload["status"],
            "panels": qualitative,
            "completed_run_manifests": completed_manifests,
            "blocked_inputs": blocked,
        }, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "run_manifests": len(completed_manifests),
        "source_rows": len(records),
        "summary_rows": len(summary_rows),
        "output_dir": str(output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
