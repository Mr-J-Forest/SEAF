#!/usr/bin/env python3
"""Export paper-ready DynaSEAF artifacts from frozen files only.

The exporter deliberately does not instantiate a model, open a checkpoint for
inference, or touch the test split.  It reads completed JSON/CSV/NPZ artifacts
already present in the workspace, writes provenance-rich tables, and makes
plots from those tables.  Missing scientific products are kept as
``NOT_FOUND`` rather than estimated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NOT_FOUND = "NOT_FOUND"
NOT_APPLICABLE_N1 = "NOT_APPLICABLE_N1"
SEEDS = (42, 123, 3407)
VARIABLES = ("TEMP", "SALT")
ABLATION_RUNS = (
    ("A0-SEAF-v1-reference", "dynaseaf_a0_seaf_reference"),
    ("A1-dynamics-aux-only", "dynaseaf_a1_dynamics_only"),
    ("A2-dynamics-plus-transport", "dynaseaf_a2_transport"),
    ("A3-dynamics-plus-transport-plus-innovation", "dynaseaf_a3_transport_innovation"),
    ("A4-full-DynaSEAF", "dynaseaf_a4_full"),
    ("no-dynamics-aux", "dynaseaf_no_dynamics_aux"),
    ("no-transport", "dynaseaf_no_transport"),
    ("no-innovation", "dynaseaf_no_innovation"),
    ("no-gate", "dynaseaf_no_gate"),
)

MAIN_VALIDATION_LOCAL = {
    42: Path("outputs/results/remote_collected/dynaseaf_full_all_seeds/seed_42"),
    123: Path("outputs/results/remote_collected/dynaseaf_full_remaining30_screen/seed_123"),
    3407: Path("outputs/results/remote_collected/dynaseaf_full_remaining30_screen/seed_3407"),
}
MAIN_VALIDATION_REMOTE = {
    42: "/root/autodl-tmp/TSC-Fusion/outputs/results/campaigns/5d106ad2d2cc1877abc23fb6b17fbfd441900509_screen/screen/dynaseaf_full/seed_42",
    123: "/root/autodl-tmp/TSC-Fusion/outputs/results/campaigns/5d106ad2d2cc1877abc23fb6b17fbfd441900509_dynaseaf_full_remaining30_screen/screen/dynaseaf_full/seed_123",
    3407: "/root/autodl-tmp/TSC-Fusion/outputs/results/campaigns/5d106ad2d2cc1877abc23fb6b17fbfd441900509_dynaseaf_full_remaining30_screen/screen/dynaseaf_full/seed_3407",
}
TEST_REMOTE_ROOT = "/root/autodl-tmp/TSC-Fusion/outputs/results/paper_ready/dynaseaf_test_eval_20260901_v1"
ABLATION_REMOTE_ROOT = "/root/autodl-tmp/TSC-Fusion/outputs/results/campaigns/01372ba982886317176140a8e9abf1879b0caa91_dynaseaf_ablation_screen30_westc/screen"
DIAGNOSTICS_REMOTE_ROOT = "/root/autodl-tmp/dynaseaf_mechanism/01372ba982886317176140a8e9abf1879b0caa91_dynaseaf_ablation_screen30_westc"


def json_load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def value_or_missing(value: Any) -> Any:
    return float(value) if finite(value) else NOT_FOUND


def nested(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def mean_sd(values: Iterable[Any]) -> tuple[Any, Any, int]:
    numeric = [float(v) for v in values if finite(v)]
    if not numeric:
        return NOT_FOUND, NOT_FOUND, 0
    if len(numeric) == 1:
        return numeric[0], NOT_APPLICABLE_N1, 1
    return fmean(numeric), stdev(numeric), len(numeric)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": NOT_FOUND}]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_if_present(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def metric_row(
    *,
    label: str,
    seed: int,
    split: str,
    evaluation: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    config: dict[str, Any] | None,
    local_eval_path: Path,
    remote_eval_path: str,
    local_run_dir: Path,
    remote_run_dir: str,
    source_campaign: str,
    source_role: str = "frozen_run",
    status: str = "completed",
) -> dict[str, Any]:
    evaluation = evaluation or {}
    summary = summary or {}
    config = config or {}
    row: dict[str, Any] = {
        "label": label,
        "seed": seed,
        "split": split,
        "status": status,
        "source_role": source_role,
        "source_campaign": source_campaign,
        "local_evaluation_path": str(local_eval_path.resolve()) if local_eval_path.is_file() else NOT_FOUND,
        "remote_evaluation_path": remote_eval_path,
        "local_run_dir": str(local_run_dir.resolve()) if local_run_dir.is_dir() else NOT_FOUND,
        "remote_run_dir": remote_run_dir,
        "evaluation_split_in_file": evaluation.get("evaluation_split", NOT_FOUND),
        "sample_count": len(nested(evaluation, "evaluation_provenance", "samples") or []) or NOT_FOUND,
        "best_epoch": value_or_missing(summary.get("best_epoch")),
        "best_val_loss": value_or_missing(summary.get("best_val_loss")),
        "parameter_count": value_or_missing(
            nested(evaluation, "model_diagnostics", "parameter_breakdown", "total")
            or summary.get("parameter_count")
        ),
        "training_evaluation_scope": summary.get("evaluation_scope", NOT_FOUND),
        "training_evaluation_file": summary.get("evaluation_file", NOT_FOUND),
        "git_dirty": summary.get("git_dirty", NOT_FOUND),
        "training_source_hash": summary.get("training_source_hash", NOT_FOUND),
    }
    metric_sources = {
        "normalized_MAE": evaluation.get("normalized_mae"),
        "normalized_RMSE": evaluation.get("normalized_rmse"),
        "normalized_R2": evaluation.get("normalized_r2"),
        "overall_MAE": evaluation.get("mae"),
        "overall_RMSE": evaluation.get("rmse"),
        "overall_R2": evaluation.get("r2"),
        "Macro_SS_AP": nested(
            evaluation,
            "baseline_comparison",
            "physical",
            "anomaly_persistence",
            "macro",
            "mse_skill",
            "mean",
        ),
        "Macro_SS_DAP": nested(
            evaluation,
            "baseline_comparison",
            "physical",
            "damped_anomaly_persistence",
            "macro",
            "mse_skill",
            "mean",
        ),
    }
    for name, value in metric_sources.items():
        row[name] = value_or_missing(value)
    for variable in VARIABLES:
        for space, report_name in (("normalized", "normalized_report"), ("physical", "physical_report")):
            report = nested(evaluation, report_name, "by_variable", variable) or {}
            for metric in ("mae", "rmse", "r2"):
                row[f"{variable}_{space}_{metric.upper()}"] = value_or_missing(report.get(metric))
        row[f"{variable}_physical_RMSE"] = value_or_missing(evaluation.get(f"physical_rmse_{variable}"))
        row[f"{variable}_physical_MAE"] = value_or_missing(evaluation.get(f"physical_mae_{variable}"))
        row[f"{variable}_normalized_RMSE"] = value_or_missing(evaluation.get(f"normalized_rmse_{variable}"))
        row[f"{variable}_normalized_MAE"] = value_or_missing(evaluation.get(f"normalized_mae_{variable}"))
        row[f"{variable}_R2"] = value_or_missing(evaluation.get(f"r2_{variable}"))
    return row


def source_campaign_from_summary(summary: dict[str, Any] | None, fallback: str) -> str:
    result_dir = str((summary or {}).get("result_dir", ""))
    marker = "/campaigns/"
    if marker in result_dir:
        return result_dir.split(marker, 1)[1].split("/", 1)[0]
    return fallback


def load_run_metadata(run_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    summary = json_load(run_dir / "run_summary.json")
    config = json_load(run_dir / "config.json") or json_load(run_dir / "eval_harness_config.json")
    provenance = json_load(run_dir / "provenance_manifest.json")
    return summary, config, provenance


def collect_main_records(project_root: Path, test_source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        local_run = project_root / MAIN_VALIDATION_LOCAL[seed]
        local_eval = local_run / "validation_results.json"
        summary, config, provenance = load_run_metadata(local_run)
        status = "completed"
        if summary is None and provenance:
            status = str(provenance.get("validation_acceptance", {}).get("accepted_for_validation_aggregate", "accepted"))
        validation_rows.append(
            metric_row(
                label="DynaSEAF",
                seed=seed,
                split="validation",
                evaluation=json_load(local_eval),
                summary=summary,
                config=config,
                local_eval_path=local_eval,
                remote_eval_path=f"{MAIN_VALIDATION_REMOTE[seed]}/validation_results.json",
                local_run_dir=local_run,
                remote_run_dir=MAIN_VALIDATION_REMOTE[seed],
                source_campaign=source_campaign_from_summary(summary, "5d106ad2d2cc1877abc23fb6b17fbfd441900509_dynaseaf_full_remaining30_screen" if seed != 42 else "5d106ad2d2cc1877abc23fb6b17fbfd441900509_screen"),
                status=status,
            )
        )
        test_run = test_source / f"seed_{seed}"
        test_eval = test_run / "evaluation_results.json"
        test_summary = json_load(test_run / "run_summary.json")
        test_config = json_load(test_run / "eval_harness_config.json") or json_load(test_run / "config.json")
        test_status = "completed_existing_test_evaluation" if test_eval.is_file() else NOT_FOUND
        test_rows.append(
            metric_row(
                label="DynaSEAF",
                seed=seed,
                split="test",
                evaluation=json_load(test_eval),
                summary=test_summary,
                config=test_config,
                local_eval_path=test_eval,
                remote_eval_path=f"{TEST_REMOTE_ROOT}/seed_{seed}/evaluation_results.json",
                local_run_dir=test_run,
                remote_run_dir=f"{TEST_REMOTE_ROOT}/seed_{seed}",
                source_campaign=source_campaign_from_summary(summary, "5d106ad2d2cc1877abc23fb6b17fbfd441900509_test_eval_export"),
                source_role="frozen_checkpoint_test_export",
                status=test_status,
            )
        )
    return validation_rows, test_rows


def collect_ablation_records(project_root: Path, ablation_source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, run_name in ABLATION_RUNS:
        run_dir = ablation_source / "screen" / run_name / "seed_42"
        eval_path = run_dir / "validation_results.json"
        summary, config, provenance = load_run_metadata(run_dir)
        rows.append(
            metric_row(
                label=label,
                seed=42,
                split="validation",
                evaluation=json_load(eval_path),
                summary=summary,
                config=config,
                local_eval_path=eval_path,
                remote_eval_path=f"{ABLATION_REMOTE_ROOT}/{run_name}/seed_42/validation_results.json",
                local_run_dir=run_dir,
                remote_run_dir=f"{ABLATION_REMOTE_ROOT}/{run_name}/seed_42",
                source_campaign="01372ba982886317176140a8e9abf1879b0caa91_dynaseaf_ablation_screen30_westc",
                source_role="exact_ablation_screen30",
                status="completed" if eval_path.is_file() else NOT_FOUND,
            )
        )
    # The main full runs are the same A4 configuration for seeds 123 and 3407,
    # but they are kept explicitly marked as a different campaign.
    for seed in (123, 3407):
        local_run = project_root / MAIN_VALIDATION_LOCAL[seed]
        eval_path = local_run / "validation_results.json"
        summary, config, provenance = load_run_metadata(local_run)
        rows.append(
            metric_row(
                label="A4-full-DynaSEAF",
                seed=seed,
                split="validation",
                evaluation=json_load(eval_path),
                summary=summary,
                config=config,
                local_eval_path=eval_path,
                remote_eval_path=f"{MAIN_VALIDATION_REMOTE[seed]}/validation_results.json",
                local_run_dir=local_run,
                remote_run_dir=MAIN_VALIDATION_REMOTE[seed],
                source_campaign=source_campaign_from_summary(summary, "5d106ad2d2cc1877abc23fb6b17fbfd441900509_dynaseaf_full_remaining30_screen"),
                source_role="same_A4_configuration_from_main_frozen_run",
                status="completed" if eval_path.is_file() else NOT_FOUND,
            )
        )
    return rows


def add_missing_ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["label"], int(r["seed"])): r for r in rows}
    all_rows: list[dict[str, Any]] = []
    for label, _ in ABLATION_RUNS:
        for seed in SEEDS:
            row = by_key.get((label, seed))
            if row is None:
                row = {
                    "label": label,
                    "seed": seed,
                    "split": "validation",
                    "status": NOT_FOUND,
                    "source_role": "no_frozen_artifact_found",
                    "source_campaign": "01372ba982886317176140a8e9abf1879b0caa91_dynaseaf_ablation_screen30_westc",
                    "local_evaluation_path": NOT_FOUND,
                    "remote_evaluation_path": NOT_FOUND,
                    "local_run_dir": NOT_FOUND,
                    "remote_run_dir": f"{ABLATION_REMOTE_ROOT}/{dict(ABLATION_RUNS)[label]}/seed_{seed}",
                }
            all_rows.append(row)
    return all_rows


def metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    excluded = {
        "label", "seed", "split", "status", "source_role", "source_campaign",
        "local_evaluation_path", "remote_evaluation_path", "local_run_dir", "remote_run_dir",
        "evaluation_split_in_file", "training_evaluation_scope", "training_evaluation_file",
        "git_dirty", "training_source_hash", "sample_count", "best_epoch", "best_val_loss",
    }
    keys = []
    for row in rows:
        for key, value in row.items():
            if key not in excluded and key not in keys and any(finite(r.get(key)) for r in rows):
                keys.append(key)
    return keys


def aggregate_metric_rows(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        buckets.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    cols = metric_columns(rows)
    for key in sorted(buckets, key=lambda item: tuple(str(v) for v in item)):
        members = buckets[key]
        base = {field: value for field, value in zip(group_fields, key)}
        base["n_rows"] = len(members)
        base["source_campaigns"] = ";".join(sorted({str(m.get("source_campaign", NOT_FOUND)) for m in members}))
        for col in cols:
            avg, sd, n = mean_sd(m.get(col) for m in members)
            base[f"{col}_mean"] = avg
            base[f"{col}_sample_sd"] = sd
            base[f"{col}_n"] = n
        output.append(base)
    return output


def extract_dimension_rows(rows: list[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if not finite(row.get("seed")):
            continue
        evaluation_path = row.get("local_evaluation_path")
        if not evaluation_path or evaluation_path == NOT_FOUND:
            continue
        evaluation = json_load(Path(evaluation_path))
        if evaluation is None:
            continue
        physical = nested(evaluation, "physical_report") or {}
        normalized = nested(evaluation, "normalized_report") or {}
        if dimension == "lead":
            pkey = "by_variable_and_lead"
            nkey = "by_variable_and_lead"
        else:
            pkey = "by_variable_and_depth"
            nkey = "by_variable_and_depth"
        p_report = physical.get(pkey) or {}
        n_report = normalized.get(nkey) or {}
        for variable, groups in p_report.items():
            for group_key, fields in groups.items():
                fields = fields if isinstance(fields, dict) else {}
                n_fields = nested(n_report, variable, group_key) or {}
                out = {
                    "label": row["label"],
                    "seed": row["seed"],
                    "split": row["split"],
                    "variable": variable,
                    "dimension_key": group_key,
                    "source_campaign": row["source_campaign"],
                    "source_evaluation_path": evaluation_path,
                    "physical_unit": fields.get("unit", NOT_FOUND),
                    "depth": fields.get("depth", NOT_FOUND) if dimension == "depth" else NOT_FOUND,
                    "physical_MSE": value_or_missing(fields.get("mse")),
                    "physical_MAE": value_or_missing(fields.get("mae")),
                    "physical_RMSE": value_or_missing(fields.get("rmse")),
                    "physical_R2": value_or_missing(fields.get("r2")),
                    "physical_correlation": value_or_missing(fields.get("correlation")),
                    "normalized_MSE": value_or_missing(n_fields.get("mse")),
                    "normalized_MAE": value_or_missing(n_fields.get("mae")),
                    "normalized_RMSE": value_or_missing(n_fields.get("rmse")),
                    "normalized_R2": value_or_missing(n_fields.get("r2")),
                    "normalized_correlation": value_or_missing(n_fields.get("correlation")),
                }
                if dimension == "lead":
                    try:
                        out["lead"] = int(str(group_key).split("_")[-1])
                    except ValueError:
                        out["lead"] = group_key
                output.append(out)
    return output


def extract_region_catalog(evaluation: dict[str, Any] | None) -> list[dict[str, Any]]:
    regions = nested(evaluation or {}, "evaluation_provenance", "regions") or {}
    rows = []
    for region_id, region in regions.items():
        region = region if isinstance(region, dict) else {}
        lon = region.get("lon_range", [NOT_FOUND, NOT_FOUND])
        lat = region.get("lat_range", [NOT_FOUND, NOT_FOUND])
        rows.append({
            "region_id": region_id,
            "region_type": region.get("region_type", NOT_FOUND),
            "lon_min": lon[0] if len(lon) > 0 else NOT_FOUND,
            "lon_max": lon[1] if len(lon) > 1 else NOT_FOUND,
            "lat_min": lat[0] if len(lat) > 0 else NOT_FOUND,
            "lat_max": lat[1] if len(lat) > 1 else NOT_FOUND,
            "center_lon": region.get("center_lon", NOT_FOUND),
            "center_lat": region.get("center_lat", NOT_FOUND),
        })
    return rows


def export_paths(
    project_root: Path,
    output: Path,
    main_validation: list[dict[str, Any]],
    main_test: list[dict[str, Any]],
    ablation: list[dict[str, Any]],
    diagnostics_source: Path,
) -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    data_dir = output / "data"
    source_dir = output / "source"
    figure_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Copy only small, already-produced source artifacts.  Checkpoints and raw
    # mechanism chunks are intentionally represented by paths in manifests.
    for split, rows, subdir in (("validation", main_validation, "main_validation"), ("test", main_test, "main_test")):
        for row in rows:
            if row["local_evaluation_path"] != NOT_FOUND:
                source = Path(row["local_evaluation_path"])
                dest = source_dir / subdir / f"seed_{row['seed']}" / source.name
                copy_if_present(source, dest)
                source_rows.append({"kind": "evaluation", "split": split, "seed": row["seed"], "label": row["label"], "local_path": str(source.resolve()), "stable_copy": str(dest.resolve()), "remote_path": row["remote_evaluation_path"], "sha256": sha256(source), "size_bytes": source.stat().st_size})
    for label, run_name in ABLATION_RUNS:
        src_dir = diagnostics_source / "screen" / run_name / "seed_42"
        for name in ("diagnostics_manifest.json", "sample_mechanism_metrics.csv"):
            source = src_dir / name
            dest = source_dir / "mechanism" / run_name / "seed_42" / name
            if copy_if_present(source, dest):
                source_rows.append({"kind": "mechanism_source", "split": "validation", "seed": 42, "label": label, "local_path": str(source.resolve()), "stable_copy": str(dest.resolve()), "remote_path": f"{DIAGNOSTICS_REMOTE_ROOT}/screen/{run_name}/seed_42/{name}", "sha256": sha256(source), "size_bytes": source.stat().st_size})
        for name in ("temp_qualitative_panel.png", "salt_qualitative_panel.png"):
            source = src_dir / name
            dest = figure_dir / "qualitative_validation_remote" / f"{run_name}_seed42_{name}"
            if copy_if_present(source, dest):
                source_rows.append({"kind": "qualitative_validation", "split": "validation", "seed": 42, "label": label, "local_path": str(source.resolve()), "stable_copy": str(dest.resolve()), "remote_path": f"{DIAGNOSTICS_REMOTE_ROOT}/screen/{run_name}/seed_42/{name}", "sha256": sha256(source), "size_bytes": source.stat().st_size})
    aggregate_source = diagnostics_source / "aggregate" / "mechanism_summary.csv"
    aggregate_json_source = diagnostics_source / "aggregate" / "mechanism_summary.json"
    qualitative_manifest_source = diagnostics_source / "aggregate" / "qualitative_manifest.json"
    for source in (aggregate_source, aggregate_json_source, qualitative_manifest_source):
        dest = data_dir / source.name
        if copy_if_present(source, dest):
            source_rows.append({"kind": "mechanism_aggregate", "split": "validation", "seed": 42, "label": "all_diagnostic_runs", "local_path": str(source.resolve()), "stable_copy": str(dest.resolve()), "remote_path": f"{DIAGNOSTICS_REMOTE_ROOT}/aggregate/{source.name}", "sha256": sha256(source), "size_bytes": source.stat().st_size})

    write_csv(data_dir / "source_index.csv", source_rows)
    write_csv(data_dir / "main_metrics_per_seed.csv", main_validation + main_test)
    main_mean_sd = aggregate_metric_rows(main_validation + main_test, ("split",))
    write_csv(data_dir / "main_metrics_mean_sd.csv", main_mean_sd)
    write_csv(data_dir / "ablation_per_seed.csv", ablation)
    write_csv(data_dir / "ablation_mean_sd.csv", aggregate_metric_rows(ablation, ("label", "split")))

    for split, rows in (("validation", main_validation), ("test", main_test)):
        lead_rows = extract_dimension_rows(rows, "lead")
        depth_rows = extract_dimension_rows(rows, "depth")
        write_csv(data_dir / f"{split}_lead_variable_per_seed.csv", lead_rows)
        write_csv(data_dir / f"{split}_lead_variable_mean_sd.csv", aggregate_metric_rows(lead_rows, ("split", "variable", "lead")))
        write_csv(data_dir / f"{split}_depth_variable_per_seed.csv", depth_rows)
        write_csv(data_dir / f"{split}_depth_variable_mean_sd.csv", aggregate_metric_rows(depth_rows, ("split", "variable", "dimension_key")))

    first_eval = json_load(project_root / MAIN_VALIDATION_LOCAL[42] / "validation_results.json")
    write_csv(data_dir / "region_catalog.csv", extract_region_catalog(first_eval))
    region_status = {
        "status": NOT_FOUND,
        "test_split": NOT_FOUND,
        "reason": "Frozen evaluation JSONs contain evaluation_provenance.region_ids/regions but no by_region metrics or raw mainline prediction arrays. Region x variable metrics cannot be reconstructed without a new evaluation, which is intentionally not run.",
        "source_evaluation_paths": [r["local_evaluation_path"] for r in main_validation if r["local_evaluation_path"] != NOT_FOUND],
        "region_catalog": str((data_dir / "region_catalog.csv").resolve()),
    }
    write_json(data_dir / "region_variable_metrics_status.json", region_status)

    # Parameter decomposition is present in the frozen evaluation JSON.
    parameter_rows = []
    seen = set()
    for row in main_validation + main_test:
        path = row.get("local_evaluation_path")
        if path == NOT_FOUND or not path:
            continue
        evaluation = json_load(Path(path)) or {}
        breakdown = nested(evaluation, "model_diagnostics", "parameter_breakdown") or {}
        for component, count in breakdown.items():
            key = (row["split"], row["seed"], component)
            if key in seen:
                continue
            seen.add(key)
            parameter_rows.append({"model": "DynaSEAF", "split": row["split"], "seed": row["seed"], "component": component, "parameter_count": count, "source": path})
    write_csv(data_dir / "parameter_breakdown.csv", parameter_rows)

    # Copy mechanism summary and create per-configuration post-processing.
    mechanism_rows: list[dict[str, Any]] = []
    for label, run_name in ABLATION_RUNS:
        csv_path = diagnostics_source / "screen" / run_name / "seed_42" / "sample_mechanism_metrics.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                raw["label"] = label
                raw["run_name"] = run_name
                mechanism_rows.append(raw)
    mechanism_fields = [
        "label", "run_name", "model", "seed", "split", "sample_index", "origin_id", "region_id", "variable", "lead",
        "final_rmse_normalized", "direct_rmse_normalized", "transport_rmse_normalized", "innovation_rmse_normalized",
        "gate_mean", "gate_p10", "gate_median", "gate_p90", "deformation_magnitude_mean", "deformation_magnitude_max", "predicted_dynamics_abs_mean",
    ]
    write_csv(data_dir / "mechanism_per_sample.csv", mechanism_rows)
    mechanism_stats = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in mechanism_rows:
        grouped.setdefault((row["label"], row["variable"], row["lead"]), []).append(row)
    mechanism_numeric = [
        "final_rmse_normalized", "direct_rmse_normalized", "transport_rmse_normalized", "innovation_rmse_normalized",
        "gate_mean", "gate_p10", "gate_median", "gate_p90", "deformation_magnitude_mean", "deformation_magnitude_max", "predicted_dynamics_abs_mean",
    ]
    for (label, variable, lead), members in sorted(grouped.items()):
        out = {"label": label, "variable": variable, "lead": lead, "split": "validation", "seed": 42, "sample_count": len(members), "source": str((diagnostics_source / "screen").resolve())}
        for field in mechanism_numeric:
            avg, sd, n = mean_sd([m.get(field) for m in members])
            out[f"{field}_mean"] = avg
            out[f"{field}_sample_sd"] = sd
            out[f"{field}_n"] = n
        # These are requested quantities with no corresponding frozen field.
        out["innovation_magnitude"] = NOT_FOUND
        out["direct_MAE"] = NOT_FOUND
        out["transport_MAE"] = NOT_FOUND
        out["final_MAE"] = NOT_FOUND
        out["future_dynamics_prediction_error"] = NOT_FOUND
        mechanism_stats.append(out)
    write_csv(data_dir / "mechanism_stats_by_config.csv", mechanism_stats)

    paired_rows = []
    for key, members in sorted(grouped.items()):
        label, variable, lead = key
        for candidate in ("transport_rmse_normalized", "innovation_rmse_normalized", "final_rmse_normalized"):
            diffs = []
            for member in members:
                if not finite(member.get(candidate)) or not finite(member.get("direct_rmse_normalized")):
                    continue
                diffs.append(float(member[candidate]) - float(member["direct_rmse_normalized"]))
            avg, sd, n = mean_sd(diffs)
            paired_rows.append({"label": label, "variable": variable, "lead": lead, "comparison": f"{candidate}-direct_rmse", "n": n, "mean_difference": avg, "sample_sd": sd, "fraction_candidate_better": (sum(d < 0 for d in diffs) / len(diffs)) if diffs else NOT_FOUND, "split": "validation", "seed": 42})
    write_csv(data_dir / "mechanism_paired_comparison.csv", paired_rows)
    write_json(data_dir / "mechanism_missing_status.json", {
        "scope": "validation only; seed42; existing diagnostic CSV/NPZ artifacts",
        "future_dynamics_prediction_error": NOT_FOUND,
        "reason_future_dynamics_truth_not_saved": "Diagnostic NPZ schema contains predicted_dynamics but not future_dynamics_targets; collector explicitly set return_future_dynamics_targets=False.",
        "innovation_magnitude": NOT_FOUND,
        "reason_innovation_magnitude": "Existing per-sample CSV stores innovation RMSE against target, not innovation absolute/magnitude; raw arrays are indexed but no magnitude aggregate was frozen.",
        "direct_transport_final_MAE": NOT_FOUND,
        "reason_MAE": "Existing per-sample CSV stores RMSE only.",
        "depth_specific_gate_deformation_stats": NOT_FOUND,
        "reason_depth": "Existing per-sample CSV aggregates over depth; raw NPZ chunks retain depth axes and are left remote rather than re-read into a new aggregate.",
        "test_mechanism_arrays": NOT_FOUND,
    })

    return {
        "source_rows": len(source_rows),
        "main_validation_rows": len(main_validation),
        "main_test_rows": len(main_test),
        "ablation_rows": len(ablation),
        "mechanism_per_sample_rows": len(mechanism_rows),
        "mechanism_stats_rows": len(mechanism_stats),
    }


def save_plot(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def configure_plot() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.facecolor": "white",
    })


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_ablation(output: Path) -> None:
    rows = read_csv_rows(output / "data/ablation_per_seed.csv")
    labels = [label for label, _ in ABLATION_RUNS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, variable in zip(axes, VARIABLES):
        means = []
        errs = []
        for label in labels:
            values = [float(r[f"{variable}_physical_RMSE"]) for r in rows if r.get("label") == label and finite(r.get(f"{variable}_physical_RMSE"))]
            mean, sd, _ = mean_sd(values)
            means.append(float(mean) if finite(mean) else np.nan)
            errs.append(float(sd) if finite(sd) else 0.0)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=errs, capsize=3, color="#4472C4")
        ax.set_xticks(x, [str(i) for i in range(len(labels))])
        ax.set_xlabel("configuration index")
        ax.set_ylabel(f"physical {variable} RMSE")
        ax.set_title(f"Ablation validation ({variable}); seed values available")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("DynaSEAF ablation screen30 — frozen validation artifacts")
    fig.text(0.5, -0.02, "0=A0 reference, 1=A1, 2=A2, 3=A3, 4=A4 full, 5–8=no_* controls; error bars are sample SD when n>1", ha="center", fontsize=8)
    # The screen30 ablation artifacts are validation-only. Keep this useful
    # diagnostic figure outside the formal/test figure area.
    save_plot(fig, output / "figures/diagnostic_only_validation/ablation_validation_physical_rmse")


def plot_main_lead(output: Path, split: str) -> None:
    rows = read_csv_rows(output / "data" / f"{split}_lead_variable_per_seed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    for ax, variable in zip(axes, VARIABLES):
        leads = sorted({int(r["lead"]) for r in rows if r.get("variable") == variable and finite(r.get("physical_RMSE"))})
        means, errs = [], []
        for lead in leads:
            vals = [float(r["physical_RMSE"]) for r in rows if r.get("variable") == variable and int(r["lead"]) == lead and finite(r.get("physical_RMSE"))]
            avg, sd, _ = mean_sd(vals)
            means.append(float(avg)); errs.append(float(sd) if finite(sd) else 0.0)
        if leads:
            ax.errorbar(leads, means, yerr=errs, marker="o", capsize=3, color="#C00000", label="DynaSEAF")
        ax.set_title(f"{split} — {variable}")
        ax.set_xlabel("lead")
        ax.set_ylabel("physical RMSE")
        ax.grid(alpha=0.25)
    fig.suptitle("Main DynaSEAF lead-wise physical RMSE; mean ± sample SD")
    save_plot(fig, output / "figures" / f"main_{split}_lead_physical_rmse")


def plot_main_depth(output: Path, split: str) -> None:
    rows = read_csv_rows(output / "data" / f"{split}_depth_variable_per_seed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, variable in zip(axes, VARIABLES):
        by_depth: dict[float, list[float]] = {}
        for r in rows:
            if r.get("variable") != variable or not finite(r.get("depth")) or not finite(r.get("physical_RMSE")):
                continue
            by_depth.setdefault(float(r["depth"]), []).append(float(r["physical_RMSE"]))
        depths = sorted(by_depth)
        means = [fmean(by_depth[d]) for d in depths]
        errs = [float(stdev(by_depth[d])) if len(by_depth[d]) > 1 else 0.0 for d in depths]
        if depths:
            ax.errorbar(depths, means, yerr=errs, marker="o", markersize=3, capsize=2, color="#70AD47")
        ax.set_title(f"{split} — {variable}")
        ax.set_xlabel("depth coordinate in frozen report")
        ax.set_ylabel("physical RMSE")
        ax.grid(alpha=0.25)
    fig.suptitle("Main DynaSEAF depth-wise physical RMSE; mean ± sample SD")
    save_plot(fig, output / "figures" / f"main_{split}_depth_physical_rmse")


def plot_mechanism(output: Path) -> None:
    rows = read_csv_rows(output / "data/mechanism_stats_by_config.csv")
    rows = [r for r in rows if r.get("label") == "A4-full-DynaSEAF"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for variable, color in zip(VARIABLES, ("#C00000", "#4472C4")):
        selected = [r for r in rows if r.get("variable") == variable]
        leads = sorted({int(r["lead"]) for r in selected})
        final = [float(next(r["final_rmse_normalized_mean"] for r in selected if int(r["lead"]) == lead)) for lead in leads]
        direct = [float(next(r["direct_rmse_normalized_mean"] for r in selected if int(r["lead"]) == lead)) for lead in leads]
        transport = [float(next(r["transport_rmse_normalized_mean"] for r in selected if int(r["lead"]) == lead)) for lead in leads]
        innovation = [float(next(r["innovation_rmse_normalized_mean"] for r in selected if int(r["lead"]) == lead)) for lead in leads]
        axes[0].plot(leads, final, marker="o", color=color, label=f"final {variable}")
        axes[0].plot(leads, direct, linestyle="--", color=color, alpha=0.65, label=f"direct {variable}")
        axes[0].plot(leads, transport, linestyle=":", color=color, alpha=0.75, label=f"transport {variable}")
        axes[0].plot(leads, innovation, linestyle="-.", color=color, alpha=0.75, label=f"innovation {variable}")
    axes[0].set_title("A4 validation component RMSE")
    axes[0].set_xlabel("lead")
    axes[0].set_ylabel("normalized RMSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=7)
    for variable, color in zip(VARIABLES, ("#C00000", "#4472C4")):
        selected = [r for r in rows if r.get("variable") == variable]
        leads = sorted({int(r["lead"]) for r in selected})
        gate = [float(next(r["gate_mean_mean"] for r in selected if int(r["lead"]) == lead)) for lead in leads]
        axes[1].plot(leads, gate, marker="o", color=color, label=variable)
    axes[1].set_title("A4 adaptive gate mean")
    axes[1].set_xlabel("lead")
    axes[1].set_ylabel("gate value")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Mechanism statistics from existing validation sample CSV; seed 42")
    save_plot(fig, output / "figures/diagnostic_only_validation/mechanism_a4_validation")


def plot_raw_validation_panel(project_root: Path, output: Path) -> dict[str, Any]:
    a4 = project_root / "outputs/dynaseaf_mechanism/01372ba982886317176140a8e9abf1879b0caa91_dynaseaf_ablation_screen30_westc/screen/dynaseaf_a4_full/seed_42/chunk_00000.npz"
    a0 = project_root / "outputs/dynaseaf_mechanism/01372ba982886317176140a8e9abf1879b0caa91_dynaseaf_ablation_screen30_westc/screen/dynaseaf_a0_seaf_reference/seed_42/chunk_00000.npz"
    if not a4.is_file() or not a0.is_file():
        return {"status": NOT_FOUND, "reason": "No local A4/A0 first diagnostic chunk; no new raw download was started."}
    with np.load(a4, allow_pickle=False) as z4, np.load(a0, allow_pickle=False) as z0:
        if "direct_forecast" not in z4.files or "forecast_normalized" not in z0.files:
            return {"status": NOT_FOUND, "reason": "Required component arrays absent."}
        sample_indices = z4["sample_indices"]
        position = int(np.where(sample_indices == 0)[0][0]) if np.any(sample_indices == 0) else 0
        lead = 0
        fields = {}
        for variable, channel in (("TEMP", 0), ("SALT", 20)):
            target = z4["target_normalized"][position, lead, channel]
            comparator = z0["forecast_normalized"][position, lead, channel]
            fields[variable] = {
                "target": target,
                "final": z4["forecast_normalized"][position, lead, channel],
                "direct": z4["direct_forecast"][position, lead, channel],
                "transport": z4["transport_forecast"][position, lead, channel],
                "innovation": z4["innovation"][position, lead, channel],
                "gate": z4["gate"][position, lead, channel],
                "deformation_magnitude": np.linalg.norm(z4["deformation"][position, lead, 0], axis=-1),
                "SEAF-v1_error": comparator - target,
            }
    fig, axes = plt.subplots(8, 2, figsize=(8, 23), squeeze=False)
    names = ("target", "final", "direct", "transport", "innovation", "gate", "deformation_magnitude", "SEAF-v1_error")
    for col, variable in enumerate(VARIABLES):
        for row, name in enumerate(names):
            image = axes[row, col].imshow(fields[variable][name], cmap="coolwarm" if name != "gate" else "viridis", aspect="auto")
            axes[row, col].set_title(f"{variable} — {name}")
            axes[row, col].set_xlabel("x index")
            axes[row, col].set_ylabel("y index")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.suptitle("Existing validation decomposition — A4 seed42, sample 0, lead 1; normalized/anomaly space", y=0.995)
    save_plot(fig, output / "figures/qualitative_validation_a4_seed42_sample0_lead1")
    return {
        "status": "completed_from_existing_npz",
        "split": "validation",
        "seed": 42,
        "sample_index": 0,
        "origin_region_source": "mechanism CSV sample_index=0",
        "lead": 1,
        "variables": list(VARIABLES),
        "fields": list(names),
        "source_npz": str(a4.resolve()),
        "comparator_source_npz": str(a0.resolve()),
        "units": "normalized/anomaly; spatial axes are array indices because coordinate arrays were not stored in NPZ",
        "test_panel": NOT_FOUND,
        "latest_anomaly": NOT_FOUND,
    }


def write_artifact_paths(output: Path, main_validation: list[dict[str, Any]], main_test: list[dict[str, Any]], ablation: list[dict[str, Any]]) -> None:
    rows = []
    for row in main_validation + main_test + ablation:
        mapping = {
            "checkpoint": f"{row.get('remote_run_dir', NOT_FOUND)}/best_model.pth",
            "config": f"{row.get('remote_run_dir', NOT_FOUND)}/config.json",
            "run_summary": f"{row.get('remote_run_dir', NOT_FOUND)}/run_summary.json",
            "_SUCCESS": f"{row.get('remote_run_dir', NOT_FOUND)}/_SUCCESS",
            "validation_results": f"{row.get('remote_run_dir', NOT_FOUND)}/validation_results.json",
            "evaluation_results": row.get("remote_evaluation_path", NOT_FOUND) if row.get("split") == "test" else NOT_FOUND,
        }
        for artifact, remote_path in mapping.items():
            local_run = Path(row.get("local_run_dir", "")) if row.get("local_run_dir") != NOT_FOUND else Path(".")
            local_name = "evaluation_results.json" if artifact == "evaluation_results" else artifact + (".json" if artifact in {"config", "run_summary", "validation_results"} else "")
            local_path = local_run / local_name
            rows.append({"label": row.get("label"), "seed": row.get("seed"), "split": row.get("split"), "source_campaign": row.get("source_campaign"), "artifact": artifact, "local_path": str(local_path.resolve()) if local_path.is_file() else NOT_FOUND, "remote_path": remote_path, "present": local_path.is_file()})
    write_csv(output / "data/artifact_paths.csv", rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--test-source", default="tmp/remote_test_eval_20260901_v1", type=Path)
    parser.add_argument("--ablation-source", default="tmp/remote_ablation_screen30", type=Path)
    parser.add_argument("--diagnostics-source", default="tmp/remote_diagnostics_metadata", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output = (project_root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    test_source = (project_root / args.test_source).resolve() if not args.test_source.is_absolute() else args.test_source.resolve()
    ablation_source = (project_root / args.ablation_source).resolve() if not args.ablation_source.is_absolute() else args.ablation_source.resolve()
    diagnostics_source = (project_root / args.diagnostics_source).resolve() if not args.diagnostics_source.is_absolute() else args.diagnostics_source.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    validation, test = collect_main_records(project_root, test_source)
    ablation = add_missing_ablation_rows(collect_ablation_records(project_root, ablation_source))
    counts = export_paths(project_root, output, validation, test, ablation, diagnostics_source)
    plot_ablation(output)
    plot_main_lead(output, "validation")
    plot_main_lead(output, "test")
    plot_main_depth(output, "validation")
    plot_main_depth(output, "test")
    plot_mechanism(output)
    qualitative_status = plot_raw_validation_panel(project_root, output)
    write_json(output / "data/qualitative_status.json", qualitative_status)
    write_artifact_paths(output, validation, test, ablation)
    missing = {
        "main_region_variable_metrics": NOT_FOUND,
        "ablation_test_metrics": NOT_FOUND,
        "ablation_seeds_123_3407_except_A4": "NOT_FOUND",
        "test_mechanism_arrays": NOT_FOUND,
        "test_qualitative_panels": NOT_FOUND,
        "formal_test_ablation_plot": NOT_FOUND,
        "formal_test_mechanism_plot": NOT_FOUND,
        "future_dynamics_prediction_error": NOT_FOUND,
        "mechanism_innovation_magnitude": NOT_FOUND,
        "mechanism_direct_transport_final_MAE": NOT_FOUND,
        "mechanism_depth_specific_gate_deformation": NOT_FOUND,
        "mainline_seed3407_run_summary": NOT_FOUND,
        "mainline_seed3407_test_checkpoint_selection_persisted_evidence": "PARTIAL_NOT_VERIFIABLE",
    }
    write_json(output / "data/missing_items.json", missing)
    write_json(output / "export_manifest.json", {
        "schema": "dynaseaf-paper-artifacts-export-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output),
        "scope": {
            "training_started": False,
            "test_evaluation_started": False,
            "checkpoint_rewritten": False,
            "source_campaign_main": "5d106ad2d2cc1877abc23fb6b17fbfd441900509_screen plus 5d106ad2d2cc1877abc23fb6b17fbfd441900509_dynaseaf_full_remaining30_screen",
            "source_campaign_ablation": "01372ba982886317176140a8e9abf1879b0caa91_dynaseaf_ablation_screen30_westc",
            "main_validation_seeds": list(SEEDS),
            "main_test_seeds": list(SEEDS),
            "ablation_validation_seed42": True,
            "test_split_used_for_checkpoint_selection": False,
            "formal_test_figures_require_test_predictions": True,
            "validation_only_figures_are_diagnostic": True,
        },
        "commands": {
            "export": f"python scripts/export_dynaseaf_paper_artifacts.py --output {output}",
            "source_test_eval": "existing remote evaluation_results.json; not re-run in this export",
            "source_mechanism_aggregate": "existing remote aggregate/mechanism_summary.csv; not recomputed from checkpoints",
            "plot_rendering": "same exporter command; PDF vector plus PNG dpi=300",
        },
        "counts": counts,
        "qualitative_status": qualitative_status,
        "missing_items": missing,
        "source_paths": {
            "remote_project": "/root/TSC-Fusion",
            "remote_diagnostics": DIAGNOSTICS_REMOTE_ROOT,
            "remote_test_eval": TEST_REMOTE_ROOT,
            "local_partial_raw_diagnostics": str((project_root / "outputs/dynaseaf_mechanism").resolve()),
        },
    })
    (output / "README.md").write_text(
        "# DynaSEAF paper-ready artifacts\n\n"
        "This directory was generated from frozen JSON/CSV/NPZ artifacts. It does not retrain, rerun test, or rewrite checkpoints.\n\n"
        "- `data/main_metrics_per_seed.csv`: main validation/test values.\n"
        "- `data/main_metrics_mean_sd.csv`: mean and sample SD across available seeds.\n"
        "- `data/ablation_per_seed.csv`: A0-A4 and controls; missing seeds are `NOT_FOUND`.\n"
        "- `data/*lead*` and `data/*depth*`: raw and aggregated mainline details.\n"
        "- `data/mechanism_*`: validation seed42 mechanism CSVs and paired summaries.\n"
        "- `figures/`: PDF vector figures and 300-dpi PNGs.\n"
        "- `data/missing_items.json`: explicit non-availability boundaries.\n\n"
        "The complete raw diagnostic NPZ chunks remain at the remote paths recorded in the copied diagnostics manifests; they were not downloaded locally.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "completed", "output": str(output), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
