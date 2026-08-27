#!/usr/bin/env python3
"""Aggregate deterministic experiment runs without manually copying metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


NUMERIC_FIELDS = (
    "temp_rmse",
    "salt_rmse",
    "temp_skill_ap",
    "salt_skill_ap",
    "macro_skill_ap",
    "best_val_loss",
    "best_epoch",
    "mean_epoch_time_seconds",
    "wall_time_seconds",
    "parameter_count",
    "peak_cuda_memory_gib",
)


COMMON_PROTOCOL_KEYS = (
    "data_path",
    "input_variables",
    "sequence_length",
    "prediction_length",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "split_context_policy",
    "lon_range",
    "lat_range",
    "depth_range",
    "train_stride_lon",
    "train_stride_lat",
    "val_stride_lon",
    "val_stride_lat",
    "test_stride_lon",
    "test_stride_lat",
    "ocean_threshold",
    "expected_canonical_windows_per_origin",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "grad_clip_norm",
    "scheduler_patience",
    "scheduler_factor",
    "min_lr",
    "early_stopping_patience",
    "min_delta",
    "mixed_precision",
    "mixed_precision_dtype",
    "nonfinite_grad_skip_limit",
    "compile_model",
    "group_batches_by_time",
    "gradient_loss_mode",
    "post_training_evaluation",
)

DATA_PROTOCOL_KEYS = (
    "data_identity",
    "split_context_policy",
    "train",
    "validation",
    "test",
    "loader_batches",
)


def nested(payload, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_run(run_dir: Path) -> tuple[dict | None, str | None]:
    required = ["_SUCCESS", "config.json", "run_summary.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        return None, f"missing: {', '.join(missing)}"
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        evaluation_file = summary.get("evaluation_file")
        evaluation = (
            json.loads((run_dir / evaluation_file).read_text(encoding="utf-8"))
            if evaluation_file else {}
        )
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON: {exc}"
    if summary.get("status") != "completed":
        return None, f"summary status is {summary.get('status')!r}"

    experiment = run_dir.parent.name
    comparison = nested(evaluation, "baseline_comparison", "physical", "anomaly_persistence") or {}
    peak_bytes = summary.get("peak_cuda_memory_bytes")
    row = {
        "stage": run_dir.parent.parent.name,
        "experiment": experiment,
        "seed": config.get("seed"),
        "run_dir": str(run_dir.resolve()),
        "model_type": config.get("model_type"),
        "baseline_kind": nested(config, "baseline_provenance", "kind"),
        "baseline_method_name": nested(config, "baseline_provenance", "method_name"),
        "baseline_official_code": nested(config, "baseline_provenance", "official_code"),
        "completed_epochs": summary.get("completed_epochs"),
        "max_epochs": config.get("epochs"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "early_stopping_patience": config.get("early_stopping_patience"),
        "temp_rmse": nested(evaluation, "physical_report", "by_variable", "TEMP", "rmse"),
        "salt_rmse": nested(evaluation, "physical_report", "by_variable", "SALT", "rmse"),
        "temp_skill_ap": nested(comparison, "by_variable", "TEMP", "mse_skill"),
        "salt_skill_ap": nested(comparison, "by_variable", "SALT", "mse_skill"),
        "macro_skill_ap": nested(comparison, "macro", "mse_skill", "mean"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_epoch": summary.get("best_epoch"),
        "mean_epoch_time_seconds": summary.get("mean_epoch_time_seconds"),
        "wall_time_seconds": summary.get("wall_time_seconds"),
        "parameter_count": summary.get("parameter_count"),
        "peak_cuda_memory_gib": (
            float(peak_bytes) / (1024 ** 3) if finite_number(peak_bytes) else None
        ),
        "git_commit": summary.get("git_commit"),
        "git_dirty": summary.get("git_dirty"),
        "source_hash": summary.get("source_hash"),
        "training_source_hash": summary.get("training_source_hash"),
        "training_config_fingerprint": summary.get("training_config_fingerprint"),
        "compile_requested": summary.get("compile_requested"),
        "compile_active": summary.get("compile_active"),
        "compile_fallback_used": summary.get("compile_fallback_used"),
        "evaluation_scope": summary.get("evaluation_scope"),
        "origin_block_count": nested(evaluation, "stratified_reports", "by_origin", "group_count"),
        "has_depth_metrics": bool(nested(
            evaluation, "physical_report", "by_variable_and_depth"
        )),
        "has_evaluation_provenance": bool(evaluation.get("evaluation_provenance")),
        "_config": config,
        "_data_protocol": summary.get("data_protocol"),
    }
    return row, None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def audit_protocol(rows: list[dict], allowed_differences: set[str], strict: bool) -> dict:
    source_hashes = sorted({
        row.get("training_source_hash") for row in rows if row.get("training_source_hash")
    })
    missing_source_hash = [
        row["run_dir"] for row in rows if not row.get("training_source_hash")
    ]
    protocol_mismatches = []
    for key in COMMON_PROTOCOL_KEYS:
        if key in allowed_differences:
            continue
        values: dict[str, list[str]] = {}
        for row in rows:
            encoded = canonical(row.get("_config", {}).get(key))
            values.setdefault(encoded, []).append(f"{row['experiment']}/seed_{row['seed']}")
        if len(values) > 1:
            protocol_mismatches.append({"key": key, "values": values})
    for key in DATA_PROTOCOL_KEYS:
        qualified_key = f"data_protocol.{key}"
        if qualified_key in allowed_differences or "data_protocol" in allowed_differences:
            continue
        values: dict[str, list[str]] = {}
        for row in rows:
            encoded = canonical((row.get("_data_protocol") or {}).get(key))
            values.setdefault(encoded, []).append(f"{row['experiment']}/seed_{row['seed']}")
        if len(values) > 1:
            protocol_mismatches.append({"key": qualified_key, "values": values})

    missing_evidence = []
    if strict:
        for row in rows:
            absent = []
            if not row.get("has_depth_metrics"):
                absent.append("depth metrics")
            if not row.get("origin_block_count"):
                absent.append("origin blocks")
            if not row.get("has_evaluation_provenance"):
                absent.append("evaluation provenance")
            if absent:
                missing_evidence.append({"run_dir": row["run_dir"], "missing": absent})

    return {
        "source_hashes": source_hashes,
        "missing_source_hash": missing_source_hash,
        "dirty_runs": [row["run_dir"] for row in rows if row.get("git_dirty")],
        "protocol_keys_checked": [
            key for key in COMMON_PROTOCOL_KEYS if key not in allowed_differences
        ] + [
            f"data_protocol.{key}" for key in DATA_PROTOCOL_KEYS
            if f"data_protocol.{key}" not in allowed_differences
            and "data_protocol" not in allowed_differences
        ],
        "allowed_protocol_differences": sorted(allowed_differences),
        "protocol_mismatches": protocol_mismatches,
        "missing_evidence": missing_evidence,
        "strict_pass": (
            len(source_hashes) <= 1
            and (not strict or not missing_source_hash)
            and not protocol_mismatches
            and not missing_evidence
        ),
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["experiment"], []).append(row)
    output = []
    for experiment, members in sorted(groups.items()):
        item = {"experiment": experiment, "n_seeds": len(members)}
        for field in NUMERIC_FIELDS:
            values = [float(row[field]) for row in members if finite_number(row.get(field))]
            item[f"{field}_mean"] = mean(values) if values else None
            item[f"{field}_std"] = stdev(values) if len(values) > 1 else None
        output.append(item)
    return output


def markdown_table(summary_rows: list[dict]) -> str:
    headers = ["Experiment", "N", "TEMP RMSE", "SALT RMSE", "TEMP AP skill", "SALT AP skill", "Macro AP skill", "Params"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]

    def show(row, field, digits=4):
        avg = row.get(f"{field}_mean")
        std = row.get(f"{field}_std")
        if not finite_number(avg):
            return ""
        if finite_number(std):
            return f"{avg:.{digits}f} ± {std:.{digits}f}"
        return f"{avg:.{digits}f}"

    for row in summary_rows:
        values = [
            row["experiment"],
            str(row["n_seeds"]),
            show(row, "temp_rmse"),
            show(row, "salt_rmse"),
            show(row, "temp_skill_ap"),
            show(row, "salt_skill_ap"),
            show(row, "macro_skill_ap"),
            show(row, "parameter_count", digits=0),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="outputs/results/matrix")
    parser.add_argument("--output", default="outputs/aggregate")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--allow-protocol-difference",
        action="append",
        default=[],
        help="common protocol key intentionally varied (repeatable)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_root = (root / args.results).resolve()
    output_dir = (root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted({path.parent for path in results_root.rglob("run_summary.json")})
    rows = []
    audit = {"results_root": str(results_root), "valid_runs": 0, "invalid_runs": []}
    for run_dir in run_dirs:
        row, error = load_run(run_dir)
        if error:
            audit["invalid_runs"].append({"run_dir": str(run_dir), "error": error})
        else:
            rows.append(row)
    audit["valid_runs"] = len(rows)
    audit["protocol"] = audit_protocol(
        rows,
        set(args.allow_protocol_difference),
        strict=args.strict,
    )

    summary_rows = summarize(rows)
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.md").write_text(markdown_table(summary_rows), encoding="utf-8")
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"valid runs: {len(rows)}; invalid runs: {len(audit['invalid_runs'])}")
    print(output_dir / "summary.md")
    failed = bool(audit["invalid_runs"])
    if args.strict and not audit["protocol"]["strict_pass"]:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
