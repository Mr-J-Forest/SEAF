#!/usr/bin/env python3
"""Generate DynaSEAF paper tables from completed, inspectable run artifacts.

This script never fills missing scientific values with estimates.  Missing
artifacts or metrics are rendered as ``TODO_FROM_FROZEN_RESULTS`` so that a
paper table cannot accidentally turn a development result into a claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable


TODO = "TODO_FROM_FROZEN_RESULTS"
MAIN_LABELS = ("SEAF-v1", "DynaSEAF", "FourCastNet", "ClimaX", "Swin")
ABLATION_LABELS = (
    "A0-SEAF-v1",
    "A1-dynamics-only",
    "A2-transport",
    "A3-transport-innovation",
    "A4-DynaSEAF",
    "no-dynamics-aux",
    "no-transport",
    "no-innovation",
    "no-gate",
)
STAGE_PRIORITY = {
    "smoke": 0,
    "screen": 1,
    "confirm_validation": 2,
    "final_test": 3,
}


def nested(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def metric_or_todo(value: Any) -> Any:
    return float(value) if finite(value) else TODO


def mean_and_std(values: Iterable[Any]) -> tuple[Any, Any]:
    numeric = [float(value) for value in values if finite(value)]
    if not numeric:
        return TODO, TODO
    return fmean(numeric), stdev(numeric) if len(numeric) > 1 else TODO


def model_label(config: dict, experiment: str) -> str | None:
    model_type = str(config.get("model_type", "")).lower()
    if model_type == "seaf":
        return "SEAF-v1"
    if model_type == "dynaseaf":
        return "DynaSEAF"
    baseline_labels = {
        "ofb_fourcastnet": "FourCastNet",
        "ofb-fourcastnet": "FourCastNet",
        "ofb_climax": "ClimaX",
        "ofb-climax": "ClimaX",
        "ofb_swin": "Swin",
        "ofb-swin": "Swin",
    }
    return baseline_labels.get(model_type)


def ablation_label(record: dict) -> str | None:
    if record["label"] == "SEAF-v1":
        return "A0-SEAF-v1"
    if record["label"] != "DynaSEAF":
        return None
    config = record["config"]
    name = record["experiment"].lower()
    if "dynamics_only" in name:
        return "A1-dynamics-only"
    if "transport_innovation" in name:
        return "A3-transport-innovation"
    if name.endswith("_transport") or name.endswith("-transport"):
        return "A2-transport"
    if name.endswith("_no_dynamics_aux") or "no-dynamics-aux" in name:
        return "no-dynamics-aux"
    if name.endswith("_no_transport") or "no-transport" in name:
        return "no-transport"
    if name.endswith("_no_innovation") or "no-innovation" in name:
        return "no-innovation"
    if name.endswith("_no_gate") or "no-gate" in name:
        return "no-gate"
    if bool(config.get("dynaseaf_use_adaptive_gate", True)):
        return "A4-DynaSEAF"
    return "DynaSEAF"


def _evaluation_path(run_dir: Path, summary: dict) -> Path | None:
    requested = summary.get("evaluation_file")
    candidates = [run_dir / requested] if requested else []
    candidates.extend((run_dir / name) for name in (
        "validation_results.json",
        "evaluation_results.json",
    ))
    return next((path for path in candidates if path and path.is_file()), None)


def load_records(results_root: Path, campaign: str | None, split: str) -> list[dict]:
    records = []
    for summary_path in sorted(results_root.rglob("run_summary.json")):
        run_dir = summary_path.parent
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("status") != "completed":
            continue
        evaluation_path = _evaluation_path(run_dir, summary)
        if evaluation_path is None:
            continue
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        evaluation_split = str(
            evaluation.get("evaluation_split", summary.get("evaluation_scope", ""))
        )
        if split and evaluation_split and evaluation_split != split:
            continue
        path_parts = run_dir.parts
        stage = run_dir.parent.parent.name if len(path_parts) >= 3 else "unknown"
        campaign_name = run_dir.parent.parent.parent.name if len(path_parts) >= 4 else None
        if campaign and campaign_name and campaign_name != campaign:
            continue
        experiment = run_dir.parent.name
        label = model_label(config, experiment)
        if label is None:
            continue
        physical = evaluation.get("physical_report") or {}
        variable_report = physical.get("by_variable", {})
        if not variable_report:
            variable_report = evaluation.get("normalized_report", {}).get(
                "by_variable", {}
            )
        records.append({
            "stage": stage,
            "campaign": campaign_name,
            "experiment": experiment,
            "seed": config.get("seed"),
            "label": label,
            "config": config,
            "summary": summary,
            "evaluation": evaluation,
            "variables": variable_report,
            "run_dir": str(run_dir.resolve()),
        })
    # A campaign can contain screen and confirmation runs with the same seed.
    # Prefer the explicitly later frozen stage without averaging duplicate runs.
    selected = {}
    for record in records:
        key = (record["experiment"], record["seed"], record["label"])
        previous = selected.get(key)
        if previous is None or STAGE_PRIORITY.get(record["stage"], -1) >= STAGE_PRIORITY.get(
            previous["stage"], -1
        ):
            selected[key] = record
    return sorted(selected.values(), key=lambda item: (item["label"], item["experiment"], str(item["seed"])))


def metric(record: dict, variable: str, field: str) -> Any:
    value = nested(record["variables"], variable, field)
    if value is None:
        value = nested(record["evaluation"], f"physical_{field}_{variable}")
    if value is None:
        value = nested(record["evaluation"], f"normalized_{field}_{variable}")
    return value


def comparison_metric(record: dict, baseline: str, *keys: str) -> Any:
    """Read a baseline skill without assuming physical units are available."""
    evaluation = record["evaluation"]
    for space in ("physical", "normalized"):
        value = nested(
            evaluation,
            "baseline_comparison",
            space,
            baseline,
            *keys,
        )
        if finite(value):
            return value
    return None


def aggregate_rows(records: list[dict], group_key: str, labels: Iterable[str]) -> list[dict]:
    rows = []
    for label in labels:
        if group_key == "ablation":
            members = [record for record in records if ablation_label(record) == label]
        else:
            members = [record for record in records if record.get(group_key) == label]
        temp_values = [metric(record, "TEMP", "rmse") for record in members]
        salt_values = [metric(record, "SALT", "rmse") for record in members]
        temp_mean, temp_std = mean_and_std(temp_values)
        salt_mean, salt_std = mean_and_std(salt_values)
        rows.append({
            "label": label,
            "n_seeds": len({record.get("seed") for record in members}),
            "TEMP_RMSE_mean": temp_mean,
            "TEMP_RMSE_sample_std": temp_std,
            "SALT_RMSE_mean": salt_mean,
            "SALT_RMSE_sample_std": salt_std,
            "parameter_count": (
                members[0]["summary"].get("parameter_count") if members else TODO
            ),
        })
    return rows


def lead_rows(records: list[dict]) -> list[dict]:
    values: dict[tuple[str, str, str], dict[str, list[Any]]] = {}
    for record in records:
        label = record["label"]
        report = nested(record["evaluation"], "physical_report", "by_variable_and_lead") or {}
        if not report:
            report = nested(record["evaluation"], "normalized_report", "by_variable_and_lead") or {}
        for variable, leads in report.items():
            for lead, fields in leads.items():
                bucket = values.setdefault(
                    (label, str(variable), str(lead)),
                    {"rmse": [], "ss_ap": [], "ss_dap": []},
                )
                bucket["rmse"].append(nested(fields, "rmse"))
                bucket["ss_ap"].append(
                    comparison_metric(
                        record,
                        "anomaly_persistence",
                        "by_variable_and_lead",
                        variable,
                        lead,
                        "mse_skill",
                    )
                )
                bucket["ss_dap"].append(
                    comparison_metric(
                        record,
                        "damped_anomaly_persistence",
                        "by_variable_and_lead",
                        variable,
                        lead,
                        "mse_skill",
                    )
                )
    rows = []
    for (label, variable, lead), metrics in sorted(values.items()):
        avg, spread = mean_and_std(metrics["rmse"])
        ap_avg, ap_spread = mean_and_std(metrics["ss_ap"])
        dap_avg, dap_spread = mean_and_std(metrics["ss_dap"])
        rows.append({
            "label": label,
            "variable": variable,
            "lead": lead,
            "RMSE_mean": avg,
            "RMSE_sample_std": spread,
            "SS_AP_mean": ap_avg,
            "SS_AP_sample_std": ap_spread,
            "SS_DAP_mean": dap_avg,
            "SS_DAP_sample_std": dap_spread,
        })
    return rows or [{"status": TODO}]


def depth_rows(records: list[dict]) -> list[dict]:
    values: dict[tuple[str, str, str], dict[str, list[Any]]] = {}
    for record in records:
        report = nested(record["evaluation"], "physical_report", "by_variable_and_depth") or {}
        if not report:
            report = nested(record["evaluation"], "normalized_report", "by_variable_and_depth") or {}
        for variable, depths in report.items():
            for depth_key, fields in depths.items():
                depth = fields.get("depth", depth_key) if isinstance(fields, dict) else depth_key
                bucket = values.setdefault(
                    (record["label"], str(variable), str(depth)),
                    {"rmse": [], "ss_ap": [], "ss_dap": []},
                )
                bucket["rmse"].append(nested(fields, "rmse"))
                bucket["ss_ap"].append(
                    comparison_metric(
                        record,
                        "anomaly_persistence",
                        "by_variable_and_depth",
                        variable,
                        depth_key,
                        "mse_skill",
                    )
                )
                bucket["ss_dap"].append(
                    comparison_metric(
                        record,
                        "damped_anomaly_persistence",
                        "by_variable_and_depth",
                        variable,
                        depth_key,
                        "mse_skill",
                    )
                )
    rows = []
    for (label, variable, depth), metrics in sorted(values.items()):
        avg, spread = mean_and_std(metrics["rmse"])
        ap_avg, ap_spread = mean_and_std(metrics["ss_ap"])
        dap_avg, dap_spread = mean_and_std(metrics["ss_dap"])
        rows.append({
            "label": label,
            "variable": variable,
            "depth": depth,
            "RMSE_mean": avg,
            "RMSE_sample_std": spread,
            "SS_AP_mean": ap_avg,
            "SS_AP_sample_std": ap_spread,
            "SS_DAP_mean": dap_avg,
            "SS_DAP_sample_std": dap_spread,
        })
    return rows or [{"status": TODO}]


def lead_depth_rows(records: list[dict]) -> list[dict]:
    values: dict[tuple[str, str, str, str], dict[str, list[Any]]] = {}
    for record in records:
        report = nested(
            record["evaluation"],
            "physical_report",
            "by_variable_lead_and_depth",
        ) or {}
        if not report:
            report = nested(
                record["evaluation"],
                "normalized_report",
                "by_variable_lead_and_depth",
            ) or {}
        for variable, leads in report.items():
            for lead, depths in leads.items():
                for depth_key, fields in depths.items():
                    depth = (
                        fields.get("depth", depth_key)
                        if isinstance(fields, dict) else depth_key
                    )
                    bucket = values.setdefault(
                        (record["label"], str(variable), str(lead), str(depth)),
                        {"rmse": [], "ss_ap": [], "ss_dap": []},
                    )
                    bucket["rmse"].append(nested(fields, "rmse"))
                    bucket["ss_ap"].append(
                        comparison_metric(
                            record,
                            "anomaly_persistence",
                            "by_variable_lead_and_depth",
                            variable,
                            lead,
                            depth_key,
                            "mse_skill",
                        )
                    )
                    bucket["ss_dap"].append(
                        comparison_metric(
                            record,
                            "damped_anomaly_persistence",
                            "by_variable_lead_and_depth",
                            variable,
                            lead,
                            depth_key,
                            "mse_skill",
                        )
                    )
    rows = []
    for (label, variable, lead, depth), metrics in sorted(values.items()):
        avg, spread = mean_and_std(metrics["rmse"])
        ap_avg, ap_spread = mean_and_std(metrics["ss_ap"])
        dap_avg, dap_spread = mean_and_std(metrics["ss_dap"])
        rows.append({
            "label": label,
            "variable": variable,
            "lead": lead,
            "depth": depth,
            "RMSE_mean": avg,
            "RMSE_sample_std": spread,
            "SS_AP_mean": ap_avg,
            "SS_AP_sample_std": ap_spread,
            "SS_DAP_mean": dap_avg,
            "SS_DAP_sample_std": dap_spread,
        })
    return rows or [{"status": TODO}]


def per_seed_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        summary = record["summary"]
        rows.append({
            "label": record["label"],
            "experiment": record["experiment"],
            "stage": record["stage"],
            "seed": record.get("seed", TODO),
            "status": summary.get("status", TODO),
            "parameter_count": metric_or_todo(summary.get("parameter_count")),
            "best_epoch": metric_or_todo(summary.get("best_epoch")),
            "best_val_loss": metric_or_todo(summary.get("best_val_loss")),
            "TEMP_RMSE": metric_or_todo(metric(record, "TEMP", "rmse")),
            "SALT_RMSE": metric_or_todo(metric(record, "SALT", "rmse")),
            "run_dir": record["run_dir"],
        })
    return rows or [{"status": TODO}]


def _first_diagnostic_value(record: dict, paths: Iterable[tuple[str, ...]]) -> Any:
    for source in (record.get("evaluation", {}), record.get("summary", {})):
        for path in paths:
            value = nested(source, *path)
            if finite(value):
                return float(value)
    return TODO


def dynamics_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        row = {
            "label": record["label"],
            "experiment": record["experiment"],
            "seed": record.get("seed", TODO),
            "future_dynamics_RMSE": _first_diagnostic_value(
                record,
                (
                    ("dynamics_metrics", "future_dynamics_rmse"),
                    ("dynaseaf_diagnostics", "future_dynamics_rmse"),
                    ("future_dynamics_rmse",),
                ),
            ),
            "auxiliary_loss_last": TODO,
            "run_dir": record["run_dir"],
        }
        history = record["summary"].get("dynaseaf_future_dynamics_losses")
        if isinstance(history, list) and history:
            row["auxiliary_loss_last"] = metric_or_todo(history[-1])
        for variable in ("UVEL", "VVEL", "SSHA", "MLD"):
            row[f"{variable}_RMSE"] = _first_diagnostic_value(
                record,
                (
                    ("dynamics_metrics", "by_variable", variable, "rmse"),
                    ("dynaseaf_diagnostics", "future_dynamics_rmse", variable),
                    ("future_dynamics_rmse", variable),
                ),
            )
        rows.append(row)
    return rows or [{"status": TODO}]


def gate_statistics_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        rows.append({
            "label": record["label"],
            "experiment": record["experiment"],
            "seed": record.get("seed", TODO),
            "mean_gate": _first_diagnostic_value(
                record,
                (
                    ("gate_statistics", "mean_gate"),
                    ("dynaseaf_diagnostics", "mean_gate"),
                    ("mean_gate",),
                ),
            ),
            "mean_gate_by_lead": TODO,
            "mean_gate_by_depth": TODO,
            "run_dir": record["run_dir"],
        })
    return rows or [{"status": TODO}]


def deformation_statistics_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        rows.append({
            "label": record["label"],
            "experiment": record["experiment"],
            "seed": record.get("seed", TODO),
            "mean_displacement_magnitude": _first_diagnostic_value(
                record,
                (
                    ("deformation_statistics", "mean_displacement_magnitude"),
                    ("dynaseaf_diagnostics", "mean_displacement_magnitude"),
                    ("mean_displacement_magnitude",),
                ),
            ),
            "max_displacement_magnitude": _first_diagnostic_value(
                record,
                (
                    ("deformation_statistics", "max_displacement_magnitude"),
                    ("dynaseaf_diagnostics", "max_displacement_magnitude"),
                    ("max_displacement_magnitude",),
                ),
            ),
            "run_dir": record["run_dir"],
        })
    return rows or [{"status": TODO}]


def paired_bootstrap_payload() -> dict[str, Any]:
    return {
        "status": TODO,
        "protocol": {
            "unit": "forecast_origin",
            "replicates": 10000,
            "block_length": 5,
            "confidence_level": 0.95,
            "multiple_comparison": "Benjamini-Hochberg",
        },
        "comparisons": [],
        "note": (
            "Paired frozen validation predictions are required before computing "
            "confidence intervals, p-values, or q-values."
        ),
    }


def parameter_count_payload(records: list[dict]) -> dict[str, Any]:
    rows = []
    for record in records:
        rows.append({
            "label": record["label"],
            "experiment": record["experiment"],
            "seed": record.get("seed", TODO),
            "parameter_count": metric_or_todo(
                record["summary"].get("parameter_count")
            ),
            "run_dir": record["run_dir"],
        })
    return {
        "status": "complete" if rows else TODO,
        "records": rows or [{"status": TODO}],
        "note": "Counts are read from completed run summaries; no estimate is inserted.",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        rows = [{"status": TODO}]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    if not rows:
        rows = [{"status": TODO}]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, TODO)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="outputs/results/campaigns")
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_root = (project_root / args.results).resolve()
    output_name = args.campaign or "all"
    output_dir = (
        (project_root / args.output).resolve()
        if args.output
        else (project_root / "results" / "dynaseaf" / output_name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(results_root, args.campaign, args.split)

    main_rows = aggregate_rows(records, "label", MAIN_LABELS)
    ablation_records = [
        record for record in records if ablation_label(record) is not None
    ]
    ablation_rows = []
    for label in ABLATION_LABELS:
        members = [record for record in ablation_records if ablation_label(record) == label]
        temp_mean, temp_std = mean_and_std(metric(record, "TEMP", "rmse") for record in members)
        salt_mean, salt_std = mean_and_std(metric(record, "SALT", "rmse") for record in members)
        ablation_rows.append({
            "label": label,
            "n_seeds": len({record.get("seed") for record in members}),
            "TEMP_RMSE_mean": temp_mean,
            "TEMP_RMSE_sample_std": temp_std,
            "SALT_RMSE_mean": salt_mean,
            "SALT_RMSE_sample_std": salt_std,
            "parameter_count": (
                members[0]["summary"].get("parameter_count") if members else TODO
            ),
        })

    statistical_rows = [{
        "comparison": TODO,
        "statistic": "log(MSE_reference/MSE_candidate)",
        "bootstrap_replicates": 10000,
        "block_length": 5,
        "geometric_reduction": TODO,
        "CI_95": TODO,
        "p_value": TODO,
        "BH_adjusted_q": TODO,
    }]

    table_sets = {
        "main_comparison": main_rows,
        "dynaseaf_ablation": ablation_rows,
        "lead_wise": lead_rows(records),
        "depth_wise": depth_rows(records),
        "statistical_comparison": statistical_rows,
    }
    for name, rows in table_sets.items():
        write_csv(output_dir / f"{name}.csv", rows)
        write_markdown(output_dir / f"{name}.md", rows)

    contract_sets = {
        "per_seed": per_seed_rows(records),
        "lead_skill": lead_rows(records),
        "depth_skill": depth_rows(records),
        "lead_depth_skill": lead_depth_rows(records),
        "dynamics_metrics": dynamics_rows(records),
        "gate_statistics": gate_statistics_rows(records),
        "deformation_statistics": deformation_statistics_rows(records),
    }
    for name, rows in contract_sets.items():
        write_csv(output_dir / f"{name}.csv", rows)

    summary = {
        "status": "complete" if records else TODO,
        "results_root": str(results_root),
        "campaign": args.campaign,
        "evaluation_split": args.split,
        "record_count": len(records),
        "main_labels": list(MAIN_LABELS),
        "tables": {name: len(rows) for name, rows in table_sets.items()},
        "machine_readable_contract": {
            "csv_files": {
                f"{name}.csv": len(rows) for name, rows in contract_sets.items()
            },
            "json_files": ["summary.json", "paired_bootstrap.json", "parameter_count.json"],
        },
        "missing_scientific_results": not bool(records),
        "source_run_dirs": [record["run_dir"] for record in records],
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "paired_bootstrap.json", paired_bootstrap_payload())
    write_json(output_dir / "parameter_count.json", parameter_count_payload(records))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
