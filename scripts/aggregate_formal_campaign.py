#!/usr/bin/env python3
"""Aggregate frozen SEAF campaign evaluations with an explicit quality audit.

The formal 36-run matrix is split across a reference campaign (the retained
full/direct-full-field checkpoints) and a continuation campaign (the remaining
ablations and learned baselines).  This script joins those immutable run
directories without copying checkpoints, validates the evaluation evidence,
and writes headline, lead-wise, and depth-wise summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


EXPECTED_EXPERIMENTS = {
    "full",
    "direct_full_field_strict",
    "direct_full_field_tuned",
    "no_tendency",
    "no_external_dynamics",
    "no_spectral",
    "uniform_ensemble",
    "single_head",
    "local_cnn",
    "ofb_fourcastnet_anomaly",
    "ofb_climax_anomaly",
    "ofb_swin_anomaly",
}
EXPECTED_SEEDS = {42, 123, 3407}
VARIABLES = ("TEMP", "SALT")


def _read_json(path: Path) -> dict[str, Any]:
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


def _nested(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _number(value: Any) -> float | None:
    if isinstance(value, dict) and "mean" in value:
        value = value["mean"]
    return float(value) if _finite(value) else None


def _evaluation_path(run_dir: Path, split: str) -> Path | None:
    names = ("validation_results.json", "evaluation_results.json")
    if split == "test":
        names = ("test_results.json", "evaluation_results.json")
    for name in names:
        path = run_dir / name
        if path.is_file():
            return path
    return None


def _run_dirs(campaign_root: Path, stage: str) -> list[Path]:
    stage_root = campaign_root / stage
    if not stage_root.is_dir():
        return []
    return sorted({path.parent for path in stage_root.rglob("run_summary.json")})


def _row_from_run(run_dir: Path, campaign: str, split: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    summary_path = run_dir / "run_summary.json"
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pth"
    missing = [str(path) for path in (summary_path, config_path, checkpoint_path) if not path.is_file()]
    if missing:
        return None, {"run_dir": str(run_dir), "campaign": campaign, "reason": "missing_artifact", "paths": missing}
    try:
        summary = _read_json(summary_path)
        config = _read_json(config_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, {"run_dir": str(run_dir), "campaign": campaign, "reason": f"invalid_metadata: {exc}"}
    if summary.get("status") != "completed" or int(summary.get("completed_epochs", 0) or 0) < 30:
        return None, {
            "run_dir": str(run_dir),
            "campaign": campaign,
            "reason": "training_not_completed",
            "status": summary.get("status"),
            "completed_epochs": summary.get("completed_epochs"),
        }
    evaluation_path = _evaluation_path(run_dir, split)
    if evaluation_path is None:
        return None, {"run_dir": str(run_dir), "campaign": campaign, "reason": f"missing_{split}_evaluation"}
    try:
        evaluation = _read_json(evaluation_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, {"run_dir": str(run_dir), "campaign": campaign, "reason": f"invalid_evaluation: {exc}"}

    experiment = run_dir.parent.name
    seed = config.get("seed")
    run_id = f"confirm_validation__{experiment}_seed{seed}"
    ap = _nested(evaluation, "baseline_comparison", "physical", "anomaly_persistence") or {}
    dap = _nested(evaluation, "baseline_comparison", "physical", "damped_anomaly_persistence") or {}
    physical = _nested(evaluation, "physical_report") or {}
    by_variable = physical.get("by_variable") or {}
    origin_groups = _nested(evaluation, "stratified_reports", "by_origin", "groups") or {}
    provenance = evaluation.get("evaluation_provenance") or {}
    quality = {
        "has_physical_report": bool(physical),
        "has_anomaly_persistence_comparison": bool(ap),
        "has_damped_anomaly_persistence_comparison": bool(dap),
        "has_depth_metrics": bool(physical.get("by_variable_and_depth")),
        "has_lead_metrics": bool(physical.get("by_variable_and_lead")),
        "has_origin_metrics": bool(origin_groups),
        "has_evaluation_provenance": bool(provenance),
        "origin_count": len(origin_groups),
        "sample_count": len(provenance.get("samples", [])) if isinstance(provenance, dict) else 0,
    }
    headline: dict[str, Any] = {
        "campaign": campaign,
        "run_id": run_id,
        "stage": run_dir.parent.parent.name,
        "experiment": experiment,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "evaluation_path": str(evaluation_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config_sha256": _sha256(config_path),
        "evaluation_sha256": _sha256(evaluation_path),
        "training_source_hash": summary.get("training_source_hash"),
        "training_config_fingerprint": summary.get("training_config_fingerprint"),
        "git_commit": summary.get("git_commit"),
        "git_dirty": summary.get("git_dirty"),
        "completed_epochs": summary.get("completed_epochs"),
        "best_epoch": summary.get("best_epoch"),
        "best_val_loss": summary.get("best_val_loss"),
        "parameter_count": summary.get("parameter_count"),
        "mean_epoch_time_seconds": summary.get("mean_epoch_time_seconds"),
        "wall_time_seconds": summary.get("wall_time_seconds"),
        "peak_cuda_memory_gib": (
            float(summary["peak_cuda_memory_bytes"]) / (1024 ** 3)
            if _finite(summary.get("peak_cuda_memory_bytes")) else None
        ),
        "temp_rmse": _number(_nested(physical, "by_variable", "TEMP", "rmse")),
        "salt_rmse": _number(_nested(physical, "by_variable", "SALT", "rmse")),
        "temp_ap_skill": _number(_nested(ap, "by_variable", "TEMP", "mse_skill")),
        "salt_ap_skill": _number(_nested(ap, "by_variable", "SALT", "mse_skill")),
        "macro_ap_skill": _number(_nested(ap, "macro", "mse_skill", "mean")),
        "temp_dap_skill": _number(_nested(dap, "by_variable", "TEMP", "mse_skill")),
        "salt_dap_skill": _number(_nested(dap, "by_variable", "SALT", "mse_skill")),
        "macro_dap_skill": _number(_nested(dap, "macro", "mse_skill", "mean")),
        "quality": quality,
    }
    lead_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    for variable in VARIABLES:
        lead_metrics = _nested(physical, "by_variable_and_lead", variable) or {}
        lead_ap = _nested(ap, "by_variable_and_lead", variable) or {}
        lead_dap = _nested(dap, "by_variable_and_lead", variable) or {}
        for lead_name, metrics in lead_metrics.items():
            lead_rows.append({
                "campaign": campaign,
                "run_id": run_id,
                "experiment": experiment,
                "seed": seed,
                "variable": variable,
                "lead": int(str(lead_name).rsplit("_", 1)[-1]),
                "rmse": _number(_nested(metrics, "rmse")),
                "mse": _number(_nested(metrics, "mse")),
                "ap_skill": _number(_nested(lead_ap, lead_name, "mse_skill")),
                "dap_skill": _number(_nested(lead_dap, lead_name, "mse_skill")),
            })
        depth_metrics = _nested(physical, "by_variable_and_depth", variable) or {}
        depth_ap = _nested(ap, "by_variable_and_depth", variable) or {}
        depth_dap = _nested(dap, "by_variable_and_depth", variable) or {}
        for depth_name, metrics in depth_metrics.items():
            depth_rows.append({
                "campaign": campaign,
                "run_id": run_id,
                "experiment": experiment,
                "seed": seed,
                "variable": variable,
                "depth_name": depth_name,
                "depth": _number(_nested(metrics, "depth")),
                "rmse": _number(_nested(metrics, "rmse")),
                "mse": _number(_nested(metrics, "mse")),
                "ap_skill": _number(_nested(depth_ap, depth_name, "mse_skill")),
                "dap_skill": _number(_nested(depth_dap, depth_name, "mse_skill")),
            })
    headline["_evaluation"] = evaluation
    headline["_lead_rows"] = lead_rows
    headline["_depth_rows"] = depth_rows
    return headline, None


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [key for key in rows[0] if not key.startswith("_")]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]], group_keys: tuple[str, ...], value_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in group_keys), []).append(row)
    output = []
    for group, members in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        item = dict(zip(group_keys, group))
        item["n"] = len(members)
        for key in value_keys:
            values = [float(row[key]) for row in members if _finite(row.get(key))]
            item[f"{key}_mean"] = mean(values) if values else None
            item[f"{key}_std"] = stdev(values) if len(values) > 1 else None
        output.append(item)
    return output


def _markdown(summary: list[dict[str, Any]], audit: dict[str, Any], split: str) -> str:
    lines = [
        f"# SEAF formal {split} aggregation",
        "",
        f"- Valid runs: **{audit['valid_runs']} / 36**",
        f"- Strict quality pass: **{audit['strict_pass']}**",
        f"- Generated (UTC): `{audit['generated_at_utc']}`",
        "",
        "| Experiment | N | TEMP RMSE | SALT RMSE | TEMP SS_AP | SALT SS_AP | Macro SS_AP | Macro SS_DAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def show(key: str) -> str:
            value = row.get(f"{key}_mean")
            spread = row.get(f"{key}_std")
            if not _finite(value):
                return ""
            if _finite(spread):
                return f"{float(value):.4f} +/- {float(spread):.4f}"
            return f"{float(value):.4f}"
        lines.append(
            "| " + " | ".join([
                str(row.get("experiment", "")), str(row.get("n", "")),
                show("temp_rmse"), show("salt_rmse"), show("temp_ap_skill"),
                show("salt_ap_skill"), show("macro_ap_skill"), show("macro_dap_skill"),
            ]) + " |"
        )
    if audit["invalid_runs"]:
        lines.extend(["", "## Invalid or missing runs", ""])
        for item in audit["invalid_runs"]:
            lines.append(f"- `{item.get('run_dir')}`: {item.get('reason')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--current-root", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--stage", default="confirm_validation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roots = [
        (Path(args.reference_root).resolve(), "reference"),
        (Path(args.current_root).resolve(), "current"),
    ]
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for root, campaign in roots:
        for run_dir in _run_dirs(root, args.stage):
            row, error = _row_from_run(run_dir, campaign, args.split)
            if row is not None:
                rows.append(row)
            if error is not None:
                invalid.append(error)

    rows.sort(key=lambda row: (str(row.get("experiment")), int(row.get("seed") or -1), str(row.get("campaign"))))
    experiment_seed_pairs = {(row.get("experiment"), int(row.get("seed"))) for row in rows if row.get("seed") is not None}
    expected_pairs = {(experiment, seed) for experiment in EXPECTED_EXPERIMENTS for seed in EXPECTED_SEEDS}
    missing_pairs = sorted(expected_pairs - experiment_seed_pairs)
    unexpected_experiments = sorted({str(row.get("experiment")) for row in rows} - EXPECTED_EXPERIMENTS)
    duplicate_pairs = sorted({pair for pair in experiment_seed_pairs if sum(1 for row in rows if (row.get("experiment"), int(row.get("seed"))) == pair) > 1})
    quality_failures = []
    for row in rows:
        absent = [key for key, value in row["quality"].items() if key.startswith("has_") and not value]
        if absent:
            quality_failures.append({"run_id": row["run_id"], "missing": absent})
    audit = {
        "status": "completed",
        "strict_pass": bool(
            len(rows) == 36
            and not invalid
            and not missing_pairs
            and not unexpected_experiments
            and not duplicate_pairs
            and not quality_failures
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "stage": args.stage,
        "valid_runs": len(rows),
        "expected_runs": 36,
        "invalid_runs": invalid,
        "missing_experiment_seed_pairs": [list(pair) for pair in missing_pairs],
        "unexpected_experiments": unexpected_experiments,
        "duplicate_experiment_seed_pairs": [list(pair) for pair in duplicate_pairs],
        "quality_failures": quality_failures,
        "experiments": sorted({str(row.get("experiment")) for row in rows}),
        "campaigns": sorted({str(row.get("campaign")) for row in rows}),
    }

    headline = [{key: value for key, value in row.items() if not key.startswith("_") and key != "quality"} for row in rows]
    lead_rows = [item for row in rows for item in row["_lead_rows"]]
    depth_rows = [item for row in rows for item in row["_depth_rows"]]
    headline_summary = _summarize(
        headline,
        ("experiment",),
        ("temp_rmse", "salt_rmse", "temp_ap_skill", "salt_ap_skill", "macro_ap_skill", "temp_dap_skill", "salt_dap_skill", "macro_dap_skill", "parameter_count", "best_epoch", "peak_cuda_memory_gib"),
    )
    lead_summary = _summarize(
        lead_rows,
        ("experiment", "variable", "lead"),
        ("rmse", "mse", "ap_skill", "dap_skill"),
    )
    depth_summary = _summarize(
        depth_rows,
        ("experiment", "variable", "depth_name", "depth"),
        ("rmse", "mse", "ap_skill", "dap_skill"),
    )

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "completed" if audit["strict_pass"] else "incomplete",
        "generated_at_utc": audit["generated_at_utc"],
        "split": args.split,
        "stage": args.stage,
        "records": rows,
        "audit": audit,
    }
    (output_dir / "formal_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(output_dir / "runs.csv", headline)
    _write_csv(output_dir / "summary.csv", headline_summary)
    _write_csv(output_dir / "lead_metrics.csv", lead_rows)
    _write_csv(output_dir / "lead_summary.csv", lead_summary)
    _write_csv(output_dir / "depth_metrics.csv", depth_rows)
    _write_csv(output_dir / "depth_summary.csv", depth_summary)
    (output_dir / "summary.md").write_text(_markdown(headline_summary, audit, args.split), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(output_dir), "valid_runs": len(rows), "invalid_runs": len(invalid), "strict_pass": audit["strict_pass"]}, ensure_ascii=False))
    return 0 if audit["strict_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
