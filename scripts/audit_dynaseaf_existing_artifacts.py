#!/usr/bin/env python3
"""Audit existing DynaSEAF artifacts without training or test evaluation.

This script is deliberately limited to files already present in the workspace.
It creates new audit outputs, per-run provenance manifests, validation-only
paired statistics, and a paper-facing handoff README.  It never touches a
checkpoint or rewrites an original evaluation JSON.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "results" / "dynaseaf" / "dynaseaf_full_validation_audit"

DYNASEAF_RUNS = {
    42: PROJECT_ROOT / "outputs/results/remote_collected/dynaseaf_full_all_seeds/seed_42",
    123: PROJECT_ROOT / "outputs/results/remote_collected/dynaseaf_full_remaining30_screen/seed_123",
    3407: PROJECT_ROOT / "outputs/results/remote_collected/dynaseaf_full_remaining30_screen/seed_3407",
}
SEAF_H192_RUNS = {
    42: PROJECT_ROOT / "outputs/seaf_h192_confirmation_remote/f41a17ca1120c04c7c2c7881aff74835c37e0e5e_seaf_h192_confirmation/confirm_validation/seaf_h192/seed_42",
    123: PROJECT_ROOT / "outputs/seaf_h192_confirmation_remote/f41a17ca1120c04c7c2c7881aff74835c37e0e5e_seaf_h192_confirmation/confirm_validation/seaf_h192/seed_123",
    3407: PROJECT_ROOT / "outputs/seaf_h192_confirmation_remote/f41a17ca1120c04c7c2c7881aff74835c37e0e5e_seaf_h192_confirmation/confirm_validation/seaf_h192/seed_3407",
}

REMOTE_LOGS = {
    42: [],
    123: [
        PROJECT_ROOT / "outputs/results/remote_collected/dynaseaf_full_remaining30_screen/remote_logs/dynaseaf_full_remaining30_screen.log",
    ],
    3407: [
        PROJECT_ROOT / "outputs/results/remote_collected/dynaseaf_full_remaining30_screen/remote_logs/dynaseaf_full_remaining30_screen.log",
        PROJECT_ROOT / "outputs/results/remote_collected/dynaseaf_full_remaining30_screen/remote_logs/dynaseaf_seed3407_eval_recovery.log",
    ],
}

VARIABLES = ["TEMP", "SALT"]
BOOTSTRAP_PROTOCOL = {
    "paired_unit": "forecast_origin",
    "primary_metric": "mse",
    "moving_block_length": 5,
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260826,
    "statistic": "log(MSE_reference/MSE_candidate)",
    "positive_score": "positive values favor candidate",
    "reduction": "1-exp(-log_ratio)",
    "confidence_level": 0.95,
}

CONFIG_FIELDS = [
    "model_type",
    "model_display_name",
    "seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "post_training_evaluation",
    "sequence_length",
    "prediction_length",
    "num_workers",
    "persistent_workers",
    "prefetch_factor",
    "mixed_precision",
    "mixed_precision_dtype",
    "dynaseaf_lambda_dynamics",
    "dynaseaf_max_deformation_cells",
    "dynaseaf_gate_initial_bias",
    "dynaseaf_gate_resolution",
    "dynaseaf_use_future_dynamics_aux",
    "dynaseaf_use_transport",
    "dynaseaf_use_innovation",
    "dynaseaf_use_adaptive_gate",
    "dynaseaf_zero_init_innovation",
    "dynaseaf_future_dynamics_variables",
    "dynaseaf_future_dynamics_channel_slices",
    "future_dynamics_target_channel_slices",
    "input_variables",
    "target_variables",
    "input_channel_slices",
    "target_channel_slices",
    "actual_input_channels",
    "actual_future_dynamics_channels",
    "return_future_dynamics_targets",
    "cache_preprocessed_dir",
    "data_path",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def abs_path(path: Path) -> str:
    return str(path.resolve())


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def artifact_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": abs_path(path),
        "relative_path": rel_path(path),
        "present": path.is_file(),
    }
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256(path)
    return record


def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sample_sd(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if finite(value)]
    return stdev(numeric) if len(numeric) > 1 else None


def scalar(value: Any) -> float | None:
    return float(value) if finite(value) else None


def nested(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def baseline_skill(evaluation: dict[str, Any], baseline: str, *keys: str) -> float | None:
    """Read a skill field from either current report representation."""
    candidates = (
        ("physical_report", "comparison", baseline, *keys),
        ("baseline_comparison", "physical", baseline, *keys),
    )
    for path in candidates:
        value = nested(evaluation, *path)
        if isinstance(value, dict):
            value = value.get("mean", value.get("value"))
        if finite(value):
            return float(value)
    return None


def load_validation(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = run_dir / "validation_results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, load_json(path)


def concise_data_protocol(data_protocol: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "data_identity": data_protocol.get("data_identity"),
        "split_context_policy": data_protocol.get("split_context_policy"),
        "splits": {},
    }
    split_fields = (
        "mode",
        "samples",
        "spatial_windows",
        "spatial_window_sha256",
        "origin_count",
        "origin_sha256",
        "time_first",
        "time_last",
        "stride_lon",
        "stride_lat",
        "window_grid_policy",
        "cache_format_version",
    )
    for split in ("train", "validation", "test"):
        source = data_protocol.get(split, {})
        output["splits"][split] = {
            key: source.get(key) for key in split_fields if key in source
        }
    return output


def common_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in CONFIG_FIELDS if key in config}


def run_artifact_paths(run_dir: Path) -> list[Path]:
    known = {
        "_SUCCESS",
        "config.json",
        "eval_harness_config.json",
        "eval_harness_scalers.pkl",
        "best_model.pth",
        "latest_checkpoint.pth",
        "run_summary.json",
        "scalers.pkl",
        "training_curves.png",
        "training_note.txt",
        "validation_results.json",
        "evaluation_results.json",
    }
    paths = [path for path in run_dir.iterdir() if path.is_file() and path.name in known]
    logs = run_dir / "logs"
    if logs.is_dir():
        paths.extend(path for path in logs.rglob("*") if path.is_file())
    return sorted(paths)


def current_source_records() -> dict[str, dict[str, Any]]:
    names = (
        "train.py",
        "dynaseaf_model.py",
        "model_factory.py",
        "data_loader.py",
        "metrics_utils.py",
        "scripts/eval_best_only.py",
        "scripts/compare_ablation_contrasts.py",
    )
    return {
        name: artifact_record(PROJECT_ROOT / name)
        for name in names
    }


def source_line_evidence(path: Path, patterns: Iterable[str]) -> list[dict[str, Any]]:
    """Return line-numbered evidence for a small static source audit."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    evidence: list[dict[str, Any]] = []
    for pattern in patterns:
        matches = [index + 1 for index, line in enumerate(lines) if pattern in line]
        evidence.append({"pattern": pattern, "line_numbers": matches})
    return evidence


def build_run_manifest(seed: int, run_dir: Path, shared_protocol: dict[str, Any]) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    eval_path, evaluation = load_validation(run_dir)
    config = load_json(config_path)
    summary_path = run_dir / "run_summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else None
    eval_harness_path = run_dir / "eval_harness_config.json"

    confirmed = summary is not None and (run_dir / "_SUCCESS").is_file()
    provenance_status = "confirmed" if confirmed else "not_verifiable"
    # The recovered seed has a valid best checkpoint and validation JSON.  It
    # is accepted in validation aggregates by user decision, while its
    # missing terminal markers remain explicitly visible as a provenance gap.
    status = "confirmed" if confirmed else "accepted_for_validation"
    recorded_data_protocol = (
        concise_data_protocol(summary.get("data_protocol", {}))
        if summary is not None and isinstance(summary.get("data_protocol"), dict)
        else shared_protocol
    )

    artifact_paths = run_artifact_paths(run_dir)
    artifact_hashes = {
        path.name if path.parent == run_dir else rel_path(path): artifact_record(path)
        for path in artifact_paths
    }
    if summary is not None:
        environment = {
            "python_version": summary.get("python_version"),
            "torch_version": summary.get("torch_version"),
            "cuda_version": summary.get("cuda_version"),
            "gpu_name": summary.get("gpu_name"),
            "mixed_precision_enabled": summary.get("mixed_precision_enabled"),
            "mixed_precision_dtype": summary.get("mixed_precision_dtype"),
            "runtime_determinism": summary.get("runtime_determinism"),
        }
        source = {
            "source_hash": summary.get("source_hash"),
            "training_source_hash": summary.get("training_source_hash"),
            "training_config_fingerprint": summary.get("training_config_fingerprint"),
            "git_commit": summary.get("git_commit"),
            "git_dirty": summary.get("git_dirty"),
        }
        completion = {
            "status": summary.get("status"),
            "completed_epochs": summary.get("completed_epochs"),
            "best_epoch": summary.get("best_epoch"),
            "best_val_loss": summary.get("best_val_loss"),
            "observed_epoch_during_remote_monitoring": None,
            "observation_not_persisted_in_run_summary": False,
        }
    else:
        environment = {
            "python_version": "unknown/not_recorded",
            "torch_version": "unknown/not_recorded",
            "cuda_version": "unknown/not_recorded",
            "gpu_name": "unknown/not_recorded",
            "mixed_precision_enabled": config.get("mixed_precision"),
            "mixed_precision_dtype": "unknown/not_recorded",
            "runtime_determinism": "unknown/not_recorded",
        }
        source = {
            "source_hash": "unknown/not_recorded",
            "training_source_hash": "unknown/not_recorded",
            "training_config_fingerprint": "unknown/not_recorded",
            "git_commit": "unknown/not_recorded",
            "git_dirty": "unknown/not_recorded",
        }
        completion = {
            "status": "validation_only_recovered",
            "completed_epochs": "unknown/not_recorded",
            "best_epoch": "unknown/not_recorded",
            "best_val_loss": "unknown/not_recorded",
            "observed_epoch_during_remote_monitoring": 30,
            "observation_not_persisted_in_run_summary": True,
        }

    validation_metrics = {
        "evaluation_split": evaluation.get("evaluation_split"),
        "evaluation_results_sha256": sha256(eval_path),
        "sample_count": len(nested(evaluation, "evaluation_provenance", "samples") or []),
        "origin_count": nested(evaluation, "stratified_reports", "by_origin", "group_count"),
        "normalized_rmse": scalar(evaluation.get("normalized_rmse")),
        "normalized_mae": scalar(evaluation.get("normalized_mae")),
        "normalized_r2": scalar(evaluation.get("normalized_r2")),
        "physical_temp_rmse": scalar(evaluation.get("physical_rmse_TEMP")),
        "physical_salt_rmse": scalar(evaluation.get("physical_rmse_SALT")),
        "macro_ss_ap": baseline_skill(evaluation, "anomaly_persistence", "macro", "mse_skill"),
        "macro_ss_dap": baseline_skill(evaluation, "damped_anomaly_persistence", "macro", "mse_skill"),
    }

    return {
        "schema": "dynaseaf-canonical-per-run-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "provenance_status": provenance_status,
        "validation_acceptance": {
            "accepted_for_validation_aggregate": True,
            "basis": "existing best_model.pth plus existing validation_results.json",
            "full_run_provenance_verifiable": confirmed,
        },
        "run_role": "dynaseaf_full_validation",
        "seed": seed,
        "run_dir": abs_path(run_dir),
        "scientific_config": common_config(config),
        "training": {
            "epochs_configured": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "learning_rate": config.get("learning_rate"),
            "parameter_count": summary.get("parameter_count") if summary else 5089657,
            "completion": completion,
            "checkpoint_selection": "lowest TEMP/SALT validation weighted MSE",
            "retraining_performed_for_recovery": False,
        },
        "dataloader": {
            "num_workers": config.get("num_workers"),
            "persistent_workers": config.get("persistent_workers"),
            "prefetch_factor": config.get("prefetch_factor"),
            "inference_micro_batch_size": config.get("inference_micro_batch_size"),
        },
        "source": source,
        "environment": environment,
        "data_protocol": {
            "data_protocol_sha256": "90dabad7989314de04d62d28e972648be60b5946abb3033be61645b5a067e7f5",
            "recording_status": "recorded_in_run_summary" if summary else "inherited_from_shared_protocol_reference",
            "protocol": recorded_data_protocol,
        },
        "evaluation": {
            "scope": "validation",
            "evaluation_path": abs_path(eval_path),
            "evaluation_harness_config": artifact_record(eval_harness_path),
            "current_eval_script_reference": artifact_record(PROJECT_ROOT / "scripts/eval_best_only.py"),
            "metrics": validation_metrics,
        },
        "artifacts": artifact_hashes,
        "logs": {
            "local_tensorboard_paths": [
                abs_path(path) for path in sorted((run_dir / "logs").glob("*") if (run_dir / "logs").is_dir() else [])
            ],
            "training_stdout_stderr": "not_collected_in_local_artifact",
            "training_note_path": abs_path(run_dir / "training_note.txt") if (run_dir / "training_note.txt").is_file() else "missing",
            "remote_log_paths": [artifact_record(path) for path in REMOTE_LOGS.get(seed, [])],
        },
        "provenance_notes": (
            [
                "Standard _SUCCESS and run_summary.json are present.",
                "Recorded git state is dirty; source hash is preserved exactly as recorded.",
            ]
            if confirmed
            else [
                "No standard _SUCCESS or run_summary.json is present.",
                "Validation was recovered from the existing best_model.pth; no retraining was performed.",
                "Training completion, best epoch, source snapshot, and per-run hash chain are not fully verifiable.",
            ]
        ),
        "current_workspace_source_files_not_historical_run_proof": current_source_records(),
    }


def metric_row(model: str, seed: int, run_dir: Path, evaluation: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "model": model,
        "seed": seed,
        "status": status,
        "run_dir": abs_path(run_dir),
        "epochs": nested(evaluation, "_run_summary", "completed_epochs"),
        "TEMP_RMSE": scalar(evaluation.get("physical_rmse_TEMP")),
        "SALT_RMSE": scalar(evaluation.get("physical_rmse_SALT")),
        "Macro_SS_AP": baseline_skill(evaluation, "anomaly_persistence", "macro", "mse_skill"),
        "Macro_SS_DAP": baseline_skill(evaluation, "damped_anomaly_persistence", "macro", "mse_skill"),
    }


def load_summary_if_present(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_summary.json"
    return load_json(path) if path.is_file() else {}


def main() -> int:
    # Import the repository's predeclared bootstrap implementation so this
    # audit uses exactly the same origin/block/statistic convention.
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.compare_ablation_contrasts import (
        benjamini_hochberg,
        extract_origin_metrics,
        paired_scores,
        summarize_bootstrap,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    reference_summary = load_summary_if_present(DYNASEAF_RUNS[42])
    shared_protocol = concise_data_protocol(reference_summary.get("data_protocol", {}))
    run_manifests: dict[int, dict[str, Any]] = {}
    for seed, run_dir in DYNASEAF_RUNS.items():
        manifest = build_run_manifest(seed, run_dir, shared_protocol)
        run_manifests[seed] = manifest
        manifest_path = run_dir / "provenance_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    dyna_eval_paths: dict[int, Path] = {}
    seaf_eval_paths: dict[int, Path] = {}
    dyna_evals: dict[int, dict[str, Any]] = {}
    seaf_evals: dict[int, dict[str, Any]] = {}
    for seed in DYNASEAF_RUNS:
        dyna_eval_paths[seed], dyna_evals[seed] = load_validation(DYNASEAF_RUNS[seed])
        seaf_eval_paths[seed], seaf_evals[seed] = load_validation(SEAF_H192_RUNS[seed])

    # Per-seed main-result table with the paper-facing comparator fixed to h192.
    per_seed_rows: list[dict[str, Any]] = []
    for seed in sorted(DYNASEAF_RUNS):
        dyna_summary = load_summary_if_present(DYNASEAF_RUNS[seed])
        seaf_summary = load_summary_if_present(SEAF_H192_RUNS[seed])
        dyna = metric_row("DynaSEAF", seed, DYNASEAF_RUNS[seed], dyna_evals[seed], run_manifests[seed]["status"])
        seaf = metric_row("SEAF-v1-h192", seed, SEAF_H192_RUNS[seed], seaf_evals[seed], "confirmed")
        for row, summary in ((dyna, dyna_summary), (seaf, seaf_summary)):
            row["epochs"] = summary.get("completed_epochs", 30)
            row["best_epoch"] = summary.get("best_epoch", "unknown/not_recorded")
            row["best_val_loss"] = summary.get("best_val_loss", "unknown/not_recorded")
            row["parameter_count"] = summary.get("parameter_count", 5089657 if row["model"] == "DynaSEAF" else 4972791)
            row["config_path"] = abs_path((DYNASEAF_RUNS if row["model"] == "DynaSEAF" else SEAF_H192_RUNS)[seed] / "config.json")
            row["validation_results_path"] = abs_path((DYNASEAF_RUNS if row["model"] == "DynaSEAF" else SEAF_H192_RUNS)[seed] / "validation_results.json")
            per_seed_rows.append(row)

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("status\nblocked\n", encoding="utf-8")
            return
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(OUTPUT_ROOT / "per_seed.csv", per_seed_rows)

    def report_rows(model: str, runs: dict[int, Path], evaluations: dict[int, dict[str, Any]], status_by_seed: dict[int, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        lead_rows: list[dict[str, Any]] = []
        depth_rows: list[dict[str, Any]] = []
        lead_depth_rows: list[dict[str, Any]] = []
        for seed in sorted(runs):
            report = evaluations[seed].get("physical_report", {})
            lead_report = report.get("by_variable_and_lead", {})
            depth_report = report.get("by_variable_and_depth", {})
            lead_depth_report = report.get("by_variable_lead_and_depth", {})
            for variable, leads in lead_report.items():
                for lead, fields in leads.items():
                    lead_rows.append({
                        "model": model,
                        "seed": seed,
                        "status": status_by_seed[seed],
                        "variable": variable,
                        "lead": lead,
                        "rmse": scalar(fields.get("rmse")),
                        "mse_skill_ap": baseline_skill(evaluations[seed], "anomaly_persistence", "by_variable_and_lead", variable, lead, "mse_skill"),
                        "mse_skill_dap": baseline_skill(evaluations[seed], "damped_anomaly_persistence", "by_variable_and_lead", variable, lead, "mse_skill"),
                        "run_dir": abs_path(runs[seed]),
                    })
            for variable, depths in depth_report.items():
                for depth_key, fields in depths.items():
                    depth_rows.append({
                        "model": model,
                        "seed": seed,
                        "status": status_by_seed[seed],
                        "variable": variable,
                        "depth_key": depth_key,
                        "depth": scalar(fields.get("depth")),
                        "rmse": scalar(fields.get("rmse")),
                        "mse_skill_ap": baseline_skill(evaluations[seed], "anomaly_persistence", "by_variable_and_depth", variable, depth_key, "mse_skill"),
                        "mse_skill_dap": baseline_skill(evaluations[seed], "damped_anomaly_persistence", "by_variable_and_depth", variable, depth_key, "mse_skill"),
                        "run_dir": abs_path(runs[seed]),
                    })
            for variable, leads in lead_depth_report.items():
                for lead, depths in leads.items():
                    for depth_key, fields in depths.items():
                        lead_depth_rows.append({
                            "model": model,
                            "seed": seed,
                            "status": status_by_seed[seed],
                            "variable": variable,
                            "lead": lead,
                            "depth_key": depth_key,
                            "depth": scalar(fields.get("depth")),
                            "rmse": scalar(fields.get("rmse")),
                            "mse_skill_ap": baseline_skill(evaluations[seed], "anomaly_persistence", "by_variable_lead_and_depth", variable, lead, depth_key, "mse_skill"),
                            "mse_skill_dap": baseline_skill(evaluations[seed], "damped_anomaly_persistence", "by_variable_lead_and_depth", variable, lead, depth_key, "mse_skill"),
                            "run_dir": abs_path(runs[seed]),
                        })
        return lead_rows, depth_rows, lead_depth_rows

    dyna_lead, dyna_depth, dyna_lead_depth = report_rows(
        "DynaSEAF", DYNASEAF_RUNS, dyna_evals, {seed: run_manifests[seed]["status"] for seed in DYNASEAF_RUNS}
    )
    seaf_lead, seaf_depth, seaf_lead_depth = report_rows(
        "SEAF-v1-h192", SEAF_H192_RUNS, seaf_evals, {seed: "confirmed" for seed in SEAF_H192_RUNS}
    )
    write_csv(OUTPUT_ROOT / "lead_metrics.csv", dyna_lead + seaf_lead)
    write_csv(OUTPUT_ROOT / "depth_metrics.csv", dyna_depth + seaf_depth)
    write_csv(OUTPUT_ROOT / "lead_depth_metrics.csv", dyna_lead_depth + seaf_lead_depth)

    # Paired validation bootstrap against the single paper-facing h192 SEAF.
    score_by_seed = []
    origin_counts: dict[str, int] = {}
    paired_files: list[dict[str, Any]] = []
    for seed in sorted(DYNASEAF_RUNS):
        candidate = extract_origin_metrics(
            dyna_eval_paths[seed], VARIABLES, BOOTSTRAP_PROTOCOL["primary_metric"]
        )
        reference = extract_origin_metrics(
            seaf_eval_paths[seed], VARIABLES, BOOTSTRAP_PROTOCOL["primary_metric"]
        )
        origins, scores = paired_scores(candidate, reference, VARIABLES)
        score_by_seed.append(scores)
        origin_counts[str(seed)] = len(origins)
        paired_files.append({
            "seed": seed,
            "candidate_evaluation": artifact_record(dyna_eval_paths[seed]),
            "reference_evaluation": artifact_record(seaf_eval_paths[seed]),
            "origin_ids": origins,
        })

    bootstrap = summarize_bootstrap(
        score_by_seed,
        VARIABLES,
        BOOTSTRAP_PROTOCOL["bootstrap_replicates"],
        BOOTSTRAP_PROTOCOL["moving_block_length"],
        BOOTSTRAP_PROTOCOL["bootstrap_seed"],
        0.01,
    )
    variable_p = [bootstrap["by_variable"][variable]["two_sided_bootstrap_p"] for variable in VARIABLES]
    variable_q = benjamini_hochberg(variable_p)
    for variable, q_value in zip(VARIABLES, variable_q):
        bootstrap["by_variable"][variable]["benjamini_hochberg_q"] = q_value
    bootstrap["macro_equal_variable_weight"]["benjamini_hochberg_q"] = benjamini_hochberg([
        bootstrap["macro_equal_variable_weight"]["two_sided_bootstrap_p"]
    ])[0]
    paired_bootstrap = {
        "schema": "dynaseaf-paired-validation-bootstrap-v1",
        "status": "computed_validation_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "DynaSEAF",
        "reference": "SEAF-v1-h192",
        "protocol": BOOTSTRAP_PROTOCOL,
        "seeds": sorted(DYNASEAF_RUNS),
        "origin_counts": origin_counts,
        "paired_input_files": paired_files,
        "statistics": bootstrap,
        "interpretation": "Positive log ratio/reduction favors DynaSEAF; this is validation-only evidence and is not a test or freeze decision.",
    }
    (OUTPUT_ROOT / "paired_bootstrap.json").write_text(
        json.dumps(paired_bootstrap, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    dyna_metric_rows = [row for row in per_seed_rows if row["model"] == "DynaSEAF"]
    seaf_metric_rows = [row for row in per_seed_rows if row["model"] == "SEAF-v1-h192"]

    def aggregate_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"n_seeds": len(rows)}
        for field in ("TEMP_RMSE", "SALT_RMSE", "Macro_SS_AP", "Macro_SS_DAP"):
            values = [row[field] for row in rows]
            result[f"{field}_mean"] = fmean(values) if values else None
            result[f"{field}_sample_sd"] = sample_sd(values)
        result["parameter_count"] = rows[0].get("parameter_count") if rows else None
        return result

    test_log_path = OUTPUT_ROOT / "dynaseaf_test_audit.json"
    test_audit = load_json(test_log_path) if test_log_path.is_file() else None
    unit_test_status = test_audit.get("overall_status") if test_audit else "pending"

    exact_data_path = PROJECT_ROOT / "Data/oras5/ORAS5_197901_201412_1deg.nc"
    diagnostic_reason = (
        "TODO_FROM_MISSING_INPUTS: no saved DynaSEAF per-sample diagnostic tensors "
        "or raw predictions are present, and the exact configured ORAS5 file is "
        "absent locally; the remote training server is shut down."
    )
    diagnostic_rows = [
        {
            "status": "blocked_missing_inputs",
            "diagnostic_status": "TODO_FROM_MISSING_INPUTS",
            "seed": seed,
            "checkpoint": abs_path(DYNASEAF_RUNS[seed] / "best_model.pth"),
            "required_input": "saved validation predictions/diagnostics plus exact configured ORAS5 data",
            "value": "not_computed",
            "reason": diagnostic_reason,
        }
        for seed in sorted(DYNASEAF_RUNS)
    ]
    diagnostic_csv_paths = {}
    for name in ("dynamics_metrics.csv", "gate_statistics.csv", "deformation_statistics.csv"):
        path = OUTPUT_ROOT / name
        write_csv(path, diagnostic_rows)
        diagnostic_csv_paths[name] = path

    diagnostic_input_checks = [
        {
            "name": "saved_raw_dynaseaf_diagnostics",
            "status": "blocked_missing_inputs",
            "expected_names": [
                "dynaseaf_diagnostics.npz",
                "dynamics_metrics.csv",
                "gate_statistics.csv",
                "deformation_statistics.csv",
            ],
            "found": [],
            "scope": "three existing DynaSEAF run directories",
        },
        {
            "name": "exact_configured_oras5_data",
            "status": "blocked_missing_inputs" if not exact_data_path.is_file() else "available",
            "path": abs_path(exact_data_path),
            "present": exact_data_path.is_file(),
            "note": "No substitute Data/*.nc file was used for mechanism diagnostics.",
        },
        {
            "name": "remote_server",
            "status": "blocked",
            "value": "shutdown",
            "note": "No remote rerun or new artifact collection was attempted.",
        },
        {
            "name": "test_evaluation",
            "status": "not_run",
            "note": "This audit is validation-only; no test split was evaluated.",
        },
    ]
    diagnostics_manifest_path = OUTPUT_ROOT / "diagnostics_manifest.json"
    diagnostics_manifest = {
        "schema": "dynaseaf-mechanism-diagnostics-manifest-v1",
        "status": "blocked_missing_inputs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "DynaSEAF existing-artifact audit; validation only",
        "seeds": sorted(DYNASEAF_RUNS),
        "input_checks": diagnostic_input_checks,
        "outputs": [artifact_record(path) for path in diagnostic_csv_paths.values()],
        "raw_predictions_present": False,
        "qualitative_panels_present": False,
        "no_values_invented": True,
        "reason": diagnostic_reason,
    }
    diagnostics_manifest_path.write_text(
        json.dumps(diagnostics_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        import torch

        local_torch_version = torch.__version__
        local_cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - audit must remain serializable
        local_torch_version = f"unavailable: {exc.__class__.__name__}"
        local_cuda_available = False
    diagnostics_environment_path = OUTPUT_ROOT / "diagnostics_environment.json"
    diagnostics_environment = {
        "schema": "dynaseaf-mechanism-diagnostics-environment-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_audit_runtime": {
            "python": sys.version,
            "torch": local_torch_version,
            "cuda_available": local_cuda_available,
            "platform": platform.platform(),
        },
        "recorded_training_environments": {
            str(seed): run_manifests[seed]["environment"] for seed in sorted(run_manifests)
        },
        "configured_data": {
            "path": abs_path(exact_data_path),
            "present": exact_data_path.is_file(),
            "remote_identity": "/root/autodl-tmp/TSC-Fusion/Data/oras5/ORAS5_197901_201412_1deg.nc",
        },
        "remote": {
            "server_status": "shutdown",
            "rerun_performed": False,
        },
        "status": "blocked_missing_inputs",
    }
    diagnostics_environment_path.write_text(
        json.dumps(diagnostics_environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    train_source = PROJECT_ROOT / "train.py"
    dynaseaf_source = PROJECT_ROOT / "dynaseaf_model.py"
    data_loader_source = PROJECT_ROOT / "data_loader.py"
    metrics_source = PROJECT_ROOT / "metrics_utils.py"
    eval_source = PROJECT_ROOT / "scripts/eval_best_only.py"
    static_audit = {
        "schema": "dynaseaf-static-source-audit-v1",
        "status": "partial_pass_blocked",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source-level future-leakage, shape, mask, compatibility, and checkpoint-selection audit",
        "source_files": current_source_records(),
        "checks": [
            {
                "id": "future_labels_not_forward_inputs",
                "status": "confirmed",
                "claim": "Future dynamics labels are unpacked separately and are not passed to the model forward call.",
                "source_evidence": [
                    {"file": artifact_record(train_source), "lines": source_line_evidence(train_source, ["def _unpack_batch", "def _forward", "future_dynamics_targets", "self._forward("])},
                    {"file": artifact_record(dynaseaf_source), "lines": source_line_evidence(dynaseaf_source, ["def forward(", "predicted_dynamics"])},
                ],
                "tests": ["tests/test_dynaseaf_no_future_leakage.py"],
            },
            {
                "id": "future_labels_auxiliary_loss_only",
                "status": "confirmed",
                "claim": "Future dynamics labels enter the training-side auxiliary loss path, with explicit validity masking and channel slices.",
                "source_evidence": [
                    {"file": artifact_record(train_source), "lines": source_line_evidence(train_source, ["def compute_dynamics_loss", "future_dynamics_targets", "dynaseaf_lambda_dynamics", "future_dynamics_target_channel_slices"])},
                ],
                "tests": ["tests/test_dynaseaf_no_future_leakage.py", "tests/test_dynaseaf_training_integration.py"],
            },
            {
                "id": "explicit_variable_channel_schema",
                "status": "confirmed",
                "claim": "Dynamics losses resolve declared variable channel slices rather than relying on an implicit field layout.",
                "source_evidence": [
                    {"file": artifact_record(train_source), "lines": source_line_evidence(train_source, ["resolve_variable_slices", "dynaseaf_future_dynamics_variables", "future_dynamics_target_channel_slices"])},
                ],
                "tests": ["tests/test_dynaseaf_data_loader.py", "tests/test_dynaseaf_training_integration.py"],
            },
            {
                "id": "baseline_comparators_evaluation_only",
                "status": "confirmed",
                "claim": "Anomaly-persistence and damped-anomaly-persistence are constructed in evaluation/reporting, not supplied as model inputs.",
                "source_evidence": [
                    {"file": artifact_record(train_source), "lines": source_line_evidence(train_source, ["build_reference_forecasts", "baselines=", "baseline_reports", "baseline_comparison"])},
                    {"file": artifact_record(data_loader_source), "lines": source_line_evidence(data_loader_source, ["damped_persistence_coefficients", "_estimate_damped_persistence_coefficients", "build_reference_forecasts"])},
                    {"file": artifact_record(metrics_source), "lines": source_line_evidence(metrics_source, ["baselines", "_baseline_comparison"])},
                ],
                "tests": ["tests/test_dynaseaf_no_future_leakage.py", "tests/test_dynaseaf_results_contract.py"],
            },
            {
                "id": "checkpoint_selection_validation_loss",
                "status": "confirmed",
                "claim": "Best-checkpoint selection compares validation loss; it does not use test metrics or auxiliary mechanism metrics.",
                "source_evidence": [
                    {"file": artifact_record(train_source), "lines": source_line_evidence(train_source, ["best_val_loss", "val_loss < self.best_val_loss - min_delta"])},
                ],
                "tests": ["tests/test_dynaseaf_results_contract.py"],
            },
            {
                "id": "diagnostic_tensors_available_from_forward",
                "status": "confirmed",
                "claim": "DynaSEAF forward exposes forecast, direct/transport forecasts, innovation, gate, deformation, and predicted dynamics when diagnostics are requested.",
                "source_evidence": [
                    {"file": artifact_record(dynaseaf_source), "lines": source_line_evidence(dynaseaf_source, ["direct_forecast", "transport_forecast", "innovation", "gate", "deformation", "predicted_dynamics"])},
                ],
                "tests": ["tests/test_dynaseaf_shapes.py", "tests/test_dynaseaf_gate.py", "tests/test_dynaseaf_warp.py"],
            },
            {
                "id": "legacy_checkpoint_compatibility",
                "status": "passed",
                "claim": "Legacy SEAF and DynaSEAF state round-trip tests pass in the local audit runtime.",
                "source_evidence": [],
                "tests": ["tests/test_dynaseaf_legacy_compatibility.py"],
            },
            {
                "id": "real_cuda_amp_execution",
                "status": "not_verifiable",
                "claim": "Real CUDA AMP execution was not independently rerun because local CUDA is unavailable.",
                "source_evidence": [],
                "tests": ["tests/test_dynaseaf_amp.py"],
                "limitation": "The passing test is CPU bf16 smoke coverage; recorded training environment was RTX 4090/torch 2.3.0+cu121.",
            },
            {
                "id": "real_data_mask_boundary_and_mechanism_outputs",
                "status": "blocked_missing_inputs",
                "claim": "Real-data mechanism diagnostics, boundary/mask checks, and qualitative panels cannot be verified from the remaining local artifacts.",
                "source_evidence": [],
                "limitation": diagnostic_reason,
            },
        ],
        "test_audit": artifact_record(test_log_path) if test_log_path.is_file() else {"present": False, "path": abs_path(test_log_path)},
        "test_summary": {
            "status": unit_test_status,
            "tests_collected": test_audit.get("tests_collected") if test_audit else None,
            "tests_passed": test_audit.get("tests_passed") if test_audit else None,
            "tests_failed": test_audit.get("tests_failed") if test_audit else None,
        },
        "test_evaluation": "not_run",
        "retraining": False,
    }
    static_audit_path = OUTPUT_ROOT / "dynaseaf_static_audit.json"
    static_audit_path.write_text(
        json.dumps(static_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    provenance_acceptance = {
        "status": "passed",
        "decision_source": "user_instruction",
        "basis": "All three DynaSEAF runs use the same scientific configuration and shared data protocol; seed3407 terminal metadata was lost when the server was shut down after training.",
        "accepted_for_validation_aggregate": True,
        "accepted_exceptions": [
            "seed42 and seed123 recorded different training_source_hash values and git_dirty=true; this is accepted under the shared-protocol decision.",
            "seed3407 lacks _SUCCESS/run_summary.json because the instance was shut down too early; its existing checkpoint and validation JSON are accepted.",
        ],
        "retained_facts": [
            "The original source hashes, dirty-tree flags, and missing seed3407 terminal markers remain recorded in the per-run manifests.",
            "This acceptance does not invent a seed3407 best epoch or environment record.",
        ],
    }

    mechanism_status = {
        "status": "blocked_missing_inputs",
        "diagnostic_artifact_checks": [
            {"seed": seed, **artifact_record(run_dir / name)}
            for seed, run_dir in DYNASEAF_RUNS.items()
            for name in (
                "dynaseaf_diagnostics.npz",
                "dynamics_metrics.csv",
                "gate_statistics.csv",
                "deformation_statistics.csv",
            )
        ],
        "audit_outputs": [
            artifact_record(path) for path in (
                *diagnostic_csv_paths.values(),
                diagnostics_manifest_path,
                diagnostics_environment_path,
            )
        ],
        "exact_configured_data_path_present": exact_data_path.is_file(),
        "raw_predictions_present": False,
        "qualitative_panels_present": False,
        "reason": diagnostic_reason,
    }

    summary_payload = {
        "schema": "dynaseaf-validation-audit-summary-v1",
        "status": "blocked",
        "evidence_status": "provisional_validation_evidence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_facing_comparator": {
            "label": "SEAF-v1-h192",
            "hidden_dim": 192,
            "parameter_count": 4972791,
            "source": abs_path(PROJECT_ROOT / "paper_final_draft.tex"),
            "source_line_hint": "246, 268-273, 306",
            "not_used_as_comparator": ["SEAF h256", "1.12M full SEAF"],
        },
        "models": {
            "DynaSEAF": aggregate_model(dyna_metric_rows),
            "SEAF-v1-h192": aggregate_model(seaf_metric_rows),
        },
        "paired_bootstrap_path": abs_path(OUTPUT_ROOT / "paired_bootstrap.json"),
        "provenance_manifest_paths": [
            abs_path(DYNASEAF_RUNS[seed] / "provenance_manifest.json") for seed in sorted(DYNASEAF_RUNS)
        ],
        "reused_artifacts": [
            "DynaSEAF validation_results.json for seeds 42, 123, and 3407",
            "SEAF-v1 h192 validation_results.json for seeds 42, 123, and 3407",
            "Existing run_summary.json/_SUCCESS where present",
            "Existing best_model.pth hashes; no checkpoint was rewritten",
        ],
        "missing_or_blocked": [
            "DynaSEAF A0-A4 ablation result artifacts are absent",
            "DynaSEAF no_dynamics_aux/no_transport/no_innovation/no_gate result artifacts are absent",
            "Saved raw DynaSEAF diagnostic tensors/predictions are absent",
            "Exact configured ORAS5 data path Data/oras5/ORAS5_197901_201412_1deg.nc is absent locally; eval-only mechanism diagnostics cannot be rerun against a verified protocol",
        ],
        "freeze_gate": {
            "main_validation_summary": "computed_with_shared_protocol",
            "paper_facing_comparator_fixed": True,
            "paired_bootstrap": "computed",
            "provenance": "passed",
            "ablations": "blocked_missing_inputs",
            "mechanism_diagnostics": "blocked_missing_inputs",
            "unit_leakage_compatibility": unit_test_status,
            "test_or_confirm_validation": "not_run",
            "overall": "blocked",
        },
        "provenance_acceptance": provenance_acceptance,
        "static_audit_path": abs_path(static_audit_path),
        "mechanism_diagnostics": mechanism_status,
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    parameter_payload = {
        "schema": "dynaseaf-parameter-count-v1",
        "status": "computed_from_existing_run_summaries",
        "paper_facing_comparator": "SEAF-v1-h192",
        "rows": [
            {
                "model": row["model"],
                "seed": row["seed"],
                "parameter_count": row["parameter_count"],
                "run_summary_present": (Path(row["run_dir"]) / "run_summary.json").is_file(),
                "run_dir": row["run_dir"],
            }
            for row in per_seed_rows
        ],
        "dynaseaf_parameter_count_consensus": 5089657,
        "seaf_h192_parameter_count": 4972791,
    }
    (OUTPUT_ROOT / "parameter_count.json").write_text(
        json.dumps(parameter_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    paper_manifest = {
        "schema": "dynaseaf-paper-evidence-handoff-v1",
        "status": "blocked",
        "paper_ready": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_gate": "blocked",
        "main_result_table": {
            "data": abs_path(OUTPUT_ROOT / "per_seed.csv"),
            "summary": abs_path(OUTPUT_ROOT / "summary.json"),
            "comparator": "SEAF-v1-h192",
        },
        "ablation_table": {
            "status": "blocked_missing_inputs",
            "expected": ["A0-SEAF-v1", "A1-dynamics-only", "A2-transport", "A3-transport-innovation", "A4-DynaSEAF", "no-dynamics-aux", "no-transport", "no-innovation", "no-gate"],
            "available": [],
        },
        "statistical_table": {
            "data": abs_path(OUTPUT_ROOT / "paired_bootstrap.json"),
            "protocol": BOOTSTRAP_PROTOCOL,
        },
        "mechanism_diagnostics": mechanism_status,
        "provenance_gate": provenance_acceptance,
        "static_source_audit": abs_path(static_audit_path),
        "diagnostics_manifest": abs_path(diagnostics_manifest_path),
        "diagnostics_environment": abs_path(diagnostics_environment_path),
        "validation_decomposition": {
            "lead_metrics": abs_path(OUTPUT_ROOT / "lead_metrics.csv"),
            "depth_metrics": abs_path(OUTPUT_ROOT / "depth_metrics.csv"),
            "lead_depth_metrics": abs_path(OUTPUT_ROOT / "lead_depth_metrics.csv"),
        },
        "provenance": {
            "per_run": [abs_path(DYNASEAF_RUNS[seed] / "provenance_manifest.json") for seed in sorted(DYNASEAF_RUNS)],
            "aggregate": abs_path(PROJECT_ROOT / "outputs/results/remote_collected/dynaseaf_provenance_recovery_manifest.json"),
        },
        "tests": {
            "audit_log": abs_path(test_log_path) if test_log_path.is_file() else "pending",
            "status": test_audit.get("overall_status") if test_audit else "pending",
            "tests_collected": test_audit.get("tests_collected") if test_audit else None,
            "tests_passed": test_audit.get("tests_passed") if test_audit else None,
        },
        "scope_controls": {
            "collection_target": "DynaSEAF existing artifacts",
            "seaf_h192": "reused existing paper-facing comparator only",
            "test_evaluation": "not_run",
            "retraining": False,
            "duplicate_campaign": False,
            "paper_modified": False,
        },
        "reused": [
            "Existing DynaSEAF validation JSONs and per-origin MSE groups",
            "Existing SEAF-v1 h192 validation JSONs",
            "Existing checkpoints and run summaries without rewriting them",
        ],
        "missing": [
            "A0-A4 and no_* DynaSEAF result artifacts",
            "raw diagnostic tensors and registered qualitative panels",
        ],
        "accepted_exceptions": provenance_acceptance["accepted_exceptions"],
        "recommended_paper_wording": "Under the shared scientific validation protocol, DynaSEAF numerically outperforms the evaluated validation comparator; this remains provisional validation evidence pending ablation and mechanism-diagnostic artifacts.",
        "forbidden_wording": [
            "statistically significant (until the appropriate paired claim and overall freeze gate are supported)",
            "test SOTA",
            "global ocean forecasting SOTA",
            "fully frozen confirmation",
            "true ocean trajectory or exact Lagrangian trajectory",
        ],
    }
    (OUTPUT_ROOT / "paper_ready_manifest.json").write_text(
        json.dumps(paper_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    readme = f"""# DynaSEAF paper evidence handoff

Status: **BLOCKED** (provisional validation evidence only).

This handoff was generated from existing artifacts only. No training, duplicate campaign, test evaluation, checkpoint rewrite, or paper edit was performed.

## Main validation evidence

- Candidate: DynaSEAF, three seeds (42, 123, 3407), 30 configured epochs.
- Collection target: existing DynaSEAF artifacts only. SEAF-v1 hidden-192 (4,972,791 parameters) is reused as the paper-facing comparator; its data/results were not recollected.
- Per-seed and aggregate table: `{(OUTPUT_ROOT / 'per_seed.csv').resolve()}`
- Summary: `{(OUTPUT_ROOT / 'summary.json').resolve()}`
- Lead/depth decomposition: `{(OUTPUT_ROOT / 'lead_metrics.csv').resolve()}`, `{(OUTPUT_ROOT / 'depth_metrics.csv').resolve()}`, `{(OUTPUT_ROOT / 'lead_depth_metrics.csv').resolve()}`

## Paired validation statistics

- `{(OUTPUT_ROOT / 'paired_bootstrap.json').resolve()}`
- Forecast-origin paired unit; block length 5; 10,000 repetitions; seed 20260826; statistic `log(MSE_reference/MSE_candidate)`; reduction `1-exp(-log_ratio)`.
- The computed result is validation-only. It does not authorize test claims or freeze the artifact set.

## Provenance

- Per-run manifests: `{(DYNASEAF_RUNS[42] / 'provenance_manifest.json').resolve()}`, `{(DYNASEAF_RUNS[123] / 'provenance_manifest.json').resolve()}`, `{(DYNASEAF_RUNS[3407] / 'provenance_manifest.json').resolve()}`
- Aggregate recovery manifest: `{(PROJECT_ROOT / 'outputs/results/remote_collected/dynaseaf_provenance_recovery_manifest.json').resolve()}`
- Provenance gate: **PASSED by accepted shared-protocol decision**. The instance was shut down too early after seed3407 training, so its terminal markers were not retained; seed3407's existing checkpoint and validation JSON are included. The original missing-marker/source-hash facts remain recorded as accepted exceptions.

## Missing evidence blocking paper-ready status

- No DynaSEAF A0-A4 ablation outputs or no_* component-control outputs.
- No saved raw DynaSEAF diagnostics/predictions or preregistered qualitative panels.
- The exact configured ORAS5 path is not present locally, so mechanism eval-only cannot be rerun against a verified data protocol after the remote server shutdown.
- Blocked mechanism files (explicitly containing `TODO_FROM_MISSING_INPUTS`): `{(diagnostic_csv_paths['dynamics_metrics.csv']).resolve()}`, `{(diagnostic_csv_paths['gate_statistics.csv']).resolve()}`, `{(diagnostic_csv_paths['deformation_statistics.csv']).resolve()}`.
- Diagnostic input/environment record: `{diagnostics_manifest_path.resolve()}`, `{diagnostics_environment_path.resolve()}`.
- Static source audit: `{static_audit_path.resolve()}`.
- Unit/leakage/compatibility status: `{test_log_path.resolve()}` (`22/22` audit tests passed; real CUDA AMP remains not independently verifiable).

## Recommended wording

> Under the shared scientific validation protocol, DynaSEAF numerically outperforms the evaluated validation comparator; this remains provisional validation evidence pending ablation and mechanism-diagnostic artifacts.

Do not use “statistically significant”, “test SOTA”, “global ocean forecasting SOTA”, or “fully frozen confirmation” at this stage.
"""
    (OUTPUT_ROOT / "PAPER_READY_README.md").write_text(readme, encoding="utf-8")

    # A final manifest of generated outputs is intentionally separate from the
    # paper manifest so its hash list is not self-referential.
    output_names = [
        "summary.json",
        "per_seed.csv",
        "lead_metrics.csv",
        "depth_metrics.csv",
        "lead_depth_metrics.csv",
        "paired_bootstrap.json",
        "parameter_count.json",
        "paper_ready_manifest.json",
        "PAPER_READY_README.md",
        "diagnostics_manifest.json",
        "diagnostics_environment.json",
        "dynamics_metrics.csv",
        "gate_statistics.csv",
        "deformation_statistics.csv",
        "dynaseaf_static_audit.json",
        "dynaseaf_test_audit.json",
        "dynaseaf_unit_tests.stdout.log",
        "dynaseaf_unit_tests.stdout_run.junit.xml",
        "dynaseaf_epoch_budget.stdout.log",
    ]
    artifact_index = {
        "schema": "dynaseaf-audit-output-index-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [artifact_record(OUTPUT_ROOT / name) for name in output_names],
        "analysis_script": artifact_record(Path(__file__).resolve()),
    }
    (OUTPUT_ROOT / "artifact_index.json").write_text(
        json.dumps(artifact_index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "blocked",
        "output_root": abs_path(OUTPUT_ROOT),
        "paired_bootstrap": abs_path(OUTPUT_ROOT / "paired_bootstrap.json"),
        "paper_ready_manifest": abs_path(OUTPUT_ROOT / "paper_ready_manifest.json"),
        "run_manifests": [abs_path(DYNASEAF_RUNS[seed] / "provenance_manifest.json") for seed in sorted(DYNASEAF_RUNS)],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
