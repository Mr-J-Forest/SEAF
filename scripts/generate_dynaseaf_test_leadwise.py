#!/usr/bin/env python3
"""Aggregate frozen DynaSEAF test reports into lead-wise metrics.

This script only reads existing ``evaluation_results.json`` files.  It does
not load a model, run inference, select a checkpoint, or modify any training
artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "remote_collected"
    / "dynaseaf_test_eval_20260901_v1"
)
DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "paper_ready"
    / "dynaseaf_paper_artifacts_20260902"
)
SEEDS = (42, 123, 3407)
VARIABLES = ("TEMP", "SALT")
LEADS = (1, 2, 3, 4, 5)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if value is None:
        raise KeyError(f"missing metric {key!r}")
    return float(value)


def make_plot(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"TEMP": "#1554A6", "SALT": "#D96600"}
    labels = {"TEMP": "TEMP", "SALT": "SALT"}
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), squeeze=False)
    panels = (
        (axes[0, 0], "ss_ap", "SS$_{\\rm AP}$", "(a)"),
        (axes[0, 1], "ss_dap", "SS$_{\\rm DAP}$", "(b)"),
        (axes[1, 0], "physical_rmse", "TEMP RMSE (°C)", "(c)"),
        (axes[1, 1], "physical_rmse", "SALT RMSE (PSU)", "(d)"),
    )
    for ax, metric, ylabel, panel_label in panels:
        ax.text(
            0.03,
            0.95,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        variables = ("TEMP", "SALT") if metric != "physical_rmse" else (ylabel.split()[0],)
        for variable in variables:
            rows = [row for row in summary_rows if row["variable"] == variable]
            rows.sort(key=lambda row: int(row["lead"]))
            x = np.asarray([row["lead"] for row in rows], dtype=float)
            mean = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=float)
            sd = np.asarray([row[f"{metric}_sample_sd"] for row in rows], dtype=float)
            ax.errorbar(
                x,
                mean,
                yerr=sd,
                color=colors[variable],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                capsize=2,
                label=labels[variable],
            )
        ax.set_xlabel("Forecast lead (months)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(LEADS))
        ax.grid(True, alpha=0.16, linewidth=0.6)
        if metric != "physical_rmse":
            ax.axhline(0.0, color="#6B7280", linewidth=0.7, linestyle="--", alpha=0.7)
    axes[0, 0].legend(loc="best", frameon=False)
    fig.tight_layout(pad=1.0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    data_dir = artifact_dir / "data"
    figure_dir = artifact_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for seed in SEEDS:
        path = input_dir / f"seed_{seed}" / "evaluation_results.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = read_json(path)
        if report.get("evaluation_split") != "test":
            raise ValueError(f"not a test report: {path}")
        source_records.append({
            "seed": seed,
            "path": str(path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        })
        normalized = report["normalized_report"]
        physical = report["physical_report"]
        comparisons = report["baseline_comparison"]["normalized"]
        for variable in VARIABLES:
            for lead in LEADS:
                lead_key = f"lead_{lead}"
                model_metrics = normalized["by_variable_and_lead"][variable][lead_key]
                ap_metrics = normalized["baselines"]["anomaly_persistence"]["by_variable_and_lead"][variable][lead_key]
                dap_metrics = normalized["baselines"]["damped_anomaly_persistence"]["by_variable_and_lead"][variable][lead_key]
                physical_metrics = physical["by_variable_and_lead"][variable][lead_key]
                model_mse = scalar(model_metrics, "mse")
                ap_mse = scalar(ap_metrics, "mse")
                dap_mse = scalar(dap_metrics, "mse")
                ss_ap = 1.0 - model_mse / ap_mse
                ss_dap = 1.0 - model_mse / dap_mse
                stored_ap = scalar(comparisons["anomaly_persistence"]["by_variable_and_lead"][variable][lead_key], "mse_skill")
                stored_dap = scalar(comparisons["damped_anomaly_persistence"]["by_variable_and_lead"][variable][lead_key], "mse_skill")
                if not np.isclose(ss_ap, stored_ap, rtol=0.0, atol=1e-10):
                    raise ValueError(f"AP skill mismatch for seed={seed}, {variable}, lead={lead}")
                if not np.isclose(ss_dap, stored_dap, rtol=0.0, atol=1e-10):
                    raise ValueError(f"DAP skill mismatch for seed={seed}, {variable}, lead={lead}")
                per_seed_rows.append({
                    "seed": seed,
                    "split": "test",
                    "variable": variable,
                    "lead": lead,
                    "normalized_mse": model_mse,
                    "ap_mse": ap_mse,
                    "dap_mse": dap_mse,
                    "ss_ap": ss_ap,
                    "ss_dap": ss_dap,
                    "physical_rmse": scalar(physical_metrics, "rmse"),
                })

    summary_rows: list[dict[str, Any]] = []
    for variable in VARIABLES:
        for lead in LEADS:
            selected = [
                row for row in per_seed_rows
                if row["variable"] == variable and row["lead"] == lead
            ]
            if len(selected) != len(SEEDS):
                raise ValueError(f"expected {len(SEEDS)} seeds for {variable}, lead {lead}")
            summary: dict[str, Any] = {
                "split": "test",
                "variable": variable,
                "lead": lead,
                "n_seeds": len(selected),
            }
            for metric in ("normalized_mse", "ap_mse", "dap_mse", "ss_ap", "ss_dap", "physical_rmse"):
                values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_sample_sd"] = float(values.std(ddof=1))
            summary_rows.append(summary)

    per_seed_path = data_dir / "test_leadwise_per_seed.csv"
    summary_path = data_dir / "test_leadwise_mean_sd.csv"
    with per_seed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed_rows[0]))
        writer.writeheader()
        writer.writerows(per_seed_rows)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    figure_path = figure_dir / "main_test_leadwise_metrics.pdf"
    make_plot(summary_rows, figure_path)
    # Keep a paper-local copy alongside the other figures referenced directly
    # by paper_final_draft.tex while retaining the complete artifact package.
    paper_figure_path = PROJECT_ROOT / "figures" / "generated" / "main_test_leadwise_metrics.pdf"
    paper_figure_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path, paper_figure_path)
    shutil.copy2(figure_path.with_suffix(".png"), paper_figure_path.with_suffix(".png"))
    manifest = {
        "schema": "dynaseaf-frozen-test-leadwise-v1",
        "status": "completed_from_frozen_evaluation_reports",
        "input_dir": str(input_dir),
        "sources": source_records,
        "seeds": list(SEEDS),
        "variables": list(VARIABLES),
        "leads": list(LEADS),
        "formulas": {
            "ss_ap": "1 - normalized_model_MSE / normalized_AP_MSE",
            "ss_dap": "1 - normalized_model_MSE / normalized_DAP_MSE",
            "summary": "mean and sample standard deviation across seeds",
        },
        "inference_run": False,
        "checkpoint_selection": False,
        "training_run": False,
        "artifacts": {
            "per_seed_csv": str(per_seed_path),
            "mean_sd_csv": str(summary_path),
            "figure_pdf": str(figure_path),
            "figure_png": str(figure_path.with_suffix(".png")),
            "paper_figure_pdf": str(paper_figure_path),
            "paper_figure_png": str(paper_figure_path.with_suffix(".png")),
        },
    }
    write_json(data_dir / "test_leadwise_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "rows_per_seed": len(per_seed_rows),
        "summary_rows": len(summary_rows),
        "per_seed_csv": str(per_seed_path),
        "mean_sd_csv": str(summary_path),
        "figure_pdf": str(figure_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
