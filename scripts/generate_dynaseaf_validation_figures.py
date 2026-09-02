#!/usr/bin/env python3
"""Create paper figures from the completed validation diagnostic CSVs.

The script is deliberately limited to the validation mechanism archive.  It
does not open checkpoints, iterate a data loader, or evaluate the test split.
Every plotted number is read from ``sample_mechanism_metrics.csv`` and the
corresponding completed diagnostics manifest is checked before use.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = [
    "dynaseaf_a0_seaf_reference",
    "dynaseaf_a1_dynamics_only",
    "dynaseaf_a2_transport",
    "dynaseaf_a3_transport_innovation",
    "dynaseaf_a4_full",
    "dynaseaf_no_dynamics_aux",
    "dynaseaf_no_transport",
    "dynaseaf_no_innovation",
    "dynaseaf_no_gate",
]
MODEL_LABELS = {
    "dynaseaf_a0_seaf_reference": "A0 SEAF-v1",
    "dynaseaf_a1_dynamics_only": "A1 dynamics",
    "dynaseaf_a2_transport": "A2 transport",
    "dynaseaf_a3_transport_innovation": "A3 + innovation",
    "dynaseaf_a4_full": "A4 DynaSEAF",
    "dynaseaf_no_dynamics_aux": "no dynamics aux",
    "dynaseaf_no_transport": "no transport",
    "dynaseaf_no_innovation": "no innovation",
    "dynaseaf_no_gate": "no gate",
}
VARIABLES = ("TEMP", "SALT")
COMPONENT_FIELDS = (
    "final_rmse_normalized",
    "direct_rmse_normalized",
    "transport_rmse_normalized",
    "gate_mean",
    "deformation_magnitude_mean",
    "deformation_magnitude_max",
    "predicted_dynamics_abs_mean",
)
ROW_KEY_FIELDS = ("sample_index", "origin_id", "region_id", "variable", "lead")


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def numeric(value: Any) -> float | None:
    return float(value) if finite(value) else None


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return fmean(values), stdev(values) if len(values) > 1 else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(input_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    screen_root = input_root / "screen"
    if not screen_root.is_dir():
        raise FileNotFoundError(f"missing validation screen directory: {screen_root}")

    for model_name in MODEL_ORDER:
        run_dir = screen_root / model_name / "seed_42"
        manifest_path = run_dir / "diagnostics_manifest.json"
        csv_path = run_dir / "sample_mechanism_metrics.csv"
        if not manifest_path.is_file() or not csv_path.is_file():
            raise FileNotFoundError(f"incomplete validation artifact: {run_dir}")
        manifest = load_json(manifest_path)
        required = {
            "status": "completed",
            "split": "validation",
            "test_iteration": False,
            "retraining": False,
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"manifest gate failed for {model_name}: {key}={manifest.get(key)!r}"
                )
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            model_rows = list(csv.DictReader(handle))
        if len(model_rows) != 33440:
            raise ValueError(
                f"unexpected row count for {model_name}: {len(model_rows)} (expected 33440)"
            )
        for row in model_rows:
            if row.get("split") != "validation" or row.get("model") not in {"SEAF-v1", "DynaSEAF"}:
                raise ValueError(f"unexpected row metadata in {csv_path}")
            row["source_model"] = model_name
            for field in COMPONENT_FIELDS:
                row[field] = numeric(row.get(field))
            row["lead"] = int(row["lead"])
            row["sample_index"] = int(row["sample_index"])
            rows.append(row)
        sources.append(
            {
                "model": model_name,
                "label": MODEL_LABELS[model_name],
                "seed": 42,
                "split": "validation",
                "row_count": len(model_rows),
                "manifest": str(manifest_path.resolve()),
                "sample_csv": str(csv_path.resolve()),
            }
        )
    return rows, sources


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_ablation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        model_rows = [row for row in rows if row["source_model"] == model_name]
        output: dict[str, Any] = {
            "model": model_name,
            "label": MODEL_LABELS[model_name],
            "seed": 42,
            "split": "validation",
            "row_count": len(model_rows),
        }
        variable_means: list[float] = []
        for variable in VARIABLES:
            values = [
                row["final_rmse_normalized"]
                for row in model_rows
                if row.get("variable") == variable and row["final_rmse_normalized"] is not None
            ]
            mean, spread = mean_sd(values)
            output[f"{variable}_final_rmse_normalized_mean_per_sample"] = mean
            output[f"{variable}_final_rmse_normalized_sample_sd"] = spread
            if mean is not None:
                variable_means.append(mean)
        output["macro_final_rmse_normalized_mean_per_sample"] = (
            fmean(variable_means) if variable_means else None
        )
        for field in COMPONENT_FIELDS[1:]:
            values = [row[field] for row in model_rows if row[field] is not None]
            mean, spread = mean_sd(values)
            output[f"{field}_mean_per_sample"] = mean
            output[f"{field}_sample_sd"] = spread
        result.append(output)
    return result


def aggregate_by_lead(rows: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for variable in VARIABLES:
        for lead in range(1, 6):
            subset = [
                row
                for row in rows
                if row["source_model"] == model_name
                and row.get("variable") == variable
                and row.get("lead") == lead
            ]
            output: dict[str, Any] = {
                "model": model_name,
                "label": MODEL_LABELS[model_name],
                "seed": 42,
                "split": "validation",
                "variable": variable,
                "lead": lead,
                "row_count": len(subset),
            }
            for field in COMPONENT_FIELDS:
                values = [row[field] for row in subset if row[field] is not None]
                mean, spread = mean_sd(values)
                output[f"{field}_mean_per_sample"] = mean
                output[f"{field}_sample_sd"] = spread
            result.append(output)
    return result


def aggregate_innovation_counterfactual(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare final forecasts with and without the residual correction.

    The stored innovation tensor is a residual correction, not a standalone
    forecast.  This
    comparison uses the final forecast from the full model and the final
    forecast from the matched ``no_innovation`` control at the same sample and
    lead, so negative values mean that retaining the correction lowers RMSE.
    """

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(row[field] for field in ROW_KEY_FIELDS)

    full = {
        key(row): row
        for row in rows
        if row["source_model"] == "dynaseaf_a4_full"
    }
    no_innovation = {
        key(row): row
        for row in rows
        if row["source_model"] == "dynaseaf_no_innovation"
    }
    if set(full) != set(no_innovation):
        raise ValueError("A4 and no_innovation validation samples are not aligned")

    result: list[dict[str, Any]] = []
    for variable in VARIABLES:
        for lead in range(1, 6):
            paired = [
                (full[sample_key], no_innovation[sample_key])
                for sample_key in full
                if full[sample_key]["variable"] == variable
                and full[sample_key]["lead"] == lead
                and full[sample_key]["final_rmse_normalized"] is not None
                and no_innovation[sample_key]["final_rmse_normalized"] is not None
            ]
            full_values = [pair[0]["final_rmse_normalized"] for pair in paired]
            no_innovation_values = [pair[1]["final_rmse_normalized"] for pair in paired]
            deltas = [full_value - control_value for full_value, control_value in zip(full_values, no_innovation_values)]
            mean, spread = mean_sd(deltas)
            result.append(
                {
                    "comparison": "full_minus_no_innovation",
                    "seed": 42,
                    "split": "validation",
                    "variable": variable,
                    "lead": lead,
                    "pair_count": len(deltas),
                    "full_final_rmse_normalized_mean_per_sample": fmean(full_values),
                    "no_innovation_final_rmse_normalized_mean_per_sample": fmean(no_innovation_values),
                    "delta_full_minus_no_innovation_mean_per_sample": mean,
                    "delta_full_minus_no_innovation_sample_sd": spread,
                }
            )
    return result


def style_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9dee8", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def save_figure(figure: Any, figure_dir: Path, stem: str) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    pdf = figure_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [str(png.resolve()), str(pdf.resolve())]


def make_ablation_figure(summary: list[dict[str, Any]], figure_dir: Path) -> list[str]:
    labels = [row["label"] for row in summary]
    temp = [row["TEMP_final_rmse_normalized_mean_per_sample"] for row in summary]
    salt = [row["SALT_final_rmse_normalized_mean_per_sample"] for row in summary]
    x = np.arange(len(labels))
    width = 0.38
    figure, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(x - width / 2, temp, width, label="TEMP", color="#2474a8")
    ax.bar(x + width / 2, salt, width, label="SALT", color="#d9822b")
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_ylabel("Mean per-sample normalized RMSE")
    ax.set_title("DynaSEAF validation ablation screen (seed 42)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.text(
        0.995,
        1.01,
        "lower is better; validation only",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#59636e",
    )
    style_axes(ax)
    figure.tight_layout()
    return save_figure(figure, figure_dir, "dynaseaf_ablation_validation")


def make_mechanism_figure(
    by_lead: list[dict[str, Any]],
    innovation_counterfactual: list[dict[str, Any]],
    figure_dir: Path,
) -> list[str]:
    colors = {
        "final_rmse_normalized": "#111827",
        "direct_rmse_normalized": "#2474a8",
        "transport_rmse_normalized": "#c2410c",
    }
    names = {
        "final_rmse_normalized": "final",
        "direct_rmse_normalized": "direct",
        "transport_rmse_normalized": "transport",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    for ax, variable in zip(axes[0], VARIABLES):
        subset = [row for row in by_lead if row["variable"] == variable]
        leads = [row["lead"] for row in subset]
        for field, color in colors.items():
            ax.plot(
                leads,
                [row[f"{field}_mean_per_sample"] for row in subset],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=color,
                label=names[field],
            )
        ax.set_title(variable)
        ax.set_xlabel("Forecast lead (month)")
        ax.set_ylabel("Mean per-sample normalized RMSE")
        ax.set_xticks(range(1, 6))
        style_axes(ax)
    axes[0, 1].legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1, 0]
    for variable, color in (("TEMP", "#2474a8"), ("SALT", "#d9822b")):
        subset = [row for row in innovation_counterfactual if row["variable"] == variable]
        ax.plot(
            [row["lead"] for row in subset],
            [row["delta_full_minus_no_innovation_mean_per_sample"] for row in subset],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=variable,
        )
    ax.axhline(0.0, color="#6b7280", linewidth=0.9, linestyle="--")
    ax.set_title("Residual-correction counterfactual")
    ax.set_xlabel("Forecast lead (month)")
    ax.set_ylabel("$\\Delta$ normalized RMSE\n(full $-$ without correction)")
    ax.set_xticks(range(1, 6))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(
        0.995,
        0.02,
        "negative: correction lowers final RMSE",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#59636e",
    )
    style_axes(ax)

    ax = axes[1, 1]
    for variable, color in (("TEMP", "#2474a8"), ("SALT", "#d9822b")):
        subset = [row for row in by_lead if row["variable"] == variable]
        ax.plot(
            [row["lead"] for row in subset],
            [row["gate_mean_mean_per_sample"] for row in subset],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"gate ({variable})",
        )
    dynamics = [row for row in by_lead if row["variable"] == "TEMP"]
    right = ax.twinx()
    right.plot(
        [row["lead"] for row in dynamics],
        [row["predicted_dynamics_abs_mean_mean_per_sample"] for row in dynamics],
        marker="s",
        linestyle="--",
        linewidth=1.5,
        markersize=4,
        color="#374151",
        label="predicted dynamics |·|",
    )
    ax.set_title("Adaptive gate and dynamics")
    ax.set_xlabel("Forecast lead (month)")
    ax.set_ylabel("Gate mean")
    right.set_ylabel("Predicted dynamics absolute mean")
    ax.set_xticks(range(1, 6))
    style_axes(ax)
    right.spines["top"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = right.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper right")
    figure.suptitle("DynaSEAF validation mechanism statistics (seed 42)", fontsize=12)
    return save_figure(figure, figure_dir, "dynaseaf_mechanism_validation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--figure_dir", required=True, type=Path)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    figure_dir = args.figure_dir.resolve()

    rows, sources = load_rows(input_root)
    summary = aggregate_ablation(rows)
    by_lead = aggregate_by_lead(rows, "dynaseaf_a4_full")
    innovation_counterfactual = aggregate_innovation_counterfactual(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "ablation_validation_summary.csv", summary)
    write_csv(output_dir / "dynaseaf_mechanism_by_lead_validation.csv", by_lead)
    write_csv(
        output_dir / "innovation_counterfactual_by_lead_validation.csv",
        innovation_counterfactual,
    )
    figures = make_ablation_figure(summary, figure_dir)
    figures.extend(make_mechanism_figure(by_lead, innovation_counterfactual, figure_dir))
    manifest = {
        "schema": "dynaseaf-validation-paper-figures-v2",
        "status": "computed_validation_only",
        "split": "validation",
        "seed": 42,
        "test_iteration": False,
        "retraining": False,
        "source_root": str(input_root),
        "source_rows": len(rows),
        "source_runs": sources,
        "outputs": {
            "ablation_summary": str((output_dir / "ablation_validation_summary.csv").resolve()),
            "mechanism_by_lead": str((output_dir / "dynaseaf_mechanism_by_lead_validation.csv").resolve()),
            "innovation_counterfactual": str(
                (output_dir / "innovation_counterfactual_by_lead_validation.csv").resolve()
            ),
            "figures": figures,
        },
        "interpretation_note": (
            "Ablation and mechanism values are means over validation per-sample rows "
            "from seed 42; they are not three-seed global evaluation aggregates. "
            "The stored innovation tensor is not evaluated as a standalone forecast; "
            "its effect is represented by the paired full-minus-control final-RMSE comparison."
        ),
    }
    (output_dir / "paper_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "source_rows": len(rows), "figures": figures}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
