#!/usr/bin/env python3
"""Summarize the validation-only progressive SEAF architecture screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _macro_skill(evaluation: dict, baseline_name: str) -> float | None:
    comparisons = evaluation.get("baseline_comparison", {})
    physical = comparisons.get("physical") or comparisons.get("normalized") or {}
    baseline = physical.get(baseline_name, {})
    skill = baseline.get("macro", {}).get("mse_skill")
    if isinstance(skill, dict):
        skill = skill.get("mean")
    return None if skill is None else float(skill)


def _experiment_specs(matrix_path: Path, stage: str) -> list[dict]:
    matrix = _read_json(matrix_path)
    experiments = matrix.get(stage)
    if not isinstance(experiments, list):
        raise ValueError(f"Matrix stage {stage!r} must be a concrete list")
    return [dict(item) for item in experiments]


def _json_cell(value: object) -> str:
    return "" if value is None else json.dumps(value, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        required=True,
        help="campaign name or path below outputs/results/campaigns",
    )
    parser.add_argument(
        "--matrix",
        default="configs/oras5_seaf_architecture_screen.json",
    )
    parser.add_argument("--stage", default="screen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    matrix_path = (PROJECT_ROOT / args.matrix).resolve()
    campaign_arg = Path(args.campaign)
    campaign_root = (
        campaign_arg.resolve()
        if campaign_arg.exists()
        else (PROJECT_ROOT / "outputs" / "results" / "campaigns" / args.campaign)
    )
    experiment_specs = _experiment_specs(matrix_path, args.stage)

    rows = []
    missing = []
    macro_ap_by_experiment: dict[str, float] = {}
    for index, specification in enumerate(experiment_specs):
        experiment = str(specification["name"])
        default_comparison = (
            None if index == 0 else str(experiment_specs[index - 1]["name"])
        )
        comparison_experiment = specification.get("compare_to", default_comparison)
        if comparison_experiment is not None:
            comparison_experiment = str(comparison_experiment)
        run_dir = campaign_root / args.stage / experiment / f"seed_{args.seed}"
        evaluation_path = run_dir / "validation_results.json"
        summary_path = run_dir / "run_summary.json"
        if not evaluation_path.is_file() or not summary_path.is_file():
            missing.append(str(run_dir))
            continue
        evaluation = _read_json(evaluation_path)
        summary = _read_json(summary_path)
        if evaluation.get("evaluation_split") != "validation":
            raise RuntimeError(f"Screening result is not validation-only: {run_dir}")

        diagnostics = (
            evaluation.get("model_diagnostics")
            or summary.get("model_diagnostics")
            or {}
        )
        macro_ap = _macro_skill(evaluation, "anomaly_persistence")
        macro_dap = _macro_skill(evaluation, "damped_anomaly_persistence")
        comparison_ap = macro_ap_by_experiment.get(comparison_experiment)
        delta_ap = (
            None if comparison_ap is None or macro_ap is None
            else macro_ap - comparison_ap
        )
        rows.append({
            "experiment_id": experiment,
            "comparison_experiment_id": comparison_experiment,
            "paper_model_name": diagnostics.get("model_display_name", "SEAF"),
            "seed": args.seed,
            "completed_epochs": summary.get("completed_epochs"),
            "best_epoch": summary.get("best_epoch"),
            "best_val_loss": summary.get("best_val_loss"),
            "temp_rmse": evaluation.get("physical_rmse_TEMP"),
            "salt_rmse": evaluation.get("physical_rmse_SALT"),
            "macro_ss_ap": macro_ap,
            "macro_ss_dap": macro_dap,
            "delta_macro_ss_ap_vs_previous": delta_ap,
            "passes_ap_delta_0_01": (
                None if delta_ap is None else bool(delta_ap > 0.01)
            ),
            "parameter_count": summary.get("parameter_count"),
            "router_type": diagnostics.get("router_type"),
            "lead_member_weights": diagnostics.get("lead_member_weights"),
            "spectral_fusion_scales": diagnostics.get("spectral_fusion_scales"),
            "forcing_fusion_scale": diagnostics.get("forcing_fusion_scale"),
            "parameter_breakdown": diagnostics.get("parameter_breakdown"),
            "run_dir": str(run_dir.resolve()),
        })
        if macro_ap is not None:
            macro_ap_by_experiment[experiment] = macro_ap

    if missing and not args.allow_incomplete:
        raise FileNotFoundError(
            "Incomplete architecture screen; missing result files under:\n- "
            + "\n- ".join(missing)
        )
    if not rows:
        raise RuntimeError("No completed SEAF architecture-screen runs were found")

    output_json = campaign_root / "architecture_screen_summary.json"
    output_csv = campaign_root / "architecture_screen_summary.csv"
    output_md = campaign_root / "architecture_screen_summary.md"
    output_json.write_text(
        json.dumps(
            {
                "matrix": str(matrix_path),
                "stage": args.stage,
                "seed": args.seed,
                "validation_only": True,
                "missing_runs": missing,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        for key in (
            "lead_member_weights",
            "spectral_fusion_scales",
            "parameter_breakdown",
        ):
            csv_row[key] = _json_cell(csv_row[key])
        csv_rows.append(csv_row)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# SEAF architecture screen (validation only)",
        "",
        "| Internal experiment | Compared with | TEMP RMSE | SALT RMSE | Macro SS_AP | Macro SS_DAP | ΔSS_AP | Params |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: object, digits: int = 5) -> str:
            return "—" if value is None else f"{float(value):.{digits}f}"

        params = row["parameter_count"]
        params_text = "—" if params is None else f"{int(params):,}"
        lines.append(
            f"| {row['experiment_id']} | "
            f"{row['comparison_experiment_id'] or '—'} | "
            f"{fmt(row['temp_rmse'])} | "
            f"{fmt(row['salt_rmse'])} | {fmt(row['macro_ss_ap'])} | "
            f"{fmt(row['macro_ss_dap'])} | "
            f"{fmt(row['delta_macro_ss_ap_vs_previous'])} | {params_text} |"
        )
    lines.extend([
        "",
        "All rows use the paper-facing model name **SEAF**. Experiment IDs only track internal module additions.",
        "Each `ΔSS_AP` uses the matrix's explicit `compare_to` branch when present; otherwise it falls back to the preceding row.",
        "The automatic promotion flag checks only the pre-declared `ΔMacro SS_AP > 0.01` rule; any alternative validation criterion must be reviewed explicitly.",
        "",
    ])
    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(output_json)
    print(output_csv)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
