#!/usr/bin/env python3
"""Audit and summarize the validation-only SEAF LCFF confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

try:
    from scripts.compare_ablation_contrasts import (
        benjamini_hochberg,
        extract_origin_metrics,
        paired_scores,
        summarize_bootstrap,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/ rather than the repo root.
    from compare_ablation_contrasts import (
        benjamini_hochberg,
        extract_origin_metrics,
        paired_scores,
        summarize_bootstrap,
    )


SEEDS = (42, 123, 3407)
VARIABLES = ("TEMP", "SALT")
DISPLAY_NAMES = {
    "seaf_no_lcff": "SEAF w/o LCFF",
    "seaf_lcff": "SEAF",
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def nested(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def number(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    return (
        run_dir / "validation_results.json",
        run_dir / "run_summary.json",
        run_dir / "config.json",
    )


def extract_run(
    variant: str,
    seed: int,
    run_dir: Path,
    expected_boundary: tuple[str, bool] | None = None,
) -> dict[str, Any]:
    evaluation_path, summary_path, config_path = run_paths(run_dir)
    missing = [
        str(path) for path in (evaluation_path, summary_path, config_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing required run artifacts: " + ", ".join(missing))
    evaluation = read_json(evaluation_path)
    summary = read_json(summary_path)
    config = read_json(config_path)
    if evaluation.get("evaluation_split") != "validation":
        raise RuntimeError(f"non-validation evaluation found: {evaluation_path}")
    if summary.get("status") != "completed" or int(summary.get("completed_epochs", 0)) != 30:
        raise RuntimeError(f"run is not a completed 30-epoch run: {run_dir}")
    expected = expected_boundary or {
        "seaf_no_lcff": ("spatial", False),
        "seaf_lcff": ("lead", True),
    }[variant]
    actual = (str(config.get("router_type")), bool(config.get("use_forcing_encoder")))
    if actual != expected:
        raise RuntimeError(
            f"LCFF boundary mismatch for {variant}/seed={seed}: {actual} != {expected}"
        )

    physical = evaluation.get("physical_report") or {}
    ap = nested(
        evaluation, "baseline_comparison", "physical", "anomaly_persistence"
    ) or {}
    dap = nested(
        evaluation,
        "baseline_comparison",
        "physical",
        "damped_anomaly_persistence",
    ) or {}
    diagnostics = evaluation.get("model_diagnostics") or summary.get("model_diagnostics") or {}
    row = {
        "variant": variant,
        "paper_model_name": DISPLAY_NAMES[variant],
        "seed": seed,
        "parameter_count": summary.get("parameter_count") or nested(
            diagnostics, "parameter_breakdown", "total"
        ),
        "best_epoch": summary.get("best_epoch"),
        "validation_selection_metric": evaluation.get("evaluation_loss"),
        "temp_rmse": number(nested(physical, "by_variable", "TEMP", "rmse")),
        "salt_rmse": number(nested(physical, "by_variable", "SALT", "rmse")),
        "temp_ss_ap": number(nested(ap, "by_variable", "TEMP", "mse_skill")),
        "salt_ss_ap": number(nested(ap, "by_variable", "SALT", "mse_skill")),
        "macro_ss_ap": number(nested(ap, "macro", "mse_skill")),
        "macro_ss_dap": number(nested(dap, "macro", "mse_skill")),
        "run_dir": str(run_dir.resolve()),
    }
    required_values = (
        "temp_rmse", "salt_rmse", "temp_ss_ap", "salt_ss_ap",
        "macro_ss_ap", "macro_ss_dap",
    )
    if any(row[key] is None for key in required_values):
        raise RuntimeError(f"headline metrics are incomplete: {evaluation_path}")
    return {
        "row": row,
        "evaluation": evaluation,
        "evaluation_path": evaluation_path,
        "diagnostics": diagnostics,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = (
        "parameter_count", "temp_rmse", "salt_rmse", "macro_ss_ap", "macro_ss_dap"
    )
    output = []
    for variant in ("seaf_no_lcff", "seaf_lcff"):
        members = [row for row in rows if row["variant"] == variant]
        record: dict[str, Any] = {
            "variant": variant,
            "paper_model_name": DISPLAY_NAMES[variant],
            "n": len(members),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in members]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = stdev(values)
        output.append(record)
    return output


def lead_skill(evaluation: dict[str, Any], variable: str, lead: int) -> float:
    value = nested(
        evaluation,
        "baseline_comparison",
        "physical",
        "anomaly_persistence",
        "by_variable_and_lead",
        variable,
        f"lead_{lead}",
        "mse_skill",
    )
    parsed = number(value)
    if parsed is None:
        raise RuntimeError(f"missing lead-wise AP skill: {variable}/lead_{lead}")
    return parsed


def markdown(
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    delta_summary: dict[str, Any],
    lead_summary: list[dict[str, Any]],
    router_summary: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    interaction: dict[str, Any],
    recommendation: str,
    paper_safe_paragraph: str,
) -> str:
    def pm(row: dict[str, Any], metric: str) -> str:
        return f"{row[metric + '_mean']:.6f} +/- {row[metric + '_std']:.6f}"

    lines = [
        "# SEAF LCFF confirmation (validation only)",
        "",
        "## Per-seed results",
        "",
        "| Variant | Seed | Params | TEMP RMSE | SALT RMSE | TEMP SS_AP | SALT SS_AP | Macro SS_AP | Macro SS_DAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['paper_model_name']} | {row['seed']} | {int(row['parameter_count']):,} | "
            f"{row['temp_rmse']:.6f} | {row['salt_rmse']:.6f} | "
            f"{row['temp_ss_ap']:.6f} | {row['salt_ss_ap']:.6f} | "
            f"{row['macro_ss_ap']:.6f} | {row['macro_ss_dap']:.6f} |"
        )
    lines.extend([
        "",
        "## Three-seed aggregate (mean +/- sample standard deviation)",
        "",
        "| Variant | Params | TEMP RMSE | SALT RMSE | Macro SS_AP | Macro SS_DAP |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in aggregate:
        lines.append(
            f"| {row['paper_model_name']} | {pm(row, 'parameter_count')} | "
            f"{pm(row, 'temp_rmse')} | {pm(row, 'salt_rmse')} | "
            f"{pm(row, 'macro_ss_ap')} | {pm(row, 'macro_ss_dap')} |"
        )
    lines.extend([
        "",
        "## Per-seed LCFF deltas (SEAF minus SEAF w/o LCFF)",
        "",
        "| Seed | Delta TEMP RMSE | Delta SALT RMSE | Delta Macro SS_AP | Delta Macro SS_DAP |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in deltas:
        lines.append(
            f"| {row['seed']} | {row['delta_temp_rmse']:.6f} | "
            f"{row['delta_salt_rmse']:.6f} | {row['delta_macro_ss_ap']:.6f} | "
            f"{row['delta_macro_ss_dap']:.6f} |"
        )
    lines.append(
        "\nThree-seed mean deltas: "
        f"TEMP RMSE {delta_summary['delta_temp_rmse_mean']:.6f}, "
        f"SALT RMSE {delta_summary['delta_salt_rmse_mean']:.6f}, "
        f"Macro SS_AP {delta_summary['delta_macro_ss_ap_mean']:.6f}, and "
        f"Macro SS_DAP {delta_summary['delta_macro_ss_dap_mean']:.6f}."
    )
    macro = bootstrap["macro_equal_variable_weight"]
    lines.extend([
        "",
        "## Paired moving-block bootstrap",
        "",
        f"Geometric MSE reduction: {macro['geometric_mse_reduction_fraction']:.6f}; "
        f"95% CI [{macro['ci95_geometric_mse_reduction_fraction'][0]:.6f}, "
        f"{macro['ci95_geometric_mse_reduction_fraction'][1]:.6f}]; "
        f"p={macro['two_sided_bootstrap_p']:.6g}; "
        f"BH q={macro['benjamini_hochberg_q']:.6g}; "
        f"P(candidate better)={macro['probability_candidate_better']:.6f}.",
        "",
        "## Lead-wise mean LCFF delta",
        "",
        "| Lead | TEMP Delta SS_AP | SALT Delta SS_AP |",
        "|---:|---:|---:|",
    ])
    for lead in range(1, 6):
        by_variable = {
            row["variable"]: row for row in lead_summary if row["lead"] == lead
        }
        lines.append(
            f"| {lead} | {by_variable['TEMP']['delta_ss_ap_mean']:.6f} | "
            f"{by_variable['SALT']['delta_ss_ap_mean']:.6f} |"
        )
    lines.extend([
        "",
        "## Mean LCFF router weights across seeds",
        "",
        "| Lead | Member 1 | Member 2 | Member 3 | Member 4 |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in router_summary:
        lines.append(
            f"| {row['lead']} | {row['member_1_mean']:.6f} | "
            f"{row['member_2_mean']:.6f} | {row['member_3_mean']:.6f} | "
            f"{row['member_4_mean']:.6f} |"
        )
    lines.extend([
        "",
        "## Seed-42 interaction diagnostic",
        "",
        f"Single-seed difference-in-differences: {interaction['difference_in_differences']:.6f}. "
        "This is a design diagnostic, not a statistically established interaction.",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "## Paper-safe result paragraph",
        "",
        paper_safe_paragraph,
        "",
        "All results in this report are validation-only. AP and DAP are external evaluation references; neither enters the SEAF forward pass.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42-screen-root", required=True)
    parser.add_argument("--confirmation-root", required=True)
    parser.add_argument("--forcing-only-seed42-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--contrasts", default="configs/oras5_seaf_lcff_contrasts.json"
    )
    args = parser.parse_args()

    seed42_root = Path(args.seed42_screen_root).resolve()
    confirmation_root = Path(args.confirmation_root).resolve()
    forcing_only_root = Path(args.forcing_only_seed42_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        ("seaf_no_lcff", 42): seed42_root / "screen" / "seaf_local_global" / "seed_42",
        ("seaf_lcff", 42): seed42_root / "screen" / "seaf_forcing_fusion" / "seed_42",
    }
    for variant in ("seaf_no_lcff", "seaf_lcff"):
        for seed in (123, 3407):
            run_dirs[(variant, seed)] = (
                confirmation_root / "confirm_validation" / variant / f"seed_{seed}"
            )

    extracted = {
        key: extract_run(key[0], key[1], run_dir)
        for key, run_dir in run_dirs.items()
    }
    rows = [extracted[(variant, seed)]["row"] for variant in ("seaf_no_lcff", "seaf_lcff") for seed in SEEDS]
    aggregate = summarize_rows(rows)

    deltas = []
    for seed in SEEDS:
        reference = extracted[("seaf_no_lcff", seed)]["row"]
        candidate = extracted[("seaf_lcff", seed)]["row"]
        deltas.append({
            "seed": seed,
            "delta_temp_rmse": candidate["temp_rmse"] - reference["temp_rmse"],
            "delta_salt_rmse": candidate["salt_rmse"] - reference["salt_rmse"],
            "delta_macro_ss_ap": candidate["macro_ss_ap"] - reference["macro_ss_ap"],
            "delta_macro_ss_dap": candidate["macro_ss_dap"] - reference["macro_ss_dap"],
        })
    delta_summary: dict[str, Any] = {}
    for metric in (
        "delta_temp_rmse", "delta_salt_rmse", "delta_macro_ss_ap", "delta_macro_ss_dap"
    ):
        values = [float(row[metric]) for row in deltas]
        delta_summary[f"{metric}_mean"] = mean(values)
        delta_summary[f"{metric}_std"] = stdev(values)

    lead_rows = []
    for seed in SEEDS:
        reference = extracted[("seaf_no_lcff", seed)]["evaluation"]
        candidate = extracted[("seaf_lcff", seed)]["evaluation"]
        for variable in VARIABLES:
            for lead in range(1, 6):
                lead_rows.append({
                    "seed": seed,
                    "variable": variable,
                    "lead": lead,
                    "delta_ss_ap": lead_skill(candidate, variable, lead) - lead_skill(reference, variable, lead),
                })
    lead_summary = []
    for variable in VARIABLES:
        for lead in range(1, 6):
            values = [
                row["delta_ss_ap"] for row in lead_rows
                if row["variable"] == variable and row["lead"] == lead
            ]
            lead_summary.append({
                "variable": variable,
                "lead": lead,
                "delta_ss_ap_mean": mean(values),
                "delta_ss_ap_std": stdev(values),
            })

    router_rows = []
    for seed in SEEDS:
        weights = extracted[("seaf_lcff", seed)]["diagnostics"].get("lead_member_weights")
        if not isinstance(weights, list) or len(weights) != 5:
            raise RuntimeError(f"invalid lead-router weights for seed {seed}")
        for lead, values in enumerate(weights, start=1):
            if len(values) != 4 or not all(math.isfinite(float(value)) for value in values):
                raise RuntimeError(f"invalid router vector for seed {seed}, lead {lead}")
            if not math.isclose(sum(float(value) for value in values), 1.0, abs_tol=1e-5):
                raise RuntimeError(f"router weights do not sum to one for seed {seed}, lead {lead}")
            router_rows.append({
                "seed": seed,
                "lead": lead,
                **{f"member_{index + 1}": float(value) for index, value in enumerate(values)},
            })
    router_summary = []
    for lead in range(1, 6):
        members = [row for row in router_rows if row["lead"] == lead]
        record: dict[str, Any] = {"lead": lead}
        for member in range(1, 5):
            values = [float(row[f"member_{member}"]) for row in members]
            record[f"member_{member}_mean"] = mean(values)
            record[f"member_{member}_std"] = stdev(values)
        router_summary.append(record)

    specification = read_json(Path(args.contrasts).resolve())
    protocol = specification["protocol"]
    score_by_seed = []
    origin_counts = {}
    for seed in SEEDS:
        candidate = extract_origin_metrics(
            extracted[("seaf_lcff", seed)]["evaluation_path"], list(VARIABLES), "mse"
        )
        reference = extract_origin_metrics(
            extracted[("seaf_no_lcff", seed)]["evaluation_path"], list(VARIABLES), "mse"
        )
        origins, scores = paired_scores(candidate, reference, list(VARIABLES))
        score_by_seed.append(scores)
        origin_counts[str(seed)] = len(origins)
    bootstrap = summarize_bootstrap(
        score_by_seed,
        list(VARIABLES),
        int(protocol["bootstrap_replicates"]),
        int(protocol["moving_block_length"]),
        int(protocol["bootstrap_seed"]),
        float(protocol["meaningful_reduction_fraction"]),
    )
    p_value = bootstrap["macro_equal_variable_weight"]["two_sided_bootstrap_p"]
    q_value = benjamini_hochberg([p_value])[0]
    bootstrap["macro_equal_variable_weight"]["benjamini_hochberg_q"] = q_value
    if (
        bootstrap["confirmation_status"] == "supported"
        and q_value > float(protocol["confirmation_fdr"])
    ):
        bootstrap["confirmation_status"] = "uncertain"
    bootstrap["seeds"] = list(SEEDS)
    bootstrap["origin_counts"] = origin_counts
    bootstrap["protocol"] = protocol

    interaction_runs = {
        "router_off_forcing_off": extracted[("seaf_no_lcff", 42)]["row"]["macro_ss_ap"],
        "router_on_forcing_off": extract_run(
            "seaf_no_lcff", 42,
            seed42_root / "screen" / "seaf_lead_router" / "seed_42",
            expected_boundary=("lead", False),
        )["row"]["macro_ss_ap"],
        "router_off_forcing_on": extract_run(
            "seaf_lcff", 42,
            forcing_only_root / "screen" / "seaf_forcing_fusion" / "seed_42",
            expected_boundary=("spatial", True),
        )["row"]["macro_ss_ap"],
        "router_on_forcing_on": extracted[("seaf_lcff", 42)]["row"]["macro_ss_ap"],
    }
    interaction = {
        **interaction_runs,
        "difference_in_differences": (
            interaction_runs["router_on_forcing_on"]
            - interaction_runs["router_on_forcing_off"]
            - interaction_runs["router_off_forcing_on"]
            + interaction_runs["router_off_forcing_off"]
        ),
        "label": "single-seed interaction diagnostic",
    }

    delta_ap = [row["delta_macro_ss_ap"] for row in deltas]
    positive_seeds = sum(value > 0 for value in delta_ap)
    mean_delta_ap = mean(delta_ap)
    temp_delta = mean(row["delta_temp_rmse"] for row in deltas)
    salt_delta = mean(row["delta_salt_rmse"] for row in deltas)
    supported = bootstrap["confirmation_status"] == "supported"
    if mean_delta_ap > 0 and positive_seeds == 3 and supported and temp_delta <= 0 and salt_delta <= 0:
        recommendation = "A. Promote LCFF as a formal SEAF module."
    elif mean_delta_ap <= 0 and positive_seeds <= 1 and not supported:
        recommendation = "C. Remove LCFF and retain the simpler backbone."
    else:
        recommendation = "B. Keep LCFF exploratory; evidence is insufficient."
    paper_safe_paragraph = (
        "On the validation split, LCFF was evaluated as a coupled module that "
        "combines dedicated external-forcing encoding with lead-specific "
        "aggregation of deterministic anomaly hypotheses. Across three seeds, "
        f"its mean change in Macro SS_AP was {mean_delta_ap:+.6f}, with "
        f"{positive_seeds}/3 seeds favoring LCFF. The paired moving-block "
        f"bootstrap estimated a geometric MSE reduction of "
        f"{bootstrap['macro_equal_variable_weight']['geometric_mse_reduction_fraction']:.6f} "
        f"(95% CI {bootstrap['macro_equal_variable_weight']['ci95_geometric_mse_reduction_fraction']}). "
        "These results support only the recommendation stated above and do not "
        "establish causal forcing effects, probabilistic uncertainty, or "
        "independent benefits from either LCFF component."
    )

    forbidden_artifacts = []
    for run_dir in run_dirs.values():
        for name in ("test_results.json", "evaluation_results.json", "test_metrics.json"):
            if (run_dir / name).exists():
                forbidden_artifacts.append(str((run_dir / name).resolve()))
    audit = {
        "strict_pass": not forbidden_artifacts and all(count > 0 for count in origin_counts.values()),
        "validation_only": True,
        "required_runs": 6,
        "completed_runs": len(rows),
        "seeds": list(SEEDS),
        "forbidden_test_artifacts": forbidden_artifacts,
        "positive_macro_ss_ap_seeds": positive_seeds,
        "recommendation": recommendation,
    }

    write_csv(output / "per_seed_metrics.csv", rows)
    write_csv(output / "aggregate_metrics.csv", aggregate)
    write_csv(output / "per_seed_deltas.csv", deltas)
    write_csv(output / "delta_summary.csv", [delta_summary])
    write_csv(output / "lead_deltas.csv", lead_rows)
    write_csv(output / "lead_delta_summary.csv", lead_summary)
    write_csv(output / "router_weights.csv", router_rows)
    write_csv(output / "router_weight_summary.csv", router_summary)
    (output / "paired_lcff_contrast.json").write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "seed42_interaction.json").write_text(
        json.dumps(interaction, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(
        markdown(
            rows,
            aggregate,
            deltas,
            delta_summary,
            lead_summary,
            router_summary,
            bootstrap,
            interaction,
            recommendation,
            paper_safe_paragraph,
        ),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit["strict_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
