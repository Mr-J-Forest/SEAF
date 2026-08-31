#!/usr/bin/env python3
"""Summarize matched LocalCNN anomaly vs full-field validation runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG

try:
    from scripts.compare_ablation_contrasts import (
        benjamini_hochberg,
        extract_origin_metrics,
        paired_scores,
        summarize_bootstrap,
    )
except ModuleNotFoundError:
    from compare_ablation_contrasts import (
        benjamini_hochberg,
        extract_origin_metrics,
        paired_scores,
        summarize_bootstrap,
    )


SEEDS = (42, 123, 3407)
VARIABLES = ("TEMP", "SALT")
RUNTIME_KEYS = {
    "seed",
    "note",
    "training_note",
    "result_dir",
    "explicit_result_dir",
    "resume_dir",
    "post_training_evaluation",
    "preassembled_mmap_dir",
}
TARGET_KEYS = {
    "enable_target_climatology_anomaly",
    "ablation_direct_full_field",
}
VALIDATION_ARCHIVE_FILES = (
    "config.json",
    "run_summary.json",
    "validation_results.json",
)
def is_excluded_artifact_name(name: str) -> bool:
    lowered = name.lower()
    return lowered == "evaluation_results.json" or lowered.startswith("test_")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_validation_only_archive(
    source_root: Path,
    experiment: str,
    destination_root: Path,
) -> dict[str, Any]:
    """Copy only validation-analysis inputs and inventory excluded artifacts."""
    records = []
    excluded = []
    for seed in SEEDS:
        source_dir = run_dir(source_root, experiment, seed)
        destination_dir = run_dir(destination_root, experiment, seed)
        destination_dir.mkdir(parents=True, exist_ok=True)
        for name in VALIDATION_ARCHIVE_FILES:
            source = source_dir / name
            destination = destination_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination)
            source_hash = sha256(source)
            destination_hash = sha256(destination)
            if source_hash != destination_hash:
                raise RuntimeError(f"validation archive hash mismatch: {source}")
            records.append({
                "seed": seed,
                "filename": name,
                "source": str(source.resolve()),
                "destination": str(destination.resolve()),
                "sha256": source_hash,
            })
        for path in source_dir.iterdir():
            if path.is_file() and is_excluded_artifact_name(path.name):
                excluded.append({
                    "seed": seed,
                    "path": str(path.resolve()),
                    "policy": "inventoried_by_path_only; not opened or copied",
                })
    return {
        "source_root": str(source_root.resolve()),
        "destination_root": str(destination_root.resolve()),
        "copied_validation_inputs": records,
        "excluded_source_artifacts": excluded,
    }


def nested(payload: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(payload, dict) or key not in payload:
            return None
        payload = payload[key]
    return payload


def number(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("mean")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"expected finite metric, got {value!r}")
    return float(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_dir(root: Path, experiment: str, seed: int) -> Path:
    return root / "confirm_validation" / experiment / f"seed_{seed}"


def load_run(root: Path, experiment: str, seed: int, formulation: str) -> dict[str, Any]:
    directory = run_dir(root, experiment, seed)
    evaluation_path = directory / "validation_results.json"
    summary_path = directory / "run_summary.json"
    config_path = directory / "config.json"
    for path in (evaluation_path, summary_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluation = read_json(evaluation_path)
    summary = read_json(summary_path)
    config = read_json(config_path)
    if evaluation.get("evaluation_split") != "validation":
        raise RuntimeError(f"non-validation report: {evaluation_path}")
    if summary.get("status") != "completed" or int(summary.get("completed_epochs", 0)) != 30:
        raise RuntimeError(f"incomplete 30-epoch run: {directory}")
    expected_target_anomaly = formulation == "direct_anomaly"
    if bool(config.get("enable_target_climatology_anomaly")) != expected_target_anomaly:
        raise RuntimeError(f"target formulation mismatch: {config_path}")
    if bool(config.get("ablation_direct_full_field")) == expected_target_anomaly:
        raise RuntimeError(f"direct-full-field flag mismatch: {config_path}")

    physical = evaluation.get("physical_report") or {}
    ap = nested(evaluation, "baseline_comparison", "physical", "anomaly_persistence") or {}
    dap = nested(
        evaluation, "baseline_comparison", "physical", "damped_anomaly_persistence"
    ) or {}
    row = {
        "formulation": formulation,
        "seed": seed,
        "parameter_count": summary.get("parameter_count"),
        "best_epoch": summary.get("best_epoch"),
        "temp_rmse": number(nested(physical, "by_variable", "TEMP", "rmse")),
        "salt_rmse": number(nested(physical, "by_variable", "SALT", "rmse")),
        "macro_ss_ap": number(nested(ap, "macro", "mse_skill")),
        "macro_ss_dap": number(nested(dap, "macro", "mse_skill")),
        "run_dir": str(directory.resolve()),
    }
    origins = nested(evaluation, "stratified_reports", "by_origin", "groups") or {}
    if len(origins) != 44:
        raise RuntimeError(f"expected 44 forecast origins: {evaluation_path}")
    return {
        "row": row,
        "evaluation_path": evaluation_path,
        "config": config,
        "origin_ids": sorted(origins, key=int),
    }


def normalized_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Backfill serialized legacy omissions with the model's declared defaults."""
    defaulted = {
        key: value for key, value in DEFAULT_CONFIG.items() if key not in config
    }
    return {**DEFAULT_CONFIG, **config}, defaulted


def config_differences(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    candidate, _ = normalized_config(candidate)
    reference, _ = normalized_config(reference)
    output = {}
    for key in sorted(set(candidate) | set(reference)):
        if key in RUNTIME_KEYS or candidate.get(key) == reference.get(key):
            continue
        output[key] = {
            "direct_anomaly": candidate.get(key),
            "direct_full_field": reference.get(key),
        }
    return output


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for formulation in ("direct_anomaly", "direct_full_field"):
        members = [row for row in rows if row["formulation"] == formulation]
        item: dict[str, Any] = {"formulation": formulation, "n": len(members)}
        for metric in ("parameter_count", "temp_rmse", "salt_rmse", "macro_ss_ap", "macro_ss_dap"):
            values = [float(row[metric]) for row in members]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values)
        output.append(item)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomaly-root", required=True)
    parser.add_argument("--full-field-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--contrast-spec", default="configs/oras5_local_cnn_target_contrasts.json"
    )
    args = parser.parse_args()

    anomaly_root = Path(args.anomaly_root).resolve()
    full_field_root = Path(args.full_field_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = read_json(Path(args.contrast_spec).resolve())
    protocol = spec["protocol"]

    archive_root = output / "validation_only_inputs"
    anomaly_archive_root = archive_root / "direct_anomaly"
    full_field_archive_root = archive_root / "direct_full_field"
    archive_manifest = {
        "direct_anomaly": build_validation_only_archive(
            anomaly_root, "local_cnn", anomaly_archive_root
        ),
        "direct_full_field": build_validation_only_archive(
            full_field_root, "local_cnn_full_field", full_field_archive_root
        ),
        "read_policy": (
            "All analysis reads are restricted to copied config.json, "
            "run_summary.json, and validation_results.json files."
        ),
        "test_artifacts_read": [],
    }
    (output / "validation_only_archive_manifest.json").write_text(
        json.dumps(archive_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in SEEDS:
        runs[("direct_anomaly", seed)] = load_run(
            anomaly_archive_root, "local_cnn", seed, "direct_anomaly"
        )
        runs[("direct_full_field", seed)] = load_run(
            full_field_archive_root,
            "local_cnn_full_field",
            seed,
            "direct_full_field",
        )

    parity = {}
    default_completion = {}
    for seed in SEEDS:
        anomaly = runs[("direct_anomaly", seed)]
        full_field = runs[("direct_full_field", seed)]
        if anomaly["origin_ids"] != full_field["origin_ids"]:
            raise RuntimeError(f"forecast-origin mismatch for seed {seed}")
        differences = config_differences(anomaly["config"], full_field["config"])
        parity[str(seed)] = differences
        _, anomaly_defaulted = normalized_config(anomaly["config"])
        _, full_field_defaulted = normalized_config(full_field["config"])
        default_completion[str(seed)] = {
            "direct_anomaly": anomaly_defaulted,
            "direct_full_field": full_field_defaulted,
        }
        unexpected = set(differences) - TARGET_KEYS
        if unexpected:
            raise RuntimeError(f"unexpected protocol differences for seed {seed}: {sorted(unexpected)}")
        if set(differences) != TARGET_KEYS:
            raise RuntimeError(f"missing target-only differences for seed {seed}: {differences}")

    rows = [runs[(formulation, seed)]["row"] for formulation in ("direct_anomaly", "direct_full_field") for seed in SEEDS]
    aggregate_rows = aggregate(rows)
    deltas = []
    scores = []
    for seed in SEEDS:
        anomaly_row = runs[("direct_anomaly", seed)]["row"]
        full_row = runs[("direct_full_field", seed)]["row"]
        deltas.append({
            "seed": seed,
            "delta_temp_rmse_anomaly_minus_full_field": anomaly_row["temp_rmse"] - full_row["temp_rmse"],
            "delta_salt_rmse_anomaly_minus_full_field": anomaly_row["salt_rmse"] - full_row["salt_rmse"],
            "delta_macro_ss_ap_anomaly_minus_full_field": anomaly_row["macro_ss_ap"] - full_row["macro_ss_ap"],
            "delta_macro_ss_dap_anomaly_minus_full_field": anomaly_row["macro_ss_dap"] - full_row["macro_ss_dap"],
        })
        candidate = extract_origin_metrics(
            runs[("direct_anomaly", seed)]["evaluation_path"], list(VARIABLES), "mse"
        )
        reference = extract_origin_metrics(
            runs[("direct_full_field", seed)]["evaluation_path"], list(VARIABLES), "mse"
        )
        _, seed_scores = paired_scores(candidate, reference, list(VARIABLES))
        scores.append(seed_scores)

    bootstrap = summarize_bootstrap(
        scores,
        list(VARIABLES),
        int(protocol["bootstrap_replicates"]),
        int(protocol["moving_block_length"]),
        int(protocol["bootstrap_seed"]),
        float(protocol["meaningful_reduction_fraction"]),
    )
    reports = [bootstrap["macro_equal_variable_weight"], *bootstrap["by_variable"].values()]
    q_values = benjamini_hochberg([report["two_sided_bootstrap_p"] for report in reports])
    for report, q_value in zip(reports, q_values):
        report["benjamini_hochberg_q"] = q_value
    bootstrap.update({"protocol": protocol, "seeds": list(SEEDS), "origins_per_seed": 44})

    forbidden = []
    for result in runs.values():
        directory = Path(result["row"]["run_dir"])
        forbidden.extend(
            str(path.resolve()) for path in directory.iterdir()
            if path.is_file() and is_excluded_artifact_name(path.name)
        )
    audit = {
        "strict_pass": not forbidden,
        "validation_only": True,
        "completed_runs": len(rows),
        "seeds": list(SEEDS),
        "origins_per_seed": 44,
        "target_only_config_differences": parity,
        "legacy_default_completion": default_completion,
        "derived_archive_forbidden_test_artifacts": forbidden,
        "source_artifacts_excluded": {
            formulation: manifest["excluded_source_artifacts"]
            for formulation, manifest in archive_manifest.items()
            if isinstance(manifest, dict)
        },
        "test_artifacts_read": [],
        "validation_only_archive_manifest": str(
            (output / "validation_only_archive_manifest.json").resolve()
        ),
    }

    payload = {
        "status": "completed" if audit["strict_pass"] else "failed",
        "per_seed": rows,
        "aggregate": aggregate_rows,
        "deltas": deltas,
        "paired_bootstrap": bootstrap,
        "audit": audit,
    }
    (output / "localcnn_target_contrast.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(output / "per_seed.csv", rows)
    write_csv(output / "aggregate.csv", aggregate_rows)
    write_csv(output / "deltas.csv", deltas)
    bootstrap_rows = []
    for scope, report in [("macro", bootstrap["macro_equal_variable_weight"]), *bootstrap["by_variable"].items()]:
        bootstrap_rows.append({"scope": scope, **report})
    write_csv(output / "bootstrap.csv", bootstrap_rows)

    macro = bootstrap["macro_equal_variable_weight"]
    lines = [
        "# LocalCNN target-formulation contrast (validation only)",
        "",
        "Direct anomaly is the candidate; direct full-field is the matched reference.",
        "",
        f"- Geometric MSE reduction: `{macro['geometric_mse_reduction_fraction']:.6f}`",
        f"- 95% CI: `{macro['ci95_geometric_mse_reduction_fraction']}`",
        f"- p: `{macro['two_sided_bootstrap_p']:.6g}`",
        f"- BH q: `{macro['benjamini_hochberg_q']:.6g}`",
        f"- P(candidate better): `{macro['probability_candidate_better']:.6f}`",
        "- Split: validation; test artifacts: none",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit["strict_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
