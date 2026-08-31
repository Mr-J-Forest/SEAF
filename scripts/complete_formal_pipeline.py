#!/usr/bin/env python3
"""Finish the frozen SEAF formal evidence pipeline.

This runner is intentionally resumable.  It does not train or alter any
checkpoint; it fills missing validation/test reports, aggregates the frozen
36-run matrix, runs the predeclared paired contrasts, and collects a small,
region-stratified set of prediction figures.

The direct full-field evaluations are always executed serially because their
target tensors are the memory-heavy branch under the 90 GiB cgroup limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CAMPAIGN = (
    "91dddd20f847d89a05949e3ddcac87fa39f8de942_"
    "seaf_confirm_all36_30ep_v1_mps"
)
CURRENT_CAMPAIGN = (
    "91626aab8289417c9472939ec2f5e5cf30179e67_"
    "seaf_confirm_remaining_mmap_v1"
)
REFERENCE_EXPERIMENTS = {
    "full",
    "direct_full_field_strict",
    "direct_full_field_tuned",
}
EXPECTED_EXPERIMENTS = REFERENCE_EXPERIMENTS | {
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_path(run_dir: Path, split: str) -> Path | None:
    names = ("validation_results.json", "evaluation_results.json")
    if split == "test":
        names = ("test_results.json", "evaluation_results.json")
    for name in names:
        path = run_dir / name
        if path.is_file():
            return path
    return None


def _valid_evaluation(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    # These are the fields consumed by the formal aggregator and the paired
    # origin bootstrap.  A JSON file alone is not sufficient evidence.
    return all(
        key in payload and isinstance(payload[key], dict)
        for key in ("physical_report", "baseline_comparison", "stratified_reports")
    )


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["TSC_TORCH_NUM_THREADS"] = "2"
    environment["TSC_TORCH_NUM_INTEROP_THREADS"] = "1"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "2"
    return environment


def _job_map(campaign_root: Path) -> dict[str, dict[str, Any]]:
    state_path = campaign_root / "experiment_queue_state.json"
    if not state_path.is_file():
        return {}
    state = _read_json(state_path)
    output: dict[str, dict[str, Any]] = {}
    for job in state.get("jobs", []):
        if not isinstance(job, dict):
            continue
        result_dir = job.get("result_dir")
        if result_dir:
            output[str(Path(str(result_dir)).resolve())] = job
    return output


def _resolve_config(job: dict[str, Any], run_dir: Path) -> Path:
    value = job.get("config")
    if value:
        path = Path(str(value))
        candidate = path if path.is_absolute() else PROJECT_ROOT / path
        if candidate.is_file():
            return candidate.resolve()
    local_config = run_dir / "config.json"
    if local_config.is_file():
        return local_config.resolve()
    raise FileNotFoundError(f"no config for {run_dir}")


def _formal_tasks(campaign_root: Path, label: str) -> list[dict[str, Any]]:
    jobs = _job_map(campaign_root)
    stage_root = campaign_root / "confirm_validation"
    tasks: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in stage_root.glob("*/seed_*") if path.is_dir()):
        experiment = run_dir.parent.name
        if experiment not in EXPECTED_EXPERIMENTS:
            continue
        summary_path = run_dir / "run_summary.json"
        checkpoint_path = run_dir / "best_model.pth"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            continue
        summary = _read_json(summary_path)
        if summary.get("status") != "completed":
            continue
        if int(summary.get("completed_epochs", 0) or 0) < 30:
            continue
        job = jobs.get(str(run_dir.resolve()), {})
        overrides = dict(job.get("overrides") or {})
        overrides.update(
            {
                "num_workers": 1,
                "persistent_workers": False,
                "prefetch_factor": 1,
                "compile_model": False,
                "post_training_evaluation": "validation",
            }
        )
        evaluation = _evaluation_path(run_dir, "validation")
        tasks.append(
            {
                "campaign": label,
                "experiment": experiment,
                "seed": int(job.get("seed", run_dir.name.removeprefix("seed_"))),
                "run_id": str(
                    job.get("run_id")
                    or f"confirm_validation__{experiment}_seed{run_dir.name.removeprefix('seed_')}"
                ),
                "run_dir": str(run_dir.resolve()),
                "config": str(_resolve_config(job, run_dir)),
                "overrides": overrides,
                "evaluation_path": str(evaluation) if evaluation else None,
                "existing": _valid_evaluation(evaluation),
                "heavy": experiment in {"direct_full_field_strict", "direct_full_field_tuned"},
                "checkpoint_sha256": _sha256(checkpoint_path),
            }
        )
    return sorted(tasks, key=lambda item: (not item["heavy"], item["run_id"]))


def _run_command(
    command: list[str],
    log_path: Path,
    *,
    environment: dict[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def _validation_command(task: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "eval_best_only.py"),
        "--config",
        task["config"],
        "--result_dir",
        task["run_dir"],
        "--overrides_json",
        json.dumps(task["overrides"], sort_keys=True, separators=(",", ":")),
        "--split",
        "validation",
    ]


def complete_reference_validation(
    reference_root: Path,
    output_root: Path,
    environment: dict[str, str],
) -> bool:
    tasks = _formal_tasks(reference_root, "reference")
    expected = {(experiment, seed) for experiment in REFERENCE_EXPERIMENTS for seed in EXPECTED_SEEDS}
    found = {(task["experiment"], task["seed"]) for task in tasks}
    if found != expected:
        raise RuntimeError(f"reference task set mismatch: missing={sorted(expected - found)} extra={sorted(found - expected)}")

    status_path = output_root / "reference_validation_status.json"
    records: list[dict[str, Any]] = []
    _write_json(
        status_path,
        {
            "status": "running",
            "stage": "confirm_validation",
            "total": len(tasks),
            "completed": sum(task["existing"] for task in tasks),
            "active": None,
            "updated_at_utc": _now(),
            "records": records,
        },
    )
    for index, task in enumerate(tasks, start=1):
        evaluation = _evaluation_path(Path(task["run_dir"]), "validation")
        if _valid_evaluation(evaluation):
            record = {
                "run_id": task["run_id"],
                "status": "existing",
                "returncode": 0,
                "evaluation_path": str(evaluation),
                "evaluation_sha256": _sha256(evaluation),
            }
            records.append(record)
            _write_json(
                status_path,
                {
                    "status": "running",
                    "stage": "confirm_validation",
                    "total": len(tasks),
                    "completed": len(records),
                    "active": None,
                    "updated_at_utc": _now(),
                    "records": records,
                },
            )
            continue

        log_path = output_root / "logs" / f"{task['run_id'].replace('/', '_')}.log"
        _write_json(
            status_path,
            {
                "status": "running",
                "stage": "confirm_validation",
                "total": len(tasks),
                "completed": len(records),
                "active": task["run_id"],
                "active_index": index,
                "updated_at_utc": _now(),
                "records": records,
            },
        )
        print(f"[reference-validation] {index}/{len(tasks)} {task['run_id']}", flush=True)
        returncode = _run_command(_validation_command(task), log_path, environment=environment)
        evaluation = _evaluation_path(Path(task["run_dir"]), "validation")
        ok = returncode == 0 and _valid_evaluation(evaluation)
        record = {
            "run_id": task["run_id"],
            "status": "completed" if ok else "failed",
            "returncode": returncode,
            "evaluation_path": str(evaluation) if evaluation else None,
            "evaluation_sha256": _sha256(evaluation) if ok and evaluation else None,
            "log": str(log_path),
        }
        records.append(record)
        _write_json(
            status_path,
            {
                "status": "running" if ok else "failed",
                "stage": "confirm_validation",
                "total": len(tasks),
                "completed": sum(item["status"] in {"existing", "completed"} for item in records),
                "active": None,
                "updated_at_utc": _now(),
                "records": records,
            },
        )
        if not ok:
            return False

    manifest = {
        "status": "completed",
        "stage": "confirm_validation",
        "campaign_root": str(reference_root.resolve()),
        "run_count": len(records),
        "records": records,
        "completed_at_utc": _now(),
    }
    _write_json(output_root / "reference_validation_manifest.json", manifest)
    _write_json(output_root / "reference_validation_status.json", manifest)
    return True


def _run_aggregation(
    reference_root: Path,
    current_root: Path,
    split: str,
    output: Path,
    log_path: Path,
    environment: dict[str, str],
) -> bool:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "aggregate_formal_campaign.py"),
        "--reference-root",
        str(reference_root),
        "--current-root",
        str(current_root),
        "--split",
        split,
        "--stage",
        "confirm_validation",
        "--output",
        str(output),
    ]
    return _run_command(command, log_path, environment=environment) == 0 and (
        output / "audit.json"
    ).is_file()


def _build_union(reference_root: Path, current_root: Path) -> Path:
    union_root = Path("/tmp/seaf_formal_validation_union_final")
    stage_root = union_root / "confirm_validation"
    for source_root in (reference_root, current_root):
        for source in (source_root / "confirm_validation").glob("*/seed_*"):
            if not source.is_dir():
                continue
            destination = stage_root / source.parent.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink() or destination.exists():
                continue
            destination.symlink_to(source.resolve(), target_is_directory=True)
    return union_root


def _run_contrasts(
    reference_root: Path,
    current_root: Path,
    output: Path,
    log_path: Path,
    environment: dict[str, str],
) -> bool:
    union_root = _build_union(reference_root, current_root)
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "compare_ablation_contrasts.py"),
        "--results-root",
        str(union_root),
        "--stage",
        "confirm_validation",
        "--contrasts",
        "configs/oras5_full_contrasts.json",
        "--output",
        str(output),
        "--strict",
    ]
    return _run_command(command, log_path, environment=environment) == 0 and output.is_file()


def _run_test_evaluation(
    reference_root: Path,
    current_root: Path,
    output: Path,
    log_path: Path,
    environment: dict[str, str],
) -> bool:
    manifest = output / "formal_test_manifest.json"
    if manifest.is_file():
        try:
            payload = _read_json(manifest)
            if payload.get("status") == "completed" and not payload.get("failures"):
                return True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "evaluate_formal_test.py"),
        "--reference-root",
        str(reference_root),
        "--current-root",
        str(current_root),
        "--stage",
        "confirm_validation",
        "--split",
        "test",
        "--output",
        str(output),
        "--light-workers",
        "2",
    ]
    return _run_command(command, log_path, environment=environment) == 0 and manifest.is_file()


def _select_samples(environment: dict[str, str], log_path: Path) -> list[int]:
    selector = r'''
import json
from data_loader import OceanDataset
from config import DEFAULT_CONFIG, load_config, merge_configs
cfg = DEFAULT_CONFIG.copy()
cfg = merge_configs(load_config("configs/experiments/oras5_seaf.json"), cfg)
train = OceanDataset(cfg["data_path"], cfg, mode="train")
test = train.temporal_split_view("test")
first_by_region = {}
for index, sequence in enumerate(test.sequences):
    region = int(sequence[1])
    first_by_region.setdefault(region, index)
regions = sorted(first_by_region)
chosen = [regions[0], regions[len(regions) // 2], regions[-1]]
indices = [first_by_region[region] for region in chosen]
print(json.dumps({"regions": chosen, "sample_indices": indices}))
try:
    train.dataset.close()
except Exception:
    pass
'''
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, "-c", selector],
        cwd=str(PROJECT_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(completed.stdout)
        handle.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"sample selection failed; see {log_path}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"sample selection returned no JSON; see {log_path}")
    payload = json.loads(lines[-1])
    indices = [int(value) for value in payload["sample_indices"]]
    if len(indices) != 3 or len(set(indices)) != 3:
        raise RuntimeError(f"sample selection did not return three distinct indices: {payload}")
    return indices


def _collect_evidence(
    reference_root: Path,
    output_root: Path,
    environment: dict[str, str],
) -> bool:
    main_dir = reference_root / "confirm_validation" / "full" / "seed_42"
    strict_dir = reference_root / "confirm_validation" / "direct_full_field_strict" / "seed_42"
    tuned_dir = reference_root / "confirm_validation" / "direct_full_field_tuned" / "seed_42"
    if not all(path.is_dir() for path in (main_dir, strict_dir, tuned_dir)):
        raise FileNotFoundError("prediction evidence checkpoints are incomplete")
    indices = _select_samples(environment, output_root / "sample_selection.log")
    records = []
    comparisons = [
        ("strict", strict_dir, indices),
        ("tuned", tuned_dir, [indices[1]]),
    ]
    for comparison_name, comparator_dir, selected_indices in comparisons:
        for sample_index in selected_indices:
            destination = output_root / f"sample_{sample_index}_{comparison_name}"
            summary_path = destination / "run_summary.json"
            if summary_path.is_file() and (destination / "_SUCCESS").is_file():
                records.append(
                    {
                        "comparison": comparison_name,
                        "sample_index": sample_index,
                        "output_dir": str(destination),
                        "status": "existing",
                    }
                )
                continue
            command = [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "scripts" / "collect_prediction_evidence.py"),
                "--main_model_dir",
                str(main_dir),
                "--direct_full_field_dir",
                str(comparator_dir),
                "--output_dir",
                str(destination),
                "--sample_index",
                str(sample_index),
                "--main_label",
                "SEAF",
                "--direct_label",
                "DirectFullField",
                "--depth_indices",
                "0",
                "10",
                "19",
                "--leads",
                "0",
                "2",
                "4",
            ]
            log_path = output_root / "logs" / f"sample_{sample_index}_{comparison_name}.log"
            print(f"[prediction-evidence] {comparison_name} sample={sample_index}", flush=True)
            returncode = _run_command(command, log_path, environment=environment)
            ok = returncode == 0 and (destination / "_SUCCESS").is_file()
            records.append(
                {
                    "comparison": comparison_name,
                    "sample_index": sample_index,
                    "output_dir": str(destination),
                    "status": "completed" if ok else "failed",
                    "returncode": returncode,
                    "log": str(log_path),
                }
            )
            if not ok:
                _write_json(
                    output_root / "prediction_evidence_manifest.json",
                    {"status": "failed", "records": records, "updated_at_utc": _now()},
                )
                return False
    _write_json(
        output_root / "prediction_evidence_manifest.json",
        {
            "status": "completed",
            "sample_indices": indices,
            "comparisons": ["full_vs_direct_full_field_strict", "full_vs_direct_full_field_tuned"],
            "records": records,
            "completed_at_utc": _now(),
        },
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="project root; normally inferred from this script",
    )
    args = parser.parse_args()
    del args

    reference_root = PROJECT_ROOT / "outputs" / "results" / "campaigns" / REFERENCE_CAMPAIGN
    current_root = PROJECT_ROOT / "outputs" / "results" / "campaigns" / CURRENT_CAMPAIGN
    formal_root = PROJECT_ROOT / "outputs" / "formal_results"
    validation_root = formal_root / "validation"
    test_eval_root = formal_root / "test_eval"
    test_root = formal_root / "test"
    prediction_root = PROJECT_ROOT / "outputs" / "prediction_evidence" / "formal_test"
    logs = formal_root / "pipeline_logs"
    environment = _environment()
    status_path = formal_root / "pipeline_status.json"

    def status(stage: str, state: str, **extra: Any) -> None:
        _write_json(
            status_path,
            {
                "status": state,
                "stage": stage,
                "updated_at_utc": _now(),
                "reference_campaign": REFERENCE_CAMPAIGN,
                "current_campaign": CURRENT_CAMPAIGN,
                **extra,
            },
        )

    try:
        status("reference_validation", "running")
        if not complete_reference_validation(
            reference_root,
            formal_root / "reference_validation_safe",
            environment,
        ):
            status("reference_validation", "failed")
            return 1

        status("validation_aggregation", "running")
        if not _run_aggregation(
            reference_root,
            current_root,
            "validation",
            validation_root,
            logs / "aggregate_validation.log",
            environment,
        ):
            status("validation_aggregation", "failed")
            return 1

        status("paired_ablation_contrasts", "running")
        if not _run_contrasts(
            reference_root,
            current_root,
            validation_root / "ablation_contrasts.json",
            logs / "ablation_contrasts.log",
            environment,
        ):
            status("paired_ablation_contrasts", "failed")
            return 1

        status("test_evaluation", "running")
        if not _run_test_evaluation(
            reference_root,
            current_root,
            test_eval_root,
            logs / "test_evaluation.log",
            environment,
        ):
            status("test_evaluation", "failed")
            return 1

        status("test_aggregation", "running")
        if not _run_aggregation(
            reference_root,
            current_root,
            "test",
            test_root,
            logs / "aggregate_test.log",
            environment,
        ):
            status("test_aggregation", "failed")
            return 1

        status("prediction_evidence", "running")
        if not _collect_evidence(reference_root, prediction_root, environment):
            status("prediction_evidence", "failed")
            return 1

        status(
            "complete",
            "completed",
            validation_output=str(validation_root),
            test_output=str(test_root),
            prediction_output=str(prediction_root),
            completed_at_utc=_now(),
        )
        print("[formal-pipeline] completed", flush=True)
        return 0
    except Exception as exc:  # keep a machine-readable failure record for resume/debugging
        status("exception", "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
