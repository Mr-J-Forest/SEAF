#!/usr/bin/env python3
"""Evaluate the frozen 36-run SEAF matrix on the held-out test split.

The direct full-field variants materialize a larger target tensor than the
anomaly variants.  They are therefore evaluated one at a time, while the
remaining models use two lanes.  The scheduler measures cgroup memory and
OOM counters and writes an auditable manifest without changing training
summaries or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXPERIMENTS = {
    "full", "direct_full_field_strict", "direct_full_field_tuned",
    "no_tendency", "no_external_dynamics", "no_spectral",
    "uniform_ensemble", "single_head", "local_cnn",
    "ofb_fourcastnet_anomaly", "ofb_climax_anomaly", "ofb_swin_anomaly",
}
EXPECTED_SEEDS = {42, 123, 3407}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
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


def _cgroup() -> tuple[int | None, dict[str, int]]:
    current = None
    events: dict[str, int] = {}
    try:
        current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines():
            key, value = line.split()
            events[key] = int(value)
    except (OSError, ValueError):
        pass
    return current, events


def _gpu() -> dict[str, float | None]:
    fields = "utilization.gpu,memory.used,power.draw,temperature.gpu"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        values = [float(item.strip()) for item in output.splitlines()[0].split(",")]
        return dict(zip(("gpu_util_pct", "gpu_memory_mib", "gpu_power_w", "gpu_temp_c"), values))
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {"gpu_util_pct": None, "gpu_memory_mib": None, "gpu_power_w": None, "gpu_temp_c": None}


def _resolve_config(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _tasks(campaign_root: Path, campaign_label: str, stage: str, split: str) -> list[dict[str, Any]]:
    state_path = campaign_root / "experiment_queue_state.json"
    state = _read_json(state_path)
    output = []
    for job in state.get("jobs", []):
        if not isinstance(job, dict) or job.get("stage") != stage:
            continue
        result_dir = Path(str(job.get("result_dir", ""))).resolve()
        summary_path = result_dir / "run_summary.json"
        checkpoint_path = result_dir / "best_model.pth"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            continue
        try:
            summary = _read_json(summary_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if summary.get("status") != "completed" or int(summary.get("completed_epochs", 0) or 0) < 30:
            continue
        config = _resolve_config(str(job.get("config", "")))
        if not config.is_file():
            continue
        experiment = result_dir.parent.name
        seed = int(job.get("seed"))
        run_id = str(job.get("run_id") or f"{stage}__{experiment}_seed{seed}")
        evaluation_path = result_dir / ("evaluation_results.json" if split == "test" else "validation_results.json")
        if evaluation_path.is_file():
            try:
                _read_json(evaluation_path)
                existing = True
            except (OSError, ValueError, json.JSONDecodeError):
                evaluation_path.unlink()
                existing = False
        else:
            existing = False
        overrides = dict(job.get("overrides") or {})
        overrides.update({
            "num_workers": 1,
            "persistent_workers": False,
            "prefetch_factor": 1,
            "compile_model": False,
            "post_training_evaluation": split,
        })
        output.append({
            "campaign": campaign_label,
            "run_id": run_id,
            "experiment": experiment,
            "seed": seed,
            "result_dir": str(result_dir),
            "config": str(config),
            "evaluation_path": str(evaluation_path),
            "overrides": overrides,
            "existing": existing,
            # This is derived from the target space, not from a seed or sample.
            "heavy": experiment in {"direct_full_field_strict", "direct_full_field_tuned"},
        })
    return output


def _command(task: dict[str, Any], split: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "eval_best_only.py"),
        "--config",
        task["config"],
        "--result_dir",
        task["result_dir"],
        "--overrides_json",
        json.dumps(task["overrides"], sort_keys=True, separators=(",", ":")),
        "--split",
        split,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--current-root", required=True)
    parser.add_argument("--stage", default="confirm_validation")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--light-workers", type=int, default=2)
    args = parser.parse_args()
    if args.light_workers < 1 or args.light_workers > 2:
        raise ValueError("light-workers must be 1 or 2")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = _tasks(Path(args.reference_root).resolve(), "reference", args.stage, args.split)
    tasks += _tasks(Path(args.current_root).resolve(), "current", args.stage, args.split)
    by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates = []
    for task in tasks:
        key = (task["experiment"], task["seed"])
        if key in by_pair:
            duplicates.append(task["run_id"])
        else:
            by_pair[key] = task
    tasks = sorted(by_pair.values(), key=lambda item: (not item["heavy"], item["run_id"]))
    expected = {(experiment, seed) for experiment in EXPECTED_EXPERIMENTS for seed in EXPECTED_SEEDS}
    found = set(by_pair)
    missing = sorted(expected - found)
    if missing or duplicates:
        raise RuntimeError(f"formal task set mismatch: missing={missing}, duplicates={duplicates}")

    pending = [task for task in tasks if not task["existing"]]
    records: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if task["existing"]:
            records[task["run_id"]] = {
                "run_id": task["run_id"], "campaign": task["campaign"],
                "result_dir": task["result_dir"], "evaluation_path": task["evaluation_path"],
                "returncode": 0, "evaluation_exists": True, "existing": True,
            }

    environment = os.environ.copy()
    environment["TSC_TORCH_NUM_THREADS"] = "2"
    environment["TSC_TORCH_NUM_INTEROP_THREADS"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "2"

    memory_start, events_start = _cgroup()
    peak_memory = memory_start or 0
    samples: list[dict[str, Any]] = []
    started_at = _now()
    status_path = output / "test_status.json"

    def write_status(active: list[dict[str, Any]], waiting: list[dict[str, Any]]) -> None:
        _write_json(status_path, {
            "status": "running",
            "split": args.split,
            "started_at": started_at,
            "updated_at": _now(),
            "total": len(tasks),
            "completed": sum(bool(item.get("evaluation_exists")) and item.get("returncode") == 0 for item in records.values()),
            "failed": sum(item.get("returncode") not in (0, None) for item in records.values()),
            "active": [item["task"]["run_id"] for item in active],
            "pending": [item["run_id"] for item in waiting],
            "peak_cgroup_memory_bytes": peak_memory,
            "records": list(records.values()),
        })

    def run_group(group: list[dict[str, Any]], max_workers: int, attempt: int) -> None:
        nonlocal peak_memory
        waiting = list(group)
        active: list[dict[str, Any]] = []
        write_status(active, waiting)
        while waiting or active:
            while waiting and len(active) < max_workers:
                task = waiting.pop(0)
                log_path = log_dir / f"{task['run_id'].replace('/', '_')}.attempt{attempt}.log"
                handle = log_path.open("w", encoding="utf-8")
                command = _command(task, args.split)
                handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
                handle.flush()
                process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=environment, stdout=handle, stderr=subprocess.STDOUT)
                active.append({"task": task, "process": process, "handle": handle, "started_at": _now(), "log": str(log_path), "command": command})
                write_status(active, waiting)

            current, events = _cgroup()
            if current is not None:
                peak_memory = max(peak_memory, current)
            samples.append({"timestamp": _now(), "active": [item["task"]["run_id"] for item in active], "cgroup_memory_bytes": current, "memory_events": events, **_gpu()})
            finished = []
            for item in active:
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                item["handle"].close()
                task = item["task"]
                evaluation_path = Path(task["evaluation_path"])
                result = {
                    "run_id": task["run_id"], "campaign": task["campaign"],
                    "experiment": task["experiment"], "seed": task["seed"],
                    "result_dir": task["result_dir"], "evaluation_path": str(evaluation_path),
                    "returncode": int(returncode), "evaluation_exists": evaluation_path.is_file(),
                    "attempt": attempt, "log": item["log"], "started_at": item["started_at"], "finished_at": _now(),
                }
                records[task["run_id"]] = result
                finished.append(item)
            for item in finished:
                active.remove(item)
            write_status(active, waiting)
            if active or waiting:
                time.sleep(1.0)

    heavy = [task for task in pending if task["heavy"]]
    light = [task for task in pending if not task["heavy"]]
    # The target-space distinction is the safety boundary: direct full-field
    # is serial, while the validated lightweight group stays two-way.
    run_group(heavy, 1, 1)
    run_group(light, args.light_workers, 1)
    failed = [task for task in pending if records.get(task["run_id"], {}).get("returncode") != 0 or not records.get(task["run_id"], {}).get("evaluation_exists")]
    if failed:
        run_group(failed, 1, 2)

    _, events_end = _cgroup()
    oom_delta = {key: int(events_end.get(key, 0)) - int(events_start.get(key, 0)) for key in set(events_start) | set(events_end) if key.startswith("oom") or key == "max"}
    final_records = [records[task["run_id"]] for task in tasks]
    failures = [record for record in final_records if record.get("returncode") != 0 or not record.get("evaluation_exists")]
    manifest = {
        "status": "completed" if not failures else "failed",
        "split": args.split,
        "stage": args.stage,
        "reference_root": str(Path(args.reference_root).resolve()),
        "current_root": str(Path(args.current_root).resolve()),
        "run_count": len(final_records),
        "records": final_records,
        "failures": failures,
        "resource": {
            "started_at": started_at, "finished_at": _now(), "sample_count": len(samples),
            "peak_cgroup_memory_bytes": peak_memory,
            "peak_cgroup_memory_gib": peak_memory / (1024 ** 3),
            "memory_events_start": events_start, "memory_events_end": events_end,
            "oom_event_delta": oom_delta, "samples": samples,
        },
        "protocol": {
            "direct_full_field_serial": True,
            "light_workers": args.light_workers,
            "loader_workers": 1,
            "checkpoint_policy": "frozen best_model.pth; no training resumed",
        },
        "created_at": _now(),
    }
    _write_json(output / "formal_test_manifest.json", manifest)
    _write_json(status_path, {**manifest, "status_path": str(status_path)})
    print(json.dumps({"status": manifest["status"], "run_count": len(final_records), "failures": len(failures), "peak_gib": manifest["resource"]["peak_cgroup_memory_gib"], "oom_delta": oom_delta}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
