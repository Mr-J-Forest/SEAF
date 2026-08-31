#!/usr/bin/env python3
"""Run a bounded, resource-measured parallel evaluation of frozen checkpoints.

The normal confirmation queue evaluates one checkpoint at a time because the
evaluation path materializes forecasts and baseline reports.  This helper is
deliberately separate: it first supports a two-lane probe whose outputs are
written to temporary lane directories and compared with existing serial
results.  Only a passing probe report can authorize the remaining evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TIMING_KEYS = {"evaluation_timings_seconds"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_filename(split: str) -> str:
    return "evaluation_results.json" if split == "test" else "validation_results.json"


def _campaign_state(campaign_root: Path) -> tuple[Path, dict]:
    state_path = campaign_root / "experiment_queue_state.json"
    state = _read_json(state_path)
    if not isinstance(state, dict) or not isinstance(state.get("jobs"), list):
        raise ValueError(f"invalid campaign state: {state_path}")
    return state_path, state


def _finalize_deferred_evaluation(run_dir: Path, split: str) -> None:
    """Attach a completed evaluation-only report to its training summary."""
    evaluation_path = run_dir / _evaluation_filename(split)
    summary_path = run_dir / "run_summary.json"
    evaluation = _read_json(evaluation_path)
    summary = _read_json(summary_path)
    summary["evaluation_scope"] = split
    summary["evaluation_file"] = evaluation_path.name
    summary["deferred_evaluation"] = True
    summary["deferred_evaluation_completed_at"] = datetime.now().isoformat()
    summary["primary_metrics"] = {
        "evaluation_loss": evaluation.get("evaluation_loss"),
        "evaluation_data_loss": evaluation.get("evaluation_data_loss"),
        "rmse_TEMP": evaluation.get("physical_rmse_TEMP"),
        "rmse_SALT": evaluation.get("physical_rmse_SALT"),
    }
    _atomic_json_dump(summary_path, summary)


def _recover_finished_records(state: dict, state_path: Path, split: str) -> int:
    """Recover results written before a dispatcher was paused or exited."""
    recovered = 0
    evaluation_name = _evaluation_filename(split)
    for record in state["jobs"]:
        if record.get("deferred_evaluation_scope") != split:
            continue
        result_dir = Path(str(record.get("result_dir", "")))
        evaluation_path = result_dir / evaluation_name
        if not evaluation_path.is_file():
            continue
        if record.get("evaluation_status") == "completed" and record.get("status") == "completed":
            continue
        try:
            _finalize_deferred_evaluation(result_dir, split)
            _read_json(evaluation_path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[RECOVER] skip {record.get('run_id')}: {exc}", flush=True)
            continue
        record["status"] = "completed"
        record["evaluation_status"] = "completed"
        record["evaluation_returncode"] = 0
        record["evaluation_finished_at"] = _now()
        record["evaluation_recovered"] = True
        recovered += 1
    if recovered:
        state["updated_at"] = datetime.now().isoformat()
        _atomic_json_dump(state_path, state)
    return recovered


def _cgroup_memory() -> tuple[int | None, dict[str, int]]:
    current_path = Path("/sys/fs/cgroup/memory.current")
    events_path = Path("/sys/fs/cgroup/memory.events")
    try:
        current = int(current_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        current = None
    events: dict[str, int] = {}
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            key, value = line.split()
            events[key] = int(value)
    except (OSError, ValueError):
        pass
    return current, events


def _gpu_sample() -> dict[str, float | None]:
    fields = "utilization.gpu,memory.used,power.draw,temperature.gpu"
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        values = [item.strip() for item in output.splitlines()[0].split(",")]
        names = ("utilization_gpu_pct", "memory_used_mib", "power_w", "temperature_c")
        return {name: float(value) for name, value in zip(names, values)}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {
            "utilization_gpu_pct": None,
            "memory_used_mib": None,
            "power_w": None,
            "temperature_c": None,
        }


def _json_diff(left: Any, right: Any, path: str = "", *, atol: float = 1e-6, rtol: float = 1e-6) -> dict:
    """Compare evaluation JSON while ignoring wall-clock timing fields."""
    mismatches: list[dict[str, Any]] = []
    max_abs_diff = 0.0

    def visit(a: Any, b: Any, current: str) -> None:
        nonlocal max_abs_diff
        if current.rsplit(".", 1)[-1] in _TIMING_KEYS:
            return
        if isinstance(a, bool) or isinstance(b, bool):
            if a != b:
                mismatches.append({"path": current, "left": a, "right": b})
            return
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = abs(float(a) - float(b))
            max_abs_diff = max(max_abs_diff, delta)
            allowed = atol + rtol * max(abs(float(a)), abs(float(b)))
            if delta > allowed:
                mismatches.append({"path": current, "left": a, "right": b, "abs_diff": delta})
            return
        if type(a) is not type(b):
            mismatches.append({"path": current, "left_type": type(a).__name__, "right_type": type(b).__name__})
            return
        if isinstance(a, dict):
            keys = sorted(set(a) | set(b))
            for key in keys:
                if key not in a or key not in b:
                    mismatches.append({"path": f"{current}.{key}", "missing": key not in a})
                else:
                    visit(a[key], b[key], f"{current}.{key}")
            return
        if isinstance(a, list):
            if len(a) != len(b):
                mismatches.append({"path": current, "left_length": len(a), "right_length": len(b)})
                return
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                visit(item_a, item_b, f"{current}[{index}]")
            return
        if a != b:
            mismatches.append({"path": current, "left": a, "right": b})

    visit(left, right, path or "$")
    return {"match": not mismatches, "max_abs_diff": max_abs_diff, "mismatches": mismatches[:20]}


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _job_overrides(record: dict, loader_workers: int) -> dict:
    overrides = dict(record.get("overrides") or {})
    overrides.update({
        "num_workers": int(loader_workers),
        "persistent_workers": False,
        "prefetch_factor": 1,
        "compile_model": False,
    })
    return overrides


def _select_probe_jobs(state: dict, split: str, only: list[str] | None) -> list[dict]:
    evaluation_name = _evaluation_filename(split)
    candidates = []
    for record in state["jobs"]:
        result_dir = Path(str(record.get("result_dir", "")))
        if record.get("stage") != "confirm_validation":
            continue
        if record.get("status") != "completed" or record.get("evaluation_status") != "completed":
            continue
        if not (result_dir / evaluation_name).is_file() or not (result_dir / "best_model.pth").is_file():
            continue
        if only and record.get("run_id") not in set(only):
            continue
        candidates.append(record)
    candidates.sort(key=lambda item: str(item.get("run_id")))
    if only and len(candidates) != len(only):
        found = {str(item.get("run_id")) for item in candidates}
        raise RuntimeError(f"probe jobs missing or not fully evaluated: {sorted(set(only) - found)}")
    if len(candidates) < 2:
        raise RuntimeError("parallel probe requires two completed evaluation results")
    # Prefer two different configurations so the probe is not tied to one model shape.
    selected = [candidates[0]]
    first_prefix = str(candidates[0].get("run_id")).rsplit("_seed", 1)[0]
    selected.extend(item for item in candidates[1:] if str(item.get("run_id")).rsplit("_seed", 1)[0] != first_prefix)
    if len(selected) < 2:
        selected = candidates[:2]
    return selected[:2]


def _select_remaining_jobs(state: dict, split: str) -> list[dict]:
    evaluation_name = _evaluation_filename(split)
    selected = []
    for record in state["jobs"]:
        if record.get("deferred_evaluation_scope") != split:
            continue
        # Jobs trained by the current queue are marked ``completed`` before
        # deferred evaluation starts; jobs completed in an earlier invocation
        # are marked ``pending_evaluation``.  Both states are training-ready.
        if record.get("status") not in {"completed", "pending_evaluation"}:
            continue
        if record.get("evaluation_status") in {"completed", "parallel_running"}:
            continue
        result_dir = Path(str(record.get("result_dir", "")))
        if not (result_dir / "best_model.pth").is_file():
            continue
        if (result_dir / evaluation_name).is_file():
            continue
        selected.append(record)
    selected.sort(key=lambda item: str(item.get("run_id")))
    return selected


def _make_probe_lane(source_dir: Path, lane_dir: Path) -> None:
    lane_dir.mkdir(parents=True, exist_ok=False)
    source_checkpoint = (source_dir / "best_model.pth").resolve()
    target_checkpoint = lane_dir / "best_model.pth"
    try:
        os.symlink(source_checkpoint, target_checkpoint)
    except OSError:
        shutil.copy2(source_checkpoint, target_checkpoint)


def _build_command(record: dict, result_dir: Path, split: str, loader_workers: int) -> list[str]:
    config_path = _resolve_path(str(record["config"]), PROJECT_ROOT)
    return [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "eval_best_only.py"),
        "--config",
        str(config_path),
        "--result_dir",
        str(result_dir),
        "--overrides_json",
        json.dumps(_job_overrides(record, loader_workers), sort_keys=True, separators=(",", ":")),
        "--split",
        split,
    ]


def _run_lanes(
    *,
    tasks: list[dict],
    campaign_root: Path,
    split: str,
    max_workers: int,
    loader_workers: int,
    status_path: Path,
    state: dict | None = None,
    state_path: Path | None = None,
    phase: str,
) -> dict:
    if not tasks:
        return {"records": [], "resource": {"samples": 0}}

    environment = os.environ.copy()
    environment.pop("TSC_POST_EVAL_LOCK", None)
    environment["TSC_TORCH_NUM_THREADS"] = "2"
    environment["TSC_TORCH_NUM_INTEROP_THREADS"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "2"

    pending = list(tasks)
    active: list[dict] = []
    records: list[dict] = []
    resource_samples: list[dict] = []
    peak_memory = 0
    memory_start, events_start = _cgroup_memory()
    started_at = _now()

    def write_status() -> None:
        _atomic_json_dump(status_path, {
            "phase": phase,
            "status": "running",
            "started_at": started_at,
            "updated_at": _now(),
            "total": len(tasks),
            "completed": sum(item.get("returncode") == 0 and item.get("evaluation_exists") for item in records),
            "failed": sum(item.get("returncode") not in (None, 0) for item in records),
            "active": [item["name"] for item in active],
            "pending": [str(item["record"].get("run_id")) for item in pending],
            "peak_cgroup_memory_bytes": peak_memory,
            "records": records,
        })

    while pending or active:
        while pending and len(active) < max_workers:
            task = pending.pop(0)
            record = task["record"]
            result_dir = task["result_dir"]
            result_dir.mkdir(parents=True, exist_ok=True)
            log_path = task["log_path"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = _build_command(record, result_dir, split, loader_workers)
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(f"\n=== parallel evaluation start {_now()} ===\n")
            log_handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active.append({
                "task": task,
                "record": record,
                "name": str(record.get("run_id")),
                "process": process,
                "log_handle": log_handle,
                "started_at": _now(),
                "command": command,
            })
            if state is not None:
                record["evaluation_status"] = "parallel_running"
                record["evaluation_command"] = command
                record["evaluation_started_at"] = active[-1]["started_at"]
                state["updated_at"] = datetime.now().isoformat()
                _atomic_json_dump(state_path, state)
            write_status()

        current_memory, events = _cgroup_memory()
        if current_memory is not None:
            peak_memory = max(peak_memory, current_memory)
        gpu = _gpu_sample()
        sample = {
            "timestamp": _now(),
            "phase": phase,
            "active": [item["name"] for item in active],
            "cgroup_memory_bytes": current_memory,
            "memory_events": events,
            **gpu,
        }
        resource_samples.append(sample)
        if len(resource_samples) % 5 == 0:
            write_status()

        finished = []
        for item in active:
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["log_handle"].close()
            task = item["task"]
            result_path = task["result_dir"] / _evaluation_filename(split)
            result = {
                "name": item["name"],
                "source_result_dir": str(task["source_result_dir"]),
                "result_dir": str(task["result_dir"]),
                "log": str(task["log_path"]),
                "command": item["command"],
                "started_at": item["started_at"],
                "finished_at": _now(),
                "returncode": int(returncode),
                "evaluation_exists": result_path.is_file(),
                "evaluation_path": str(result_path),
                "evaluation_sha256": _sha256(result_path) if result_path.is_file() else None,
            }
            records.append(result)
            if state is not None:
                record = item["record"]
                if returncode == 0 and result_path.is_file():
                    _finalize_deferred_evaluation(task["source_result_dir"], split)
                    record["status"] = "completed"
                    record["evaluation_status"] = "completed"
                    record["evaluation_returncode"] = 0
                    record["evaluation_finished_at"] = result["finished_at"]
                    record["evaluation_sha256"] = result["evaluation_sha256"]
                    record["evaluation_peak_cgroup_memory_bytes"] = peak_memory
                else:
                    record["status"] = "failed"
                    record["evaluation_status"] = "failed"
                    record["evaluation_returncode"] = int(returncode)
                    record["evaluation_finished_at"] = result["finished_at"]
                state["updated_at"] = datetime.now().isoformat()
                _atomic_json_dump(state_path, state)
            finished.append(item)
        for item in finished:
            active.remove(item)
        write_status()
        if active or pending:
            time.sleep(1.0)

    _, events_end = _cgroup_memory()
    resource = {
        "started_at": started_at,
        "finished_at": _now(),
        "sample_count": len(resource_samples),
        "peak_cgroup_memory_bytes": peak_memory,
        "cgroup_memory_start_bytes": memory_start,
        "cgroup_memory_end_bytes": _cgroup_memory()[0],
        "memory_events_start": events_start,
        "memory_events_end": events_end,
        "samples": resource_samples,
    }
    _atomic_json_dump(status_path, {
        "phase": phase,
        "status": "completed",
        "started_at": started_at,
        "updated_at": _now(),
        "total": len(tasks),
        "completed": sum(item.get("returncode") == 0 and item.get("evaluation_exists") for item in records),
        "failed": sum(item.get("returncode") not in (None, 0) for item in records),
        "active": [],
        "pending": [],
        "peak_cgroup_memory_bytes": peak_memory,
        "records": records,
        "resource": resource,
    })
    return {"records": records, "resource": resource}


def _probe(args: argparse.Namespace, campaign_root: Path, state_path: Path, state: dict) -> int:
    jobs = _select_probe_jobs(state, args.split, args.only)
    probe_root = campaign_root / f"parallel_eval_probe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    tasks = []
    for index, record in enumerate(jobs, start=1):
        source_dir = Path(str(record["result_dir"])).resolve()
        lane_dir = probe_root / f"lane_{index}_{record['run_id'].replace('/', '_')}"
        _make_probe_lane(source_dir, lane_dir)
        tasks.append({
            "record": record,
            "source_result_dir": source_dir,
            "result_dir": lane_dir,
            "log_path": probe_root / f"lane_{index}.log",
        })
    status_path = probe_root / "probe_status.json"
    outcome = _run_lanes(
        tasks=tasks,
        campaign_root=campaign_root,
        split=args.split,
        max_workers=2,
        loader_workers=args.loader_workers,
        status_path=status_path,
        phase="probe",
    )

    comparisons = []
    evaluation_name = _evaluation_filename(args.split)
    results_by_name = {result["name"]: result for result in outcome["records"]}
    for task in tasks:
        result = results_by_name[str(task["record"].get("run_id"))]
        baseline_path = task["source_result_dir"] / evaluation_name
        probe_path = task["result_dir"] / evaluation_name
        comparison = _json_diff(_read_json(baseline_path), _read_json(probe_path)) if probe_path.is_file() else {
            "match": False,
            "mismatches": [{"path": "$", "reason": "probe output missing"}],
        }
        comparisons.append({"name": result["name"], "comparison": comparison})

    events_start = outcome["resource"].get("memory_events_start", {})
    events_end = outcome["resource"].get("memory_events_end", {})
    oom_delta = {
        key: int(events_end.get(key, 0)) - int(events_start.get(key, 0))
        for key in set(events_start) | set(events_end)
        if key.startswith("oom") or key == "max"
    }
    peak_bytes = int(outcome["resource"].get("peak_cgroup_memory_bytes") or 0)
    max_bytes = int(args.max_memory_gib * (1024 ** 3))
    output_ok = all(item["returncode"] == 0 and item["evaluation_exists"] for item in outcome["records"])
    metrics_ok = all(item["comparison"]["match"] for item in comparisons)
    oom_ok = all(value == 0 for value in oom_delta.values())
    memory_ok = peak_bytes <= max_bytes
    passed = bool(output_ok and metrics_ok and oom_ok and memory_ok)
    report = {
        "status": "passed" if passed else "failed",
        "mode": "probe",
        "split": args.split,
        "probe_root": str(probe_root),
        "loader_workers": args.loader_workers,
        "max_workers": 2,
        "max_memory_gib": args.max_memory_gib,
        "checks": {
            "outputs": output_ok,
            "metrics_match": metrics_ok,
            "no_oom_events": oom_ok,
            "memory_under_limit": memory_ok,
        },
        "peak_cgroup_memory_bytes": peak_bytes,
        "peak_cgroup_memory_gib": peak_bytes / (1024 ** 3),
        "oom_event_delta": oom_delta,
        "comparisons": comparisons,
        "records": outcome["records"],
        "resource": outcome["resource"],
        "created_at": _now(),
    }
    report_path = probe_root / "parallel_eval_probe_report.json"
    _atomic_json_dump(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "peak_gib": report["peak_cgroup_memory_gib"], "oom_delta": oom_delta}, ensure_ascii=False), flush=True)
    return 0 if passed else 2


def _remaining(args: argparse.Namespace, campaign_root: Path, state_path: Path, state: dict) -> int:
    if not args.probe_report:
        raise RuntimeError("remaining mode requires --probe-report")
    probe_report = _read_json(Path(args.probe_report))
    if probe_report.get("status") != "passed":
        raise RuntimeError(f"refusing parallel remaining evaluation; probe status is {probe_report.get('status')!r}")
    jobs = _select_remaining_jobs(state, args.split)
    if not jobs:
        print("[REMAINING] no pending evaluation jobs", flush=True)
        return 0
    output_root = campaign_root / f"parallel_eval_remaining_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    tasks = []
    for record in jobs:
        tasks.append({
            "record": record,
            "source_result_dir": Path(str(record["result_dir"])).resolve(),
            "result_dir": Path(str(record["result_dir"])).resolve(),
            "log_path": output_root / f"{record['run_id'].replace('/', '_')}.log",
        })
    status_path = output_root / "remaining_status.json"
    outcome = _run_lanes(
        tasks=tasks,
        campaign_root=campaign_root,
        split=args.split,
        max_workers=2,
        loader_workers=args.loader_workers,
        status_path=status_path,
        state=state,
        state_path=state_path,
        phase="remaining",
    )
    failures = [item for item in outcome["records"] if item["returncode"] != 0 or not item["evaluation_exists"]]
    report = {
        "status": "completed" if not failures else "failed",
        "mode": "remaining",
        "split": args.split,
        "run_count": len(outcome["records"]),
        "failure_count": len(failures),
        "failures": failures,
        "records": outcome["records"],
        "resource": outcome["resource"],
        "created_at": _now(),
    }
    report_path = output_root / "parallel_eval_remaining_report.json"
    _atomic_json_dump(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "run_count": len(outcome["records"]), "failures": len(failures)}, ensure_ascii=False), flush=True)
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--mode", choices=("probe", "remaining"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--max-memory-gib", type=float, default=65.0)
    parser.add_argument("--probe-report")
    args = parser.parse_args(argv)
    if args.loader_workers < 0 or args.loader_workers > 1:
        raise ValueError("parallel evaluation probe supports loader-workers 0 or 1")
    campaign_root = Path(args.campaign_root).resolve()
    state_path, state = _campaign_state(campaign_root)
    recovered = _recover_finished_records(state, state_path, args.split)
    if recovered:
        print(f"[RECOVER] finalized {recovered} evaluation result(s)", flush=True)
    if args.mode == "probe":
        return _probe(args, campaign_root, state_path, state)
    return _remaining(args, campaign_root, state_path, state)


if __name__ == "__main__":
    raise SystemExit(main())
