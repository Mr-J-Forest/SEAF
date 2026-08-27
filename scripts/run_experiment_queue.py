#!/usr/bin/env python3
"""Idempotent sequential experiment runner for a single-GPU server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime


def atomic_json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def read_source_state(project_root: Path) -> dict:
    path = project_root / "source_state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def linux_process_tree_rss_bytes(
    root_pid: int,
    proc_root: Path = Path('/proc'),
) -> int | None:
    """Sum resident memory for a Linux process and all current descendants."""
    pending = [int(root_pid)]
    visited = set()
    total_kib = 0
    found = False
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        process_dir = proc_root / str(pid)
        try:
            status = (process_dir / 'status').read_text(encoding='utf-8')
        except OSError:
            continue
        for line in status.splitlines():
            if line.startswith('VmRSS:'):
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        total_kib += int(fields[1])
                        found = True
                    except ValueError:
                        pass
                break
        try:
            children = (
                process_dir / 'task' / str(pid) / 'children'
            ).read_text(encoding='utf-8').split()
            pending.extend(int(child) for child in children)
        except (OSError, ValueError):
            pass
    return total_kib * 1024 if found else None


def is_complete(run_dir: Path, expected_source_hash: str | None = None) -> bool:
    if not (run_dir / "_SUCCESS").is_file():
        return False
    summary = run_dir / "run_summary.json"
    if not summary.is_file() or not (run_dir / "config.json").is_file():
        return False
    try:
        with summary.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") != "completed":
            return False
        evaluation_file = payload.get("evaluation_file")
        if evaluation_file and not (run_dir / evaluation_file).is_file():
            return False
        if not evaluation_file and payload.get("evaluation_scope") != "none":
            return False
        if expected_source_hash and payload.get("training_source_hash") != expected_source_hash:
            raise RuntimeError(
                f"completed run has stale source hash: {run_dir} "
                f"({payload.get('training_source_hash')} != {expected_source_hash})"
            )
        return True
    except (OSError, json.JSONDecodeError):
        return False


def resolve_stage_experiments(matrix: dict, stage: str, stack: tuple[str, ...] = ()) -> list[dict]:
    """Resolve a concrete list or a declarative stage derived from another stage."""
    if stage in stack:
        raise ValueError(f"stage inheritance cycle: {' -> '.join((*stack, stage))}")
    if stage not in matrix or stage.startswith('_'):
        raise KeyError(f'unknown stage: {stage}')
    specification = matrix[stage]
    if isinstance(specification, list):
        return [dict(experiment) for experiment in specification]
    if not isinstance(specification, dict) or not specification.get('from_stage'):
        raise TypeError(f'stage {stage!r} must be a list or contain from_stage')
    base = resolve_stage_experiments(
        matrix, str(specification['from_stage']), (*stack, stage)
    )
    output = []
    for experiment in base:
        resolved = dict(experiment)
        if 'seeds' in specification:
            resolved['seeds'] = list(specification['seeds'])
        inherited_overrides = dict(resolved.get('overrides', {}))
        inherited_overrides.update(specification.get('overrides', {}))
        if inherited_overrides:
            resolved['overrides'] = inherited_overrides
        output.append(resolved)
    return output


def expand_matrix(matrix: dict, stages: list[str], only: set[str] | None) -> list[dict]:
    jobs = []
    stage_overrides_map = matrix.get("_stage_overrides", {})
    for stage in stages:
        stage_overrides = dict(stage_overrides_map.get(stage, {}))
        for experiment in resolve_stage_experiments(matrix, stage):
            name = experiment["name"]
            if only and name not in only:
                continue
            overrides = {**stage_overrides, **dict(experiment.get("overrides", {}))}
            for seed in experiment.get("seeds", [42]):
                jobs.append({
                    "stage": stage,
                    "name": name,
                    "config": experiment["config"],
                    "seed": int(seed),
                    "overrides": overrides,
                })
    return jobs


def main(
    argv: list[str] | None = None,
    project_root: Path | str | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/oras5_recent_baseline_matrix.json")
    parser.add_argument(
        "--stage", action="append", dest="stages", required=True,
        help="explicit experiment stage; repeat to run more than one",
    )
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--campaign",
        default=None,
        help="immutable campaign namespace; strongly recommended for scientific runs",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="maximum number of independent training jobs to run concurrently",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_parallel < 1:
        raise ValueError("max-parallel must be at least 1")

    project_root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    matrix_path = (project_root / args.matrix).resolve()
    with matrix_path.open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)

    stages = args.stages
    only = set(args.only) if args.only else None
    jobs = expand_matrix(matrix, stages, only)
    source_state = read_source_state(project_root)
    source_hash = source_state.get("training_source_hash")
    if args.campaign:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", args.campaign):
            raise ValueError("campaign may contain only letters, numbers, dot, underscore, and hyphen")
        if not source_hash:
            raise RuntimeError(
                "campaign runs require source_state.json with source_hash; sync the project first"
            )
        result_root = project_root / "outputs" / "results" / "campaigns" / args.campaign
        state_path = result_root / "experiment_queue_state.json"
        manifest_path = result_root / "campaign_manifest.json"
        matrix_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
        manifest = {
            "campaign": args.campaign,
            "source_state": source_state,
            "matrix_path": str(matrix_path),
            "matrix_sha256": matrix_hash,
            "stages": stages,
            "only": sorted(only) if only else None,
            "max_parallel": args.max_parallel,
        }
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("source_state", {}).get("training_source_hash") != source_hash:
                raise RuntimeError(
                    f"campaign {args.campaign!r} belongs to a different source hash"
                )
            if existing.get("matrix_sha256") != matrix_hash:
                raise RuntimeError(
                    f"campaign {args.campaign!r} belongs to a different experiment matrix"
                )
            if existing.get("stages") != stages:
                raise RuntimeError(
                    f"campaign {args.campaign!r} belongs to stages "
                    f"{existing.get('stages')}, not {stages}"
                )
            if existing.get("only") != (sorted(only) if only else None):
                raise RuntimeError(
                    f"campaign {args.campaign!r} belongs to experiment selection "
                    f"{existing.get('only')}, not {sorted(only) if only else None}"
                )
            if int(existing.get("max_parallel", 1)) != args.max_parallel:
                raise RuntimeError(
                    f"campaign {args.campaign!r} belongs to max_parallel "
                    f"{existing.get('max_parallel', 1)}, not {args.max_parallel}"
                )
        else:
            atomic_json_dump(manifest_path, manifest)
    else:
        state_path = project_root / "outputs" / "experiment_queue_state.json"
        result_root = project_root / "outputs" / "results" / "matrix"
    log_dir = (
        project_root / "run_logs" / args.campaign
        if args.campaign else project_root / "run_logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    # All queues rooted in the same checkout share the host-memory-heavy
    # evaluation phase, even when they belong to different campaigns.
    evaluation_lock_path = project_root / "outputs" / ".post_training_evaluation.lock"

    state = {
        "matrix": str(matrix_path),
        "max_parallel": args.max_parallel,
        "post_training_evaluation_lock": str(evaluation_lock_path),
        "updated_at": datetime.now().isoformat(),
        "jobs": [],
    }
    failures = 0
    pending = []

    for position, job in enumerate(jobs, start=1):
        run_id = f"{job['stage']}__{job['name']}_seed{job['seed']}"
        run_dir = result_root / job["stage"] / job["name"] / f"seed_{job['seed']}"
        log_path = log_dir / f"{run_id}.log"
        record = dict(job, run_id=run_id, result_dir=str(run_dir), log=str(log_path))

        if is_complete(run_dir, expected_source_hash=source_hash if args.campaign else None):
            record["status"] = "skipped_completed"
            state["jobs"].append(record)
            state["updated_at"] = datetime.now().isoformat()
            atomic_json_dump(state_path, state)
            print(f"[{position}/{len(jobs)}] SKIP completed {run_id}", flush=True)
            continue

        config_path = (project_root / job["config"]).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)

        command = [
            args.python,
            "-u",
            str(project_root / "train.py"),
            "--config",
            str(config_path),
            "--seed",
            str(job["seed"]),
            "--note",
            run_id,
        ]
        if job.get("overrides"):
            command.extend([
                "--overrides_json",
                json.dumps(job["overrides"], sort_keys=True, separators=(",", ":")),
            ])
        checkpoint = run_dir / "latest_checkpoint.pth"
        if checkpoint.is_file():
            command.extend(["--resume_dir", str(run_dir)])
            record["mode"] = "resume"
        else:
            command.extend(["--result_dir", str(run_dir)])
            record["mode"] = "fresh"

        record["command"] = command
        if args.dry_run:
            record["status"] = "dry_run"
            state["jobs"].append(record)
            print(" ".join(command), flush=True)
            continue

        record["status"] = "pending"
        state["jobs"].append(record)
        pending.append({
            "position": position,
            "record": record,
            "run_dir": run_dir,
            "log_path": log_path,
            "command": command,
        })

    state["updated_at"] = datetime.now().isoformat()
    atomic_json_dump(state_path, state)
    if args.dry_run:
        return 0

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TSC_POST_EVAL_LOCK"] = str(evaluation_lock_path)
    active = []
    last_state_write = 0.0

    try:
        while pending or active:
            while pending and len(active) < args.max_parallel:
                item = pending.pop(0)
                record = item["record"]
                log_handle = item["log_path"].open("a", encoding="utf-8")
                record["status"] = "running"
                record["started_at"] = datetime.now().isoformat()
                print(
                    f"[{item['position']}/{len(jobs)}] START "
                    f"{record['run_id']} ({record['mode']})",
                    flush=True,
                )
                log_handle.write(f"\n=== queue start {datetime.now().isoformat()} ===\n")
                log_handle.flush()
                process = subprocess.Popen(
                    item["command"],
                    cwd=project_root,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                record["pid"] = process.pid
                item.update({
                    "process": process,
                    "log_handle": log_handle,
                    "peak_tree_rss": None,
                })
                active.append(item)

            state["updated_at"] = datetime.now().isoformat()
            atomic_json_dump(state_path, state)
            finished = []
            abort_returncode = None

            for item in active:
                process = item["process"]
                record = item["record"]
                current_tree_rss = linux_process_tree_rss_bytes(process.pid)
                if current_tree_rss is not None:
                    item["peak_tree_rss"] = max(
                        item["peak_tree_rss"] or 0,
                        current_tree_rss,
                    )
                    record["peak_process_tree_rss_bytes"] = item["peak_tree_rss"]

                returncode = process.poll()
                if returncode is None:
                    continue

                item["log_handle"].close()
                record["returncode"] = returncode
                record["finished_at"] = datetime.now().isoformat()
                if returncode == 0 and is_complete(
                    item["run_dir"],
                    expected_source_hash=source_hash if args.campaign else None,
                ):
                    record["status"] = "completed"
                    print(
                        f"[{item['position']}/{len(jobs)}] DONE {record['run_id']}",
                        flush=True,
                    )
                else:
                    record["status"] = "failed"
                    failures += 1
                    print(
                        f"[{item['position']}/{len(jobs)}] FAILED {record['run_id']}; "
                        f"see {item['log_path']}",
                        flush=True,
                    )
                    if not args.continue_on_error:
                        abort_returncode = returncode or 1
                finished.append(item)

            for item in finished:
                active.remove(item)

            state["updated_at"] = datetime.now().isoformat()
            atomic_json_dump(state_path, state)

            if abort_returncode is not None:
                for item in active:
                    item["process"].terminate()
                for item in active:
                    process = item["process"]
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    item["log_handle"].close()
                    item["record"]["returncode"] = process.returncode
                    item["record"]["finished_at"] = datetime.now().isoformat()
                    item["record"]["status"] = "cancelled_after_failure"
                state["updated_at"] = datetime.now().isoformat()
                atomic_json_dump(state_path, state)
                return abort_returncode

            if active and not finished:
                now = time.monotonic()
                if now - last_state_write >= 5.0:
                    state["updated_at"] = datetime.now().isoformat()
                    atomic_json_dump(state_path, state)
                    last_state_write = now
                time.sleep(1.0)
    except BaseException:
        for item in active:
            process = item["process"]
            if process.poll() is None:
                process.terminate()
            item["log_handle"].close()
        raise

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
