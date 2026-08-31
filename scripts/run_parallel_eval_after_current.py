#!/usr/bin/env python3
"""Hand the deferred evaluation phase from the serial queue to the probe.

The caller must pause the queue dispatcher while its current evaluation child
continues.  This coordinator waits for that child to exit, verifies that no
evaluation worker remains, removes only the now-idle dispatcher, and then
runs the guarded two-lane probe followed by the remaining evaluations.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace").replace("\x00", " ")
    except OSError:
        return ""


def _eval_pids() -> list[int]:
    try:
        rows = subprocess.check_output(
            ["ps", "-eo", "pid=,stat=,args="], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for row in rows:
        if "eval_best_only.py" not in row:
            continue
        try:
            fields = row.strip().split(maxsplit=2)
            if len(fields) < 3 or fields[1].startswith("Z"):
                continue
            pids.append(int(fields[0]))
        except (IndexError, ValueError):
            continue
    return pids


def _wait_pid_exit(pid: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while os.path.exists(f"/proc/{pid}"):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"evaluation pid {pid} did not exit within {timeout:.0f}s")
        time.sleep(2.0)


def _wait_no_evaluators(timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        pids = _eval_pids()
        if not pids:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"evaluation workers remain after current child exit: {pids}")
        time.sleep(2.0)


def _kill_idle_queue(queue_pid: int, campaign: str) -> None:
    command = _cmdline(queue_pid)
    if not command:
        print(f"[HANDOFF] queue pid {queue_pid} already exited", flush=True)
        return
    if "run_experiment_queue.py" not in command or campaign not in command:
        raise RuntimeError(f"queue pid mismatch; refusing to kill {queue_pid}: {command}")
    os.kill(queue_pid, signal.SIGKILL)
    print(f"[HANDOFF] removed idle serial dispatcher pid={queue_pid}", flush=True)


def _latest_probe_report(campaign_root: Path) -> Path:
    candidates = [Path(item) for item in glob.glob(str(campaign_root / "parallel_eval_probe_*" / "parallel_eval_probe_report.json"))]
    if not candidates:
        raise FileNotFoundError("parallel probe report was not created")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run(command: list[str], log_handle) -> int:
    print("[HANDOFF] RUN " + json.dumps(command, ensure_ascii=False), flush=True)
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(f"[HANDOFF] returncode={completed.returncode}", flush=True)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-pid", type=int, required=True)
    parser.add_argument("--current-eval-pid", type=int, required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--wait-timeout", type=float, default=12 * 60 * 60)
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--max-memory-gib", type=float, default=65.0)
    args = parser.parse_args(argv)

    campaign_root = Path(args.campaign_root).resolve()
    log_path = campaign_root / "parallel_eval_handoff.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        print(f"[HANDOFF] waiting for current evaluation pid={args.current_eval_pid}", flush=True)
        _wait_pid_exit(args.current_eval_pid, args.wait_timeout)
        print("[HANDOFF] current evaluation exited", flush=True)
        _wait_no_evaluators(120.0)
        _kill_idle_queue(args.queue_pid, args.campaign)

        probe_command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / "parallel_deferred_evaluation.py"),
            "--campaign-root",
            str(campaign_root),
            "--mode",
            "probe",
            "--split",
            "validation",
            "--loader-workers",
            str(args.loader_workers),
            "--max-memory-gib",
            str(args.max_memory_gib),
        ]
        if _run(probe_command, log_handle) != 0:
            print("[HANDOFF] probe failed; remaining evaluations were not started", flush=True)
            return 2

        report_path = _latest_probe_report(campaign_root)
        remaining_command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / "parallel_deferred_evaluation.py"),
            "--campaign-root",
            str(campaign_root),
            "--mode",
            "remaining",
            "--split",
            "validation",
            "--loader-workers",
            str(args.loader_workers),
            "--max-memory-gib",
            str(args.max_memory_gib),
            "--probe-report",
            str(report_path),
        ]
        return _run(remaining_command, log_handle)


if __name__ == "__main__":
    raise SystemExit(main())
