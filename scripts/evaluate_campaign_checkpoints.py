#!/usr/bin/env python3
"""Evaluate frozen campaign checkpoints on a held-out split, serially.

This driver is intentionally separate from training.  It reads the immutable
queue manifest, invokes ``eval_best_only.py`` for each completed run, and
records a machine-readable manifest.  The serial execution is important for
the ORAS5 memory budget because evaluation materializes physical forecasts and
baseline reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_config(config_ref: str) -> Path:
    path = Path(config_ref)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign_root",
        required=True,
        help="campaign result directory containing experiment_queue_state.json",
    )
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument(
        "--stage",
        default="confirm_validation",
        help="queue stage to evaluate; default is the frozen confirmation stage",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="optional run names to evaluate; default evaluates every completed job",
    )
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root).resolve()
    state_path = campaign_root / "experiment_queue_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"queue state not found: {state_path}")
    state = _read_json(state_path)
    jobs = state.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"queue state has no jobs list: {state_path}")

    selected_names = set(args.only) if args.only is not None else None
    selected = []
    failures = []
    skipped = []
    for job in jobs:
        if not isinstance(job, dict) or job.get("stage") != args.stage:
            continue
        name = str(job.get("name", ""))
        if selected_names is not None and name not in selected_names:
            continue
        if job.get("status") != "completed":
            skipped.append({"name": name, "reason": f"status={job.get('status')!r}"})
            continue
        result_dir = Path(str(job.get("result_dir", ""))).resolve()
        checkpoint = result_dir / "best_model.pth"
        if not checkpoint.is_file():
            skipped.append({"name": name, "reason": "best_model.pth missing", "result_dir": str(result_dir)})
            continue
        selected.append((job, result_dir))

    if selected_names is not None:
        found = {str(job.get("name", "")) for job, _ in selected}
        found.update(item["name"] for item in skipped)
        missing = sorted(selected_names - found)
        if missing:
            raise ValueError(f"requested runs are absent from campaign state: {missing}")
    if not selected:
        raise RuntimeError("no completed checkpoints selected for evaluation")
    if skipped:
        raise RuntimeError(
            "campaign is incomplete; refusing partial frozen evaluation: "
            + json.dumps(skipped, ensure_ascii=False)
        )

    log_path = campaign_root / f"{args.split}_evaluation_driver.log"
    manifest_path = campaign_root / f"{args.split}_evaluation_manifest.json"
    records = []
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as log_handle:
        for index, (job, result_dir) in enumerate(selected, start=1):
            name = str(job["name"])
            config_path = _resolve_config(str(job["config"]))
            overrides = dict(job.get("overrides", {}))
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "eval_best_only.py"),
                "--config",
                str(config_path),
                "--result_dir",
                str(result_dir),
                "--overrides_json",
                json.dumps(overrides, sort_keys=True, separators=(",", ":")),
                "--split",
                args.split,
            ]
            log_handle.write(
                f"\n=== [{index}/{len(selected)}] {name} "
                f"{datetime.now(timezone.utc).isoformat()} ===\n"
            )
            log_handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
            log_handle.flush()
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            expected_filename = (
                "evaluation_results.json" if args.split == "test"
                else "validation_results.json"
            )
            evaluation_path = result_dir / expected_filename
            checkpoint_path = result_dir / "best_model.pth"
            record = {
                "name": name,
                "result_dir": str(result_dir),
                "config": str(config_path),
                "config_sha256": _sha256(config_path)
                if config_path.is_file() else None,
                "returncode": int(completed.returncode),
                "evaluation_path": str(evaluation_path),
                "evaluation_exists": evaluation_path.is_file(),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path)
                if checkpoint_path.is_file() else None,
                "evaluation_sha256": _sha256(evaluation_path)
                if evaluation_path.is_file() else None,
            }
            records.append(record)
            if completed.returncode != 0 or not evaluation_path.is_file():
                failures.append(record)
                _write_json(
                    manifest_path,
                    {
                        "status": "failed",
                        "split": args.split,
                        "stage": args.stage,
                        "campaign_root": str(campaign_root),
                        "records": records,
                        "failures": failures,
                        "started_at_utc": started,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "log": str(log_path),
                    },
                )
                raise RuntimeError(f"evaluation failed for {name}; see {log_path}")

    manifest = {
        "status": "completed",
        "split": args.split,
        "stage": args.stage,
        "campaign_root": str(campaign_root),
        "run_count": len(records),
        "records": records,
        "failures": failures,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "log": str(log_path),
        "note": "Evaluation only; no checkpoint was resumed for further training.",
    }
    _write_json(manifest_path, manifest)
    (campaign_root / f"{args.split}_evaluation_SUCCESS").write_text(
        manifest["completed_at_utc"] + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
