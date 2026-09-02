#!/usr/bin/env python3
"""Collect validation-only DynaSEAF decomposition artifacts.

The collector deliberately never iterates the test loader.  It loads an
already-trained checkpoint, runs the validation split, writes normalized
forecast/target/component tensors in compressed chunks, and emits streaming
per-sample mechanism statistics plus qualitative panels.  It does not resume
training or modify the training result directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COMPONENT_KEYS = (
    "direct_forecast",
    "transport_forecast",
    "innovation",
    "gate",
    "deformation",
    "predicted_dynamics",
)
TARGET_VARIABLES = ("TEMP", "SALT")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve()), "present": path.is_file()}
    if path.is_file():
        result["size_bytes"] = path.stat().st_size
        result["sha256"] = sha256(path)
    return result


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite_values(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    return values[np.isfinite(values)]


def scalar_stats(array: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
    values = finite_values(array)
    if values.size == 0:
        return None, None, None, None
    return (
        float(values.mean()),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 90)),
    )


def rmse(array: np.ndarray) -> float | None:
    values = finite_values(array)
    return float(np.sqrt(np.mean(values * values))) if values.size else None


def mean_abs(array: np.ndarray) -> float | None:
    values = finite_values(array)
    return float(np.mean(np.abs(values))) if values.size else None


def parse_dtype(config: dict[str, Any]) -> torch.dtype:
    requested = str(config.get("mixed_precision_dtype", "bfloat16")).lower()
    if requested in {"float16", "fp16", "half"}:
        return torch.float16
    return torch.bfloat16


def load_run_config(config_path: Path) -> dict[str, Any]:
    from config import DEFAULT_CONFIG, load_config, merge_configs

    return merge_configs(load_config(config_path), DEFAULT_CONFIG)


def load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device):
    from model_factory import create_ocean_model

    model = create_ocean_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", -1) + 1)
    return model, checkpoint, best_epoch


def close_loader(loader: Any) -> None:
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if loader is not None and hasattr(loader, "_iterator"):
        loader._iterator = None
    dataset = getattr(loader, "dataset", None)
    source = getattr(dataset, "dataset", None)
    close = getattr(source, "close", None)
    if callable(close):
        close()


def load_validation_loader(config: dict[str, Any]):
    from data_loader import create_data_loaders

    loaders = create_data_loaders(
        config["data_path"],
        config,
        batch_size=int(config.get("batch_size", 76)),
        num_workers=int(config.get("num_workers", 2)),
        persistent_workers=bool(config.get("persistent_workers", True)),
        prefetch_factor=int(config.get("prefetch_factor", 1)),
    )
    return loaders[1], loaders


def slice_map(dataset: Any, config: dict[str, Any]) -> dict[str, slice]:
    raw = getattr(dataset, "target_channel_slices", {}) or {}
    if not raw:
        raw = config.get("target_channel_slices", {}) or {}
    return {
        name: slice(int(bounds.start), int(bounds.stop))
        if isinstance(bounds, slice)
        else slice(int(bounds[0]), int(bounds[1]))
        for name, bounds in raw.items()
    }


def component_or_none(output: dict[str, torch.Tensor], key: str, shape: tuple[int, ...]) -> np.ndarray | None:
    value = output.get(key)
    if value is None:
        return None
    return value.detach().float().cpu().numpy()


def component_metric(
    array: np.ndarray | None,
    sample_pos: int,
    lead_pos: int,
    channels: slice | None = None,
) -> np.ndarray | None:
    if array is None:
        return None
    value = array[sample_pos, lead_pos]
    if channels is not None and value.ndim >= 3:
        value = value[channels]
    return np.asarray(value)


def deformation_magnitude(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim < 1 or array.shape[-1] != 2:
        return None
    return np.sqrt(np.sum(array.astype(np.float64) ** 2, axis=-1))


def write_chunk(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    np.savez_compressed(path, **arrays)
    sample_indices = arrays["sample_indices"]
    return {
        **artifact(path),
        "sample_count": int(len(sample_indices)),
        "sample_index_min": int(sample_indices.min()) if len(sample_indices) else None,
        "sample_index_max": int(sample_indices.max()) if len(sample_indices) else None,
    }


def plot_qualitative(
    output_dir: Path,
    first: dict[str, np.ndarray],
    dataset: Any,
    channels: dict[str, slice],
) -> list[Path]:
    if not first:
        return []
    sample_idx = int(first["sample_indices"][0])
    _, region_idx = dataset.sequences[sample_idx]
    coords = dataset.all_regions_data[int(region_idx)].get("coords", {})
    lons = np.asarray(coords.get("lons"))
    lats = np.asarray(coords.get("lats"))
    outputs: list[Path] = []
    is_dynaseaf = "direct_forecast" in first
    for variable in TARGET_VARIABLES:
        ch = channels.get(variable)
        if ch is None:
            continue
        rows = ["target", "final"]
        if is_dynaseaf:
            rows.extend(["direct", "transport", "gate"])
        figure, axes = plt.subplots(
            len(rows), 3,
            figsize=(13, 3.2 * len(rows)),
            squeeze=False,
        )
        for col in range(3):
            lead = min(col, first["target_normalized"].shape[1] - 1)
            target = first["target_normalized"][0, lead, ch].mean(axis=0)
            final = first["forecast_normalized"][0, lead, ch].mean(axis=0)
            maps: dict[str, np.ndarray] = {"target": target, "final": final}
            if is_dynaseaf:
                maps["direct"] = first["direct_forecast"][0, lead, ch].mean(axis=0)
                maps["transport"] = first["transport_forecast"][0, lead, ch].mean(axis=0)
                maps["gate"] = first["gate"][0, lead, ch].mean(axis=0)
            for row, name in enumerate(rows):
                values = maps[name]
                image = axes[row, col].pcolormesh(
                    lons, lats, values, shading="auto", cmap="viridis"
                )
                axes[row, col].set_title(f"{name}, lead {lead + 1}")
                axes[row, col].set_xlabel("Longitude")
                axes[row, col].set_ylabel("Latitude")
                figure.colorbar(image, ax=axes[row, col], shrink=0.8)
        figure.suptitle(
            f"{variable} normalized validation decomposition — sample {sample_idx}",
            fontsize=14,
        )
        figure.tight_layout()
        path = output_dir / f"{variable.lower()}_qualitative_panel.png"
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help="do not write the large preprocessing cache during one-off collection",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="override the preprocessing cache directory when caching is enabled",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = result_dir / "best_model.pth"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    disk = shutil.disk_usage(output_dir)
    if disk.free < 1 * 1024**3:
        raise RuntimeError(
            f"diagnostic output disk has less than 1 GiB free: {disk.free / 1024**3:.2f} GiB"
        )

    config = load_run_config(config_path)
    config["return_future_dynamics_targets"] = False
    # Validation never materializes future dynamics labels.  Do not let an
    # unused training-only target list split the physical preprocessing cache
    # from the baseline configuration.
    config.pop("future_dynamics_target_variables", None)
    if args.cache_dir is not None:
        config["cache_preprocessed_dir"] = str(args.cache_dir.resolve())
    if args.disable_cache:
        config["cache_preprocessed"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_dynaseaf = str(config.get("model_type", "seaf")).lower() == "dynaseaf"
    model, checkpoint, best_epoch = load_model(config, checkpoint_path, device)
    validation_loader, all_loaders = load_validation_loader(config)
    validation_dataset = validation_loader.dataset
    channels = slice_map(validation_dataset, config)
    amp_enabled = bool(config.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = parse_dtype(config)
    chunk_records: list[dict[str, Any]] = []
    first_arrays: dict[str, np.ndarray] = {}
    sample_metric_path = output_dir / "sample_mechanism_metrics.csv"
    fieldnames = [
        "model",
        "seed",
        "split",
        "sample_index",
        "origin_id",
        "region_id",
        "variable",
        "lead",
        "final_rmse_normalized",
        "direct_rmse_normalized",
        "transport_rmse_normalized",
        "innovation_rmse_normalized",
        "gate_mean",
        "gate_p10",
        "gate_median",
        "gate_p90",
        "deformation_magnitude_mean",
        "deformation_magnitude_max",
        "predicted_dynamics_abs_mean",
    ]
    total_samples = 0
    chunk_index = 0
    with sample_metric_path.open("w", encoding="utf-8", newline="") as metric_handle:
        writer = csv.DictWriter(metric_handle, fieldnames=fieldnames)
        writer.writeheader()
        with torch.inference_mode():
            for batch in validation_loader:
                inputs, targets, sample_indices = batch[:3]
                if args.max_samples is not None:
                    remaining = max(0, int(args.max_samples) - total_samples)
                    if remaining <= 0:
                        break
                    if len(sample_indices) > remaining:
                        inputs = inputs[:remaining]
                        targets = targets[:remaining]
                        sample_indices = sample_indices[:remaining]
                inputs_device = inputs.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    raw_output = model(
                        inputs_device,
                        return_diagnostics=True,
                    ) if is_dynaseaf else model(inputs_device)
                if isinstance(raw_output, dict):
                    forecast_tensor = raw_output["forecast"]
                    batch_arrays = {
                        "sample_indices": np.asarray(sample_indices.cpu(), dtype=np.int64),
                        "forecast_normalized": forecast_tensor.detach().float().cpu().numpy(),
                        "target_normalized": targets.float().cpu().numpy(),
                    }
                    for key in COMPONENT_KEYS:
                        value = raw_output.get(key)
                        if torch.is_tensor(value):
                            batch_arrays[key] = value.detach().float().cpu().numpy()
                else:
                    batch_arrays = {
                        "sample_indices": np.asarray(sample_indices.cpu(), dtype=np.int64),
                        "forecast_normalized": raw_output.detach().float().cpu().numpy(),
                        "target_normalized": targets.float().cpu().numpy(),
                    }
                if not first_arrays:
                    first_arrays = {
                        key: value[:1].copy() if isinstance(value, np.ndarray) else value
                        for key, value in batch_arrays.items()
                    }
                batch_count = len(batch_arrays["sample_indices"])
                for start in range(0, batch_count, args.chunk_size):
                    end = min(batch_count, start + args.chunk_size)
                    chunk_arrays = {
                        key: value[start:end]
                        for key, value in batch_arrays.items()
                    }
                    chunk_path = output_dir / f"chunk_{chunk_index:05d}.npz"
                    chunk_records.append(write_chunk(chunk_path, chunk_arrays))
                    chunk_index += 1

                for position, sample_index_value in enumerate(batch_arrays["sample_indices"]):
                    sample_index = int(sample_index_value)
                    origin_id, region_id = validation_dataset.sequences[sample_index]
                    for variable in TARGET_VARIABLES:
                        ch = channels.get(variable)
                        if ch is None:
                            continue
                        for lead_pos in range(batch_arrays["target_normalized"].shape[1]):
                            target_value = component_metric(batch_arrays["target_normalized"], position, lead_pos, ch)
                            row: dict[str, Any] = {
                                "model": "DynaSEAF" if is_dynaseaf else "SEAF-v1",
                                "seed": config.get("seed"),
                                "split": "validation",
                                "sample_index": sample_index,
                                "origin_id": int(origin_id),
                                "region_id": int(region_id),
                                "variable": variable,
                                "lead": lead_pos + 1,
                                "final_rmse_normalized": rmse(component_metric(batch_arrays["forecast_normalized"], position, lead_pos, ch) - target_value) if target_value is not None else None,
                                "direct_rmse_normalized": None,
                                "transport_rmse_normalized": None,
                                "innovation_rmse_normalized": None,
                                "gate_mean": None,
                                "gate_p10": None,
                                "gate_median": None,
                                "gate_p90": None,
                                "deformation_magnitude_mean": None,
                                "deformation_magnitude_max": None,
                                "predicted_dynamics_abs_mean": None,
                            }
                            for key, output_field in (
                                ("direct_forecast", "direct_rmse_normalized"),
                                ("transport_forecast", "transport_rmse_normalized"),
                                ("innovation", "innovation_rmse_normalized"),
                            ):
                                value = component_metric(batch_arrays.get(key), position, lead_pos, ch)
                                row[output_field] = rmse(value - target_value) if value is not None and target_value is not None else None
                            gate_value = component_metric(batch_arrays.get("gate"), position, lead_pos, ch)
                            if gate_value is not None:
                                mean, p10, median, p90 = scalar_stats(gate_value)
                                row.update({
                                    "gate_mean": mean,
                                    "gate_p10": p10,
                                    "gate_median": median,
                                    "gate_p90": p90,
                                })
                            deformation_value = deformation_magnitude(
                                component_metric(batch_arrays.get("deformation"), position, lead_pos, ch)
                            )
                            if deformation_value is not None:
                                row["deformation_magnitude_mean"] = mean_abs(deformation_value)
                                finite_deformation = finite_values(deformation_value)
                                row["deformation_magnitude_max"] = float(finite_deformation.max()) if finite_deformation.size else None
                            dynamics_value = component_metric(
                                batch_arrays.get("predicted_dynamics"), position, lead_pos
                            )
                            if dynamics_value is not None:
                                row["predicted_dynamics_abs_mean"] = mean_abs(dynamics_value)
                            writer.writerow(row)
                total_samples += batch_count
                print(f"[DIAGNOSTICS] validation samples {total_samples}", flush=True)

    qualitative_paths = plot_qualitative(output_dir, first_arrays, validation_dataset, channels)
    run_manifest = {
        "schema": "dynaseaf-validation-diagnostics-run-v1",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "DynaSEAF" if is_dynaseaf else "SEAF-v1",
        "seed": config.get("seed"),
        "split": "validation",
        "test_iteration": False,
        "retraining": False,
        "result_dir": str(result_dir),
        "config": artifact(config_path),
        "checkpoint": artifact(checkpoint_path),
        "best_epoch": best_epoch,
        "device": {
            "type": device.type,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "amp_enabled": amp_enabled,
            "amp_dtype": str(amp_dtype),
        },
        "data": {
            "path": config.get("data_path"),
            "dataset_mode": "val",
            "sample_count": total_samples,
            "target_variables": list(TARGET_VARIABLES),
            "target_channel_slices": {
                name: [int(value.start), int(value.stop)] for name, value in channels.items()
            },
        },
        "tensor_space": "normalized model/anomaly space",
        "diagnostic_keys": sorted(key for key in COMPONENT_KEYS if key in first_arrays),
        "chunks": chunk_records,
        "sample_mechanism_metrics": artifact(sample_metric_path),
        "qualitative_panels": [artifact(path) for path in qualitative_paths],
        "storage": {
            "output_dir": str(output_dir),
            "free_bytes_after": shutil.disk_usage(output_dir).free,
            "platform": platform.platform(),
        },
        "checkpoint_metadata": {
            "epoch": checkpoint.get("epoch"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_val_loss": checkpoint.get("best_val_loss"),
        },
    }
    dump_json(output_dir / "diagnostics_manifest.json", run_manifest)
    close_loader(validation_loader)
    for loader in all_loaders:
        if loader is not validation_loader:
            close_loader(loader)
    print(json.dumps({
        "status": "completed",
        "model": run_manifest["model"],
        "seed": run_manifest["seed"],
        "samples": total_samples,
        "output_dir": str(output_dir),
        "diagnostics_keys": run_manifest["diagnostic_keys"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
