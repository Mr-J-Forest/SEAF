#!/usr/bin/env python3
"""Collect reproducible regional forecast evidence for trained checkpoints.

The script deliberately keeps model inference, frozen baselines, numeric arrays,
figures, and provenance together.  It is intended for the paper's qualitative
evidence module; headline claims must still use the complete validation/test
reports produced by training.

Examples
--------
python scripts/collect_prediction_evidence.py \
    --main_model_dir outputs/results/campaigns/.../confirm_validation/full/seed_42 \
    --direct_full_field_dir outputs/results/campaigns/.../confirm_validation/direct_full_field_strict/seed_42 \
    --output_dir outputs/prediction_evidence/frozen_test_region

The same script can be run with only ``--main_model_dir`` when the comparator
checkpoint is not available yet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG
from data_loader import OceanDataset
from metrics_utils import compute_metric_report
from predict import SmartOceanPredictor


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#2E8B57",
    "red": "#B64342",
    "teal": "#42949E",
    "neutral": "#666666",
}


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, slice):
        return [value.start, value.stop]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
    return cleaned or "model"


def _finite_values(*arrays: np.ndarray) -> np.ndarray:
    chunks = []
    for array in arrays:
        values = np.asarray(array)
        finite = values[np.isfinite(values)]
        if finite.size:
            chunks.append(finite.reshape(-1))
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)


def _metric_value(mapping: Mapping[str, Any], key: str) -> Optional[float]:
    value = mapping.get(key)
    if isinstance(value, Mapping) and "mean" in value:
        value = value.get("mean")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _save_figure(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.2)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _target_slices(dataset: OceanDataset, channel_count: int) -> Dict[str, slice]:
    slices = getattr(dataset, "target_channel_slices", {}) or {}
    if slices:
        return {
            str(name): slice(int(value.start), int(value.stop))
            for name, value in slices.items()
        }
    output: Dict[str, slice] = {}
    offset = 0
    for variable in dataset.target_variables:
        levels = int(dataset.all_regions_data[0]["data"][variable].shape[1])
        output[variable] = slice(offset, offset + levels)
        offset += levels
    if offset != channel_count:
        raise ValueError(f"target channel schema mismatch: {offset} != {channel_count}")
    return output


def _load_test_view(predictor: SmartOceanPredictor):
    """Load the train payload once and create a memory-sharing test view.

    This follows the training/evaluation data path and avoids rebuilding a
    second full spatial payload with externally supplied scalers.
    """
    train_dataset = OceanDataset(
        predictor.config["data_path"],
        predictor.config,
        mode="train",
    )
    test_dataset = train_dataset.temporal_split_view("test")
    return train_dataset, test_dataset


def _select_sample(
    dataset: OceanDataset,
    sample_index: Optional[int],
    base_time_index: Optional[int],
    target_lon: Optional[float],
    target_lat: Optional[float],
) -> int:
    if len(dataset) == 0:
        raise RuntimeError("test dataset contains no forecast samples")
    if sample_index is not None:
        if not 0 <= int(sample_index) < len(dataset):
            raise IndexError(f"sample index out of range: {sample_index}")
        return int(sample_index)

    if target_lon is None:
        configured_lon = dataset.config.get("lon_range", [0.0, 0.0])
        target_lon = float(np.mean(configured_lon))
    if target_lat is None:
        configured_lat = dataset.config.get("lat_range", [0.0, 0.0])
        target_lat = float(np.mean(configured_lat))

    region_scores = []
    for region_idx, region in enumerate(dataset.all_regions_data):
        coords = region.get("coords", {})
        lons = np.asarray(coords.get("lons", []), dtype=np.float64)
        lats = np.asarray(coords.get("lats", []), dtype=np.float64)
        if not lons.size or not lats.size:
            continue
        center_lon = float(np.mean(lons))
        center_lat = float(np.mean(lats))
        region_scores.append(
            ((center_lon - float(target_lon)) ** 2 + (center_lat - float(target_lat)) ** 2,
             region_idx)
        )
    if not region_scores:
        raise RuntimeError("test dataset contains no coordinate-bearing regions")
    selected_region = min(region_scores)[1]

    candidate_indices = [
        idx for idx, (_, region_idx) in enumerate(dataset.sequences)
        if int(region_idx) == int(selected_region)
    ]
    if not candidate_indices:
        raise RuntimeError(f"no test samples found for region {selected_region}")
    if base_time_index is None:
        return int(candidate_indices[0])

    desired_start = int(base_time_index) - int(dataset.sequence_length) + 1
    return int(min(
        candidate_indices,
        key=lambda idx: abs(int(dataset.sequences[idx][0]) - desired_start),
    ))


def _model_label(model_dir: Path, requested: Optional[str]) -> str:
    if requested:
        return _safe_name(requested)
    name = model_dir.name
    if name.startswith("seed_") and model_dir.parent.name:
        return _safe_name(model_dir.parent.name)
    return _safe_name(name)


def _collect_one(
    model_dir: Path,
    output_dir: Path,
    sample_index: Optional[int],
    base_time_index: Optional[int],
    target_lon: Optional[float],
    target_lat: Optional[float],
    label: Optional[str],
    leads: Sequence[int],
    depth_indices: Sequence[int],
) -> Dict[str, Any]:
    model_label = _model_label(model_dir, label)
    model_output = output_dir / model_label
    model_output.mkdir(parents=True, exist_ok=True)

    predictor = SmartOceanPredictor(
        model_dir=str(model_dir),
        config=DEFAULT_CONFIG.copy(),
        output_dir=str(model_output / "predictor_cache"),
    )
    train_dataset = None
    test_dataset = None
    try:
        train_dataset, test_dataset = _load_test_view(predictor)
        selected_index = _select_sample(
            test_dataset,
            sample_index=sample_index,
            base_time_index=base_time_index,
            target_lon=target_lon,
            target_lat=target_lat,
        )
        input_tensor, target_tensor = test_dataset[selected_index]
        import torch

        with torch.no_grad():
            output_tensor = predictor.model(
                input_tensor.unsqueeze(0).to(predictor.device)
            ).detach().cpu().numpy()
        target_tensor_np = target_tensor.unsqueeze(0).numpy()

        prediction = test_dataset.inverse_transform_targets(
            output_tensor,
            sample_indices=[selected_index],
        )[0].astype(np.float32, copy=False)
        target = test_dataset.inverse_transform_targets(
            target_tensor_np,
            sample_indices=[selected_index],
        )[0].astype(np.float32, copy=False)
        baselines = test_dataset.build_reference_forecasts(
            sample_indices=[selected_index],
            spaces=("physical",),
        )["physical"]
        climatology = baselines["climatology"][0].astype(np.float32, copy=False)
        anomaly_prediction = prediction - climatology
        anomaly_target = target - climatology
        anomaly_baselines = {
            name: values[0] - climatology
            for name, values in baselines.items()
        }

        channel_slices = _target_slices(test_dataset, prediction.shape[1])
        report = compute_metric_report(
            prediction[np.newaxis, ...],
            target[np.newaxis, ...],
            list(test_dataset.target_variables),
            channel_slices=channel_slices,
            baselines=baselines,
            metric_space="physical",
            depth_values=[float(value) for value in test_dataset.levels],
        )
        report["target_variables"] = list(test_dataset.target_variables)
        report["target_channel_slices"] = {
            name: [int(value.start), int(value.stop)]
            for name, value in channel_slices.items()
        }
        provenance = test_dataset.build_sample_provenance([selected_index])
        region_idx = int(test_dataset.sequences[selected_index][1])
        region = test_dataset.all_regions_data[region_idx]
        region_coords = region.get("coords", {})

        array_path = model_output / "regional_evidence.npz"
        np.savez_compressed(
            array_path,
            prediction=prediction,
            target=target,
            climatology=climatology,
            anomaly_prediction=anomaly_prediction,
            anomaly_target=anomaly_target,
            anomaly_persistence=anomaly_baselines["anomaly_persistence"],
            damped_anomaly_persistence=anomaly_baselines["damped_anomaly_persistence"],
            ap_physical=baselines["anomaly_persistence"][0],
            dap_physical=baselines["damped_anomaly_persistence"][0],
            persistence_physical=baselines["persistence"][0],
            lons=np.asarray(region_coords["lons"], dtype=np.float64),
            lats=np.asarray(region_coords["lats"], dtype=np.float64),
            levels=np.asarray(test_dataset.levels, dtype=np.float32),
        )

        metadata = {
            "status": "completed",
            "model_label": model_label,
            "model_dir": str(model_dir.resolve()),
            "checkpoint": str(Path(predictor.model_path).resolve()),
            "sample_index_in_test_view": int(selected_index),
            "region_index": region_idx,
            "region_lon_range": [float(value) for value in region.get("lon_range", [])],
            "region_lat_range": [float(value) for value in region.get("lat_range", [])],
            "base_time_index": int(test_dataset.sequences[selected_index][0] + test_dataset.sequence_length - 1),
            "forecast_time_indices": [
                int(value)
                for value in range(
                    test_dataset.sequences[selected_index][0] + test_dataset.sequence_length,
                    test_dataset.sequences[selected_index][0]
                    + test_dataset.sequence_length
                    + test_dataset.prediction_length,
                )
            ],
            "target_variables": list(test_dataset.target_variables),
            "target_channel_slices": {
                name: [int(value.start), int(value.stop)]
                for name, value in channel_slices.items()
            },
            "prediction_shape": list(prediction.shape),
            "array_file": array_path.name,
            "provenance": provenance,
            "config_semantics": {
                key: predictor.config.get(key)
                for key in (
                    "enable_climatology_anomaly",
                    "enable_target_climatology_anomaly",
                    "ablation_direct_full_field",
                    "ablation_disable_spectral",
                    "ablation_disable_ensemble",
                    "ablation_uniform_ensemble",
                    "include_tendency_features",
                    "external_dynamic_variables",
                    "cache_preprocessed_dir",
                )
            },
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(model_output / "regional_evidence.json", metadata)
        _write_json(model_output / "metrics.json", report)
        _render_model_figures(
            model_output,
            model_label,
            prediction,
            target,
            climatology,
            baselines,
            anomaly_prediction,
            anomaly_target,
            anomaly_baselines,
            np.asarray(region_coords["lons"], dtype=np.float64),
            np.asarray(region_coords["lats"], dtype=np.float64),
            np.asarray(test_dataset.levels, dtype=np.float32),
            report,
            channel_slices=channel_slices,
            leads=leads,
            depth_indices=depth_indices,
        )
        return {
            "label": model_label,
            "output_dir": str(model_output.resolve()),
            "metadata": metadata,
            "report": report,
            "arrays": {
                "prediction": prediction,
                "target": target,
                "climatology": climatology,
                "anomaly_prediction": anomaly_prediction,
                "anomaly_target": anomaly_target,
                "anomaly_persistence": anomaly_baselines["anomaly_persistence"],
                "damped_anomaly_persistence": anomaly_baselines["damped_anomaly_persistence"],
                "ap_physical": baselines["anomaly_persistence"][0],
                "lons": np.asarray(region_coords["lons"], dtype=np.float64),
                "lats": np.asarray(region_coords["lats"], dtype=np.float64),
                "levels": np.asarray(test_dataset.levels, dtype=np.float32),
            },
        }
    finally:
        # The view shares the xarray handle with the train dataset.
        if train_dataset is not None and getattr(train_dataset, "dataset", None) is not None:
            train_dataset.dataset.close()


def _variable_metrics(report: Mapping[str, Any], variable: str, baseline: str):
    model_by_lead = report.get("by_variable_and_lead", {}).get(variable, {})
    baseline_report = report.get("baselines", {}).get(baseline, {})
    baseline_by_lead = baseline_report.get("by_variable_and_lead", {}).get(variable, {})
    comparison = report.get("comparison", {}).get(baseline, {})
    comparison_by_lead = comparison.get("by_variable_and_lead", {}).get(variable, {})
    output = []
    for lead_name, model_metrics in model_by_lead.items():
        output.append({
            "lead": int(str(lead_name).split("_")[-1]),
            "rmse": _metric_value(model_metrics, "rmse"),
            "mae": _metric_value(model_metrics, "mae"),
            "r2": _metric_value(model_metrics, "r2"),
            "baseline_rmse": _metric_value(baseline_by_lead.get(lead_name, {}), "rmse"),
            "ss_ap": _metric_value(comparison_by_lead.get(lead_name, {}), "mse_skill")
            if baseline == "anomaly_persistence" else None,
            "rmse_improvement_pct": _metric_value(
                comparison_by_lead.get(lead_name, {}), "rmse_improvement_pct"
            ),
        })
    return output


def _render_model_figures(
    output_dir: Path,
    model_label: str,
    prediction: np.ndarray,
    target: np.ndarray,
    climatology: np.ndarray,
    baselines: Mapping[str, np.ndarray],
    anomaly_prediction: np.ndarray,
    anomaly_target: np.ndarray,
    anomaly_baselines: Mapping[str, np.ndarray],
    lons: np.ndarray,
    lats: np.ndarray,
    levels: np.ndarray,
    report: Mapping[str, Any],
    channel_slices: Optional[Mapping[str, slice]] = None,
    leads: Sequence[int] = (0, 2, 4),
    depth_indices: Optional[Sequence[int]] = None,
) -> None:
    del climatology
    if channel_slices is None:
        channel_slices = _infer_channel_slices(prediction, report)
    else:
        channel_slices = dict(channel_slices)
    n_depth = len(levels)
    if depth_indices is None:
        depth_indices = (0, n_depth // 2, n_depth - 1)
    selected_depths = [int(idx) for idx in depth_indices if 0 <= int(idx) < n_depth]
    selected_leads = [int(idx) for idx in leads if 0 <= int(idx) < prediction.shape[0]]
    if not selected_leads or not selected_depths:
        return

    unit_map = {"TEMP": "°C", "SALT": "PSU"}
    for variable, ch_slice in channel_slices.items():
        display_label = f"{variable} ({unit_map.get(variable, 'physical')})"
        var_pred = prediction[:, ch_slice]
        var_target = target[:, ch_slice]
        var_ap = np.asarray(baselines["anomaly_persistence"])[0][:, ch_slice]
        var_dap = np.asarray(baselines["damped_anomaly_persistence"])[0][:, ch_slice]
        var_anom_pred = anomaly_prediction[:, ch_slice]
        var_anom_target = anomaly_target[:, ch_slice]
        var_anom_ap = np.asarray(anomaly_baselines["anomaly_persistence"])[:, ch_slice]
        var_anom_dap = np.asarray(anomaly_baselines["damped_anomaly_persistence"])[:, ch_slice]

        for depth_idx in selected_depths:
            depth_label = _depth_label(levels, depth_idx)
            physical_fields = [
                ("Truth", var_target[:, depth_idx]),
                (model_label, var_pred[:, depth_idx]),
                ("AP", var_ap[:, depth_idx]),
                ("DAP", var_dap[:, depth_idx]),
            ]
            _plot_comparison_grid(
                output_dir / f"region_{_safe_name(variable)}_{_safe_name(depth_label)}_physical",
                f"{display_label} regional forecast — {depth_label}",
                physical_fields,
                var_pred[:, depth_idx] - var_target[:, depth_idx],
                var_ap[:, depth_idx] - var_target[:, depth_idx],
                selected_leads,
                lons,
                lats,
                error_label=f"{model_label} error",
                second_error_label="AP error",
                error_pair=True,
                anomaly=False,
            )
            anomaly_fields = [
                ("Truth anomaly", var_anom_target[:, depth_idx]),
                (f"{model_label} anomaly", var_anom_pred[:, depth_idx]),
                ("AP anomaly", var_anom_ap[:, depth_idx]),
                ("DAP anomaly", var_anom_dap[:, depth_idx]),
            ]
            _plot_comparison_grid(
                output_dir / f"region_{_safe_name(variable)}_{_safe_name(depth_label)}_anomaly",
                f"{display_label} anomaly evolution — {depth_label}",
                anomaly_fields,
                var_anom_pred[:, depth_idx] - var_anom_target[:, depth_idx],
                var_anom_ap[:, depth_idx] - var_anom_target[:, depth_idx],
                selected_leads,
                lons,
                lats,
                error_label=f"{model_label} anomaly error",
                second_error_label="AP anomaly error",
                error_pair=True,
                anomaly=True,
            )

    analysis = {
        "model_label": model_label,
        "variables": {},
        "interpretation": (
            "Positive ss_ap means lower MSE than anomaly persistence for the "
            "selected region and lead. This regional view is qualitative and "
            "does not replace the complete validation/test report."
        ),
    }
    for variable in channel_slices:
        analysis["variables"][variable] = {
            "lead_metrics_vs_ap": _variable_metrics(
                report, variable, "anomaly_persistence"
            ),
            "lead_metrics_vs_dap": _variable_metrics(
                report, variable, "damped_anomaly_persistence"
            ),
        }
    _write_json(output_dir / "regional_analysis.json", analysis)


def _infer_channel_slices(prediction: np.ndarray, report: Mapping[str, Any]) -> Dict[str, slice]:
    variables = list(
        report.get("target_variables")
        or report.get("_target_variables")
        or (["TEMP", "SALT"] if prediction.shape[1] >= 2 else ["target"])
    )
    # The formal ORAS5 target is depth-resolved TEMP followed by SALT.  Prefer
    # explicit slices saved by the collector; otherwise split equally only for
    # the unusual one-variable fallback.
    explicit = report.get("target_channel_slices")
    if isinstance(explicit, Mapping):
        return {
            name: slice(int(bounds[0]), int(bounds[1]))
            for name, bounds in explicit.items()
        }
    if len(variables) == 1:
        return {variables[0]: slice(0, prediction.shape[1])}
    if prediction.shape[1] % len(variables) != 0:
        raise ValueError(
            "collector requires explicit target_channel_slices for unequal variable depths"
        )
    width = prediction.shape[1] // len(variables)
    return {
        name: slice(idx * width, (idx + 1) * width)
        for idx, name in enumerate(variables)
    }


def _depth_label(levels: np.ndarray, depth_idx: int) -> str:
    if 0 <= int(depth_idx) < len(levels):
        value = float(levels[int(depth_idx)])
        return f"depth_{value:g}m"
    return f"depth_{int(depth_idx)}"


def _plot_comparison_grid(
    output_base: Path,
    title: str,
    fields: Sequence[tuple[str, np.ndarray]],
    model_error: np.ndarray,
    ap_error: np.ndarray,
    leads: Sequence[int],
    lons: np.ndarray,
    lats: np.ndarray,
    error_label: str,
    second_error_label: str,
    error_pair: bool,
    anomaly: bool,
) -> None:
    row_labels = [label for label, _ in fields]
    rows = len(fields) + (2 if error_pair else 1)
    fig, axes = plt.subplots(
        rows,
        len(leads),
        figsize=(4.2 * len(leads), 2.9 * rows),
        squeeze=False,
    )
    cmap = "RdBu_r" if anomaly else "turbo"
    for col, lead in enumerate(leads):
        common = _finite_values(*(values[lead] for _, values in fields))
        if common.size:
            if anomaly:
                limit = max(abs(float(common.min())), abs(float(common.max())), 1e-8)
                vmin, vmax = -limit, limit
            else:
                vmin, vmax = float(common.min()), float(common.max())
                if vmax <= vmin:
                    vmax = vmin + 1e-8
        else:
            vmin, vmax = 0.0, 1.0
        for row, (label, values) in enumerate(fields):
            ax = axes[row, col]
            image = ax.pcolormesh(
                lons,
                lats,
                values[lead],
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{label}\nlead {lead + 1}", fontsize=9)
            ax.set_xlabel("Longitude", fontsize=8)
            ax.set_ylabel("Latitude", fontsize=8)
            fig.colorbar(image, ax=ax, shrink=0.8)
        for row, (error, label) in enumerate(
            ((model_error, error_label), (ap_error, second_error_label)),
            start=len(fields),
        ):
            ax = axes[row, col]
            values = error[lead]
            finite = _finite_values(values)
            limit = max(float(np.max(np.abs(finite))), 1e-8) if finite.size else 1.0
            image = ax.pcolormesh(
                lons,
                lats,
                values,
                shading="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            ax.set_title(f"{label}\nlead {lead + 1}", fontsize=9)
            ax.set_xlabel("Longitude", fontsize=8)
            ax.set_ylabel("Latitude", fontsize=8)
            fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=13)
    _save_figure(fig, output_base)


def _render_cross_model_figures(
    output_dir: Path,
    main: Mapping[str, Any],
    comparator: Mapping[str, Any],
    leads: Sequence[int],
) -> None:
    main_label = main["label"]
    comparator_label = comparator["label"]
    main_arrays = main["arrays"]
    comparator_arrays = comparator["arrays"]
    target = np.asarray(main_arrays["target"], dtype=np.float32)
    comparator_target = np.asarray(comparator_arrays["target"], dtype=np.float32)
    if target.shape != comparator_target.shape:
        raise ValueError(
            "main/comparator evidence selected incompatible target shapes: "
            f"{target.shape} != {comparator_target.shape}"
        )
    # Both predictors are scored against the same physical target, but their
    # inverse transformations can differ by a few float32 ulps (especially
    # near zero anomaly values).  Use a physical-unit tolerance rather than
    # np.allclose's default atol=1e-8, which spuriously rejects those values.
    if not np.allclose(
        target,
        comparator_target,
        rtol=1e-5,
        atol=1e-5,
        equal_nan=True,
    ):
        difference = np.abs(target - comparator_target)
        finite_difference = difference[np.isfinite(difference)]
        max_difference = float(np.max(finite_difference)) if finite_difference.size else None
        mean_difference = float(np.mean(finite_difference)) if finite_difference.size else None
        raise ValueError(
            "main/comparator evidence selected different target fields: "
            f"max_abs_diff={max_difference}, mean_abs_diff={mean_difference}"
        )
    prediction = main_arrays["prediction"]
    comparator_prediction = comparator_arrays["prediction"]
    n_channels = prediction.shape[1]
    channel_slices = _infer_channel_slices(prediction, main["report"])
    lons = np.asarray(main_arrays.get("lons"))
    lats = np.asarray(main_arrays.get("lats"))
    levels = np.asarray(main_arrays.get("levels"))
    if lons.size == 0 or lats.size == 0:
        evidence_npz = np.load(Path(main["output_dir"]) / "regional_evidence.npz")
        lons = evidence_npz["lons"]
        lats = evidence_npz["lats"]
        levels = evidence_npz["levels"]
    depth_idx = 0
    for variable, ch_slice in channel_slices.items():
        if ch_slice.stop > n_channels:
            continue
        fields = [
            ("Truth", target[:, ch_slice][:, depth_idx]),
            (main_label, prediction[:, ch_slice][:, depth_idx]),
            (comparator_label, comparator_prediction[:, ch_slice][:, depth_idx]),
            ("AP", main_arrays["ap_physical"][:, ch_slice][:, depth_idx]),
        ]
        _plot_comparison_grid(
            output_dir / f"cross_model_{_safe_name(variable)}_surface",
            f"Regional physical comparison — {variable} surface",
            fields,
            prediction[:, ch_slice][:, depth_idx] - target[:, ch_slice][:, depth_idx],
            main_arrays["ap_physical"][:, ch_slice][:, depth_idx]
            - target[:, ch_slice][:, depth_idx],
            leads,
            lons,
            lats,
            error_label=f"{main_label} error",
            second_error_label="AP error",
            error_pair=True,
            anomaly=False,
        )

    _render_metric_comparison(output_dir, main, comparator)


def _render_metric_comparison(
    output_dir: Path,
    main: Mapping[str, Any],
    comparator: Mapping[str, Any],
) -> None:
    variables = list(main["report"].get("by_variable", {}).keys())
    if not variables:
        return
    fig, axes = plt.subplots(2, len(variables), figsize=(5.2 * len(variables), 8), squeeze=False)
    colors = {
        main["label"]: PALETTE["blue_main"],
        comparator["label"]: PALETTE["red"],
    }
    for col, variable in enumerate(variables):
        for model in (main, comparator):
            report = model["report"]
            leads = _variable_metrics(report, variable, "anomaly_persistence")
            x = [item["lead"] for item in leads]
            rmse = [item["rmse"] for item in leads]
            skill = [item["ss_ap"] for item in leads]
            axes[0, col].plot(
                x,
                rmse,
                "o-",
                label=model["label"],
                color=colors[model["label"]],
                linewidth=2,
            )
            axes[1, col].plot(
                x,
                skill,
                "o-",
                label=model["label"],
                color=colors[model["label"]],
                linewidth=2,
            )
        axes[0, col].set_title(f"{variable} regional RMSE")
        axes[0, col].set_xlabel("Lead month")
        axes[0, col].set_ylabel("RMSE (physical units)")
        axes[0, col].grid(alpha=0.25)
        axes[0, col].legend(frameon=False)
        axes[1, col].axhline(0.0, color=PALETTE["neutral"], linewidth=1)
        axes[1, col].set_title(f"{variable} skill vs AP")
        axes[1, col].set_xlabel("Lead month")
        axes[1, col].set_ylabel("SS_AP")
        axes[1, col].grid(alpha=0.25)
        axes[1, col].legend(frameon=False)
    _save_figure(fig, output_dir / "regional_metric_comparison")

    comparison = {
        "main_model": main["label"],
        "comparator_model": comparator["label"],
        "variables": {},
        "interpretation": (
            "SS_AP = 1 - MSE_model/MSE_AP. Values above zero indicate that "
            "the selected model improves on anomaly persistence for that lead "
            "and region; this is not a substitute for paired test statistics."
        ),
    }
    for variable in variables:
        comparison["variables"][variable] = {
            "main": _variable_metrics(main["report"], variable, "anomaly_persistence"),
            "comparator": _variable_metrics(
                comparator["report"], variable, "anomaly_persistence"
            ),
        }
    _write_json(output_dir / "regional_model_comparison.json", comparison)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main_model_dir", required=True)
    parser.add_argument("--direct_full_field_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--base_time_index", type=int, default=None)
    parser.add_argument("--target_lon", type=float, default=None)
    parser.add_argument("--target_lat", type=float, default=None)
    parser.add_argument("--main_label", default="SEAF")
    parser.add_argument("--direct_label", default="DirectFullField")
    parser.add_argument(
        "--depth_indices",
        type=int,
        nargs="+",
        default=[0, 10, 19],
        help="depth-channel indices within each target variable",
    )
    parser.add_argument(
        "--leads",
        type=int,
        nargs="+",
        default=[0, 2, 4],
        help="0-indexed forecast leads to render",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    main_result = _collect_one(
        Path(args.main_model_dir).resolve(),
        output_dir,
        sample_index=args.sample_index,
        base_time_index=args.base_time_index,
        target_lon=args.target_lon,
        target_lat=args.target_lat,
        label=args.main_label,
        leads=args.leads,
        depth_indices=args.depth_indices,
    )
    if args.direct_full_field_dir:
        comparator_result = _collect_one(
            Path(args.direct_full_field_dir).resolve(),
            output_dir,
            sample_index=args.sample_index,
            base_time_index=args.base_time_index,
            target_lon=args.target_lon,
            target_lat=args.target_lat,
            label=args.direct_label,
            leads=args.leads,
            depth_indices=args.depth_indices,
        )
        # The first implementation uses the same default render settings for
        # both checkpoints; render the requested cross-model comparison here.
        _render_cross_model_figures(output_dir, main_result, comparator_result, args.leads)

    summary = {
        "status": "completed",
        "output_dir": str(output_dir),
        "main_model": main_result["metadata"],
        "direct_full_field_model": (
            comparator_result["metadata"] if args.direct_full_field_dir else None
        ),
        "render_settings": {
            "depth_indices": args.depth_indices,
            "leads": args.leads,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "run_summary.json", summary)
    (output_dir / "_SUCCESS").write_text(
        summary["completed_at_utc"] + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
