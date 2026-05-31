"""Shared forecast metric helpers for training and prediction reports."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _safe_float(value):
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _basic_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, Optional[float]]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = pred - target
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    if pred_flat.size == 0 or target_flat.size == 0:
        corr = np.nan
        r2 = np.nan
    else:
        corr = np.corrcoef(pred_flat, target_flat)[0, 1]
        target_mean = np.mean(target_flat)
        ss_tot = np.sum((target_flat - target_mean) ** 2)
        ss_res = np.sum((pred_flat - target_flat) ** 2)
        r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return {
        'mse': _safe_float(mse),
        'mae': _safe_float(mae),
        'rmse': _safe_float(rmse),
        'correlation': _safe_float(corr),
        'r2': _safe_float(r2),
    }


def _improvement(model_rmse: Optional[float], baseline_rmse: Optional[float]) -> Optional[float]:
    if model_rmse is None or baseline_rmse is None or baseline_rmse == 0:
        return None
    return _safe_float((baseline_rmse - model_rmse) / baseline_rmse * 100.0)


def compute_metric_report(
    pred: np.ndarray,
    target: np.ndarray,
    target_variables,
    channel_slices: Optional[Dict[str, slice]] = None,
    baselines: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """Compute overall, per-variable, per-lead, and baseline comparison metrics."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_variables = list(target_variables or [])
    channel_slices = channel_slices or {}

    report = {
        'overall': _basic_metrics(pred, target),
        'by_variable': {},
        'by_lead': {},
    }

    var_count = max(1, len(target_variables))
    total_channels = pred.shape[2]
    fallback_channels_per_var = total_channels // var_count

    for var_idx, var_name in enumerate(target_variables or ['target']):
        ch_slice = channel_slices.get(var_name)
        if ch_slice is None:
            start_ch = var_idx * fallback_channels_per_var
            end_ch = (var_idx + 1) * fallback_channels_per_var
        else:
            start_ch = ch_slice.start or 0
            end_ch = ch_slice.stop or start_ch
        if end_ch <= start_ch:
            continue
        report['by_variable'][var_name] = _basic_metrics(
            pred[:, :, start_ch:end_ch, :, :],
            target[:, :, start_ch:end_ch, :, :],
        )

    for lead_idx in range(pred.shape[1]):
        report['by_lead'][f'lead_{lead_idx + 1}'] = _basic_metrics(
            pred[:, lead_idx],
            target[:, lead_idx],
        )

    if baselines:
        baseline_report = {}
        comparison = {}
        model_rmse = report['overall']['rmse']
        for name, baseline_pred in baselines.items():
            baseline_metrics = _basic_metrics(baseline_pred, target)
            baseline_report[name] = baseline_metrics
            comparison[f'rmse_improvement_vs_{name}_pct'] = _improvement(
                model_rmse,
                baseline_metrics.get('rmse'),
            )
        report['baselines'] = baseline_report
        report['comparison'] = comparison

    return report

