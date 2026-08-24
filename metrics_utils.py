"""Shared forecast metric helpers for training and prediction reports."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def _safe_float(value):
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _metric_sufficient_statistics(
    pred: np.ndarray,
    target: np.ndarray,
    chunk_elements: int = 4_000_000,
    *,
    center_spatial: bool = False,
    shared_reference: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Accumulate scalar metric statistics without materializing giant flats.

    ``np.corrcoef(pred.reshape(-1), target.reshape(-1))`` constructs a 2xN
    temporary.  During full-map evaluation that was repeated for every
    variable, lead, baseline, and metric space, resulting in tens of GiB of
    memory traffic.  These additive statistics are sufficient for exactly the
    same MAE/MSE/correlation/R2 definitions and keep peak temporary memory
    bounded by ``chunk_elements``.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} != {target.shape}")
    if shared_reference is not None:
        shared_reference = np.asarray(shared_reference)
        if shared_reference.shape != pred.shape:
            raise ValueError(
                'shared reference shape mismatch: '
                f'{shared_reference.shape} != {pred.shape}'
            )

    stats = {
        'count': 0,
        'sum_pred': 0.0,
        'sum_target': 0.0,
        'sum_pred_sq': 0.0,
        'sum_target_sq': 0.0,
        'sum_cross': 0.0,
        'sum_abs_error': 0.0,
        'sum_sq_error': 0.0,
    }
    if pred.size == 0:
        return stats

    # Slice only along the leading dimension. Each temporary is therefore
    # bounded even when a channel slice is non-contiguous and must be copied.
    elements_per_item = max(1, int(np.prod(pred.shape[1:], dtype=np.int64)))
    items_per_chunk = max(1, int(chunk_elements) // elements_per_item)
    for start in range(0, pred.shape[0], items_per_chunk):
        stop = min(start + items_per_chunk, pred.shape[0])
        pred_chunk = np.asarray(pred[start:stop], dtype=np.float64)
        target_chunk = np.asarray(target[start:stop], dtype=np.float64)
        if shared_reference is not None:
            reference_chunk = np.asarray(
                shared_reference[start:stop], dtype=np.float64
            )
            pred_chunk = pred_chunk - reference_chunk
            target_chunk = target_chunk - reference_chunk
        if center_spatial:
            pred_chunk = pred_chunk - np.nanmean(
                pred_chunk, axis=(-2, -1), keepdims=True
            )
            target_chunk = target_chunk - np.nanmean(
                target_chunk, axis=(-2, -1), keepdims=True
            )
        pred_chunk = pred_chunk.reshape(-1)
        target_chunk = target_chunk.reshape(-1)
        valid = np.isfinite(pred_chunk) & np.isfinite(target_chunk)
        if not np.all(valid):
            pred_chunk = pred_chunk[valid]
            target_chunk = target_chunk[valid]
        if pred_chunk.size == 0:
            continue
        diff = pred_chunk - target_chunk

        stats['count'] += int(pred_chunk.size)
        stats['sum_pred'] += float(np.sum(pred_chunk, dtype=np.float64))
        stats['sum_target'] += float(np.sum(target_chunk, dtype=np.float64))
        stats['sum_pred_sq'] += float(np.dot(pred_chunk, pred_chunk))
        stats['sum_target_sq'] += float(np.dot(target_chunk, target_chunk))
        stats['sum_cross'] += float(np.dot(pred_chunk, target_chunk))
        stats['sum_abs_error'] += float(np.sum(np.abs(diff), dtype=np.float64))
        stats['sum_sq_error'] += float(np.dot(diff, diff))
    return stats


def _metrics_from_statistics(stats: Dict[str, float]) -> Dict[str, Optional[float]]:
    count = int(stats['count'])
    if count == 0:
        return {key: None for key in ('mse', 'mae', 'rmse', 'correlation', 'r2')}

    mse = stats['sum_sq_error'] / count
    mae = stats['sum_abs_error'] / count
    rmse = np.sqrt(mse)
    centered_pred_ss = stats['sum_pred_sq'] - stats['sum_pred'] ** 2 / count
    centered_target_ss = stats['sum_target_sq'] - stats['sum_target'] ** 2 / count
    centered_cross = stats['sum_cross'] - stats['sum_pred'] * stats['sum_target'] / count
    # Round-off can make a theoretically zero sum of squares slightly negative.
    centered_pred_ss = max(0.0, centered_pred_ss)
    centered_target_ss = max(0.0, centered_target_ss)
    if centered_pred_ss == 0 or centered_target_ss == 0:
        corr = np.nan
    else:
        corr = centered_cross / np.sqrt(centered_pred_ss * centered_target_ss)
        corr = float(np.clip(corr, -1.0, 1.0))
    r2 = (
        np.nan
        if centered_target_ss == 0
        else 1.0 - stats['sum_sq_error'] / centered_target_ss
    )

    return {
        'mse': _safe_float(mse),
        'mae': _safe_float(mae),
        'rmse': _safe_float(rmse),
        'correlation': _safe_float(corr),
        'r2': _safe_float(r2),
    }


def _merge_statistics(*items: Dict[str, float]) -> Dict[str, float]:
    keys = (
        'count',
        'sum_pred',
        'sum_target',
        'sum_pred_sq',
        'sum_target_sq',
        'sum_cross',
        'sum_abs_error',
        'sum_sq_error',
    )
    merged = {key: 0 for key in keys}
    for stats in items:
        for key in keys:
            merged[key] += stats[key]
    return merged


def _basic_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, Optional[float]]:
    metrics = _metrics_from_statistics(_metric_sufficient_statistics(pred, target))

    return metrics


def _structure_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, Optional[float]]:
    metrics = _basic_metrics(pred, target)
    return {
        'correlation': metrics['correlation'],
        'r2': metrics['r2'],
    }


def _finite_summary(values) -> Dict[str, Optional[float]]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'mean': None, 'median': None}
    return {
        'mean': _safe_float(np.mean(values)),
        'median': _safe_float(np.median(values)),
    }


def resolve_variable_slices(target_variables, channel_slices, total_channels):
    """Resolve and validate an exact target-variable channel partition.

    Scientific metrics cannot infer a multi-variable schema by evenly splitting
    channels: that silently becomes wrong as soon as variables have different
    depth counts.  The only schema-free case accepted is one anonymous/single
    target, for which all channels unambiguously belong to that target.
    """
    names = list(target_variables or ['target'])
    total_channels = int(total_channels)
    if total_channels <= 0:
        raise ValueError('目标通道数必须为正整数')
    if len(set(names)) != len(names):
        raise ValueError(f'目标变量名称重复: {names}')

    raw_slices = channel_slices or {}
    if not raw_slices and len(names) == 1:
        return {names[0]: slice(0, total_channels)}

    resolved = {}
    coverage = np.zeros(total_channels, dtype=np.int8)
    for var_name in names:
        bounds = raw_slices.get(var_name)
        if isinstance(bounds, slice):
            start_ch, end_ch = bounds.start, bounds.stop
        elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
            start_ch, end_ch = bounds[0], bounds[1]
        else:
            raise ValueError(f'目标变量 {var_name!r} 缺少显式 channel slice')
        if start_ch is None or end_ch is None:
            raise ValueError(f'目标变量 {var_name!r} 的 channel slice 必须有明确边界')
        start_ch, end_ch = int(start_ch), int(end_ch)
        if not 0 <= start_ch < end_ch <= total_channels:
            raise ValueError(
                f'目标变量 {var_name!r} 的 channel slice [{start_ch}, {end_ch}) '
                f'超出总通道数 {total_channels}'
            )
        coverage[start_ch:end_ch] += 1
        resolved[var_name] = slice(start_ch, end_ch)

    overlapping = np.flatnonzero(coverage > 1).tolist()
    uncovered = np.flatnonzero(coverage == 0).tolist()
    if overlapping or uncovered:
        raise ValueError(
            '目标 channel slices 必须无重叠且完整覆盖输出通道；'
            f'重叠={overlapping}, 未覆盖={uncovered}'
        )
    return resolved


def _variable_slices(target_variables, channel_slices, total_channels):
    """Backward-compatible private alias for the strict public resolver."""
    return resolve_variable_slices(target_variables, channel_slices, total_channels)


def _depth_slices(ch_slice: slice, depth_values: Optional[Sequence[float]]):
    """Map a variable's channel slice to physical depth labels when possible."""
    if depth_values is None:
        return []
    start = int(ch_slice.start or 0)
    stop = int(ch_slice.stop or start)
    values = list(depth_values)
    if stop - start != len(values):
        return []
    return [
        (f'depth_{idx + 1}', slice(start + idx, start + idx + 1), _safe_float(value))
        for idx, value in enumerate(values)
    ]


def _field_metric_values(
    pred: np.ndarray,
    target: np.ndarray,
    structure_only: bool = False,
    chunk_size: int = 2048,
):
    keys = ('correlation', 'r2') if structure_only else ('rmse', 'correlation', 'r2')
    collected = {key: [] for key in keys}
    if pred.shape != target.shape or pred.ndim != 5:
        return {key: np.asarray([], dtype=np.float64) for key in keys}

    # 每个 (sample, lead, channel) 是一个二维场。原实现逐场调用
    # np.corrcoef，正式评估会产生数十万次 Python/NumPy 小调用。
    # 这里保持同一公式，但将场展平后分块向量化，并限制临时数组大小。
    chunk_size = max(1, int(chunk_size))
    fields_per_sample = max(1, int(np.prod(pred.shape[1:-2], dtype=np.int64)))
    samples_per_chunk = max(1, chunk_size // fields_per_sample)
    field_size = pred.shape[-2] * pred.shape[-1]

    for start in range(0, pred.shape[0], samples_per_chunk):
        stop = min(start + samples_per_chunk, pred.shape[0])
        # Convert only a bounded sample block. Channel views are often
        # non-contiguous, so reshaping the complete array would silently copy
        # the whole validation split.
        pred_chunk = np.asarray(pred[start:stop], dtype=np.float64).reshape(-1, field_size)
        target_chunk = np.asarray(target[start:stop], dtype=np.float64).reshape(-1, field_size)
        valid = np.isfinite(pred_chunk) & np.isfinite(target_chunk)
        count = valid.sum(axis=1).astype(np.float64)
        pred_valid = np.where(valid, pred_chunk, 0.0)
        target_valid = np.where(valid, target_chunk, 0.0)
        diff = np.where(valid, pred_chunk - target_chunk, 0.0)
        sse = np.einsum('ij,ij->i', diff, diff)

        safe_count = np.maximum(count, 1.0)
        sum_pred = pred_valid.sum(axis=1)
        sum_target = target_valid.sum(axis=1)
        pred_ss = np.einsum('ij,ij->i', pred_valid, pred_valid) - sum_pred ** 2 / safe_count
        target_ss = np.einsum('ij,ij->i', target_valid, target_valid) - sum_target ** 2 / safe_count
        covariance = (
            np.einsum('ij,ij->i', pred_valid, target_valid)
            - sum_pred * sum_target / safe_count
        )
        pred_ss = np.maximum(pred_ss, 0.0)
        target_ss = np.maximum(target_ss, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            correlation = covariance / np.sqrt(pred_ss * target_ss)
            r2 = 1.0 - sse / target_ss
        invalid_structure = (count <= 1) | (pred_ss <= 0) | (target_ss <= 0)
        correlation[invalid_structure] = np.nan
        r2[(count == 0) | (target_ss <= 0)] = np.nan

        if not structure_only:
            with np.errstate(divide='ignore', invalid='ignore'):
                rmse = np.sqrt(sse / count)
            rmse[count == 0] = np.nan
            collected['rmse'].append(rmse)
        collected['correlation'].append(correlation)
        collected['r2'].append(r2)

    return {
        key: np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)
        for key, parts in collected.items()
    }


def _macro_field_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    variable_slices,
    allow_unit_aggregation: bool,
    structure_only: bool = False,
) -> Dict:
    """Summarize field RMSE and dimensionless structure metrics.

    Field MSE/MAE are intentionally omitted: for equal-size fields their means
    are identical to globally flattened MSE/MAE and add no information.
    """
    if pred.ndim != 5 or target.ndim != 5:
        return {}

    keys = ('correlation', 'r2') if structure_only else ('rmse', 'correlation', 'r2')
    by_variable = {}
    raw_by_variable = {}
    for var_name, ch_slice in variable_slices.items():
        values = _field_metric_values(
            pred[:, :, ch_slice, :, :],
            target[:, :, ch_slice, :, :],
            structure_only=structure_only,
        )
        raw_by_variable[var_name] = values
        by_variable[var_name] = {key: _finite_summary(items) for key, items in values.items()}

    all_values = {
        key: np.concatenate([
            values[key] for values in raw_by_variable.values() if values[key].size
        ]) if any(values[key].size for values in raw_by_variable.values()) else np.asarray([], dtype=np.float64)
        for key in keys
    }
    result = {
        'by_variable': by_variable,
        'dimensionless_overall': {
            key: _finite_summary(all_values[key]) for key in ('correlation', 'r2')
        },
    }
    if allow_unit_aggregation and not structure_only:
        result['overall'] = {
            key: _finite_summary(items) for key, items in all_values.items()
        }
    return result


def _improvement(model_rmse: Optional[float], baseline_rmse: Optional[float]) -> Optional[float]:
    if model_rmse is None or baseline_rmse is None or baseline_rmse == 0:
        return None
    return _safe_float((baseline_rmse - model_rmse) / baseline_rmse * 100.0)


def _mse_skill(model_mse: Optional[float], baseline_mse: Optional[float]) -> Optional[float]:
    if model_mse is None or baseline_mse is None or baseline_mse == 0:
        return None
    return _safe_float(1.0 - model_mse / baseline_mse)


def _standard_sections(
    pred: np.ndarray,
    target: np.ndarray,
    target_variables,
    channel_slices: Dict[str, slice],
    allow_unit_aggregation: bool,
    structure_only: bool = False,
    depth_values: Optional[Sequence[float]] = None,
    include_macro_field: bool = True,
    include_depth: bool = True,
    center_spatial: bool = False,
    shared_reference: Optional[np.ndarray] = None,
) -> Dict:
    slices = _variable_slices(target_variables, channel_slices, pred.shape[2])
    report = {
        'overall': None,
        'by_variable': {},
        'by_lead': {},
        'by_variable_and_lead': {},
        'by_variable_and_depth': {},
        'by_variable_lead_and_depth': {},
    }

    # Each element is reduced exactly once for scalar metrics. Per-variable,
    # per-lead, and overall results are then assembled from additive sufficient
    # statistics instead of rescanning the same large tensors.
    detailed_stats = {}
    for var_name, ch_slice in slices.items():
        lead_stats = []
        lead_metrics = {}
        depth_entries = _depth_slices(ch_slice, depth_values) if include_depth else []
        depth_lead_stats = {depth_name: [] for depth_name, _, _ in depth_entries}
        depth_lead_metrics = {}
        for lead_idx in range(pred.shape[1]):
            if depth_entries:
                per_depth_stats = []
                per_depth_metrics = {}
                for depth_name, depth_slice, depth_value in depth_entries:
                    depth_stats = _metric_sufficient_statistics(
                        pred[:, lead_idx:lead_idx + 1, depth_slice, :, :],
                        target[:, lead_idx:lead_idx + 1, depth_slice, :, :],
                        center_spatial=center_spatial,
                        shared_reference=(
                            shared_reference[:, lead_idx:lead_idx + 1, depth_slice, :, :]
                            if shared_reference is not None else None
                        ),
                    )
                    depth_lead_stats[depth_name].append(depth_stats)
                    per_depth_stats.append(depth_stats)
                    metrics = _metrics_from_statistics(depth_stats)
                    per_depth_metrics[depth_name] = {
                        'depth': depth_value,
                        **(
                            {key: metrics[key] for key in ('correlation', 'r2')}
                            if structure_only else metrics
                        ),
                    }
                stats = _merge_statistics(*per_depth_stats)
                depth_lead_metrics[f'lead_{lead_idx + 1}'] = per_depth_metrics
            else:
                stats = _metric_sufficient_statistics(
                    pred[:, lead_idx:lead_idx + 1, ch_slice, :, :],
                    target[:, lead_idx:lead_idx + 1, ch_slice, :, :],
                    center_spatial=center_spatial,
                    shared_reference=(
                        shared_reference[:, lead_idx:lead_idx + 1, ch_slice, :, :]
                        if shared_reference is not None else None
                    ),
                )
            lead_stats.append(stats)
            metrics = _metrics_from_statistics(stats)
            lead_metrics[f'lead_{lead_idx + 1}'] = (
                {key: metrics[key] for key in ('correlation', 'r2')}
                if structure_only else metrics
            )
        detailed_stats[var_name] = lead_stats
        variable_metrics = _metrics_from_statistics(_merge_statistics(*lead_stats))
        report['by_variable'][var_name] = (
            {key: variable_metrics[key] for key in ('correlation', 'r2')}
            if structure_only else variable_metrics
        )
        report['by_variable_and_lead'][var_name] = lead_metrics
        if depth_entries:
            report['by_variable_lead_and_depth'][var_name] = depth_lead_metrics
            per_depth = {}
            for depth_name, _, depth_value in depth_entries:
                metrics = _metrics_from_statistics(
                    _merge_statistics(*depth_lead_stats[depth_name])
                )
                per_depth[depth_name] = {
                    'depth': depth_value,
                    **(
                        {key: metrics[key] for key in ('correlation', 'r2')}
                        if structure_only else metrics
                    ),
                }
            report['by_variable_and_depth'][var_name] = per_depth

    if allow_unit_aggregation:
        all_stats = [stats for per_var in detailed_stats.values() for stats in per_var]
        expected_count = int(pred.size)
        covered_count = sum(int(stats['count']) for stats in all_stats)
        overall_stats = (
            _merge_statistics(*all_stats)
            if covered_count == expected_count
            else _metric_sufficient_statistics(
                pred,
                target,
                center_spatial=center_spatial,
                shared_reference=shared_reference,
            )
        )
        overall_metrics = _metrics_from_statistics(overall_stats)
        report['overall'] = (
            {key: overall_metrics[key] for key in ('correlation', 'r2')}
            if structure_only else overall_metrics
        )

        for lead_idx in range(pred.shape[1]):
            per_lead = [stats[lead_idx] for stats in detailed_stats.values()]
            covered = sum(int(stats['count']) for stats in per_lead)
            expected = int(pred[:, lead_idx:lead_idx + 1].size)
            lead_stats = (
                _merge_statistics(*per_lead)
                if covered == expected
                else _metric_sufficient_statistics(
                    pred[:, lead_idx:lead_idx + 1],
                    target[:, lead_idx:lead_idx + 1],
                    center_spatial=center_spatial,
                    shared_reference=(
                        shared_reference[:, lead_idx:lead_idx + 1]
                        if shared_reference is not None else None
                    ),
                )
            )
            lead_metrics = _metrics_from_statistics(lead_stats)
            report['by_lead'][f'lead_{lead_idx + 1}'] = (
                {key: lead_metrics[key] for key in ('correlation', 'r2')}
                if structure_only else lead_metrics
            )

    if include_macro_field:
        if center_spatial or shared_reference is not None:
            raise ValueError(
                'transformed reports must disable macro-field rescans'
            )
        report['macro_field'] = _macro_field_metrics(
            pred,
            target,
            slices,
            allow_unit_aggregation=allow_unit_aggregation,
            structure_only=structure_only,
        )
    return report


def _comparison_metrics(model_metrics, baseline_metrics):
    return {
        'rmse_improvement_pct': _improvement(
            model_metrics.get('rmse'), baseline_metrics.get('rmse')
        ),
        'mse_skill': _mse_skill(
            model_metrics.get('mse'), baseline_metrics.get('mse')
        ),
    }


def _baseline_comparison(
    model_report: Dict,
    baseline_report: Dict,
    allow_unit_aggregation: bool,
) -> Dict:
    comparison = {
        'overall': None,
        'by_variable': {},
        'by_variable_and_lead': {},
        'by_variable_and_depth': {},
        'by_variable_lead_and_depth': {},
    }
    if allow_unit_aggregation:
        comparison['overall'] = _comparison_metrics(
            model_report['overall'],
            baseline_report['overall'],
        )

    macro_rmse_improvement = []
    macro_mse_skill = []
    for var_name, model_metrics in model_report['by_variable'].items():
        baseline_metrics = baseline_report['by_variable'][var_name]
        comparison['by_variable'][var_name] = _comparison_metrics(
            model_metrics,
            baseline_metrics,
        )
        lead_comparison = {}
        for lead_name, model_lead_metrics in model_report['by_variable_and_lead'][var_name].items():
            values = _comparison_metrics(
                model_lead_metrics,
                baseline_report['by_variable_and_lead'][var_name][lead_name],
            )
            lead_comparison[lead_name] = values
            if values['rmse_improvement_pct'] is not None:
                macro_rmse_improvement.append(values['rmse_improvement_pct'])
            if values['mse_skill'] is not None:
                macro_mse_skill.append(values['mse_skill'])
        comparison['by_variable_and_lead'][var_name] = lead_comparison

        depth_comparison = {}
        for depth_name, model_depth_metrics in model_report.get(
            'by_variable_and_depth', {}
        ).get(var_name, {}).items():
            values = _comparison_metrics(
                model_depth_metrics,
                baseline_report['by_variable_and_depth'][var_name][depth_name],
            )
            values['depth'] = model_depth_metrics.get('depth')
            depth_comparison[depth_name] = values
        if depth_comparison:
            comparison['by_variable_and_depth'][var_name] = depth_comparison

        lead_depth_comparison = {}
        for lead_name, model_depths in model_report.get(
            'by_variable_lead_and_depth', {}
        ).get(var_name, {}).items():
            per_depth = {}
            for depth_name, model_depth_metrics in model_depths.items():
                values = _comparison_metrics(
                    model_depth_metrics,
                    baseline_report['by_variable_lead_and_depth'][var_name][lead_name][depth_name],
                )
                values['depth'] = model_depth_metrics.get('depth')
                per_depth[depth_name] = values
            lead_depth_comparison[lead_name] = per_depth
        if lead_depth_comparison:
            comparison['by_variable_lead_and_depth'][var_name] = lead_depth_comparison

    comparison['macro'] = {
        'rmse_improvement_pct': _finite_summary(macro_rmse_improvement),
        'mse_skill': _finite_summary(macro_mse_skill),
    }
    return comparison


def compute_metric_report(
    pred: np.ndarray,
    target: np.ndarray,
    target_variables,
    channel_slices: Optional[Dict[str, slice]] = None,
    baselines: Optional[Dict[str, np.ndarray]] = None,
    metric_space: str = 'physical',
    depth_values: Optional[Sequence[float]] = None,
    include_depth: bool = True,
) -> Dict:
    """Compute unit-safe forecast metrics and baseline skill reports.

    Physical metrics from multiple variables are never aggregated because their
    units differ. Normalized/anomaly metrics may be aggregated across variables.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    target_variables = list(target_variables or [])
    channel_slices = channel_slices or {}
    if pred.shape != target.shape:
        raise ValueError(f"预测与目标形状不一致: {pred.shape} != {target.shape}")
    if pred.ndim != 5:
        raise ValueError(f"指标计算期望 5D 数组 (N,T,C,H,W)，收到: {pred.shape}")
    if metric_space not in {'physical', 'normalized'}:
        raise ValueError(f"不支持的 metric_space: {metric_space}")

    variable_slices = _variable_slices(target_variables, channel_slices, pred.shape[2])
    allow_unit_aggregation = metric_space == 'normalized' or len(variable_slices) <= 1
    report = _standard_sections(
        pred,
        target,
        target_variables,
        channel_slices,
        allow_unit_aggregation=allow_unit_aggregation,
        depth_values=depth_values,
        include_depth=include_depth,
    )
    report['metric_space'] = metric_space
    report['unit_aggregation_allowed'] = allow_unit_aggregation
    report['aggregation_note'] = (
        "overall/by_lead aggregate normalized channels with compatible units."
        if allow_unit_aggregation
        else "Physical variables have different units; use by_variable and by_variable_and_lead."
    )

    report['spatial_mean_removed'] = _standard_sections(
        pred,
        target,
        target_variables,
        channel_slices,
        allow_unit_aggregation=allow_unit_aggregation,
        structure_only=True,
        depth_values=depth_values,
        include_macro_field=False,
        include_depth=include_depth,
        center_spatial=True,
    )

    if baselines:
        baseline_report = {}
        comparison = {}
        for name, baseline_pred in baselines.items():
            baseline_pred = np.asarray(baseline_pred)
            if baseline_pred.shape != target.shape:
                raise ValueError(
                    f"baseline {name} 形状不一致: {baseline_pred.shape} != {target.shape}"
                )
            baseline_report[name] = _standard_sections(
                baseline_pred,
                target,
                target_variables,
                channel_slices,
                allow_unit_aggregation=allow_unit_aggregation,
                depth_values=depth_values,
                include_macro_field=False,
                include_depth=include_depth,
            )
            comparison[name] = _baseline_comparison(
                report,
                baseline_report[name],
                allow_unit_aggregation=allow_unit_aggregation,
            )
        report['baselines'] = baseline_report
        report['comparison'] = comparison

        climatology = baselines.get('climatology')
        if climatology is not None:
            climatology = np.asarray(climatology)
            report['climatology_residual'] = _standard_sections(
                pred,
                target,
                target_variables,
                channel_slices,
                allow_unit_aggregation=allow_unit_aggregation,
                structure_only=True,
                depth_values=depth_values,
                include_macro_field=False,
                include_depth=include_depth,
                shared_reference=climatology,
            )
            report['climatology_residual']['note'] = (
                "Only correlation/R2 are reported: subtracting the same climatology "
                "does not change MAE/MSE/RMSE."
            )

    return report


def compute_sample_group_report(
    pred: np.ndarray,
    target: np.ndarray,
    sample_group_ids: Sequence,
    target_variables,
    channel_slices: Optional[Dict[str, slice]] = None,
    baselines: Optional[Dict[str, np.ndarray]] = None,
    metric_space: str = 'physical',
    depth_values: Optional[Sequence[float]] = None,
    include_depth: bool = False,
) -> Dict:
    """Compute compact, paired metrics for sample-level blocks (e.g. origins).

    The report deliberately omits macro-field diagnostics. Its purpose is to
    preserve independent temporal blocks for later paired confidence intervals
    without storing hundreds of MiB of raw predictions per experiment.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    group_ids = np.asarray(sample_group_ids)
    if pred.shape != target.shape or pred.ndim != 5:
        raise ValueError('grouped metrics require matching 5D prediction/target arrays')
    if group_ids.ndim != 1 or group_ids.shape[0] != pred.shape[0]:
        raise ValueError('sample_group_ids must be one-dimensional with length N')

    channel_slices = channel_slices or {}
    slices = _variable_slices(target_variables, channel_slices, pred.shape[2])
    allow_unit_aggregation = metric_space == 'normalized' or len(slices) <= 1
    output = {
        'metric_space': metric_space,
        'group_count': int(np.unique(group_ids).size),
        'groups': {},
    }
    for group_id in np.unique(group_ids):
        positions = np.flatnonzero(group_ids == group_id)
        contiguous = positions.size > 0 and (
            positions.size == 1 or np.all(np.diff(positions) == 1)
        )
        selector = (
            slice(int(positions[0]), int(positions[-1]) + 1)
            if contiguous else positions
        )
        model_report = _standard_sections(
            pred[selector],
            target[selector],
            target_variables,
            channel_slices,
            allow_unit_aggregation=allow_unit_aggregation,
            depth_values=depth_values,
            include_macro_field=False,
            include_depth=include_depth,
        )
        item = {
            'sample_count': int(positions.size),
            'metrics': model_report,
            'baseline_comparison': {},
        }
        if baselines:
            for name, baseline_pred in baselines.items():
                baseline_pred = np.asarray(baseline_pred)
                if baseline_pred.shape != target.shape:
                    raise ValueError(
                        f'baseline {name} shape mismatch: {baseline_pred.shape} != {target.shape}'
                    )
                baseline_report = _standard_sections(
                    baseline_pred[selector],
                    target[selector],
                    target_variables,
                    channel_slices,
                    allow_unit_aggregation=allow_unit_aggregation,
                    depth_values=depth_values,
                    include_macro_field=False,
                    include_depth=include_depth,
                )
                item['baseline_comparison'][name] = _baseline_comparison(
                    model_report,
                    baseline_report,
                    allow_unit_aggregation=allow_unit_aggregation,
                )
        output['groups'][str(group_id)] = item
    return output


def compute_period_group_report(
    pred: np.ndarray,
    target: np.ndarray,
    period_ids: np.ndarray,
    target_variables,
    channel_slices: Optional[Dict[str, slice]] = None,
    baselines: Optional[Dict[str, np.ndarray]] = None,
    metric_space: str = 'physical',
    depth_values: Optional[Sequence[float]] = None,
    include_depth: bool = False,
) -> Dict:
    """Aggregate metrics by a sample/lead label such as calendar month."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    period_ids = np.asarray(period_ids)
    if pred.shape != target.shape or pred.ndim != 5:
        raise ValueError('period metrics require matching 5D prediction/target arrays')
    if period_ids.shape != pred.shape[:2]:
        raise ValueError('period_ids must have shape (N, prediction_length)')

    flattened_pred = pred.reshape(-1, 1, *pred.shape[2:])
    flattened_target = target.reshape(-1, 1, *target.shape[2:])
    flattened_periods = period_ids.reshape(-1)
    flattened_baselines = {
        name: np.asarray(values).reshape(-1, 1, *pred.shape[2:])
        for name, values in (baselines or {}).items()
    }
    return compute_sample_group_report(
        flattened_pred,
        flattened_target,
        flattened_periods,
        target_variables,
        channel_slices=channel_slices,
        baselines=flattened_baselines,
        metric_space=metric_space,
        depth_values=depth_values,
        include_depth=include_depth,
    )
