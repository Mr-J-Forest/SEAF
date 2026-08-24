#!/usr/bin/env python3
"""Run dense overlap-tile inference and render unit-safe full-map figures."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from config import DEFAULT_CONFIG
from metrics_utils import compute_metric_report, resolve_variable_slices
from predict import SmartOceanPredictor


def _finite_limits(*arrays):
    values = np.concatenate([np.asarray(arr)[np.isfinite(arr)] for arr in arrays])
    if values.size == 0:
        return 0.0, 1.0
    return float(values.min()), float(values.max())


def _sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _build_provenance(predictor, model_dir):
    project_root = Path(__file__).resolve().parent
    checkpoint = Path(predictor.model_path).resolve()
    model_config = Path(model_dir).resolve() / predictor.config['config_filename']
    data_path = Path(predictor.config['data_path'])
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()
    data_stat = data_path.stat()
    return {
        'source_state': _read_json(project_root / 'source_state.json'),
        'training_run_summary': _read_json(Path(model_dir) / 'run_summary.json'),
        'checkpoint': {
            'path': str(checkpoint),
            'size_bytes': checkpoint.stat().st_size,
            'sha256': _sha256(checkpoint),
        },
        'model_config': {
            'path': str(model_config),
            'sha256': _sha256(model_config),
        },
        'dataset': {
            'path': str(data_path),
            'realpath': os.path.realpath(data_path),
            'size_bytes': data_stat.st_size,
            'mtime_ns': data_stat.st_mtime_ns,
        },
    }


def _target_slices(result):
    target_vars = list(result.get('target_variables') or ['target'])
    raw_slices = result.get('target_channel_slices') or {}
    total_channels = result['blended_pred'].shape[0]
    return resolve_variable_slices(target_vars, raw_slices, total_channels)


def plot_full_map_result(result, pred_step, output_dir, model_index):
    full_lons = result['lons']
    full_lats = result['lats']
    blended_pred = result['blended_pred']
    blended_target = result['blended_target']
    levels = np.asarray(result.get('levels', []))

    for var_name, ch_slice in _target_slices(result).items():
        var_pred = blended_pred[ch_slice]
        var_target = blended_target[ch_slice]
        n_depth = var_pred.shape[0]
        var_levels = levels[:n_depth] if levels.size >= n_depth else np.arange(n_depth)
        ncols = min(n_depth, 5)
        nrows = (n_depth + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 4.5, nrows * 8), squeeze=False)
        fig.suptitle(
            f'{var_name} — Prediction Step {pred_step + 1} '
            '(100%-ocean overlap-tile mosaic)',
            fontsize=14,
        )
        for depth_idx in range(n_depth):
            row, col = divmod(depth_idx, ncols)
            pred_slice = var_pred[depth_idx]
            target_slice = var_target[depth_idx]
            vmin, vmax = _finite_limits(pred_slice, target_slice)
            for plot_row, data, label in (
                (row * 2, pred_slice, 'Prediction'),
                (row * 2 + 1, target_slice, 'Target'),
            ):
                ax = axes[plot_row, col]
                im = ax.pcolormesh(full_lons, full_lats, data, shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
                ax.set_title(f'{label}  depth={var_levels[depth_idx]:g}m')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                plt.colorbar(im, ax=ax, shrink=0.8)
        for depth_idx in range(n_depth, nrows * ncols):
            row, col = divmod(depth_idx, ncols)
            axes[row * 2, col].axis('off')
            axes[row * 2 + 1, col].axis('off')
        plt.tight_layout()
        path = os.path.join(output_dir, f'fullmap_{model_index}_step{pred_step + 1}_{var_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4), squeeze=False)
        fig.suptitle(f'{var_name} Error (Prediction - Target), Step {pred_step + 1}', fontsize=14)
        for depth_idx in range(n_depth):
            row, col = divmod(depth_idx, ncols)
            error = var_pred[depth_idx] - var_target[depth_idx]
            finite = np.abs(error[np.isfinite(error)])
            max_err = max(float(finite.max()), 1e-12) if finite.size else 1.0
            ax = axes[row, col]
            im = ax.pcolormesh(full_lons, full_lats, error, shading='auto', cmap='RdBu_r', vmin=-max_err, vmax=max_err)
            ax.set_title(f'depth={var_levels[depth_idx]:g}m')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, shrink=0.8)
        for depth_idx in range(n_depth, nrows * ncols):
            row, col = divmod(depth_idx, ncols)
            axes[row, col].axis('off')
        plt.tight_layout()
        path = os.path.join(output_dir, f'fullmap_{model_index}_step{pred_step + 1}_{var_name}_error.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

    coverage = np.asarray(result.get('coverage_mask', result['weight_sum'] > 0), dtype=bool)
    ocean = np.asarray(result.get('ocean_domain_mask', coverage), dtype=bool)
    coverage_class = np.zeros_like(coverage, dtype=np.uint8)
    coverage_class[ocean] = 1
    coverage_class[ocean & coverage] = 2
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(
        full_lons,
        full_lats,
        coverage_class,
        shading='auto',
        cmap=ListedColormap(['#d9d9d9', '#f4a261', '#2a9d8f']),
        vmin=0,
        vmax=2,
    )
    ocean_fraction = result.get('ocean_coverage_fraction')
    ocean_text = (
        f', ocean={ocean_fraction * 100:.2f}%'
        if ocean_fraction is not None else ''
    )
    ax.set_title(
        f'100%-ocean window coverage (domain={coverage.mean() * 100:.2f}%{ocean_text})'
    )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    colorbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(['land', 'uncovered ocean', 'covered ocean'])
    path = os.path.join(output_dir, f'fullmap_{model_index}_step{pred_step + 1}_coverage.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def save_full_map_result(result, pred_step, output_dir, model_label, provenance):
    """Persist numeric fields and provenance; figures alone are not evidence."""
    prefix = f'fullmap_{model_label}_step{pred_step + 1}'
    array_path = os.path.join(output_dir, prefix + '.npz')
    prediction = np.asarray(result['blended_pred'], dtype=np.float32)
    target = np.asarray(result['blended_target'], dtype=np.float32)
    np.savez_compressed(
        array_path,
        prediction=prediction,
        target=target,
        error=prediction - target,
        weight_sum=np.asarray(result['weight_sum'], dtype=np.float32),
        coverage_mask=np.asarray(result['coverage_mask'], dtype=bool),
        ocean_domain_mask=np.asarray(result['ocean_domain_mask'], dtype=bool),
        lons=np.asarray(result['lons']),
        lats=np.asarray(result['lats']),
        levels=np.asarray(result.get('levels', []), dtype=np.float32),
    )
    channel_slices = {
        name: slice(int(bounds[0]), int(bounds[1]))
        for name, bounds in result.get('target_channel_slices', {}).items()
    }
    physical_metrics = compute_metric_report(
        prediction[np.newaxis, np.newaxis, ...],
        target[np.newaxis, np.newaxis, ...],
        list(result.get('target_variables', [])),
        channel_slices=channel_slices,
        metric_space='physical',
        depth_values=[float(value) for value in result.get('levels', [])],
    )
    metadata = {
        key: value for key, value in result.items()
        if key not in {
            'blended_pred', 'blended_target', 'weight_sum', 'coverage_mask',
            'ocean_domain_mask',
            'lons', 'lats', 'levels'
        }
    }
    metadata.update({
        'model_label': model_label,
        'array_file': os.path.basename(array_path),
        'prediction_units': 'physical by target variable',
        'saved_at_utc': datetime.now(timezone.utc).isoformat(),
        'provenance': provenance,
        'physical_metrics_on_covered_cells': physical_metrics,
    })
    metadata_path = os.path.join(output_dir, prefix + '.json')
    with open(metadata_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"  Saved numeric result: {array_path}")
    return array_path, metadata_path


def main():
    parser = argparse.ArgumentParser(description="Full-map overlap-tile prediction")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', type=int, help='Model index in outputs/results/')
    model_group.add_argument('--model_dir', type=str, help='Deterministic training result directory')
    parser.add_argument('--steps', type=int, nargs='+', default=[0, 2, 4], help='0-indexed lead steps')
    parser.add_argument('--base_time_index', type=int, default=None,
                        help='历史序列最后一个全局时间索引；默认使用测试段首个可预测时刻')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.model_dir:
        model_dir = os.path.abspath(args.model_dir)
        if not os.path.isdir(model_dir):
            print(f"Error: model directory does not exist: {model_dir}")
            sys.exit(1)
        model_label = os.path.basename(model_dir)
    else:
        results_base = config['results_dir']
        candidates = [d for d in os.listdir(results_base) if d.startswith(f'{args.model}_results_')]
        if not candidates:
            print(f"Error: no results directory for model {args.model} in {results_base}")
            sys.exit(1)
        model_dir = os.path.join(results_base, sorted(candidates)[-1])
        model_label = str(args.model)
    output_dir = args.output_dir or os.path.join(model_dir, 'full_map')
    os.makedirs(output_dir, exist_ok=True)

    predictor = SmartOceanPredictor(
        model_index=args.model,
        model_dir=args.model_dir,
        config=config,
        output_dir=output_dir,
    )
    pred_len = int(predictor.config['prediction_length'])
    provenance = _build_provenance(predictor, model_dir)
    started = time.time()
    completed_steps = []
    for step in args.steps:
        if not 0 <= step < pred_len:
            print(f"Skip invalid step {step}; valid range is [0, {pred_len - 1}]")
            continue
        print(f"\n--- Full-map prediction, step {step + 1}/{pred_len} ---")
        result = predictor.predict_full_map(
            base_time_index=args.base_time_index,
            pred_step=step,
        )
        array_path, metadata_path = save_full_map_result(
            result, step, output_dir, model_label, provenance
        )
        plot_full_map_result(result, step, output_dir, model_label)
        completed_steps.append({
            'prediction_step': int(step),
            'coverage_fraction': float(result['coverage_fraction']),
            'ocean_coverage_fraction': result['ocean_coverage_fraction'],
            'valid_windows': int(result['valid_windows']),
            'array_file': os.path.basename(array_path),
            'metadata_file': os.path.basename(metadata_path),
        })

    if not completed_steps:
        raise RuntimeError("没有合法的预测步，未生成全图结果")

    summary = {
        'status': 'completed',
        'model_dir': os.path.abspath(model_dir),
        'model_label': model_label,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'wall_time_seconds': time.time() - started,
        'steps': completed_steps,
        'provenance': provenance,
    }
    with open(os.path.join(output_dir, 'run_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, '_SUCCESS'), 'w', encoding='utf-8') as handle:
        handle.write(summary['completed_at_utc'] + '\n')

    print(f"\nDone. All figures saved to {output_dir}")


if __name__ == '__main__':
    main()
