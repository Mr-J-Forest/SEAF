#!/usr/bin/env python3
"""Audit the immutable data/split/window protocol without preprocessing caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG, load_config, merge_configs, validate_config
from data_loader import OceanDataset


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def time_label(value) -> str:
    return str(np.datetime_as_string(value, unit='D')) if np.issubdtype(
        np.asarray(value).dtype, np.datetime64
    ) else str(value)


def read_json_if_available(path: Path):
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def split_summary(total_steps: int, config: dict, window_count: int) -> dict:
    sequence_length = int(config['sequence_length'])
    prediction_length = int(config['prediction_length'])
    train_end = int(total_steps * float(config['train_ratio']))
    val_end = int(total_steps * (
        float(config['train_ratio']) + float(config['val_ratio'])
    ))
    boundaries = {
        'train': (0, train_end - 1),
        'val': (train_end, val_end - 1),
        'test': (val_end, total_steps - 1),
    }
    output = {}
    context_policy = config.get('split_context_policy', 'carry_history')
    required_length = sequence_length + prediction_length
    for name, (segment_start, segment_end) in boundaries.items():
        if name == 'train' or context_policy == 'strict_segment':
            earliest_start = segment_start
        else:
            earliest_start = max(0, segment_start - sequence_length)
        latest_start = segment_end - required_length + 1
        origins = []
        for history_start in range(earliest_start, latest_start + 1):
            target_start = history_start + sequence_length
            target_end = target_start + prediction_length - 1
            if target_start >= segment_start and target_end <= segment_end:
                origins.append({
                    'history_start': int(history_start),
                    'history_end': int(target_start - 1),
                    'target_start': int(target_start),
                    'target_end': int(target_end),
                })
        output[name] = {
            'segment_start': int(segment_start),
            'segment_end': int(segment_end),
            'origin_count': len(origins),
            'sample_count': int(len(origins) * window_count),
            'first_origin': origins[0] if origins else None,
            'last_origin': origins[-1] if origins else None,
        }
    return output


def audit_window_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    ocean_mask: np.ndarray,
    lon_span: float,
    lat_span: float,
    stride_lon: float,
    stride_lat: float,
    ocean_threshold: float,
) -> dict:
    lon_starts = OceanDataset._window_starts(
        float(lons.min()), float(lons.max()), lon_span, stride_lon
    )
    lat_starts = OceanDataset._window_starts(
        float(lats.min()), float(lats.max()), lat_span, stride_lat
    )
    coverage = np.zeros_like(ocean_mask, dtype=bool)
    valid_ranges = []
    window_shapes = set()
    for lon_start in lon_starts:
        lon_idx = np.flatnonzero((lons >= lon_start) & (lons <= lon_start + lon_span))
        for lat_start in lat_starts:
            lat_idx = np.flatnonzero((lats >= lat_start) & (lats <= lat_start + lat_span))
            if not len(lon_idx) or not len(lat_idx):
                continue
            window_shapes.add((int(len(lat_idx)), int(len(lon_idx))))
            window_mask = ocean_mask[np.ix_(lat_idx, lon_idx)]
            if float(window_mask.mean()) < ocean_threshold:
                continue
            coverage[np.ix_(lat_idx, lon_idx)] = True
            valid_ranges.append({
                'lon_range': [float(lon_start), float(lon_start + lon_span)],
                'lat_range': [float(lat_start), float(lat_start + lat_span)],
            })

    ocean_points = int(ocean_mask.sum())
    covered_points = int(coverage.sum())
    terminal_lon = float(lons.max() - lon_span)
    terminal_lat = float(lats.max() - lat_span)
    return {
        'policy': OceanDataset._WINDOW_GRID_POLICY,
        'stride_lon': float(stride_lon),
        'stride_lat': float(stride_lat),
        'longitude_anchor_count': len(lon_starts),
        'latitude_anchor_count': len(lat_starts),
        'candidate_window_count': int(len(lon_starts) * len(lat_starts)),
        'valid_pure_ocean_window_count': len(valid_ranges),
        'window_grid_shapes': [list(shape) for shape in sorted(window_shapes)],
        'terminal_longitude_anchor': terminal_lon,
        'terminal_latitude_anchor': terminal_lat,
        'terminal_longitude_anchor_present': bool(
            lon_starts and np.isclose(lon_starts[-1], terminal_lon)
        ),
        'terminal_latitude_anchor_present': bool(
            lat_starts and np.isclose(lat_starts[-1], terminal_lat)
        ),
        'covered_grid_points': covered_points,
        'full_domain_coverage_fraction': covered_points / int(coverage.size),
        'ocean_grid_coverage_fraction': (
            covered_points / ocean_points if ocean_points else None
        ),
        'valid_ranges': valid_ranges,
    }


def audit_mask_stability(dataset: xr.Dataset, variables: list[str], reference: np.ndarray) -> dict:
    report = {}
    for name in variables:
        if name not in dataset.data_vars:
            report[name] = {'available': False}
            continue
        stable = True
        mismatch_count = 0
        finite_count = 0
        total_count = 0
        finite_by_time = []
        data = dataset[name]
        for time_index in range(int(data.sizes.get('TIME', 1))):
            block = np.asarray(data.isel(TIME=time_index).values)
            finite = np.isfinite(block)
            finite_by_time.append(int(finite.sum()))
            finite_count += int(finite.sum())
            total_count += int(finite.size)
            if finite.ndim == 2:
                finite = finite[np.newaxis, ...]
            mismatch_count += int(np.count_nonzero(finite != reference[np.newaxis, ...]))
            if mismatch_count:
                stable = False
        report[name] = {
            'available': True,
            'finite_fraction': finite_count / total_count if total_count else None,
            'mask_matches_reference_for_all_selected_times_and_levels': stable,
            'mask_mismatch_count': mismatch_count,
            'all_missing_time_indices': [
                index for index, count in enumerate(finite_by_time) if count == 0
            ],
            'all_missing_time_labels': [
                time_label(dataset.TIME.values[index])
                for index, count in enumerate(finite_by_time) if count == 0
            ],
            'finite_points_per_time_min': min(finite_by_time) if finite_by_time else None,
            'finite_points_per_time_max': max(finite_by_time) if finite_by_time else None,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/full.json')
    parser.add_argument('--data', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--hash-file', action='store_true')
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    config_path = (project_root / args.config).resolve()
    config = merge_configs(load_config(config_path), DEFAULT_CONFIG.copy())
    validate_config(config)
    data_path = Path(args.data or config['data_path'])
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    stat = data_path.stat()
    with xr.open_dataset(data_path) as raw:
        dataset = raw.sel(LEVEL=slice(*config['depth_range']))
        lons = np.asarray(dataset.LONGITUDE.values, dtype=np.float64)
        lats = np.asarray(dataset.LATITUDE.values, dtype=np.float64)
        levels = np.asarray(dataset.LEVEL.values, dtype=np.float64)
        times = np.asarray(dataset.TIME.values)
        reference_mask = np.isfinite(
            np.asarray(dataset['TEMP'].isel(TIME=0, LEVEL=0).values)
        )

        lon_span = float(config['lon_range'][1] - config['lon_range'][0])
        lat_span = float(config['lat_range'][1] - config['lat_range'][0])
        canonical = audit_window_grid(
            lons, lats, reference_mask, lon_span, lat_span,
            float(config['train_stride_lon']), float(config['train_stride_lat']),
            float(config['ocean_threshold']),
        )
        dense = audit_window_grid(
            lons, lats, reference_mask, lon_span, lat_span,
            float(config['inference_stride_lon']), float(config['inference_stride_lat']),
            float(config['ocean_threshold']),
        )
        variables_to_check = sorted(set(config['input_variables']) | set(config['target_variables']))
        mask_stability = audit_mask_stability(dataset, variables_to_check, reference_mask)

        payload = {
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'config_path': str(config_path),
            'config_sha256': file_sha256(config_path),
            'resolved_config_sha256': hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode('utf-8')
            ).hexdigest(),
            'audit_script_sha256': file_sha256(Path(__file__).resolve()),
            'source_state': read_json_if_available(project_root / 'source_state.json'),
            'data_identity': {
                'path': str(data_path),
                'size_bytes': int(stat.st_size),
                'mtime_ns': int(stat.st_mtime_ns),
                'sha256': file_sha256(data_path) if args.hash_file else None,
            },
            'dimensions': {name: int(size) for name, size in dataset.sizes.items()},
            'coordinate_ranges': {
                'longitude': [float(lons.min()), float(lons.max())],
                'latitude': [float(lats.min()), float(lats.max())],
                'depth': [float(levels.min()), float(levels.max())],
                'time': [time_label(times[0]), time_label(times[-1])],
            },
            'coordinate_resolution': {
                'longitude': sorted(set(float(value) for value in np.diff(lons))),
                'latitude': sorted(set(float(value) for value in np.diff(lats))),
            },
            'selected_depth_values': [float(value) for value in levels],
            'variables': sorted(str(name) for name in dataset.data_vars),
            'configured_input_variables': list(config['input_variables']),
            'configured_target_variables': list(config['target_variables']),
            'surface_ocean_fraction': float(reference_mask.mean()),
            'mask_stability': mask_stability,
            'window_grids': {
                'canonical_training_and_evaluation': canonical,
                'dense_overlap_tile_inference': dense,
            },
            'split_protocol': {
                'context_policy': config.get('split_context_policy'),
                'sequence_length': int(config['sequence_length']),
                'prediction_length': int(config['prediction_length']),
                'splits': split_summary(
                    len(times), config, canonical['valid_pure_ocean_window_count']
                ),
            },
            'protocol_assertions': {
                'canonical_split_strides_equal': bool(
                    config['train_stride_lon'] == config['val_stride_lon'] == config['test_stride_lon']
                    and config['train_stride_lat'] == config['val_stride_lat'] == config['test_stride_lat']
                ),
                'terminal_anchors_present': bool(
                    canonical['terminal_longitude_anchor_present']
                    and canonical['terminal_latitude_anchor_present']
                    and dense['terminal_longitude_anchor_present']
                    and dense['terminal_latitude_anchor_present']
                ),
                'all_configured_variables_available': all(
                    name in dataset.data_vars for name in variables_to_check
                ),
            },
        }

    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = (project_root / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + '.tmp')
        temporary.write_text(serialized + '\n', encoding='utf-8')
        os.replace(temporary, output_path)
        print(output_path)
    else:
        print(serialized)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
