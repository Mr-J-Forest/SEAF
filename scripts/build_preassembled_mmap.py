#!/usr/bin/env python3
"""Build the shared ORAS5 channel-preassembled mmap cache."""

import argparse
import gc
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from data_loader import OceanDataset  # noqa: E402


_ARRAY_BASENAMES = {
    'inputs_f32.npy',
    'fullfield_targets_f32.npy',
    'target_climatology_f32.npy',
}
_METADATA_BASENAMES = {
    'manifest.json',
    'state.pkl',
    '_SUCCESS',
}


def _open_partial_array(path: Path, shape: tuple[int, ...]) -> tuple[Path | None, np.memmap]:
    """Open an array for writing, resuming when the final file already exists.

    A finished array from a previous interrupted run is reused as-is
    (read-only); a stale ``*.partial`` from a crashed run is discarded.
    """
    partial = path.with_name(path.name + '.partial')
    if path.exists():
        array = np.lib.format.open_memmap(path, mode='r', dtype=np.float32)
        if tuple(array.shape) != tuple(shape):
            raise ValueError(
                f'已有缓存数组 shape 不匹配: {path} {array.shape} != {tuple(shape)}'
            )
        return None, array
    if partial.exists():
        partial.unlink()
    array = np.lib.format.open_memmap(
        partial, mode='w+', dtype=np.float32, shape=shape
    )
    return partial, array


def _finish_array(partial: Path | None, array: np.memmap, destination: Path) -> None:
    if partial is None:
        return
    array.flush()
    os.replace(partial, destination)


def _channel_first(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4:
        return np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        return np.asarray(array[:, np.newaxis, :, :], dtype=np.float32)
    raise ValueError(f'不支持的时间轴数组 shape: {array.shape}')


def _atomic_json(path: Path, payload: dict) -> None:
    partial = path.with_name(path.name + '.partial')
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    os.replace(partial, path)


def _atomic_pickle(path: Path, payload: dict) -> None:
    partial = path.with_name(path.name + '.partial')
    with partial.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _load_dataset(
    config_path: Path,
    data_path: Path,
    cache_preprocessed: bool,
) -> OceanDataset:
    config = load_config(str(config_path))
    config['preassembled_mmap_dir'] = None
    config['cache_preprocessed'] = bool(cache_preprocessed)
    return OceanDataset(str(data_path), config, mode='train')


def build_cache(
    anomaly_config_path: Path,
    fullfield_config_path: Path,
    data_path: Path,
    output_dir: Path,
    cache_preprocessed: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in output_dir.iterdir():
        name = entry.name
        is_stripped_partial = (
            name.endswith('.partial')
            and name[: -len('.partial')] in _ARRAY_BASENAMES
        )
        if name in _ARRAY_BASENAMES or name in _METADATA_BASENAMES or is_stripped_partial:
            continue
        raise FileExistsError(
            f'输出目录已包含无法续传的文件: {entry}（仅允许数组和已知元数据文件）'
        )

    # A rebuild may be repairing metadata after an interrupted run.  Remove
    # only the marker, so readers cannot accept a manifest while it is being
    # regenerated; arrays remain available for true checkpoint-style resume.
    success_marker = output_dir / '_SUCCESS'
    if success_marker.exists():
        success_marker.unlink()

    print('[1/4] 加载 direct-anomaly 预处理数据', flush=True)
    anomaly = _load_dataset(
        anomaly_config_path, data_path, cache_preprocessed
    )
    input_variables = list(anomaly.actual_input_variables)
    if any(name in input_variables for name in ('SPATIAL_ENCODING', 'TIME_ENCODING')):
        raise ValueError('正式 mmap builder 不支持当前未启用的位置/时间编码')

    region_count = len(anomaly.all_regions_data)
    month_count = len(anomaly.times)
    reference = anomaly.all_regions_data[0]['normalized_data'][anomaly.input_variables[0]]
    height, width = reference.shape[-2:]

    input_slices = {}
    input_channels = 0
    for variable in input_variables:
        array = _channel_first(anomaly.all_regions_data[0]['normalized_data'][variable])
        input_slices[variable] = [input_channels, input_channels + int(array.shape[1])]
        input_channels += int(array.shape[1])

    target_slices = {}
    target_channels = 0
    for variable in anomaly.target_variables:
        array = _channel_first(anomaly.all_regions_data[0]['normalized_data'][variable])
        target_slices[variable] = [target_channels, target_channels + int(array.shape[1])]
        target_channels += int(array.shape[1])

    input_path = output_dir / 'inputs_f32.npy'
    input_partial, inputs = _open_partial_array(
        input_path,
        (region_count, month_count, input_channels, height, width),
    )
    climatology_path = output_dir / 'target_climatology_f32.npy'
    climatology_partial, climatology = _open_partial_array(
        climatology_path,
        (region_count, anomaly.climatology_period, target_channels, height, width),
    )

    regions = []
    print('[2/4] 写入预拼接输入和目标气候态', flush=True)
    if input_partial is None and climatology_partial is None:
        print('  inputs/climatology 已存在，跳过重写（断点续传）', flush=True)
    for region_idx, region in enumerate(anomaly.all_regions_data):
        if input_partial is not None:
            for variable in input_variables:
                start, stop = input_slices[variable]
                values = _channel_first(region['normalized_data'][variable])
                expected = (month_count, stop - start, height, width)
                if values.shape != expected:
                    raise ValueError(
                        f'{variable} region={region_idx} shape={values.shape}, expected={expected}'
                    )
                inputs[region_idx, :, start:stop, :, :] = values
        if climatology_partial is not None:
            for variable in anomaly.target_variables:
                start, stop = target_slices[variable]
                values = _channel_first(region['climatology'][variable])
                climatology[region_idx, :, start:stop, :, :] = values
        coords = region['coords']
        regions.append({
            'lon_range': [float(value) for value in region['lon_range']],
            'lat_range': [float(value) for value in region['lat_range']],
            'region_type': region.get('region_type', 'sliding'),
            'lons': np.asarray(coords['lons'], dtype=np.float32).tolist(),
            'lats': np.asarray(coords['lats'], dtype=np.float32).tolist(),
        })
        if (region_idx + 1) % 10 == 0 or region_idx + 1 == region_count:
            print(f'  inputs: {region_idx + 1}/{region_count}', flush=True)
    _finish_array(input_partial, inputs, input_path)
    _finish_array(climatology_partial, climatology, climatology_path)
    del inputs, climatology

    anomaly_scalers = anomaly.scalers
    damped_coefficients = anomaly.damped_persistence_coefficients
    available_variables = anomaly.available_variables
    sequence_length = int(anomaly.sequence_length)
    prediction_length = int(anomaly.prediction_length)
    climatology_period = int(anomaly.climatology_period)
    lons = np.asarray(anomaly.lons).tolist()
    lats = np.asarray(anomaly.lats).tolist()
    levels = np.asarray(anomaly.levels).tolist()
    times = [str(value) for value in np.asarray(anomaly.times)]
    time_period_indices = np.asarray(
        anomaly.time_period_indices, dtype=np.int64
    ).tolist()
    del anomaly
    gc.collect()

    print('[3/4] 加载 direct-full-field 目标空间并写入目标 mmap', flush=True)
    fullfield = _load_dataset(
        fullfield_config_path, data_path, cache_preprocessed
    )
    if len(fullfield.all_regions_data) != region_count:
        raise ValueError('direct anomaly/full-field 的区域数量不一致')
    for variable, scaler in anomaly_scalers.items():
        other = fullfield.scalers.get(variable)
        if other is None:
            raise ValueError(f'direct-full-field 缺少输入 scaler: {variable}')
        if not (
            np.array_equal(np.asarray(scaler.mean_), np.asarray(other.mean_))
            and np.array_equal(np.asarray(scaler.scale_), np.asarray(other.scale_))
        ):
            raise ValueError(f'两种目标协议的输入 scaler 不一致: {variable}')

    fullfield_target_path = output_dir / 'fullfield_targets_f32.npy'
    target_partial, targets = _open_partial_array(
        fullfield_target_path,
        (region_count, month_count, target_channels, height, width),
    )
    fullfield_scaler_names = {}
    if target_partial is not None:
        for region_idx, region in enumerate(fullfield.all_regions_data):
            normalized_targets = region['normalized_target_data']
            for variable in fullfield.target_variables:
                start, stop = target_slices[variable]
                values = _channel_first(normalized_targets[variable])
                targets[region_idx, :, start:stop, :, :] = values
                fullfield_scaler_names[variable] = fullfield._target_scaler_name(variable)
            if (region_idx + 1) % 10 == 0 or region_idx + 1 == region_count:
                print(f'  targets: {region_idx + 1}/{region_count}', flush=True)
        _finish_array(target_partial, targets, fullfield_target_path)
    else:
        print('  fullfield targets 已存在，跳过重写（断点续传）', flush=True)
        fullfield_scaler_names = {
            variable: fullfield._target_scaler_name(variable)
            for variable in fullfield.target_variables
        }
    del targets

    state_scalers = dict(anomaly_scalers)
    state_scalers.update(fullfield.scalers)
    _atomic_pickle(output_dir / 'state.pkl', {
        'scalers': state_scalers,
        'damped_persistence_coefficients': damped_coefficients,
    })
    del fullfield
    gc.collect()

    data_stat = data_path.stat()
    manifest = {
        'format_version': OceanDataset._PREASSEMBLED_MMAP_FORMAT_VERSION,
        'data_identity': {
            'size': int(data_stat.st_size),
            'mtime_ns': int(data_stat.st_mtime_ns),
        },
        'sequence_length': sequence_length,
        'prediction_length': prediction_length,
        'available_variables': available_variables,
        'lons': lons,
        'lats': lats,
        'levels': levels,
        'times': times,
        'time_period_indices': time_period_indices,
        'regions': regions,
        'input_file': input_path.name,
        'input_shape': [region_count, month_count, input_channels, height, width],
        'input_channel_slices': input_slices,
        'fullfield_target_file': fullfield_target_path.name,
        'fullfield_target_shape': [
            region_count, month_count, target_channels, height, width
        ],
        'fullfield_target_scalers': fullfield_scaler_names,
        'target_climatology_file': climatology_path.name,
        'target_climatology_shape': [
            region_count,
            climatology_period,
            target_channels,
            height,
            width,
        ],
        'target_channel_slices': target_slices,
        'files': {
            path.name: int(path.stat().st_size)
            for path in (input_path, fullfield_target_path, climatology_path)
        },
    }
    _atomic_json(output_dir / 'manifest.json', manifest)
    (output_dir / '_SUCCESS').write_text('format_version=1\n', encoding='utf-8')
    print(
        '[4/4] 完成: '
        f"{sum(manifest['files'].values()) / 2**30:.3f} GiB array payload",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--anomaly-config',
        type=Path,
        default=PROJECT_ROOT / 'configs/experiments/oras5_seaf.json',
    )
    parser.add_argument(
        '--fullfield-config',
        type=Path,
        default=PROJECT_ROOT / 'configs/experiments/oras5_ablation_direct_full_field.json',
    )
    parser.add_argument(
        '--data-path',
        type=Path,
        default=PROJECT_ROOT / 'Data/oras5/ORAS5_197901_201412_1deg.nc',
    )
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--disable-preprocessed-cache',
        action='store_true',
        help=(
            '不读写 .cache/preprocessed 中间缓存（每次 ~9GB/协议）。'
            '构建产物本身不依赖这些缓存；磁盘紧张时必须禁用。'
        ),
    )
    args = parser.parse_args()
    build_cache(
        args.anomaly_config.resolve(),
        args.fullfield_config.resolve(),
        args.data_path.resolve(),
        args.output_dir.resolve(),
        cache_preprocessed=not args.disable_preprocessed_cache,
    )


if __name__ == '__main__':
    main()
