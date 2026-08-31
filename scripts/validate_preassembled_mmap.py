#!/usr/bin/env python3
"""Verify tensor and reference-forecast parity for the preassembled mmap loader."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from data_loader import OceanDataset  # noqa: E402


DEFAULT_CONFIGS = (
    'configs/experiments/oras5_seaf.json',
    'configs/experiments/oras5_ablation_direct_full_field.json',
    'configs/experiments/oras5_ablation_no_tendency.json',
    'configs/experiments/oras5_ablation_no_external_dynamics.json',
)


def _capture(dataset: OceanDataset) -> dict:
    split_views = {
        'train': dataset,
        'validation': dataset.temporal_split_view('val'),
        'test': dataset.temporal_split_view('test'),
    }
    captured = {}
    for split, view in split_views.items():
        indices = [0, min(75, len(view) - 1), len(view) // 2, len(view) - 1]
        indices = sorted(set(indices))
        samples = []
        for index in indices:
            inputs, targets = view[index]
            samples.append({
                'index': index,
                'inputs': inputs.numpy().copy(),
                'targets': targets.numpy().copy(),
            })
        captured[split] = {
            'length': len(view),
            'sequences': list(view.sequences),
            'samples': samples,
        }
    references = dataset.build_reference_forecasts(
        sample_indices=[0], spaces=('physical', 'normalized')
    )
    captured['references'] = {
        space: {name: values.copy() for name, values in forecasts.items()}
        for space, forecasts in references.items()
    }
    captured['input_channel_slices'] = dict(dataset.input_channel_slices)
    captured['target_channel_slices'] = dict(dataset.target_channel_slices)
    return captured


def _compare(expected: dict, actual: dict, label: str) -> dict:
    result = {'config': label, 'max_abs_error': 0.0, 'arrays_checked': 0}
    if expected['input_channel_slices'] != actual['input_channel_slices']:
        raise AssertionError(f'{label}: input channel slices differ')
    if expected['target_channel_slices'] != actual['target_channel_slices']:
        raise AssertionError(f'{label}: target channel slices differ')
    for split in ('train', 'validation', 'test'):
        if expected[split]['length'] != actual[split]['length']:
            raise AssertionError(f'{label}/{split}: dataset lengths differ')
        if expected[split]['sequences'] != actual[split]['sequences']:
            raise AssertionError(f'{label}/{split}: sequence manifests differ')
        for old_sample, new_sample in zip(
            expected[split]['samples'], actual[split]['samples']
        ):
            if old_sample['index'] != new_sample['index']:
                raise AssertionError(f'{label}/{split}: sample indices differ')
            for key in ('inputs', 'targets'):
                old = old_sample[key]
                new = new_sample[key]
                error = float(np.max(np.abs(old - new)))
                result['max_abs_error'] = max(result['max_abs_error'], error)
                result['arrays_checked'] += 1
                if not np.array_equal(old, new):
                    raise AssertionError(
                        f'{label}/{split}/{old_sample["index"]}/{key}: max_abs={error}'
                    )
    for space in ('physical', 'normalized'):
        for name, old in expected['references'][space].items():
            new = actual['references'][space][name]
            error = float(np.max(np.abs(old - new)))
            result['max_abs_error'] = max(result['max_abs_error'], error)
            result['arrays_checked'] += 1
            if not np.allclose(old, new, rtol=0.0, atol=1e-5):
                raise AssertionError(
                    f'{label}/references/{space}/{name}: max_abs={error}'
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mmap-dir', type=Path, required=True)
    parser.add_argument(
        '--legacy-cache-dir',
        type=Path,
        default=Path('/tmp/seaf_cache/oras5_1979_2014'),
    )
    parser.add_argument(
        '--data-path',
        type=Path,
        default=PROJECT_ROOT / 'Data/oras5/ORAS5_197901_201412_1deg.nc',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'outputs/mmap_validation.json',
    )
    args = parser.parse_args()

    results = []
    for relative_config in DEFAULT_CONFIGS:
        config_path = PROJECT_ROOT / relative_config
        config = load_config(str(config_path))
        config['cache_preprocessed_dir'] = str(args.legacy_cache_dir)
        # Rebuild the legacy path in memory instead of relying on stale cache
        # keys from an earlier preprocessing schema. This keeps the comparison
        # tied to the same current raw-data protocol without writing another
        # multi-gigabyte compressed cache.
        config['cache_preprocessed'] = False
        config['preassembled_mmap_dir'] = None
        print(f'[OLD] {relative_config}', flush=True)
        legacy = OceanDataset(str(args.data_path), config, mode='train')
        expected = _capture(legacy)
        del legacy
        gc.collect()

        config['preassembled_mmap_dir'] = str(args.mmap_dir)
        print(f'[MMAP] {relative_config}', flush=True)
        mmap_dataset = OceanDataset(str(args.data_path), config, mode='train')
        actual = _capture(mmap_dataset)
        results.append(_compare(expected, actual, relative_config))
        del mmap_dataset, expected, actual
        gc.collect()

    payload = {'status': 'passed', 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
