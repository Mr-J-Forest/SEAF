#!/usr/bin/env python3
"""Resolve and validate every experiment before consuming server GPU time."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG, load_config, merge_configs, validate_config
from scripts.run_experiment_queue import expand_matrix


DIAGNOSTIC_STAGES = {
    'global_lr_calibrate',
    'paper_reimplementation_lr_calibrate',
    'calibrate_batch_diagnostic',
    'sensitivity',
}
VALIDATION_SCREEN_STAGES = {
    'screen',
    'paper_reimplementation_baselines',
}
VALIDATION_CONFIRMATION_STAGES = {
    'confirm_validation',
    'paper_reimplementation_confirm_validation',
}
PAPER_REIMPLEMENTATION_STAGES = {
    'paper_reimplementation_lr_calibrate',
    'paper_reimplementation_baselines',
    'paper_reimplementation_confirm_validation',
}
NO_TEST_STAGES = {
    'smoke',
    'postfix_smoke',
} | DIAGNOSTIC_STAGES | VALIDATION_SCREEN_STAGES | VALIDATION_CONFIRMATION_STAGES


def max_expected_windows(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return max(int(count) for count in value.values())
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', default='configs/experiment_matrix.json')
    parser.add_argument('--contrasts', default='configs/ablation_contrasts.json')
    args = parser.parse_args()

    matrix_path = (PROJECT_ROOT / args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text(encoding='utf-8'))
    stages = [name for name in matrix if not name.startswith('_')]
    errors: list[str] = []
    rows = []
    run_ids = []
    resolved_configs = {}

    for stage in stages:
        for job in expand_matrix(matrix, [stage], only=None):
            config_path = (PROJECT_ROOT / job['config']).resolve()
            if not config_path.is_file():
                errors.append(f"{stage}/{job['name']}: missing config {config_path}")
                continue
            config = merge_configs(load_config(config_path), DEFAULT_CONFIG.copy())
            config.update(job.get('overrides', {}))
            try:
                validate_config(config)
            except Exception as exc:
                errors.append(f"{stage}/{job['name']}: {exc}")
                continue
            resolved_configs[(stage, job['name'])] = config

            scope = config['post_training_evaluation']
            if stage in NO_TEST_STAGES and scope == 'test':
                errors.append(
                    f"{stage}/{job['name']}: test evaluation is forbidden before confirmation"
                )
            if stage in VALIDATION_SCREEN_STAGES and scope != 'validation':
                errors.append(f"{stage}/{job['name']}: screening must evaluate validation")
            if stage in VALIDATION_CONFIRMATION_STAGES and scope != 'validation':
                errors.append(f"{stage}/{job['name']}: confirmation must evaluate validation")
            if stage == 'final_test' and scope != 'test':
                errors.append(f"{stage}/{job['name']}: final test stage must evaluate test")
            if scope == 'test' and stage != 'final_test':
                errors.append(f"{stage}/{job['name']}: test is reserved for final_test")
            if stage in DIAGNOSTIC_STAGES:
                if scope != 'none':
                    errors.append(f"{stage}/{job['name']}: diagnostic stage must not run reports")

            if stage in PAPER_REIMPLEMENTATION_STAGES:
                provenance = config.get('baseline_provenance', {})
                if not isinstance(provenance, dict):
                    errors.append(
                        f"{stage}/{job['name']}: baseline_provenance must be an object"
                    )
                elif (
                    provenance.get('kind') != 'paper_reimplementation'
                    or provenance.get('official_code') is not False
                    or not provenance.get('method_name')
                ):
                    errors.append(
                        f"{stage}/{job['name']}: paper reimplementation must be "
                        "explicitly marked as non-official"
                    )

            if (
                config.get('enable_global_token_bank', False)
                and config.get('global_token_bank_scope') == 'time_group'
            ):
                if not config.get('group_batches_by_time', False):
                    errors.append(f"{stage}/{job['name']}: time-group GTB requires grouped batches")
                stride_pairs = {
                    (
                        float(config[f'{split}_stride_lon']),
                        float(config[f'{split}_stride_lat']),
                    )
                    for split in ('train', 'val', 'test')
                }
                if len(stride_pairs) != 1:
                    errors.append(f"{stage}/{job['name']}: time-group GTB requires one canonical grid")
                expected = max_expected_windows(
                    config.get('expected_canonical_windows_per_origin')
                )
                if expected is not None and int(config['batch_size']) < expected:
                    errors.append(
                        f"{stage}/{job['name']}: batch_size={config['batch_size']} "
                        f"cannot hold expected GTB group={expected}"
                    )

            run_id = f"{stage}__{job['name']}_seed{job['seed']}"
            run_ids.append(run_id)
            rows.append({
                'stage': stage,
                'name': job['name'],
                'seed': int(job['seed']),
                'model_type': config['model_type'],
                'epochs': int(config['epochs']),
                'learning_rate': float(config['learning_rate']),
                'batch_size': int(config['batch_size']),
                'evaluation_scope': scope,
            })

    duplicate_run_ids = sorted(name for name, count in Counter(run_ids).items() if count > 1)
    if duplicate_run_ids:
        errors.append(f'duplicate run ids: {duplicate_run_ids}')

    contrast_path = (PROJECT_ROOT / args.contrasts).resolve()
    contrasts = json.loads(contrast_path.read_text(encoding='utf-8'))
    contrast_names = [item['name'] for item in contrasts.get('contrasts', [])]
    duplicate_contrasts = sorted(
        name for name, count in Counter(contrast_names).items() if count > 1
    )
    if duplicate_contrasts:
        errors.append(f'duplicate contrast names: {duplicate_contrasts}')
    for contrast in contrasts.get('contrasts', []):
        for role in ('candidate', 'reference'):
            experiment = contrast[role]
            config = resolved_configs.get(('screen', experiment))
            if config is None:
                errors.append(
                    f"contrast {contrast['name']}: {role} {experiment!r} is not in screen"
                )
                continue
            missing_variables = sorted(
                set(contrast['variables']) - set(config['target_variables'])
            )
            if missing_variables:
                errors.append(
                    f"contrast {contrast['name']}: {role} lacks targets {missing_variables}"
                )
    protocol = contrasts.get('protocol', {})
    if int(protocol.get('moving_block_length', 0)) < int(DEFAULT_CONFIG['prediction_length']):
        errors.append('moving_block_length must be at least prediction_length')
    for key in ('screening_fdr', 'confirmation_fdr'):
        value = float(protocol.get(key, 0.0))
        if not 0.0 < value < 1.0:
            errors.append(f'{key} must be in (0, 1)')

    summary = {
        'matrix': str(matrix_path),
        'stages': {stage: sum(row['stage'] == stage for row in rows) for stage in stages},
        'resolved_job_count': len(rows),
        'contrast_count': len(contrast_names),
        'errors': errors,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
