#!/usr/bin/env python3
"""Select coarse learning rates from validation-only calibration runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


NAME_PATTERN = re.compile(r'^lr_(?P<family>.+)_(?P<label>3e4|8e4|15e4|3e3)$')
EXPECTED_RATES = {3e-4, 8e-4, 1.5e-3, 3e-3}
COARSE_GRID_BOUNDARIES = {min(EXPECTED_RATES), max(EXPECTED_RATES)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True, help='campaign result root')
    parser.add_argument('--stage', default='global_lr_calibrate')
    parser.add_argument('--tail-epochs', type=int, default=3)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    stage_root = Path(args.results_root).resolve() / args.stage
    families = {}
    source_hashes = set()
    errors = []
    for run_dir in sorted(stage_root.glob('*/seed_*')):
        match = NAME_PATTERN.fullmatch(run_dir.parent.name)
        if not match:
            continue
        required = [run_dir / '_SUCCESS', run_dir / 'config.json', run_dir / 'run_summary.json']
        if not all(path.is_file() for path in required):
            errors.append(f'incomplete run: {run_dir}')
            continue
        config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
        summary = json.loads((run_dir / 'run_summary.json').read_text(encoding='utf-8'))
        if summary.get('evaluation_scope') != 'none':
            errors.append(f'calibration accessed an evaluation report: {run_dir}')
            continue
        losses = [float(value) for value in summary.get('validation_selection_losses', [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            errors.append(f'missing/nonfinite validation loss history: {run_dir}')
            continue
        tail = losses[-max(1, int(args.tail_epochs)):]
        row = {
            'experiment': run_dir.parent.name,
            'learning_rate': float(config['learning_rate']),
            'best_validation_loss': float(min(losses)),
            'tail_validation_loss_median': float(median(tail)),
            'best_epoch': summary.get('best_epoch'),
            'completed_epochs': summary.get('completed_epochs'),
            'run_dir': str(run_dir),
        }
        families.setdefault(match.group('family'), []).append(row)
        if summary.get('training_source_hash'):
            source_hashes.add(summary['training_source_hash'])

    selections = {}
    for family, rows in sorted(families.items()):
        rates = {row['learning_rate'] for row in rows}
        if rates != EXPECTED_RATES:
            errors.append(
                f'{family}: expected rates={sorted(EXPECTED_RATES)}, got={sorted(rates)}'
            )
            continue
        ranked = sorted(
            rows,
            key=lambda row: (
                row['tail_validation_loss_median'],
                row['best_validation_loss'],
                row['learning_rate'],
            ),
        )
        selected_rate = ranked[0]['learning_rate']
        requires_supplemental_calibration = selected_rate in COARSE_GRID_BOUNDARIES
        selections[family] = {
            'selected_learning_rate': selected_rate,
            'selection_rule': 'minimum median validation-selection loss over final tail epochs',
            'requires_supplemental_calibration': requires_supplemental_calibration,
            'ranking': ranked,
        }
        if requires_supplemental_calibration:
            errors.append(
                f'{family}: selected learning rate {selected_rate:g} is a coarse-grid '
                'boundary; run a supplemental interior grid before freezing it'
            )

    if len(source_hashes) > 1:
        errors.append(f'mixed training source hashes: {sorted(source_hashes)}')
    payload = {
        'status': 'completed' if not errors else 'incomplete',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'stage_root': str(stage_root),
        'tail_epochs': int(args.tail_epochs),
        'training_source_hashes': sorted(source_hashes),
        'selections': selections,
        'errors': errors,
        'note': (
            'Only selections without supplemental-calibration requirements may be '
            'applied to frozen screen configs. Then revalidate, resync, and start '
            'a new campaign.'
        ),
    }
    output_path = (
        Path(args.output).resolve()
        if args.output else Path(args.results_root).resolve() / 'selected_learning_rates.json'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, output_path)
    print(output_path)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
