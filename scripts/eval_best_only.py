#!/usr/bin/env python3
"""Evaluate an already-trained best checkpoint WITHOUT resuming training.

When a bug is fixed purely in the evaluation/export path (not in training
numerics), the trained weights stored in ``best_model.pth`` remain valid. This
script re-computes metrics on those stored weights using ``train.py``'s own
``OceanModelTrainer.evaluate``, deliberately bypassing the strict cross-version
resume guards in ``load_checkpoint`` — we are NOT resuming training across code
versions, only scoring frozen weights.

This is the intended escape hatch: re-training from scratch would reproduce the
same weights but waste hours of GPU time for an evaluation-only bug.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import OceanModelTrainer, load_config, merge_configs, DEFAULT_CONFIG


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, help='experiment config JSON')
    ap.add_argument('--result_dir', required=True,
                    help='campaign result dir holding best_model.pth')
    ap.add_argument('--overrides_json', default='{}',
                    help='same overrides used for the original training')
    ap.add_argument('--split', default='validation')
    args = ap.parse_args()

    config = DEFAULT_CONFIG.copy()
    config = merge_configs(load_config(args.config), config)
    config.update(json.loads(args.overrides_json))
    config['explicit_result_dir'] = args.result_dir
    config['post_training_evaluation'] = args.split
    # Avoid clobbering the training artifacts written by the real run.
    config['config_filename'] = 'eval_harness_config.json'
    config['scalers_filename'] = 'eval_harness_scalers.pkl'

    trainer = OceanModelTrainer(config)
    if not trainer._load_best_model_weights():
        print(f'ERROR: best model weights not found in {args.result_dir}',
              file=sys.stderr)
        sys.exit(2)

    results = trainer.evaluate(split=args.split)

    out_filename = (
        'evaluation_results.json' if args.split == 'test'
        else 'validation_results.json'
    )
    out_path = os.path.join(args.result_dir, out_filename)
    with open(out_path, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f'\nEvaluation written to {out_path}')
    print(f"RMSE={results.get('rmse')} MAE={results.get('mae')} "
          f"R2={results.get('r2')} units={results.get('metric_units')}")


if __name__ == '__main__':
    main()
