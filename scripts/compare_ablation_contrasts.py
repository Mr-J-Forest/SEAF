#!/usr/bin/env python3
"""Paired moving-block bootstrap for predeclared ablation contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'expected JSON object: {path}')
    return payload


def evaluation_path(run_dir: Path) -> Path:
    for filename in ('validation_results.json', 'evaluation_results.json'):
        path = run_dir / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f'no evaluation report under {run_dir}')


def extract_origin_metrics(path: Path, variables: list[str], metric: str) -> dict[str, dict[str, float]]:
    payload = load_json(path)
    origin_report = payload.get('stratified_reports', {}).get('by_origin', {})
    groups = origin_report.get('groups', {})
    if not groups:
        raise ValueError(f'origin-stratified metrics missing: {path}')
    output = {}
    for origin, group in groups.items():
        by_variable = group.get('metrics', {}).get('by_variable', {})
        values = {}
        for variable in variables:
            value = by_variable.get(variable, {}).get(metric)
            if value is None or not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError(
                    f'invalid {metric} for origin={origin}, variable={variable}: {path}'
                )
            values[variable] = float(value)
        output[str(origin)] = values
    return output


def circular_block_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    block_length = max(1, min(int(block_length), length))
    blocks = int(math.ceil(length / block_length))
    starts = rng.integers(0, length, size=blocks)
    sampled = [
        (start + offset) % length
        for start in starts
        for offset in range(block_length)
    ]
    return np.asarray(sampled[:length], dtype=np.int64)


def paired_scores(candidate: dict, reference: dict, variables: list[str]) -> tuple[list[str], np.ndarray]:
    origins = sorted(set(candidate) & set(reference), key=lambda value: int(value))
    if set(candidate) != set(reference):
        raise ValueError(
            'paired comparison requires identical forecast origins; '
            f'candidate_only={sorted(set(candidate) - set(reference))}, '
            f'reference_only={sorted(set(reference) - set(candidate))}'
        )
    eps = np.finfo(np.float64).tiny
    scores = np.empty((len(origins), len(variables)), dtype=np.float64)
    for origin_idx, origin in enumerate(origins):
        for variable_idx, variable in enumerate(variables):
            candidate_value = max(candidate[origin][variable], eps)
            reference_value = max(reference[origin][variable], eps)
            scores[origin_idx, variable_idx] = math.log(reference_value / candidate_value)
    return origins, scores


def summarize_bootstrap(
    score_by_seed: list[np.ndarray],
    variables: list[str],
    replicates: int,
    block_length: int,
    seed: int,
    meaningful_reduction_fraction: float,
) -> dict:
    rng = np.random.default_rng(seed)
    observed_by_variable = np.mean(np.concatenate(score_by_seed, axis=0), axis=0)
    observed_macro = float(observed_by_variable.mean())
    draws = np.empty((replicates, len(variables)), dtype=np.float64)
    seed_count = len(score_by_seed)
    for replicate in range(replicates):
        sampled_seed_indices = rng.integers(0, seed_count, size=seed_count)
        seed_draws = []
        for seed_idx in sampled_seed_indices:
            values = score_by_seed[int(seed_idx)]
            indices = circular_block_indices(len(values), block_length, rng)
            seed_draws.append(values[indices].mean(axis=0))
        draws[replicate] = np.mean(seed_draws, axis=0)
    macro_draws = draws.mean(axis=1)
    meaningful_score = -math.log1p(-meaningful_reduction_fraction)

    def stats(observed: float, samples: np.ndarray) -> dict:
        low, high = np.quantile(samples, [0.025, 0.975])
        probability_better = float(np.mean(samples > 0.0))
        lower_tail = (float(np.count_nonzero(samples <= 0.0)) + 1.0) / (len(samples) + 1.0)
        upper_tail = (float(np.count_nonzero(samples >= 0.0)) + 1.0) / (len(samples) + 1.0)
        return {
            'mean_log_mse_ratio': float(observed),
            'geometric_mse_reduction_fraction': float(1.0 - math.exp(-observed)),
            'ci95_log_mse_ratio': [float(low), float(high)],
            'ci95_geometric_mse_reduction_fraction': [
                float(1.0 - math.exp(-low)),
                float(1.0 - math.exp(-high)),
            ],
            'probability_candidate_better': probability_better,
            'two_sided_bootstrap_p': float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
            'probability_reduction_at_least_threshold': float(
                np.mean(samples >= meaningful_score)
            ),
        }

    by_variable = {
        variable: stats(float(observed_by_variable[idx]), draws[:, idx])
        for idx, variable in enumerate(variables)
    }
    macro = stats(observed_macro, macro_draws)
    worst_variable_reduction = min(
        values['geometric_mse_reduction_fraction'] for values in by_variable.values()
    )
    if (
        macro['geometric_mse_reduction_fraction'] >= meaningful_reduction_fraction
        and macro['probability_candidate_better'] >= 0.8
        and worst_variable_reduction >= -meaningful_reduction_fraction
    ):
        screening_decision = 'advance'
    elif (
        macro['geometric_mse_reduction_fraction'] <= -meaningful_reduction_fraction
        and macro['probability_candidate_better'] <= 0.2
    ):
        screening_decision = 'do_not_advance'
    else:
        screening_decision = 'inconclusive'
    if macro['ci95_log_mse_ratio'][0] > 0 and worst_variable_reduction >= 0:
        confirmation_status = 'supported'
    elif macro['ci95_log_mse_ratio'][1] < 0:
        confirmation_status = 'contradicted'
    else:
        confirmation_status = 'uncertain'
    return {
        'macro_equal_variable_weight': macro,
        'by_variable': by_variable,
        'screening_decision': screening_decision,
        'confirmation_status': confirmation_status,
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted q-values."""
    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = count - reverse_rank + 1
        raw = float(p_values[int(index)]) * count / rank
        running = min(running, raw)
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def common_seed_dirs(stage_root: Path, candidate: str, reference: str) -> list[tuple[int, Path, Path]]:
    candidate_root = stage_root / candidate
    reference_root = stage_root / reference
    candidate_seeds = {
        int(path.name.removeprefix('seed_')): path
        for path in candidate_root.glob('seed_*') if path.is_dir()
    }
    reference_seeds = {
        int(path.name.removeprefix('seed_')): path
        for path in reference_root.glob('seed_*') if path.is_dir()
    }
    return [
        (seed, candidate_seeds[seed], reference_seeds[seed])
        for seed in sorted(set(candidate_seeds) & set(reference_seeds))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True, help='campaign result root')
    parser.add_argument('--stage', default='screen')
    parser.add_argument('--contrasts', default='configs/ablation_contrasts.json')
    parser.add_argument('--output', default=None)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_root = Path(args.results_root).resolve()
    stage_root = results_root / args.stage
    contrast_path = (project_root / args.contrasts).resolve()
    specification = load_json(contrast_path)
    protocol = specification['protocol']
    metric = str(protocol['primary_metric'])
    block_length = int(protocol['moving_block_length'])
    replicates = int(protocol['bootstrap_replicates'])
    bootstrap_seed = int(protocol['bootstrap_seed'])
    meaningful = float(protocol['meaningful_reduction_fraction'])
    screening_fdr = float(protocol.get('screening_fdr', 0.10))
    confirmation_fdr = float(protocol.get('confirmation_fdr', 0.05))
    reports = []
    missing = []

    for contrast_index, contrast in enumerate(specification['contrasts']):
        seed_dirs = common_seed_dirs(
            stage_root, contrast['candidate'], contrast['reference']
        )
        if not seed_dirs:
            missing.append({
                'contrast': contrast['name'],
                'candidate': contrast['candidate'],
                'reference': contrast['reference'],
            })
            continue
        variables = list(contrast['variables'])
        score_by_seed = []
        origin_counts = {}
        used_seeds = []
        for seed, candidate_dir, reference_dir in seed_dirs:
            candidate = extract_origin_metrics(
                evaluation_path(candidate_dir), variables, metric
            )
            reference = extract_origin_metrics(
                evaluation_path(reference_dir), variables, metric
            )
            origins, scores = paired_scores(candidate, reference, variables)
            score_by_seed.append(scores)
            origin_counts[str(seed)] = len(origins)
            used_seeds.append(seed)
        summary = summarize_bootstrap(
            score_by_seed,
            variables,
            replicates,
            block_length,
            bootstrap_seed + contrast_index,
            meaningful,
        )
        reports.append({
            **contrast,
            'metric': metric,
            'seeds': used_seeds,
            'origin_counts': origin_counts,
            **summary,
        })

    q_values = benjamini_hochberg([
        report['macro_equal_variable_weight']['two_sided_bootstrap_p']
        for report in reports
    ])
    for report, q_value in zip(reports, q_values):
        report['macro_equal_variable_weight']['benjamini_hochberg_q'] = q_value
        if report['screening_decision'] == 'advance' and q_value > screening_fdr:
            report['screening_decision'] = 'inconclusive'
        if report['confirmation_status'] == 'supported' and q_value > confirmation_fdr:
            report['confirmation_status'] = 'uncertain'

    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'results_root': str(results_root),
        'stage': args.stage,
        'contrast_specification': str(contrast_path),
        'contrast_specification_sha256': sha256(contrast_path),
        'analysis_script_sha256': sha256(Path(__file__).resolve()),
        'protocol': protocol,
        'completed_contrasts': reports,
        'missing_contrasts': missing,
    }
    if args.strict and missing:
        payload['status'] = 'incomplete'
    else:
        payload['status'] = 'completed'

    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    output_path = Path(args.output).resolve() if args.output else results_root / f'{args.stage}_ablation_contrasts.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary.write_text(serialized + '\n', encoding='utf-8')
    os.replace(temporary, output_path)
    print(output_path)
    print(
        f"completed={len(reports)}, missing={len(missing)}, "
        f"status={payload['status']}"
    )
    return 1 if args.strict and missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
