# DynaSEAF results ledger

This file is intentionally a results ledger, not a placeholder for invented
science numbers. The three-seed validation artifacts, the isolated three-seed
held-out test artifacts, and the validation-only component screen used by the
current paper revision are frozen below. Paired-bootstrap fields below are
populated only from their corresponding frozen artifacts; no training or
checkpoint rewriting is involved.

## Frozen SEAF-v1 reference

The existing paper-facing artifact records:

- parameter count: 4,972,791;
- seed 42 best epoch: 27;
- best validation loss: 0.710064;
- physical RMSE: TEMP 0.5091, SALT 0.0766.

These are reference values only; they must not be copied into a DynaSEAF row.

## Frozen DynaSEAF validation evidence

The validation evidence is stored in the remote campaign outputs:

- seed 42: `outputs/results/campaigns/5d106ad2d2cc1877abc23fb6b17fbfd441900509_screen/screen/dynaseaf_full/seed_42/validation_results.json`;
- seeds 123 and 3407: `outputs/results/campaigns/5d106ad2d2cc1877abc23fb6b17fbfd441900509_dynaseaf_full_remaining30_screen/screen/dynaseaf_full/seed_{123,3407}/validation_results.json`.

The three-seed mean ± sample standard deviation is:

- TEMP RMSE: `0.493996 ± 0.000544`;
- SALT RMSE: `0.075463 ± 0.000183`;
- Macro SS_AP: `0.363731 ± 0.001784`;
- Macro SS_DAP: `0.216963 ± 0.002239`;
- parameters: `5,089,657`.

## Frozen DynaSEAF test evidence

The held-out test checkpoints and `evaluation_results.json` files are stored
under `outputs/results/remote_collected/dynaseaf_test_eval_20260901_v1/` for seeds
42, 123, and 3407. The test manifest was evaluated after validation checkpoint
selection and was not used for training or checkpoint selection.

The three-seed mean ± sample standard deviation is:

- normalized RMSE: `0.808980 ± 0.003693`;
- physical TEMP RMSE: `0.475020 ± 0.002211`;
- physical SALT RMSE: `0.076159 ± 0.000382`;
- Macro SS_AP: `0.352181 ± 0.005197`;
- Macro SS_DAP: `0.203966 ± 0.006576`.

The fixed-checkpoint lead-wise test export is
`outputs/results/paper_ready/dynaseaf_paper_artifacts_20260902/data/test_leadwise_mean_sd.csv`.
It contains TEMP/SALT SS_AP, SS_DAP, and physical RMSE for held-out leads 1--5,
with the corresponding per-seed CSV and manifest in the same directory.

## Frozen DynaSEAF--SEAF-v1 paired validation statistics

The paired comparison reuses the existing origin-level validation artifacts at
`outputs/results/dynaseaf/dynaseaf_full_validation_audit/` and applies the
target-formulation moving-block bootstrap protocol: forecast-origin paired
units, block length 5, 10,000 replicates, statistic
`log(MSE_SEAF-v1 / MSE_DynaSEAF)`, and geometric reduction
`1 - exp(-log_ratio)`. The candidate/reference SHA256 hashes in
`paired_bootstrap.json` match the three frozen seed files (42, 123, and 3407).

The resulting reductions and two-sided bootstrap significance values are:

- TEMP: `4.54%`, 95% CI `[2.40%, 6.45%]`, `p=0.00020`, BH `q=0.00020`;
- SALT: `3.92%`, 95% CI `[2.74%, 4.97%]`, `p=0.00020`, BH `q=0.00020`;
- macro equal-variable average: `4.23%`, 95% CI `[3.11%, 5.31%]`,
  `p=0.00020`, BH `q=0.00020`.

These are frozen validation statistics against the inherited SEAF-v1
reference, not a test-set checkpoint-selection procedure.

## Frozen validation ablation and mechanism evidence

The component screen is stored under
`outputs/results/remote_collected/dynaseaf_validation_mechanism_screen30/`.
It contains seed-42 validation exports for all nine requested configurations:
the A0--A4 ladder plus `no_dynamics_aux`, `no_transport`,
`no_innovation`, and `no_gate`. Each configuration has 33,440 records (3,344
origin-window samples, five leads, and two variables), a completed diagnostics manifest,
and TEMP/SALT qualitative panels. The manifests record
`split=validation`, `test_iteration=false`, and `retraining=false`.

The paper-facing summaries generated from these frozen CSVs are:

- `aggregate_paper/ablation_validation_summary.csv`;
- `aggregate_paper/dynaseaf_mechanism_by_lead_validation.csv`;
- `aggregate_paper/innovation_counterfactual_by_lead_validation.csv`;
- `aggregate_paper/paper_figure_manifest.json`.

The corresponding figures are
`figures/generated/dynaseaf_ablation_validation.{png,pdf}` and
`figures/generated/dynaseaf_mechanism_validation.{png,pdf}`, with the two
qualitative panels in `figures/generated/dynaseaf_validation_*_panel.png`.
These are validation-only descriptive diagnostics and are not substituted for
the three-seed held-out test claim. The raw remote CSV exports retain an
`innovation_rmse_normalized` column from the original exporter, but it is not
used in the paper-facing summaries because the innovation residual is not a
standalone forecast. Its effect is represented by
`innovation_counterfactual_by_lead_validation.csv` instead.

## Required campaign files

For each campaign, the machine-readable output contract is:

```text
results/dynaseaf/<campaign>/summary.json
results/dynaseaf/<campaign>/per_seed.csv
results/dynaseaf/<campaign>/lead_skill.csv
results/dynaseaf/<campaign>/depth_skill.csv
results/dynaseaf/<campaign>/lead_depth_skill.csv
results/dynaseaf/<campaign>/dynamics_metrics.csv
results/dynaseaf/<campaign>/gate_statistics.csv
results/dynaseaf/<campaign>/deformation_statistics.csv
results/dynaseaf/<campaign>/paired_bootstrap.json
results/dynaseaf/<campaign>/parameter_count.json
```

`scripts/generate_dynaseaf_paper_tables.py` consumes completed run summaries
and evaluation reports to generate the paper-facing tables. It preserves
`TODO_FROM_FROZEN_RESULTS` when a field or campaign is absent, and always emits
the complete CSV/JSON contract listed above.

## Current status

| Item | Status |
|---|---|
| Local implementation and unit tests | complete |
| Matrix/config validation | complete |
| CPU/AMP model smoke | complete |
| ORAS5 server smoke | complete: `dynaseaf_smoke_v1`, 2 epochs, seed 42; development check only |
| Three-seed validation confirmation | complete; frozen validation artifacts above |
| Three-seed held-out test evaluation | complete; isolated frozen test artifacts above |
| DynaSEAF A0--A4 and component ablations | complete; seed-42 validation screen above |
| Per-sample mechanism diagnostics and qualitative panels | complete; validation-only exports above |
| Paired bootstrap / q-values | complete; frozen validation only |
| Full DynaSEAF parameter count from formal artifact | complete: 5,089,657 |
