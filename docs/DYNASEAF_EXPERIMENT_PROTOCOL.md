# DynaSEAF experiment protocol

## Frozen comparison contract

All DynaSEAF entries inherit the paper-facing ORAS5 SEAF configuration:

- ORAS5 ICDC 1° data, 12-month history, 5-month joint TEMP/SALT forecast;
- chronological train/validation/test fractions 0.7777778/0.1111111/0.1111111
  with `carry_history`;
- training-period monthly climatology and scalers only;
- anomaly inputs and anomaly TEMP/SALT targets;
- 30 epochs for every formal calibration/screen/ablation/confirmation stage;
  smoke remains 2 epochs. The legacy 80-epoch budget is stale and forbidden;
- the existing spatial windows, sampler, masks, optimizer family, metrics, and
  checkpoint rule;
- validation-only selection during development; the legacy test split stays
  sealed until the method and settings are frozen.

AP and DAP remain external evaluation references. They are not inputs, model
branches, targets, losses, or checkpoint criteria.

## Incremental ladder

| Stage | Configuration | Change from previous row |
|---|---|---|
| A0 | `oras5_seaf_h192.json` | Frozen SEAF-v1 reference |
| A1 | `oras5_dynaseaf_dynamics_only.json` | Future dynamics head + auxiliary loss |
| A2 | `oras5_dynaseaf_transport.json` | Learned effective warp, fixed 0.5 mixture |
| A3 | `oras5_dynaseaf_transport_innovation.json` | Zero-initialized innovation head |
| A4 | `oras5_dynaseaf.json` | Adaptive target/depth-resolved gate |

Component controls are `no_dynamics_aux`, `no_transport`, `no_innovation`, and
`no_gate`. They are defined in the corresponding experiment JSON files and use
the same validation selection rule.

## Stages and seeds

1. Local unit tests and CPU/bfloat16 smoke only.
2. Stage B one-seed screen: seed 42, validation only; screen only
   `dynaseaf_lambda_dynamics`, `dynaseaf_max_deformation_cells`, and, if needed,
   the gate prior.
3. Three-seed confirmation: seeds 42, 123, and 3407, reporting mean and sample
   standard deviation.
4. One frozen legacy-test evaluation after all model and training settings are
   locked.

The active matrices are:

- `configs/oras5_dynaseaf_screen_matrix.json` (6 screen jobs);
- `configs/oras5_dynaseaf_ablation_matrix.json` (1 smoke, 9 screen, and 27
  confirmation jobs).

Run `python scripts/validate_experiment_matrix.py --matrix <matrix>` before
training. Both matrices currently resolve without validation errors.

## Server commands

After syncing the current local workspace and verifying the AutoDL SSH endpoint,
the exact screen command is:

```bash
cd /root/TSC-Fusion
nohup .venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_dynaseaf_ablation_matrix.json \
  --stage screen --campaign dynaseaf_screen_v1 \
  --max-parallel 2 \
  > run_logs/dynaseaf_screen.log 2>&1 < /dev/null &
```

Training is started only by an explicit queue command; importing the
implementation does not launch jobs. The existing mmap v1 cache has no
future-dynamics array; the loader detects that and safely falls back to regular
preprocessing for auxiliary-supervision runs.

## Statistical protocol

The planned primary statistic is the forecast-origin paired moving-block
bootstrap with 10,000 replicates, block length 5, and statistic
`log(MSE_reference/MSE_candidate)`. Report geometric MSE reduction,
95% confidence interval, p-value, and BH-adjusted q-value. Until frozen result
artifacts exist, every such value is `TODO_FROM_FROZEN_RESULTS`.
