# SEAF

SEAF (**Spectral-Ensemble Anomaly Forecaster**) directly predicts future three-dimensional ocean temperature and salinity anomalies. Its formal architecture is intentionally compact:

1. a low-mode spatial spectral encoder;
2. multiple joint TEMP/SALT forecast heads;
3. a spatial ensemble gate;
4. causal TEMP/SALT tendency channels and external ocean-dynamics inputs.

Let the training-only climatology be $C_t$ and the anomaly be $A_t=Y_t-C_t$. SEAF learns

\[
\widehat A_{t+1:t+K}=f_\theta(X_{t-J+1:t}),
\qquad
\widehat Y_{t+h}=C_{t+h}+\widehat A_{t+h}.
\]

SEAF contains no anomaly-persistence skip, persistence projection, or learned persistence scale. Anomaly persistence remains an evaluation baseline, not a component of the formal model. A zero anomaly in physical anomaly space corresponds to climatology; zero in the network's standardized output space should not be interpreted directly as physical climatology.

## Formal input contract

The frozen ORAS5 configuration is [`configs/experiments/oras5_seaf.json`](configs/experiments/oras5_seaf.json). It uses:

- prognostic state anomalies: `TEMP`, `SALT`;
- causal tendencies: one-step backward differences of `TEMP`, `SALT`;
- anomalized external dynamics: `UVEL`, `VVEL`, `SSHA`, `MLD`, `TAUX`, `TAUY`, `QNET`, `WFLUX`;
- targets: joint future `TEMP` and `SALT` anomalies for all configured leads and depths.

Climatologies and scalers are fitted from training years only. The repository does not maintain separate temperature-only or salinity-only formal models.

## Implementation

- [`seaf_model.py`](seaf_model.py): spectral encoder, joint anomaly forecast members, and ensemble gate.
- [`model_factory.py`](model_factory.py): SEAF and comparison-model construction.
- [`data_loader.py`](data_loader.py): leakage-safe anomalies, tendencies, external fields, and reference forecasts.
- [`train.py`](train.py), [`predict.py`](predict.py): training and evaluation.

Removed experimental modules—AP skip, thermohaline memory, a separate 3-D branch, fusion transformer, global token bank, and local parallel branch—are not part of SEAF.

## Data preparation

```bash
.venv/bin/python scripts/prepare_oras5.py
```

The dataset configuration expects `Data/oras5/ORAS5_197901_201412_1deg.nc`.

## Validation and training

```bash
.venv/bin/python scripts/validate_experiment_matrix.py
.venv/bin/python main.py --config configs/experiments/oras5_smoke.json --mode train
```

Validate the unified formal matrix:

```bash
.venv/bin/python scripts/validate_experiment_matrix.py \
  --matrix configs/oras5_seaf_full_matrix.json \
  --contrasts configs/oras5_full_contrasts.json
```

The unified matrix contains LR calibration, smoke, and three-seed validation confirmation stages for SEAF, strict and validation-tuned full-field controls, architecture ablations, a local CNN control, and the three learned architecture adapters. Exact campaign commands and final-test freezing rules are in [`NEXT_STEPS_SEAF.md`](NEXT_STEPS_SEAF.md).

Run a selected stage:

```bash
.venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_seaf_full_matrix.json \
  --stage confirm_validation \
  --campaign <training_source_hash>_seaf_confirm_v1 \
  --max-parallel 2
```

The strict full-field control keeps the anomaly-centered inputs and changes only the target prediction space. A separate validation-tuned full-field comparator tests whether the target result survives architecture-specific LR selection. Uniform multi-head averaging and a single-head control separately test the spatial gate and the value of multiple forecast hypotheses.

## Baselines and metrics

Every evaluation includes climatology, persistence, anomaly persistence, and training-only damped anomaly persistence. FourCastNet/AFNO, ClimaX, and Swin use the same direct-anomaly target as SEAF; none receives an AP skip.

The primary reference score remains

\[
SS_{AP}=1-\frac{MSE_{model}}{MSE_{AP}}.
\]

Positive values indicate that a direct anomaly forecast improves on anomaly persistence. Reports also retain lead-, depth-, season-, variable-, and region-resolved results where available.

## Server synchronization

The historical remote directory remains `/root/TSC-Fusion` for deployment compatibility.

```bash
./sync_to_server.sh
```

`sync_to_server.sh` accepts `SEAF_SERVER`, `SEAF_SERVER_PORT`, `SEAF_REMOTE_DIR`, and `SEAF_SSH_BIN`; the older `TSC_*` variables remain fallback aliases for existing server scripts.
