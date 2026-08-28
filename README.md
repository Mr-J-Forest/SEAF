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

SEAF contains no anomaly-persistence skip, persistence projection, or learned persistence scale. Anomaly persistence remains an evaluation baseline, not a component of the formal model.

## Formal input contract

The frozen ORAS5 configuration is [`configs/experiments/oras5_seaf.json`](configs/experiments/oras5_seaf.json). It uses:

- prognostic state: `TEMP`, `SALT`;
- causal tendencies: one-step backward differences of `TEMP`, `SALT`;
- external dynamics: `UVEL`, `VVEL`, `SSHA`, `MLD`, `TAUX`, `TAUY`, `QNET`, `WFLUX`;
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

Run the SEAF/recent-baseline screen:

```bash
.venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_recent_baseline_matrix.json \
  --stage screen \
  --campaign <training_source_hash>_screen \
  --max-parallel 2
```

Run formal ablations:

```bash
.venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_ablation_matrix.json \
  --stage screen \
  --campaign <training_source_hash>_ablation \
  --max-parallel 2
```

The six predeclared variants isolate the anomaly target, causal tendencies, external dynamics, spectral encoder, and ensemble gate. The former `no_ap_residual` variant is now the full SEAF model and is no longer an ablation.

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
