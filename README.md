# APEX

APEX (**Anomaly-Persistence EXtrapolator**) predicts joint three-dimensional ocean temperature and salinity evolution beyond anomaly persistence. The formal model is intentionally narrow:

1. a fixed anomaly-persistence (AP) skip connection;
2. a low-mode spectral encoder;
3. multiple residual forecast heads with a spatial ensemble gate;
4. causal TEMP/SALT tendencies and external ocean-dynamics inputs.

The network predicts only the correction to anomaly persistence:

\[
\hat Y_{t+h}=C_{t+h}+A_t+f_\theta(X_{t-k:t}),
\qquad A_t=Y_t-C_t.
\]

Residual heads are zero-initialized. Before learning, APEX therefore equals anomaly persistence exactly. Climatologies and all normalization statistics are fitted from the training period only.

## Formal input contract

The frozen ORAS5 configuration is [`configs/experiments/oras5_apex.json`](configs/experiments/oras5_apex.json). It uses:

- prognostic state: `TEMP`, `SALT`;
- causal tendencies: one-step backward differences of `TEMP`, `SALT`;
- external dynamics: `UVEL`, `VVEL`, `SSHA`, `MLD`, `TAUX`, `TAUY`, `QNET`, `WFLUX`;
- targets: joint `TEMP` and `SALT` forecasts for all configured leads and depths.

The repository does not maintain separate temperature-only or salinity-only formal models.

## Model implementation

- [`apex_model.py`](apex_model.py): APEX spectral encoder, residual members, ensemble gate, and fixed AP skip.
- [`model_factory.py`](model_factory.py): APEX and comparison-model construction.
- [`data_loader.py`](data_loader.py): leakage-safe anomalies, tendencies, external fields, and reference forecasts.
- [`train.py`](train.py), [`predict.py`](predict.py): training and evaluation.

Removed experimental modules—thermohaline memory, a separate 3-D branch, fusion transformer, global token bank, and local parallel branch—are not part of the formal implementation.

## Data preparation

Download and prepare ORAS5 on the training server:

```bash
.venv/bin/python scripts/prepare_oras5.py
```

The dataset configuration expects:

```text
Data/oras5/ORAS5_197901_201412_1deg.nc
```

## Validation and training

Validate the frozen ablation protocol:

```bash
.venv/bin/python scripts/validate_experiment_matrix.py
```

Run a smoke test:

```bash
.venv/bin/python main.py --config configs/experiments/oras5_smoke.json --mode train
```

Run the APEX/recent-baseline screen:

```bash
.venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_recent_baseline_matrix.json \
  --stage screen \
  --campaign <training_source_hash>_screen \
  --max-parallel 2
```

Run the formal ablations:

```bash
.venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_ablation_matrix.json \
  --stage screen \
  --campaign <training_source_hash>_ablation \
  --max-parallel 2
```

The seven predeclared variants isolate AP residual learning, anomaly targets, causal tendencies, external dynamics, the spectral branch, and the ensemble gate. Test remains sealed during selection; screening and ablations evaluate validation only.

## Baselines and metrics

Every evaluation includes climatology, persistence, anomaly persistence, and training-only damped anomaly persistence. Recent neural adapters are FourCastNet/AFNO, ClimaX, and Swin Transformer under the same data and AP-residual protocol.

The primary comparison against AP is:

\[
SS_{AP}=1-\frac{MSE_{model}}{MSE_{AP}}.
\]

Positive values indicate improvement beyond anomaly persistence. Reports also retain lead-, depth-, season-, variable-, and region-resolved results where available.

## Server synchronization

The historical remote directory remains `/root/TSC-Fusion` for deployment compatibility, although the model and repository identity are APEX.

```bash
./sync_to_server.sh
```

`sync_to_server.sh` accepts `APEX_SERVER`, `APEX_SERVER_PORT`, `APEX_REMOTE_DIR`, and `APEX_SSH_BIN`; legacy `TSC_*` environment variables remain fallback aliases for existing server scripts.
