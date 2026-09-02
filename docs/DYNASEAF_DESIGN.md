# DynaSEAF design

## Scope

DynaSEAF is an incremental wrapper around the existing SEAF-v1 anomaly
forecaster. It reuses `SEAFNet.encode_features` and
`SEAFNet.forecast_from_features`; it does not copy or replace the SEAF
backbone. The default protocol is the same 12-month history, 5-month
TEMP/SALT anomaly forecast, training-only climatology, normalization, masks,
and split as the frozen ORAS5 experiment.

## Forward decomposition

For history `X` and forecast lead `h`, the model computes

```text
F = SEAFNet.encode_features(X)
A_dir(h) = SEAFNet.forecast_from_features(F)
D_hat(h) = FutureDynamicsHead(F, e_h)
Delta(h) = DeformationHead(F, D_hat(h), e_h)
A_transport(h) = W(A_t, Delta(h))
R(h) = InnovationHead(F, D_hat(h), e_h)
g(h) = sigmoid(TransportDirectGate(F, D_hat(h), e_h))

A_hat(h) = (1 - g(h)) A_dir(h) + g(h) A_transport(h) + R(h)
```

`e_h` is a small embedding-plus-MLP for leads 1..5. `D_hat` predicts the
configured future UVEL/VVEL/SSHA/MLD channels from history only. `Delta` is a
learned effective displacement in grid-cell units, bounded by
`dynaseaf_max_deformation_cells` through `tanh`; it is not described as a true
Lagrangian trajectory. TEMP and SALT may share the deformation field while
their anomaly channels remain separate.

`DifferentiableAnomalyWarp` uses `grid_sample` with explicit coordinate
normalization and samples the source at destination minus displacement, so
positive `dx` moves a structure toward increasing longitude-column index and
positive `dy` toward increasing latitude-row index. When a validity mask or
non-finite source is present, values and masks are sampled separately and
renormalized, preventing land/NaN values from diluting an ocean cell.

The innovation output convolution is zero-initialized. The gate output
convolution is also zero-initialized and its bias is set to
`dynaseaf_gate_initial_bias` (default sigmoid prior about 0.15), allowing the
network to start close to direct SEAF while learning a spatial,
lead-conditioned gate.

## Training contract

The main objective remains the existing variable-balanced forecasting loss:

```text
L_main = 0.5 MSE_TEMP + 0.5 MSE_SALT
L_dyn  = 1/4 (L_UVEL + L_VVEL + L_SSHA + L_MLD)
L      = L_main + dynaseaf_lambda_dynamics * L_dyn
```

The future dynamics target is appended by `OceanDataset` only for a DynaSEAF
training dataset when `return_future_dynamics_targets=true`. Validation, test,
prediction, checkpoint selection, and AP/DAP evaluation never consume that
target. `model(x)` returns only the final forecast tensor. The explicit
`model(x, return_diagnostics=True)` API returns the decomposition fields for
diagnostic export.

## Component map

| Component | Implementation |
|---|---|
| Lead embedding | `LeadEmbedding` |
| Future dynamics | `FutureDynamicsHead` |
| Shared conditioning | `DynamicsConditioner` |
| Effective displacement | `DeformationHead` |
| Mask-aware warp | `DifferentiableAnomalyWarp` |
| Non-transport residual | `InnovationHead` |
| Direct/transport gate | `TransportDirectGate` |
| Model wrapper | `DynaSEAFNet` |

Parameter counts are emitted by `DynaSEAFNet.parameter_breakdown()` and copied
to run summaries from the instantiated model. The ORAS5 full DynaSEAF count is
`TODO_FROM_FROZEN_RESULTS` until the server-side protocol run completes; no
count is guessed here.
