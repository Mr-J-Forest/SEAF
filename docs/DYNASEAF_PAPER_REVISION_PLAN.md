# DynaSEAF paper revision plan

The paper is revised incrementally from the existing SEAF manuscript. The
three-seed DynaSEAF validation and held-out test artifacts are frozen and are
used for the mainline paper rows. The component screen and mechanism exports
are frozen on validation at seed 42 and are kept separate from the mainline
test claim.

## Keep unchanged

- direct anomaly formulation `A_t = Y_t - C_t`;
- training-period climatology reconstruction;
- full-field versus direct-anomaly target study;
- CLIM/PERS/AP/DAP references and persistence-aware evaluation;
- existing FourCastNet, ClimaX, and Swin comparisons;
- the SEAF-v1 paper-facing checkpoint and protocol.

## Add from the frozen mainline results

1. Title/abstract: introduce “DynaSEAF: Dynamics-Guided Transport for Ocean
   Anomaly Forecasting” as the second progression after
   full-field to direct anomaly forecasting.
2. Method: define future-dynamics auxiliary prediction, learned effective
   deformation, differentiable anomaly warp, direct/transport gate, and
   innovation residual using the equation in `DYNASEAF_DESIGN.md`.
3. Experiments: state the unchanged 12/5 ORAS5 protocol, A0–A4 ladder,
   component ablations, three seeds, and validation-only development policy.
4. Results: add the main comparison table with exactly SEAF-v1, DynaSEAF,
   FourCastNet, ClimaX, and Swin; report the A0--A4 and component-control
   screen as validation-only descriptive evidence.
5. Mechanism analysis: report future-dynamics magnitude, mean gate by lead,
   displacement magnitude, direct/transport/final branch errors, and the
   paired full-versus-no-innovation final-RMSE counterfactual. Do not evaluate
   the innovation residual as a standalone forecast. Add qualitative panels
   for the target, final forecast, transport, direct forecast, and gate.
6. Statistics: add forecast-origin paired moving-block bootstrap results with
   the frozen 10,000-replicate, block-5 protocol and BH-adjusted q-values.

## Writing constraints

Use “learned effective deformation” rather than “true trajectory” or “exact
Lagrangian trajectory”. Do not claim improvement, significance, or a parameter
count until it is read from a frozen result artifact. Keep AP/DAP external to
the forward path and do not use legacy test results for development decisions.

## Current edit status

`paper_final_draft.tex` now presents DynaSEAF as the mainline and reports the
frozen three-seed validation and held-out test aggregates. The architecture
figure has been updated, and the generated PDF has been compiled and visually
checked. A0--A4, `no_dynamics_aux`, `no_transport`, `no_innovation`,
`no_gate`, per-sample diagnostic exports, mechanism statistics, and
validation qualitative panels are now included from the frozen validation
screen. Paired-bootstrap results remain pending. No held-out test prediction
maps were available in the collected test JSONs, so the paper does not label
validation panels as test figures.
