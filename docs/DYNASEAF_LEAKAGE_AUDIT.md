# DynaSEAF leakage audit

This checklist records the implementation boundaries that must hold for every
training or evaluation run.

| Check | Evidence / boundary | Status |
|---|---|---|
| Future dynamics into `x` | `OceanDataset` keeps auxiliary fields after the first two batch fields; `actual_input_variables` is built only from configured history inputs | PASS (unit coverage) |
| Future ground truth into forward | `DynaSEAFNet.forward(x, return_diagnostics=False, valid_mask=None)` has no future-label argument and uses only model-predicted dynamics | PASS |
| Label shuffling | `test_dynaseaf_no_future_leakage.py` checks identical `model(x)` outputs and auxiliary-loss-only dependence | PASS |
| Auxiliary labels in validation/test | `temporal_split_view` and independent eval construction force `return_future_dynamics_targets=False` | PASS |
| Checkpoint selection | `validate_epoch` returns weighted TEMP/SALT validation MSE; it does not add `L_dyn` | PASS |
| AP/DAP | Reference forecasts are built in evaluation utilities and are not present in DynaSEAF modules or loss | PASS |
| Gate inputs | Gate conditions on `F`, model-generated `D_hat`, and lead embedding | PASS |
| NaN/land handling | Warp samples validity masks separately and renormalizes valid values | PASS (unit coverage) |
| Split/climatology/scaler | DynaSEAF inherits the existing dataset config and train-only preprocessing | PASS by configuration; server smoke pending |
| Test-set access | Matrices reserve test for a later frozen evaluation | PASS |

## Things not used as evidence

The following do not enter model input or checkpoint selection: AP, DAP, SS_AP,
future-dynamics validation accuracy, test RMSE, test SS_AP, or any manually
selected sample. The optional diagnostic file contains model-generated
components only and is not used to score or select a checkpoint.

## Remaining external verification

The local ORAS5 NetCDF is not present in this workspace, and the previously
configured AutoDL endpoint was unavailable at implementation time. Therefore
server data-pipeline smoke, one-seed screen, three-seed confirmation, and
legacy-test evaluation remain `TODO_FROM_FROZEN_RESULTS`.
