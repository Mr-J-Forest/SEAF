# SEAF hidden-192 confirmation plan

The user selected hidden width 192 directly. This is the only capacity
candidate; widths 128, 144, and 160 are not screened.

## Frozen comparison

- Candidate: complete **SEAF** with LCFF and `seaf_hidden_dim=192`.
- Reference: the existing three-seed complete **SEAF** with LCFF and
  `seaf_hidden_dim=112`.
- Paper-facing name: **SEAF** for both; `seaf_h192` is an internal experiment
  identifier only.
- Seeds: 42, 123, 3407.
- Budget: exactly 30 epochs.
- Evaluation: validation only. Test access is forbidden.

All settings other than hidden width remain fixed: direct TEMP/SALT anomaly
targets, profile mixer, local/global spectral path, four deterministic anomaly
hypotheses, LCFF, optimizer, learning rate, batch size, scheduler, mixed
precision, gradient clipping, preprocessing, checkpoint selection, and
evaluator. AP and DAP remain external references and do not enter the forward
pass.

The hidden-192 model has 4,972,791 parameters, compared with 2,211,351 for the
hidden-112 reference. The experiment must report whether the extra capacity
improves TEMP/SALT RMSE and AP/DAP skill consistently enough to justify the
larger model.

## Execution matrix

| Internal experiment | Seed 42 | Seed 123 | Seed 3407 |
|---|---|---|---|
| `seaf_h192` | NEEDS RUN | NEEDS RUN | NEEDS RUN |

Use `configs/oras5_seaf_h192_confirmation.json`, deferred validation, and
`max-parallel=1` initially. The 192-channel model has not yet been memory-
calibrated under concurrent training, so concurrency must not be increased
without a measured cgroup/GPU-memory preflight.

After all runs finish, compare hidden 192 against the existing hidden-112 SEAF
using three-seed mean and sample standard deviation, per-seed deltas, and the
same paired forecast-origin moving-block bootstrap protocol. Do not access test
until the capacity decision is confirmed by the user. Keep the server on after
completion.
