# SEAF LCFF confirmation plan

This plan evaluates **Lead-Conditioned Forcing Fusion (LCFF)** as one atomic
module inside **SEAF**. The complete paper-facing model is always named
**SEAF**, while its formal ablation is **SEAF w/o LCFF**. Identifiers such as
`seaf_lcff` are experiment bookkeeping only and must not be used as model names
in the paper.

## Exact code boundary

Both variants use direct TEMP/SALT anomaly targets, hidden width 112, the
temporal/depth profile mixer, local/global spatial paths, two 8x8-mode spectral
blocks, four deterministic anomaly hypotheses, the same training loss and the
same original external-variable input contract. Neither variant contains an AP
skip. AP and DAP are external evaluation references only.

- **SEAF w/o LCFF** sets `use_forcing_encoder=false` and
  `router_type=spatial`. External variables are not removed: they remain in the
  shared context tensor and are encoded with tendency channels. The stable
  sample- and location-dependent spatial gate aggregates the four hypotheses,
  with one spatial mixture shared across forecast leads.
- **SEAF** sets `use_forcing_encoder=true` and `router_type=lead`; this is the
  complete model containing LCFF.
  External channels are separated from the shared context tensor, encoded by a
  dedicated forcing branch, and added to the shared features through a learned
  scalar initialized to 0.1. Four forcing-informed deterministic hypotheses are
  then aggregated by learned global weights of shape 5x4, one mixture per lead.

The lead router does not directly consume the forcing tensor. The precise
interpretation is that dedicated forcing features first enter the shared
anomaly representation, after which lead-specific weights aggregate the
resulting forcing-informed hypotheses.

## Compatibility audit and run matrix

Current code strictly loads both archived seed-42 checkpoints. Their effective
configurations differ only in `use_forcing_encoder` and `router_type`; hidden
width, target, data protocol, optimization, 30-epoch budget, preprocessing,
checkpoint selection, validation evaluator, and 44 forecast-origin IDs match.
The archived checkpoints reproduce the expected parameter counts and module
diagnostics, so seed 42 is reusable.

| Variant | Seed 42 | Seed 123 | Seed 3407 |
|---|---|---|---|
| SEAF w/o LCFF | EXISTING | NEEDS RUN | NEEDS RUN |
| SEAF | EXISTING | NEEDS RUN | NEEDS RUN |

Archived seed-42 sources:

- w/o LCFF: campaign `b717f069fbb1481e52cb9bd537f910810a4e9fd6_seaf_arch_screen30`, run `screen/seaf_local_global/seed_42`.
- +LCFF: the same campaign, run `screen/seaf_forcing_fusion/seed_42`.

The later lead-free forcing run remains an interaction diagnostic and is not
the formal LCFF ablation.

## Missing-run execution

Use `configs/oras5_seaf_lcff_confirmation.json`. It contains exactly four
missing runs: two variants x seeds 123 and 3407. Run all jobs for 30 epochs and
defer validation so the four frozen checkpoints are evaluated serially. Do not
access test and do not launch unrelated ablations.

```bash
cd /root/TSC-Fusion
nohup .venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_seaf_lcff_confirmation.json \
  --stage confirm_validation \
  --campaign <training_source_hash>_seaf_lcff_confirmation \
  --max-parallel 2 \
  --defer-post-training-evaluation \
  > run_logs/<training_source_hash>_seaf_lcff_confirmation.log 2>&1 < /dev/null &
```

The server must remain on after completion until the user explicitly changes
that instruction.

## Required final analysis

After all four missing runs finish, combine them with the two archived seed-42
runs and report:

1. Per-seed parameter count, best epoch, validation selection metric, TEMP/SALT
   RMSE, TEMP/SALT SS_AP, Macro SS_AP, and Macro SS_DAP.
2. Three-seed mean and sample standard deviation for headline metrics.
3. Per-seed LCFF deltas and the number of seeds with positive Macro SS_AP delta.
4. Paired forecast-origin moving-block bootstrap with block length 5, 10,000
   replicates, seed 20260826, and BH adjustment.
5. Three-seed lead-wise TEMP/SALT SS_AP deltas and LCFF router weights.
6. The existing seed-42 2x2 interaction diagnostic, clearly labeled single-seed.

Only after local artifact transfer and strict audit should the architecture be
classified as Promote, Exploratory, or Remove. No paper-writing task is sent
before training results are complete and the user has confirmed the result.
