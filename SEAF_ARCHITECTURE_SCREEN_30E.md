# SEAF 30-epoch architecture screen

## Scope and naming

The paper-facing model name is always **SEAF**. Internal experiment IDs describe
module additions only; iteration labels such as `v1` or `v2` must not appear in
paper tables, captions, checkpoints selected for the paper, or prose.

SEAF directly predicts five future TEMP/SALT anomalies. Future climatology is
added outside the network. AP and DAP are evaluation references only; this code
path contains no persistence skip.

## Implemented modules

1. `TemporalDepthMixer` applies grouped depth Conv1D and grouped temporal Conv1D
   only to the depth-resolved TEMP/SALT history, followed by a 1x1 projection.
2. The local/global blocks combine a two-layer 3x3 local residual update with the
   existing 8x8 low-mode spectral update. The learned spectral scale starts at
   0.1 and is logged after training.
3. The lead router uses zero-initialized logits with shape `[5, 4]` and softmax
   over the four members. It remains available as an ablation, but the completed
   screen showed a negative adjacent gain, so the default candidate keeps the
   spatial gate with `router_type=spatial`.
4. The forcing encoder receives UVEL, VVEL, SSHA, MLD, TAUX, TAUY, QNET, and
   WFLUX separately from TEMP/SALT. Its learned fusion scale starts at 0.1 and is
   logged after training. The default forcing-fusion candidate branches from
   `seaf_local_global` and therefore does not inherit the lead router.

All channel groups are resolved from `input_channel_slices`; no ORAS5 channel
offset is hard-coded in the model.

## Screening matrix

The matrix is `configs/oras5_seaf_architecture_screen.json`. It contains six
seed-42 validation runs, all fixed to exactly 30 epochs:

| Internal ID | Purpose | Parameters at the audited ORAS5 schema |
|---|---|---:|
| `seaf_reference` | archived hidden-64 SEAF reference | 1,122,884 |
| `seaf_capacity_control` | old architecture, hidden 112 | 2,182,172 |
| `seaf_profile_mixer` | capacity control + temporal/depth mixer | 1,752,636 |
| `seaf_local_global` | previous + local/global blocks | 2,205,118 |
| `seaf_lead_router` | local/global branch + lead-aware router (not retained) | 2,198,582 |
| `seaf_forcing_fusion` | local/global branch + separate forcing encoder; spatial gate retained | 2,217,887 |

The capacity control isolates width from module effects. Module decisions should
therefore use the consecutive comparisons beginning with
`seaf_capacity_control -> seaf_profile_mixer`, not the archived hidden-64 anchor.
The two final experiments are branches: both lead-router and forcing-fusion
effects are compared directly with `seaf_local_global` through the matrix's
`compare_to` field.

## Local verification

```bash
python scripts/validate_experiment_matrix.py \
  --matrix configs/oras5_seaf_architecture_screen.json

python -m pytest tests/test_pipeline_integrity.py \
  -k "modular_seaf or lead_router or legacy_seaf_checkpoint or full_modular_seaf or small_seaf_forward_and_backward or ablation_removes_disabled_parameters" \
  -q
```

The required checks are: legacy strict state-dict loading, finite forward output,
one backward step, output shape `[B,5,40,H,W]` under the formal schema, separated
profile/context/forcing indices, normalized lead weights, and a final parameter
count between 2M and 5M.

## Server execution

First sync the audited local source. From the local workspace:

```bash
TSC_SERVER=root@connect.westc.seetacloud.com \
TSC_SERVER_PORT=48312 \
./sync_to_server.sh
```

Then run on the server:

```bash
cd /root/TSC-Fusion

.venv/bin/python scripts/validate_experiment_matrix.py \
  --matrix configs/oras5_seaf_architecture_screen.json

SCREEN_SOURCE_HASH=$(.venv/bin/python -c \
  'import json; print(json.load(open("source_state.json"))["training_source_hash"])')
SCREEN_CAMPAIGN="${SCREEN_SOURCE_HASH}_seaf_arch_screen30"

nohup .venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_seaf_architecture_screen.json \
  --stage screen \
  --campaign "$SCREEN_CAMPAIGN" \
  --max-parallel 2 \
  --defer-post-training-evaluation \
  --continue-on-error \
  > "run_logs/${SCREEN_CAMPAIGN}.log" 2>&1 < /dev/null &
```

Do not raise `--max-parallel` above 2 without a new host-memory measurement.
Screening must not evaluate the test split.

Monitor with:

```bash
tail -f "run_logs/${SCREEN_CAMPAIGN}.log"
```

After all six jobs and deferred validation evaluations finish:

```bash
.venv/bin/python scripts/summarize_seaf_architecture_screen.py \
  --campaign "$SCREEN_CAMPAIGN"
```

This writes `architecture_screen_summary.{json,csv,md}` under the campaign root.
Each run also retains `validation_results.json`, `run_summary.json`, `config.json`,
the best checkpoint, training curves, and the complete log. The summaries include
TEMP/SALT RMSE, Macro SS_AP, Macro SS_DAP, parameter count and breakdown, learned
lead/member weights, spectral scales, and forcing scale.

## Decision rule

Use validation only. Retain a newly added module when its consecutive Macro SS_AP
gain is greater than 0.01, or when another metric declared before inspecting the
result shows a clear, scientifically relevant gain. Do not protect a module that
fails this screen. Do not inspect test performance while choosing the model.

Only after one configuration is frozen as clearly stronger should it proceed to
seeds 42, 123, and 3407 for validation confirmation. Test evaluation remains a
single final action after architecture and checkpoint-selection rules are frozen.

The completed seed-42 screen measured `seaf_lead_router` at a Macro SS_AP change
of -0.002673 relative to `seaf_local_global`; it is therefore excluded from the
default candidate. The archived `seaf_forcing_fusion` result was trained before
this decision and still included the lead router. It must not be reported as the
performance of the new lead-free forcing-fusion candidate; that configuration
requires a fresh validation run.

## Known risks

- A one-seed screen is a triage step, not paper evidence.
- The hidden-112 capacity control is required because the requested 2M--5M target
  cannot be met by the grouped modules at hidden 64 without adding unnecessary
  parameters.
- UVEL/VVEL are depth-resolved even though they belong to the forcing group; they
  are kept out of the TEMP/SALT depth mixer and encoded as forcing channels.
- The learned fusion scales can collapse toward zero or grow too large. Their
  final values are logged and must be inspected alongside validation skill.
- Existing evidence did not establish the spectral branch or spatial gate as
  statistically significant. The screen must be interpreted as new validation
  evidence, not as confirmation of old claims.
