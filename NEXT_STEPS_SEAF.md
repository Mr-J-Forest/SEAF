# SEAF 完整实验交接与执行计划

本文档是 SEAF 正式实验的唯一执行清单。接手者应严格按顺序完成 P0–P6，不能复用任何包含 AP skip、persistence projection 或 learned persistence scale 的旧训练结果，也不能在查看 test 结果后重新调参。

## 0. 最终研究问题与冻结协议

SEAF 直接预测未来 TEMP/SALT anomaly：

\[
A_t=Y_t-C_t,\qquad
\widehat A_{t+1:t+5}=f_\theta(X_{t-11:t}),\qquad
\widehat Y_{t+h}=C_{t+h}+\widehat A_{t+h}.
\]

其中 climatology 和 scaler 只能由 training years 拟合。Anomaly persistence (AP) 只作为评价参考，不能进入任何正式模型的计算图。

“零预测对应 climatology”只适用于物理 anomaly 空间中的零异常；标准化网络输出的数值零不能直接解释为物理 climatology。正式输入是 state anomalies、causal TEMP/SALT tendencies 和 anomalized external dynamics/forcing；空间 Fourier mixer 不应被描述为 temporal spectral modeling。

冻结数据协议：

- dataset：`ORAS5_ICDC_r1x1_opa0_1979_2014`；
- history：12 months；forecast：5 months；
- targets：joint TEMP/SALT，所有配置深度；
- input state：TEMP、SALT；
- causal tendency：TEMP、SALT 的一步后向差分；
- external dynamics/forcing：UVEL、VVEL、SSHA、MLD、TAUX、TAUY、QNET、WFLUX；
- split：现有 chronological train/validation/test，`split_context_policy=carry_history`；
- formal seeds：42、123、3407；
- formal maximum epochs：30；scheduler patience：8；early-stopping patience：25；smoke 固定 2 epochs；
- ORAS5 SEAF batch：76，DataLoader workers：2；大模型基线可用其已校准的 batch 8；
- server concurrency：`--max-parallel 2`，不得在未重新测量 cgroup 峰值前提高；
- 远端正式 campaign 的预处理 cache 使用 `/tmp/seaf_cache/oras5_1979_2014`，训练结果建议放在 `/tmp/seaf_runs/` 对应的 campaign symlink 下，避免占满 50 GB 项目盘；`/tmp` 仅用于本次容器会话，完成后必须归档结果；
- validation 用于 LR 选择和 checkpoint 选择；test 在全部配置冻结后只打开一次。

## P0. 正式运行前必须完成的仓库工作

以下工作没有完成时，不得同步并启动正式 campaign。

### P0.1 补齐统一正式矩阵

使用 `configs/oras5_seaf_full_matrix.json`，包含三个训练阶段：

1. `global_lr_calibrate`：5 个模型族 × 3 个 LR，共 15 jobs；
2. `smoke`：12 个唯一配置 × seed 42，共 12 jobs；
3. `confirm_validation`：12 个唯一配置 × 3 seeds，共 36 jobs。

不得把 `final_test` 写成普通训练 stage。当前 queue 会为新 stage 创建新的 result directory，存在重新训练的风险；最终测试必须从 `confirm_validation` 的 `best_model.pth` 做 eval-only。

### P0.2 将 ensemble 消融拆开

当前 `ablation_disable_ensemble=true` 同时删除多成员和 gate，只能解释为 single-head control。必须补充：

- `uniform_ensemble`：保留 4 个 member heads，空间权重固定为 \(1/4\)；
- `single_head`：1 个 member，无 gate；可复用并重命名现有 no-ensemble 行；
- `local_cnn`：non-spectral control encoder + single head。

建议配置文件：

- `configs/experiments/oras5_ablation_uniform_ensemble.json`；
- `configs/experiments/oras5_ablation_single_head.json`；
- `configs/experiments/oras5_baseline_local_cnn_anomaly.json`。

实现必须是通用 config flag，不得按 seed、样本或结果值写特例。测试至少验证：uniform weights 和为 1、member 数不变、single-head 不构造 gate、local CNN 不包含 Fourier 参数。

### P0.3 增加 frozen-checkpoint 批量测试入口

增加一个 campaign-level eval-only driver，例如 `scripts/evaluate_campaign_checkpoints.py`：

- 输入 `confirm_validation` campaign root、matrix、split=`test`；
- 只加载每个 run 的 `best_model.pth`，绝不调用训练循环；
- 检查 12 experiments × 3 seeds 全部存在且 source hash 一致；
- test 输出写入独立目录或独立文件，不能覆盖 validation report；
- 记录 checkpoint SHA256、config fingerprint、training source hash、split identity；
- 支持断点续跑，但已完成结果必须通过 provenance 检查后才能跳过。

同时增加对应测试，证明 eval-only 不改变 checkpoint mtime/hash，也不创建新的训练 checkpoint。

### P0.4 更新 contrast specification

建立或更新 `configs/oras5_full_contrasts.json`，预声明下列比较：

| Contrast | Candidate | Reference | 解释 |
|---|---|---|---|
| strict anomaly target | full | direct_full_field_strict | 只改变 target prediction space |
| tuned full-field challenge | full | direct_full_field_tuned | 排除 full-field LR 未调优解释 |
| spectral branch | full | no_spectral | spectral vs matched local encoder |
| spatial gate | full | uniform_ensemble | learned spatial gate |
| multiple hypotheses | full | single_head | multi-member ensemble |
| tendency | full | no_tendency | causal change direction |
| external dynamics | full | no_external_dynamics | physical forcing information |
| compact learned control | full | local_cnn | full architecture vs local CNN |
| FourCastNet | full | ofb_fourcastnet_anomaly | learned baseline |
| ClimaX | full | ofb_climax_anomaly | learned baseline |
| Swin | full | ofb_swin_anomaly | learned baseline |

primary metric 使用 forecast-origin MSE；moving block length 5；10,000 bootstrap replicates；confirmation FDR \(q=0.05\)；positive score 为 `log(reference_mse/candidate_mse)`。

### P0.5 验证脚本必须覆盖的约束

扩展 `scripts/validate_experiment_matrix.py`，至少拒绝：

- 任何 AP skip/persistence projection/learned persistence scale；
- 任何非 test stage 的 test evaluation；
- 缺失的 12×3 confirmation jobs；
- ablation 自行改变 full SEAF 的 LR（direct-full-field tuned comparator 除外）；
- baseline target 不是 direct anomaly；
- 不一致的 dataset、split、history、lead、target variables；
- duplicate run IDs；
- baseline provenance 缺失；
- `final_test` 被配置成训练 stage。

完成后在本地和服务器都运行完整 tests 与 matrix validation。

## 1. 唯一正式训练配置清单

### 1.1 SEAF、目标对照与消融（9 个）

| Name | Config/改动 | 角色 |
|---|---|---|
| `full` | `oras5_seaf.json` | 完整 SEAF |
| `direct_full_field_strict` | 保持 anomaly inputs，只关闭 anomaly target，并继承 Full LR | strict target-only control |
| `direct_full_field_tuned` | 与 strict 相同，但使用独立 validation-calibrated LR | strong full-field challenge |
| `no_spectral` | Fourier encoder 换为 matched local Conv/GN/GELU encoder | spectral ablation |
| `uniform_ensemble` | 4 heads，固定均匀平均 | gate ablation |
| `single_head` | 1 head，无 gate | multi-hypothesis ablation |
| `no_tendency` | 移除 TEMP/SALT tendency | information ablation |
| `no_external_dynamics` | 移除 8 个外部动力/强迫变量 | information ablation |
| `local_cnn` | no spectral + single head | simple learned anomaly baseline |

### 1.2 学习型架构基线（3 个）

以下全部从头重跑，使用相同 direct-anomaly task，且不含 AP skip：

| Name | Existing config | Provenance label |
|---|---|---|
| `ofb_fourcastnet_anomaly` | `oras5_ofb_fourcastnet_anomaly.json` | OceanForecastBench architecture adapter，非官方复现 |
| `ofb_climax_anomaly` | `oras5_ofb_climax_anomaly.json` | OceanForecastBench architecture adapter，非官方复现 |
| `ofb_swin_anomaly` | `oras5_ofb_swin_anomaly.json` | OceanForecastBench architecture adapter，非官方复现 |

共 12 个唯一可训练配置。`local_cnn` 同时服务于 baseline 表和组合消融表，只训练一次。如果 tuned full-field 最终选择的 LR 与 Full 完全一致，可在冻结 manifest 中声明两者等价并复用 strict checkpoint，否则必须分别训练。

旧 campaign 中的 Joint AP SS `0.3400`、TEMP/SALT RMSE `0.5086/0.0777` 和 anomaly-target `41.76%` 改善只能作为历史初步证据。它们不能归属于当前紧凑 `SEAFNet`，正式论文数值必须来自本计划的新 campaign。

### 1.3 解析参考（不训练，但必须在所有 validation/test evaluation 中重算）

- climatology；
- full-field persistence；
- anomaly persistence (AP)；
- training-only damped anomaly persistence (DAP)。

### 1.4 不进入主表的模型

`TianHai`、`FuXi Ocean`、`FuXi ONS`、`AxiomOcean` 当前是 `official_code=false` 的 `local_paper_level_proxy`，且尚未完成同一 ORAS5 direct-anomaly protocol 审计。它们不能作为权威 SOTA baseline 混入主结果表。如后续确需运行，必须单独建立 supplementary proxy campaign，明确标注“非官方代理实现”，且不得用于主要 superiority claim。

## 2. LR 校准：15 个短训练任务

只校准五个模型族；所有纯消融继承 full SEAF 的冻结 LR。

| Family | LR grid |
|---|---|
| `seaf` | 3e-4, 8e-4, 1.5e-3 |
| `direct_full_field_tuned` | 3e-4, 8e-4, 1.5e-3 |
| `ofb_fourcastnet` | 3e-6, 1e-5, 3e-5 |
| `ofb_climax` | 1.5e-4, 5e-4, 1e-3 |
| `ofb_swin` | 1.5e-5, 5e-5, 1.5e-4 |

统一 seed 42、30 epochs、scheduler patience 6、early-stopping patience 30、`post_training_evaluation=none`。正式选择规则是最后 5 个 epoch validation-selection loss 的 median 最小，其次 best validation loss 最小；不再使用 8 epochs 的短校准预算。

若某个模型族满足以下任一条件，不得立即冻结 LR：

- 最优 LR 位于 coarse grid 边界；
- 前两名的 tail-median validation loss 相差不足 1%；
- 30 epoch 时最优候选仍在明显下降，尚未形成稳定排序。

此时可对该模型族补充 interior LR，但每个补充任务仍固定为 30 epochs；不得自行延长到 50 或 80 epochs。补充运行仍然只能访问 validation-selection loss，不能生成或查看 test report。

服务器运行：

```bash
cd /root/TSC-Fusion
mkdir -p run_logs
nohup .venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_seaf_full_matrix.json \
  --stage global_lr_calibrate \
  --campaign <H0>_seaf_lr_v1 \
  --max-parallel 2 \
  > run_logs/<H0>_seaf_lr_v1.queue.log 2>&1 < /dev/null &
```

选择 LR：

```bash
.venv/bin/python scripts/select_learning_rates.py \
  --results-root outputs/results/campaigns/<H0>_seaf_lr_v1 \
  --stage global_lr_calibrate \
  --matrix configs/oras5_seaf_full_matrix.json \
  --tail-epochs 5 \
  --output outputs/results/campaigns/<H0>_seaf_lr_v1/selected_learning_rates.json
```

若最佳 LR 位于 grid 边界，脚本会要求补充 interior grid；未完成补充校准前不得冻结。选定 LR 后更新正式配置和 matrix，重新运行 tests/validation，再次 `./sync_to_server.sh`。此时 training source hash 会从 H0 变为 H1；后续正式 campaign 必须全部使用 H1。

## 3. Smoke：12 个短任务

在 H1 上对 12 个唯一配置运行 seed 42、1–2 epochs、无正式 evaluation。Smoke 必须确认：

- 所有模型输入/输出 shape 正确；
- TEMP/SALT 均有有限 loss；
- 无 CUDA OOM、NaN/Inf、nonfinite-gradient skip；
- 两个并发作业的进程树内存仍在 90 GiB cgroup 内；
- evaluation lock 和 checkpoint 写入行为正确。

运行命令：

```bash
nohup .venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_seaf_full_matrix.json \
  --stage smoke \
  --campaign <H1>_seaf_smoke_v1 \
  --max-parallel 2 \
  > run_logs/<H1>_seaf_smoke_v1.queue.log 2>&1 < /dev/null &
```

验收：12/12 `_SUCCESS`，queue state 中无 failed/cancelled，所有 run 的 `training_source_hash=H1`。

## 4. 三种子正式 validation confirmation：36 个训练任务

只有 smoke 全部通过后才能启动：

```bash
nohup .venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_seaf_full_matrix.json \
  --stage confirm_validation \
  --campaign <H1>_seaf_confirm_v1 \
  --max-parallel 2 \
  > run_logs/<H1>_seaf_confirm_v1.queue.log 2>&1 < /dev/null &
```

任务数：

\[
12\ \text{configs}\times3\ \text{seeds}=36\ \text{training jobs}.
\]

并发 2 时共有 18 个调度波次。完整 post-training validation evaluation 继续由全局 lock 串行化。

每个 run 必须保存：

- `config.json`；
- `best_model.pth` 和 checkpoint provenance；
- `run_summary.json`；
- validation evaluation report；
- TEMP/SALT overall metrics；
- lead、depth、forecast-origin 分层指标；
- CLIM/PERS/AP/DAP comparisons；
- parameter count、wall time、epoch time、peak CUDA/process-tree memory；
- source hash、training source hash、config fingerprint。

正式确认验收：36/36 `_SUCCESS`；3 seeds 完整；同一 H1；无 test access；无缺失 origin/depth/provenance 数据。

## 5. Validation 统计、冻结与一次性 final test

### P5.1 Validation 聚合

```bash
.venv/bin/python scripts/aggregate_results.py \
  --results outputs/results/campaigns/<H1>_seaf_confirm_v1 \
  --output outputs/aggregate/<H1>_seaf_confirm_v1 \
  --strict \
  --allow-protocol-difference batch_size \
  --allow-protocol-difference learning_rate \
  --allow-protocol-difference weight_decay \
  --allow-protocol-difference group_batches_by_time
```

这些 difference 只允许来自预声明的模型优化设置；dataset/split/history/lead/targets 不允许不同。

### P5.2 Paired hierarchical bootstrap

```bash
.venv/bin/python scripts/compare_ablation_contrasts.py \
  --results-root outputs/results/campaigns/<H1>_seaf_confirm_v1 \
  --stage confirm_validation \
  --contrasts configs/oras5_full_contrasts.json \
  --output outputs/aggregate/<H1>_seaf_confirm_v1/confirm_contrasts.json \
  --strict
```

报告三种子的 mean±std、paired bootstrap 95% CI、p、BH-FDR q，以及至少 1% MSE reduction 的概率。对 tendency/external，如果均值为正但 q≥0.05，只能称为 physically motivated information design，不能声称显著提升。

`SS_AP>0` 的样本均值只能表述为“平均上优于 AP”。只有 paired 95% CI 下界大于 0 且对应比较满足 `q<0.05` 时，才能写“存在统计证据表明模型捕获了超越 AP 的额外可预测信息”。这不等价于识别了真实因果动力机制。

### P5.3 冻结清单

在打开 test 前生成不可变 manifest，至少包含：

- 36 个 `best_model.pth` 的路径与 SHA256；
- 12 个 resolved config fingerprints；
- H1 training source hash；
- matrix 和 contrast SHA256；
- validation aggregate/contrast SHA256；
- test split identity；
- 声明“此后不再调参”。

### P5.4 一次性 final test

使用 P0.3 的 eval-only driver，对冻结的 36 个 checkpoint 运行 test。严禁创建新训练 run，严禁根据 test 结果重训。测试后再次聚合并运行相同 contrasts，输出到独立目录：

- `outputs/final_test/<H1>_seaf_confirm_v1/`；
- `outputs/aggregate/<H1>_seaf_final_test/`。

每个 test evaluation 必须重新计算 CLIM、PERS、AP、DAP，确保所有方法使用完全相同 forecast origins 和 ocean mask。

## 6. 论文所需的全部数值与推理图数据

### P6.1 主结果表

主表包括：Full SEAF、Local CNN、FourCastNet、ClimaX、Swin、CLIM、PERS、AP、DAP。

必须导出：

- joint/equal-variable TEMP+SALT MSE；
- TEMP RMSE、SALT RMSE；
- `SS_AP = 1 - MSE_model/MSE_AP`；
- 3-seed mean±std 和 95% CI；
- parameter count、训练时间、推理时间、峰值显存。

### P6.2 消融表

包括 Full、Strict direct full-field、Tuned direct full-field、No spectral、Uniform ensemble、Single head、No tendency、No external、Local CNN。Strict 行用于 target-only claim，tuned 行用于鲁棒性挑战；报告 absolute metric、相对 Full 的变化、paired CI 和 FDR q。

### P6.3 Lead/depth/season/region 数据

至少导出：

- leads 1–5；
- 每个配置深度层，主文重点 0、50、200 m（若网格没有精确值，必须使用并记录最近可用层）；
- TEMP/SALT 分开；
- seasons DJF/MAM/JJA/SON；
- forecast-origin 逐样本 MSE；
- 若 region mask 已正式定义，再输出区域分解；不得看结果后临时挑区域。

### P6.4 Dense regional inference maps

在打开 test 前固定可视化选择规则：

- 主文 origin：test midpoint；lead 5；depth 50 m 或最近层；
- 补充 origins：test 时间轴 10%、50%、90% 位置；
- leads：1、3、5；depths：0、50、200 m 或最近层；
- variables：TEMP、SALT；
- models：AP、Full SEAF、validation 上最强 learned baseline；
- learned model map 使用 3-seed prediction mean，同时保留 seed spread。

主文每个变量至少包含 Truth、AP、SEAF、`|e_AP|-|e_SEAF|`。补充材料加入最强 learned baseline 及更多 origin/lead/depth。

当前 `predict_full_map.py` 若仍只保存 model/target，必须先扩展为同时保存：

- physical truth；
- anomaly truth；
- climatology；
- AP 和 DAP forecasts；
- 每个 seed 的 learned prediction；
- 3-seed mean/std；
- ocean mask、lat/lon/depth/time/lead metadata；
- absolute error 与 AP-relative improvement；
- checkpoint/config/source provenance。

建议输出 NetCDF/Zarr 作为数据源，PNG/PDF 只作为渲染产物。任何图都必须可从保存的数据重新生成。

## 7. 可选但预声明的敏感性实验

只有 36 个正式 confirmation 全部完成后再运行，使用 validation、seed 42，不打开 test：

- spectral modes：4×4、8×8、12×12，其中 8×8 复用 Full seed 42，只新增 2 jobs；
- ensemble members：2、4、8，其中 4 复用 Full seed 42，只新增 2 jobs。

共新增 4 个 validation-only jobs。它们只用于说明 8×8 modes 和 4 members 的选择，不进入主 superiority claim，也不得触发正式模型重调。

## 8. 总任务预算

| 阶段 | 任务数 | 是否正式训练 |
|---|---:|---|
| LR calibration | 15 + 必要时 refinement | 每个 coarse run 30 epochs |
| Smoke | 12 | 2 epochs |
| Three-seed confirmation | 36 | 每个 30 epochs |
| Frozen final-test evaluation | 36 | 仅评估，不训练 |
| Optional sensitivity | 4 | validation-only 短训练 |

无 refinement 时核心训练预算为 15 + 12 + 36 = 63 个训练任务，其中论文正式结果来自 36 个 confirmation runs。若 tuned full-field LR 与 Full LR 完全一致并经 manifest 验证，可复用 strict checkpoint，将 smoke/confirmation 各减少 1/3 个任务。若某些模型族触发 LR refinement，应把补充任务记入 calibration campaign manifest，不得为了保持固定任务数而跳过。无 refinement、无 checkpoint 复用且执行预声明敏感性分析时，总训练任务为 67 个。

## 9. 每阶段交接时必须报告

接手者每完成一个阶段，应在对话中给出：

1. campaign name 和 training source hash；
2. 完成/失败/跳过 job 数；
3. queue state、aggregate、contrast 或 freeze manifest 的绝对路径；
4. 任何 protocol deviation；
5. 下一阶段是否满足启动门槛。

不得只粘贴单个最佳数字，也不得从旧 campaign 手工复制结果。论文中使用的每个数字和图都必须追溯到同一冻结协议下的新 campaign、checkpoint 和机器可读输出。
