# DynaSEAF 设计与协议/泄漏审计（2026-09-01）

分支：`feature/dynaseaf-transport-innovation`（基线 commit `a22da9b`）。
本文档回答三件事：模型怎么分解、未来动力学如何防泄漏、实验协议怎么跑。

## 1. 模型设计（`dynaseaf_model.py`）

DynaSEAF = 冻结的 SEAF-v1 direct path + transport–innovation 分解外衣：

```
features = SEAFNet.encode_features(x)          # 复用冻结编码器（不复制 backbone）
direct    = SEAFNet.forecast_from_features(f)  # 冻结 direct 多头预测
dyn       = FutureDynamicsHead(f, lead)        # 预测未来 UVEL/VVEL/SSHA/MLD anomaly
deform    = DeformationHead(f, dyn, lead)      # 每通道 2D 形变场，|disp| ≤ max_deformation_cells
transport = grid_sample(latest_anomaly, deform)  # 对最近一个月异常场做可微平流
innovation= InnovationHead(f, dyn, lead)       # zero-init 残差修正
gate      = TransportDirectGate(f, dyn, lead)  # zero-init → sigmoid≈0.15 起步
forecast  = (1-gate)*direct + gate*transport + innovation
```

要点：

- `forward` **只接受历史输入**，签名上没有未来动力学参数；ground-truth 未来场永远进不了网络。
- 未来动力学仅两处用途：(a) 作为 heads 的 conditioning（用模型自预测的 dyn），(b) 训练期辅助损失
  `L = L_task + lambda_dynamics * L_dyn`（`dynaseaf_lambda_dynamics`，默认 0.10）。
- `return_diagnostics=True` 导出 direct/transport/innovation/gate/deformation/predicted_dynamics，
  供机理分析与论文图使用；`predict.py` 推理路径默认关闭该开关，保证与 SEAF-v1 输出契约一致。
- 参数分解由 `parameter_breakdown()` 报告；目标总量 <10M（direct 4.97M + 分解头）。

## 2. 数据与标签（`data_loader.py`）

- 仅当 `model_type='dynaseaf'` 且 `dynaseaf_use_future_dynamics_aux=true` 且 split=='train' 时，
  dataset 才返回 `future_dynamics` 及其 valid mask（变量 UVEL/VVEL/SSHA/MLD，取自输入 anomaly 通道的未来切片）。
- val/test/推理 dataset 一律 `return_future_dynamics_targets=False`；若 mmap v1 缓存缺未来通道则自动回退
  常规预处理路径（带 `/tmp/seaf_cache` 缓存），不会静默编造标签。
- batch 解包统一走 `train.py::_unpack_batch`：前两项恒为 (inputs, targets)，后续字段由消费方显式决定，避免辅助标签漏进 loss 或评估。

## 3. 泄漏审计清单（逐条核过）

| 风险 | 结论 |
|------|------|
| 未来真值进入 forward | 无：forward 无 dynamics 参数；测试 `test_dynaseaf_no_future_leakage.py` 验证改变未来标签不影响输出 |
| 辅助标签进入评估/预测 | 无：val/test dataset 不返回 future_dynamics；评估样本索引契约由 `return_sample_index` 全 split 保持 |
| 辅助标签进入主 loss | 无：aux loss 只在训练分支加权累加，权重来自 config，验证集指标不含 dyn 项 |
| gate 用未来信息 | 无：gate 只条件于 features + 模型自预测 dyn + lead embedding |
| warp 越界/NaN | 有限差分形变经 `max_deformation_cells` 裁剪；invalid 单元 mask 后零填充（`test_dynaseaf_mask.py`、`test_dynaseaf_warp.py`） |
| SEAF-v1 行为改变 | SEAF 路径零改动（encode_features/forecast_from_features 只是把原 forward 拆成两步）；config 新键对 `model_type='seaf'` 惰性；回归测试确认 |
| AMP bf16 | warp/grid_sample 在 autocast 下数值稳定，溢出按既有跳步协议处理（`test_dynaseaf_amp.py`） |

## 4. 实验协议与矩阵

消融阶梯（验证集，seed 42 筛选 → 42/123/3407 确认，test 冻结前禁用）：

| 名称 | 配置 | 定义 |
|------|------|------|
| A0 | `oras5_seaf_h192.json` | SEAF-v1 冻结基线 |
| A1 | `oras5_dynaseaf_dynamics_only.json` | A0 + 未来动力学辅助预测（transport/innovation/gate 关） |
| A2 | `oras5_dynaseaf_transport.json` | A1 + transport（固定 0.5 直连/平流混合） |
| A3 | `oras5_dynaseaf_transport_innovation.json` | A2 + innovation（仍固定混合） |
| A4 | `oras5_dynaseaf.json` | Full DynaSEAF（+ adaptive gate） |
| 对照 | `oras5_dynaseaf_no_dynamics_aux.json` | A4 去掉辅助损失（模型自预测 dyn 仍作 conditioning） |
| 对照 | `oras5_dynaseaf_no_transport.json` | A4 去掉平流路径 |
| 对照 | `oras5_dynaseaf_no_innovation.json` | A4 去掉 innovation |
| 对照 | `oras5_dynaseaf_no_gate.json` | A4 的 adaptive gate 换成固定 0.5 混合 |

矩阵（均通过 `scripts/validate_experiment_matrix.py`，0 errors）：

- `configs/oras5_dynaseaf_screen_matrix.json` — 超参筛选（λ_dyn 0.05/0.2、max_deformation 0.5/2.0），screen 6 jobs；
- `configs/oras5_dynaseaf_ablation_matrix.json` — smoke 1 + screen 9 + confirm_validation 27 = 37 jobs。

## 5. 运行手册（服务器）

```bash
# 0) 本地同步
TSC_SERVER=root@connect.westc.seetacloud.com TSC_SERVER_PORT=48312 ./sync_to_server.sh

# 1) smoke（小区域 2 epochs，验证管线）
cd /root/TSC-Fusion
nohup .venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_dynaseaf_ablation_matrix.json --stage smoke \
  --campaign dynaseaf_smoke_v1 --max-parallel 1 \
  > run_logs/dynaseaf_smoke.log 2>&1 < /dev/null &

# 2) screen（A0-A4 + 对照，30 epochs，seed 42）
nohup .venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_dynaseaf_ablation_matrix.json --stage screen \
  --campaign dynaseaf_screen_v1 --max-parallel 2 \
  > run_logs/dynaseaf_screen.log 2>&1 < /dev/null &

# 3) confirm_validation（30 epochs，3 seeds）在 screen 结果裁决后启动
```

正式预算统一为 30 epochs（包括 screen、ablation、confirm_validation 和 LR
calibration）；smoke 固定为 2 epochs。旧矩阵中的 80 epochs 是 stale 配置，
已被修正并由矩阵校验器硬性拒绝，禁止用 80 轮启动任何后续正式任务。

注意：DynaSEAF 开启辅助损失时 mmap v1 缓存不含未来通道，首跑会退回常规预处理（自动、安全）；
若要恢复 mmap 加速需构建含 future dynamics 的 v2 缓存（`scripts/build_preassembled_mmap.py` 扩展，暂缓）。

## 6. 测试覆盖（本地全绿）

- `tests/test_dynaseaf_shapes.py` / `_warp` / `_gate` / `_mask` / `_legacy` / `_amp` / `_no_future_leakage`
  + `tests/dynaseaf_test_utils.py`：形状/契约/数值稳定性/泄漏共 14+ 用例；
- 既有回归 `test_pipeline_integrity.py` + `test_amp_overflow.py` 68 passed；
- 已知预存失败（与本分支无关）：`test_apex_restore.py`（model_factory 未注册 apex_restored）。
