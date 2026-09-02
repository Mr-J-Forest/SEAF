# CURRENT SEAF Architecture Map (frozen reference for DynaSEAF work)

> 冻结日期：2026-09-01（分支 `feature/dynaseaf-transport-innovation`，基线 commit `a22da9b`）。
> 本文档冻结 paper-facing SEAF-v1 的全部科学口径；DynaSEAF 的任何改动不得回写此基线。

## 1. 模型与代码入口

| 项目 | 值 |
|------|-----|
| 模型类 | `SEAFNet`（`seaf_model.py`） |
| 工厂 | `model_factory.create_ocean_model`，`model_type="seaf"` |
| 训练入口 | `train.py`（`OceanModelTrainer`）；`main.py` 为包装入口 |
| 预测入口 | `predict.py`（`SmartOceanPredictor`，overlap-tile + cosine 融合） |
| 队列入口 | `scripts/run_experiment_queue.py` + `scripts/validate_experiment_matrix.py` |

结构：低模态空间 Fourier 编码（`LowModeSpectralEncoder`）→ 4 个确定性 member heads → spatial/lead ensemble gate；可选模块沿配置链启用：temporal-depth mixer（`use_temporal_depth_mixer`）、local path、lead router（`router_type="lead"`）、forcing encoder（`use_forcing_encoder`）。

## 2. paper-facing 配置链

```
configs/experiments/oras5_seaf_h192.json        # SEAF-v1 正式入口（hidden 192）
└── oras5_seaf_lcff.json                        # router_type=lead, use_forcing_encoder=true
    └── oras5_seaf_local_global.json            # use_local_path=true
        └── oras5_seaf_profile_mixer.json       # use_temporal_depth_mixer=true
            └── oras5_seaf_capacity_control.json
                └── oras5_seaf.json             # model_type=seaf, lr=3e-4
                    └── ../datasets/oras5_icdc_1deg_1979_2014.json
                        └── ../experiments/formal_base.json
```

## 3. 数据协议（ORAS5，冻结）

| 项 | 值 |
|----|-----|
| 数据 | `Data/oras5/ORAS5_197901_201412_1deg.nc`（1979-01 至 2014-12） |
| 输入变量 | TEMP, SALT, UVEL, VVEL, SSHA, MLD, TAUX, TAUY, QNET, WFLUX |
| 目标变量 | TEMP, SALT（anomaly 空间，40 个深度通道） |
| 区域 | lon 130–162°E, lat 6.5–27.5°N, depth 0–1000 m |
| 序列 | 12 个月历史 → 5 个月联合 TEMP/SALT 预报 |
| 切分 | train 0.7778 / val 0.1111 / test 0.1111，时间连续切分，`carry_history` |
| 滑窗 | 训练/评估 stride 8；推理 overlap-tile stride 4 |
| 标准化 | 训练期 scaler；输入与目标均使用训练期月气候态 anomaly |
| 预处理缓存 | 服务器 `/tmp/seaf_cache/oras5_1979_2014`；mmap v1（不含未来动力学通道） |

## 4. 训练协议（冻结）

| 项 | 值 |
|----|-----|
| epochs | 所有正式阶段 30；smoke 2；旧 80 轮预算已废止且不得启动 |
| batch | 151（`group_batches_by_time`，每个起报时次的全部 canonical 窗） |
| optimizer | Adam，lr 3e-4（oras5_seaf.json 覆盖默认 1.57e-4），weight_decay 1e-4 |
| loss | 加权 MSE（TEMP 0.5 / SALT 0.5），无 gradient loss |
| AMP | bfloat16 autocast；梯度瞬时溢出跳步，连续 30 次判发散 |
| checkpoint | 仅按 TEMP/SALT validation objective 选择 best |
| 服务器 | RTX 5090 / 90 GiB cgroup；队列 `--max-parallel 2`，每作业 num_workers 2 |
| 评估 | `post_training_evaluation: validation`；test 在协议冻结前禁用 |

## 5. SEAF-v1 正式基线数字（不得改动）

来自 `outputs/seaf_h192_confirmation_remote/.../seed_42/run_summary.json`：

- 参数量：**4,972,791**
- seed 42，best epoch 27，best val loss **0.710064**
- RMSE_TEMP **0.5091**，RMSE_SALT **0.0766**
- 混合精度：bfloat16

## 6. AP / DAP 口径

AP（anomaly persistence）与 DAP（damped AP）基线由 `data_loader.py` 的 reference 路径生成
（`damped_persistence_coefficients` 在训练段按 pooled origin 回归估计、clip 到 [0,1]）。
它们只出现在评估报告中，**不进入模型 forward、输入、target 或 loss**。

## 7. 已知预存问题（与 DynaSEAF 无关）

`tests/test_apex_restore.py::test_factory_adds_apex_without_changing_seaf` 在基线 commit 上即失败：
`model_factory.py` 未注册 `apex_restored`。本分支不修复、不扩展该 baseline。
