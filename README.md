# TSC-Fusion

TSC-Fusion 是一个面向海洋温度与盐度月尺度多步预报的研究代码库。当前任务使用过去 12 个月的 TEMP、SALT、SSHA、UWND、VWND，预测未来 5 个月、0–1000 m 共 20 个深度层的 TEMP 与 SALT。

当前阶段以收集可复核证据为先。实验筛选、多随机种子确认和全域拼图尚未完成前，不应把任何旧表格、旧图片或单次运行写成论文结论。完整执行顺序见 [NEXT_STEPS_TSC_FUSION.md](NEXT_STEPS_TSC_FUSION.md)。

## 已冻结的数据协议

- 数据：`Data/FullData_preprocessed.nc`，121 个时步，实际时间标记为 `200901`–`201901`。
- 网格：360×180；选取 0–1000 m 的 20 个深度层。
- 窗口：经度跨度 32°、纬度跨度 21°，实际张量空间尺寸为 22×33。
- canonical 网格：经纬步长均为 8°，显式加入最东/最北终端锚点，共 151 个 100% 纯海洋窗口。
- 时间切分：目标时段严格按 60%/20%/20% 顺序切分；验证和测试允许使用分界前已经观测到的历史。
- 样本数：train 56×151=8456，validation 20×151=3020，test 21×151=3171。
- 密集拼图：4°×4° 步长，共 632 个 100% 纯海洋窗口。它覆盖全球网格的 35.73%、海洋格点的 78.72%，因此准确名称是“全球开阔海域拼图”，不是海岸到海岸的全海洋覆盖。

服务器审计文件会记录数据 SHA-256、窗口列表、split 边界、掩膜稳定性和源码指纹。可重跑：

```bash
python scripts/audit_dataset_protocol.py \
  --config configs/experiments/full.json \
  --hash-file \
  --output outputs/evidence/dataset_protocol.json
```

## 架构

数据管线先用训练时段计算逐窗口月气候态。TEMP/SALT 默认以气候态 anomaly 作为模型量，同时把月气候态背景作为输入特征。所有 scaler 仅由训练时段拟合。深度折叠进固定通道顺序，不使用在所有样本上恒定、没有新增信息的独立深度编码；经纬度采用接缝连续的周期 Fourier 特征。静态空间特征和纯时间特征以 singleton 轴存储，只在取样时广播。

TSC-Fusion 主干包含：

- Thermohaline Memory：从 TEMP/SALT 剖面通道学习软水团原型并写回网格特征；
- Local Branch：处理展平的时间×通道局地纹理；
- Low-mode Spectral Branch：在单窗口内进行低频二维 Fourier 滤波；
- Spatiotemporal Structure Branch：在时间、纬度、经度上做 3D 卷积，深度仍位于通道中；
- Fusion Transformer：细化窗口内空间 token；
- Global Token Bank：让同一起报月份的 151 个 canonical 窗口共享远地上下文；
- Gated Ensemble 与 Persistence Residual：生成 5 个 lead 的 TEMP/SALT anomaly，再恢复到物理量。

训练时，一个 time-group batch 必须包含同一起报月份的全部 151 个窗口。密集拼图推理采用严格两遍流程：先在 canonical 8° 网格建立全局 token bank，再用 4° 网格 micro-batch 渲染；改变渲染切批或裁剪范围不会改变 bank。

## 指标口径

- TEMP 与 SALT 的物理量 MSE/MAE/RMSE 分开报告，禁止把 °C 与 PSU 混成一个 overall 数值。
- 跨变量 overall 只在 normalized/anomaly 空间计算。
- 报告按变量、lead、深度、起报月份和气候期分层的指标。
- 同时报告 climatology、persistence、anomaly-persistence 基线及 MSE skill。
- `spatial_mean_removed`、`climatology_residual` 和 macro-field 指标用于检查模型是否只复原背景态。
- 消融比较按相同起报月份配对，并用长度 5 的 circular moving-block bootstrap；多重 contrast 使用 Benjamini–Hochberg 校正。

## 安装与检查

```bash
pip install -r requirements.txt
python -m compileall -q config.py data_loader.py metrics_utils.py \
  convlstm_model.py train.py predict.py predict_full_map.py scripts tests
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/validate_experiment_matrix.py
```

`validate_experiment_matrix.py` 会解析所有配置继承、检查测试集隔离、GTB 完整分组、任务 ID 唯一性，并核对预声明消融 contrast。

## 训练

单次调试：

```bash
python -u train.py \
  --config configs/experiments/postfix_smoke.json \
  --seed 42 \
  --note local_smoke
```

正式实验必须使用 campaign 隔离结果目录：

```bash
python -u scripts/run_experiment_queue.py \
  --stage screen \
  --campaign <training_source_hash>_screen
```

阶段约束：

- `global_lr_calibrate`：只保存训练/验证曲线，不生成验证或测试报告；
- `screen`：只生成 `validation_results.json`；
- `confirm_validation`：只对筛选留下的候选做 3-seed validation 确认；
- 当前没有可运行的 `final_test` 阶段；只有 validation 冻结架构后才显式创建并一次性运行；
- 默认手工运行只评估 validation，不会自动读取 test 指标。

每个成功目录包含 `_SUCCESS`、完整 `config.json`、`run_summary.json`、checkpoint、scaler 和评估文件。`run_summary.json` 保存训练源码哈希、配置指纹、实际数据几何、运行时确定性设置和硬件信息。

## 汇总与消融统计

```bash
python scripts/aggregate_results.py \
  --results outputs/results/campaigns/<campaign>/screen \
  --output outputs/aggregate/<campaign>_screen \
  --strict \
  --allow-protocol-difference input_variables \
  --allow-protocol-difference expected_canonical_windows_per_origin \
  --allow-protocol-difference data_protocol.train

python scripts/compare_ablation_contrasts.py \
  --results-root outputs/results/campaigns/<campaign> \
  --stage screen \
  --strict
```

预声明的 32 个 contrast 位于 `configs/ablation_contrasts.json`，覆盖主要模块、交互作用、联合温盐训练和输入变量价值。不能根据跑出的结果临时改比较方向或筛选阈值。

## 全球开阔海域拼图

```bash
python -u predict_full_map.py \
  --model_dir outputs/results/campaigns/<campaign>/confirm_validation/<model>/seed_42 \
  --base_time_index 95 \
  --steps 0 2 4
```

输出包括数值 `.npz`、逐变量物理指标、coverage/ocean mask、来源 JSON 和图。未覆盖格点为 NaN，不会用极小权重伪装成有效预测。经度 Fourier 输入在 0°/360° 接缝连续，但卷积不做周期 padding；海岸和高纬窄海域也未覆盖。这些限制必须出现在论文中。

## 服务器

```bash
./sync_to_server.sh
ssh -p 39323 root@connect.westd.seetacloud.com
```

远端 `/root/TSC-Fusion` 指向 `/root/autodl-tmp/TSC-Fusion`。当前实例为 1×RTX 5090 32GB、50GB 临时盘，宿主可见 208 逻辑 CPU、容器 CPU quota 25 核，容器内存上限 90GiB。后台任务应把整个进程从 SSH 会话分离：

```bash
cd /root/TSC-Fusion
nohup setsid .venv/bin/python -u scripts/run_experiment_queue.py \
  --stage postfix_smoke --campaign <hash>_postfix \
  > run_logs/postfix.log 2>&1 < /dev/null &
```

运行中的 campaign 期间不要再次同步代码，否则 `source_state.json` 会与正在运行的进程不一致。

## 证据边界

- `paper_reimplementation_models.py` 只是本仓库中的论文启发式复现/代理，不是相关工作的官方实现，默认不进入正式主表。
- `figures/gen_fig_visualization.py` 是硬编码旧结果 ID 的历史草稿，已加显式保护，不能用于当前论文证据。
- 论文权威源是工作区外的当前 `.tex`；不要读取或引用仓库中的 `Paper.md`/旧 Paper 目录。
