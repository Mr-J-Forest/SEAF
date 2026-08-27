# TSC-Fusion

TSC-Fusion 是一个面向海洋温度与盐度月尺度多步预报的研究代码库。当前任务使用过去 12 个月的 TEMP、SALT、SSHA、UWND、VWND，预测未来 5 个月、0–1000 m 共 20 个深度层的 TEMP 与 SALT。

当前阶段以收集可复核证据为先。实验筛选、多随机种子确认和全域拼图尚未完成前，不应把任何旧表格、旧图片或单次运行写成论文结论。完整执行顺序见 [NEXT_STEPS_TSC_FUSION.md](NEXT_STEPS_TSC_FUSION.md)。

## ORAS5 公开数据轨道

项目提供 [ECMWF ORAS5](https://cds.climate.copernicus.eu/datasets/reanalysis-oras5?tab=overview) 的独立公开数据轨道。准备脚本读取 ICDC 发布的 1° 控制成员 `opa0`，并强制应用 ICDC 官方[修正版 land-sea mask](https://icdc.cen.uni-hamburg.de/thredds/fileServer/ftpthredds/EASYInit/oras5/r1x1/Correction_to_ORAS5_r1x1_files.pdf)。原始 r1x1 文件不能绕过这一步直接用于实验。

先查看下载与磁盘估算，不写入数据：

```bash
python scripts/prepare_oras5.py --dry-run
```

准备完整的 1979–2014 数据：

```bash
python -u scripts/prepare_oras5.py
```

ICDC 支持 HTTP Range 时，可以用可恢复的并发分段下载加速大归档；例如服务器上使用 16 条连接：

```bash
python -u scripts/prepare_oras5.py --download-workers 16
```

每个分段写入独立临时文件，全部完成后才按顺序合并并原子替换；中断后会继续已有的 `.part`/分段文件。若服务端不支持 Range 或没有 `Content-Length`，脚本会自动退回单连接模式。

### 快速区域轨道（OPeNDAP）

ICDC 同时公开逐月 NetCDF 的 OPeNDAP 服务。若只需要当前 ORAS5 实验区域，
`prepare_oras5_opendap_region.py` 会在服务端切片后并发读取，不下载全球年归档：

```bash
python scripts/prepare_oras5_opendap_region.py --dry-run
python -u scripts/prepare_oras5_opendap_region.py --workers 8
```

若服务器允许更多进程，可用 `--workers 16 --executor process`；这里必须使用独立进程隔离
netCDF4 原生 OPeNDAP 状态，不能把 16 路直接改成线程。`auto` 会在 workers 大于 8 时
自动选择进程池。

默认输出 `Data/oras5/ORAS5_197901_201412_1deg_region.nc`，覆盖
`130–162°E, 6.5–27.5°N`，仍包含 10 个物理变量、432 个月和 20 个深度层，
但经纬度维度只保留该区域。它对应显式的
`configs/experiments/oras5_tsc_ap_residual_region_opendap.json`，并将滑窗关闭为
单区域实验；不能把这个区域文件冒充完整全球 76 窗口协议。脚本按变量×月份保存状态，
中断后可直接重启继续。

脚本逐年流式处理归档，默认不保留 tar.gz；中断后可从已经完成的变量×年份继续。输出为 `Data/oras5/ORAS5_197901_201412_1deg.nc`，包含 432 个月、20 个 0–1000 m 深度，以及 TEMP、SALT、UVEL、VVEL、SSHA、MLD、TAUX、TAUY、QNET、WFLUX。以 1979 年归档大小外推，完整下载约 15 GiB；浮点数据未压缩上界约 9 GiB，实际 NetCDF 会压缩。

ORAS5 主实验使用 `configs/experiments/oras5_tsc_ap_residual.json`：1979–2006 train、2007–2010 validation、2011–2014 test；窗口在 1000 m 参考层要求 100% 有效海水，8° canonical 网格共有 76 个窗口。模型采用固定系数 1 的 anomaly-persistence skip，残差输出层零初始化，因此训练开始时严格等于 anomaly persistence，而不是可学习缩放的近似版本。输入额外包含 TEMP/SALT 物理量的因果一步后向差分；tendency 使用独立的训练期 scaler，第一个时间步固定为零，不读取未来值。

```bash
python scripts/audit_dataset_protocol.py \
  --config configs/experiments/oras5_tsc_ap_residual.json \
  --output outputs/evidence/oras5_dataset_protocol.json

python -u train.py \
  --config configs/experiments/oras5_smoke.json \
  --note oras5_pipeline_smoke
```

同数据协议的基础对照配置为 `oras5_convlstm_baseline.json` 和 `oras5_cnn_baseline.json`。它们是本仓库实现的 sanity baselines，不应标成外部论文的官方复现。

每次 ORAS5 评估都会同时构造 Climatology、Persistence、Anomaly Persistence 和 Damped Anomaly Persistence。Damped AP 的系数只用 1979–2006 训练段估计，按目标变量、lead 和深度分别做 pooled lag regression，并限制在 `[0, 1]`；验证/测试数据不参与系数估计。

### ORAS5 消融实验

冻结消融矩阵为 `configs/oras5_ablation_matrix.json`，覆盖 AP residual、anomaly 目标、显式 tendency、外部动力/表面输入、TSC Memory、Spectral、3D 时空分支、Ensemble gate，以及 TEMP/SALT 联合训练。所有结构消融共享完整模型的冻结超参数，不为每个消融单独挑学习率。

```bash
python scripts/validate_experiment_matrix.py \
  --matrix configs/oras5_ablation_matrix.json

python scripts/run_experiment_queue.py \
  --matrix configs/oras5_ablation_matrix.json \
  --stage screen \
  --dry-run

python scripts/compare_ablation_contrasts.py \
  --results-root outputs/results/campaigns/<training_source_hash>_oras5_ablation \
  --stage screen \
  --contrasts configs/oras5_ablation_contrasts.json \
  --strict
```

`screen` 为 11 个单 seed、30 epoch 的 validation-only 任务；`confirm_validation` 将相同 11 个配置扩展到 3 个 seed、80 epoch。消融矩阵没有 test 阶段。

### 近期公开架构基线

正式的 ORAS5 对比另提供三套来自 2025 年
[OceanForecastBench](https://github.com/Ocean-Intelligent-Forecasting/OceanForecastBench)
公开基准的架构适配器：

| 配置 | ORAS5 参数量 | 保留的核心机制 | ORAS5 任务适配 |
|---|---:|---|---|
| `oras5_ofb_fourcastnet_ap_residual.json` | 21.67M | FourCastNet 的 AFNO 频域块 | 12 个月多变量 patch stem + 5 lead residual head |
| `oras5_ofb_climax_ap_residual.json` | 65.61M | ClimaX 的变量 tokenization、cross-attention 聚合和 ViT | 每个 ORAS5 变量/特征组单独编码 |
| `oras5_ofb_swin_ap_residual.json` | 36.47M | Swin 的分层特征和 shifted-window attention | 两尺度 encoder-decoder + 5 lead residual head |

三者都在本地 ORAS5 train split 从头训练，共享相同的输入、目标、气候态、标准化器和
fixed anomaly-persistence skip；残差 head 均零初始化，所以初始预测逐元素严格等于 AP。
它们是有明确源码 commit 和许可记录的 **architecture adapters**，不是官方权重，也不能把
本项目得到的数值称为 OceanForecastBench 官方成绩。详细来源见各配置的
`baseline_provenance` 和 `THIRD_PARTY_NOTICES.md`。

验证并查看冻结的 4 模型队列：

```bash
python scripts/validate_experiment_matrix.py \
  --matrix configs/oras5_recent_baseline_matrix.json

python scripts/run_experiment_queue.py \
  --matrix configs/oras5_recent_baseline_matrix.json \
  --stage global_lr_calibrate \
  --dry-run
```

先在服务器运行各模型独立的 validation-only 学习率网格：

```bash
nohup setsid .venv/bin/python -u scripts/run_experiment_queue.py \
  --matrix configs/oras5_recent_baseline_matrix.json \
  --stage global_lr_calibrate \
  --campaign <training_source_hash>_oras5_recent_lr \
  --max-parallel 2 \
  > run_logs/oras5_recent_lr.log 2>&1 < /dev/null &

.venv/bin/python scripts/select_learning_rates.py \
  --results-root outputs/results/campaigns/<training_source_hash>_oras5_recent_lr \
  --stage global_lr_calibrate \
  --matrix configs/oras5_recent_baseline_matrix.json
```

只有非边界最优学习率可以写回对应配置；若最优值位于网格边界，先补内部网格。随后提交、
重新同步并用新 source hash 启动 `screen`，不能在看过 test 后回头调参。

矩阵的 `screen` 只读 validation；`confirm_validation` 使用 3 个 seed。测试集条目只保留为
`_final_test_template`，必须在模型与超参数冻结后显式转成 `final_test` 阶段。XiHe 官方发布物
是针对 GLORYS12 日尺度固定层位的 ONNX 推理模型，FuXi-Ocean 当前公开论文也没有可直接
重训的同协议代码，因此二者没有被伪装成 ORAS5 月尺度可训练基线。

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
  --campaign <training_source_hash>_screen \
  --max-parallel 2
```

`--max-parallel 2` 同时运行两个彼此隔离的训练作业。ORAS5 配置相应固定为每个作业
2 个 DataLoader worker；在 90 GiB cgroup 内存和 24 GiB GPU 上的双作业实测峰值约为
75 GiB RAM、13 GiB 显存。队列会用跨进程锁串行执行训练后的完整 validation/test
评估，避免两个大预测数组同时汇总。提高到 3 之前必须重新做内存峰值测试。

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
