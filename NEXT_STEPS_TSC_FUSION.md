# TSC-Fusion 下一步执行与证据收集计划

更新日期：2026-08-24

## 1. 总原则与当前结论

当前优先级不是写论文，而是把数据、代码、实验和统计证据全部冻结并收集齐。只有在消融实验确认哪些模块真正有用、最终架构重新训练并且测试集只使用一次之后，才修改论文正文。

论文唯一权威源为：

```text
C:\Users\ysx\Desktop\Project\论文\Paper\latex\official\paper.tex
SHA-256: f34ce2cd18c4aeb51f932873e6bf5f2a4a1644774f17f8f41f7f1c1f79559523
```

仓库中的旧 Markdown 和旧 Paper 目录均不作为事实来源，也不恢复、引用或据此修改结论。

截至本计划更新时：

- 本地与服务器 Python 编译检查通过，57 个回归测试通过；
- 实验矩阵解析为 152 个作业，当前可运行阶段无测试集评估，32 个预声明 contrast 均能解析；
- 服务器数据协议已完成独立审计；
- 早期两次 smoke 的 `returncode=-9` 来自旧的重复常量编码、三份预处理数组和 worker 预取造成的主存放大；它们没有完整结果，不能被解释为模型性能。共享只读预处理、紧凑广播编码、`num_workers=0` 和分块指标已在服务器复验；
- 中间 smoke 还暴露了队列 RSS 监控改造后的父进程收尾变量错误；已用可执行的队列成功路径回归测试修复，而不是针对该次 campaign 手工绕过；
- 最终训练源码哈希为 `39f083b72d013519ddefd0d1f748fec7451c2aed`，campaign `39f083b72d01_postfix_final` 的 TSC-Fusion、ConvLSTM、CNN 均 `returncode=0`、`_SUCCESS` 齐全且无 OOM；
- `39f083b72d01_lr_v1` 的 20 作业学习率校准已经在服务器串行启动；
- 学习率校准、30 组筛选、多随机种子确认和最终全图结果尚未完成，因此目前没有可写入论文的新性能数字。

最终 smoke 的机器可读资源结果如下。它只验证链路、数值有限性和运行预算；不同模型的 1-epoch loss 不用于性能排名。

| 模型 | 参数量 | 1-epoch best val loss | epoch 时间 | 峰值 CUDA allocated | 峰值进程 RSS | wall time | 训练后报告 |
|---|---:|---:|---:|---:|---:|---:|---|
| TSC-Fusion | 2,764,415 | 1.757312 | 52.2 s | 7.78 GiB | 24.64 GiB | 433.5 s | validation |
| ConvLSTM | 6,712,184 | 2.749334 | 77.7 s | 19.63 GiB | 12.07 GiB | 114.0 s | none |
| CNN | 1,197,480 | 2.447887 | 45.4 s | 2.10 GiB | 11.98 GiB | 81.8 s | none |

TSC-Fusion 的 1-epoch technical validation sanity 为 normalized RMSE `1.325636`、R² `0.471039`，TEMP RMSE `0.610899 °C`，SALT RMSE `0.098573 PSU`，macro-field R² 均值 `0.855086`。去气候态残差 R² 只有 TEMP `0.425041`、SALT `0.494734`，且相对 anomaly-persistence 的 macro MSE skill 为 `-0.138823`；这恰好说明不能把高物理量 R² 或单轮 smoke 写进论文，必须继续正式训练、基线比较和消融。

## 2. 已冻结的数据事实

服务器当前数据文件：

```text
Data/FullData_preprocessed.nc
SHA-256: ee0bd5e9d97a69a0cca8a189914aea274d4f99eef61b3feabacdf27c2cc61206
```

审计结果：

| 项目 | 冻结事实 |
|---|---|
| 时间 | 121 个时间步，`200901` 至 `201901` |
| 水平网格 | 180×360，1° 全球规则网格 |
| 深度 | 0–1000 m，共 20 层 |
| 输入历史 | 12 个月 |
| 预测长度 | 未来 5 个月 |
| 输入变量 | TEMP、SALT、SSHA、UWND、VWND |
| 目标变量 | TEMP、SALT |
| canonical 窗口 | 32°×21°，张量为 33×22 格点，8° 步长并加入终端锚点 |
| 窗口筛选 | 表层 TEMP 掩膜下 100% 纯海洋窗口，共 151 个 |
| train | 目标时间 12–71；56 个起报 ×151 窗口 = 8456 样本 |
| validation | 目标时间 72–95；20 个起报 ×151 窗口 = 3020 样本 |
| test | 目标时间 96–120；21 个起报 ×151 窗口 = 3171 样本 |
| dense 拼图 | 4° 步长，632 个纯海洋窗口 |
| dense 覆盖 | 全网格 35.73%，海洋格点 78.72% |

验证和测试的预测目标严格位于各自时间段，历史输入可以承接分界前已经观测到的数据。统计检验的独立配对单位是起报月份，不是高度重叠的 3020/3171 个空间窗口。

数据限制也必须冻结：SSHA 的全局有限比例约为 55.72%，UWND/VWND 约为 83.09%；风场在 5 个时间步全局缺失（索引 99、104–107）。逐时间片优先使用同片空间均值，全空片只允许使用训练段统计量，不能查看 validation/test 值。因此风场、SSHA 和全部表面强迫都有单独输入消融，不能预设其有效。

## 3. 新架构实际如何工作

### 3.1 数据空间

1. 对每个纯海洋窗口读取 12 个月历史。
2. 月气候态只由训练时间段计算。
3. TEMP/SALT 默认减去对应月份的训练期气候态，形成 anomaly；模型目标也是未来 TEMP/SALT anomaly。
4. 所有 scaler 只在训练时间段拟合，然后用于三个 split。
5. 输入还包含 TEMP/SALT 月气候态背景、周期经纬度 Fourier 编码和时间编码。当前 124 个输入通道由 43 个观测通道（TEMP 20、SALT 20、SSHA/UWND/VWND 各 1）、40 个 TEMP/SALT 气候态通道、32 个空间 Fourier 通道和 9 个时间通道组成。空间编码只存一帧，时间编码只存 `1×1` 空间轴，取 12 个月样本时再零拷贝广播。
6. 深度被折叠进固定 channel 顺序。网络实际张量是 `B,T,C,H,W`，不是带独立深度轴的 `B,T,C,D,H,W`。单独的“深度编码”因在所有样本、时间和格点上完全相同且不增加信息，已从架构中删除；SSHA/UWND/VWND 仍是各自一个表面通道，不复制成 20 层。
7. train/validation/test 在空间预处理相同时共享同一套只读大数组，只重新生成各自的时间索引和序列，避免三份多 GiB 数据常驻内存。

### 3.2 单窗口编码

设输入为 `B×12×C×22×33`：

- **Thermohaline Memory** 只截取 TEMP/SALT 剖面通道，将每个水柱软路由到 8 个可学习原型，用小型 Transformer 更新原型，再把水团上下文写回网格；
- **Local Branch** 把时间和通道展平，用二维卷积提取局地结构；
- **Low-mode Spectral Branch** 对单窗口的水平面做二维 FFT，只学习低频对角 Fourier 权重，逆变换后再用 `1×1` 卷积混合通道；它不是跨深度的谱卷积；
- **Spatiotemporal Structure Branch** 对时间、纬度、经度做 3D 卷积和时间注意力池化；深度仍在 channel 中；
- 三个分支拼接后经卷积残差融合，并可由窗口内 Fusion Transformer 细化空间 token。

### 3.3 跨窗口上下文与解码

训练和 canonical 验证时，一个 time-group batch 必须包含同一起报月份的全部 151 个窗口。Global Token Bank 从每个窗口的融合特征做空间池化，151 个 token 作为共享 key/value，每个格点特征作为 query，从而引入远地海域上下文。禁止把不同起报月份混入同一个 bank。

解码端由 4 个预测头和空间门控权重组成。模型同时预测未来 5 个 lead 的 TEMP/SALT anomaly，并加上最后一个历史时刻的可学习 persistence residual。最终先做 inverse scaling，再按目标月份加回训练期气候态，得到 °C 和 PSU 物理量。

### 3.4 全球开阔海域拼图

密集 4° 网格有 632 个窗口，不能一次放入 GPU。严格推理采用两遍流程：

1. 在冻结的 8° canonical 网格上编码全部 151 个窗口并建立一次全局 bank；
2. 在 4° dense 网格上按 micro-batch 编码/解码，但每批使用同一个 canonical bank；
3. 用余弦 taper 对重叠窗口加权拼接；未覆盖格点保持 NaN；
4. 保存预测、目标、blend weight、coverage mask、ocean mask、物理指标和来源哈希。

这只能称为“全球开阔海域拼图”。当前没有海岸到海岸覆盖。输入经度 Fourier 特征已在 0°/360° 接缝连续，但卷积本身没有周期 padding，不能声称完整全球海洋预测。

## 4. 已修复并需要继续守住的科学口径

- 物理量 TEMP 与 SALT 分开报告，禁止把 °C 和 PSU 混成 overall RMSE/MSE/MAE；跨变量 overall 只在 normalized/anomaly 空间给出。
- 指标按变量、lead、深度、起报月份和气候周期保存，并报告 correlation、R²、field RMSE 等可复算统计量。
- climatology、persistence、anomaly-persistence 三个基线使用同一训练期统计量。
- 目标通道映射必须显式、无重叠、完整覆盖；不再按变量数平均切通道。
- 物理量恢复、样本 provenance 或基线构造失败时直接终止，不能静默换指标空间。
- checkpoint 选择使用归一化目标 MSE，不包含可选 gradient regularizer，保证有/无梯度损失的消融可比。
- 缺失值兜底、气候态和 scaler 不得读取 validation/test 统计量。
- Global Token Bank 的训练 batch 必须完整覆盖同一起报的 151 个 canonical 窗口；密集推理使用外部两遍 bank，结果应对 micro-batch 切分不变。
- 默认训练后只评估 validation。当前矩阵没有可运行的 `final_test` 阶段。
- 所有正式运行保存完整配置、源码哈希、数据协议、随机种子、参数量、峰值显存、epoch 时间、checkpoint 和 `_SUCCESS`。

## 5. 消融实验设计：先找出真正有用的模块

### 5.1 预声明比较

`configs/experiment_matrix.json` 的 `screen` 包含 30 个模型，`configs/ablation_contrasts.json` 预声明 32 个有方向的比较。论文启发式模型不混入这 30 个模块筛选作业，而是走独立的 `paper_reimplementation_*` 阶段。主要组别为：

| 组别 | 必做比较 |
|---|---|
| 主干模块 | TSC、Spectral、3D、Fusion Transformer、GTB |
| 解码模块 | Gated Ensemble、Persistence Residual |
| 表征 | anomaly target、climatology feature、位置编码、时间编码 |
| 损失 | 无梯度损失、向量梯度、梯度幅值 |
| 输入 | 无风场、无 SSHA、仅 TEMP/SALT |
| 多任务 | 联合 TEMP/SALT 对各单任务的影响 |
| TSC 条件效应 | 联合任务与 TEMP/SALT 单任务下分别测试 |
| 交互作用 | GTB×Spectral、TSC×3D、Ensemble×Persistence、位置×时间、anomaly×气候态特征 |
| 覆盖范围 | 完整架构 vs local-only；全球窗口训练 vs 单区域训练（两者均无 GTB） |
| 通用基线 | 参数和训练协议可追溯的 ConvLSTM、CNN |

不能只做“full 减一个模块”。成对 factorial 对比用于识别模块是否只在另一个模块存在时才有效。

### 5.2 基线分层与官方实现引入计划

“有论文”“有作者代码”和“能用作者权重直接推理”是三种不同证据等级，不能统一写成 official baseline。基线固定分为以下五层：

| 分组 | 模型 | 当前状态 | 论文中允许的表述与用途 |
|---|---|---|---|
| `classical_baselines` | persistence、climatology、anomaly-persistence | 已由同一训练期统计量构造 | 无训练下界和 skill 参照 |
| `in_repo_baselines` | CNN、ConvLSTM | 已进入主矩阵 | 与 TSC-Fusion 使用同一数据协议重训的通用基线 |
| `benchmark_architecture_adapters` | OceanForecastBench-FourCastNet、ClimaX、SwinTransformer | 已进入独立 ORAS5 可运行矩阵 | 基于公开核心架构、在 ORAS5 AP-residual 协议从头重训；必须标成非官方 architecture adapter |
| `paper_reimplementation_baselines` | TianHai、FuXi-Ocean、FuXi-ONS、AxiomOcean | 已配置独立 LR、单 seed 和多 seed 阶段 | 只能称 paper-level reimplementation / paper-inspired proxy，不能称作者官方模型 |
| `official_native_protocol_references` | ORCA-DL、NeuralOM、WenHai、XiHe | 有作者代码或权重，但协议不兼容 | 单独外部参考表或定性案例；不得与本项目主表数值直接排名 |

#### 主表优先接入的官方代码

按与当前月尺度 12→5 联合温盐任务的适配难度、架构互补性和单卡 RTX 5090 可执行性，冻结以下优先级。该优先级必须在查看最终 test 结果前确定，不能根据谁的测试成绩更好再选择基线。

1. **OceanForecastBench-FourCastNet**：AFNO 频域路线，正式 ORAS5 配置为 20.87M 参数。来源：[论文](https://arxiv.org/abs/2511.18732)、[公开实现](https://github.com/Ocean-Intelligent-Forecasting/OceanForecastBench)。
2. **OceanForecastBench-ClimaX**：变量 tokenization、cross-attention 聚合和 ViT 路线，正式 ORAS5 配置为 63.64M 参数。来源：[ClimaX 官方实现](https://github.com/microsoft/ClimaX)、[OceanForecastBench 适配](https://github.com/Ocean-Intelligent-Forecasting/OceanForecastBench)。
3. **OceanForecastBench-SwinTransformer**：层级 shifted-window attention 路线，正式 ORAS5 配置为 36.10M 参数。来源：[Swin 官方实现](https://github.com/microsoft/Swin-Transformer)、[OceanForecastBench 适配](https://github.com/Ocean-Intelligent-Forecasting/OceanForecastBench)。
4. **PredFormer / SimVPv2（可选）**：只有前三项完成 screen 后仍缺少纯时空轻量对照时再接入；不能为扩大表格临时增加模型。

三项已冻结在 `configs/oras5_recent_baseline_matrix.json`。输入/输出 stem 按 12→5 ORAS5 任务适配，核心 AFNO、变量聚合和 shifted-window 机制保留；所有配置均记录源码 commit、许可和必要修改，并明确 `official_code=false`。OFB-ResNet 与现有 CNN 信息重叠，不再重复加入。

#### 官方代码适配的公平性规则

- 主表中的模型必须使用相同的 train/validation/test 时间切分、12→5 历史/预测长度、输入变量、TEMP/SALT 目标、海洋掩膜和指标实现；不同原论文的原生数字不能抄入本项目主表。
- 官方预训练权重若因日/月尺度、网格、深度或通道不同而不兼容，只使用其架构和作者训练代码；若使用任何迁移初始化，必须另立实验并明确标注，不能与从头训练结果混合。
- 每个外部实现固定仓库 URL、commit SHA/tag、许可证、原始配置、适配补丁、参数量和环境；没有明确许可证时不能直接复制代码进仓库，须先确认授权方式。
- 适配只允许数据接口、输入/输出头、空间尺寸和训练框架所必需的修改；核心模块发生实质变化时降级为 `paper_reimplementation`，不得继续称为官方代码适配。
- 三个必做模型先分别通过 shape、forward/backward、NaN、峰值显存、保存/恢复和一个 validation batch 的 smoke，再为每个模型族运行各自预声明、围绕公开默认值的三点 LR 网格。
- 公平训练采用相同的最大 epoch、early-stop 信息边界和 seeds；可保留作者推荐的优化器或正则，但差异必须预先写入配置和报告。若要做严格架构对照，再补一组统一优化器/损失，而不是训练后选择更好的一种口径。
- 主结果至少报告 3 seeds，并同时给出 TEMP/SALT、5 个 lead、20 个深度、参数量、峰值显存和 wall time；单 seed 只用于 smoke 和筛选。

#### 有权重但只能按原生协议参考的模型

- **ORCA-DL**（Science Advances 2025）是最接近当前月尺度、约 1° 任务的外部模型，作者提供权重和推理程序，官方报告单卡测试约需 12 GB 显存。ORAS5 轨道已有海流和风应力，但 ORCA-DL 的原生 16 层全球网格、CMIP 训练分布、GODAS 初始化、标准化统计和季节到年代际 rollout 与本项目 20 层区域 12→5 任务不同；官方训练还使用 4×A100 FSDP。因此只能单列 native-protocol 参考，不能把权重直接放入公平主表。来源：[论文](https://www.science.org/doi/full/10.1126/sciadv.adu2488)、[官方代码](https://github.com/OpenEarthLab/ORCA-DL)。
- **NeuralOM**（AAAI 2026）有 MIT 许可证、训练代码和预训练资源，但它是全球 0.5° 日尺度、多通道 S2S 模型，完整数据资源也不适合当前 50 GB 临时盘。保留为后续外部验证，不作为当前必跑项。来源：[AAAI 论文](https://ojs.aaai.org/index.php/AAAI/article/view/38495)、[官方代码](https://github.com/YuanGao-YG/NeuralOM)。
- **WenHai**（Nature Communications 2025）和 **XiHe** 有官方 ONNX 权重，但都是全球 1/12° 日尺度预测，需要海流以及更多大气/海洋变量；只允许出现在外部原生协议表中。来源：[WenHai 论文](https://www.nature.com/articles/s41467-025-57389-2)、[WenHai 代码](https://github.com/Cuiyingzhe/WenHai)、[XiHe 代码](https://github.com/Ocean-Intelligent-Forecasting/XiHe-GlobalOceanForecasting)。

当前已实现的 `paper_reimplementation_baselines` 包括 `tianhai_paper`、`fuxi_ocean_paper`、`fuxi_ons_paper` 和 `axiomocean_paper`。它们继续使用独立的 `paper_reimplementation_lr_calibrate`、`paper_reimplementation_baselines` 和 `paper_reimplementation_confirm_validation`，结果不参与 32 个模块消融 contrast，也不能被用来证明对作者原始系统的直接超越。

### 5.3 统计单位和规则

每个 contrast 对相同起报月份配对：

```text
score = log(MSE_reference / MSE_candidate)
```

正值表示 candidate 更好。TEMP/SALT 先各自计算，再等权形成 macro，不能按通道数或数值尺度加权。bootstrap 使用：

- circular moving block，block length = 5；
- 10,000 次重复，seed = 20260824；
- 多随机种子确认时先重采样 seed，再在 seed 内重采样起报 block；
- 32 个 contrast 用 Benjamini–Hochberg 控制多重比较。

单种子筛选进入下一阶段需同时满足：macro 几何 MSE 降幅至少 1%，`P(candidate better) ≥ 0.8`，任一变量不得恶化超过 1%，且 BH `q ≤ 0.10`；否则标记 `inconclusive` 或 `do_not_advance`。

多种子确认支持一个模块需同时满足：macro 95% CI 下界大于 0、TEMP/SALT 都不恶化、BH `q ≤ 0.05`。还要检查 5 个 lead、20 个深度、各气候月份和 persistence/climatology skill，避免总体均值掩盖局部失败。

### 5.4 模块取舍

- `supported`：保留，并报告效果量、区间和计算代价；
- `contradicted`：从最终架构删除；
- `uncertain`：默认不进入精简架构，除非有预先说明的物理必要性，再做一次独立诊断；
- 删除模块后必须形成一个新的 **pruned architecture**，重新校准学习率并用 3 个 seed 在 validation 上与原 full 比较；不能把若干独立消融的最佳数字直接拼成最终模型；
- 若大模块收益成立但参数量差异很大，再加参数量近似（建议 ±5%）的容量匹配控制，区分“模块机制”与“只是参数更多”；
- 同时比较参数量、峰值显存、平均 epoch 时间和 MSE，提取质量—成本 Pareto 前沿。

风场若显示收益，必须追加缺失期敏感性分析；GTB 若显示收益，必须检查不同海域/季节的收益和 dense 两遍推理一致性；TSC 若显示收益，必须展示按深度和水团相关区域的增益，而不只给总体均值。

## 6. 服务器分阶段运行方案

### 6.1 服务器与资源边界

```text
SSH: root@connect.westd.seetacloud.com:39323
项目: /root/TSC-Fusion -> /root/autodl-tmp/TSC-Fusion
Python: .venv/bin/python -> /root/miniconda3
GPU: 1×RTX 5090 32GB
CPU: 宿主可见 208 逻辑核；容器 CPU quota 25 核
宿主 RAM: 约 754 GiB
容器内存上限: 90 GiB，无 swap
可用临时盘: 约 50GB
```

正式配置统一使用 `batch_size=151`、`num_workers=0`、`persistent_workers=false`。之前的多 worker 预取会复制大 time-group batch，是已有 `-9` 失败的主要技术嫌疑。运行中的 campaign 禁止再次同步源码。

### 6.2 同步与预检

本地：

```bash
./sync_to_server.sh
ssh -p 39323 root@connect.westd.seetacloud.com
```

远端：

```bash
cd /root/TSC-Fusion
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python scripts/validate_experiment_matrix.py
.venv/bin/python scripts/audit_dataset_protocol.py \
  --config configs/experiments/full.json \
  --hash-file \
  --output outputs/evidence/dataset_protocol_final.json
```

有效的 v6 cache 不应因切换实验配置而清空；cache key 已覆盖预处理语义，复用是预期行为。只有日志明确指出某一个 key 缺文件、元数据不一致或没有 `_SUCCESS` 时，才隔离该 key。先核对绝对根目录和精确 key，再移动到隔离目录，禁止批量 `rm -rf`：

```bash
CACHE_ROOT=$(readlink -f .cache/preprocessed)
test "$CACHE_ROOT" = "/root/autodl-tmp/TSC-Fusion/.cache/preprocessed"
KEY="<日志中精确的 cache key>"
TARGET=$(readlink -f "$CACHE_ROOT/$KEY")
case "$TARGET" in "$CACHE_ROOT"/*) ;; *) exit 1 ;; esac
test -f "$TARGET/payload/metadata.json"
mkdir -p "$CACHE_ROOT/.quarantine"
mv -- "$TARGET" "$CACHE_ROOT/.quarantine/${KEY}.$(date +%Y%m%dT%H%M%S)"
```

### 6.3 阶段、配置与应产生产物

| 阶段 | 配置/数量 | epoch | 读取哪个 split | 该阶段应回答什么 |
|---|---:|---:|---|---|
| `postfix_smoke` | TSC-Fusion、ConvLSTM、CNN，共 3 个 | 1 | TSC 仅 validation；基线不做训练后报告 | 数据共享、完整 GTB batch、前反向、保存和恢复能否跑通；不比较性能 |
| `global_lr_calibrate` | 5 个模型族 ×4 个 LR，共 20 个 | 8 | 不生成最终 split 报告，只使用训练过程中的 validation selection loss | 为 full、ConvLSTM、CNN、TEMP-only、SALT-only 选择粗学习率 |
| `screen` | 30 个模型，seed 42 | 30 | validation | 计算 32 个预声明 contrast，筛掉无效模块 |
| `confirm_validation` | 只运行筛选留下的模型，3 seeds | 80 | validation | 给出层级 block-bootstrap 置信区间并冻结 pruned architecture |
| pruned 复验 | full 与 pruned，各 3 seeds | 重新校准后确定 | validation | 验证组合删减后的整体模型确实优于或不劣于 full |
| `paper_reimplementation_lr_calibrate` | 4 个论文级模型 ×4 个 LR，共 16 个 | 8 | 不生成最终 split 报告，只使用 validation selection loss | 分别为 TianHai、FuXi-Ocean、FuXi-ONS、AxiomOcean 复现选择 LR |
| `paper_reimplementation_baselines` | 4 个论文级复现，seed 42 | 30 | validation | 在同协议下做单 seed 可行性和初筛；不冒充官方结果 |
| `paper_reimplementation_confirm_validation` | 预先冻结的论文级模型，3 seeds | 80 | validation | 获得论文级复现的多 seed 不确定性和资源统计 |
| ORAS5 `global_lr_calibrate`（独立矩阵） | 4 个模型族 ×3 个预声明 LR，共 12 个 | 8 | 只使用 validation selection loss | 在读 test 前为每个架构选择非边界学习率 |
| ORAS5 `screen`（独立矩阵） | TSC-Fusion + OFB-FourCastNet/ClimaX/Swin，共 4 个 | 30 | validation | 在统一 fixed-AP residual 任务上筛选近期公开架构 |
| ORAS5 `confirm_validation`（独立矩阵） | 上述 4 个模型 ×3 seeds | 80 | validation | 报告参数量、资源与多 seed 的 AP skill 不确定性 |
| `final_test` | 冻结后的 pruned、full 和必要基线，各 3 seeds | 与确认一致 | test，只运行一次 | 论文最终泛化结果；当前尚未启用该阶段 |
| dense full-map | 代表 seed/checkpoint，3 个起报时刻 × lead 1/3/5 | 不训练 | 最终冻结 test 时刻 | 开阔海域空间结构、覆盖范围、拼接一致性 |

所有训练作业在单张 GPU 上串行执行。旧数据轨道共享设置为 `batch_size=151`、AMP 开启、`num_workers=0`、`persistent_workers=false`、`torch.compile=false`、按起报月份分组；full 才启用 time-group GTB，ConvLSTM/CNN 不启用 GTB。ORAS5 轨道的 TSC-Fusion 使用完整 time-group `batch_size=76`，三个 OFB adapter 不使用 GTB、使用各自冻结配置中的 `batch_size=8`。两个矩阵的 `screen` 都固定为 30 epoch、scheduler patience 4、early-stop patience 10；`confirm_validation` 固定为最多 80 epoch、patience 8/25。不同轨道或模型族的学习率不能在查看 test 后临时改变。

OFB 三个 architecture adapter 已通过 shape、forward/backward、奇数空间尺寸和严格 AP 初值测试，运行入口是 `configs/oras5_recent_baseline_matrix.json`。它与旧数据轨道的 `configs/experiment_matrix.json` 完全分离；启动时必须同时传 `--matrix`，不得误用旧矩阵的同名 `screen`。最终 test 条目仍只以 `_final_test_template` 保存，直到模型、优化器和 validation 选择全部冻结。

按当前 smoke 速度粗估，LR 校准约需 2–4 GPU 小时，30 个 screen 作业约需 10–15 GPU 小时；confirmation 的耗时取决于晋级候选数，不能预先把全部 30 个模型都跑 3 seeds。每阶段启动前至少确认 15GB 可用磁盘；阶段结束后先汇总 JSON，再决定保留哪些中间 checkpoint。时间估计只用于排程，不是结果，也不得据此提前裁剪模型。

每阶段完成后应得到两层产物：campaign 层的 `campaign_manifest.json`、`experiment_queue_state.json` 和逐作业日志；run 层的解析后 `config.json`、`run_summary.json`、训练曲线、best/latest checkpoint、scaler、评估 JSON（若该阶段允许）及 `_SUCCESS`。缺任一层均视为不完整运行。

smoke 启动示例：

```bash
cd /root/TSC-Fusion
mkdir -p run_logs
HASH=$(.venv/bin/python -c "import json; print(json.load(open('source_state.json'))['training_source_hash'][:12])")
CAMPAIGN="${HASH}_postfix_final"
nohup setsid .venv/bin/python -u scripts/run_experiment_queue.py \
  --stage postfix_smoke \
  --campaign "$CAMPAIGN" \
  > "run_logs/${CAMPAIGN}.log" 2>&1 < /dev/null &
```

监控：

```bash
tail -f "run_logs/${CAMPAIGN}.log"
nvidia-smi
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
```

只有 3 个目录均有 `_SUCCESS`、源码哈希一致、没有非有限 loss，才进入学习率阶段。任何 `-9` 都先记录 RSS/cgroup/GPU/最后一个日志阶段，不能把失败运行当成负面模型结果。

学习率阶段：

```bash
CAMPAIGN="${HASH}_lr_v1"
nohup setsid .venv/bin/python -u scripts/run_experiment_queue.py \
  --stage global_lr_calibrate \
  --campaign "$CAMPAIGN" \
  > "run_logs/${CAMPAIGN}.log" 2>&1 < /dev/null &

.venv/bin/python scripts/select_learning_rates.py \
  --results-root "outputs/results/campaigns/${CAMPAIGN}" \
  --stage global_lr_calibrate \
  --output "outputs/evidence/${CAMPAIGN}_selection.json"
```

每个模型族按最后 3 个 epoch 的 validation selection loss 中位数选择 LR。如果最优值位于搜索边界，围绕该边界补一轮，不直接接受。选定后更新对应配置、重新验证矩阵、重新同步并使用新的源码 hash 启动 screen；LR 选择 JSON 必须随新 campaign 一起归档。

论文级复现基线使用完全独立的 campaign，不能追加到正在运行的模块消融 campaign：

```bash
PAPER_LR_CAMPAIGN="${HASH}_paper_reimplementation_lr_v1"
nohup setsid .venv/bin/python -u scripts/run_experiment_queue.py \
  --stage paper_reimplementation_lr_calibrate \
  --campaign "$PAPER_LR_CAMPAIGN" \
  > "run_logs/${PAPER_LR_CAMPAIGN}.log" 2>&1 < /dev/null &

# 冻结四个模型各自的 LR、更新配置并产生新的源码 hash 后再启动：
PAPER_CAMPAIGN="<new_hash>_paper_reimplementation_v1"
nohup setsid .venv/bin/python -u scripts/run_experiment_queue.py \
  --stage paper_reimplementation_baselines \
  --campaign "$PAPER_CAMPAIGN" \
  > "run_logs/${PAPER_CAMPAIGN}.log" 2>&1 < /dev/null &
```

单 seed validation 结果只用于检查复现是否有基本预测能力和是否值得进入确认阶段。确认模型集合必须在查看 test 前写入 campaign manifest，再运行：

```bash
.venv/bin/python -u scripts/run_experiment_queue.py \
  --stage paper_reimplementation_confirm_validation \
  --only tianhai_paper fuxi_ocean_paper fuxi_ons_paper axiomocean_paper \
  --campaign "<hash>_paper_reimplementation_confirm_v1"
```

若其中某个模型因实现错误或资源限制未进入确认，必须在运行前从 `--only` 中删除并写明客观原因；不能因为它的 validation 分数较差而事后删除。官方代码适配模型待实现后沿用同一 campaign 隔离方式，但不得复用 `paper_reimplementation_*` 标签。

筛选和统计：

```bash
CAMPAIGN="<new_hash>_screen_v1"
nohup setsid .venv/bin/python -u scripts/run_experiment_queue.py \
  --stage screen \
  --campaign "$CAMPAIGN" \
  > "run_logs/${CAMPAIGN}.log" 2>&1 < /dev/null &

.venv/bin/python scripts/aggregate_results.py \
  --results "outputs/results/campaigns/${CAMPAIGN}/screen" \
  --output "outputs/aggregate/${CAMPAIGN}" \
  --strict \
  --allow-protocol-difference input_variables \
  --allow-protocol-difference expected_canonical_windows_per_origin \
  --allow-protocol-difference data_protocol.train

.venv/bin/python scripts/compare_ablation_contrasts.py \
  --results-root "outputs/results/campaigns/${CAMPAIGN}" \
  --stage screen \
  --strict \
  --output "outputs/evidence/${CAMPAIGN}_contrasts.json"
```

确认阶段必须使用 `--only` 固定候选集合，新 campaign manifest 会锁定选择，不能中途增删：

```bash
.venv/bin/python -u scripts/run_experiment_queue.py \
  --stage confirm_validation \
  --only full <候选1> <候选2> \
  --campaign "<hash>_confirm_v1"
```

在 validation 冻结最终模块、学习率、epoch/early-stop 规则和 seeds 后，才把隐藏模板改成显式 `final_test`，运行矩阵校验并写一份冻结决策记录。之后 test 只执行一次；不得根据 test 结果返回选择模块或超参数。

### 6.4 最终 dense 拼图

最终代表 checkpoint 至少运行输入结束索引 95、105、115，lead 0、2、4：

```bash
.venv/bin/python -u predict_full_map.py \
  --model_dir outputs/results/campaigns/<campaign>/confirm_validation/<model>/seed_42 \
  --base_time_index 95 --steps 0 2 4
```

另外两个起报时刻分别运行。三次结果都要核对：canonical bank token 数、dense 窗口数、海洋覆盖率、未覆盖 NaN、不同 micro-batch 大小的数值一致性，以及 TEMP/SALT 分变量、分 lead、分深度指标。

## 7. 必须收集齐的证据清单

### 数据与协议

- 数据文件 SHA-256、mtime、shape、变量、时间标签和深度值；
- train/validation/test 的目标边界、历史承接规则、起报列表；
- canonical/dense 窗口完整坐标列表、掩膜稳定性、海洋覆盖率；
- 各输入变量 finite fraction、全空时间片和填补规则。

### 每次训练

- campaign manifest、训练源码 hash、矩阵 hash、完整解析后 config；
- seed、硬件、PyTorch/CUDA、确定性开关；
- 参数量、峰值显存、进程自身峰值 RSS、队列监测的进程树峰值 RSS、wall time、初始化/训练/评估分阶段时间和每 epoch 时间；
- train objective 和 validation selection loss 全曲线；
- best epoch/checkpoint、scaler、通道 schema、数据协议；
- `_SUCCESS` 和失败运行的明确 failure record。

### 指标与统计

- TEMP/SALT 的物理 MAE/RMSE/MSE、correlation、R²；
- normalized/anomaly overall；
- 5 个 lead、20 个深度、20 个 validation 起报和气候月份分层；
- 三个简单基线及 skill；
- 32 个 contrast 的配对 origin、效果量、95% CI、概率、p、BH q；
- 3-seed 原始值和层级 bootstrap 结果，不能只保留均值。

### 空间与计算证据

- full-map 数值 NPZ，而不只是 PNG；
- coverage/ocean mask、blend weight、窗口列表和预测来源；
- 典型成功区域与失败区域，不能只挑最好看的图；
- 质量—参数量—显存—速度表；
- 与外部方法比较时保存官方仓库 URL、commit SHA/tag、许可证、原始配置、适配补丁和环境锁定文件；仓库内的论文启发式代理不能冒充官方 baseline；
- 官方代码适配模型保存从头训练与迁移初始化的明确标记；原生官方权重保存权重校验和、原始输入协议和预处理版本；
- `paper_reimplementation_baselines`、`official_code_adapted_baselines` 和 `official_native_protocol_references` 分目录汇总，表格标题和图例不得省略证据等级。

## 8. 论文何时以及如何修改

已对权威 `paper.tex` 做只读核对，当前不能直接沿用的内容包括：文中数据口径仍是 2005–2020 年西太平洋，而实际数据是 2009-01 至 2019-01 的全球规则网格；文中输入仍写 PTEMP/PDEN/SPICE，而代码输入是 TEMP/SALT/SSHA/UWND/VWND；文中把深度写成独立 `D` 轴，而实现把深度折叠进 channel；文中的局地/频谱混合方式与当前低频对角 FFT 加 `1×1` 混合不一致；文中尚未描述训练期 anomaly/climatology、Global Token Bank、canonical/dense 两遍推理和联合温盐的新协议；旧实验配置、数字和图也没有当前哈希下的可追溯证据。因此现在只保存差异清单，不改正文、不搬用旧数字。

只有满足以下 evidence freeze 才开始改 `.tex`：

1. dataset audit、最终源码 hash、配置和所有 seeds 完整；
2. screen 与 confirmation contrast 报告完成；
3. pruned architecture 在 validation 上复验完成；
4. final test 一次性运行完成；
5. full-map 数值、coverage 和资源统计完整；
6. 所有表格和图都能从保存的 JSON/NPZ 重新生成；
7. 引用逐条核验，不引入无法确认的论文或数字。

届时直接修改权威 `.tex`，重点重写：

- **Abstract/Introduction**：只保留被消融和多 seed 支持的贡献；无效模块从标题、摘要和贡献列表删除；
- **Data**：改为实际的 2009-01 至 2019-01 全球数据、20 层、五个输入变量和连续时间切分；
- **Method**：明确 anomaly/climatology、深度折叠进通道、三分支、GTB time-group 和 dense 两遍推理；不得写不存在的 PTEMP/PDEN/SPICE 或独立深度轴操作；
- **Experiments**：写真实 batch=151、学习率校准、30/80 epoch 阶段、3 seeds、validation 筛选和一次性 test；分别说明通用基线、官方代码适配、论文级复现与官方原生协议参考，主表只纳入同协议重训结果；
- **Metrics**：TEMP/SALT 分单位报告，说明 origin-level block bootstrap、BH 校正和三个基线；
- **Results**：先主结果，再模块效果量和区间，再分 lead/深度/区域、计算代价和失败案例；
- **Figures**：全部由冻结 NPZ/JSON 生成，旧硬编码图片不得复用；
- **Limitations**：开阔海域覆盖 78.72%、海岸缺失、卷积没有周期 padding、风场缺测、验证/测试起报数有限、空间窗口重叠，以及外部 baseline 可比性。

TEOS-10、密度/Spice 诊断目前没有经过代码和数值验证，不能保留为已完成实验。若确实需要，必须先实现、加测试、保存可复算数据并单独审计。

## 9. Go / No-Go 门槛

- **进入 LR 校准（已通过）**：服务器 3 个 postfix smoke 全部 `_SUCCESS`，无 `-9`/NaN/协议漂移；当前 LR campaign 已启动。
- **进入 screen**：每个模型族 LR 不在搜索边界，选择文件和源码已归档。
- **进入 confirmation**：30 个 screen 作业完整，32 个 contrast 状态为 completed，候选规则未事后修改。
- **冻结 pruned architecture**：3-seed validation 支持且无 TEMP/SALT/关键 lead 明显退化，资源收益已记录。
- **开放 final test**：模型、训练配置、随机种子、统计代码和论文声明草案全部先冻结。
- **开始改论文**：final test、full-map、消融、资源和复现证据全部齐全。

任何门槛不满足时，停在该阶段解决根因；不得用手选样本、改统计口径、换 test 配置或引用旧图表绕过。
