# TSC-Fusion 海洋温盐预测框架

基于 PyTorch 的 TSC-Fusion 时空预测框架，针对海洋温度/盐度场提供统一配置、全球窗口训练、气候态 anomaly 建模、overlap-tile 推理融合与可扩展的模型架构。

---

## 目录

1. [概览](#概览)
2. [项目结构](#项目结构)
3. [安装与环境](#安装与环境)
4. [快速上手](#快速上手)
5. [数据管线](#数据管线)
6. [模型架构](#模型架构)
7. [配置参考](#配置参考)
8. [训练流程](#训练流程)
9. [预测与评估](#预测与评估)
10. [推理融合](#推理融合)
11. [进阶用法](#进阶用法)
12. [常见问题](#常见问题)
13. [更新日志](#更新日志)

---

## 概览

- **任务**：基于历史 12 步海洋变量（TEMP、SALT、SSHA、UWND、VWND）预测未来 5 步温度/盐度场。
- **总体范式**：从全海洋有效窗口训练一个共享总模型；任意窗口输入历史序列后，模型输出该窗口未来预测。
- **数据工程**：在全海洋数据范围内进行 2D（经度+纬度）滑动窗口采样，仅保留 100% 纯海洋覆盖的 32°×21° 窗口；训练/验证/测试使用不同步长（密集/稀疏），推理阶段使用密集重叠滑窗 + 余弦权重融合实现全图无缝拼接。
- **气候态 anomaly**：默认基于训练时间段计算每个窗口的月气候态，TEMP/SALT 以 anomaly 形式训练和预测，同时把气候态作为输入特征；评估和预测输出会自动加回气候态恢复到物理量。
- **模型**：TSC-Fusion 架构 — 热盐结构记忆、全局频谱低模态分支、三维时空结构分支、局部空间分支、Global Token Bank 跨窗口注意力、特征融合与门控集成预测。
- **数据源**：`Data/FullData_preprocessed.nc`。

---

## 项目结构

```text
├── config.py                        # 统一配置中心（所有参数集中管理）
├── convlstm_model.py                # TSC-Fusion 模型定义
├── data_loader.py                   # 数据加载、2D滑动窗口、气候态anomaly、标准化、编码
├── train.py                         # 训练主程序
├── predict.py                       # 预测脚本 + overlap-tile 全图推理
├── main.py                          # 统一入口（train/test_data/test_model）
├── font_config.py                   # 中文字体配置
├── paper_reimplementation_models.py # 论文复现参考模型
├── requirements.txt                 # 依赖列表
├── README.md                        # 项目文档（本文件）
└── Data/
    ├── FullData_preprocessed.nc     # 预处理后的海洋数据
    ├── Description.txt
    └── IO_Description.txt
```

---

## 安装与环境

```bash
pip install -r requirements.txt
```

核心依赖：PyTorch ≥1.9、xarray、matplotlib、scikit-learn、numpy、tqdm。

建议使用 GPU（CUDA）训练。若显存不足，可在 `config.py` 中降低 `batch_size` 或调整 `hidden_dims`。

---

## 快速上手

### 1. 环境检查（可选）

```bash
python main.py --mode test_data      # 验证数据加载与预处理
python main.py --mode test_model     # 验证模型结构
```

### 2. 开始训练

```bash
python train.py
```

- 终端提示输入训练备注（可留空）。
- 自动加载统一配置并校验。
- 2D 滑动窗口搜索 100% 海洋区域，按模式选择步长。
- 使用训练时间段计算月气候态，TEMP/SALT 默认转为 anomaly 学习目标。
- 训练/验证/测试按时间顺序 60%/20%/20% 划分。
- 自动保存 `best_model.pth`、`latest_checkpoint.pth`、`config.json`、`training_curves.png`、`evaluation_results.json`。
- TensorBoard 日志写入 `logs/`。

训练结果目录示例：

```
outputs/results/0_results_20250531_120000_experiment/
├── best_model.pth
├── latest_checkpoint.pth
├── config.json
├── training_curves.png
├── evaluation_results.json
├── training_note.txt
└── logs/
```

### 3. 预测

先查看可用模型编号：

```bash
ls outputs/results
```

使用编号运行预测：

```bash
python predict.py --model 0

# 可选参数
python predict.py --model 0 --samples 5
```

- 根据编号精确选择训练结果目录。
- 脚本读取 `config.json` 并加载 `best_model.pth`。
- 测试集包含所有 100% 海洋窗口（稀疏步长）。
- 输出预测图、误差分析、`predictions.npz`、`metrics.json`。

---

## 数据管线

### 数据描述

- **输入变量**：TEMP、SALT、SSHA、UWND、VWND
- **目标变量**：TEMP（可配置为 TEMP + SALT 双变量）
- **序列长度**：12 步输入 → 5 步预测
- **深度范围**：0–1000 m（可配置 `depth_range`）

### 2D 滑动窗口策略

在整个海洋数据范围内进行经度+纬度双向滑动：

| 模式 | 经度步长 | 纬度步长 | 说明 |
|---|---|---|---|
| Train | 8° | 8° | 密集滑窗，最大化训练样本 |
| Val | 16° | 10° | 稀疏重叠，减少数据重复 |
| Test | 16° | 10° | 稀疏重叠，干净评估 |

- **窗口尺寸**：32°×21°（由 `lon_range`/`lat_range` 跨度决定）
- **筛选条件**：仅保留 100% 海洋覆盖的窗口（`ocean_threshold=1.0`）
- 所有模式（train/val/test）均包含全部有效窗口
- 每个窗口内的数据按时间维度 60%/20%/20% 划分

### 气候态 anomaly 策略

默认启用 `enable_climatology_anomaly=True`：

- 对 `anomaly_variables`（默认 TEMP、SALT）按窗口、月份、深度和网格计算训练期月气候态。
- 模型学习 `原始值 - 月气候态` 的标准化 anomaly，减少季节循环和区域均值差异对总模型的干扰。
- `CLIMATOLOGY_TEMP` / `CLIMATOLOGY_SALT` 会作为额外输入通道，帮助模型知道当前窗口的局地背景态。
- `inverse_transform_targets()`、训练评估和 `predict.py` 会自动执行 scaler inverse 并加回对应未来月份气候态，输出物理量。

### 预处理流程

1. xarray 加载 NetCDF，仅按深度切片（保留完整经纬度范围）。
2. 2D 滑动搜索 100% 海洋区域。
3. 对每个有效窗口切片数据，填充缺失值。
4. 使用训练集时间段计算月气候态，并将目标变量转换为 anomaly。
5. 使用训练集时间段的全局统计量进行 StandardScaler 标准化（防止数据泄露）。
6. 拼接气候态、空间、深度和时间编码通道。
7. 构造 (input_sequence, target_sequence) 对并返回 PyTorch 张量。

---

## 模型架构

### TSC-Fusion 主干

- **热盐结构记忆**：基于 TEMP、SALT、PTEMP、PDEN、SPICE 的水团原型路由与 Transformer 原型更新。
- **全局频谱分支**：使用 2D FFT/RFFT 的低频模态建模大尺度海洋空间结构。
- **三维结构分支**：以变量-时间-空间立方体为输入，使用 3D 卷积与时间注意力聚合结构特征。
- **局部空间分支**：对展平后的历史序列执行 Conv + GroupNorm + GELU 局部上下文建模。
- **特征融合与门控集成**：拼接多分支特征，经 1×1 融合、残差细化、空间 Transformer 和多成员门控集成输出多步预测。
- **Global Token Bank**：将同一历史起点 batch 内各空间窗口池化为全局 token bank，每个窗口的网格特征通过 cross-attention 读取其它窗口上下文，用同一套权重补充远场信号。该模块解决的是 ENSO、季风遥相关、上游传播等可能超出 32°×21° 局地窗口的问题；窗口内 spectral 分支仍只负责窗口内低频结构。
- **持久性残差**：从最后一个历史步的目标变量构造 residual base，提升多步预测稳定性。

### 配置开关

| 模块 | 配置键 | 说明 |
|---|---|---|
| 空间位置编码 | `enable_positional_encoding` | 经纬度/深度正弦-余弦编码 |
| 时间编码 | `enable_time_encoding` | 月份傅里叶 + 年份趋势 |
| 气候态 anomaly | `enable_climatology_anomaly` | 训练期月气候态 + anomaly 目标 |
| Global Token Bank | `enable_global_token_bank` | 同时间跨窗口 attention，引入远地空间上下文 |
| TSC 消融 | `ablation_disable_tsc` | 关闭热盐结构记忆，用于消融实验 |
| 频谱消融 | `ablation_disable_spectral` | 关闭全局频谱分支 |
| 3D 消融 | `ablation_disable_3d` | 关闭三维结构分支 |
| 集成消融 | `ablation_disable_ensemble` | 关闭门控集成，退化为单预测头 |

### 损失函数

- **主损失**：MSE（温度/盐度可配置权重 `temp_weight`/`salt_weight`）
- **梯度损失**：Sobel 梯度分布匹配（`use_gradient_loss`，权重 `gradient_loss_weight`）

---

## 配置参考

### 核心数据配置

```python
DATA_CONFIG = {
    'data_path': 'Data/FullData_preprocessed.nc',
    'input_variables': ['TEMP', 'SALT', 'SSHA', 'UWND', 'VWND'],
    'target_variables': ['TEMP'],
    'sequence_length': 12,
    'prediction_length': 5,
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'lon_range': [130.5, 162.5],    # 窗口经度跨度 = 32°
    'lat_range': [6.5, 27.5],       # 窗口纬度跨度 = 21°
    'depth_range': [0, 1000.0],

    # 2D 滑动窗口
    'sliding_enabled': True,
    'ocean_threshold': 1.0,         # 仅 100% 海洋
    'train_stride_lon': 8.0,        # 训练密集步长
    'train_stride_lat': 8.0,
    'val_stride_lon': 16.0,         # 验证稀疏步长
    'val_stride_lat': 10.0,
    'test_stride_lon': 16.0,        # 测试稀疏步长
    'test_stride_lat': 10.0,

    # 推理融合
    'inference_stride_lon': 4.0,    # 推理密集步长
    'inference_stride_lat': 4.0,
    'taper_ratio': 0.25,            # 余弦衰减带比例
    'min_blend_weight': 1e-3,       # 边缘最低权重

    # 气候态 / anomaly
    'anomaly_variables': ['TEMP', 'SALT'],
    'climatology_period': 12,
    'include_climatology_features': True,
    'climatology_feature_variables': ['TEMP', 'SALT'],
}
```

### 核心训练配置

```python
TRAINING_CONFIG = {
    'epochs': 20,
    'learning_rate': 1.57e-4,
    'batch_size': 8,
    'group_batches_by_time': True,
    'weight_decay': 1e-4,
    'grad_clip_norm': 1.0,
    'scheduler_patience': 10,
    'early_stopping_patience': 20,
    'temp_weight': 0.5,
    'salt_weight': 0.5,
    'use_gradient_loss': True,
    'gradient_loss_weight': 0.1,
}
```

完整参数列表见 `config.py`。

---

## 训练流程

1. **初始化**：加载配置 → 创建数据加载器（自动检测通道数）→ 构建模型。
2. **数据增强**：2D 滑动搜索 100% 海洋窗口，train/val/test 按步长分别采样。
3. **同时间分组**：`group_batches_by_time=True` 时，DataLoader 将同一历史起点的不同空间窗口放入同一 batch，供 Global Token Bank 做跨窗口注意力。
4. **目标构造**：用训练时间段计算月气候态，TEMP/SALT 默认转为 anomaly，并把局地气候态拼回输入通道。
5. **训练循环**：
   - 每 epoch 执行 train_epoch + validate_epoch。
   - 梯度裁剪（`max_norm=1.0`）+ ReduceLROnPlateau 调度。
   - 温度收敛后自动触发额外降学习率（优化盐度）。
   - NaN/Inf 检测与自动跳过。
   - 自动保存最佳模型和检查点。
6. **评估**：加载最佳模型，在测试集上计算 MSE/MAE/RMSE/Correlation/R²；主指标为恢复气候态后的物理量，同时保留 normalized 指标。分组 batch 会携带样本 index，保证反标准化、加回气候态和 baseline 对齐原始窗口。

---

## 预测与评估

### 标准预测

```bash
python predict.py --model <编号>
```

- 使用 test 步长（稀疏重叠）窗口进行逐窗口预测。
- anomaly 模式下自动将预测 anomaly 加回未来月份气候态，保存物理量结果。
- 输出温度/盐度对比图、误差趋势图、区域 RMSE 热力图。
- 指标保存在 `metrics.json` 中。

### 全图 Overlap-Tile 推理

```python
from predict import SmartOceanPredictor

predictor = SmartOceanPredictor(model_index=0)
result = predictor.predict_full_map(
    target_lon_range=[130.5, 162.5],
    target_lat_range=[6.5, 27.5],
)

# result['blended_pred']   — (C, H, W) 融合预测
# result['blended_target'] — (C, H, W) 融合真实值
# result['weight_sum']     — (H, W) 每格点权重和
# result['lons'] / ['lats'] — 坐标
```

全图推理会把同一输入时间的候选窗口按 `batch_size` 成批送入模型，因此启用 Global Token Bank 时，密拼窗口之间也会共享远地上下文；若 batch 只有 1 个窗口，该模块自动退化为局地推理。

**融合策略**：
1. 在目标区域以 `inference_stride`（默认 4°）生成密集重叠窗口。
2. 每个窗口通过同一套 `OceanDataset` 管线构造输入，保持气候态、位置、深度、时间编码与训练一致。
3. 使用 2D 余弦衰减权重（`taper_ratio=0.25`）加权累积。
4. 最终预测 = `Σ(W_i × pred_i) / Σ(W_i)`，消除拼接伪影。
5. 预留 NaN/陆地 mask 支持（`build_validity_mask`），非海洋区域权重置零。

---

## 进阶用法

### 自定义配置

```python
from config import DEFAULT_CONFIG, update_config

custom = update_config(DEFAULT_CONFIG,
    epochs=200,
    learning_rate=1e-3,
    batch_size=16,
)
```

### 快速实验

```python
quick_config = {
    'epochs': 5,
    'sliding_enabled': False,
    'batch_size': 4,
    'device': 'cpu',
}
```

### 服务器部署

项目部署路径：`/root/TSC-Fusion/`

SSH 连接：
```bash
ssh -p 36401 root@connect.westd.seetacloud.com
cd /root/TSC-Fusion
```

---

## 常见问题

### 数据管线

- **Q**: 找不到有效窗口？**A**: 检查 `ocean_threshold`（100% 海洋可能过于严格），可下调至 0.9。
- **Q**: 验证/测试集序列数为 0？**A**: 减小 `sequence_length`/`prediction_length` 或增加 `val_ratio`/`test_ratio`。
- **Q**: 窗口之间空间形状不一致？**A**: 数据网格应均匀分布；若不一致，`_merge_and_normalize_data` 会自动处理。
- **Q**: 为什么默认预测 anomaly？**A**: 海温/盐度有强季节循环和区域背景差异；让总模型学习 anomaly 通常比直接拟合原始值更稳，输出阶段会自动加回气候态。

### 训练

- **Q**: CUDA 内存不足？**A**: 降低 `batch_size`、`hidden_dims` 或关闭滑动增强。
- **Q**: 训练太慢？**A**: 启用 `mixed_precision=True`、`compile_model=True`。
- **Q**: 损失为 NaN？**A**: 降低 `learning_rate`、检查数据中极端异常值。

---

## 更新日志

### v4.2.0 · 2026-05-31

- **Global Token Bank**：同一历史时间起点的多空间窗口共享 token bank，并通过 cross-attention 引入远地上下文。
- **同时间 batch sampler**：新增 `group_batches_by_time`，保证 token bank 内样本时间一致，避免把不同月份/年份状态混到同一次 attention。
- **评估对齐**：训练评估支持分组 batch 的样本 index 对齐，物理量恢复、气候态加回和 persistence/climatology baseline 指标仍对应原始窗口。

### v4.1.0 · 2026-05-31

- **全球总模型训练范式强化**：保持全海洋窗口共享模型，任意窗口历史输入→该窗口未来预测。
- **训练期月气候态 anomaly**：TEMP/SALT 默认减去训练段月气候态后训练，降低季节循环和区域均值混杂。
- **气候态输入通道**：新增 `CLIMATOLOGY_TEMP` / `CLIMATOLOGY_SALT` 特征，让模型显式感知局地背景态。
- **物理量恢复**：训练评估、标准预测和全图推理统一使用 `inverse_transform_targets()` 加回未来月份气候态。
- **时间解析修正**：兼容 `YYYYMM` 整数 TIME 坐标，月份傅里叶编码和气候态月份索引保持一致。

### v4.0.0 · 2025-05-31

- **2D 滑动窗口数据工程**：经纬度双向滑动，仅保留 100% 海洋覆盖区域。
- **分模式步长**：Train 8°/8°、Val 16°/10°、Test 16°/10°，经纬度独立配置。
- **Overlap-tile 推理融合**：密集滑窗 + 余弦衰减权重全图无缝拼接。
- **模型输出泛化**：模型不再绑定特定空间区域，任意窗口输入→该窗口预测。
- 移除 ARIMA/XGBoost 集成模块与单点/单变量预测脚本。
- 移除赤道对称扩增（被 2D 滑动取代）。
- 新增 TSC-Fusion 多模块架构（热盐结构记忆、全局频谱建模、三维结构建模、门控集成）。

### v3.0.0 · 2025-09-18

- 统一配置系统：`config.py` 管理全部参数，自动验证与保存。
- 预测脚本重构：自动模型发现、配置匹配、温盐分离分析。
- 可选时空编码：经纬度/深度正余弦 + 月份傅里叶 + 年份趋势。

### v2.0.0 · 2025-09-18

- 全球滑动窗口（仅经度方向）数据增强，训练样本量 ×71。
- 取消早停、优化批次与调度器。

### v1.0.0 · 初始版本

- 基础 ConvLSTM 预测框架，支持 NetCDF 数据加载与训练/预测流程。

---

打造海洋时空预测的统一实验平台。
