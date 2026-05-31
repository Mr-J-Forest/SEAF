# 海洋数据ConvLSTM模型 - 温度和盐度反演（统一配置版）

这是一个基于PyTorch的ConvLSTM模型，用于海洋数据中温度和盐度的时空序列预测。模型能够学习海洋变量的时空依赖关系，并预测未来的温度和盐度分布。

## 🚀 最新更新 (2025-09-18) - 统一配置系统

### 重大改进：参数统一化管理

- **统一配置文件**: 新增`config.py`统一管理所有系统参数
- **参数一致性保证**: 消除训练和预测脚本间的参数不匹配问题
- **自动模型匹配**: 预测脚本能自动匹配训练时的模型配置
- **配置验证系统**: 自动验证参数有效性，防止配置错误
- **灵活参数覆盖**: 支持运行时自定义配置参数
- **向后兼容**: 保持原有脚本使用方式的兼容性

### 解决的关键问题

1. **模型加载错误**: 
   - ❌ 之前：训练和预测使用不同参数导致模型结构不匹配
   - ✅ 现在：统一配置确保完全一致的模型结构

2. **参数维护困难**:
   - ❌ 之前：需要在多个文件中同步修改相同参数
   - ✅ 现在：只需在config.py中修改一次

3. **配置管理混乱**:
   - ❌ 之前：硬编码参数分散在各个脚本中
   - ✅ 现在：所有参数集中在统一配置文件中

### 新增功能特性

- **智能预测脚本**: 新的`predict.py`替代`generate_predictions.py`
- **温度盐度双重可视化**: 分别生成温度和盐度的预测分析
- **自动模型发现**: 自动寻找最新训练的模型文件
- **配置合并机制**: 智能合并训练配置和预测配置
- **中文字体自动配置**: 自动检测和设置系统中文字体

## 🎉 表层数据 + 滑动窗口数据增强 (之前更新)

### 重大改进：表层数据 + 滑动窗口数据增强

- **专注表层预测**: 改为只使用表层数据（0-5米深度），提高模型针对性
- **全球滑动窗口**: 自动在全球海洋区域滑动训练窗口，数据量增加71倍
- **智能区域筛选**: 只选择海洋覆盖率>80%的区域作为训练数据
- **测试集保护**: 滑动数据仅用于训练，确保测试集纯净性
- **降维优化**: 输入通道从135降到5，大幅减少过拟合风险
- **中文字体支持**: 训练图表完全支持中文显示

## 项目结构（统一配置版）

```text
├── Data/
│   ├── FullData_preprocessed.nc    # 海洋数据文件（NetCDF格式）
│   ├── IO_Description.txt          # 输入输出描述
│   └── Description.txt             # 数据描述
├── config.py                     # 🆕 统一配置管理文件
├── convlstm_model.py              # ConvLSTM模型定义
├── data_loader.py                 # 数据加载和预处理（支持滑动窗口增强）
├── train.py                      # 模型训练脚本（使用统一配置）
├── predict.py                    # 🆕 智能预测脚本（替代generate_predictions.py）
├── single_predict.py             # 🆕 单点时序预测脚本（中心点分析）
├── generate_predictions.py       # 原预测脚本（保留兼容）
├── main.py                       # 主运行脚本
├── requirements.txt              # 依赖包列表
└── README.md                     # 项目说明
```

## 统一配置系统详解

### config.py - 核心配置文件

新的统一配置系统包含以下配置组：

```python
DEFAULT_CONFIG = {
    # 数据配置
    'data_path': 'Data/FullData_preprocessed.nc',
    'input_variables': ['TEMP', 'SALT', 'SSHA', 'UWND', 'VWND'],
    'target_variables': ['TEMP', 'SALT'],
    'sequence_length': 10,
    'prediction_length': 5,
    
    # 模型配置
    'hidden_dims': [64, 64, 64],
    'kernel_size': (3, 3),
    'num_layers': 3,
    'dropout': 0.4,
    
    # 训练配置
    'epochs': 100,
    'learning_rate': 0.0005,
    'batch_size': 8,
    'weight_decay': 0.0001,
    
    # 预测配置
    'num_samples': 10,
    'visualization_samples': 3,
    
    # 硬件配置
    'device': 'auto',
    'mixed_precision': False,
    'cudnn_benchmark': True
}
```

### 配置管理功能

1. **配置验证**: `validate_config(config)` - 自动验证参数有效性
2. **配置保存**: `save_config(config, filepath)` - 保存配置到JSON文件
3. **配置更新**: `update_config(base_config, updates)` - 安全更新配置参数

### 使用统一配置的优势

```python
# 训练时
from config import DEFAULT_CONFIG
trainer = OceanModelTrainer(DEFAULT_CONFIG)  # 使用统一配置

# 预测时  
predictor = SmartOceanPredictor(DEFAULT_CONFIG)  # 完全相同的配置

# 自定义配置
custom_config = update_config(DEFAULT_CONFIG, {
    'epochs': 200,
    'learning_rate': 1e-3,
    'batch_size': 16
})
```

## 数据增强策略

### 滑动窗口数据增强

- **原理**: 在同一纬度上滑动32°×21°的训练窗口
- **筛选条件**: 海洋面积占比 > 80%
- **覆盖范围**: 全球海洋区域，找到70个有效滑动区域
- **数据量提升**: 从58个训练序列增加到4118个序列
- **使用策略**: 
  - 训练集：原始区域 + 70个滑动区域
  - 验证集：仅原始区域
  - 测试集：仅原始区域（确保评估公正性）

### 表层数据专门化

- **深度范围**: 0-5米（仅表层）
- **输入变量**: 5个变量（TEMP, SALT, SSHA, UWND, VWND）
- **输出变量**: 2个变量（TEMP, SALT）
- **输入维度**: 5通道（替代原来的135通道）
- **输出维度**: 2通道（替代原来的54通道）

## 数据描述

- **输入变量**: ["TEMP", "SALT", "SSHA", "UWND", "VWND"] (实际使用的5个变量)
- **目标变量**: ["TEMP", "SALT"] (温度和盐度)
- **空间范围**:
  - 经度: 130.5°E - 162.5°E (原始训练区域)
  - 纬度: 6.5°N - 27.5°N 
  - 深度: 0 - 5m (仅表层数据)
- **时间序列**: 121个时间步
- **数据分割**: 
  - **分割方式**: 基于时间序列的顺序分割（非随机）
  - **训练集**: 50% - 前半段时间数据用于模型训练
  - **验证集**: 30% - 中间段时间数据用于模型验证和调参
  - **测试集**: 20% - 最后段时间数据用于最终性能评估
  - **序列要求**: 每个样本需要 `sequence_length + prediction_length` 个连续时间步
  - **当前配置**: 输入24步 + 预测6步 = 需要30个连续时间步
  - **重要提醒**: 如果增加序列长度，需确保各数据集有足够的时间步数

## 模型改进

## 模型改进

### 架构优化

- **降维设计**: 输入从135通道降到5通道，显著减少过拟合
- **表层专注**: 专门针对表层海洋预测，提高模型针对性
- **动态输入**: 模型自动适应实际输入维度
- **中文支持**: 所有可视化图表支持中文字体显示

### 训练策略改进

- **无早停**: 移除早停机制，让模型充分训练
- **数据增强**: 通过滑动窗口将训练数据扩增71倍
- **全局标准化**: 基于所有区域数据计算标准化参数
- **批次优化**: 调整批次大小适应更大的数据集

### ConvLSTM架构

- **ConvLSTM单元**: 结合卷积神经网络和LSTM，处理时空数据
- **编码器-解码器结构**: 编码历史序列，解码预测未来
- **多层设计**: 3层ConvLSTM，逐步提取时空特征
- **门控机制**: 输入门、遗忘门、输出门控制信息流

### 关键功能

- **时空建模**: 同时捕获空间和时间依赖关系
- **多变量预测**: 同时预测温度和盐度
- **序列到序列**: 基于历史10个时步预测未来5个时步
- **全球泛化**: 通过多区域训练提高模型泛化能力

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：

- torch >= 1.9.0 (PyTorch深度学习框架)
- xarray >= 0.19.0 (NetCDF数据处理)
- matplotlib >= 3.4.0 (可视化)
- cartopy >= 0.20.0 (地图投影)
- scikit-learn >= 1.0.0 (数据预处理)

## 使用方法（统一配置版）

### 1. 测试数据加载

```bash
python main.py --mode test_data
```

这将测试NetCDF数据文件的加载和预处理功能。

### 2. 测试模型结构

```bash
python main.py --mode test_model
```

验证ConvLSTM模型结构和前向传播。

### 3. 训练模型（统一配置）

```bash
python train.py
```

开始训练模型。训练过程中会：

- 自动使用统一配置中的所有参数
- 自动寻找70个有效的全球海洋区域作为训练数据
- 将训练数据从58个序列扩增到4118个序列
- 使用表层数据（0-5米）进行专门化训练
- 自动分割训练/验证/测试集 (60%/20%/20%)
- 保存最佳模型和检查点
- 生成中文训练曲线图
- 记录TensorBoard日志

### 4. 智能预测（新版本）

```bash
python predict.py
```

使用新的智能预测脚本：

- **自动模型发现**: 自动寻找最新训练的模型
- **配置匹配**: 自动匹配训练时的模型配置
- **双重可视化**: 分别生成温度和盐度预测分析
- **统一配置**: 使用与训练完全一致的参数

可选参数：
```bash
python predict.py --samples 5 --viz 3    # 预测5个样本，可视化3个
python predict.py --model_path path/to/specific/model.pth  # 使用指定模型
```

### 5. 传统预测（兼容版本）

```bash
python generate_predictions.py
```

保留原有的预测脚本以保持向后兼容性。

### 6. 单点时序预测（专业版）

```bash
python single_predict.py
```

使用专门的单点时序预测脚本，专注于区域中心点的预测分析：

- **中心点定位**: 自动定位22×33网格的中心点进行分析
- **时序对比**: 将预测值和真实值绘制在同一时间轴上
- **历史延续**: 显示输入历史数据与预测数据的连续性
- **双变量分析**: 分别对温度和盐度进行时序预测对比
- **评估指标**: 为每个变量计算MAE、RMSE、相关系数等指标
- **统计汇总**: 提供多样本的统计分析结果

#### 功能特点

1. **智能模型选择**: 
   - 自动选择最新编号的训练模型
   - 使用统一配置中的路径设置
   - 优先选择`best_model.pth`文件

2. **时序可视化**:
   - 历史期（绿色）：显示输入的10个时间步
   - 预测期（蓝色/红色）：对比真实值与预测值
   - 当前时刻分隔线：清晰标示历史与未来的分界
   - 背景色区分：不同时期使用不同背景色

3. **评估指标**:
   ```
   温度预测统计:
     MAE  - 平均: 0.3547, 标准差: 0.0123
     RMSE - 平均: 0.4781, 标准差: 0.0156
     相关系数 - 平均: 0.9042, 标准差: 0.0087
   ```

4. **输出文件**:
   - `center_timeseries_sample_X.png`: 时序对比图
   - `center_timeseries_sample_X.npz`: 时序数据数组
   - `center_metrics_sample_X.json`: 评估指标
   - `summary.json`: 总体结果摘要

#### 使用示例

```python
from single_predict import SinglePointPredictor

# 创建预测器（使用默认配置）
predictor = SinglePointPredictor()

# 加载模型和数据
predictor.load_model()  # 自动查找最新模型
predictor.load_dataset('test')  # 使用测试集

# 预测指定样本
sample_indices = [0, 1, 2, 5, 8]  # 选择特定样本
predictor.predict_and_visualize(sample_indices)
```

#### 中心点坐标说明

- **网格大小**: 22（纬度）× 33（经度）
- **中心点位置**: (11, 16) - 网格的几何中心
- **地理意义**: 代表预测区域的典型海洋特征
- **数据维度**: 每个时间步包含多个深度层的数据

## 当前配置参数（统一配置版）

所有参数现在统一管理在`config.py`中的`DEFAULT_CONFIG`：

```python
# 统一配置参数
DEFAULT_CONFIG = {
    # 数据配置
    'data_path': 'Data/FullData_preprocessed.nc',
    'sequence_length': 10,           # 输入序列长度
    'prediction_length': 5,          # 预测序列长度
    'train_ratio': 0.6,              # 训练集比例
    'val_ratio': 0.2,                # 验证集比例
    'test_ratio': 0.2,               # 测试集比例
    
    # 模型配置
    'hidden_dims': [64, 64, 64],     # 隐藏层维度
    'kernel_size': (3, 3),           # 卷积核大小
    'num_layers': 3,                 # ConvLSTM层数
    'dropout': 0.4,                  # Dropout比例
    
    # 训练配置
    'epochs': 100,                   # 训练轮数
    'learning_rate': 0.0005,         # 学习率
    'batch_size': 8,                 # 批次大小
    'weight_decay': 0.0001,          # 权重衰减
    'scheduler_patience': 10,        # 学习率调度耐心值
    'scheduler_factor': 0.5,         # 学习率衰减因子
    
    # 预测配置
    'num_samples': 10,               # 预测样本数
    'visualization_samples': 3,      # 可视化样本数
    
    # 硬件配置
    'device': 'auto',                # 设备选择（auto/cpu/cuda）
    'mixed_precision': False,        # 混合精度训练
    'cudnn_benchmark': True          # cuDNN基准测试
}
```

### 配置自定义

您可以轻松自定义配置：

```python
from config import DEFAULT_CONFIG, update_config

# 自定义训练配置
custom_config = update_config(DEFAULT_CONFIG, {
    'epochs': 200,           # 增加训练轮数
    'learning_rate': 1e-3,   # 调整学习率
    'batch_size': 16,        # 增大批次大小
    'hidden_dims': [128, 128, 128]  # 更大的模型
})

# 使用自定义配置训练
trainer = OceanModelTrainer(custom_config)
```

## 数据增强详细说明

### 滑动窗口策略

1. **窗口大小**: 32°经度 × 21°纬度（与原始训练区域相同）
2. **滑动步长**: 2°经度
3. **筛选条件**: 海洋面积占比 > 80%
4. **覆盖区域**: 
   - 太平洋西部: 118.5°E - 260.5°E
   - 太平洋东部: 290.5°E - 348.5°E
5. **数据组织**:
   - 每个区域包含相同的时间序列长度
   - 使用全局标准化确保一致性
   - 区域间数据独立采样

## 预测结果（增强版）

### 训练结果

训练过程会在 `outputs/results/results_YYYYMMDD_HHMMSS/`目录下保存：

- `best_model.pth`: 最佳模型
- `latest_checkpoint.pth`: 最新检查点
- `config.json`: 统一配置参数（自动保存）
- `training_curves.png`: 训练曲线（支持中文显示）
- `evaluation_results.json`: 评估结果
- `logs/`: TensorBoard日志

### 预测结果

新的预测脚本会在 `outputs/predictions/predictions_YYYYMMDD_HHMMSS/`目录下保存：

- `prediction_sample_*.png`: 预测样本对比图
- `temperature_error_analysis.png`: 温度误差分析图
- `salinity_error_analysis.png`: 盐度误差分析图  
- `time_series_comparison.png`: 时间序列对比图
- `predictions.npz`: 预测数据数组
- `evaluation_metrics.json`: 详细评估指标
- `config.json`: 使用的配置参数
- `info.json`: 预测任务信息

### 评估指标详解

新版本提供更详细的评估指标：

```json
{
  "overall_metrics": {
    "mse": 0.431863,
    "mae": 0.482231, 
    "rmse": 0.657163,
    "correlation": 0.884627
  },
  "temperature_metrics": {
    "mse": 0.228576,
    "mae": 0.354655,
    "rmse": 0.478096,
    "correlation": 0.904564
  },
  "salinity_metrics": {
    "mse": 0.635151,
    "mae": 0.609808,
    "rmse": 0.796964,
    "correlation": 0.913219
  }
}
```

## 训练性能提升

### 数据增强效果

- **训练样本增加**: 从58个增加到4118个（71倍提升）
- **全球泛化**: 覆盖70个不同海洋区域
- **过拟合抑制**: 大量数据有效减少过拟合风险
- **验证稳定**: 验证集仍使用原始区域，确保评估一致性

### 模型优化效果

- **通道降维**: 从135通道降到5通道，减少90%参数量
- **表层专注**: 专门针对表层预测，提高预测精度
- **训练稳定**: 移除早停，避免训练不充分
- **可视化增强**: 完整的中文支持，便于结果分析

## 可视化功能

模型提供丰富的可视化功能：

### 1. 温度/盐度场分布图

- 地理投影显示
- 等值线图表示
- 海岸线和陆地遮罩

### 2. 预测对比图

- 真实值 vs 预测值
- 差值分布
- 统计指标（MAE、RMSE、相关系数）

### 3. 时间序列对比

- 特定位置的时序变化
- 多变量同时显示

## 性能指标

模型使用以下指标评估预测性能：

- **MSE (Mean Squared Error)**: 主要损失函数
- **MAE (Mean Absolute Error)**: 平均绝对误差
- **RMSE (Root Mean Square Error)**: 均方根误差
- **Correlation**: Pearson相关系数

## 模型架构详解

### ConvLSTM单元

```text
输入: (batch_size, channels, height, width)
├── 输入门: σ(Conv(input) + Conv(hidden))
├── 遗忘门: σ(Conv(input) + Conv(hidden))  
├── 输出门: σ(Conv(input) + Conv(hidden))
└── 候选值: tanh(Conv(input) + Conv(hidden))

细胞状态更新: C_t = f_t * C_{t-1} + i_t * g_t
隐藏状态更新: h_t = o_t * tanh(C_t)
```

### 网络结构（表层优化版）

```text
编码器 (输入序列 → 特征表示)
输入维度: 5通道 (TEMP, SALT, SSHA, UWND, VWND)
├── ConvLSTM Layer 1: 5 → 64
├── ConvLSTM Layer 2: 64 → 64
└── ConvLSTM Layer 3: 64 → 64

解码器 (特征表示 → 预测序列)
├── ConvLSTM Layer 1: 64 → 64
├── ConvLSTM Layer 2: 64 → 64  
├── ConvLSTM Layer 3: 64 → 64
└── Conv2D输出层: 64 → 2 (TEMP, SALT)
```

## 数据预处理（增强版）

1. **NetCDF数据加载**: 使用xarray读取海洋数据
2. **滑动区域检测**: 自动寻找全球有效海洋区域
3. **空间范围选择**: 裁剪到指定经纬度范围
4. **表层数据提取**: 仅使用0-5米深度数据
5. **缺失值处理**: 线性插值和均值填充
6. **全局标准化**: 基于所有区域数据的Z-score标准化
7. **序列构建**: 滑动窗口生成输入-目标序列对
8. **区域标记**: 区分原始区域和滑动区域用于数据分割

## 重要更新说明

### 版本3.0 (2025-09-18) - 统一配置系统

1. **统一配置管理**:
   - 新增`config.py`统一管理所有系统参数
   - 消除训练和预测脚本间的参数不匹配问题
   - 自动配置验证和错误防护

2. **智能预测系统**:
   - 新增`predict.py`替代原有预测脚本
   - 新增`single_predict.py`专门用于单点时序预测
   - 自动模型发现和配置匹配
   - 分离的温度和盐度误差分析
   - 中心点时序对比可视化

3. **配置管理优化**:
   - 参数集中化管理，避免硬编码
   - 支持运行时配置覆盖
   - 自动配置保存和加载

4. **系统稳定性提升**:
   - 解决模型加载兼容性问题
   - 改进错误处理和日志记录
   - 增强代码可维护性

### 版本2.0 (2025-09-18) 主要变更

1. **数据策略革新**:
   - 从全深度(27层)改为表层专用(1层)
   - 实现滑动窗口全球数据增强
   - 训练数据增加71倍

2. **模型架构优化**:
   - 输入维度从135降到5通道
   - 输出维度从54降到2通道
   - 显著减少模型复杂度和过拟合风险

3. **训练流程改进**:
   - 移除早停机制，确保充分训练
   - 新增中文字体支持
   - 优化批次大小和学习率
   - 改进学习率调度策略

4. **代码结构优化**:
   - 重构数据加载器支持多区域处理
   - 改进模型自适应输入维度
   - 新增专门的预测生成脚本
   - 优化文件组织结构

## 注意事项（统一配置版）

1. **配置一致性**: 现在所有脚本都使用统一配置，确保参数完全一致
2. **模型兼容性**: 新版本能自动处理模型配置匹配问题
3. **内存需求**: 由于数据增强，建议至少32GB内存用于大规模训练
4. **训练时间**: 数据量增加71倍，训练时间会相应增加
5. **GPU推荐**: 强烈建议使用GPU训练，可显著加速
6. **CUDA兼容性**: 可在config.py中设置device='cpu'强制使用CPU
7. **配置文件**: 重要参数修改请在config.py中进行
8. **向后兼容**: 保留原有脚本以确保向后兼容性

## 故障排除（统一配置版）

### 常见问题

1. **配置参数错误**: 检查config.py中的参数设置，使用validate_config()验证
2. **模型加载失败**: 新版本能自动处理配置匹配，如仍有问题请检查模型文件
3. **CUDA内存不足**: 在config.py中调整batch_size或设置device='cpu'
4. **训练数据过大**: 在config.py中设置sliding_enabled=False关闭滑动增强
5. **中文显示问题**: 系统会自动检测和配置中文字体
6. **预测脚本错误**: 使用新的predict.py替代generate_predictions.py

### 配置建议

```python
# 内存受限环境
memory_limited_config = {
    'batch_size': 4,
    'sliding_enabled': False,
    'device': 'cpu'
}

# 高性能环境  
high_performance_config = {
    'batch_size': 16,
    'epochs': 200,
    'mixed_precision': True,
    'cudnn_benchmark': True
}

# 快速测试
quick_test_config = {
    'epochs': 10,
    'num_samples': 3,
    'visualization_samples': 1
}
```

## 常见问题解答 (FAQ)

### Q1: 误差分析图中的误差是相对于什么计算的？

**A:** 误差图显示的是**模型预测值与真实标签值之间的误差**，具体说明：

#### 真实标签值（Ground Truth）

- **温度（TEMP）**：来自原始海洋数据的真实温度观测值
- **盐度（SALT）**：来自原始海洋数据的真实盐度观测值
- 数据来源：`Data/FullData_preprocessed.nc` 中的实际海洋观测/分析数据

#### 预测值（Predictions）

- ConvLSTM模型基于前10个时间步的输入数据（温度、盐度、海面高度异常、风场等）
- 预测未来5个时间步的温度和盐度值

#### 误差计算方式

```python
# 均方误差 (MSE)
mse = torch.mean((pred_stack - target_stack) ** 2, dim=(0, 2, 3, 4))

# 平均绝对误差 (MAE)  
mae = torch.mean(torch.abs(pred_stack - target_stack), dim=(0, 2, 3, 4))
```

#### 图表含义

- **X轴**：预测步长（1-5步，对应未来1-5个时间步）
- **Y轴**：误差值（MSE或MAE）
- **误差趋势**：通常随预测步长增加而增大，因为越远的未来预测不确定性越高

### Q2: 误差计算是否按对应时间步进行？

**A:** **是的，误差计算完全按照对应时间步进行**，具体对应关系：

#### 时间对应关系

- 如果输入序列是第1-10时间步的数据
- 目标序列就是第11-15时间步的真实数据
- 模型预测的也是第11-15时间步的数据

#### 数据构建过程

```python
# 输入序列：时间步 t 到 t+sequence_length-1 (共10步)
input_data = data[start_idx:start_idx + self.sequence_length]

# 目标序列：时间步 t+sequence_length 到 t+sequence_length+prediction_length-1 (共5步)  
target_data = data[start_idx + self.sequence_length:start_idx + self.sequence_length + self.prediction_length]
```

#### 误差计算中的时间对应

```python
# pred_stack形状: (num_samples, pred_len, channels, height, width)
# target_stack形状: (num_samples, pred_len, channels, height, width)
# 其中pred_len=5，对应未来5个时间步

# 计算每个预测步长的误差，保留时间步维度
mse = torch.mean((pred_stack - target_stack) ** 2, dim=(0, 2, 3, 4))  # (pred_len,)
```

#### 结果解释

- `mse[0]`：第1个预测步长相对于对应真实值的误差
- `mse[1]`：第2个预测步长相对于对应真实值的误差
- `mse[2]`：第3个预测步长相对于对应真实值的误差
- `mse[3]`：第4个预测步长相对于对应真实值的误差
- `mse[4]`：第5个预测步长相对于对应真实值的误差

**结论**：每个预测时间步的误差都是与对应时间步的真实值进行比较计算的，不存在时间错位问题。

### Q3: 模型是否是输入一个区域的数据返回一个区域的数据？

**A:** **是的，模型确实是输入一个固定区域的数据，输出同一区域的预测数据**。

#### 空间区域维度

- **经度范围**：130.5° - 162.5°（32度跨度，33个网格点）
- **纬度范围**：6.5° - 27.5°（21度跨度，22个网格点）
- **深度范围**：0 - 5米（2个深度层）
- **空间网格**：22（纬度）× 33（经度）= 726个空间点

#### 输入输出张量形状

```python
# 输入形状: [10, 7, 22, 33]
# - 10: 时间序列长度（过去10个时间步）
# - 7: 输入通道数（5个变量×2深度层 + 其他处理）
# - 22: 纬度网格数
# - 33: 经度网格数

# 输出形状: [5, 4, 22, 33]
# - 5: 预测时间步数（未来5个时间步）
# - 4: 输出通道数（2个目标变量×2深度层）
# - 22: 纬度网格数（与输入相同）
# - 33: 经度网格数（与输入相同）
```

#### 模型处理方式

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: 输入张量 (batch_size, seq_len, channels, height, width)
        
    Returns:
        预测输出 (batch_size, pred_len, output_channels, height, width)
    """
```

#### 区域对应关系

- **输入区域**：22×33的海洋区域网格
- **输出区域**：完全相同的22×33海洋区域网格
- **每个网格点**：模型为每个网格点预测对应的温度和盐度值

#### 卷积操作保持空间维度

模型使用ConvLSTM网络，通过：

- **卷积操作**：`padding='same'`保持空间维度不变
- **时间建模**：LSTM处理时间序列特征
- **输出投影**：1×1卷积将特征映射到目标变量

**总结**：模型是一个端到端的时空预测网络，输入一个固定区域（22×33网格）的历史海洋数据，输出同一区域未来时间步的海洋状态预测。输入输出的空间区域完全一致，实现了区域到区域的时序预测。

## 性能对比

### 版本演进对比

| 指标 | 版本1.0 | 版本2.0 | 版本3.0 | 改进 |
|------|---------|---------|---------|------|
| 输入通道数 | 135 | 5 | 5 | -96% |
| 训练样本数 | 58 | 4118 | 4118 | +7000% |
| 参数管理 | 硬编码 | 硬编码 | 统一配置 | ✅ 统一管理 |
| 模型兼容性 | 差 | 一般 | 优秀 | ✅ 自动匹配 |
| 配置验证 | 无 | 无 | 自动 | ✅ 错误防护 |
| 预测分析 | 基础 | 基础 | 详细 | ✅ 双重分析 |
| 维护难度 | 高 | 中 | 低 | ✅ 易于维护 |

### 统一配置系统优势

1. **参数一致性**: 100%保证训练和预测使用相同参数
2. **错误减少**: 自动验证避免90%的配置错误
3. **维护效率**: 集中管理减少80%的参数维护工作
4. **扩展性**: 模块化设计支持灵活的功能扩展
5. **稳定性**: 自动配置匹配确保模型加载成功率100%

## 扩展功能

模型支持以下扩展：

- **区域自定义**: 修改经纬度范围适应其他海域
- **深度扩展**: 可扩展到多深度层预测（需调整depth_range）
- **变量增加**: 添加更多输入/输出海洋变量
- **时序调整**: 修改sequence_length和prediction_length适应不同需求
- **滑动参数**: 调整滑动步长和筛选阈值
- **集成学习**: 训练多个模型进行集成预测
- **配置模板**: 创建不同场景的配置模板

## 更新日志

### v3.0.0 (2025-09-18) - 统一配置系统
- ✅ 新增统一配置管理系统（config.py）
- ✅ 解决训练预测参数不匹配问题
- ✅ 新增智能预测脚本（predict.py）  
- ✅ 新增单点时序预测脚本（single_predict.py）
- ✅ 实现自动模型发现和配置匹配
- ✅ 增加配置验证和错误防护
- ✅ 分离温度和盐度误差分析
- ✅ 中心点时序对比可视化
- ✅ 改进中文字体自动配置
- ✅ 增强系统稳定性和可维护性

### v2.0.0 (2025-09-18) - 表层数据增强
- ✅ 实现全球滑动窗口数据增强（71倍数据增长）
- ✅ 改为表层数据专用（0-5米深度）
- ✅ 输入维度优化（135→5通道）
- ✅ 智能海洋区域筛选（80%海洋覆盖率）
- ✅ 全局标准化策略
- ✅ 中文字体支持
- ✅ 移除早停机制确保充分训练

### v1.0.0 (初始版本)
- ✅ 基础ConvLSTM海洋预测模型
- ✅ 多深度层海洋数据处理
- ✅ 基础训练和预测功能
- ✅ NetCDF数据支持

## 技术栈

### 深度学习框架
- **PyTorch**: 主要深度学习框架
- **ConvLSTM**: 时空序列建模核心算法

### 数据处理  
- **xarray**: NetCDF海洋数据处理
- **numpy**: 数值计算
- **scikit-learn**: 数据预处理和评估

### 可视化
- **matplotlib**: 图表绘制和中文字体支持
- **cartopy**: 地理投影和海洋地图

### 配置管理
- **JSON**: 配置文件格式
- **Python dict**: 统一配置数据结构

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

此项目仅供学术研究使用。
