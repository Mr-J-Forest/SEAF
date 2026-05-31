#!/usr/bin/env python3
"""
统一配置文件
定义所有训练和预测相关的参数
"""

# ========== 模块开关集中管理 ==========
MODULE_SWITCHES = {
    'enable_positional_encoding': True,   # 空间位置编码
    'enable_time_encoding': True,         # 时间傅里叶编码
    'enable_climatology_anomaly': True,   # 使用训练期月气候态构造 anomaly 学习目标
}

# ========== 数据相关参数 ==========
DATA_CONFIG = {
    'data_path': 'Data/FullData_preprocessed.nc',
    'input_variables': ['TEMP', 'SALT', 'SSHA', 'UWND', 'VWND'],
    'target_variables': ['TEMP', 'SALT'],
    'sequence_length': 12,
    'prediction_length': 5,
    # 数据分割比例 - 为了确保验证集有足够数据，调整分割比例
    # 注意：数据分割是基于时间序列进行的，不是随机打乱
    # 分割机制：
    # 1. 时间维度分割：按时间顺序将数据切分为三个连续段
    #    - 训练集：时间步 0 到 train_ratio*total_steps
    #    - 验证集：时间步 train_ratio*total_steps 到 (train_ratio+val_ratio)*total_steps  
    #    - 测试集：时间步 (train_ratio+val_ratio)*total_steps 到 total_steps
    # 2. 序列要求：每个样本需要 sequence_length + prediction_length 个连续时间步
    #    - 当前需要：24 + 6 = 30 个连续时间步
    #    - 如果某个数据集的时间步少于30，将无法创建样本
    # 3. 空间维度：每个时间段包含所有空间位置的数据
    # 4. 数据增强：训练时启用滑动窗口增强，测试时只用原始区域
    'train_ratio': 0.6,    # 训练集比例：前50%时间段
    'val_ratio': 0.2,      # 验证集比例：中间30%时间段
    'test_ratio': 0.2,     # 测试集比例：最后20%时间段

    # 空间范围
    'lon_range': [130.5, 162.5],  # 经度范围
    'lat_range': [6.5, 27.5],     # 纬度范围
    'depth_range': [0, 1000.0],      # 深度范围（米）
    
    # 数据增强 — 2D 滑动窗口采集100%海洋区域
    'sliding_enabled': True,       # 是否启用2D滑动窗口（经纬度双向滑动）
    'ocean_threshold': 1.0,        # 海洋面积占比阈值（1.0 = 仅保留100%海洋窗口）
    'lon_step': 2.0,              # 经纬度滑动步长（兜底默认值，优先使用分模式步长）

    # 分模式滑动步长（支持经纬度分别设置）
    # Train: 密集滑窗
    'train_stride_lon': 8.0,
    'train_stride_lat': 8.0,
    # Val: 稀疏重叠
    'val_stride_lon': 16.0,
    'val_stride_lat': 10.0,
    # Test: 稀疏重叠
    'test_stride_lon': 16.0,
    'test_stride_lat': 10.0,

    # 推理阶段密集重叠滑窗步长（overlap-tile prediction）
    'inference_stride_lon': 4.0,
    'inference_stride_lat': 4.0,

    # Overlap-tile 融合参数
    'taper_ratio': 0.25,           # 边缘余弦衰减带占窗口尺寸比例（四周各25%）
    'min_blend_weight': 1e-3,      # 融合时边缘最低权重，避免除零/空洞

    # 气候态 / anomaly 建模
    'anomaly_variables': ['TEMP', 'SALT'],       # 对这些变量减去训练期月气候态后再标准化
    'climatology_period': 12,                    # 月气候态周期
    'include_climatology_features': True,        # 将目标变量气候态作为额外输入通道
    'climatology_feature_variables': ['TEMP', 'SALT'],

    # 预处理持久化缓存（避免每次训练重复滑窗搜索、气候态计算、标准化拟合）
    'cache_preprocessed': True,                  # 是否启用预处理缓存
    'cache_preprocessed_dir': '.cache/preprocessed',  # 缓存存放路径
}

# ========== 模型相关参数 ==========
MODEL_CONFIG = {
    'model_type': 'tsc_fusion',           # 模型类型: 'convlstm' 或 'tsc_fusion'
    'hidden_dims': [64, 96, 128, 128],  # 隐藏层维度
    'kernel_size': (3, 3),              # 卷积核大小
    'num_layers': 4,                    # 层数
    'dropout': 0.05,                     # Dropout率，温度喜欢0.1，盐度喜欢0.05 (过拟合时可适当调大)

    'tsc_variables': ['TEMP', 'SALT', 'PTEMP', 'PDEN', 'SPICE'],
    'tsc_num_prototypes': 8,
    'tsc_hidden_dim': 32,
    'tsc_output_dim': 16,
    'tsc_attention_heads': 4,
    'tsc_memory_layers': 1,
    'tsc_ffn_dim': 64,
    'tsc_fusion_hidden_dim': 64,
    'tsc_fusion_spectral_modes': [8, 8],
    'tsc_fusion_spectral_layers': 2,
    'tsc_fusion_3d_layers': 2,
    'tsc_fusion_ensemble_members': 4,
    'tsc_fusion_transformer_heads': 8,
    'tsc_fusion_transformer_layers': 1,
    'tsc_fusion_transformer_ffn_dim': 256,
    'tsc_fusion_persistence_init': 0.5,
    'enable_global_token_bank': True,
    'global_token_bank_heads': 4,
    'global_token_bank_ffn_dim': 128,
    'global_token_bank_dropout': 0.05,
}

# ========== 训练相关参数 ==========
TRAINING_CONFIG = {
    'epochs': 20,                 # 训练轮数（快速验证稳定性，可按需调大）
    'learning_rate': 1.57e-4,         # 学习率 温度喜欢1.57e-4
    'batch_size': 8,               # 批次大小（验证期减小以加快迭代）
    'num_workers': 4,              # 数据加载进程数 (设置为0以避免验证集卡死)
    'persistent_workers': True,    # 多进程数据加载是否持久化worker
    'prefetch_factor': 4,          # 每个worker预取的批次数（num_workers>0时生效）
    'pin_memory': True,            # 是否使用pin_memory
    'group_batches_by_time': True, # 将同一历史起点的不同窗口组织到同一batch，供Global Token Bank使用
    
    # 优化器参数
    'weight_decay': 1e-4,          # 权重衰减
    'grad_clip_norm': 1.0,         # 梯度裁剪
    
    # 学习率调度
    'scheduler_patience': 10,       # 学习率调度耐心值
    'scheduler_factor': 0.5,        # 学习率衰减因子
    'min_lr': 1e-6,                # 最小学习率
    # 温度收敛后进一步压低学习率以优化盐度
    'temp_lr_threshold': 0.05,      # 当温度损失低于该值时触发额外降学习率
    'temp_lr_decay_factor': 0.5,    # 额外降学习率倍数
    'temp_lr_cooldown': 1,          # 两次触发之间的最小epoch间隔
    'temp_lr_min': 1e-6,            # 额外策略的学习率下限
    
    # 早停
    'early_stopping_patience': 20,  # 早停耐心值
    'min_delta': 1e-6,             # 最小改善阈值
    
    # 损失函数权重
    'temp_weight': 0.5,            # 温度损失权重
    'salt_weight': 0.5,            # 盐度损失权重
    'use_gradient_loss': True,     # 是否启用梯度分布匹配损失
    'gradient_loss_weight': 0.1,   # 梯度损失权重
    
    # 保存设置
    'save_best_only': True,        # 只保存最佳模型
    'save_last': True,             # 保存最后一个检查点
}

# ========== 预测相关参数 ==========
PREDICTION_CONFIG = {
    'num_samples': 10,             # 预测样本数量
    'visualization_samples': 3,    # 可视化样本数量
    'dpi': 150,                    # 图像分辨率
    'figsize_single': (16, 10),    # 单变量图像大小
    'figsize_combined': (12, 10),  # 组合图像大小
    'figsize_error': (14, 6),      # 误差分析图像大小
}

# ========== 硬件和性能参数 ==========
HARDWARE_CONFIG = {
    'device': 'auto',              # 设备选择：'auto', 'cpu', 'cuda'
    'mixed_precision': True,       # 启用混合精度训练(AMP)，提升RTX 5090性能
    'compile_model': True,         # 启用torch.compile加速
    'cudnn_benchmark': True,       # 启用cuDNN benchmark
}

# ========== 输出和日志参数 ==========
OUTPUT_CONFIG = {
    'output_base_dir': 'outputs',
    'results_dir': 'outputs/results',
    'predictions_dir': 'outputs/predictions',
    'logs_dir': 'logs',
    
    # 文件命名
    'timestamp_format': '%Y%m%d_%H%M%S',
    'model_filename': 'best_model.pth',
    'checkpoint_filename': 'latest_checkpoint.pth',
    'config_filename': 'config.json',
    'metrics_filename': 'metrics.json',
    'info_filename': 'info.json',
    
    # 日志设置
    'log_interval': 50,            # 训练日志输出间隔
    'val_log_interval': 1,         # 验证日志输出间隔
}

# ========== 特征编码参数 ==========
ENCODING_CONFIG = {
    'enable_positional_encoding': MODULE_SWITCHES['enable_positional_encoding'],  # 是否启用经纬度/深度正弦-余弦位置编码
    'positional_encoding_frequencies': 8, # 经纬度编码频率数量（总通道数 = 4 * 频率数）
    'depth_encoding_frequencies': 4,      # 深度编码频率数量（总通道数 = 2 * 深度层数 * 频率数）
    'enable_time_encoding': MODULE_SWITCHES['enable_time_encoding'],        # 是否启用时间傅里叶编码
    'time_encoding_frequencies': 4,       # 时间编码频率数量（总通道数 = 2 * 频率数）
    'time_encoding_period': 12,           # 时间周期（默认12表示月份）
    'include_year_trend': True,           # 是否加入年份趋势特征
}

# ========== 合并所有配置 ==========
def get_unified_config():
    """获取统一的配置字典"""
    config = {}
    config.update(MODULE_SWITCHES)
    config.update(DATA_CONFIG)
    config.update(MODEL_CONFIG)
    config.update(TRAINING_CONFIG)
    config.update(PREDICTION_CONFIG)
    config.update(HARDWARE_CONFIG)
    config.update(OUTPUT_CONFIG)
    config.update(ENCODING_CONFIG)
    return config

# 默认配置
DEFAULT_CONFIG = get_unified_config()

# ========== 配置验证函数 ==========
def validate_config(config):
    """验证配置参数的合理性"""
    errors = []
    
    # 验证数据分割比例
    if config['train_ratio'] + config['val_ratio'] > 1.0:
        errors.append("训练集和验证集比例之和不能超过1.0")
    
    # 验证序列长度
    if config['sequence_length'] <= 0 or config['prediction_length'] <= 0:
        errors.append("序列长度必须大于0")

    if config.get('enable_climatology_anomaly', False):
        if int(config.get('climatology_period', 0)) <= 0:
            errors.append("climatology_period 必须为正整数")
        anomaly_variables = config.get('anomaly_variables', [])
        if not isinstance(anomaly_variables, (list, tuple)) or len(anomaly_variables) == 0:
            errors.append("启用 enable_climatology_anomaly 时 anomaly_variables 不能为空")
    
    # 验证模型参数
    if len(config['hidden_dims']) != config['num_layers']:
        errors.append("hidden_dims长度必须等于num_layers")
    
    # 验证学习率
    if config['learning_rate'] <= 0 or config['learning_rate'] >= 1:
        errors.append("学习率必须在(0,1)范围内")
    
    # 验证批次大小
    if config['batch_size'] <= 0:
        errors.append("批次大小必须大于0")

    if config.get('temp_lr_threshold', 0) < 0:
        errors.append("temp_lr_threshold 应为非负")
    if config.get('temp_lr_decay_factor', 1) <= 0:
        errors.append("temp_lr_decay_factor 应为正")
    if config.get('temp_lr_cooldown', 0) < 0:
        errors.append("temp_lr_cooldown 应为非负整数")
    if str(config.get('model_type', '')).lower() in {
        'tsc_fusion',
        'tscglobal',
        'tsc_global_axiom_ensemble',
        'tsc-spectrum-axiom-ensemble',
        'tsc_spectrum_axiom_ensemble'
    }:
        if config.get('tsc_fusion_hidden_dim', 0) <= 0:
            errors.append("tsc_fusion_hidden_dim 必须为正")
        spectral_modes = config.get('tsc_fusion_spectral_modes', [8, 8])
        if len(spectral_modes) != 2 or spectral_modes[0] <= 0 or spectral_modes[1] <= 0:
            errors.append("tsc_fusion_spectral_modes 必须包含两个正整数")
        if config.get('tsc_fusion_ensemble_members', 0) <= 0:
            errors.append("tsc_fusion_ensemble_members 必须为正")
        if config.get('tsc_fusion_transformer_layers', 0) > 0:
            hidden = config.get('tsc_fusion_hidden_dim', 0)
            heads = config.get('tsc_fusion_transformer_heads', 1)
            if heads <= 0:
                errors.append("tsc_fusion_transformer_heads 必须为正")
            elif hidden % heads != 0:
                errors.append("tsc_fusion_hidden_dim 必须能被 tsc_fusion_transformer_heads 整除")
        if config.get('enable_global_token_bank', False):
            hidden = config.get('tsc_fusion_hidden_dim', 0)
            heads = config.get('global_token_bank_heads', 1)
            if heads <= 0:
                errors.append("global_token_bank_heads 必须为正")
            elif hidden % heads != 0:
                errors.append("tsc_fusion_hidden_dim 必须能被 global_token_bank_heads 整除")
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

# ========== 配置更新函数 ==========
def update_config(base_config, **kwargs):
    """更新配置参数"""
    config = base_config.copy()
    config.update(kwargs)
    validate_config(config)
    return config

# ========== 配置保存和加载函数 ==========
import json
import os

def save_config(config, filepath):
    """保存配置到文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def load_config(filepath):
    """从文件加载配置"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_configs(file_config, default_config=None):
    """合并文件配置和默认配置"""
    if default_config is None:
        default_config = DEFAULT_CONFIG
    
    merged = default_config.copy()
    merged.update(file_config)
    return merged

if __name__ == "__main__":
    # 测试配置
    config = get_unified_config()
    print("统一配置参数:")
    for section in ['DATA', 'MODEL', 'TRAINING', 'PREDICTION', 'HARDWARE', 'OUTPUT']:
        print(f"\n{section} CONFIG:")
        section_config = globals()[f'{section}_CONFIG']
        for key, value in section_config.items():
            print(f"  {key}: {value}")
    
    # 验证配置
    try:
        validate_config(config)
        print("\n✓ 配置验证通过")
    except ValueError as e:
        print(f"\n✗ 配置验证失败: {e}")
