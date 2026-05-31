#!/usr/bin/env python3
"""
统一配置文件
定义所有训练和预测相关的参数
"""

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
    'depth_range': [0, 5.0],      # 深度范围（米）
    
    # 数据增强
    'sliding_enabled': True,       # 是否启用滑动窗口数据增强
    'ocean_threshold': 0.8,        # 海洋面积占比阈值
    'lon_step': 2.0,              # 经度滑动步长
}

# ========== 模型相关参数 ==========
MODEL_CONFIG = {
    'hidden_dims': [64, 64, 64],   # 隐藏层维度
    'kernel_size': (3, 3),         # 卷积核大小
    'num_layers': 3,               # 层数
    'dropout': 0.4,                # Dropout率
}

# ========== 训练相关参数 ==========
TRAINING_CONFIG = {
    'epochs': 100,                 # 训练轮数
    'learning_rate': 5e-4,         # 学习率
    'batch_size': 8,               # 批次大小
    'num_workers': 4,              # 数据加载进程数
    'pin_memory': True,            # 是否使用pin_memory
    
    # 优化器参数
    'weight_decay': 1e-4,          # 权重衰减
    'grad_clip_norm': 1.0,         # 梯度裁剪
    
    # 学习率调度
    'scheduler_patience': 10,       # 学习率调度耐心值
    'scheduler_factor': 0.5,        # 学习率衰减因子
    'min_lr': 1e-6,                # 最小学习率
    
    # 早停
    'early_stopping_patience': 20,  # 早停耐心值
    'min_delta': 1e-6,             # 最小改善阈值
    
    # 损失函数权重
    'temp_weight': 0.7,            # 温度损失权重
    'salt_weight': 0.3,            # 盐度损失权重
    
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

# ========== 合并所有配置 ==========
def get_unified_config():
    """获取统一的配置字典"""
    config = {}
    config.update(DATA_CONFIG)
    config.update(MODEL_CONFIG)
    config.update(TRAINING_CONFIG)
    config.update(PREDICTION_CONFIG)
    config.update(HARDWARE_CONFIG)
    config.update(OUTPUT_CONFIG)
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
    
    # 验证模型参数
    if len(config['hidden_dims']) != config['num_layers']:
        errors.append("hidden_dims长度必须等于num_layers")
    
    # 验证学习率
    if config['learning_rate'] <= 0 or config['learning_rate'] >= 1:
        errors.append("学习率必须在(0,1)范围内")
    
    # 验证批次大小
    if config['batch_size'] <= 0:
        errors.append("批次大小必须大于0")
    
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