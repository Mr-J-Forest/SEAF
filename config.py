#!/usr/bin/env python3
"""
统一配置文件
定义所有训练和预测相关的参数
"""

import math

# ========== 模块开关集中管理 ==========
MODULE_SWITCHES = {
    'enable_positional_encoding': True,   # 空间位置编码
    'enable_time_encoding': True,         # 时间傅里叶编码
    'enable_climatology_anomaly': True,   # 输入变量使用训练期月气候态 anomaly
    'enable_target_climatology_anomaly': True,  # 目标变量使用 anomaly；可独立消融
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
    # 2. 序列要求：每个样本需要 sequence_length + prediction_length 个连续时间步。
    #    carry_history 下验证/测试可使用分界前已观测历史，但预测目标不能越界。
    # 3. 空间维度：每个时间段包含所有空间位置的数据
    # 4. Train/validation/test 使用同一 canonical 空间网格。
    'train_ratio': 0.6,    # 训练集比例：前60%时间段
    'val_ratio': 0.2,      # 验证集比例：中间20%时间段
    'test_ratio': 0.2,     # 测试集比例：最后20%时间段
    # 验证/测试的预测目标严格位于各自分段；历史输入可承接分界前已观测数据。
    # 这对应滚动起报，避免无意义地丢弃每个分段开头 sequence_length 个月。
    'split_context_policy': 'carry_history',

    # 空间范围
    'lon_range': [130.5, 162.5],  # 经度范围
    'lat_range': [6.5, 27.5],     # 纬度范围
    'depth_range': [0, 1000.0],      # 深度范围（米）
    
    # 数据增强 — 2D 滑动窗口采集100%海洋区域
    'sliding_enabled': True,       # 是否启用2D滑动窗口（经纬度双向滑动）
    'ocean_threshold': 1.0,        # 海洋面积占比阈值（1.0 = 仅保留100%海洋窗口）
    # None 使用表层判断；设置为深度（米）时要求该深度层也满足海洋覆盖率。
    # ORAS5 0-1000 m 实验用 1000，避免把海底以下的填充值当作可预测海水。
    'ocean_coverage_depth': None,
    'lon_step': 2.0,              # 经纬度滑动步长（兜底默认值，优先使用分模式步长）

    # 分模式滑动步长（支持经纬度分别设置）
    # Train/val/test: 同一个 8° canonical 网格
    'train_stride_lon': 8.0,
    'train_stride_lat': 8.0,
    # Val/Test 使用与训练相同的 canonical 网格，保证评价空间采样一致。
    'val_stride_lon': 8.0,
    'val_stride_lat': 8.0,
    'test_stride_lon': 8.0,
    'test_stride_lat': 8.0,
    # 对当前冻结数据文件和 terminal-anchor 网格的审计结果；加载器据此检测协议漂移。
    'expected_canonical_windows_per_origin': 151,

    # 推理阶段密集重叠滑窗步长（overlap-tile prediction）
    'inference_stride_lon': 4.0,
    'inference_stride_lat': 4.0,

    # Overlap-tile 融合参数
    'taper_ratio': 0.25,           # 边缘余弦衰减带占窗口尺寸比例（四周各25%）
    'min_blend_weight': 1e-3,      # 融合时边缘最低权重，避免除零/空洞
    'inference_micro_batch_size': 32,

    # 气候态 / anomaly 建模
    'anomaly_variables': ['TEMP', 'SALT'],       # 对这些变量减去训练期月气候态后再标准化
    'target_anomaly_variables': ['TEMP', 'SALT'],
    'climatology_period': 12,                    # 月气候态周期
    'include_climatology_features': True,        # 将目标变量气候态作为额外输入通道
    'climatology_feature_variables': ['TEMP', 'SALT'],
    'climatology_baseline_variables': ['TEMP', 'SALT'],
    # 显式物理量后向差分只使用当前及过去观测，并使用独立训练期 scaler。
    'include_tendency_features': False,
    'tendency_feature_variables': ['TEMP', 'SALT'],
    # SEAF 正式输入中的外部动力与表面强迫变量；数据集配置覆盖实际列表。
    'external_dynamic_variables': ['SSHA', 'UWND', 'VWND'],

    # 预处理持久化缓存（避免每次训练重复滑窗搜索、气候态计算、标准化拟合）
    'cache_preprocessed': True,                  # 是否启用预处理缓存
    'cache_preprocessed_dir': '.cache/preprocessed',  # 缓存存放路径
}

# ========== 模型相关参数 ==========
MODEL_CONFIG = {
    'model_type': 'seaf',
    'model_display_name': 'SEAF',
    'hidden_dims': [64, 96, 128, 128],  # 隐藏层维度
    'kernel_size': (3, 3),              # 卷积核大小
    'num_layers': 4,                    # 层数
    'dropout': 0.05,                     # Dropout；最终值由验证集协议确定

    'seaf_hidden_dim': 64,
    'seaf_spectral_modes': [8, 8],
    'seaf_spectral_layers': 2,
    'seaf_ensemble_members': 4,
    # 模块化 SEAF 结构开关。默认值保持已归档模型的前向与 state_dict 兼容；
    # 新结构只通过显式实验配置逐项启用。
    'use_temporal_depth_mixer': False,
    'use_local_path': False,
    'use_spectral_path': True,
    'router_type': 'spatial',  # spatial | uniform | lead
    'use_forcing_encoder': False,
    'seaf_spectral_scale_init': 0.1,
    'seaf_forcing_scale_init': 0.1,
    'ablation_disable_spectral': False,
    'ablation_disable_ensemble': False,
    'ablation_uniform_ensemble': False,
    'ablation_direct_full_field': False,

    # DynaSEAF transport--innovation decomposition.  These keys are inert for
    # model_type='seaf' and therefore do not alter the frozen SEAF path.
    'dynaseaf_future_dynamics_variables': ['UVEL', 'VVEL', 'SSHA', 'MLD'],
    'dynaseaf_lambda_dynamics': 0.10,
    'dynaseaf_max_deformation_cells': 1.0,
    'dynaseaf_gate_resolution': 'target',  # target | depth
    'dynaseaf_gate_initial_bias': -1.7346,  # sigmoid ~= 0.15
    'dynaseaf_zero_init_innovation': True,
    'dynaseaf_use_future_dynamics_aux': True,
    'dynaseaf_use_transport': True,
    'dynaseaf_use_innovation': True,
    'dynaseaf_use_adaptive_gate': True,
    'dynaseaf_use_temporal_depth_mixer': False,
}

# ========== 训练相关参数 ==========
TRAINING_CONFIG = {
    'epochs': 50,                 # 训练轮数
    'learning_rate': 1.57e-4,         # 默认值；正式运行先执行分模型学习率校准
    'batch_size': 151,             # 当前冻结数据协议每个起报时次的全部 canonical 空间窗
    'seed': 42,                    # 默认固定随机种子；正式实验使用多个 seed 重复
    # 一个 canonical time-group batch 很大；多 worker × 预取会复制整批张量，
    # 在容器内存限额下没有收益且可能被外部 OOM guard 杀死。
    'num_workers': 0,
    'persistent_workers': False,
    'prefetch_factor': 2,          # 仅 num_workers>0 时生效
    'pin_memory': True,            # 是否使用pin_memory
    'group_batches_by_time': True, # 将同一起报时次的空间窗口组织到同一批次
    
    # 优化器参数
    'optimizer_type': 'adam',      # 可选 adam / adamw；基线按公开协议显式覆盖
    'optimizer_betas': [0.9, 0.999],
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
    'target_loss_weights': {'TEMP': 0.5, 'SALT': 0.5},  # 通用分变量损失权重
    'use_gradient_loss': True,     # 是否启用梯度分布匹配损失
    'gradient_loss_weight': 0.1,   # 梯度损失权重
    'gradient_loss_mode': 'vector', # vector 保留梯度方向；magnitude 仅用于敏感性对照
    
    # 保存设置
    'save_best_only': True,        # 只保存最佳模型
    'save_last': True,             # 保存最后一个检查点
    'strict_resume_provenance': True,  # 禁止跨配置或跨源码版本拼接训练轨迹
    # none=仅训练曲线，validation=消融筛选，test=协议冻结后的最终确认。
    # 默认只读验证集，避免手工运行 train.py 时无意中反复查看测试集。
    # 只有冻结后的 final_test 阶段才能在实验矩阵中显式覆盖为 test。
    'post_training_evaluation': 'validation',
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
    'mixed_precision_dtype': 'auto',  # AMP 精度：'auto'(bf16优先,不支持回退fp16)|'bfloat16'|'float16'
    'nonfinite_grad_skip_limit': 30,  # 连续非有限梯度跳过上限；损失/输出有限但梯度溢出时按 AMP 语义跳步，超过该上限判定为真实发散
    'compile_model': False,        # 实测动态形状 compile 更慢且会重复编译，正式协议使用 eager
    'cudnn_benchmark': False,      # 固定 seed 的正式实验关闭 autotuner，减少运行间漂移
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
    'scalers_filename': 'scalers.pkl',  # 与模型一起保存训练期标准化器
    'info_filename': 'info.json',
    
    # 日志设置
    'log_interval': 50,            # 训练日志输出间隔
    'val_log_interval': 1,         # 验证日志输出间隔
}

# ========== 特征编码参数 ==========
ENCODING_CONFIG = {
    'enable_positional_encoding': MODULE_SWITCHES['enable_positional_encoding'],  # 是否启用周期经纬度 Fourier 编码
    'positional_encoding_frequencies': 8, # 经纬度编码频率数量（总通道数 = 4 * 频率数）
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
    train_ratio = float(config.get('train_ratio', 0))
    val_ratio = float(config.get('val_ratio', 0))
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        errors.append("train_ratio 必须为正、val_ratio 必须非负，且二者之和必须小于1.0")
    if config.get('split_context_policy', 'carry_history') not in {'carry_history', 'strict_segment'}:
        errors.append("split_context_policy 必须为 carry_history 或 strict_segment")
    
    # 验证序列长度
    if config['sequence_length'] <= 0 or config['prediction_length'] <= 0:
        errors.append("序列长度必须大于0")

    if config.get('enable_climatology_anomaly', False):
        if int(config.get('climatology_period', 0)) <= 0:
            errors.append("climatology_period 必须为正整数")
        anomaly_variables = config.get('anomaly_variables', [])
        if not isinstance(anomaly_variables, (list, tuple)) or len(anomaly_variables) == 0:
            errors.append("启用 enable_climatology_anomaly 时 anomaly_variables 不能为空")
        elif not set(anomaly_variables).issubset(set(config.get('input_variables', []))):
            errors.append("anomaly_variables 必须是 input_variables 的子集")

    if config.get(
        'enable_target_climatology_anomaly',
        config.get('enable_climatology_anomaly', False),
    ):
        target_anomaly_variables = config.get(
            'target_anomaly_variables', config.get('target_variables', [])
        )
        if not isinstance(target_anomaly_variables, (list, tuple)) or not target_anomaly_variables:
            errors.append(
                "启用 enable_target_climatology_anomaly 时 "
                "target_anomaly_variables 不能为空"
            )
        elif not set(target_anomaly_variables).issubset(
            set(config.get('target_variables', []))
        ):
            errors.append("target_anomaly_variables 必须是 target_variables 的子集")

    if config.get('include_climatology_features', False):
        feature_variables = config.get('climatology_feature_variables', [])
        if not isinstance(feature_variables, (list, tuple)) or len(feature_variables) == 0:
            errors.append("启用 include_climatology_features 时 climatology_feature_variables 不能为空")
        elif not set(feature_variables).issubset(set(config.get('input_variables', []))):
            errors.append("climatology_feature_variables 必须是 input_variables 的子集")

    if config.get('include_tendency_features', False):
        tendency_variables = config.get('tendency_feature_variables', [])
        if not isinstance(tendency_variables, (list, tuple)) or not tendency_variables:
            errors.append("启用 include_tendency_features 时 tendency_feature_variables 不能为空")
        elif not set(tendency_variables).issubset(set(config.get('input_variables', []))):
            errors.append("tendency_feature_variables 必须是 input_variables 的子集")

    external_variables = config.get('external_dynamic_variables', [])
    if not isinstance(external_variables, (list, tuple)):
        errors.append("external_dynamic_variables 必须是列表")
    elif not set(external_variables).issubset(set(config.get('input_variables', []))):
        errors.append("external_dynamic_variables 必须是 input_variables 的子集")

    ocean_threshold = float(config.get('ocean_threshold', 1.0))
    if not 0.0 <= ocean_threshold <= 1.0:
        errors.append("ocean_threshold 必须位于[0, 1]")
    ocean_coverage_depth = config.get('ocean_coverage_depth')
    if ocean_coverage_depth is not None and float(ocean_coverage_depth) < 0:
        errors.append("ocean_coverage_depth 必须为非负深度或 None")
    for key in (
        'train_stride_lon', 'train_stride_lat', 'val_stride_lon', 'val_stride_lat',
        'test_stride_lon', 'test_stride_lat', 'inference_stride_lon', 'inference_stride_lat'
    ):
        if float(config.get(key, 0)) <= 0:
            errors.append(f"{key} 必须为正")
    
    # 验证模型参数
    if len(config['hidden_dims']) != config['num_layers']:
        errors.append("hidden_dims长度必须等于num_layers")
    
    # 验证学习率
    if config['learning_rate'] <= 0 or config['learning_rate'] >= 1:
        errors.append("学习率必须在(0,1)范围内")
    
    # 验证批次大小
    if config['batch_size'] <= 0:
        errors.append("批次大小必须大于0")
    expected_windows = config.get('expected_canonical_windows_per_origin')
    if expected_windows is not None:
        if isinstance(expected_windows, dict):
            required_splits = {'train', 'validation', 'test'}
            if set(expected_windows) != required_splits:
                errors.append(
                    "expected_canonical_windows_per_origin 映射必须恰好包含 "
                    "train、validation、test"
                )
            elif any(int(value) <= 0 for value in expected_windows.values()):
                errors.append("expected_canonical_windows_per_origin 的 split 值必须为正整数")
        elif int(expected_windows) <= 0:
            errors.append("expected_canonical_windows_per_origin 必须为正整数或 split 映射")
    if int(config.get('epochs', 0)) <= 0:
        errors.append("epochs 必须为正整数")
    expected_parameter_count = config.get('expected_parameter_count')
    if expected_parameter_count is not None and int(expected_parameter_count) <= 0:
        errors.append("expected_parameter_count 必须为正整数或 None")
    if int(config.get('early_stopping_patience', 0)) <= 0:
        errors.append("early_stopping_patience 必须为正整数")
    if float(config.get('grad_clip_norm', 0)) <= 0:
        errors.append("grad_clip_norm 必须为正")
    if config.get('gradient_loss_mode', 'vector') not in {'vector', 'magnitude'}:
        errors.append("gradient_loss_mode 必须为 vector 或 magnitude")
    if config.get('post_training_evaluation', 'validation') not in {
        'none', 'validation', 'test'
    }:
        errors.append("post_training_evaluation 必须为 none、validation 或 test")

    target_variables = list(config.get('target_variables', []))
    target_weights = config.get('target_loss_weights', {})
    if not target_variables:
        errors.append("target_variables 不能为空")
    if not isinstance(target_weights, dict):
        errors.append("target_loss_weights 必须为字典")
    else:
        missing_weights = [name for name in target_variables if name not in target_weights]
        if missing_weights:
            errors.append(f"target_loss_weights 缺少变量: {missing_weights}")
        elif any(float(target_weights[name]) < 0 for name in target_variables):
            errors.append("target_loss_weights 不能为负")
        elif sum(float(target_weights[name]) for name in target_variables) <= 0:
            errors.append("target_loss_weights 总和必须为正")

    normalized_model_type = str(config.get('model_type', '')).lower()
    seaf_model_types = {'seaf'}
    dynaseaf_model_types = {'dynaseaf'}
    recent_baseline_types = {
        'ofb_fourcastnet', 'ofb-fourcastnet',
        'ofb_climax', 'ofb-climax',
        'ofb_swin', 'ofb-swin',
    }
    supported_model_types = {
        *seaf_model_types,
        *dynaseaf_model_types,
        *recent_baseline_types,
        'tianhai_paper', 'tianhai-reimpl',
        'fuxi_ocean_paper', 'fuxi-ocean-reimpl',
        'fuxi_ons_paper', 'fuxi-ons-reimpl',
        'axiomocean_paper', 'axiom-ocean-reimpl',
    }
    if normalized_model_type not in supported_model_types:
        errors.append(f"未知 model_type: {config.get('model_type')!r}")

    if normalized_model_type in seaf_model_types:
        input_anomaly = bool(config.get('enable_climatology_anomaly', False))
        target_anomaly = bool(config.get(
            'enable_target_climatology_anomaly', input_anomaly
        ))
        direct_full_field = bool(config.get('ablation_direct_full_field', False))
        if not input_anomaly:
            errors.append("SEAF 正式协议要求输入使用 climatology anomaly")
        elif not set(config.get('target_variables', [])).issubset(
            set(config.get('anomaly_variables', []))
        ):
            errors.append("SEAF 要求所有 target_variables 都属于输入 anomaly_variables")
        if direct_full_field == target_anomaly:
            errors.append(
                "ablation_direct_full_field 必须与 "
                "enable_target_climatology_anomaly 互斥"
            )
        if target_anomaly and not set(config.get('target_variables', [])).issubset(
            set(config.get('target_anomaly_variables', []))
        ):
            errors.append("SEAF anomaly target 必须覆盖所有 target_variables")
        if int(config.get('seaf_hidden_dim', 0)) <= 0:
            errors.append("seaf_hidden_dim 必须为正")
        spectral_modes = config.get('seaf_spectral_modes', [8, 8])
        if (
            not isinstance(spectral_modes, (list, tuple))
            or len(spectral_modes) != 2
            or any(int(value) <= 0 for value in spectral_modes)
        ):
            errors.append("seaf_spectral_modes 必须包含两个正整数")
        if int(config.get('seaf_spectral_layers', 0)) <= 0:
            errors.append("seaf_spectral_layers 必须为正")
        if int(config.get('seaf_ensemble_members', 0)) <= 0:
            errors.append("seaf_ensemble_members 必须为正")
        router_type = str(config.get('router_type', 'spatial')).lower()
        if router_type not in {'spatial', 'uniform', 'lead'}:
            errors.append("router_type 必须为 spatial、uniform 或 lead")
        if (
            router_type == 'lead'
            and not config.get('ablation_disable_ensemble', False)
            and int(config.get('seaf_ensemble_members', 0)) < 2
        ):
            errors.append("lead router 至少需要两个 ensemble members")
        if (
            config.get('ablation_uniform_ensemble', False)
            and router_type == 'lead'
        ):
            errors.append("lead router 与 uniform-ensemble 消融不能同时启用")
        for key in ('seaf_spectral_scale_init', 'seaf_forcing_scale_init'):
            value = float(config.get(key, 0.1))
            if not 0.0 <= value <= 1.0:
                errors.append(f"{key} 必须位于[0, 1]")
        if (
            config.get('use_forcing_encoder', False)
            and not config.get('external_dynamic_variables', [])
        ):
            errors.append(
                "启用 use_forcing_encoder 时 external_dynamic_variables 不能为空"
            )
        if (
            config.get('ablation_disable_ensemble', False)
            and config.get('ablation_uniform_ensemble', False)
        ):
            errors.append("single-head 与 uniform-ensemble 消融不能同时启用")
        if (
            config.get('ablation_uniform_ensemble', False)
            and int(config.get('seaf_ensemble_members', 0)) < 2
        ):
            errors.append("uniform-ensemble 消融至少需要两个 member")

    if normalized_model_type in dynaseaf_model_types:
        if config.get('use_gradient_loss', False):
            errors.append(
                "DynaSEAF 第一阶段必须关闭 gradient loss；该实验只比较主任务与 dynamics auxiliary loss"
            )
        input_anomaly = bool(config.get('enable_climatology_anomaly', False))
        target_anomaly = bool(config.get(
            'enable_target_climatology_anomaly', input_anomaly
        ))
        if not input_anomaly:
            errors.append("DynaSEAF 正式协议要求输入使用 climatology anomaly")
        if not target_anomaly:
            errors.append("DynaSEAF 正式协议要求 TEMP/SALT target 使用 anomaly")
        if not set(config.get('target_variables', [])).issubset(
            set(config.get('anomaly_variables', []))
        ):
            errors.append("DynaSEAF 要求所有 target_variables 都属于输入 anomaly_variables")
        if not set(config.get('target_variables', [])).issubset(
            set(config.get('target_anomaly_variables', []))
        ):
            errors.append("DynaSEAF anomaly target 必须覆盖所有 target_variables")

        dynamics_variables = config.get(
            'dynaseaf_future_dynamics_variables',
            ['UVEL', 'VVEL', 'SSHA', 'MLD'],
        )
        if (
            not isinstance(dynamics_variables, (list, tuple))
            or not dynamics_variables
            or not all(isinstance(name, str) and name for name in dynamics_variables)
            or len(set(dynamics_variables)) != len(dynamics_variables)
        ):
            errors.append(
                "dynaseaf_future_dynamics_variables 必须是非空且不重复的字符串列表"
            )
        try:
            lambda_dynamics = float(config.get('dynaseaf_lambda_dynamics', 0.1))
            max_deformation = float(
                config.get('dynaseaf_max_deformation_cells', 1.0)
            )
            gate_bias = float(config.get('dynaseaf_gate_initial_bias', -1.7346))
            if not math.isfinite(lambda_dynamics) or lambda_dynamics < 0:
                errors.append("dynaseaf_lambda_dynamics 必须为非负有限数")
            if not math.isfinite(max_deformation) or max_deformation < 0:
                errors.append("dynaseaf_max_deformation_cells 必须为非负有限数")
            if not math.isfinite(gate_bias) or not -20.0 <= gate_bias <= 20.0:
                errors.append("dynaseaf_gate_initial_bias 必须位于[-20, 20]")
        except (TypeError, ValueError):
            errors.append(
                "dynaseaf_lambda_dynamics、dynaseaf_max_deformation_cells、"
                "dynaseaf_gate_initial_bias 必须为数值"
            )
        if str(config.get('dynaseaf_gate_resolution', 'target')).lower() not in {
            'target', 'channel', 'target_channel', 'depth', 'shared_depth'
        }:
            errors.append("dynaseaf_gate_resolution 必须为 target 或 depth")
        dyna_boolean_defaults = {
            'dynaseaf_zero_init_innovation': True,
            'dynaseaf_use_future_dynamics_aux': True,
            'dynaseaf_use_transport': True,
            'dynaseaf_use_innovation': True,
            'dynaseaf_use_adaptive_gate': True,
            'dynaseaf_use_temporal_depth_mixer': False,
        }
        for key, default in dyna_boolean_defaults.items():
            if not isinstance(config.get(key, default), bool):
                errors.append(f"{key} 必须为布尔值")

    if normalized_model_type in recent_baseline_types:
        if not config.get('enable_climatology_anomaly', False):
            errors.append("OceanForecastBench adapters 直接异常场预测要求启用 anomaly")
        elif not config.get(
            'enable_target_climatology_anomaly',
            config.get('enable_climatology_anomaly', False),
        ):
            errors.append("OceanForecastBench adapters 的 target 必须是 anomaly")
        elif not set(config.get('target_variables', [])).issubset(
            set(config.get('target_anomaly_variables', []))
        ):
            errors.append(
                "OceanForecastBench adapters 要求 targets 属于 "
                "target_anomaly_variables"
            )

    for forbidden_key in (
        'enable_persistence_residual',
        'enable_persistence_projection',
        'learned_persistence_scale',
        'ap_scale',
    ):
        if config.get(forbidden_key, False):
            errors.append(f"正式模型禁止启用 {forbidden_key}")

    if normalized_model_type in recent_baseline_types:
        patch_size = int(config.get('baseline_patch_size', 0))
        embed_dim = int(config.get('baseline_embed_dim', 0))
        depth = int(config.get('baseline_depth', 0))
        if patch_size <= 0:
            errors.append("baseline_patch_size 必须为正")
        if embed_dim <= 0 or embed_dim % 4 != 0:
            errors.append("baseline_embed_dim 必须为正且能被4整除")
        if normalized_model_type not in {'ofb_swin', 'ofb-swin'} and depth <= 0:
            errors.append("baseline_depth 必须为正")
        if float(config.get('baseline_mlp_ratio', 0)) <= 0:
            errors.append("baseline_mlp_ratio 必须为正")
        for key in ('baseline_drop_rate', 'baseline_attention_dropout',
                    'baseline_drop_path_rate'):
            value = float(config.get(key, 0.0))
            if not 0.0 <= value < 1.0:
                errors.append(f"{key} 必须位于[0, 1)")
        if normalized_model_type in {'ofb_fourcastnet', 'ofb-fourcastnet'}:
            blocks = int(config.get('afno_num_blocks', 0))
            fraction = float(config.get('afno_hard_thresholding_fraction', 0.0))
            if blocks <= 0 or embed_dim % max(1, blocks) != 0:
                errors.append("afno_num_blocks 必须为正且整除 baseline_embed_dim")
            if not 0.0 < fraction <= 1.0:
                errors.append("afno_hard_thresholding_fraction 必须位于(0, 1]")
        elif normalized_model_type in {'ofb_climax', 'ofb-climax'}:
            heads = int(config.get('baseline_num_heads', 0))
            if heads <= 0 or embed_dim % max(1, heads) != 0:
                errors.append("baseline_num_heads 必须为正且整除 baseline_embed_dim")
        else:
            depths = config.get('swin_depths', [])
            heads = config.get('swin_num_heads', [])
            if (
                not isinstance(depths, (list, tuple))
                or len(depths) != 3
                or any(int(value) <= 0 for value in depths)
            ):
                errors.append("swin_depths 必须包含3个正整数")
            if (
                not isinstance(heads, (list, tuple))
                or len(heads) != 3
                or any(int(value) <= 0 for value in heads)
            ):
                errors.append("swin_num_heads 必须包含3个正整数")
            elif isinstance(depths, (list, tuple)) and len(depths) == 3 and (
                embed_dim % int(heads[0]) != 0
                or (2 * embed_dim) % int(heads[1]) != 0
                or embed_dim % int(heads[2]) != 0
            ):
                errors.append("Swin 各 stage 通道数必须能被对应 heads 整除")
            if int(config.get('swin_window_size', 0)) <= 1:
                errors.append("swin_window_size 必须大于1")

    if str(config.get('optimizer_type', 'adam')).lower() not in {'adam', 'adamw'}:
        errors.append("optimizer_type 必须为 adam 或 adamw")
    optimizer_betas = config.get('optimizer_betas', [0.9, 0.999])
    if (
        not isinstance(optimizer_betas, (list, tuple))
        or len(optimizer_betas) != 2
        or not all(0.0 <= float(value) < 1.0 for value in optimizer_betas)
    ):
        errors.append("optimizer_betas 必须包含两个位于[0, 1)的数")

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

def load_config(filepath, _stack=None):
    """加载 JSON 配置，并解析相对路径 ``extends`` 继承链。"""
    path = os.path.abspath(os.fspath(filepath))
    stack = tuple(_stack or ())
    if path in stack:
        cycle = " -> ".join((*stack, path))
        raise ValueError(f"配置 extends 出现循环: {cycle}")

    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置顶层必须是对象: {path}")

    parent_refs = config.pop('extends', None)
    if parent_refs is None:
        return config
    if isinstance(parent_refs, str):
        parent_refs = [parent_refs]
    if not isinstance(parent_refs, list) or not parent_refs or not all(
        isinstance(item, str) and item.strip() for item in parent_refs
    ):
        raise ValueError(f"extends 必须是非空路径字符串或路径列表: {path}")

    merged = {}
    for parent_ref in parent_refs:
        parent_path = parent_ref
        if not os.path.isabs(parent_path):
            parent_path = os.path.join(os.path.dirname(path), parent_path)
        merged.update(load_config(parent_path, _stack=(*stack, path)))
    merged.update(config)
    return merged

def merge_configs(file_config, default_config=None):
    """合并文件配置和默认配置"""
    if default_config is None:
        default_config = DEFAULT_CONFIG
    
    merged = default_config.copy()
    merged.update(file_config)
    if 'enable_target_climatology_anomaly' not in file_config:
        merged['enable_target_climatology_anomaly'] = bool(
            file_config.get(
                'enable_climatology_anomaly',
                merged.get('enable_climatology_anomaly', False),
            )
        )
    if 'target_anomaly_variables' not in file_config:
        merged['target_anomaly_variables'] = list(
            file_config.get('target_variables', merged.get('target_variables', []))
        )
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
