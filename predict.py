#!/usr/bin/env python3
"""
ConvLSTM海洋预测脚本 - 统一配置版本
自动匹配模型结构和权重，生成可视化预测结果
使用统一配置文件确保与训练脚本参数一致
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import json
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Sequence
import warnings
warnings.filterwarnings('ignore')

# 导入统一配置
from config import DEFAULT_CONFIG, load_config, merge_configs, save_config
from convlstm_model import create_ocean_model
from data_loader import OceanDataset
from font_config import setup_chinese_fonts
from metrics_utils import compute_metric_report

# 初始化中文字体
setup_chinese_fonts()


def generate_cosine_weights(height: int, width: int, taper_ratio: float = 0.25,
                            min_weight: float = 1e-3) -> np.ndarray:
    """
    生成2D余弦衰减权重，用于 overlap-tile 融合。

    窗口中心区域权重=1，四周 taper_ratio 比例区域做余弦衰减。
    W_2d = W_h × W_w（外积），边缘最低权重 clamp 到 min_weight。

    Args:
        height: 窗口高度（纬度方向格点数）
        width: 窗口宽度（经度方向格点数）
        taper_ratio: 衰减带占窗口尺寸的比例，默认0.25
        min_weight: 边缘最低权重，避免除零/空洞

    Returns:
        (height, width) float32 权重数组
    """
    taper_h = max(1, int(height * taper_ratio))
    taper_w = max(1, int(width * taper_ratio))

    def _cosine_taper_1d(size: int, taper: int) -> np.ndarray:
        w = np.ones(size, dtype=np.float32)
        for i in range(taper):
            cos_val = 0.5 * (1.0 - np.cos(np.pi * i / max(taper - 1, 1)))
            val = min_weight + (1.0 - min_weight) * cos_val
            w[i] = val
            w[-(i + 1)] = val
        return w

    h_weights = _cosine_taper_1d(height, taper_h)
    w_weights = _cosine_taper_1d(width, taper_w)
    return np.outer(h_weights, w_weights).astype(np.float32)


def build_validity_mask(data_2d: np.ndarray) -> np.ndarray:
    """
    根据数据构建有效性 mask：NaN → weight=0，有效 → weight=1。

    用于融合时将陆地/缺失区域排除在加权平均之外。

    Args:
        data_2d: (H, W) 二维数据数组

    Returns:
        (H, W) float32 mask 数组
    """
    mask = (~np.isnan(data_2d)).astype(np.float32)
    return mask


class SmartOceanPredictor:
    """智能海洋预测器 - 使用统一配置并按编号加载模型"""
    
    def __init__(self,
                 model_index: int,
                 config: Optional[Dict] = None,
                 output_dir: Optional[str] = None,
                 selected_variables: Optional[Sequence[str]] = None):
        """
        初始化预测器
        
        Args:
            model_index: 结果目录编号（outputs/results/<index>_*）
            config: 配置字典，如果为None则使用默认配置
            output_dir: 输出目录，默认自动生成
        """
        # 使用统一配置
        if config is None:
            self.config = DEFAULT_CONFIG.copy()
        else:
            self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_index = model_index
        
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime(self.config['timestamp_format'])
            self.output_dir = os.path.join(
                self.config['predictions_dir'],
                f"predictions_model{model_index}_{timestamp}"
            )
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model_dir, self.model_path = self._resolve_model_by_index(model_index)
        print(f"使用模型目录: {self.model_dir}")
        print(f"使用权重: {self.model_path}")
        
        # 加载模型
        self._load_model()

        self.target_variables = self.config.get('target_variables', [])
        self.selected_variables = list(selected_variables) if selected_variables else None
        if self.selected_variables:
            # 保留与模型配置交集，保持原始顺序
            filtered = [var for var in self.target_variables if var in self.selected_variables]
            if not filtered:
                print(f"警告: 选择的变量 {self.selected_variables} 不在模型目标变量 {self.target_variables} 中，采用模型默认变量")
                self.active_target_variables = list(self.target_variables)
            else:
                self.active_target_variables = filtered
        else:
            self.active_target_variables = list(self.target_variables)

        self.target_channel_slices: Dict[str, slice] = self._slice_map_from_config('target_channel_slices')
        
        print(f"预测结果将保存到: {self.output_dir}")

    def _slice_map_from_config(self, key: str) -> Dict[str, slice]:
        raw_map = self.config.get(key, {})
        slice_map: Dict[str, slice] = {}
        if not isinstance(raw_map, dict):
            return slice_map
        for name, bounds in raw_map.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                slice_map[name] = slice(int(bounds[0]), int(bounds[1]))
        return slice_map

    def _resolve_model_by_index(self, model_index: int) -> Tuple[str, str]:
        base_dir = self.config['results_dir']
        dirs = sorted([d for d in os.listdir(base_dir)
                       if os.path.isdir(os.path.join(base_dir, d)) and d.split('_')[0].isdigit()])
        candidates = [d for d in dirs if int(d.split('_')[0]) == model_index]
        if not candidates:
            raise ValueError(f"未找到编号 {model_index} 的模型目录")
        result_dir = os.path.join(base_dir, candidates[0])
        config_path = os.path.join(result_dir, self.config['config_filename'])
        model_path = os.path.join(result_dir, self.config['model_filename'])
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        model_config = load_config(config_path)
        merged_config = merge_configs(model_config, self.config)
        if 'enable_climatology_anomaly' not in model_config:
            # Older checkpoints were trained without anomaly/climatology channels.
            merged_config['enable_climatology_anomaly'] = False
            merged_config['include_climatology_features'] = False
        self.config = merged_config
        return result_dir, model_path
    
    def _load_model(self):
        """加载模型"""
        print("加载模型...")
        
        # 创建模型
        self.model = create_ocean_model(self.config).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"模型训练轮数: {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"最佳验证损失: {checkpoint['best_val_loss']:.6f}")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("模型加载成功!")

    def _extract_variable_channels(self, tensor: torch.Tensor, variables: Sequence[str]) -> torch.Tensor:
        """根据选定变量提取通道"""
        if not variables or not self.target_channel_slices:
            return tensor
        parts = []
        for var in variables:
            sl = self.target_channel_slices.get(var)
            if sl is None:
                continue
            parts.append(tensor[:, sl, :, :])
        if not parts:
            return tensor
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=1)
    
    def predict(self, num_samples: Optional[int] = None):
        """
        进行预测
        
        Args:
            num_samples: 预测样本数量，如果为None则使用配置中的默认值
        """
        if num_samples is None:
            num_samples = self.config['num_samples']
            
        print(f"开始预测 {num_samples} 个样本...")
        
        # 创建测试数据集（包含所有100%海洋窗口）
        test_dataset = OceanDataset(self.config['data_path'], self.config, mode='test')
        dataset_slices = getattr(test_dataset, 'target_channel_slices', {}) or {}
        if dataset_slices:
            self.target_channel_slices = dataset_slices
        
        if len(test_dataset) == 0:
            print("测试集为空，无法进行预测")
            return
        
        # 限制样本数量
        num_samples = min(num_samples, len(test_dataset))
        
        predictions = []
        targets = []
        inputs_list = []
        sample_indices = []
        
        print(f"数据集大小: {len(test_dataset)}")
        print(f"实际预测样本数: {num_samples}")

        with torch.no_grad():
            for i in range(num_samples):
                sample = test_dataset[i]
                if isinstance(sample, (list, tuple)) and len(sample) == 3:
                    inputs, target, _ = sample
                else:
                    inputs, target = sample
                inputs = inputs.unsqueeze(0).to(self.device)  # 添加batch维度

                # 模型预测
                output = self.model(inputs)

                # 检查输出是否有效
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"警告: 样本 {i} 的预测结果包含NaN或Inf，跳过")
                    continue

                output_cpu = output.cpu()

                # 转换回CPU并保存
                predictions.append(output_cpu.squeeze(0))
                targets.append(target)
                inputs_list.append(inputs.cpu().squeeze(0))
                sample_indices.append(i)
                
                if (i + 1) % 5 == 0:
                    print(f"已完成 {i + 1}/{num_samples} 个样本的预测")
        
        if len(predictions) == 0:
            print("所有预测结果都包含异常值，无法生成有效预测")
            return
        
        print(f"成功预测 {len(predictions)} 个样本")
        
        # 反标准化预测结果和目标
        predictions_original, targets_original = self._denormalize_predictions(
            predictions,
            targets,
            test_dataset,
            sample_indices=sample_indices,
        )
        
        # 计算评估指标（使用反标准化后的数据）
        metrics = self._compute_metrics(
            predictions_original,
            targets_original,
            normalized_predictions=predictions,
            normalized_targets=targets,
            dataset=test_dataset,
            sample_indices=sample_indices,
        )
        
        # 保存和可视化结果
        self._save_results(predictions_original, targets_original, inputs_list, test_dataset)
        self._visualize_results(predictions_original, targets_original, inputs_list, metrics, test_dataset)
        
        print(f"预测完成！结果保存在: {self.output_dir}")

    def predict_full_map(self, target_lon_range=None, target_lat_range=None,
                         base_time_index=None, pred_step: int = 0):
        """
        全图 overlap-tile 推理：用密集滑窗覆盖目标区域，逐窗口预测后用余弦权重融合。

        Args:
            target_lon_range: [min_lon, max_lon]，默认使用数据完整经度范围
            target_lat_range: [min_lat, max_lat]，默认使用数据完整纬度范围
            base_time_index: 输入序列结束的时间索引（在 test 时间段内），默认取 test 段最后一个可用位置
            pred_step: 取第几个预测步的输出（0-indexed）

        Returns:
            dict: {
                'blended_pred': (channels, H, W) 融合后的预测,
                'blended_target': (channels, H, W) 融合后的真实值,
                'weight_sum': (H, W) 每个格点的权重和,
                'lons': 经度坐标,
                'lats': 纬度坐标
            }
        """
        return self._predict_full_map_with_dataset(
            target_lon_range=target_lon_range,
            target_lat_range=target_lat_range,
            base_time_index=base_time_index,
            pred_step=pred_step,
        )

        taper_ratio = self.config.get('taper_ratio', 0.25)
        min_weight = self.config.get('min_blend_weight', 1e-3)
        stride_lon = self.config.get('inference_stride_lon', 4.0)
        stride_lat = self.config.get('inference_stride_lat', 4.0)
        seq_len = self.config['sequence_length']

        # 1. 获取 scalers（用 train 模式快速获取）
        print("获取标准化参数...")
        ref_dataset = OceanDataset(self.config['data_path'], self.config, mode='train')
        scalers = ref_dataset.scalers
        input_vars = self.config['input_variables']
        target_vars = self.config['target_variables']
        available_vars = ref_dataset.available_variables
        ref_dataset.dataset.close()

        # 2. 打开数据
        ds = xr.open_dataset(self.config['data_path'])
        ds = ds.sel(LEVEL=slice(self.config['depth_range'][0], self.config['depth_range'][1]))

        all_lons = ds.LONGITUDE.values.astype(np.float64)
        all_lats = ds.LATITUDE.values.astype(np.float64)
        times = ds.TIME.values

        # 验证输入变量
        valid_input_vars = [v for v in input_vars if v in available_vars and v in ds.data_vars]

        # 3. 确定目标范围
        if target_lon_range is None:
            target_lon_range = [float(all_lons.min()), float(all_lons.max())]
        if target_lat_range is None:
            target_lat_range = [float(all_lats.min()), float(all_lats.max())]

        # 窗口尺寸（度）
        win_lon = self.config['lon_range'][1] - self.config['lon_range'][0]
        win_lat = self.config['lat_range'][1] - self.config['lat_range'][0]

        # 4. 确定时间索引
        total_t = len(times)
        train_end = int(total_t * self.config.get('train_ratio', 0.6))
        val_end = int(total_t * (self.config.get('train_ratio', 0.6) + self.config.get('val_ratio', 0.2)))
        test_start = val_end

        if base_time_index is None:
            base_time_index = min(test_start + seq_len, total_t - self.config['prediction_length'] - 1)
        base_time_index = max(test_start + seq_len - 1, min(base_time_index, total_t - self.config['prediction_length'] - 1))

        print(f"全图推理: 输入序列结束于时间索引 {base_time_index}")
        print(f"  目标范围: 经度 [{target_lon_range[0]:.1f}, {target_lon_range[1]:.1f}], "
              f"纬度 [{target_lat_range[0]:.1f}, {target_lat_range[1]:.1f}]")
        print(f"  推理步长: 经度={stride_lon}°, 纬度={stride_lat}°")
        print(f"  余弦衰减: taper_ratio={taper_ratio}, min_weight={min_weight}")

        # 5. 生成窗口位置网格
        lon_starts = np.arange(target_lon_range[0], target_lon_range[1] - win_lon + stride_lon * 0.5, stride_lon)
        lat_starts = np.arange(target_lat_range[0], target_lat_range[1] - win_lat + stride_lat * 0.5, stride_lat)

        if len(lon_starts) == 0:
            lon_starts = np.array([target_lon_range[0]])
        if len(lat_starts) == 0:
            lat_starts = np.array([target_lat_range[0]])

        print(f"  窗口网格: {len(lon_starts)} (经度) × {len(lat_starts)} (纬度) = {len(lon_starts) * len(lat_starts)} 个窗口")

        # 6. 确定全图格点范围
        lon_res = float(all_lons[1] - all_lons[0]) if len(all_lons) > 1 else 1.0
        lat_res = float(all_lats[1] - all_lats[0]) if len(all_lats) > 1 else 1.0

        full_lon_start_idx = max(0, int(np.searchsorted(all_lons, target_lon_range[0])))
        full_lon_end_idx = min(len(all_lons), int(np.searchsorted(all_lons, target_lon_range[1])) + 1)
        full_lat_start_idx = max(0, int(np.searchsorted(all_lats, target_lat_range[0])))
        full_lat_end_idx = min(len(all_lats), int(np.searchsorted(all_lats, target_lat_range[1])) + 1)

        full_h = full_lat_end_idx - full_lat_start_idx
        full_w = full_lon_end_idx - full_lon_start_idx
        full_lons = all_lons[full_lon_start_idx:full_lon_end_idx]
        full_lats = all_lats[full_lat_start_idx:full_lat_end_idx]

        # 窗口格点数（从第一个窗口推算）
        first_win_lon_end = target_lon_range[0] + win_lon
        first_win_lat_end = target_lat_range[0] + win_lat
        win_w = int(np.searchsorted(all_lons, first_win_lon_end) - np.searchsorted(all_lons, target_lon_range[0]))
        win_h = int(np.searchsorted(all_lats, first_win_lat_end) - np.searchsorted(all_lats, target_lat_range[0]))

        # 输出通道数（从模型配置获取）
        # 用第一个预测样本的形状推断
        num_target_vars = len(target_vars)

        # 7. 余弦权重模板
        cos_weights = generate_cosine_weights(win_h, win_w, taper_ratio, min_weight)  # (win_h, win_w)

        # 8. 累积器
        # 输出通道数 = 深度层数 × 目标变量数，用第一个窗口的实际输出确定
        total_out_channels = None
        blended_pred = None
        blended_target = None
        weight_sum = np.zeros((full_h, full_w), dtype=np.float64)

        # 9. 逐窗口预测并融合
        self.model.eval()
        valid_windows = 0

        with torch.no_grad():
            for lat_start in lat_starts:
                for lon_start in lon_starts:
                    lon_end = lon_start + win_lon
                    lat_end = lat_start + win_lat

                    # 切片数据
                    try:
                        win_ds = ds.sel(
                            LONGITUDE=slice(lon_start, lon_end),
                            LATITUDE=slice(lat_start, lat_end)
                        )
                    except Exception:
                        continue

                    # 检查数据有效性
                    if 'TEMP' not in win_ds.data_vars:
                        continue
                    temp_check = win_ds.TEMP.isel(TIME=0, LEVEL=0).values
                    if temp_check.size == 0 or np.all(np.isnan(temp_check)):
                        continue

                    # 提取输入序列
                    input_seq_data = {}
                    for var in valid_input_vars:
                        if var in win_ds.data_vars:
                            data = win_ds[var].values
                            input_seq_data[var] = data[base_time_index - seq_len + 1:base_time_index + 1]

                    # 提取目标序列
                    target_seq_data = {}
                    for var in target_vars:
                        if var in win_ds.data_vars:
                            data = win_ds[var].values
                            t_start = base_time_index + 1
                            t_end = t_start + self.config['prediction_length']
                            target_seq_data[var] = data[t_start:t_end]

                    if not input_seq_data:
                        continue

                    # 填充缺失值 + 标准化
                    for var in input_seq_data:
                        if np.isnan(input_seq_data[var]).any():
                            input_seq_data[var] = self._fill_nan(input_seq_data[var])
                    for var in target_seq_data:
                        if np.isnan(target_seq_data[var]).any():
                            target_seq_data[var] = self._fill_nan(target_seq_data[var])

                    # 构建输入张量 (seq_len, lat, lon, channels)
                    input_arrays = []
                    for var in valid_input_vars:
                        if var in input_seq_data:
                            v = input_seq_data[var]
                            if v.ndim == 4:  # (seq, level, lat, lon) → (seq, lat, lon, level)
                                v = v.transpose(0, 2, 3, 1)
                            elif v.ndim == 3:
                                v = v[:, :, :, np.newaxis]
                            else:
                                continue
                            # 标准化
                            if var in scalers:
                                orig_shape = v.shape
                                v_2d = v.reshape(-1, 1)
                                v_2d = scalers[var].transform(v_2d)
                                v = v_2d.reshape(orig_shape)
                            input_arrays.append(v.astype(np.float32))

                    if not input_arrays:
                        continue

                    input_tensor = torch.from_numpy(
                        np.concatenate(input_arrays, axis=-1)
                    ).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)  # (1, seq, C, H, W)

                    # 构建目标张量
                    target_arrays = []
                    for var in target_vars:
                        if var in target_seq_data:
                            v = target_seq_data[var]
                            if v.ndim == 4:
                                v = v.transpose(0, 2, 3, 1)
                            elif v.ndim == 3:
                                v = v[:, :, :, np.newaxis]
                            else:
                                continue
                            if var in scalers:
                                orig_shape = v.shape
                                v_2d = v.reshape(-1, 1)
                                v_2d = scalers[var].transform(v_2d)
                                v = v_2d.reshape(orig_shape)
                            target_arrays.append(v.astype(np.float32))

                    target_tensor = None
                    if target_arrays:
                        target_tensor = torch.from_numpy(
                            np.concatenate(target_arrays, axis=-1)
                        ).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)

                    # 模型预测
                    output = self.model(input_tensor)  # (1, pred_len, C, H, W)
                    pred = output[0, pred_step].cpu().numpy()  # (C, H, W)
                    target = target_tensor[0, pred_step].cpu().numpy() if target_tensor is not None else None

                    if total_out_channels is None:
                        total_out_channels = pred.shape[0]
                        blended_pred = np.zeros((total_out_channels, full_h, full_w), dtype=np.float64)
                        blended_target = np.zeros((total_out_channels, full_h, full_w), dtype=np.float64)

                    # 反标准化
                    for var_idx, var_name in enumerate(target_vars):
                        if var_name in scalers:
                            ch_slice = self.target_channel_slices.get(var_name)
                            if ch_slice is not None:
                                ch_start, ch_stop = ch_slice.start, ch_slice.stop
                                if ch_start is not None and ch_stop is not None:
                                    var_pred = pred[ch_start:ch_stop]
                                    var_shape = var_pred.shape
                                    var_2d = var_pred.reshape(-1, 1)
                                    var_2d = scalers[var_name].inverse_transform(var_2d)
                                    pred[ch_start:ch_stop] = var_2d.reshape(var_shape)
                                    if target is not None:
                                        var_target = target[ch_start:ch_stop]
                                        var_2d_t = var_target.reshape(-1, 1)
                                        var_2d_t = scalers[var_name].inverse_transform(var_2d_t)
                                        target[ch_start:ch_stop] = var_2d_t.reshape(var_shape)

                    # 确定在全图中的位置
                    lon_idx_start = int(np.searchsorted(all_lons, lon_start)) - full_lon_start_idx
                    lat_idx_start = int(np.searchsorted(all_lats, lat_start)) - full_lat_start_idx
                    lon_idx_end = lon_idx_start + win_w
                    lat_idx_end = lat_idx_start + win_h

                    # Clip to full map bounds
                    p_h_start = 0
                    p_w_start = 0
                    p_h_end = win_h
                    p_w_end = win_w

                    if lat_idx_start < 0:
                        p_h_start = -lat_idx_start
                        lat_idx_start = 0
                    if lon_idx_start < 0:
                        p_w_start = -lon_idx_start
                        lon_idx_start = 0
                    if lat_idx_end > full_h:
                        p_h_end = win_h - (lat_idx_end - full_h)
                        lat_idx_end = full_h
                    if lon_idx_end > full_w:
                        p_w_end = win_w - (lon_idx_end - full_w)
                        lon_idx_end = full_w

                    if lat_idx_end <= lat_idx_start or lon_idx_end <= lon_idx_start:
                        continue

                    # 应用余弦权重并累积
                    weights_slice = cos_weights[p_h_start:p_h_end, p_w_start:p_w_end]  # (h_slice, w_slice)
                    weights_3d = weights_slice[np.newaxis, :, :]  # (1, h, w)

                    pred_slice = pred[:, p_h_start:p_h_end, p_w_start:p_w_end]  # (C, h, w)

                    blended_pred[:, lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end] += pred_slice * weights_3d
                    if target is not None:
                        target_slice = target[:, p_h_start:p_h_end, p_w_start:p_w_end]
                        blended_target[:, lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end] += target_slice * weights_3d

                    weight_sum[lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end] += weights_slice
                    valid_windows += 1

        ds.close()

        if valid_windows == 0:
            raise RuntimeError("没有有效窗口可用于全图推理")

        # 10. 归一化
        weight_sum_3d = np.maximum(weight_sum, min_weight)[np.newaxis, :, :]
        blended_pred /= weight_sum_3d
        blended_target /= weight_sum_3d

        print(f"全图推理完成: {valid_windows} 个有效窗口融合")
        print(f"  输出尺寸: {blended_pred.shape}")

        return {
            'blended_pred': blended_pred.astype(np.float32),
            'blended_target': blended_target.astype(np.float32),
            'weight_sum': weight_sum.astype(np.float32),
            'lons': full_lons,
            'lats': full_lats
        }

    def _predict_full_map_with_dataset(self, target_lon_range=None, target_lat_range=None,
                                       base_time_index=None, pred_step: int = 0):
        """使用 OceanDataset 生成密集窗口，确保训练和全图推理的输入通道一致。"""
        taper_ratio = self.config.get('taper_ratio', 0.25)
        min_weight = self.config.get('min_blend_weight', 1e-3)
        stride_lon = self.config.get('inference_stride_lon', 4.0)
        stride_lat = self.config.get('inference_stride_lat', 4.0)
        seq_len = self.config['sequence_length']
        pred_len = self.config['prediction_length']

        print("获取训练期标准化参数...")
        ref_dataset = OceanDataset(self.config['data_path'], self.config, mode='train')
        scalers = ref_dataset.scalers

        print("构建密集推理窗口数据集...")
        dense_dataset = OceanDataset(
            self.config['data_path'],
            self.config,
            mode='test',
            scalers=scalers,
            override_stride_lon=stride_lon,
            override_stride_lat=stride_lat,
        )

        try:
            all_lons = dense_dataset.lons.astype(np.float64)
            all_lats = dense_dataset.lats.astype(np.float64)
            times = dense_dataset.times

            if target_lon_range is None:
                target_lon_range = [float(all_lons.min()), float(all_lons.max())]
            if target_lat_range is None:
                target_lat_range = [float(all_lats.min()), float(all_lats.max())]

            total_t = len(times)
            val_end = int(total_t * (
                self.config.get('train_ratio', 0.6) + self.config.get('val_ratio', 0.2)
            ))
            test_start = val_end

            if base_time_index is None:
                base_time_index = min(test_start + seq_len, total_t - pred_len - 1)
            base_time_index = max(test_start + seq_len - 1, min(base_time_index, total_t - pred_len - 1))
            sequence_start = base_time_index - seq_len + 1
            pred_step = max(0, min(int(pred_step), pred_len - 1))

            lon_indices = np.where((all_lons >= target_lon_range[0]) & (all_lons <= target_lon_range[1]))[0]
            lat_indices = np.where((all_lats >= target_lat_range[0]) & (all_lats <= target_lat_range[1]))[0]
            if len(lon_indices) == 0 or len(lat_indices) == 0:
                raise ValueError("目标经纬度范围没有覆盖任何网格点")

            full_lon_start_idx = int(lon_indices[0])
            full_lon_end_idx = int(lon_indices[-1]) + 1
            full_lat_start_idx = int(lat_indices[0])
            full_lat_end_idx = int(lat_indices[-1]) + 1
            full_lons = all_lons[full_lon_start_idx:full_lon_end_idx]
            full_lats = all_lats[full_lat_start_idx:full_lat_end_idx]
            full_h = len(full_lats)
            full_w = len(full_lons)

            candidate_indices = []
            for sample_idx, (start_idx, region_idx) in enumerate(dense_dataset.sequences):
                if start_idx != sequence_start:
                    continue
                region = dense_dataset.all_regions_data[region_idx]
                lon_range = region['lon_range']
                lat_range = region['lat_range']
                if lon_range[1] < target_lon_range[0] or lon_range[0] > target_lon_range[1]:
                    continue
                if lat_range[1] < target_lat_range[0] or lat_range[0] > target_lat_range[1]:
                    continue
                candidate_indices.append(sample_idx)

            if not candidate_indices:
                raise RuntimeError("没有匹配目标时间和范围的推理窗口")

            print(f"全图推理: 输入序列结束于时间索引 {base_time_index}")
            print(f"  目标范围: 经度 [{target_lon_range[0]:.1f}, {target_lon_range[1]:.1f}], "
                  f"纬度 [{target_lat_range[0]:.1f}, {target_lat_range[1]:.1f}]")
            print(f"  推理步长: 经度={stride_lon}°, 纬度={stride_lat}°")
            print(f"  候选窗口: {len(candidate_indices)}")

            blended_pred = None
            blended_target = None
            weight_sum = np.zeros((full_h, full_w), dtype=np.float64)
            valid_windows = 0
            inference_batch_size = max(1, int(self.config.get('batch_size', 8)))

            self.model.eval()
            with torch.no_grad():
                for offset in range(0, len(candidate_indices), inference_batch_size):
                    batch_indices = candidate_indices[offset:offset + inference_batch_size]
                    batch_inputs = []
                    batch_targets = []
                    for sample_idx in batch_indices:
                        inputs, target_tensor = dense_dataset[sample_idx]
                        batch_inputs.append(inputs)
                        batch_targets.append(target_tensor)

                    inputs_tensor = torch.stack(batch_inputs, dim=0).to(self.device)
                    outputs = self.model(inputs_tensor)
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"警告: batch {offset // inference_batch_size} 输出包含 NaN/Inf，逐窗口过滤")

                    pred_phys_batch = dense_dataset.inverse_transform_targets(
                        outputs.cpu().numpy(),
                        sample_indices=batch_indices,
                    )[:, pred_step]
                    target_phys_batch = dense_dataset.inverse_transform_targets(
                        torch.stack(batch_targets, dim=0).numpy(),
                        sample_indices=batch_indices,
                    )[:, pred_step]

                    for local_idx, sample_idx in enumerate(batch_indices):
                        pred_phys = pred_phys_batch[local_idx]
                        target_phys = target_phys_batch[local_idx]
                        if not np.isfinite(pred_phys).all():
                            print(f"警告: 窗口 {sample_idx} 输出包含 NaN/Inf，已跳过")
                            continue

                        _, region_idx = dense_dataset.sequences[sample_idx]
                        region = dense_dataset.all_regions_data[region_idx]
                        coords = region['coords']
                        window_lons = coords['lons'].astype(np.float64)
                        window_lats = coords['lats'].astype(np.float64)

                        if blended_pred is None:
                            out_channels = pred_phys.shape[0]
                            blended_pred = np.zeros((out_channels, full_h, full_w), dtype=np.float64)
                            blended_target = np.zeros((out_channels, full_h, full_w), dtype=np.float64)

                        win_h = len(window_lats)
                        win_w = len(window_lons)
                        weights = generate_cosine_weights(win_h, win_w, taper_ratio, min_weight)

                        lon_idx_start = int(np.searchsorted(all_lons, window_lons[0])) - full_lon_start_idx
                        lat_idx_start = int(np.searchsorted(all_lats, window_lats[0])) - full_lat_start_idx
                        lon_idx_end = lon_idx_start + win_w
                        lat_idx_end = lat_idx_start + win_h

                        p_h_start = 0
                        p_w_start = 0
                        p_h_end = win_h
                        p_w_end = win_w

                        if lat_idx_start < 0:
                            p_h_start = -lat_idx_start
                            lat_idx_start = 0
                        if lon_idx_start < 0:
                            p_w_start = -lon_idx_start
                            lon_idx_start = 0
                        if lat_idx_end > full_h:
                            p_h_end = win_h - (lat_idx_end - full_h)
                            lat_idx_end = full_h
                        if lon_idx_end > full_w:
                            p_w_end = win_w - (lon_idx_end - full_w)
                            lon_idx_end = full_w

                        if lat_idx_end <= lat_idx_start or lon_idx_end <= lon_idx_start:
                            continue

                        weights_slice = weights[p_h_start:p_h_end, p_w_start:p_w_end]
                        weights_3d = weights_slice[np.newaxis, :, :]
                        pred_slice = pred_phys[:, p_h_start:p_h_end, p_w_start:p_w_end]
                        target_slice = target_phys[:, p_h_start:p_h_end, p_w_start:p_w_end]

                        blended_pred[:, lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end] += pred_slice * weights_3d
                        blended_target[:, lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end] += target_slice * weights_3d
                        weight_sum[lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end] += weights_slice
                        valid_windows += 1

            if valid_windows == 0 or blended_pred is None:
                raise RuntimeError("没有有效窗口可用于全图推理")

            weight_sum_3d = np.maximum(weight_sum, min_weight)[np.newaxis, :, :]
            blended_pred /= weight_sum_3d
            blended_target /= weight_sum_3d

            print(f"全图推理完成: {valid_windows} 个有效窗口融合")
            print(f"  输出尺寸: {blended_pred.shape}")

            return {
                'blended_pred': blended_pred.astype(np.float32),
                'blended_target': blended_target.astype(np.float32),
                'weight_sum': weight_sum.astype(np.float32),
                'lons': full_lons,
                'lats': full_lats,
            }
        finally:
            if getattr(dense_dataset, 'dataset', None) is not None:
                dense_dataset.dataset.close()
            if getattr(ref_dataset, 'dataset', None) is not None:
                ref_dataset.dataset.close()

    @staticmethod
    def _fill_nan(data: np.ndarray) -> np.ndarray:
        """填充 NaN 值（使用有效值均值）"""
        if not np.isnan(data).any():
            return data
        valid_mask = ~np.isnan(data)
        if valid_mask.sum() == 0:
            return np.zeros_like(data)
        global_mean = float(np.nanmean(data))
        if data.ndim == 4:
            for t in range(data.shape[0]):
                for d in range(data.shape[1]):
                    sl = data[t, d]
                    nan_mask = np.isnan(sl)
                    if nan_mask.any():
                        sl_mean = np.nanmean(sl)
                        sl[nan_mask] = sl_mean if not np.isnan(sl_mean) else global_mean
                        data[t, d] = sl
        elif data.ndim == 3:
            for t in range(data.shape[0]):
                sl = data[t]
                nan_mask = np.isnan(sl)
                if nan_mask.any():
                    sl_mean = np.nanmean(sl)
                    sl[nan_mask] = sl_mean if not np.isnan(sl_mean) else global_mean
                    data[t] = sl
        data[np.isnan(data)] = global_mean
        return data

    def _denormalize_predictions(self, predictions, targets, dataset, sample_indices=None):
        """反标准化预测结果和目标数据，anomaly 模式下同时加回气候态。"""
        print("反标准化预测结果...")

        if hasattr(dataset, 'inverse_transform_targets'):
            try:
                pred_np = torch.stack(predictions).numpy()
                target_np = torch.stack(targets).numpy()
                pred_original = dataset.inverse_transform_targets(
                    pred_np,
                    sample_indices=sample_indices,
                )
                target_original = dataset.inverse_transform_targets(
                    target_np,
                    sample_indices=sample_indices,
                )
                predictions_original = [torch.from_numpy(arr.astype(np.float32)) for arr in pred_original]
                targets_original = [torch.from_numpy(arr.astype(np.float32)) for arr in target_original]
                print("反标准化完成")
                return predictions_original, targets_original
            except Exception as exc:
                print(f"警告: 目标物理量恢复失败，回退到普通反标准化: {exc}")

        if not hasattr(dataset, 'scalers') or not dataset.scalers:
            print("警告: 未找到标准化器，使用原始数据")
            return predictions, targets

        predictions_original = []
        targets_original = []
        
        target_variables = self.config['target_variables']
        channel_slices = getattr(dataset, 'target_channel_slices', {}) or self.target_channel_slices
        
        for i in range(len(predictions)):
            pred = predictions[i].clone()  # (pred_len, channels, height, width)
            target = targets[i].clone()
            
            # 对每个目标变量进行反标准化
            for var_idx, var_name in enumerate(target_variables):
                if var_name in dataset.scalers:
                    scaler = dataset.scalers[var_name]
                    slice_indices = channel_slices.get(var_name, slice(var_idx, var_idx + 1))

                    pred_var = pred[:, slice_indices, :, :]
                    original_shape = pred_var.shape
                    pred_2d = pred_var.reshape(-1, 1).numpy()
                    pred_denorm = scaler.inverse_transform(pred_2d)
                    pred[:, slice_indices, :, :] = torch.from_numpy(pred_denorm.reshape(original_shape)).to(pred.dtype)

                    target_var = target[:, slice_indices, :, :]
                    original_shape = target_var.shape
                    target_2d = target_var.reshape(-1, 1).numpy()
                    target_denorm = scaler.inverse_transform(target_2d)
                    target[:, slice_indices, :, :] = torch.from_numpy(target_denorm.reshape(original_shape)).to(target.dtype)
            
            predictions_original.append(pred)
            targets_original.append(target)
        
        print("反标准化完成")
        return predictions_original, targets_original
    
    def _save_results(self, predictions, targets, inputs_list, dataset):
        """保存预测结果"""
        print("保存预测结果...")
        
        # 转换为numpy数组
        predictions_np = torch.stack(predictions).numpy()
        targets_np = torch.stack(targets).numpy()
        inputs_np = torch.stack(inputs_list).numpy()
        
        # 保存原始结果
        np.savez(
            os.path.join(self.output_dir, 'predictions.npz'),
            predictions=predictions_np,
            targets=targets_np,
            inputs=inputs_np
        )
        
        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # 保存模型路径信息
        info = {
            'model_path': self.model_path,
            'model_index': self.model_index,
            'model_dir': self.model_dir,
            'data_path': self.config['data_path'],
            'num_samples': len(predictions),
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(self.output_dir, 'info.json'), 'w') as f:
            json.dump(info, f, indent=4)
    
    def _visualize_results(self, predictions, targets, inputs_list, metrics, dataset=None):
        """可视化预测结果"""
        print("生成可视化结果...")
        
        # 获取经纬度坐标
        if dataset is not None:
            lons = dataset.lons
            lats = dataset.lats
        else:
            # 如果没有提供数据集，使用默认的索引坐标
            lons = None
            lats = None
        
        # 选择几个样本进行可视化
        num_vis = min(3, len(predictions))
        
        for i in range(num_vis):
            pred = predictions[i]  # (pred_len, channels, height, width)
            target = targets[i]
            
            # 假设前面的通道是温度，后面的是盐度
            num_channels = pred.shape[1]
            channels_per_var = num_channels // len(self.config['target_variables'])  # 每个变量的通道数
            temp_channels = channels_per_var  # 温度变量的通道数
            
            # 绘制温度预测对比图
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle(f'样本 {i+1} 海洋温度预测结果对比', fontsize=16, fontweight='bold')
            
            # 在图像上添加指标文本
            if 'temperature' in metrics:
                temp_metrics = metrics['temperature']
                metrics_text = (
                    f"MSE: {temp_metrics['MSE']:.6f}\n"
                    f"MAE: {temp_metrics['MAE']:.6f}\n"
                    f"RMSE: {temp_metrics['RMSE']:.6f}\n"
                    f"Corr: {temp_metrics['Correlation']:.6f}\n"
                    f"R²: {temp_metrics['R2']:.6f}"
                )
                fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure, fontsize=10, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 温度对比
            for t in range(min(3, pred.shape[0])):
                # 预测温度 (取所有温度通道的平均)
                pred_temp = pred[t, :temp_channels].mean(dim=0).numpy()
                target_temp = target[t, :temp_channels].mean(dim=0).numpy()
                
                # 预测结果
                im1 = axes[0, t].imshow(pred_temp, cmap='RdYlBu_r', aspect='auto')
                axes[0, t].set_title(f'预测海温 第{t+1}步', fontsize=12, fontweight='bold')
                
                # 设置坐标轴
                if lons is not None and lats is not None:
                    # 使用实际经纬度坐标
                    lon_ticks = [0, len(lons)//4, len(lons)//2, 3*len(lons)//4, len(lons)-1]
                    lat_ticks = [0, len(lats)//4, len(lats)//2, 3*len(lats)//4, len(lats)-1]
                    
                    axes[0, t].set_xticks(lon_ticks)
                    axes[0, t].set_yticks(lat_ticks)
                    axes[0, t].set_xticklabels([f'{lons[i]:.1f}°E' for i in lon_ticks])
                    axes[0, t].set_yticklabels([f'{lats[i]:.1f}°N' for i in lat_ticks])
                else:
                    axes[0, t].set_xlabel('经度方向', fontsize=10)
                    axes[0, t].set_ylabel('纬度方向', fontsize=10)
                
                cbar1 = plt.colorbar(im1, ax=axes[0, t], shrink=0.8)
                cbar1.set_label('温度 (°C)', fontsize=9)
                
                # 真实结果
                im2 = axes[1, t].imshow(target_temp, cmap='RdYlBu_r', aspect='auto')
                axes[1, t].set_title(f'真实海温 第{t+1}步', fontsize=12, fontweight='bold')
                
                # 设置坐标轴
                if lons is not None and lats is not None:
                    axes[1, t].set_xticks(lon_ticks)
                    axes[1, t].set_yticks(lat_ticks)
                    axes[1, t].set_xticklabels([f'{lons[i]:.1f}°E' for i in lon_ticks])
                    axes[1, t].set_yticklabels([f'{lats[i]:.1f}°N' for i in lat_ticks])
                else:
                    axes[1, t].set_xlabel('经度方向', fontsize=10)
                    axes[1, t].set_ylabel('纬度方向', fontsize=10)
                
                cbar2 = plt.colorbar(im2, ax=axes[1, t], shrink=0.8)
                cbar2.set_label('温度 (°C)', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'temperature_prediction_sample_{i}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # 绘制盐度预测对比图
            if temp_channels < num_channels:  # 确保有盐度通道
                fig, axes = plt.subplots(2, 3, figsize=(16, 10))
                fig.suptitle(f'样本 {i+1} 海洋盐度预测结果对比', fontsize=16, fontweight='bold')
                
                # 在图像上添加盐度指标文本
                if 'salinity' in metrics:
                    salt_metrics = metrics['salinity']
                    metrics_text = (
                        f"MSE: {salt_metrics['MSE']:.6f}\n"
                        f"MAE: {salt_metrics['MAE']:.6f}\n"
                        f"RMSE: {salt_metrics['RMSE']:.6f}\n"
                        f"Corr: {salt_metrics['Correlation']:.6f}\n"
                        f"R²: {salt_metrics['R2']:.6f}"
                    )
                    fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure, fontsize=10, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                # 盐度对比
                for t in range(min(3, pred.shape[0])):
                    # 预测盐度 (取所有盐度通道的平均)
                    pred_salt = pred[t, temp_channels:].mean(dim=0).numpy()
                    target_salt = target[t, temp_channels:].mean(dim=0).numpy()
                    
                    # 预测结果
                    im1 = axes[0, t].imshow(pred_salt, cmap='viridis', aspect='auto')
                    axes[0, t].set_title(f'预测盐度 第{t+1}步', fontsize=12, fontweight='bold')
                    
                    # 设置坐标轴
                    if lons is not None and lats is not None:
                        axes[0, t].set_xticks(lon_ticks)
                        axes[0, t].set_yticks(lat_ticks)
                        axes[0, t].set_xticklabels([f'{lons[i]:.1f}°E' for i in lon_ticks])
                        axes[0, t].set_yticklabels([f'{lats[i]:.1f}°N' for i in lat_ticks])
                    else:
                        axes[0, t].set_xlabel('经度方向', fontsize=10)
                        axes[0, t].set_ylabel('纬度方向', fontsize=10)
                    
                    cbar1 = plt.colorbar(im1, ax=axes[0, t], shrink=0.8)
                    cbar1.set_label('盐度 (PSU)', fontsize=9)
                    
                    # 真实结果
                    im2 = axes[1, t].imshow(target_salt, cmap='viridis', aspect='auto')
                    axes[1, t].set_title(f'真实盐度 第{t+1}步', fontsize=12, fontweight='bold')
                    
                    # 设置坐标轴
                    if lons is not None and lats is not None:
                        axes[1, t].set_xticks(lon_ticks)
                        axes[1, t].set_yticks(lat_ticks)
                        axes[1, t].set_xticklabels([f'{lons[i]:.1f}°E' for i in lon_ticks])
                        axes[1, t].set_yticklabels([f'{lats[i]:.1f}°N' for i in lat_ticks])
                    else:
                        axes[1, t].set_xlabel('经度方向', fontsize=10)
                        axes[1, t].set_ylabel('纬度方向', fontsize=10)
                    
                    cbar2 = plt.colorbar(im2, ax=axes[1, t], shrink=0.8)
                    cbar2.set_label('盐度 (PSU)', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'salinity_prediction_sample_{i}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
            
            # 绘制温度和盐度的组合对比图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'样本 {i+1} 温度盐度综合预测对比 (第1步)', fontsize=16, fontweight='bold')
            
            # 在图像上添加整体指标文本
            overall_metrics = metrics['overall']
            metrics_text = (
                f"整体指标:\n"
                f"MSE: {overall_metrics['MSE']:.6f}\n"
                f"MAE: {overall_metrics['MAE']:.6f}\n"
                f"RMSE: {overall_metrics['RMSE']:.6f}\n"
                f"Corr: {overall_metrics['Correlation']:.6f}\n"
                f"R²: {overall_metrics['R2']:.6f}"
            )
            fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # 第一个预测步长的温度和盐度
            pred_temp_0 = pred[0, :temp_channels].mean(dim=0).numpy()
            target_temp_0 = target[0, :temp_channels].mean(dim=0).numpy()
            
            # 预测温度
            im1 = axes[0, 0].imshow(pred_temp_0, cmap='RdYlBu_r', aspect='auto')
            axes[0, 0].set_title('预测温度', fontsize=12, fontweight='bold')
            
            # 设置坐标轴
            if lons is not None and lats is not None:
                axes[0, 0].set_xticks(lon_ticks)
                axes[0, 0].set_yticks(lat_ticks)
                axes[0, 0].set_xticklabels([f'{lons[i]:.1f}°E' for i in lon_ticks])
                axes[0, 0].set_yticklabels([f'{lats[i]:.1f}°N' for i in lat_ticks])
            else:
                axes[0, 0].set_xlabel('经度方向', fontsize=10)
                axes[0, 0].set_ylabel('纬度方向', fontsize=10)
            
            cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
            cbar1.set_label('温度 (°C)', fontsize=9)
            
            # 真实温度
            im2 = axes[0, 1].imshow(target_temp_0, cmap='RdYlBu_r', aspect='auto')
            axes[0, 1].set_title('真实温度', fontsize=12, fontweight='bold')
            
            # 设置坐标轴
            if lons is not None and lats is not None:
                axes[0, 1].set_xticks(lon_ticks)
                axes[0, 1].set_yticks(lat_ticks)
                axes[0, 1].set_xticklabels([f'{lons[i]:.1f}°E' for i in lon_ticks])
                axes[0, 1].set_yticklabels([f'{lats[i]:.1f}°N' for i in lat_ticks])
            else:
                axes[0, 1].set_xlabel('经度方向', fontsize=10)
                axes[0, 1].set_ylabel('纬度方向', fontsize=10)
            
            cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            cbar2.set_label('温度 (°C)', fontsize=9)
            
            if temp_channels < num_channels:
                pred_salt_0 = pred[0, temp_channels:].mean(dim=0).numpy()
                target_salt_0 = target[0, temp_channels:].mean(dim=0).numpy()
                
                # 预测盐度
                im3 = axes[1, 0].imshow(pred_salt_0, cmap='viridis', aspect='auto')
                axes[1, 0].set_title('预测盐度', fontsize=12, fontweight='bold')
                
                # 设置坐标轴
                if lons is not None and lats is not None:
                    axes[1, 0].set_xticks(lon_ticks)
                    axes[1, 0].set_yticks(lat_ticks)
                    axes[1, 0].set_xticklabels([f'{lons[i]:.1f}°E' for i in lon_ticks])
                    axes[1, 0].set_yticklabels([f'{lats[i]:.1f}°N' for i in lat_ticks])
                else:
                    axes[1, 0].set_xlabel('经度方向', fontsize=10)
                    axes[1, 0].set_ylabel('纬度方向', fontsize=10)
                
                cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
                cbar3.set_label('盐度 (PSU)', fontsize=9)
                
                # 真实盐度
                im4 = axes[1, 1].imshow(target_salt_0, cmap='viridis', aspect='auto')
                axes[1, 1].set_title('真实盐度', fontsize=12, fontweight='bold')
                
                # 设置坐标轴
                if lons is not None and lats is not None:
                    axes[1, 1].set_xticks(lon_ticks)
                    axes[1, 1].set_yticks(lat_ticks)
                    axes[1, 1].set_xticklabels([f'{lons[i]:.1f}°E' for i in lon_ticks])
                    axes[1, 1].set_yticklabels([f'{lats[i]:.1f}°N' for i in lat_ticks])
                else:
                    axes[1, 1].set_xlabel('经度方向', fontsize=10)
                    axes[1, 1].set_ylabel('纬度方向', fontsize=10)
                
                cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
                cbar4.set_label('盐度 (PSU)', fontsize=9)
            else:
                axes[1, 0].text(0.5, 0.5, '无盐度数据', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 1].text(0.5, 0.5, '无盐度数据', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 0].set_visible(False)
                axes[1, 1].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'combined_prediction_sample_{i}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # 生成误差分析图
        self._plot_error_analysis(predictions, targets)
        
        # 生成温度和盐度的单独误差分析
        self._plot_variable_error_analysis(predictions, targets)

        # 生成按空间区域的误差热力图
        self._plot_regional_error(predictions, targets, dataset)

    def _plot_regional_error(self, predictions, targets, dataset=None):
        """绘制空间各区域（格点）RMSE热力图，并保存矩阵数据"""
        print("生成按空间区域的误差热力图...")

        pred_stack = torch.stack(predictions)  # (num_samples, pred_len, channels, height, width)
        target_stack = torch.stack(targets)

        # 基础形状
        num_channels = pred_stack.shape[2]
        height = pred_stack.shape[3]
        width = pred_stack.shape[4]

        # 通道按变量分组（更通用：按 target_variables 等分）
        target_variables = self.config.get('target_variables', [])
        var_count = max(1, len(target_variables))
        channels_per_var = num_channels // var_count if var_count > 0 else num_channels

        # 计算整体 RMSE 空间分布
        overall_rmse_map = torch.sqrt(torch.mean((pred_stack - target_stack) ** 2, dim=(0, 1, 2)))  # (H, W)

        # 计算温度/盐度 RMSE 空间分布（若存在）
        temp_rmse_map = None
        salt_rmse_map = None

        if channels_per_var > 0:
            temp_slice = slice(0, channels_per_var)
            temp_rmse_map = torch.sqrt(torch.mean((pred_stack[:, :, temp_slice, :, :] -
                                                   target_stack[:, :, temp_slice, :, :]) ** 2,
                                                  dim=(0, 1, 2)))
        if channels_per_var < num_channels:
            salt_slice = slice(channels_per_var, num_channels)
            salt_rmse_map = torch.sqrt(torch.mean((pred_stack[:, :, salt_slice, :, :] -
                                                   target_stack[:, :, salt_slice, :, :]) ** 2,
                                                  dim=(0, 1, 2)))

        # 准备经纬度刻度
        lons = getattr(dataset, 'lons', None) if dataset is not None else None
        lats = getattr(dataset, 'lats', None) if dataset is not None else None
        lon_ticks = None
        lat_ticks = None
        if lons is not None and lats is not None and len(lons) == width and len(lats) == height:
            lon_ticks = [0, len(lons)//4, len(lons)//2, 3*len(lons)//4, len(lons)-1]
            lat_ticks = [0, len(lats)//4, len(lats)//2, 3*len(lats)//4, len(lats)-1]

        # 绘图：整体 / 温度 / 盐度 RMSE
        cols = 1 + (1 if temp_rmse_map is not None else 0) + (1 if salt_rmse_map is not None else 0)
        figsize = (6 * cols + 2, 6)
        fig, axes = plt.subplots(1, cols, figsize=figsize)
        if cols == 1:
            axes = [axes]

        idx = 0
        im0 = axes[idx].imshow(overall_rmse_map.numpy(), cmap='inferno', aspect='auto')
        axes[idx].set_title('整体RMSE（混合单位）', fontsize=12, fontweight='bold')
        if lon_ticks is not None and lat_ticks is not None:
            axes[idx].set_xticks(lon_ticks)
            axes[idx].set_yticks(lat_ticks)
            axes[idx].set_xticklabels([f'{lons[i]:.1f}°E' for i in lon_ticks])
            axes[idx].set_yticklabels([f'{lats[i]:.1f}°N' for i in lat_ticks])
        cbar0 = plt.colorbar(im0, ax=axes[idx], shrink=0.8)
        cbar0.set_label('RMSE [混合单位]', fontsize=9)
        idx += 1

        if temp_rmse_map is not None:
            im1 = axes[idx].imshow(temp_rmse_map.numpy(), cmap='RdYlBu_r', aspect='auto')
            axes[idx].set_title('温度RMSE (°C)', fontsize=12, fontweight='bold')
            if lon_ticks is not None and lat_ticks is not None:
                axes[idx].set_xticks(lon_ticks)
                axes[idx].set_yticks(lat_ticks)
                axes[idx].set_xticklabels([f'{lons[i]:.1f}°E' for i in lon_ticks])
                axes[idx].set_yticklabels([f'{lats[i]:.1f}°N' for i in lat_ticks])
            cbar1 = plt.colorbar(im1, ax=axes[idx], shrink=0.8)
            cbar1.set_label('RMSE [°C]', fontsize=9)
            idx += 1

        if salt_rmse_map is not None:
            im2 = axes[idx].imshow(salt_rmse_map.numpy(), cmap='viridis', aspect='auto')
            axes[idx].set_title('盐度RMSE (PSU)', fontsize=12, fontweight='bold')
            if lon_ticks is not None and lat_ticks is not None:
                axes[idx].set_xticks(lon_ticks)
                axes[idx].set_yticks(lat_ticks)
                axes[idx].set_xticklabels([f'{lons[i]:.1f}°E' for i in lon_ticks])
                axes[idx].set_yticklabels([f'{lats[i]:.1f}°N' for i in lat_ticks])
            cbar2 = plt.colorbar(im2, ax=axes[idx], shrink=0.8)
            cbar2.set_label('RMSE [PSU]', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regional_error.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 保存矩阵数据，便于后续分析
        save_dict = {
            'overall_rmse': overall_rmse_map.numpy()
        }
        if temp_rmse_map is not None:
            save_dict['temperature_rmse'] = temp_rmse_map.numpy()
        if salt_rmse_map is not None:
            save_dict['salinity_rmse'] = salt_rmse_map.numpy()
        if lons is not None and lats is not None:
            save_dict['lons'] = np.asarray(lons)
            save_dict['lats'] = np.asarray(lats)

        np.savez(os.path.join(self.output_dir, 'regional_metrics.npz'), **save_dict)
    
    def _plot_error_analysis(self, predictions, targets):
        """绘制误差分析图"""
        print("生成误差分析...")
        
        pred_stack = torch.stack(predictions)  # (num_samples, pred_len, channels, height, width)
        target_stack = torch.stack(targets)
        
        # 计算整体误差
        mse = torch.mean((pred_stack - target_stack) ** 2, dim=(0, 2, 3, 4))  # (pred_len,)
        mae = torch.mean(torch.abs(pred_stack - target_stack), dim=(0, 2, 3, 4))  # (pred_len,)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # MSE随时间变化
        ax1.plot(range(1, len(mse)+1), mse.numpy(), 'o-', linewidth=2, markersize=6)
        ax1.set_title('均方误差随预测步长变化', fontsize=14, fontweight='bold')
        ax1.set_xlabel('预测步长', fontsize=12)
        ax1.set_ylabel('均方误差 (MSE) [混合单位]', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, len(mse)+1))
        
        # MAE随时间变化
        ax2.plot(range(1, len(mae)+1), mae.numpy(), 'o-', color='orange', linewidth=2, markersize=6)
        ax2.set_title('平均绝对误差随预测步长变化', fontsize=14, fontweight='bold')
        ax2.set_xlabel('预测步长', fontsize=12)
        ax2.set_ylabel('平均绝对误差 (MAE) [混合单位]', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(1, len(mae)+1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_variable_error_analysis(self, predictions, targets):
        """绘制温度和盐度的单独误差分析图"""
        print("生成温度和盐度误差分析...")
        
        pred_stack = torch.stack(predictions)  # (num_samples, pred_len, channels, height, width)
        target_stack = torch.stack(targets)
        
        num_channels = pred_stack.shape[2]
        temp_channels = num_channels // 2
        
        # 分别计算温度和盐度的误差
        if temp_channels > 0:
            # 温度误差
            pred_temp = pred_stack[:, :, :temp_channels, :, :]
            target_temp = target_stack[:, :, :temp_channels, :, :]
            temp_mse = torch.mean((pred_temp - target_temp) ** 2, dim=(0, 2, 3, 4))
            temp_mae = torch.mean(torch.abs(pred_temp - target_temp), dim=(0, 2, 3, 4))
        
        if temp_channels < num_channels:
            # 盐度误差
            pred_salt = pred_stack[:, :, temp_channels:, :, :]
            target_salt = target_stack[:, :, temp_channels:, :, :]
            salt_mse = torch.mean((pred_salt - target_salt) ** 2, dim=(0, 2, 3, 4))
            salt_mae = torch.mean(torch.abs(pred_salt - target_salt), dim=(0, 2, 3, 4))
        
        # 创建图形
        if temp_channels > 0 and temp_channels < num_channels:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('温度和盐度预测误差分析', fontsize=16, fontweight='bold')
            
            # 温度MSE
            axes[0, 0].plot(range(1, len(temp_mse)+1), temp_mse.numpy(), 'o-', 
                           color='red', linewidth=2, markersize=6, label='温度')
            axes[0, 0].set_title('温度均方误差随预测步长变化', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('预测步长', fontsize=10)
            axes[0, 0].set_ylabel('均方误差 (MSE) [°C²]', fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xticks(range(1, len(temp_mse)+1))
            
            # 温度MAE
            axes[0, 1].plot(range(1, len(temp_mae)+1), temp_mae.numpy(), 'o-', 
                           color='red', linewidth=2, markersize=6, label='温度')
            axes[0, 1].set_title('温度平均绝对误差随预测步长变化', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('预测步长', fontsize=10)
            axes[0, 1].set_ylabel('平均绝对误差 (MAE) [°C]', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xticks(range(1, len(temp_mae)+1))
            
            # 盐度MSE
            axes[1, 0].plot(range(1, len(salt_mse)+1), salt_mse.numpy(), 'o-', 
                           color='blue', linewidth=2, markersize=6, label='盐度')
            axes[1, 0].set_title('盐度均方误差随预测步长变化', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('预测步长', fontsize=10)
            axes[1, 0].set_ylabel('均方误差 (MSE) [PSU²]', fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xticks(range(1, len(salt_mse)+1))
            
            # 盐度MAE
            axes[1, 1].plot(range(1, len(salt_mae)+1), salt_mae.numpy(), 'o-', 
                           color='blue', linewidth=2, markersize=6, label='盐度')
            axes[1, 1].set_title('盐度平均绝对误差随预测步长变化', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('预测步长', fontsize=10)
            axes[1, 1].set_ylabel('平均绝对误差 (MAE) [PSU]', fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xticks(range(1, len(salt_mae)+1))
            
        elif temp_channels > 0:
            # 只有温度
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('温度预测误差分析', fontsize=16, fontweight='bold')
            
            ax1.plot(range(1, len(temp_mse)+1), temp_mse.numpy(), 'o-', 
                    color='red', linewidth=2, markersize=6)
            ax1.set_title('温度均方误差随预测步长变化', fontsize=12, fontweight='bold')
            ax1.set_xlabel('预测步长', fontsize=10)
            ax1.set_ylabel('均方误差 (MSE) [°C²]', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(range(1, len(temp_mse)+1))
            
            ax2.plot(range(1, len(temp_mae)+1), temp_mae.numpy(), 'o-', 
                    color='red', linewidth=2, markersize=6)
            ax2.set_title('温度平均绝对误差随预测步长变化', fontsize=12, fontweight='bold')
            ax2.set_xlabel('预测步长', fontsize=10)
            ax2.set_ylabel('平均绝对误差 (MAE) [°C]', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(range(1, len(temp_mae)+1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'variable_error_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _compute_metrics(
        self,
        predictions,
        targets,
        normalized_predictions=None,
        normalized_targets=None,
        dataset=None,
        sample_indices=None,
    ):
        """计算评估指标"""
        print("计算评估指标...")
        
        pred_stack = torch.stack(predictions)
        target_stack = torch.stack(targets)

        def _to_serializable(value):
            if isinstance(value, (np.ndarray, np.generic)):
                value = float(value)
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (float, int)):
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    return None
                return float(value)
            return value
        
        # 获取损失权重
        # 获取目标变量信息
        target_variables = self.config.get('target_variables', [])
        var_count = max(1, len(target_variables))
        num_channels = pred_stack.shape[2]
        channels_per_var = num_channels // var_count
        channel_slices = getattr(dataset, 'target_channel_slices', {}) if dataset is not None else self.target_channel_slices
        if not channel_slices:
            channel_slices = self.target_channel_slices

        baseline_refs = {'physical': {}, 'normalized': {}}
        if dataset is not None and hasattr(dataset, 'build_reference_forecasts'):
            try:
                baseline_refs = dataset.build_reference_forecasts(sample_indices=sample_indices)
            except Exception as exc:
                print(f"警告: baseline 指标构造失败: {exc}")

        pred_np = pred_stack.numpy()
        target_np = target_stack.numpy()
        physical_report = compute_metric_report(
            pred_np,
            target_np,
            target_variables,
            channel_slices=channel_slices,
            baselines=baseline_refs.get('physical'),
        )

        normalized_report = None
        if normalized_predictions is not None and normalized_targets is not None:
            norm_pred_np = torch.stack(normalized_predictions).numpy()
            norm_target_np = torch.stack(normalized_targets).numpy()
            normalized_report = compute_metric_report(
                norm_pred_np,
                norm_target_np,
                target_variables,
                channel_slices=channel_slices,
                baselines=baseline_refs.get('normalized'),
            )

        temp_weight = self.config.get('temp_weight', 0.7)
        salt_weight = self.config.get('salt_weight', 0.3)

        weighted_loss = 0.0
        temp_loss = 0.0
        salt_loss = 0.0

        if var_count == 1:
            # 单变量直接计算整体MSE
            temp_loss = torch.mean((pred_stack - target_stack) ** 2).item()
            weighted_loss = temp_loss
        else:
            # 双变量（保留原逻辑）
            pred_temp = pred_stack[:, :, :channels_per_var, :, :]
            target_temp = target_stack[:, :, :channels_per_var, :, :]
            temp_loss = torch.mean((pred_temp - target_temp) ** 2).item()
            weighted_loss += temp_weight * temp_loss

            pred_salt = pred_stack[:, :, channels_per_var:, :, :]
            target_salt = target_stack[:, :, channels_per_var:, :, :]
            salt_loss = torch.mean((pred_salt - target_salt) ** 2).item()
            weighted_loss += salt_weight * salt_loss
        
        # 整体指标
        mse = torch.mean((pred_stack - target_stack) ** 2).item()
        mae = torch.mean(torch.abs(pred_stack - target_stack)).item()
        rmse = np.sqrt(mse)
        
        # 计算相关系数
        # 转换为 float64 以确保计算精度
        pred_flat = pred_stack.reshape(-1).numpy().astype(np.float64)
        target_flat = target_stack.reshape(-1).numpy().astype(np.float64)
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]

        # 计算R^2
        target_mean = target_flat.mean()
        ss_tot = np.sum((target_flat - target_mean) ** 2)
        ss_res = np.sum((pred_flat - target_flat) ** 2)
        r2 = float('nan') if ss_tot == 0 else 1 - ss_res / ss_tot
        
        metrics = {
            'overall': {
                'MSE': _to_serializable(mse),
                'MAE': _to_serializable(mae),
                'RMSE': _to_serializable(rmse),
                'Correlation': _to_serializable(correlation),
                'R2': _to_serializable(r2)
            },
            'physical_report': physical_report,
            'normalized_report': normalized_report,
            'baseline_reports': {
                'physical': physical_report.get('baselines', {}),
                'normalized': normalized_report.get('baselines', {}) if normalized_report else {},
            },
            'baseline_comparison': {
                'physical': physical_report.get('comparison', {}),
                'normalized': normalized_report.get('comparison', {}) if normalized_report else {},
            },
            'num_samples': len(predictions)
        }
            # 仅在多变量时保留加权结构
        if var_count > 1:
            metrics['weighted'] = {
                'Weighted_MSE': _to_serializable(weighted_loss),
                'Temp_MSE': _to_serializable(temp_loss),
                'Salt_MSE': _to_serializable(salt_loss),
                'Temp_Weight': _to_serializable(temp_weight),
                'Salt_Weight': _to_serializable(salt_weight)
            }
        else:
            metrics['weighted'] = {
                'Weighted_MSE': _to_serializable(weighted_loss),
                f'{target_variables[0] if target_variables else "Var"}_MSE': _to_serializable(temp_loss)
            }
        
        # 分别计算温度和盐度的指标
        # 按变量分别计算指标
        for var_idx in range(var_count):
            start_ch = var_idx * channels_per_var
            end_ch = (var_idx + 1) * channels_per_var
            pred_var = pred_stack[:, :, start_ch:end_ch, :, :]
            target_var = target_stack[:, :, start_ch:end_ch, :, :]

            var_mse = torch.mean((pred_var - target_var) ** 2).item()
            var_mae = torch.mean(torch.abs(pred_var - target_var)).item()
            var_rmse = np.sqrt(var_mse)

            pred_var_flat = pred_var.reshape(-1).numpy()
            target_var_flat = target_var.reshape(-1).numpy()
            var_corr = np.corrcoef(pred_var_flat, target_var_flat)[0, 1]
            var_target_mean = target_var_flat.mean()
            var_ss_tot = np.sum((target_var_flat - var_target_mean) ** 2)
            var_ss_res = np.sum((pred_var_flat - target_var_flat) ** 2)
            var_r2 = float('nan') if var_ss_tot == 0 else 1 - var_ss_res / var_ss_tot

            name_map = {
                'TEMP': 'temperature',
                'SALT': 'salinity'
            }
            var_name_key = target_variables[var_idx] if var_idx < len(target_variables) else f'var{var_idx}'
            metrics_key = name_map.get(var_name_key, var_name_key.lower())
            metrics[metrics_key] = {
                'MSE': _to_serializable(var_mse),
                'MAE': _to_serializable(var_mae),
                'RMSE': _to_serializable(var_rmse),
                'Correlation': _to_serializable(var_corr),
                'R2': _to_serializable(var_r2)
            }
        
        # 保存指标
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("评估指标:")
        print("  整体指标:")
        for key, value in metrics['overall'].items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")
        
        print("  加权损失:")
        for key, value in metrics['weighted'].items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")
        
        if 'temperature' in metrics:
            print("  温度指标:")
            for key, value in metrics['temperature'].items():
                print(f"    {key}: {value:.6f}")
        
        if 'salinity' in metrics:
            print("  盐度指标:")
            for key, value in metrics['salinity'].items():
                print(f"    {key}: {value:.6f}")

        if normalized_report is not None:
            norm_overall = normalized_report.get('overall', {})
            norm_rmse = norm_overall.get('rmse')
            norm_mae = norm_overall.get('mae')
            norm_r2 = norm_overall.get('r2')
            print("  Normalized/anomaly 指标:")
            print(
                "    "
                f"RMSE: {norm_rmse:.6f}" if norm_rmse is not None else "    RMSE: nan",
                "|",
                f"MAE: {norm_mae:.6f}" if norm_mae is not None else "MAE: nan",
                "|",
                f"R2: {norm_r2:.6f}" if norm_r2 is not None else "R2: nan",
            )

        print("  Baseline 对比 (RMSE improvement %, 越高越好):")
        for space_name, comparison in metrics['baseline_comparison'].items():
            if not comparison:
                continue
            print(f"    [{space_name}]")
            for key, value in comparison.items():
                if value is None:
                    print(f"      {key}: nan")
                else:
                    print(f"      {key}: {value:.2f}%")
        
        print(f"  样本数量: {metrics['num_samples']}")
        
        return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ConvLSTM 海洋预测脚本")
    parser.add_argument('--model', type=int, required=True, help='选择 outputs/results 下模型编号')
    parser.add_argument('--samples', type=int, default=None, help='预测样本数（覆盖配置）')
    parser.add_argument('--output_dir', type=str, default=None, help='自定义预测输出目录')
    args = parser.parse_args()
    print("ConvLSTM海洋预测系统 - 统一配置版本")
    print("=" * 50)
    
    # 使用统一配置
    config = DEFAULT_CONFIG.copy()
    
    # 可以在这里覆盖特定的预测参数
    config.update({
        # 'num_samples': 20,           # 覆盖预测样本数
        # 'visualization_samples': 5,  # 覆盖可视化样本数
    })
    
    print("使用统一配置:")
    print(f"  数据路径: {config['data_path']}")
    print(f"  预测样本数: {config['num_samples']}")
    print(f"  可视化样本数: {config['visualization_samples']}")
    print("=" * 50)
    
    if not os.path.exists(config['data_path']):
        print(f"错误: 数据文件不存在: {config['data_path']}")
        return
    
    try:
        # 创建预测器
        predictor = SmartOceanPredictor(model_index=args.model, config=config, output_dir=args.output_dir)

        # 进行预测
        predictor.predict(num_samples=args.samples)

        print("预测任务完成!")

    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
