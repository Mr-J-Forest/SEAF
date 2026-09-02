#!/usr/bin/env python3
"""
SEAF 海洋预测脚本 - 统一配置版本
自动匹配模型结构和权重，生成可视化预测结果
使用统一配置文件确保与训练脚本参数一致
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Sequence

# 导入统一配置
from config import DEFAULT_CONFIG, load_config, merge_configs, save_config
from model_factory import create_ocean_model
from data_loader import OceanDataset, TimeGroupedBatchSampler
from font_config import setup_chinese_fonts
from metrics_utils import compute_metric_report, resolve_variable_slices

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


def finalize_weighted_blend(blended_pred, blended_target, weight_sum):
    """归一化 overlap-tile 累积值，并把未覆盖格点显式标记为 NaN。"""
    coverage_mask = np.asarray(weight_sum) > 0
    pred = np.full_like(blended_pred, np.nan, dtype=np.float64)
    target = np.full_like(blended_target, np.nan, dtype=np.float64)
    if np.any(coverage_mask):
        denom = weight_sum[coverage_mask]
        pred[:, coverage_mask] = blended_pred[:, coverage_mask] / denom
        target[:, coverage_mask] = blended_target[:, coverage_mask] / denom
    return pred, target, coverage_mask


def interleaved_batches(indices, batch_size):
    """把空间有序窗口交错分配到 batch，增加每批的远场覆盖。"""
    indices = list(indices)
    batch_size = max(1, int(batch_size))
    if not indices:
        return []
    batch_count = int(np.ceil(len(indices) / batch_size))
    return [indices[offset::batch_count] for offset in range(batch_count)]


def balanced_group_sample_positions(group_sizes, total_samples):
    """Allocate retained examples across forecast origins and their windows."""
    sizes = [max(0, int(value)) for value in group_sizes]
    remaining = min(max(0, int(total_samples)), sum(sizes))
    output = []
    group_count = len(sizes)
    for group_idx, size in enumerate(sizes):
        groups_left = group_count - group_idx
        quota = min(size, int(np.ceil(remaining / groups_left))) if groups_left else 0
        if quota <= 0:
            positions = []
        elif quota == 1 and group_count > 1:
            positions = [int(round(group_idx * (size - 1) / (group_count - 1)))]
        else:
            positions = np.linspace(0, size - 1, num=quota, dtype=int).tolist()
        output.append(positions)
        remaining -= len(positions)
    return output


class SmartOceanPredictor:
    """智能海洋预测器 - 使用统一配置并按编号加载模型"""
    
    def __init__(self,
                 model_index: Optional[int] = None,
                 config: Optional[Dict] = None,
                 output_dir: Optional[str] = None,
                 selected_variables: Optional[Sequence[str]] = None,
                 model_dir: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            model_index: 结果目录编号（outputs/results/<index>_*）
            model_dir: 确定性的结果目录（与 model_index 二选一）
            config: 配置字典，如果为None则使用默认配置
            output_dir: 输出目录，默认自动生成
        """
        # 使用统一配置
        if config is None:
            self.config = DEFAULT_CONFIG.copy()
        else:
            self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if (model_index is None) == (model_dir is None):
            raise ValueError("model_index 与 model_dir 必须且只能提供一个")
        self.model_index = model_index
        self.model_label = str(model_index) if model_index is not None else os.path.basename(os.path.abspath(model_dir))
        
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime(self.config['timestamp_format'])
            self.output_dir = os.path.join(
                self.config['predictions_dir'],
                f"predictions_model{self.model_label}_{timestamp}"
            )
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if model_dir is not None:
            self.model_dir, self.model_path = self._resolve_model_directory(model_dir)
        else:
            self.model_dir, self.model_path = self._resolve_model_by_index(model_index)
        print(f"使用模型目录: {self.model_dir}")
        print(f"使用权重: {self.model_path}")
        self.preprocessing_scalers = self._load_preprocessing_scalers()
        
        # 加载模型
        self._load_model()
        self.is_dynaseaf = (
            str(self.config.get('model_type', 'seaf')).lower() == 'dynaseaf'
        )

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
        return self._resolve_model_directory(os.path.join(base_dir, candidates[0]))

    def _resolve_model_directory(self, model_dir: str) -> Tuple[str, str]:
        result_dir = os.path.abspath(model_dir)
        if not os.path.isdir(result_dir):
            raise FileNotFoundError(f"模型目录不存在: {result_dir}")
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
            merged_config['enable_target_climatology_anomaly'] = False
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

    def _load_preprocessing_scalers(self):
        """加载训练时保存的 scaler，确保预测与训练处于同一模型空间。"""
        filename = self.config.get('scalers_filename', 'scalers.pkl')
        path = os.path.join(self.model_dir, filename)
        if not os.path.isfile(path):
            print("警告: 旧模型未保存训练 scaler，将从 train 数据管线重建")
            return None
        with open(path, 'rb') as f:
            scalers = pickle.load(f)
        print(f"加载训练期标准化器: {path}")
        return scalers

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
    
    def predict(
        self,
        num_samples: Optional[int] = None,
        save_dynaseaf_diagnostics: bool = False,
    ):
        """
        进行预测
        
        Args:
            num_samples: 预测样本数量，如果为None则使用配置中的默认值
            save_dynaseaf_diagnostics: 是否额外保存 DynaSEAF 的分解张量；
                该导出只调用模型生成的 diagnostics，不读取未来动力学标签。
        """
        if num_samples is None:
            num_samples = self.config['num_samples']
            
        print(f"开始预测 {num_samples} 个样本...")
        
        # 创建测试数据集（包含所有100%海洋窗口）
        scalers = self.preprocessing_scalers
        inference_config = dict(self.config)
        inference_config['return_future_dynamics_targets'] = False
        if scalers is None:
            ref_dataset = OceanDataset(
                inference_config['data_path'], inference_config, mode='train'
            )
            scalers = ref_dataset.scalers
            if getattr(ref_dataset, 'dataset', None) is not None:
                ref_dataset.dataset.close()
        test_dataset = OceanDataset(
            inference_config['data_path'], inference_config, mode='test', scalers=scalers
        )
        dataset_slices = getattr(test_dataset, 'target_channel_slices', {}) or {}
        if dataset_slices:
            self.target_channel_slices = dataset_slices
        
        if len(test_dataset) == 0:
            print("测试集为空，无法进行预测")
            if getattr(test_dataset, 'dataset', None) is not None:
                test_dataset.dataset.close()
            return
        
        # 限制样本数量
        num_samples = min(num_samples, len(test_dataset))
        
        predictions = []
        targets = []
        inputs_list = []
        sample_indices = []
        dynaseaf_diagnostics = {}
        
        print(f"数据集大小: {len(test_dataset)}")
        print(f"实际预测样本数: {num_samples}")

        test_dataset.return_sample_index = True
        inference_batch_size = max(2, int(self.config.get('batch_size', 4)))
        grouped_sampler = TimeGroupedBatchSampler(
            test_dataset, inference_batch_size, shuffle=False
        )
        all_batches = list(grouped_sampler)
        # Retain balanced examples across temporal origins and spatial order.
        needed_batches = min(len(all_batches), num_samples)
        selected_positions = np.linspace(
            0, len(all_batches) - 1, num=needed_batches, dtype=int
        ) if needed_batches else np.array([], dtype=int)
        selected_batches = [all_batches[int(pos)] for pos in selected_positions]
        retained_positions = balanced_group_sample_positions(
            [len(batch) for batch in selected_batches], num_samples
        )
        loader = DataLoader(
            test_dataset,
            batch_sampler=selected_batches,
            num_workers=0,
            pin_memory=bool(self.config.get('pin_memory', True)),
        )
        with torch.no_grad():
            for group_idx, batch in enumerate(loader):
                batch_inputs, batch_targets, batch_indices = batch[:3]
                remaining = num_samples - len(predictions)
                if remaining <= 0:
                    break
                model_output = self.model(
                    batch_inputs.to(self.device, non_blocking=True),
                    return_diagnostics=(
                        bool(save_dynaseaf_diagnostics) and self.is_dynaseaf
                    ),
                ) if bool(save_dynaseaf_diagnostics) and self.is_dynaseaf else self.model(
                    batch_inputs.to(self.device, non_blocking=True)
                )
                if isinstance(model_output, dict):
                    outputs = model_output['forecast'].cpu()
                    batch_diagnostics = {
                        name: value.detach().float().cpu()
                        for name, value in model_output.items()
                        if name != 'forecast' and torch.is_tensor(value)
                    }
                else:
                    outputs = model_output.cpu()
                    batch_diagnostics = {}
                for local_idx in retained_positions[group_idx]:
                    sample_idx = int(batch_indices[local_idx])
                    output = outputs[local_idx]
                    if not torch.isfinite(output).all():
                        print(f"警告: 样本 {sample_idx} 的预测结果包含NaN或Inf，跳过")
                        continue
                    predictions.append(output)
                    targets.append(batch_targets[local_idx])
                    inputs_list.append(batch_inputs[local_idx])
                    sample_indices.append(sample_idx)
                    for name, value in batch_diagnostics.items():
                        dynaseaf_diagnostics.setdefault(name, []).append(
                            value[local_idx]
                        )
                print(f"已完成 {len(predictions)}/{num_samples} 个样本的预测")
        
        if len(predictions) == 0:
            print("所有预测结果都包含异常值，无法生成有效预测")
            if getattr(test_dataset, 'dataset', None) is not None:
                test_dataset.dataset.close()
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
        self._save_results(
            predictions_original,
            targets_original,
            inputs_list,
            test_dataset,
            sample_indices,
            dynaseaf_diagnostics=dynaseaf_diagnostics
            if save_dynaseaf_diagnostics and self.is_dynaseaf else None,
        )
        self._visualize_results(
            predictions_original,
            targets_original,
            metrics,
            test_dataset,
            sample_indices,
        )
        if getattr(test_dataset, 'dataset', None) is not None:
            test_dataset.dataset.close()
        
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
        ref_dataset = None
        scalers = self.preprocessing_scalers
        inference_config = dict(self.config)
        inference_config['return_future_dynamics_targets'] = False
        if scalers is None:
            ref_dataset = OceanDataset(
                inference_config['data_path'], inference_config, mode='train'
            )
            scalers = ref_dataset.scalers

        print("构建密集推理窗口数据集...")
        dense_dataset = OceanDataset(
            inference_config['data_path'],
            inference_config,
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
                base_time_index = test_start - 1
            base_time_index = max(
                max(seq_len - 1, test_start - 1),
                min(base_time_index, total_t - pred_len - 1),
            )
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
            surface_temp = np.asarray(
                dense_dataset.dataset['TEMP'].isel(TIME=0, LEVEL=0).values
            )
            ocean_domain_mask = np.isfinite(surface_temp)[
                full_lat_start_idx:full_lat_end_idx,
                full_lon_start_idx:full_lon_end_idx,
            ]

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
            blended_baselines = {}
            weight_sum = np.zeros((full_h, full_w), dtype=np.float64)
            valid_windows = 0
            inference_batch_size = max(
                1,
                int(self.config.get(
                    'inference_micro_batch_size',
                    self.config.get('batch_size', 8),
                )),
            )
            inference_batches = interleaved_batches(candidate_indices, inference_batch_size)

            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch_indices in enumerate(inference_batches):
                    batch_inputs = []
                    batch_targets = []
                    for sample_idx in batch_indices:
                        inputs, target_tensor = dense_dataset[sample_idx]
                        batch_inputs.append(inputs)
                        batch_targets.append(target_tensor)

                    inputs_tensor = torch.stack(batch_inputs, dim=0).to(self.device)
                    outputs = self.model(inputs_tensor)
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"警告: batch {batch_idx} 输出包含 NaN/Inf，逐窗口过滤")

                    pred_phys_batch = dense_dataset.inverse_transform_targets(
                        outputs.cpu().numpy(),
                        sample_indices=batch_indices,
                    )[:, pred_step]
                    target_phys_batch = dense_dataset.inverse_transform_targets(
                        torch.stack(batch_targets, dim=0).numpy(),
                        sample_indices=batch_indices,
                    )[:, pred_step]
                    baseline_batch = dense_dataset.build_reference_forecasts(
                        sample_indices=batch_indices,
                        spaces=("physical",),
                    ).get("physical", {})
                    baseline_phys_batch = {
                        name: values[:, pred_step]
                        for name, values in baseline_batch.items()
                    }

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
                            blended_baselines = {
                                name: np.zeros((out_channels, full_h, full_w), dtype=np.float64)
                                for name in baseline_phys_batch
                            }

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
                        for name, baseline_values in baseline_phys_batch.items():
                            baseline_slice = baseline_values[local_idx, :, p_h_start:p_h_end, p_w_start:p_w_end]
                            blended_baselines[name][
                                :, lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end
                            ] += baseline_slice * weights_3d
                        weight_sum[lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end] += weights_slice
                        valid_windows += 1

            if valid_windows == 0 or blended_pred is None:
                raise RuntimeError("没有有效窗口可用于全图推理")

            blended_pred, blended_target, coverage_mask = finalize_weighted_blend(
                blended_pred, blended_target, weight_sum
            )
            finalized_baselines = {}
            for name, values in blended_baselines.items():
                finalized, _, _ = finalize_weighted_blend(
                    values,
                    values,
                    weight_sum,
                )
                finalized_baselines[name] = finalized.astype(np.float32)
            coverage_pct = float(np.mean(coverage_mask) * 100.0)
            ocean_points = int(ocean_domain_mask.sum())
            covered_ocean_points = int(np.count_nonzero(coverage_mask & ocean_domain_mask))
            ocean_coverage_fraction = (
                covered_ocean_points / ocean_points if ocean_points else None
            )

            print(f"全图推理完成: {valid_windows} 个有效窗口融合")
            print(f"  输出尺寸: {blended_pred.shape}")
            print(f"  目标区域有效覆盖率: {coverage_pct:.2f}%（未覆盖格点为 NaN）")
            if ocean_coverage_fraction is not None:
                print(f"  目标区域海洋格点覆盖率: {ocean_coverage_fraction * 100:.2f}%")

            return {
                'blended_pred': blended_pred.astype(np.float32),
                'blended_target': blended_target.astype(np.float32),
                'blended_baselines': finalized_baselines,
                'weight_sum': weight_sum.astype(np.float32),
                'coverage_mask': coverage_mask,
                'ocean_domain_mask': ocean_domain_mask,
                'lons': full_lons,
                'lats': full_lats,
                'target_variables': list(self.config.get('target_variables', [])),
                'target_channel_slices': {
                    name: [sl.start, sl.stop]
                    for name, sl in dense_dataset.target_channel_slices.items()
                },
                'levels': dense_dataset.levels.astype(np.float32),
                'base_time_index': int(base_time_index),
                'forecast_time_index': int(base_time_index + 1 + pred_step),
                'prediction_step': int(pred_step),
                'valid_windows': int(valid_windows),
                'candidate_windows': int(len(candidate_indices)),
                'inference_micro_batch_size': int(inference_batch_size),
                'coverage_fraction': coverage_pct / 100.0,
                'ocean_coverage_fraction': ocean_coverage_fraction,
                'covered_ocean_grid_points': covered_ocean_points,
                'ocean_grid_points': ocean_points,
                'window_ocean_threshold': float(dense_dataset.ocean_threshold),
                'target_lon_range': [float(value) for value in target_lon_range],
                'target_lat_range': [float(value) for value in target_lat_range],
                'inference_stride_lon': float(stride_lon),
                'inference_stride_lat': float(stride_lat),
            }
        finally:
            if getattr(dense_dataset, 'dataset', None) is not None:
                dense_dataset.dataset.close()
            if ref_dataset is not None and getattr(ref_dataset, 'dataset', None) is not None:
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
        if not hasattr(dataset, 'inverse_transform_targets'):
            raise RuntimeError('预测 dataset 不支持严格物理量恢复')
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
        except Exception as exc:
            raise RuntimeError('目标物理量恢复失败；拒绝改变指标口径后继续') from exc
        predictions_original = [
            torch.from_numpy(arr.astype(np.float32)) for arr in pred_original
        ]
        targets_original = [
            torch.from_numpy(arr.astype(np.float32)) for arr in target_original
        ]
        print("反标准化完成")
        return predictions_original, targets_original
    
    def _save_results(
        self,
        predictions,
        targets,
        inputs_list,
        dataset,
        sample_indices,
        dynaseaf_diagnostics: Optional[Dict[str, List[torch.Tensor]]] = None,
    ):
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
            inputs=inputs_np,
            sample_indices=np.asarray(sample_indices, dtype=np.int64),
        )

        diagnostics_file = None
        if dynaseaf_diagnostics:
            diagnostics_arrays = {}
            for name, values in dynaseaf_diagnostics.items():
                if values:
                    diagnostics_arrays[name] = torch.stack(values).numpy()
            if diagnostics_arrays:
                diagnostics_file = 'dynaseaf_diagnostics.npz'
                np.savez_compressed(
                    os.path.join(self.output_dir, diagnostics_file),
                    sample_indices=np.asarray(sample_indices, dtype=np.int64),
                    **diagnostics_arrays,
                )
        
        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # 保存模型路径信息
        provenance = []
        for sample_idx in sample_indices:
            start_idx, region_idx = dataset.sequences[int(sample_idx)]
            region = dataset.all_regions_data[int(region_idx)]
            target_start = int(start_idx + dataset.sequence_length)
            provenance.append({
                'sample_index': int(sample_idx),
                'region_index': int(region_idx),
                'history_start_index': int(start_idx),
                'history_end_index': target_start - 1,
                'forecast_start_index': target_start,
                'forecast_end_index': target_start + int(dataset.prediction_length) - 1,
                'lon_range': [float(value) for value in region.get('lon_range', [])],
                'lat_range': [float(value) for value in region.get('lat_range', [])],
            })

        info = {
            'model_path': self.model_path,
            'model_index': self.model_index,
            'model_label': self.model_label,
            'model_dir': self.model_dir,
            'data_path': self.config['data_path'],
            'num_samples': len(predictions),
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction_units': 'physical',
            'input_space': 'normalized model features',
            'sample_provenance': provenance,
        }
        if diagnostics_file is not None:
            info['dynaseaf_diagnostics_file'] = diagnostics_file
            info['dynaseaf_diagnostics_keys'] = sorted(
                dynaseaf_diagnostics.keys()
            )
        
        with open(os.path.join(self.output_dir, 'info.json'), 'w') as f:
            json.dump(info, f, indent=4)
    
    @staticmethod
    def _variable_plot_style(variable):
        styles = {
            'TEMP': ('Temperature', '°C', 'RdYlBu_r'),
            'SALT': ('Salinity', 'PSU', 'viridis'),
        }
        return styles.get(variable, (variable, 'physical units', 'viridis'))

    def _visualize_results(
        self,
        predictions,
        targets,
        metrics,
        dataset,
        sample_indices,
    ):
        """Generate schema-safe plots on each sample's real geographic window."""
        print("生成可追溯的分变量可视化结果...")
        if dataset is None:
            raise ValueError('可视化需要 dataset 以恢复每个窗口的真实坐标')
        if len(predictions) != len(targets) or len(predictions) != len(sample_indices):
            raise ValueError('可视化的预测、目标与 sample_indices 数量不一致')
        if not predictions:
            return

        total_channels = int(predictions[0].shape[1])
        target_variables = list(self.config.get('target_variables', []))
        channel_slices = resolve_variable_slices(
            target_variables,
            getattr(dataset, 'target_channel_slices', {}),
            total_channels,
        )
        physical_by_variable = metrics.get('physical_report', {}).get('by_variable', {})

        for output_idx in range(min(3, len(predictions))):
            sample_index = int(sample_indices[output_idx])
            if sample_index < 0 or sample_index >= len(dataset.sequences):
                raise IndexError(f'可视化 sample_index 超出数据集范围: {sample_index}')
            _, region_idx = dataset.sequences[sample_index]
            region = dataset.all_regions_data[int(region_idx)]
            coords = region.get('coords', {})
            lons = np.asarray(coords.get('lons'))
            lats = np.asarray(coords.get('lats'))
            height, width = predictions[output_idx].shape[-2:]
            if lons.shape != (width,) or lats.shape != (height,):
                raise ValueError(
                    f'样本 {sample_index} 坐标形状与窗口不一致: '
                    f'lons={lons.shape}, lats={lats.shape}, grid={(height, width)}'
                )

            for variable, ch_slice in channel_slices.items():
                display_name, unit, cmap = self._variable_plot_style(variable)
                lead_count = min(3, int(predictions[output_idx].shape[0]))
                fig, axes = plt.subplots(
                    2, lead_count,
                    figsize=(5.2 * lead_count, 8.5),
                    squeeze=False,
                )
                fig.suptitle(
                    f'{display_name} depth-mean — sample {sample_index}',
                    fontsize=14,
                )
                variable_metrics = physical_by_variable.get(variable, {})
                metric_lines = []
                for label, key in (('RMSE', 'rmse'), ('MAE', 'mae'), ('R²', 'r2')):
                    value = variable_metrics.get(key)
                    metric_lines.append(
                        f'{label}: {value:.5g}' if value is not None else f'{label}: n/a'
                    )
                fig.text(
                    0.01, 0.98, '\n'.join(metric_lines),
                    va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
                )

                for lead_idx in range(lead_count):
                    pred_map = predictions[output_idx][lead_idx, ch_slice].mean(dim=0).numpy()
                    target_map = targets[output_idx][lead_idx, ch_slice].mean(dim=0).numpy()
                    finite = np.concatenate((
                        pred_map[np.isfinite(pred_map)],
                        target_map[np.isfinite(target_map)],
                    ))
                    vmin = float(finite.min()) if finite.size else 0.0
                    vmax = float(finite.max()) if finite.size else 1.0
                    if vmax <= vmin:
                        vmax = vmin + 1e-12
                    for row, values, label in (
                        (0, pred_map, 'Prediction'),
                        (1, target_map, 'Target'),
                    ):
                        image_artist = axes[row, lead_idx].pcolormesh(
                            lons, lats, values,
                            shading='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                        )
                        axes[row, lead_idx].set_title(
                            f'{label}, lead {lead_idx + 1}'
                        )
                        axes[row, lead_idx].set_xlabel('Longitude')
                        axes[row, lead_idx].set_ylabel('Latitude')
                        plt.colorbar(
                            image_artist,
                            ax=axes[row, lead_idx],
                            shrink=0.8,
                            label=unit,
                        )
                plt.tight_layout()
                safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', variable)
                plt.savefig(
                    os.path.join(
                        self.output_dir,
                        f'{safe_name}_prediction_sample_{sample_index}.png',
                    ),
                    dpi=150,
                    bbox_inches='tight',
                )
                plt.close(fig)

        self._plot_variable_lead_errors(
            predictions, targets, dataset, channel_slices
        )
        self._plot_geographic_error(
            predictions, targets, dataset, sample_indices, channel_slices
        )

    def _plot_variable_lead_errors(
        self,
        predictions,
        targets,
        dataset,
        channel_slices,
    ):
        """Plot lead-wise physical errors separately for every target variable."""
        del dataset
        pred_stack = torch.stack(predictions)
        target_stack = torch.stack(targets)
        fig, axes = plt.subplots(
            len(channel_slices), 2,
            figsize=(13, 4.5 * len(channel_slices)),
            squeeze=False,
        )
        saved = {}
        for row, (variable, ch_slice) in enumerate(channel_slices.items()):
            _, unit, _ = self._variable_plot_style(variable)
            error = pred_stack[:, :, ch_slice] - target_stack[:, :, ch_slice]
            mse = torch.mean(error ** 2, dim=(0, 2, 3, 4)).numpy()
            mae = torch.mean(torch.abs(error), dim=(0, 2, 3, 4)).numpy()
            leads = np.arange(1, len(mse) + 1)
            axes[row, 0].plot(leads, mse, 'o-', linewidth=2)
            axes[row, 0].set_title(f'{variable} MSE by lead')
            axes[row, 0].set_ylabel(f'MSE [{unit}²]')
            axes[row, 1].plot(leads, mae, 'o-', linewidth=2)
            axes[row, 1].set_title(f'{variable} MAE by lead')
            axes[row, 1].set_ylabel(f'MAE [{unit}]')
            for axis in axes[row]:
                axis.set_xlabel('Forecast lead')
                axis.set_xticks(leads)
                axis.grid(True, alpha=0.3)
            saved[f'{variable}_mse_by_lead'] = mse
            saved[f'{variable}_mae_by_lead'] = mae
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, 'variable_error_analysis.png'),
            dpi=150,
            bbox_inches='tight',
        )
        plt.close(fig)
        np.savez_compressed(
            os.path.join(self.output_dir, 'variable_error_by_lead.npz'),
            **saved,
        )

    def _plot_geographic_error(
        self,
        predictions,
        targets,
        dataset,
        sample_indices,
        channel_slices,
    ):
        """Accumulate sampled-window errors on the actual global grid."""
        global_lons = np.asarray(dataset.lons)
        global_lats = np.asarray(dataset.lats)
        grid_shape = (len(global_lats), len(global_lons))
        squared_error = {
            variable: np.zeros(grid_shape, dtype=np.float64)
            for variable in channel_slices
        }
        finite_count = {
            variable: np.zeros(grid_shape, dtype=np.int64)
            for variable in channel_slices
        }

        for output_idx, sample_index in enumerate(sample_indices):
            _, region_idx = dataset.sequences[int(sample_index)]
            coords = dataset.all_regions_data[int(region_idx)].get('coords', {})
            window_lons = np.asarray(coords.get('lons'))
            window_lats = np.asarray(coords.get('lats'))
            lon_indices = np.searchsorted(global_lons, window_lons)
            lat_indices = np.searchsorted(global_lats, window_lats)
            if (
                np.any(lon_indices >= len(global_lons))
                or np.any(lat_indices >= len(global_lats))
                or not np.allclose(global_lons[lon_indices], window_lons)
                or not np.allclose(global_lats[lat_indices], window_lats)
            ):
                raise ValueError(f'样本 {sample_index} 的窗口坐标不能映射到全局网格')
            grid_index = np.ix_(lat_indices, lon_indices)
            for variable, ch_slice in channel_slices.items():
                difference = (
                    predictions[output_idx][:, ch_slice]
                    - targets[output_idx][:, ch_slice]
                ).numpy()
                finite = np.isfinite(difference)
                sample_sse = np.where(finite, difference ** 2, 0.0).sum(axis=(0, 1))
                sample_count = finite.sum(axis=(0, 1))
                squared_error[variable][grid_index] += sample_sse
                finite_count[variable][grid_index] += sample_count

        fig, axes = plt.subplots(
            1, len(channel_slices),
            figsize=(7 * len(channel_slices), 4.5),
            squeeze=False,
        )
        saved = {
            'lons': global_lons,
            'lats': global_lats,
            'sample_indices': np.asarray(sample_indices, dtype=np.int64),
        }
        for col, variable in enumerate(channel_slices):
            rmse = np.full(grid_shape, np.nan, dtype=np.float32)
            covered = finite_count[variable] > 0
            rmse[covered] = np.sqrt(
                squared_error[variable][covered] / finite_count[variable][covered]
            )
            _, unit, cmap = self._variable_plot_style(variable)
            artist = axes[0, col].pcolormesh(
                global_lons, global_lats, rmse,
                shading='auto', cmap=cmap,
            )
            axes[0, col].set_title(
                f'{variable} RMSE on sampled geographic cells'
            )
            axes[0, col].set_xlabel('Longitude')
            axes[0, col].set_ylabel('Latitude')
            plt.colorbar(artist, ax=axes[0, col], shrink=0.8, label=unit)
            saved[f'{variable}_rmse'] = rmse
            saved[f'{variable}_finite_count'] = finite_count[variable]
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, 'sampled_geographic_error.png'),
            dpi=150,
            bbox_inches='tight',
        )
        plt.close(fig)
        np.savez_compressed(
            os.path.join(self.output_dir, 'sampled_geographic_metrics.npz'),
            **saved,
        )

    def _legacy_visualize_results(self, predictions, targets, inputs_list, metrics, dataset=None):
        """可视化预测结果"""
        raise RuntimeError('旧版等分目标通道可视化已停用，请使用 schema-safe 可视化')
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
        self._legacy_plot_error_analysis(predictions, targets)
        
        # 生成温度和盐度的单独误差分析
        self._legacy_plot_variable_error_analysis(predictions, targets)

        # 生成按空间区域的误差热力图
        self._legacy_plot_regional_error(predictions, targets, dataset)

    def _legacy_plot_regional_error(self, predictions, targets, dataset=None):
        """绘制空间各区域（格点）RMSE热力图，并保存矩阵数据"""
        raise RuntimeError('旧版窗口相对坐标误差图已停用，请使用地理网格聚合')
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
    
    def _legacy_plot_error_analysis(self, predictions, targets):
        """绘制误差分析图"""
        raise RuntimeError('物理变量禁止生成混合单位整体误差图')
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
    
    def _legacy_plot_variable_error_analysis(self, predictions, targets):
        """绘制温度和盐度的单独误差分析图"""
        raise RuntimeError('旧版等分目标通道误差图已停用，请使用显式 channel schema')
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
        num_channels = pred_stack.shape[2]
        channel_slices = getattr(dataset, 'target_channel_slices', {}) if dataset is not None else self.target_channel_slices
        if not channel_slices:
            channel_slices = self.target_channel_slices
        resolved_slices = resolve_variable_slices(
            target_variables,
            channel_slices,
            num_channels,
        )

        if dataset is None or not hasattr(dataset, 'build_reference_forecasts'):
            raise RuntimeError('预测 dataset 不支持冻结基线构造')
        try:
            physical_baselines = dataset.build_reference_forecasts(
                sample_indices=sample_indices,
                spaces=('physical',),
            ).get('physical', {})
        except Exception as exc:
            raise RuntimeError('物理量冻结基线构造失败；拒绝输出不完整指标') from exc

        pred_np = pred_stack.numpy()
        target_np = target_stack.numpy()
        raw_levels = getattr(dataset, 'levels', None) if dataset is not None else None
        depth_values = (
            [float(value) for value in np.asarray(raw_levels).reshape(-1)]
            if raw_levels is not None else None
        )
        physical_report = compute_metric_report(
            pred_np,
            target_np,
            target_variables,
            channel_slices=channel_slices,
            baselines=physical_baselines,
            metric_space='physical',
            depth_values=depth_values,
        )

        normalized_report = None
        if normalized_predictions is not None and normalized_targets is not None:
            norm_pred_np = torch.stack(normalized_predictions).numpy()
            norm_target_np = torch.stack(normalized_targets).numpy()
            try:
                normalized_baselines = dataset.build_reference_forecasts(
                    sample_indices=sample_indices,
                    spaces=('normalized',),
                ).get('normalized', {})
            except Exception as exc:
                raise RuntimeError(
                    '标准化冻结基线构造失败；拒绝输出不完整指标'
                ) from exc
            normalized_report = compute_metric_report(
                norm_pred_np,
                norm_target_np,
                target_variables,
                channel_slices=channel_slices,
                baselines=normalized_baselines,
                metric_space='normalized',
                depth_values=depth_values,
                include_depth=False,
            )

        # 加权损失必须在训练使用的 normalized/anomaly 空间计算。
        loss_pred_stack = torch.stack(normalized_predictions) if normalized_predictions is not None else pred_stack
        loss_target_stack = torch.stack(normalized_targets) if normalized_targets is not None else target_stack
        raw_weights = self.config.get('target_loss_weights', {})
        weight_sum = sum(float(raw_weights.get(name, 0.0)) for name in resolved_slices)
        variable_weights = {
            name: (
                float(raw_weights.get(name, 0.0)) / weight_sum
                if weight_sum > 0 else 1.0 / len(resolved_slices)
            )
            for name in resolved_slices
        }
        variable_losses = {}
        weighted_loss = 0.0
        for var_name, ch_slice in resolved_slices.items():
            var_loss = torch.mean(
                (loss_pred_stack[:, :, ch_slice, :, :] - loss_target_stack[:, :, ch_slice, :, :]) ** 2
            ).item()
            variable_losses[var_name] = var_loss
            weighted_loss += variable_weights[var_name] * var_loss

        # 跨变量整体指标只取单位一致的报告；多物理变量时使用 normalized/anomaly。
        physical_overall = physical_report.get('overall')
        aggregate_report = physical_report if physical_overall is not None else normalized_report
        aggregate_metrics = (aggregate_report or {}).get('overall') or {}
        
        metrics = {
            'overall': {
                'MSE': _to_serializable(aggregate_metrics.get('mse')),
                'MAE': _to_serializable(aggregate_metrics.get('mae')),
                'RMSE': _to_serializable(aggregate_metrics.get('rmse')),
                'Correlation': _to_serializable(aggregate_metrics.get('correlation')),
                'R2': _to_serializable(aggregate_metrics.get('r2')),
                'metric_space': 'physical' if physical_overall is not None else 'normalized/anomaly',
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
        metrics['weighted'] = {
            'Weighted_MSE': _to_serializable(weighted_loss),
            'metric_space': 'normalized/anomaly' if normalized_predictions is not None else 'physical',
            'by_variable': {
                var_name: {
                    'MSE': _to_serializable(variable_losses[var_name]),
                    'weight': _to_serializable(variable_weights[var_name]),
                }
                for var_name in variable_losses
            },
        }
        
        # 分别计算温度和盐度的指标
        # 按变量分别计算指标
        for var_name_key in resolved_slices:
            var_report = physical_report['by_variable'][var_name_key]

            name_map = {
                'TEMP': 'temperature',
                'SALT': 'salinity'
            }
            metrics_key = name_map.get(var_name_key, var_name_key.lower())
            metrics[metrics_key] = {
                'MSE': _to_serializable(var_report.get('mse')),
                'MAE': _to_serializable(var_report.get('mae')),
                'RMSE': _to_serializable(var_report.get('rmse')),
                'Correlation': _to_serializable(var_report.get('correlation')),
                'R2': _to_serializable(var_report.get('r2')),
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

        primary_report = metrics.get('physical_report') or normalized_report or {}
        macro_r2 = (
            primary_report.get('macro_field', {})
            .get('dimensionless_overall', {})
            .get('r2', {})
            .get('mean')
        )
        print("  去背景稳健指标:")
        print(f"    Macro-field R2 mean: {macro_r2:.6f}" if macro_r2 is not None else "    Macro-field R2 mean: nan")
        spatial_by_var = primary_report.get('spatial_mean_removed', {}).get('by_variable', {})
        clim_by_var = primary_report.get('climatology_residual', {}).get('by_variable', {})
        for var_name in target_variables:
            spatial_r2 = spatial_by_var.get(var_name, {}).get('r2')
            clim_metrics = clim_by_var.get(var_name, {})
            spatial_text = f"{spatial_r2:.6f}" if spatial_r2 is not None else "nan"
            clim_r2 = clim_metrics.get('r2')
            clim_corr = clim_metrics.get('correlation')
            clim_r2_text = f"{clim_r2:.6f}" if clim_r2 is not None else "nan"
            clim_corr_text = f"{clim_corr:.6f}" if clim_corr is not None else "nan"
            print(f"    [{var_name}] Spatial-mean-removed R2: {spatial_text}")
            if clim_metrics:
                print(f"    [{var_name}] Climatology-residual Corr/R2: {clim_corr_text} / {clim_r2_text}")

        print("  Baseline 对比 (按变量计算，skill > 0 表示优于基线):")
        for space_name, comparisons in metrics['baseline_comparison'].items():
            if not comparisons:
                continue
            print(f"    [{space_name}]")
            for baseline_name, comparison in comparisons.items():
                macro_skill = comparison.get('macro', {}).get('mse_skill', {}).get('mean')
                macro_text = f"{macro_skill:.6f}" if macro_skill is not None else "nan"
                print(f"      {baseline_name}: macro MSE skill={macro_text}")
                for var_name, values in comparison.get('by_variable', {}).items():
                    skill = values.get('mse_skill')
                    improvement = values.get('rmse_improvement_pct')
                    skill_text = f"{skill:.6f}" if skill is not None else "nan"
                    improvement_text = f"{improvement:.2f}%" if improvement is not None else "nan"
                    print(f"        [{var_name}] skill={skill_text}, RMSE improvement={improvement_text}")
        
        print(f"  样本数量: {metrics['num_samples']}")
        
        return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ConvLSTM 海洋预测脚本")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', type=int, help='选择 outputs/results 下模型编号')
    model_group.add_argument('--model_dir', type=str, help='直接指定确定性的训练结果目录')
    parser.add_argument('--samples', type=int, default=None, help='预测样本数（覆盖配置）')
    parser.add_argument('--output_dir', type=str, default=None, help='自定义预测输出目录')
    parser.add_argument(
        '--save-dynaseaf-diagnostics',
        action='store_true',
        help='DynaSEAF 额外导出 direct/transport/gate/innovation/dynamics 分解',
    )
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
        predictor = SmartOceanPredictor(
            model_index=args.model,
            model_dir=args.model_dir,
            config=config,
            output_dir=args.output_dir,
        )

        # 进行预测
        predictor.predict(
            num_samples=args.samples,
            save_dynaseaf_diagnostics=args.save_dynaseaf_diagnostics,
        )

        print("预测任务完成!")

    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
