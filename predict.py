#!/usr/bin/env python3
"""
ConvLSTM海洋预测脚本 - 统一配置版本
自动匹配模型结构和权重，生成可视化预测结果
使用统一配置文件确保与训练脚本参数一致
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import xarray as xr
import os
import json
import glob
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入统一配置
from config import DEFAULT_CONFIG, load_config, merge_configs, save_config
from convlstm_model import OceanConvLSTMPredictor
from data_loader import OceanDataset

# 设置matplotlib以支持中文显示
def setup_chinese_fonts():
    """设置中文字体支持"""
    import matplotlib
    
    # 设置matplotlib使用Agg后端（无GUI，适合服务器环境）
    matplotlib.use('Agg')
    
    # 获取系统所有可用字体
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    print(f"系统总字体数: {len(available_fonts)}")
    
    # 中文字体候选列表（按优先级排序）- 现在包含新安装的字体
    chinese_fonts = [
        'Noto Sans CJK SC',        # Google Noto 简体中文 (新安装)
        'WenQuanYi Micro Hei',     # 文泉驿微米黑 (新安装)
        'WenQuanYi Zen Hei',       # 文泉驿正黑 (新安装)
        'SimHei',                  # 黑体 (Windows)
        'Microsoft YaHei',         # 微软雅黑 (Windows)
        'Source Han Sans SC',      # 思源黑体 (Adobe)
        'Hiragino Sans GB',        # 冬青黑体 (macOS)
        'PingFang SC',             # 苹方 (macOS)
        'Arial Unicode MS',        # Unicode 字体 (macOS)
        'STHeiti',                 # 华文黑体
        'STSong',                  # 华文宋体
        'Liberation Sans',         # 自由字体
        'FreeSans',               # GNU字体
        'DejaVu Sans'              # 备用字体
    ]
    
    # 查找可用的中文字体
    found_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            found_font = font
            break
    
    # 设置字体
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font] + chinese_fonts
        print(f"✓ 找到并设置中文字体: {found_font}")
    else:
        # 如果没有找到专门的中文字体，使用Unicode编码
        plt.rcParams['font.sans-serif'] = chinese_fonts
        print("⚠ 未找到专门的中文字体，使用Unicode编码显示")
    
    # 重要：设置负号正确显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置字体编码为UTF-8
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 设置字体大小
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    return found_font

# 初始化中文字体
setup_chinese_fonts()

class SmartOceanPredictor:
    """智能海洋预测器 - 使用统一配置自动匹配模型"""
    
    def __init__(self, config: Optional[Dict] = None, output_dir: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
            output_dir: 输出目录，默认自动生成
        """
        # 使用统一配置
        if config is None:
            self.config = DEFAULT_CONFIG.copy()
        else:
            self.config = config
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime(self.config['timestamp_format'])
            self.output_dir = os.path.join(self.config['predictions_dir'], f"predictions_{timestamp}")
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 自动查找并加载最佳匹配的模型
        self.model_path, self.model_config = self._find_best_model()
        print(f"使用模型: {self.model_path}")
        print(f"模型配置与统一配置合并完成")
        
        # 加载模型
        self._load_model()
        
        print(f"预测结果将保存到: {self.output_dir}")
    
    def _find_best_model(self) -> Tuple[str, Dict]:
        """查找最佳匹配的模型和配置"""
        print("正在搜索可用的模型...")
        
        # 查找所有模型文件
        model_patterns = [
            f"{self.config['results_dir']}/**/{self.config['model_filename']}",
            f"{self.config['results_dir']}/**/{self.config['checkpoint_filename']}",
            f"{self.config['results_dir']}/**/*.pth"
        ]
        
        model_candidates = []
        for pattern in model_patterns:
            model_candidates.extend(glob.glob(pattern, recursive=True))
        
        if not model_candidates:
            raise FileNotFoundError("未找到任何模型文件(.pth)")
        
        # 按修改时间排序，优先使用最新的
        model_candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # 尝试每个模型，找到第一个能成功加载的
        for model_path in model_candidates:
            print(f"尝试模型: {model_path}")
            
            # 查找对应的config.json
            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, self.config['config_filename'])
            
            if not os.path.exists(config_path):
                print(f"  跳过: 未找到配置文件 {config_path}")
                continue
            
            try:
                # 读取模型配置
                model_config = load_config(config_path)
                
                # 合并模型配置和统一配置
                merged_config = merge_configs(model_config, self.config)
                
                # 尝试创建模型并加载权重
                test_model = OceanConvLSTMPredictor(merged_config)
                checkpoint = torch.load(model_path, map_location='cpu')
                
                if 'model_state_dict' in checkpoint:
                    test_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    test_model.load_state_dict(checkpoint)
                
                print(f"  成功: 模型结构匹配")
                
                # 更新当前配置为合并后的配置
                self.config = merged_config
                return model_path, merged_config
                
            except Exception as e:
                print(f"  失败: {str(e)[:100]}...")
                continue
        
        raise RuntimeError("未找到任何可用的模型，请检查模型文件和配置是否匹配")
    
    def _load_model(self):
        """加载模型"""
        print("加载模型...")
        
        # 创建模型
        self.model = OceanConvLSTMPredictor(self.config).to(self.device)
        
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
    
    def predict(self, num_samples: Optional[int] = None):
        """
        进行预测
        
        Args:
            num_samples: 预测样本数量，如果为None则使用配置中的默认值
        """
        if num_samples is None:
            num_samples = self.config['num_samples']
            
        print(f"开始预测 {num_samples} 个样本...")
        
        # 创建测试数据集
        test_dataset = OceanDataset(self.config['data_path'], self.config, mode='test')
        
        if len(test_dataset) == 0:
            print("测试集为空，无法进行预测")
            return
        
        # 限制样本数量
        num_samples = min(num_samples, len(test_dataset))
        
        predictions = []
        targets = []
        inputs_list = []
        
        print(f"数据集大小: {len(test_dataset)}")
        print(f"实际预测样本数: {num_samples}")
        
        with torch.no_grad():
            for i in range(num_samples):
                inputs, target = test_dataset[i]
                inputs = inputs.unsqueeze(0).to(self.device)  # 添加batch维度
                
                # 模型预测
                output = self.model(inputs)
                
                # 检查输出是否有效
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"警告: 样本 {i} 的预测结果包含NaN或Inf，跳过")
                    continue
                
                # 转换回CPU并保存
                predictions.append(output.cpu().squeeze(0))
                targets.append(target)
                inputs_list.append(inputs.cpu().squeeze(0))
                
                if (i + 1) % 5 == 0:
                    print(f"已完成 {i + 1}/{num_samples} 个样本的预测")
        
        if len(predictions) == 0:
            print("所有预测结果都包含异常值，无法生成有效预测")
            return
        
        print(f"成功预测 {len(predictions)} 个样本")
        
        # 反标准化预测结果和目标
        predictions_original, targets_original = self._denormalize_predictions(predictions, targets, test_dataset)
        
        # 计算评估指标（使用反标准化后的数据）
        metrics = self._compute_metrics(predictions_original, targets_original)
        
        # 保存和可视化结果
        self._save_results(predictions_original, targets_original, inputs_list, test_dataset)
        self._visualize_results(predictions_original, targets_original, inputs_list, metrics, test_dataset)
        
        print(f"预测完成！结果保存在: {self.output_dir}")
    
    def _denormalize_predictions(self, predictions, targets, dataset):
        """反标准化预测结果和目标数据"""
        print("反标准化预测结果...")
        
        if not hasattr(dataset, 'scalers') or not dataset.scalers:
            print("警告: 未找到标准化器，使用原始数据")
            return predictions, targets
        
        predictions_original = []
        targets_original = []
        
        target_variables = self.config['target_variables']
        total_channels = predictions[0].shape[1]  # 获取总通道数
        channels_per_var = total_channels // len(target_variables)  # 每个变量的通道数
        
        for i in range(len(predictions)):
            pred = predictions[i].clone()  # (pred_len, channels, height, width)
            target = targets[i].clone()
            
            # 对每个目标变量进行反标准化
            for var_idx, var_name in enumerate(target_variables):
                if var_name in dataset.scalers:
                    scaler = dataset.scalers[var_name]
                    
                    # 计算该变量对应的通道范围
                    start_ch = var_idx * channels_per_var
                    end_ch = (var_idx + 1) * channels_per_var
                    
                    # 反标准化预测结果
                    pred_var = pred[:, start_ch:end_ch, :, :]  # 选择该变量的所有通道
                    original_shape = pred_var.shape
                    pred_2d = pred_var.reshape(-1, 1)
                    pred_denorm = scaler.inverse_transform(pred_2d)
                    pred[:, start_ch:end_ch, :, :] = torch.tensor(pred_denorm.reshape(original_shape), dtype=pred.dtype)
                    
                    # 反标准化目标数据
                    target_var = target[:, start_ch:end_ch, :, :]
                    original_shape = target_var.shape
                    target_2d = target_var.reshape(-1, 1)
                    target_denorm = scaler.inverse_transform(target_2d)
                    target[:, start_ch:end_ch, :, :] = torch.tensor(target_denorm.reshape(original_shape), dtype=target.dtype)
            
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
                metrics_text = f"MSE: {temp_metrics['MSE']:.6f}\nMAE: {temp_metrics['MAE']:.6f}\nRMSE: {temp_metrics['RMSE']:.6f}\nCorr: {temp_metrics['Correlation']:.6f}"
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
                    metrics_text = f"MSE: {salt_metrics['MSE']:.6f}\nMAE: {salt_metrics['MAE']:.6f}\nRMSE: {salt_metrics['RMSE']:.6f}\nCorr: {salt_metrics['Correlation']:.6f}"
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
            metrics_text = f"整体指标:\nMSE: {overall_metrics['MSE']:.6f}\nMAE: {overall_metrics['MAE']:.6f}\nRMSE: {overall_metrics['RMSE']:.6f}\nCorr: {overall_metrics['Correlation']:.6f}"
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
    
    def _compute_metrics(self, predictions, targets):
        """计算评估指标"""
        print("计算评估指标...")
        
        pred_stack = torch.stack(predictions)
        target_stack = torch.stack(targets)
        
        # 获取损失权重
        temp_weight = self.config.get('temp_weight', 0.7)
        salt_weight = self.config.get('salt_weight', 0.3)
        
        # 计算加权损失
        num_channels = pred_stack.shape[2]
        temp_channels = num_channels // 2
        
        weighted_loss = 0.0
        temp_loss = 0.0
        salt_loss = 0.0
        
        if temp_channels > 0:
            pred_temp = pred_stack[:, :, :temp_channels, :, :]
            target_temp = target_stack[:, :, :temp_channels, :, :]
            temp_loss = torch.mean((pred_temp - target_temp) ** 2).item()
            weighted_loss += temp_weight * temp_loss
        
        if temp_channels < num_channels:
            pred_salt = pred_stack[:, :, temp_channels:, :, :]
            target_salt = target_stack[:, :, temp_channels:, :, :]
            salt_loss = torch.mean((pred_salt - target_salt) ** 2).item()
            weighted_loss += salt_weight * salt_loss
        
        # 整体指标
        mse = torch.mean((pred_stack - target_stack) ** 2).item()
        mae = torch.mean(torch.abs(pred_stack - target_stack)).item()
        rmse = np.sqrt(mse)
        
        # 计算相关系数
        pred_flat = pred_stack.reshape(-1).numpy()
        target_flat = target_stack.reshape(-1).numpy()
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        
        metrics = {
            'overall': {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'Correlation': correlation
            },
            'weighted': {
                'Weighted_MSE': weighted_loss,
                'Temp_MSE': temp_loss,
                'Salt_MSE': salt_loss,
                'Temp_Weight': temp_weight,
                'Salt_Weight': salt_weight
            },
            'num_samples': len(predictions)
        }
        
        # 分别计算温度和盐度的指标
        num_channels = pred_stack.shape[2]
        temp_channels = num_channels // 2
        
        if temp_channels > 0:
            # 温度指标
            pred_temp = pred_stack[:, :, :temp_channels, :, :]
            target_temp = target_stack[:, :, :temp_channels, :, :]
            
            temp_mse = torch.mean((pred_temp - target_temp) ** 2).item()
            temp_mae = torch.mean(torch.abs(pred_temp - target_temp)).item()
            temp_rmse = np.sqrt(temp_mse)
            
            pred_temp_flat = pred_temp.reshape(-1).numpy()
            target_temp_flat = target_temp.reshape(-1).numpy()
            temp_correlation = np.corrcoef(pred_temp_flat, target_temp_flat)[0, 1]
            
            metrics['temperature'] = {
                'MSE': temp_mse,
                'MAE': temp_mae,
                'RMSE': temp_rmse,
                'Correlation': temp_correlation
            }
        
        if temp_channels < num_channels:
            # 盐度指标
            pred_salt = pred_stack[:, :, temp_channels:, :, :]
            target_salt = target_stack[:, :, temp_channels:, :, :]
            
            salt_mse = torch.mean((pred_salt - target_salt) ** 2).item()
            salt_mae = torch.mean(torch.abs(pred_salt - target_salt)).item()
            salt_rmse = np.sqrt(salt_mse)
            
            pred_salt_flat = pred_salt.reshape(-1).numpy()
            target_salt_flat = target_salt.reshape(-1).numpy()
            salt_correlation = np.corrcoef(pred_salt_flat, target_salt_flat)[0, 1]
            
            metrics['salinity'] = {
                'MSE': salt_mse,
                'MAE': salt_mae,
                'RMSE': salt_rmse,
                'Correlation': salt_correlation
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
        
        print(f"  样本数量: {metrics['num_samples']}")
        
        return metrics


def main():
    """主函数"""
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
        predictor = SmartOceanPredictor(config)
        
        # 进行预测
        predictor.predict()
        
        print("预测任务完成!")
        
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
