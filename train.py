"""
海洋数据ConvLSTM模型训练脚本
使用统一配置文件确保参数一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
import random
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入统一配置
from config import DEFAULT_CONFIG, save_config, validate_config, update_config
from convlstm_model import create_ocean_model
from data_loader import create_data_loaders
from font_config import setup_chinese_fonts

# 初始化中文字体
setup_chinese_fonts()


def _sanitize_note_for_path(note: str, max_length: int = 40) -> str:
    """将备注内容转换为适合路径命名的短标识"""
    if not note:
        return ""
    cleaned = note.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[\\/:*?\"<>|]", "-", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("-_")
    if not cleaned:
        return ""
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("-_")
    return cleaned

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OceanModelTrainer:
    """
    海洋模型训练器
    """

    def __init__(self, config: dict = None):
        """
        初始化训练器

        Args:
            config: 统一配置字典，如果为None则使用默认配置
        """
        if config is None:
            config = DEFAULT_CONFIG

        self.config = config

        # Set seed for reproducibility (if specified in config)
        seed = config.get('seed', None)
        if seed is not None:
            set_seed(seed)
        elif config.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        self.ensemble_enabled = config.get('enable_arima_xgboost', False)
        if self.ensemble_enabled:
            print("Warning: ARIMA-XGBoost ensemble is no longer supported. Ignoring.")
            self.ensemble_enabled = False
        
        # 设置设备
        if config['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config['device'])
        
        # 更新配置中的设备，确保后续模型初始化使用正确的设备字符串
        config['device'] = str(self.device)
        print(f"使用设备: {self.device}")
        
        # 设置性能优化（仅在未指定随机种子时启用，以确保可复现性）
        if config.get('cudnn_benchmark', False) and config.get('seed') is None:
            torch.backends.cudnn.benchmark = True
            print("启用cuDNN benchmark")
        
        # 创建数据加载器
        print("创建数据加载器...")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            config['data_path'], 
            config, 
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            persistent_workers=config.get('persistent_workers', False),
            prefetch_factor=config.get('prefetch_factor', 2),
        )
        
        # 获取第一个批次来确定实际的输入输出维度
        print("检测数据维度...")
        sample_batch = next(iter(self.train_loader))
        if isinstance(sample_batch, (list, tuple)):
            if len(sample_batch) < 2:
                raise ValueError("训练数据批次缺少输入或目标张量")
            sample_input = sample_batch[0]
            sample_target = sample_batch[1]
        else:
            raise TypeError("训练数据批次类型不受支持，期望为tuple或list")
        actual_input_channels = sample_input.shape[2]  # [batch, seq, channels, height, width]
        actual_output_channels = sample_target.shape[2]  # [batch, pred, channels, height, width]
        
        print(f"实际输入通道数: {actual_input_channels}")
        print(f"实际输出通道数: {actual_output_channels}")
        
        # 更新配置
        config['actual_input_channels'] = actual_input_channels
        config['actual_output_channels'] = actual_output_channels

        def _serialize_channel_slices(slice_map):
            serialized = {}
            for name, channel_slice in getattr(slice_map, "items", lambda: [])():
                if isinstance(channel_slice, slice):
                    serialized[name] = [channel_slice.start, channel_slice.stop]
            return serialized

        train_dataset = getattr(self.train_loader, 'dataset', None)
        if train_dataset is not None:
            input_slices = _serialize_channel_slices(getattr(train_dataset, 'input_channel_slices', {}))
            target_slices = _serialize_channel_slices(getattr(train_dataset, 'target_channel_slices', {}))
            if input_slices:
                config['input_channel_slices'] = input_slices
            if target_slices:
                config['target_channel_slices'] = target_slices
        
        # 创建模型
        self.model = create_ocean_model(config).to(self.device)
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 设置损失函数和优化器
        self.criterion = nn.MSELoss()
        
        # 获取损失权重
        self.temp_weight = config.get('temp_weight', 0.7)
        self.salt_weight = config.get('salt_weight', 0.3)
        
        # 计算每个变量的通道数
        self.target_variables = config['target_variables']
        self.channels_per_var = None  # 将在第一次前向传播时确定

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config['scheduler_factor'], 
            patience=config['scheduler_patience'], 
            min_lr=config['min_lr']
        )
        # 温度收敛后追加压低学习率以便盐度优化
        self.temp_lr_threshold = config.get('temp_lr_threshold', 0.05)
        self.temp_lr_decay_factor = config.get('temp_lr_decay_factor', 0.5)
        self.temp_lr_cooldown = config.get('temp_lr_cooldown', 1)
        self.temp_lr_min = config.get('temp_lr_min', config.get('min_lr', 1e-6))
        self.last_temp_decay_epoch = -1
        self.last_epoch_temp_loss = None
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
        
        # 创建结果目录（自动编号）或使用现有目录
        if config.get('resume_dir') and os.path.exists(config['resume_dir']):
            self.result_dir = config['resume_dir']
            print(f"恢复训练，使用现有目录: {self.result_dir}")
        else:
            base_dir = config['results_dir']
            os.makedirs(base_dir, exist_ok=True)
            timestamp = datetime.now().strftime(config['timestamp_format'])
            note_slug = _sanitize_note_for_path(config.get('training_note', ''))
            result_dir_name = f"results_{timestamp}"
            if note_slug:
                result_dir_name += f"_{note_slug}"
            result_index = self._next_result_index(base_dir)
            numbered_name = f"{result_index}_{result_dir_name}"
            self.result_dir = os.path.join(base_dir, numbered_name)
            os.makedirs(self.result_dir, exist_ok=True)
            print(f"结果目录编号: {result_index} -> {self.result_dir}")
        
        # TensorBoard日志
        self.writer = SummaryWriter(os.path.join(self.result_dir, 'logs'))
        
        # 保存配置
        config_file = os.path.join(self.result_dir, config['config_filename'])
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

        # 保存训练备注
        if config.get('training_note'):
            note_path = os.path.join(self.result_dir, 'training_note.txt')
            # 如果是追加模式，使用 'a'
            mode = 'a' if config.get('resume_dir') else 'w'
            with open(note_path, mode, encoding='utf-8') as note_file:
                if mode == 'a':
                    note_file.write("\n--- 继续训练 ---\n")
                note_file.write(f"备注: {config['training_note']}\n")
                note_file.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    @staticmethod
    def _next_result_index(base_dir: str) -> int:
        existing = []
        try:
            for name in os.listdir(base_dir):
                path = os.path.join(base_dir, name)
                if not os.path.isdir(path):
                    continue
                prefix = name.split('_', 1)[0]
                if prefix.isdigit():
                    existing.append(int(prefix))
        except FileNotFoundError:
            return 0
        return max(existing) + 1 if existing else 0

    def compute_gradient_loss(self, pred, target):
        """
        计算梯度分布匹配损失 (Gradient Profile Loss)
        L_gp = MSE(|∇P|, |∇T|)
        """
        # pred, target: (B, T, C, H, W)
        # 计算空间梯度 (Sobel算子近似)
        # 定义Sobel核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        b, t, c, h, w = pred.shape
        pred_flat = pred.reshape(b*t*c, 1, h, w)
        target_flat = target.reshape(b*t*c, 1, h, w)
        
        # 计算梯度
        grad_pred_x = F.conv2d(pred_flat, sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred_flat, sobel_y, padding=1)
        grad_pred_mag = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-6)
        
        grad_target_x = F.conv2d(target_flat, sobel_x, padding=1)
        grad_target_y = F.conv2d(target_flat, sobel_y, padding=1)
        grad_target_mag = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-6)
        
        # 计算梯度的MSE损失
        loss = F.mse_loss(grad_pred_mag, grad_target_mag)
        return loss

    def compute_weighted_loss(self, outputs, targets):
        """
        计算温度和盐度的加权损失
        
        Args:
            outputs: 模型输出 (batch_size, pred_len, channels, height, width)
            targets: 目标值 (batch_size, pred_len, channels, height, width)
            
        Returns:
            加权损失值
        """
        # 支持单变量或多变量（当前主要是1或2变量）
        num_vars = len(self.target_variables)
        total_channels = outputs.shape[2]
        if self.channels_per_var is None:
            self.channels_per_var = total_channels // max(1, num_vars)

        # 计算梯度损失
        grad_loss = 0.0
        if self.config.get('use_gradient_loss', True):
            grad_loss = self.compute_gradient_loss(outputs, targets)
            grad_weight = self.config.get('gradient_loss_weight', 0.1) # 默认权重0.1
            grad_loss = grad_loss * grad_weight

        if num_vars == 1:
            mse_loss = self.criterion(outputs, targets)
            total_loss = mse_loss + grad_loss
            return total_loss, mse_loss.item(), 0.0, 0.0

        # 双变量拆分（保持原逻辑）
        temp_channels = self.channels_per_var
        temp_outputs = outputs[:, :, :temp_channels, :, :]
        temp_targets = targets[:, :, :temp_channels, :, :]
        salt_outputs = outputs[:, :, temp_channels:, :, :]
        salt_targets = targets[:, :, temp_channels:, :, :]

        temp_loss = self.criterion(temp_outputs, temp_targets)
        salt_loss = self.criterion(salt_outputs, salt_targets)

        weighted_loss = self.temp_weight * temp_loss + self.salt_weight * salt_loss + grad_loss

        return weighted_loss, temp_loss.item(), salt_loss.item(), 0.0
    
    def train_epoch(self) -> float:
        """
        训练一个epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1} 训练')
        temp_loss_sum = 0.0
        temp_loss_count = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs = inputs.to(self.device)  # (batch_size, seq_len, channels, height, width)
            targets = targets.to(self.device)  # (batch_size, pred_len, channels, height, width)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算加权损失
            loss, temp_loss_val, salt_loss_val, _ = self.compute_weighted_loss(outputs, targets)
            
            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"警告: 第{batch_idx}批次损失为NaN，跳过该批次")
                continue
                
            # 检查输出是否包含异常值
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"警告: 第{batch_idx}批次输出包含NaN或Inf，跳过该批次")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 检查梯度
            if torch.isnan(grad_norm):
                print(f"警告: 第{batch_idx}批次梯度为NaN，跳过参数更新")
                self.optimizer.zero_grad()
                continue
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            if len(self.target_variables) > 1:
                postfix = {
                    'Loss': f'{loss.item():.6f}',
                    'Temp': f'{temp_loss_val:.6f}',
                    'Salt': f'{salt_loss_val:.6f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.6f}'
                }
                progress_bar.set_postfix(postfix)
                temp_loss_sum += float(temp_loss_val)
                temp_loss_count += 1
            else:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.6f}'
                })
            
            # 记录到TensorBoard
            global_step = self.epoch * num_batches + batch_idx
            self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            if len(self.target_variables) > 1:
                self.writer.add_scalar('Loss/Train_Temp', temp_loss_val, global_step)
                self.writer.add_scalar('Loss/Train_Salt', salt_loss_val, global_step)
        
        avg_loss = total_loss / num_batches
        self.last_epoch_temp_loss = (temp_loss_sum / temp_loss_count) if temp_loss_count > 0 else None
        return avg_loss
    
    def validate_epoch(self) -> float:
        """
        验证一个epoch
        
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        # 检查验证集是否为空
        if num_batches == 0:
            print("警告: 验证集为空，跳过验证")
            return float('inf')  # 返回无穷大表示无法验证
        
        valid_batches = 0  # 记录有效批次数量
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f'Epoch {self.epoch+1} 验证')
            
            for batch in progress_bar:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # 计算加权损失
                loss, temp_loss_val, salt_loss_val, _ = self.compute_weighted_loss(outputs, targets)
                
                # 检查损失和输出是否包含异常值
                if not (torch.isnan(loss) or torch.isnan(outputs).any() or torch.isinf(outputs).any()):
                    total_loss += loss.item()
                    valid_batches += 1
                else:
                    print(f"验证中发现NaN/Inf，跳过该批次")
                    continue
                
                if len(self.target_variables) > 1:
                    postfix = {
                        'Val Loss': f'{loss.item():.6f}',
                        'Temp': f'{temp_loss_val:.6f}',
                        'Salt': f'{salt_loss_val:.6f}',
                        'Avg Val Loss': f'{total_loss/max(valid_batches, 1):.6f}'
                    }
                    progress_bar.set_postfix(postfix)
                else:
                    progress_bar.set_postfix({
                        'Val Loss': f'{loss.item():.6f}',
                        'Avg Val Loss': f'{total_loss/max(valid_batches, 1):.6f}'
                    })
                
                # 记录验证损失到TensorBoard
                if valid_batches == 1:  # 只记录第一个批次，避免过多记录
                    val_step = self.epoch
                    self.writer.add_scalar('Loss/Val_Batch', loss.item(), val_step)
                    if len(self.target_variables) > 1:
                        self.writer.add_scalar('Loss/Val_Temp', temp_loss_val, val_step)
                        self.writer.add_scalar('Loss/Val_Salt', salt_loss_val, val_step)
        
        # 防止除零错误
        if valid_batches == 0:
            print("警告: 所有验证批次都包含异常值，无法计算验证损失")
            return float('inf')
        
        avg_loss = total_loss / valid_batches
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """
        保存模型检查点
        
        Args:
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.result_dir, 'latest_checkpoint.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.result_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证损失: {self.best_val_loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            print(f"从检查点恢复训练，epoch: {self.epoch}, 最佳验证损失: {self.best_val_loss:.6f}")
        else:
            print(f"检查点文件不存在: {checkpoint_path}")

    def _load_best_model_weights(self) -> bool:
        """尝试加载最佳模型权重。"""
        best_model_path = os.path.join(self.result_dir, self.config['model_filename'])
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
            return True
        return False
    
    def plot_training_curves(self):
        """
        绘制训练曲线
        """
        if len(self.train_losses) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        
        # 损失曲线
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', linewidth=2, label='训练损失')
        plt.plot(epochs, self.val_losses, 'r-', linewidth=2, label='验证损失')
        plt.xlabel('训练轮次 (Epoch)', fontsize=12)
        plt.ylabel('损失值 (Loss)', fontsize=12)
        plt.title('训练和验证损失曲线', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加一些美化
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, epochs: int = None, resume: bool = False):
        """
        训练模型
        
        Args:
            epochs: 训练轮数，如果为None则使用配置中的默认值
            resume: 是否从检查点恢复训练
        """
        if epochs is None:
            epochs = self.config['epochs']
            
        if resume:
            checkpoint_path = os.path.join(self.result_dir, self.config['checkpoint_filename'])
            self.load_checkpoint(checkpoint_path)
        
        print(f"开始训练，总共 {epochs} 个epoch")
        print(f"结果保存在: {self.result_dir}")
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            print(f"Epoch {epoch+1} 训练完成，开始验证...")
            # 验证
            val_loss = self.validate_epoch()
            
            # 更新学习率
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # 如果学习率发生变化，打印通知
            if new_lr != old_lr:
                print(f"学习率调整: {old_lr:.2e} -> {new_lr:.2e}")
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 温度收敛后额外降学习率，便于盐度优化
            if (
                len(self.target_variables) > 1
                and self.last_epoch_temp_loss is not None
                and self.last_epoch_temp_loss <= self.temp_lr_threshold
                and (epoch - self.last_temp_decay_epoch) >= self.temp_lr_cooldown
            ):
                manual_old_lr = self.optimizer.param_groups[0]['lr']
                manual_new_lr = max(manual_old_lr * self.temp_lr_decay_factor, self.temp_lr_min)
                if manual_new_lr < manual_old_lr:
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = manual_new_lr
                    self.last_temp_decay_epoch = epoch
                    print(f"温度收敛触发额外降学习率: {manual_old_lr:.2e} -> {manual_new_lr:.2e} (TempLoss={self.last_epoch_temp_loss:.4f})")
                    self.writer.add_scalar('Learning_Rate_TempTriggered', manual_new_lr, epoch)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 检查是否为最佳模型（跳过无穷大的验证损失）
            is_best = False
            if not np.isinf(val_loss) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
                print(f"[BEST] New best val loss: {val_loss:.6f}")
            elif np.isinf(val_loss):
                print("[WARN] Val loss is inf, skipping model update")
            
            # 保存检查点
            self.save_checkpoint(is_best)
            
            # 绘制训练曲线
            if (epoch + 1) % 10 == 0:
                self.plot_training_curves()
            
            # 打印epoch信息
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"训练损失: {train_loss:.6f} | "
                  f"验证损失: {val_loss:.6f} | "
                  f"最佳验证损失: {self.best_val_loss:.6f} | "
                  f"学习率: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"时间: {epoch_time:.1f}s")
            
            # 检查损失是否为NaN
            if np.isnan(train_loss) or (not np.isinf(val_loss) and np.isnan(val_loss)):
                print("检测到NaN损失，停止训练")
                break
        
        # 最终保存
        self.plot_training_curves()

        self.writer.close()
        
        print(f"训练完成！最佳验证损失: {self.best_val_loss:.6f}")
        print(f"结果保存在: {self.result_dir}")
    
    def evaluate(self, use_best_model: bool = True) -> Dict:
        """
        评估模型
        
        Args:
            use_best_model: 是否使用最佳模型
            
        Returns:
            评估结果字典
        """
        if use_best_model:
            if self._load_best_model_weights():
                print("使用最佳模型进行评估")
            else:
                print("未找到最佳模型权重，使用当前权重评估")
        
        self.model.eval()
        
        # 在测试集上评估
        test_loss = 0.0
        num_batches = len(self.test_loader)
        
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="测试集评估"):
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss, _, _, _ = self.compute_weighted_loss(outputs, targets)
                test_loss += loss.item()
                
                # 收集预测结果
                predictions.append(outputs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        
        # 检查是否有测试数据
        if num_batches == 0:
            print("警告: 测试集为空，跳过评估")
            results = {
                'test_loss': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'correlation': 0.0
            }
            return results
        
        avg_test_loss = test_loss / num_batches
        
        # 检查是否有预测结果
        if len(predictions) == 0:
            print("警告: 没有预测结果，跳过指标计算")
            results = {
                'test_loss': avg_test_loss,
                'mae': float('inf'),
                'rmse': float('inf'),
                'correlation': 0.0
            }
            return results
        
        # 计算其他指标
        predictions = np.concatenate(predictions, axis=0)
        targets_array = np.concatenate(targets_list, axis=0)
        
        # 转换为 float64 以确保计算精度 (防止大数组累加时的精度丢失)
        predictions_f64 = predictions.astype(np.float64)
        targets_f64 = targets_array.astype(np.float64)
        
        # 标准化空间指标保留为 normalized_*，主指标优先使用反标准化后的物理单位。
        normalized_mae = np.mean(np.abs(predictions_f64 - targets_f64))
        normalized_rmse = np.sqrt(np.mean((predictions_f64 - targets_f64) ** 2))

        eval_dataset = getattr(self.test_loader, 'dataset', None)
        scalers = getattr(eval_dataset, 'scalers', {}) if eval_dataset is not None else {}
        channel_slices = getattr(eval_dataset, 'target_channel_slices', {}) if eval_dataset is not None else {}
        num_vars = len(self.target_variables)
        total_channels = predictions.shape[2]
        fallback_channels_per_var = total_channels // max(1, num_vars)

        physical_predictions_f64 = predictions_f64.copy()
        physical_targets_f64 = targets_f64.copy()
        physical_units_available = False

        if eval_dataset is not None and hasattr(eval_dataset, 'inverse_transform_targets'):
            try:
                physical_predictions_f64 = eval_dataset.inverse_transform_targets(predictions_f64).astype(np.float64)
                physical_targets_f64 = eval_dataset.inverse_transform_targets(targets_f64).astype(np.float64)
                physical_units_available = True
            except Exception as exc:
                print(f"警告: 目标物理量恢复失败，回退到普通反标准化: {exc}")

        if not physical_units_available:
            for i, var_name in enumerate(self.target_variables):
                scaler = scalers.get(var_name) if scalers else None
                if scaler is None:
                    continue
                ch_slice = channel_slices.get(var_name)
                if ch_slice is None:
                    start_ch = i * fallback_channels_per_var
                    end_ch = (i + 1) * fallback_channels_per_var
                else:
                    start_ch = ch_slice.start or 0
                    end_ch = ch_slice.stop or start_ch
                if end_ch <= start_ch:
                    continue

                pred_var = physical_predictions_f64[:, :, start_ch:end_ch, :, :]
                target_var = physical_targets_f64[:, :, start_ch:end_ch, :, :]
                pred_shape = pred_var.shape
                target_shape = target_var.shape
                physical_predictions_f64[:, :, start_ch:end_ch, :, :] = scaler.inverse_transform(
                    pred_var.reshape(-1, 1)
                ).reshape(pred_shape)
                physical_targets_f64[:, :, start_ch:end_ch, :, :] = scaler.inverse_transform(
                    target_var.reshape(-1, 1)
                ).reshape(target_shape)
                physical_units_available = True

        metric_predictions_f64 = physical_predictions_f64 if physical_units_available else predictions_f64
        metric_targets_f64 = physical_targets_f64 if physical_units_available else targets_f64

        # 计算整体MAE和RMSE
        mae = np.mean(np.abs(metric_predictions_f64 - metric_targets_f64))
        rmse = np.sqrt(np.mean((metric_predictions_f64 - metric_targets_f64) ** 2))
        
        # 计算整体相关系数
        if predictions.size > 0 and targets_array.size > 0:
            # corrcoef 内部会自动处理精度，但传入 float64 更稳妥
            correlation = np.corrcoef(metric_predictions_f64.flatten(), metric_targets_f64.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            target_mean = np.mean(metric_targets_f64)
            ss_tot = np.sum((metric_targets_f64 - target_mean) ** 2)
            ss_res = np.sum((metric_predictions_f64 - metric_targets_f64) ** 2)
            r2 = float('nan') if ss_tot == 0 else 1 - ss_res / ss_tot
        else:
            correlation = 0.0
            r2 = float('nan')
        
        results = {
            'test_loss': float(avg_test_loss),
            'mae': float(mae),
            'rmse': float(rmse),
            'metric_units': 'physical' if physical_units_available else 'normalized',
            'physical_mae': float(mae) if physical_units_available else None,
            'physical_rmse': float(rmse) if physical_units_available else None,
            'normalized_mae': float(normalized_mae),
            'normalized_rmse': float(normalized_rmse),
            'correlation': float(correlation),
            'r2': float(r2) if not np.isnan(r2) else None,
            # uppercase aliases for ablation script compatibility
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R^2': float(r2) if not np.isnan(r2) else None,
        }

        # 计算分变量指标
        for i, var_name in enumerate(self.target_variables):
            ch_slice = channel_slices.get(var_name) if channel_slices else None
            if ch_slice is None:
                start_ch = i * fallback_channels_per_var
                end_ch = (i + 1) * fallback_channels_per_var
            else:
                start_ch = ch_slice.start or 0
                end_ch = ch_slice.stop or start_ch
            
            # 使用 float64 切片
            pred_var = metric_predictions_f64[:, :, start_ch:end_ch, :, :]
            target_var = metric_targets_f64[:, :, start_ch:end_ch, :, :]
            pred_var_norm = predictions_f64[:, :, start_ch:end_ch, :, :]
            target_var_norm = targets_f64[:, :, start_ch:end_ch, :, :]
            
            var_mae = np.mean(np.abs(pred_var - target_var))
            var_rmse = np.sqrt(np.mean((pred_var - target_var) ** 2))
            var_normalized_mae = np.mean(np.abs(pred_var_norm - target_var_norm))
            var_normalized_rmse = np.sqrt(np.mean((pred_var_norm - target_var_norm) ** 2))
            
            if pred_var.size > 0 and target_var.size > 0:
                var_corr = np.corrcoef(pred_var.flatten(), target_var.flatten())[0, 1]
                if np.isnan(var_corr): var_corr = 0.0
                
                v_mean = np.mean(target_var)
                v_ss_tot = np.sum((target_var - v_mean) ** 2)
                v_ss_res = np.sum((pred_var - target_var) ** 2)
                var_r2 = float('nan') if v_ss_tot == 0 else 1 - v_ss_res / v_ss_tot
            else:
                var_corr = 0.0
                var_r2 = float('nan')
            
            results[f'mae_{var_name}'] = float(var_mae)
            results[f'rmse_{var_name}'] = float(var_rmse)
            results[f'physical_mae_{var_name}'] = float(var_mae) if physical_units_available else None
            results[f'physical_rmse_{var_name}'] = float(var_rmse) if physical_units_available else None
            results[f'normalized_mae_{var_name}'] = float(var_normalized_mae)
            results[f'normalized_rmse_{var_name}'] = float(var_normalized_rmse)
            results[f'correlation_{var_name}'] = float(var_corr)
            results[f'r2_{var_name}'] = float(var_r2) if not np.isnan(var_r2) else None

        print(f"测试结果:")
        if not np.isnan(avg_test_loss):
            print(f"  测试损失: {avg_test_loss:.6f}")
        else:
            print(f"  测试损失: nan")
        if not np.isnan(mae):
            print(f"  MAE: {mae:.6f}")
        else:
            print(f"  MAE: nan")
        if not np.isnan(rmse):
            print(f"  RMSE: {rmse:.6f} ({results['metric_units']})")
        else:
            print(f"  RMSE: nan")
        print(f"  Normalized MAE/RMSE: {normalized_mae:.6f} / {normalized_rmse:.6f}")
        if not np.isnan(correlation):
            print(f"  相关系数: {correlation:.6f}")
        else:
            print(f"  相关系数: nan")
        if r2 is not None and not np.isnan(r2):
            print(f"  R^2: {r2:.6f}")
        else:
            print("  R^2: nan")
            
        # 打印分变量结果
        for var_name in self.target_variables:
            print(f"  [{var_name}] MAE: {results[f'mae_{var_name}']:.6f} | RMSE: {results[f'rmse_{var_name}']:.6f} | R^2: {results[f'r2_{var_name}'] if results[f'r2_{var_name}'] is not None else 'nan'}")
        
        # 保存评估结果
        with open(os.path.join(self.result_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        return results


def main():
    """
    主函数
    """
    # 设置中文字体
    setup_chinese_fonts()
    
    # 使用统一配置文件
    config = DEFAULT_CONFIG.copy()
    
    # 可以在这里覆盖特定的参数
    # config['epochs'] = 200           # 覆盖默认的epoch数
    # config['learning_rate'] = 1e-3   # 覆盖默认的学习率
    # config['batch_size'] = 16        # 覆盖默认的batch_size
    
    # 验证配置
    validate_config(config)
    
    print("海洋数据ConvLSTM模型训练")
    print("=" * 50)
    print("使用统一配置文件:")
    print(f"  数据路径: {config['data_path']}")
    print(f"  序列长度: {config['sequence_length']}")
    print(f"  预测长度: {config['prediction_length']}")
    print(f"  隐藏维度: {config['hidden_dims']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  Dropout: {config['dropout']}")
    print("-" * 50)
    print("TSC-Fusion 组件状态:")
    print(f"  [PosEncode] 位置编码: {'开启' if config.get('enable_positional_encoding', False) else '关闭'}")
    print(f"  [TimeEncode] 时间编码: {'开启' if config.get('enable_time_encoding', False) else '关闭'}")
    print(f"  [TSC] 热盐结构记忆: {'关闭(消融)' if config.get('ablation_disable_tsc', False) else '开启'}")
    print(f"  [Spectral] 全局频谱分支: {'关闭(消融)' if config.get('ablation_disable_spectral', False) else '开启'}")
    print(f"  [3D] 三维结构分支: {'关闭(消融)' if config.get('ablation_disable_3d', False) else '开启'}")
    print(f"  [Ensemble] 门控集成: {'关闭(消融)' if config.get('ablation_disable_ensemble', False) else '开启'}")
    print("=" * 50)
    
    # 获取本次训练备注
    try:
        training_note = input("请输入本次训练的备注（可留空直接回车）: ").strip()
    except EOFError:
        training_note = ""

    if training_note:
        print(f"本次训练备注: {training_note}")
        config['training_note'] = training_note
    else:
        print("未输入备注，使用默认配置继续训练。")

    try:
        # 创建训练器
        trainer = OceanModelTrainer(config)
        
        # 训练模型（使用配置中的默认epochs）
        trainer.train()
        
        # 评估模型
        trainer.evaluate()
        
        print("训练和评估完成！")
        print(f"结果保存在: {trainer.result_dir}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
