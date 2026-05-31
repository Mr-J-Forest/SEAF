"""
海洋数据ConvLSTM模型训练脚本
使用统一配置文件确保参数一致性
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

# 导入统一配置
from config import DEFAULT_CONFIG, save_config, validate_config, update_config
from convlstm_model import OceanConvLSTMPredictor
from data_loader import create_data_loaders

# 设置中文字体支持
def setup_chinese_fonts():
    """设置中文字体支持"""
    # 获取系统所有可用字体
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    # 中文字体候选列表（按优先级排序）
    chinese_fonts = [
        'WenQuanYi Micro Hei', # 文泉驿微米黑 (Linux)
        'WenQuanYi Zen Hei',   # 文泉驿正黑 (Linux)
        'SimHei',              # 黑体 (Windows)
        'Microsoft YaHei',     # 微软雅黑 (Windows)
        'Noto Sans CJK SC',    # 思源黑体 (Google)
        'Source Han Sans SC',  # 思源黑体 (Adobe)
        'Hiragino Sans GB',    # 冬青黑体 (macOS)
        'PingFang SC',         # 苹方 (macOS)
        'Arial Unicode MS',    # Unicode 字体 (macOS)
        'STHeiti',             # 华文黑体
        'STSong',              # 华文宋体
        'DejaVu Sans'          # 备用字体
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
        print(f"✓ 成功设置中文字体: {found_font}")
    else:
        # 如果没有找到中文字体，使用matplotlib默认配置
        plt.rcParams['font.sans-serif'] = chinese_fonts
        print("⚠ 未找到中文字体，使用默认配置")
    
    # 设置负号正确显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置字体大小
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    return found_font

# 初始化中文字体
setup_chinese_fonts()

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
        
        # 设置设备
        if config['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config['device'])
        print(f"使用设备: {self.device}")
        
        # 设置性能优化
        if config['cudnn_benchmark']:
            torch.backends.cudnn.benchmark = True
            print("启用cuDNN benchmark")
        
        # 创建数据加载器
        print("创建数据加载器...")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            config['data_path'], 
            config, 
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        # 获取第一个批次来确定实际的输入输出维度
        print("检测数据维度...")
        sample_batch = next(iter(self.train_loader))
        sample_input, sample_target = sample_batch
        actual_input_channels = sample_input.shape[2]  # [batch, seq, channels, height, width]
        actual_output_channels = sample_target.shape[2]  # [batch, pred, channels, height, width]
        
        print(f"实际输入通道数: {actual_input_channels}")
        print(f"实际输出通道数: {actual_output_channels}")
        
        # 更新配置
        config['actual_input_channels'] = actual_input_channels
        config['actual_output_channels'] = actual_output_channels
        
        # 创建模型（使用实际维度）
        self.model = OceanConvLSTMPredictor(config).to(self.device)
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
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
        
        # 创建结果目录
        base_dir = config['results_dir']
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime(config['timestamp_format'])
        self.result_dir = f"{base_dir}/results_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # TensorBoard日志
        self.writer = SummaryWriter(os.path.join(self.result_dir, 'logs'))
        
        # 保存配置
        config_file = os.path.join(self.result_dir, config['config_filename'])
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def compute_weighted_loss(self, outputs, targets):
        """
        计算温度和盐度的加权损失
        
        Args:
            outputs: 模型输出 (batch_size, pred_len, channels, height, width)
            targets: 目标值 (batch_size, pred_len, channels, height, width)
            
        Returns:
            加权损失值
        """
        # 确定通道划分（如果还没有确定）
        if self.channels_per_var is None:
            total_channels = outputs.shape[2]
            num_vars = len(self.target_variables)
            self.channels_per_var = total_channels // num_vars
        
        temp_channels = self.channels_per_var
        
        # 分离温度和盐度
        temp_outputs = outputs[:, :, :temp_channels, :, :]
        temp_targets = targets[:, :, :temp_channels, :, :]
        
        salt_outputs = outputs[:, :, temp_channels:, :, :]
        salt_targets = targets[:, :, temp_channels:, :, :]
        
        # 计算各变量损失
        temp_loss = self.criterion(temp_outputs, temp_targets)
        salt_loss = self.criterion(salt_outputs, salt_targets)
        
        # 加权求和
        weighted_loss = self.temp_weight * temp_loss + self.salt_weight * salt_loss
        
        return weighted_loss, temp_loss.item(), salt_loss.item()
    
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
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)  # (batch_size, seq_len, channels, height, width)
            targets = targets.to(self.device)  # (batch_size, pred_len, channels, height, width)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # (batch_size, pred_len, output_channels, height, width)
            
            # 计算加权损失
            loss, temp_loss_val, salt_loss_val = self.compute_weighted_loss(outputs, targets)
            
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
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Temp': f'{temp_loss_val:.6f}',
                'Salt': f'{salt_loss_val:.6f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.6f}'
            })
            
            # 记录到TensorBoard
            global_step = self.epoch * num_batches + batch_idx
            self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            self.writer.add_scalar('Loss/Train_Temp', temp_loss_val, global_step)
            self.writer.add_scalar('Loss/Train_Salt', salt_loss_val, global_step)
        
        avg_loss = total_loss / num_batches
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
            
            for inputs, targets in progress_bar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # 计算加权损失
                loss, temp_loss_val, salt_loss_val = self.compute_weighted_loss(outputs, targets)
                
                # 检查损失和输出是否包含异常值
                if not (torch.isnan(loss) or torch.isnan(outputs).any() or torch.isinf(outputs).any()):
                    total_loss += loss.item()
                    valid_batches += 1
                else:
                    print(f"验证中发现NaN/Inf，跳过该批次")
                    continue
                
                progress_bar.set_postfix({
                    'Val Loss': f'{loss.item():.6f}',
                    'Temp': f'{temp_loss_val:.6f}',
                    'Salt': f'{salt_loss_val:.6f}',
                    'Avg Val Loss': f'{total_loss/max(valid_batches, 1):.6f}'
                })
                
                # 记录验证损失到TensorBoard
                if valid_batches == 1:  # 只记录第一个批次，避免过多记录
                    val_step = self.epoch
                    self.writer.add_scalar('Loss/Val_Batch', loss.item(), val_step)
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
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 检查是否为最佳模型（跳过无穷大的验证损失）
            is_best = False
            if not np.isinf(val_loss) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
                print(f"✓ 新的最佳验证损失: {val_loss:.6f}")
            elif np.isinf(val_loss):
                print("⚠️  验证损失为无穷大，跳过模型更新")
            
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
            best_model_path = os.path.join(self.result_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("使用最佳模型进行评估")
        
        self.model.eval()
        
        # 在测试集上评估
        test_loss = 0.0
        num_batches = len(self.test_loader)
        
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="测试集评估"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
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
        
        # 计算MAE和RMSE
        mae = np.mean(np.abs(predictions - targets_array))
        rmse = np.sqrt(np.mean((predictions - targets_array) ** 2))
        
        # 计算相关系数
        if predictions.size > 0 and targets_array.size > 0:
            correlation = np.corrcoef(predictions.flatten(), targets_array.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        results = {
            'test_loss': float(avg_test_loss),
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation)
        }
        
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
            print(f"  RMSE: {rmse:.6f}")
        else:
            print(f"  RMSE: nan")
        if not np.isnan(correlation):
            print(f"  相关系数: {correlation:.6f}")
        else:
            print(f"  相关系数: nan")
        
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
    print("=" * 50)
    
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