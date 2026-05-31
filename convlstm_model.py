"""
海洋数据温度和盐度反演的ConvLSTM模型
基于时空卷积LSTM网络预测未来的温度和盐度分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM单元 - 卷积长短期记忆网络的基本单元
    结合了卷积神经网络的空间特征提取能力和LSTM的时序建模能力
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int], 
                 bias: bool = True, padding: str = 'same'):
        """
        初始化ConvLSTM单元
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏状态维度  
            kernel_size: 卷积核大小
            bias: 是否使用偏置
            padding: 填充方式
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2 if padding == 'same' else 0
        self.bias = bias
        
        # 输入到隐藏状态的卷积层 (输入门、遗忘门、候选值、输出门)
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
        # 隐藏状态到隐藏状态的卷积层
        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor: torch.Tensor, cur_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_tensor: 输入张量 (batch_size, input_dim, height, width)
            cur_state: 当前状态 (hidden_state, cell_state)
            
        Returns:
            新的隐藏状态和细胞状态
        """
        h_cur, c_cur = cur_state
        
        # 计算输入到隐藏状态的卷积
        combined_ih = self.conv_ih(input_tensor)
        
        # 计算隐藏状态到隐藏状态的卷积
        combined_hh = self.conv_hh(h_cur)
        
        # 组合输入和隐藏状态的贡献
        combined = combined_ih + combined_hh
        
        # 分离四个门的输出
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_dim, dim=1)
        
        # 计算门控值
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = torch.tanh(cc_g)     # 候选值
        
        # 更新细胞状态和隐藏状态
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, image_size: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化隐藏状态和细胞状态
        
        Args:
            batch_size: 批次大小
            image_size: 图像尺寸 (height, width)
            device: 设备
            
        Returns:
            初始化的隐藏状态和细胞状态
        """
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """
    多层ConvLSTM网络
    用于处理海洋数据的时空序列预测
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], kernel_size: Tuple[int, int],
                 num_layers: int, batch_first: bool = True, bias: bool = True, 
                 return_all_layers: bool = False):
        """
        初始化ConvLSTM网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 每层的隐藏维度列表
            kernel_size: 卷积核大小
            num_layers: 层数
            batch_first: 是否批次优先
            bias: 是否使用偏置
            return_all_layers: 是否返回所有层的输出
        """
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        # 构建ConvLSTM层
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dims[i],
                kernel_size=self.kernel_size,
                bias=self.bias
            ))
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, input_tensor: torch.Tensor, hidden_state: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        """
        前向传播
        
        Args:
            input_tensor: 输入张量 (batch_size, seq_len, input_dim, height, width)
            hidden_state: 初始隐藏状态
            
        Returns:
            输出张量和最终隐藏状态
        """
        if not self.batch_first:
            # 如果不是批次优先，转换维度 (seq_len, batch_size, input_dim, height, width)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        batch_size, seq_len = input_tensor.size(0), input_tensor.size(1)
        height, width = input_tensor.size(3), input_tensor.size(4)
        
        # 初始化隐藏状态
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width), input_tensor.device)
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size: int, image_size: Tuple[int, int], device: torch.device) -> List:
        """
        初始化所有层的隐藏状态
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size, device))
        return init_states


class OceanConvLSTMPredictor(nn.Module):
    """
    海洋数据ConvLSTM预测模型
    专门用于温度和盐度的时空序列预测
    """
    
    def __init__(self, config: dict):
        """
        初始化海洋预测模型
        
        Args:
            config: 配置字典，包含模型参数
        """
        super(OceanConvLSTMPredictor, self).__init__()
        
        self.config = config
        
        # 计算实际的输入和输出维度
        if 'actual_input_channels' in config and 'actual_output_channels' in config:
            # 使用实际检测到的维度
            self.input_dim = config['actual_input_channels']
            self.output_dim = config['actual_output_channels']
            print(f"使用检测到的实际维度 - 输入: {self.input_dim}, 输出: {self.output_dim}")
        else:
            # 使用默认计算方式（向后兼容）
            num_depth_levels = 2  # 实际数据显示有2个深度层
            self.input_dim = len(config['input_variables']) * num_depth_levels
            self.output_dim = len(config['target_variables']) * num_depth_levels
            print(f"使用默认计算维度 - 输入: {self.input_dim}, 输出: {self.output_dim}")
        
        self.seq_len = config.get('sequence_length', 10)  # 输入序列长度
        self.pred_len = config.get('prediction_length', 5)  # 预测序列长度
        
        # ConvLSTM编码器
        self.encoder = ConvLSTM(
            input_dim=self.input_dim,
            hidden_dims=config.get('hidden_dims', [64, 64, 64]),
            kernel_size=config.get('kernel_size', (3, 3)),
            num_layers=config.get('num_layers', 3),
            batch_first=True,
            bias=True,
            return_all_layers=True  # 返回所有层的状态，供解码器使用
        )
        
        # ConvLSTM解码器
        self.decoder = ConvLSTM(
            input_dim=config.get('hidden_dims', [64, 64, 64])[-1],
            hidden_dims=config.get('hidden_dims', [64, 64, 64]),
            kernel_size=config.get('kernel_size', (3, 3)),
            num_layers=config.get('num_layers', 3),
            batch_first=True,
            bias=True,
            return_all_layers=True  # 也设置为True以保持一致性
        )
        
        # 输出投影层
        self.output_proj = nn.Conv2d(
            in_channels=config.get('hidden_dims', [64, 64, 64])[-1],
            out_channels=self.output_dim,
            kernel_size=1,
            padding=0
        )
        
        # 批归一化
        self.batch_norm = nn.BatchNorm2d(self.output_dim)
        
        # Dropout正则化
        self.dropout = nn.Dropout2d(config.get('dropout', 0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, channels, height, width)
            
        Returns:
            预测输出 (batch_size, pred_len, output_channels, height, width)
        """
        batch_size = x.size(0)
        
        # 编码阶段 - 处理输入序列
        encoder_outputs, encoder_states = self.encoder(x)
        
        # 获取编码器的最后输出作为初始输入
        last_output = encoder_outputs[-1][:, -1, :, :, :]  # (batch_size, hidden_dim, height, width)
        
        # 解码阶段 - 生成预测序列
        decoder_input = last_output.unsqueeze(1)  # (batch_size, 1, hidden_dim, height, width)
        
        # 使用编码器的最终状态初始化解码器
        decoder_hidden = encoder_states
        
        predictions = []
        
        for t in range(self.pred_len):
            # 解码器前向传播
            decoder_outputs, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            current_output = decoder_outputs[-1][:, -1, :, :, :]  # 获取当前时步的输出
            
            # 投影到目标维度
            pred = self.output_proj(current_output)
            pred = self.batch_norm(pred)
            pred = self.dropout(pred)
            
            predictions.append(pred)
            
            # 使用当前输出作为下一时步的输入
            decoder_input = current_output.unsqueeze(1)
        
        # 堆叠预测结果
        predictions = torch.stack(predictions, dim=1)
        
        return predictions
    
    def predict_sequence(self, x: torch.Tensor, future_steps: int) -> torch.Tensor:
        """
        预测未来多个时步
        
        Args:
            x: 输入序列 (batch_size, seq_len, channels, height, width)
            future_steps: 未来预测步数
            
        Returns:
            预测结果 (batch_size, future_steps, output_channels, height, width)
        """
        self.eval()
        with torch.no_grad():
            # 临时修改预测长度
            original_pred_len = self.pred_len
            self.pred_len = future_steps
            
            predictions = self.forward(x)
            
            # 恢复原始预测长度
            self.pred_len = original_pred_len
            
        return predictions


def create_ocean_model(config: dict) -> OceanConvLSTMPredictor:
    """
    创建海洋预测模型的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        配置好的海洋预测模型
    """
    return OceanConvLSTMPredictor(config)


# 默认配置
DEFAULT_CONFIG = {
    'input_variables': ["TEMP", "SALT", "SSHA", "UWND", "VWND"],  # 使用实际存在的变量
    'target_variables': ["TEMP", "SALT"],
    'sequence_length': 10,
    'prediction_length': 5,
    'hidden_dims': [64, 64, 64],
    'kernel_size': (3, 3),
    'num_layers': 3,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'batch_size': 4,
    'epochs': 100
}


if __name__ == "__main__":
    # 测试模型
    config = DEFAULT_CONFIG.copy()
    model = create_ocean_model(config)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    channels = len(config['input_variables'])
    height, width = 32, 21  # 根据数据描述的经纬度范围计算
    
    test_input = torch.randn(batch_size, seq_len, channels, height, width)
    
    print(f"模型配置: {config}")
    print(f"输入形状: {test_input.shape}")
    
    # 前向传播测试
    output = model(test_input)
    print(f"输出形状: {output.shape}")
    print("ConvLSTM模型创建成功！")