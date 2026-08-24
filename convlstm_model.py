"""
海洋数据温度和盐度反演的ConvLSTM模型
基于时空卷积LSTM网络预测未来的温度和盐度分布
"""

import torch
import torch.nn as nn
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
                 return_all_layers: bool = False, residual_between_layers: bool = False,
                 dropout: float = 0.0):
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
            residual_between_layers: 是否在层间使用残差连接
            dropout: 层间Dropout概率
        """
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.residual_between_layers = residual_between_layers
        self.dropout = dropout

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
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

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

            if self.residual_between_layers and cur_layer_input.shape[2] == layer_output.shape[2]:
                layer_output = layer_output + cur_layer_input

            # Apply dropout between layers
            if layer_idx < self.num_layers - 1 and self.dropout > 0:
                B_drop, T_drop, C_drop, H_drop, W_drop = layer_output.shape
                layer_output_flat = layer_output.view(B_drop * T_drop, C_drop, H_drop, W_drop)
                layer_output_flat = self.dropout_layer(layer_output_flat)
                layer_output = layer_output_flat.view(B_drop, T_drop, C_drop, H_drop, W_drop)

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


class ResidualRefiner(nn.Module):
    """带自适应GroupNorm的残差细化模块"""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        group_count = min(8, channels)
        while channels % group_count != 0 and group_count > 1:
            group_count -= 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=group_count, num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = self.dropout(out)
        return self.activation(residual + out)


class TransformerRefiner(nn.Module):
    """空间注意力Transformer细化模块"""

    def __init__(self, channels: int, heads: int, layers: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
            activation='gelu',
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.view(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        tokens = self.encoder(tokens)
        tokens = tokens.permute(0, 2, 1).view(b, c, h, w)
        return tokens


class GlobalTokenBankAttention(nn.Module):
    """Cross-window attention over windows from the same historical time."""

    def __init__(self, channels: int, heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.token_norm = nn.LayerNorm(channels)
        self.query_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, channels),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def pool_tokens(x: torch.Tensor) -> torch.Tensor:
        """Return one raw token per spatial window."""
        return x.mean(dim=(2, 3))

    def forward(
        self,
        x: torch.Tensor,
        bank_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        pooled_tokens = self.pool_tokens(x) if bank_tokens is None else bank_tokens
        if pooled_tokens.ndim != 2 or pooled_tokens.shape[1] != c:
            raise ValueError(
                f'global token bank must have shape (windows, {c}), got {tuple(pooled_tokens.shape)}'
            )
        if pooled_tokens.shape[0] <= 1:
            return x

        bank = self.token_norm(pooled_tokens).unsqueeze(0).expand(b, -1, -1)

        queries = x.view(b, c, h * w).permute(0, 2, 1)
        queries = self.query_norm(queries)
        context, _ = self.attn(queries, bank, bank, need_weights=False)
        context = context + self.ffn(self.out_norm(context))
        context = context.permute(0, 2, 1).view(b, c, h, w)
        return x + torch.tanh(self.gate) * context


class ThermohalineMemory(nn.Module):
    """Learned water-mass memory over the configured profile channels.

    The module routes each grid column to soft water-mass prototypes, updates
    those prototype tokens with a tiny Transformer, then writes the water-mass
    context back to the Eulerian grid as extra model input channels. The active
    configuration uses TEMP/SALT only; other variables are used only when they
    are both present in ``input_variables`` and listed in ``tsc_variables``.
    """

    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        self.variables = config.get(
            'tsc_variables',
            ['TEMP', 'SALT', 'PTEMP', 'PDEN', 'SPICE']
        )
        self.selected_slices = self._resolve_slices(config.get('input_channel_slices', {}), input_dim)
        selected_channels = sum(stop - start for start, stop in self.selected_slices)
        self.selected_channels = selected_channels

        hidden_dim = int(config.get('tsc_hidden_dim', 32))
        output_dim = int(config.get('tsc_output_dim', 16))
        prototype_count = int(config.get('tsc_num_prototypes', 8))
        heads = int(config.get('tsc_attention_heads', 4))
        dropout = float(config.get('dropout', 0.0))

        if hidden_dim % heads != 0:
            heads = 1

        self.profile_proj = nn.Sequential(
            nn.Conv2d(selected_channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=self._groups(hidden_dim), num_channels=hidden_dim),
            nn.GELU(),
        )
        self.router = nn.Conv2d(hidden_dim, prototype_count, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=max(hidden_dim * 2, int(config.get('tsc_ffn_dim', hidden_dim * 2))),
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.prototype_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(config.get('tsc_memory_layers', 1))
        )
        self.prototype_bias = nn.Parameter(torch.zeros(prototype_count, hidden_dim))
        self.grid_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=self._groups(hidden_dim), num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
        )

    @staticmethod
    def _groups(channels: int) -> int:
        group_count = min(8, channels)
        while channels % group_count != 0 and group_count > 1:
            group_count -= 1
        return group_count

    def _resolve_slices(self, raw_slices, input_dim: int) -> List[Tuple[int, int]]:
        selected = []
        for name in self.variables:
            bounds = raw_slices.get(name) if isinstance(raw_slices, dict) else None
            if isinstance(bounds, slice):
                start, stop = bounds.start, bounds.stop
            elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                start, stop = int(bounds[0]), int(bounds[1])
            else:
                continue
            if start is None or stop is None:
                continue
            start = max(0, int(start))
            stop = min(input_dim, int(stop))
            if stop > start:
                selected.append((start, stop))

        if not selected:
            raise ValueError(
                'ThermohalineMemory 未找到任何配置变量的输入通道；'
                '拒绝静默把全部输入变量当作温盐剖面'
            )
        return selected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        profile_parts = [x[:, :, start:stop, :, :] for start, stop in self.selected_slices]
        profiles = torch.cat(profile_parts, dim=2)
        b, t, c, h, w = profiles.shape
        flat = profiles.reshape(b * t, c, h, w)

        features = self.profile_proj(flat)
        assignment = torch.softmax(self.router(features), dim=1)
        denom = assignment.sum(dim=(2, 3)).clamp_min(1e-6)
        tokens = torch.einsum('bkhw,bchw->bkc', assignment, features) / denom.unsqueeze(-1)
        tokens = self.prototype_encoder(tokens + self.prototype_bias.unsqueeze(0))
        water_context = torch.einsum('bkhw,bkc->bchw', assignment, tokens)
        fused = self.grid_fuse(torch.cat([features, water_context], dim=1))
        return fused.view(b, t, -1, h, w)


def _adaptive_group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class ConvGNAct2d(nn.Module):
    """Small 2D conv block with stable normalization for low-batch training."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_adaptive_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvGNAct3d(nn.Module):
    """3D conv block over the sequence-spatial cube."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 dropout: float = 0.0):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_adaptive_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralLowModeMixer(nn.Module):
    """Global low-frequency spatial mixer using learnable diagonal Fourier modes."""

    def __init__(self, channels: int, modes_y: int = 8, modes_x: int = 8, dropout: float = 0.0):
        super().__init__()
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)
        scale = 0.02
        self.weight_pos = nn.Parameter(scale * torch.randn(channels, self.modes_y, self.modes_x, 2))
        self.weight_neg = nn.Parameter(scale * torch.randn(channels, self.modes_y, self.modes_x, 2))
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_adaptive_group_count(channels), channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        mh = max(1, min(self.modes_y, h // 2 if h > 1 else 1))
        mw = max(1, min(self.modes_x, w // 2 + 1))

        fft_input = x.float()
        spectrum = torch.fft.rfft2(fft_input, norm='ortho')
        filtered = torch.zeros_like(spectrum)

        pos_weight = torch.view_as_complex(self.weight_pos[:, :mh, :mw].contiguous()).to(spectrum.dtype)
        neg_weight = torch.view_as_complex(self.weight_neg[:, :mh, :mw].contiguous()).to(spectrum.dtype)
        filtered[:, :, :mh, :mw] = spectrum[:, :, :mh, :mw] * pos_weight.unsqueeze(0)
        filtered[:, :, -mh:, :mw] = spectrum[:, :, -mh:, :mw] * neg_weight.unsqueeze(0)

        spatial = torch.fft.irfft2(filtered, s=(h, w), norm='ortho').to(dtype=x.dtype)
        return x + self.mix(spatial)


class LowModeSpectralBranch(nn.Module):
    """Flattened time-channel branch with learnable low-mode spatial filtering."""

    def __init__(self, input_channels: int, hidden_dim: int, modes_y: int, modes_x: int,
                 layers: int = 2, dropout: float = 0.0):
        super().__init__()
        blocks = [ConvGNAct2d(input_channels, hidden_dim, dropout=dropout)]
        for _ in range(max(1, layers)):
            blocks.append(SpectralLowModeMixer(hidden_dim, modes_y=modes_y, modes_x=modes_x, dropout=dropout))
            blocks.append(ConvGNAct2d(hidden_dim, hidden_dim, kernel_size=1, dropout=dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatiotemporalStructureBranch(nn.Module):
    """3D convolutional branch over channel-time-latitude-longitude cubes."""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        blocks = [ConvGNAct3d(input_dim, hidden_dim, dropout=dropout)]
        for _ in range(max(0, layers - 1)):
            blocks.append(ConvGNAct3d(hidden_dim, hidden_dim, dropout=dropout))
        self.encoder = nn.Sequential(*blocks)
        self.temporal_score = nn.Conv3d(hidden_dim, 1, kernel_size=1)
        self.post = ConvGNAct2d(hidden_dim, hidden_dim, kernel_size=1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        z = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.encoder(z)
        weights = torch.softmax(self.temporal_score(z), dim=2)
        pooled = (z * weights).sum(dim=2)
        return self.post(pooled)


class TSCFusionNet(nn.Module):
    """Thermohaline-memory, spectral, spatiotemporal, and gated fusion forecaster."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.seq_len = int(config.get('sequence_length', 12))
        self.pred_len = int(config.get('prediction_length', 5))
        self.target_variables = config.get('target_variables', [])

        if 'actual_input_channels' in config and 'actual_output_channels' in config:
            self.input_dim = int(config['actual_input_channels'])
            self.output_dim = int(config['actual_output_channels'])
            print(f"使用检测到的实际维度 - 输入: {self.input_dim}, 输出: {self.output_dim}")
        else:
            assumed_depth_levels = int(config.get('assumed_depth_levels', 2))
            self.input_dim = len(config.get('input_variables', [])) * assumed_depth_levels
            self.output_dim = len(self.target_variables) * assumed_depth_levels
            print(f"使用默认计算维度 (假设depth={assumed_depth_levels}) - 输入: {self.input_dim}, 输出: {self.output_dim}")

        hidden_dim = int(config.get('tsc_fusion_hidden_dim', 64))
        dropout = float(config.get('dropout', 0.05))
        spectral_modes = config.get('tsc_fusion_spectral_modes', [8, 8])
        modes_y = int(spectral_modes[0]) if len(spectral_modes) > 0 else 8
        modes_x = int(spectral_modes[1]) if len(spectral_modes) > 1 else modes_y

        self.disable_tsc = bool(config.get('ablation_disable_tsc', False))
        self.disable_spectral = bool(config.get('ablation_disable_spectral', False))
        self.disable_3d = bool(config.get('ablation_disable_3d', False))
        self.disable_ensemble = bool(config.get('ablation_disable_ensemble', False))

        tsc_out_dim = int(config.get('tsc_output_dim', 16))
        self.thermohaline_memory = (
            None if self.disable_tsc else ThermohalineMemory(self.input_dim, config)
        )
        self.augmented_dim = self.input_dim + (0 if self.disable_tsc else tsc_out_dim)
        ablation_tags = []
        if self.disable_tsc: ablation_tags.append("no-TSC")
        if self.disable_spectral: ablation_tags.append("no-spectral")
        if self.disable_3d: ablation_tags.append("no-3D")
        if self.disable_ensemble: ablation_tags.append("no-ensemble")
        tag = f" [ablation: {', '.join(ablation_tags)}]" if ablation_tags else ""
        print(
            "启用 TSC-Fusion 主干 "
            f"(hidden={hidden_dim}, augmented_channels={self.augmented_dim}, modes=({modes_y},{modes_x})){tag}"
        )

        tsc_chan = self.input_dim if self.disable_tsc else self.augmented_dim
        flat_channels = self.seq_len * tsc_chan
        self.local_branch = nn.Sequential(
            ConvGNAct2d(flat_channels, hidden_dim, dropout=dropout),
            ConvGNAct2d(hidden_dim, hidden_dim, dropout=dropout),
        )
        self.spectral_branch = LowModeSpectralBranch(
            flat_channels,
            hidden_dim,
            modes_y=modes_y,
            modes_x=modes_x,
            layers=int(config.get('tsc_fusion_spectral_layers', 2)),
            dropout=dropout,
        ) if not self.disable_spectral else None
        struct_in_dim = self.augmented_dim if not self.disable_tsc else self.input_dim
        self.structure_branch = SpatiotemporalStructureBranch(
            struct_in_dim,
            hidden_dim,
            layers=int(config.get('tsc_fusion_3d_layers', 2)),
            dropout=dropout,
        ) if not self.disable_3d else None

        n_branches = 1 + (0 if self.disable_spectral else 1) + (0 if self.disable_3d else 1)
        self.fusion = nn.Sequential(
            ConvGNAct2d(hidden_dim * n_branches, hidden_dim, kernel_size=1, dropout=dropout),
            ResidualRefiner(hidden_dim, dropout=dropout),
        )
        transformer_layers = int(config.get('tsc_fusion_transformer_layers', 1))
        if transformer_layers > 0:
            heads = int(config.get('tsc_fusion_transformer_heads', 8))
            if hidden_dim % heads != 0:
                heads = 1
            self.fusion_transformer = TransformerRefiner(
                hidden_dim,
                heads=heads,
                layers=transformer_layers,
                ffn_dim=int(config.get('tsc_fusion_transformer_ffn_dim', hidden_dim * 4)),
                dropout=dropout,
            )
        else:
            self.fusion_transformer = None

        if config.get('enable_global_token_bank', False):
            bank_heads = int(config.get('global_token_bank_heads', 4))
            if hidden_dim % bank_heads != 0:
                bank_heads = 1
            self.global_token_bank = GlobalTokenBankAttention(
                hidden_dim,
                heads=bank_heads,
                ffn_dim=int(config.get('global_token_bank_ffn_dim', hidden_dim * 2)),
                dropout=float(config.get('global_token_bank_dropout', dropout)),
            )
            print(f"启用 Global Token Bank (heads={bank_heads})")
        else:
            self.global_token_bank = None

        members = 1 if self.disable_ensemble else int(config.get('tsc_fusion_ensemble_members', 4))
        out_channels = self.pred_len * self.output_dim
        self.member_heads = nn.ModuleList([
            nn.Sequential(
                ConvGNAct2d(hidden_dim, hidden_dim, dropout=dropout),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            )
            for _ in range(members)
        ])
        self.ensemble_gate = None if self.disable_ensemble else nn.Sequential(
            nn.Conv2d(hidden_dim, max(8, hidden_dim // 2), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(8, hidden_dim // 2), members, kernel_size=1),
            nn.Softmax(dim=1),
        )

        self.enable_persistence_residual = bool(config.get('enable_persistence_residual', True))
        self.persistence_slices = (
            self._resolve_target_input_slices(config) if self.enable_persistence_residual else []
        )
        if self.enable_persistence_residual and not self.persistence_slices:
            raise ValueError(
                '启用 persistence residual 时，所有目标变量必须能映射到输入通道'
            )
        persistence_channels = sum(stop - start for start, stop in self.persistence_slices)
        self.persistence_proj = None
        if persistence_channels > 0 and persistence_channels != self.output_dim:
            self.persistence_proj = nn.Conv2d(persistence_channels, self.output_dim, kernel_size=1)
        if self.enable_persistence_residual:
            self.persistence_scale = nn.Parameter(
                torch.tensor(float(config.get('tsc_fusion_persistence_init', 0.5)), dtype=torch.float32)
            )
        else:
            self.register_parameter('persistence_scale', None)

    def _resolve_target_input_slices(self, config: dict) -> List[Tuple[int, int]]:
        raw_slices = config.get('input_channel_slices', {})
        selected = []
        if isinstance(raw_slices, dict):
            for name in self.target_variables:
                bounds = raw_slices.get(name)
                if isinstance(bounds, slice):
                    start, stop = bounds.start, bounds.stop
                elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                    start, stop = int(bounds[0]), int(bounds[1])
                else:
                    continue
                if start is None or stop is None:
                    continue
                start = max(0, int(start))
                stop = min(self.input_dim, int(stop))
                if stop > start:
                    selected.append((start, stop))
        return selected

    def _persistence_base(self, raw_x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.persistence_slices:
            return None
        last = raw_x[:, -1]
        parts = [last[:, start:stop] for start, stop in self.persistence_slices]
        base = torch.cat(parts, dim=1)
        if self.persistence_proj is not None:
            base = self.persistence_proj(base)
        if base.shape[1] != self.output_dim:
            return None
        return base

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a window without applying cross-window context or heads."""
        b = x.shape[0]

        if self.disable_tsc:
            aug = x
        else:
            tsc_features = self.thermohaline_memory(x)
            aug = torch.cat([x, tsc_features], dim=2)

        t, c = aug.shape[1], aug.shape[2]
        h, w = aug.shape[3], aug.shape[4]
        flat = aug.reshape(b, t * c, h, w)

        local_features = self.local_branch(flat)

        branch_features = [local_features]
        if self.disable_spectral:
            pass  # skip
        else:
            branch_features.append(self.spectral_branch(flat))

        if self.disable_3d:
            pass  # skip
        else:
            branch_features.append(self.structure_branch(aug))

        fused = self.fusion(torch.cat(branch_features, dim=1))
        if self.fusion_transformer is not None:
            fused = fused + self.fusion_transformer(fused)
        return fused

    def build_global_token_bank(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Build raw cross-window tokens for exact two-pass inference."""
        if self.global_token_bank is None:
            return None
        return self.global_token_bank.pool_tokens(self.encode_features(x))

    def decode_features(
        self,
        fused: torch.Tensor,
        raw_x: torch.Tensor,
        global_bank_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, _, h, w = fused.shape
        if self.global_token_bank is not None:
            fused = self.global_token_bank(fused, bank_tokens=global_bank_tokens)

        if self.disable_ensemble:
            # Single head, no gate — direct prediction
            out_channels = self.pred_len * self.output_dim
            predictions = self.member_heads[0](fused).reshape(b, self.pred_len, self.output_dim, h, w)
        else:
            member_outputs = []
            for head in self.member_heads:
                member = head(fused).reshape(b, self.pred_len, self.output_dim, h, w)
                member_outputs.append(member)
            members = torch.stack(member_outputs, dim=1)
            weights = self.ensemble_gate(fused).unsqueeze(2).unsqueeze(3)
            predictions = (members * weights).sum(dim=1)

        persistence = self._persistence_base(raw_x)
        if persistence is not None and self.persistence_scale is not None:
            predictions = predictions + self.persistence_scale * persistence.unsqueeze(1)
        return predictions

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        global_bank_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets
        fused = self.encode_features(x)
        return self.decode_features(fused, x, global_bank_tokens=global_bank_tokens)


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
        self.seq_len = config.get('sequence_length', 10)  # 输入序列长度
        self.pred_len = config.get('prediction_length', 5)  # 预测序列长度

        # 计算实际的输入和输出维度
        if 'actual_input_channels' in config and 'actual_output_channels' in config:
            self.input_dim = config['actual_input_channels']
            self.output_dim = config['actual_output_channels']
            print(f"使用检测到的实际维度 - 输入: {self.input_dim}, 输出: {self.output_dim}")
        else:
            # 不再强制假设固定深度层数量，直接按变量数作为基准（仍保持向后兼容）
            assumed_depth_levels = config.get('assumed_depth_levels', 2)
            self.input_dim = len(config['input_variables']) * assumed_depth_levels
            self.output_dim = len(config['target_variables']) * assumed_depth_levels
            print(f"使用默认计算维度 (假设depth={assumed_depth_levels}) - 输入: {self.input_dim}, 输出: {self.output_dim}")

        self.encoder_input_dim = self.input_dim

        # ConvLSTM编码器
        self.encoder = ConvLSTM(
            input_dim=self.encoder_input_dim,
            hidden_dims=config.get('hidden_dims', [64, 64, 64]),
            kernel_size=config.get('kernel_size', (3, 3)),
            num_layers=config.get('num_layers', 3),
            batch_first=True,
            bias=True,
            return_all_layers=True,  # 返回所有层的状态，供解码器使用
            residual_between_layers=False,
            dropout=config.get('dropout', 0.0),
        )

        # ConvLSTM解码器
        self.decoder = ConvLSTM(
            input_dim=config.get('hidden_dims', [64, 64, 64])[-1],
            hidden_dims=config.get('hidden_dims', [64, 64, 64]),
            kernel_size=config.get('kernel_size', (3, 3)),
            num_layers=config.get('num_layers', 3),
            batch_first=True,
            bias=True,
            return_all_layers=True,  # 也设置为True以保持一致性
            residual_between_layers=False,
            dropout=config.get('dropout', 0.0),
        )

        # 输出投影层
        final_hidden_dim = config.get('hidden_dims', [64, 64, 64])[-1]
        self.output_proj = nn.Conv2d(
            in_channels=final_hidden_dim,
            out_channels=self.output_dim,
            kernel_size=1,
            padding=0
        )

        # 旧 ConvLSTM 路径保留为简洁基线；TSC-Fusion 的细化由主干内部实现。
        dropout = config.get('dropout', 0.0)
        self.residual_refiner = ResidualRefiner(final_hidden_dim, dropout=dropout)

        # 批归一化
        self.batch_norm = nn.BatchNorm2d(self.output_dim)

        # Dropout正则化
        self.dropout = nn.Dropout2d(config.get('dropout', 0.1))

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, channels, height, width)

        Returns:
            预测输出 (batch_size, pred_len, output_channels, height, width)
        """
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

            # 细化解码器输出
            if self.residual_refiner is not None:
                current_output = self.residual_refiner(current_output)

            # 投影到目标维度
            pred = self.output_proj(current_output)
            pred = self.batch_norm(pred)
            pred = self.dropout(pred)

            predictions.append(pred)

            # 使用当前输出作为下一时步的输入
            decoder_input = current_output.unsqueeze(1)

        # 堆叠预测结果
        return torch.stack(predictions, dim=1)

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


class SimpleCNN(nn.Module):
    """
    简单的CNN基准模型
    将时间维度堆叠到通道维度进行预测
    """
    def __init__(self, config: dict):
        super(SimpleCNN, self).__init__()
        self.config = config

        # 计算输入输出维度
        if 'actual_input_channels' in config and 'actual_output_channels' in config:
            self.input_dim = config['actual_input_channels']
            self.output_dim = config['actual_output_channels']
        else:
            assumed_depth_levels = config.get('assumed_depth_levels', 2)
            self.input_dim = len(config['input_variables']) * assumed_depth_levels
            self.output_dim = len(config['target_variables']) * assumed_depth_levels

        self.seq_len = config.get('sequence_length', 10)
        self.pred_len = config.get('prediction_length', 5)

        # 输入通道数 = 特征数 * 序列长度
        self.in_channels = self.input_dim * self.seq_len
        # 输出通道数 = 目标特征数 * 预测长度
        self.out_channels = self.output_dim * self.pred_len

        hidden_dims = config.get('hidden_dims', [64, 128, 64])

        layers = []
        # 第一层
        layers.append(nn.Conv2d(self.in_channels, hidden_dims[0], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_dims[0]))
        layers.append(nn.ReLU(inplace=True))

        # 中间层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

        # 输出层
        layers.append(nn.Conv2d(hidden_dims[-1], self.out_channels, kernel_size=1))

        self.cnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, channels, height, width)
        b, s, c, h, w = x.shape
        # Flatten seq and channels: (batch, seq*channels, height, width)
        x = x.view(b, s * c, h, w)

        out = self.cnn(x)

        # Reshape output: (batch, pred_len, out_channels, height, width)
        out = out.view(b, self.pred_len, self.output_dim, h, w)
        return out


def create_ocean_model(config: dict) -> nn.Module:
    """
    创建海洋预测模型的工厂函数

    Args:
        config: 配置字典

    Returns:
        配置好的海洋预测模型
    """
    model_type = config.get('model_type', 'convlstm')
    try:
        from paper_reimplementation_models import (
            create_paper_reimplementation_model,
            is_paper_reimplementation,
        )
        if is_paper_reimplementation(str(model_type)):
            return create_paper_reimplementation_model(config)
    except ImportError:
        pass

    normalized_type = str(model_type).lower()
    if normalized_type == 'cnn':
        return SimpleCNN(config)
    if normalized_type in {'tsc_fusion', 'tscglobal', 'tsc_global_axiom_ensemble',
                            'tsc-spectrum-axiom-ensemble', 'tsc_spectrum_axiom_ensemble'}:
        return TSCFusionNet(config)
    if normalized_type == 'convlstm':
        return OceanConvLSTMPredictor(config)
    raise ValueError(f'未知 model_type: {model_type!r}；拒绝静默回退到 ConvLSTM')


# Backward-compatible import aliases. They do not imply equivalence to any
# external named model; formal reports use the generic TSCFusionNet name.
GlobalSpectralBranch = LowModeSpectralBranch
Axiom3DStructureBranch = SpatiotemporalStructureBranch
TSCGlobalAxiomEnsembleNet = TSCFusionNet


# 默认配置
DEFAULT_CONFIG = {
    'input_variables': ["TEMP", "SALT", "SSHA", "UWND", "VWND"],  # 使用实际存在的变量
    'target_variables': ["TEMP", "SALT"],
    'sequence_length': 10,
    'prediction_length': 5,
    'hidden_dims': [64, 96, 128, 128],
    'kernel_size': (3, 3),
    'num_layers': 4,
    'dropout': 0.2,
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
    # channels = len(config['input_variables'])
    # 根据模型初始化逻辑，输入维度应该是 input_variables * assumed_depth_levels
    channels = model.input_dim
    height, width = 32, 21  # 根据数据描述的经纬度范围计算

    test_input = torch.randn(batch_size, seq_len, channels, height, width)

    print(f"模型配置: {config}")
    print(f"输入形状: {test_input.shape}")

    # 前向传播测试
    output = model(test_input)
    print(f"输出形状: {output.shape}")
    print("ConvLSTM模型创建成功！")
