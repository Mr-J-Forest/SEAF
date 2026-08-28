#!/usr/bin/env python3
"""OceanForecastBench architecture adapters for the ORAS5 forecast contract.

The three backbones below preserve the defining mechanisms used by the public
OceanForecastBench baselines: AFNO filtering (FourCastNet), variable
tokenization/aggregation (ClimaX), and hierarchical shifted-window attention
(SwinTransformer).  The input stem and output head are intentionally adapted
to this repository's ``(B, T, C, H, W) -> (B, lead, C_out, H, W)`` contract.

These are train-from-scratch architecture adapters, not official pretrained
models and not reproductions of published OceanForecastBench scores.  Source
and license provenance is recorded in each experiment config and in
``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bounds(value, *, name: str) -> tuple[int, int]:
    if isinstance(value, slice):
        start, stop = value.start, value.stop
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        start, stop = value[0], value[1]
    else:
        raise ValueError(f"{name} 缺少有效的 channel slice: {value!r}")
    if start is None or stop is None or int(start) < 0 or int(stop) <= int(start):
        raise ValueError(f"{name} channel slice 非法: {value!r}")
    return int(start), int(stop)


def _ordered_channel_slices(config: dict, input_dim: int) -> list[tuple[str, int, int]]:
    raw = config.get("input_channel_slices", {})
    if not isinstance(raw, dict) or not raw:
        raise ValueError("ClimaX adapter 要求显式 input_channel_slices")
    items = [(str(name), *_bounds(value, name=str(name))) for name, value in raw.items()]
    items.sort(key=lambda item: item[1])
    cursor = 0
    for name, start, stop in items:
        if start != cursor:
            raise ValueError(
                "ClimaX variable tokenization 要求 channel slices 无重叠且完整覆盖输入；"
                f"在 {name!r} 前期望起点 {cursor}，实际 {start}"
            )
        cursor = stop
    if cursor != input_dim:
        raise ValueError(
            "ClimaX variable tokenization 未覆盖全部输入通道："
            f"covered={cursor}, actual_input_channels={input_dim}"
        )
    return items


def _pad_to_patch(x: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, int, int]:
    height, width = x.shape[-2:]
    pad_h = (-height) % patch_size
    pad_w = (-width) % patch_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, height, width


def _unpatchify(
    values: torch.Tensor,
    patch_size: int,
    output_channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert ``(B, Hp, Wp, patch^2*C)`` tokens back to a cropped field."""
    batch, patch_h, patch_w, features = values.shape
    expected = patch_size * patch_size * output_channels
    if features != expected:
        raise ValueError(f"patch head 输出 {features}，期望 {expected}")
    field = values.reshape(
        batch, patch_h, patch_w, patch_size, patch_size, output_channels
    )
    field = field.permute(0, 5, 1, 3, 2, 4).contiguous()
    field = field.reshape(
        batch, output_channels, patch_h * patch_size, patch_w * patch_size
    )
    return field[:, :, :height, :width]


def _sincos_position(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a deterministic rectangular 2-D sine/cosine position encoding."""
    if dim % 4 != 0:
        raise ValueError(f"position embedding dim 必须能被4整除，实际 {dim}")
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = torch.exp(-math.log(10000.0) * omega / max(1, quarter - 1))
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    y_phase = y[:, None] * omega[None, :]
    x_phase = x[:, None] * omega[None, :]
    y_embed = torch.cat((y_phase.sin(), y_phase.cos()), dim=-1)
    x_embed = torch.cat((x_phase.sin(), x_phase.cos()), dim=-1)
    position = torch.cat(
        (
            y_embed[:, None, :].expand(-1, width, -1),
            x_embed[None, :, :].expand(height, -1, -1),
        ),
        dim=-1,
    )
    return position.unsqueeze(0).to(dtype=dtype)


class DropPath(nn.Module):
    """Per-sample stochastic depth without a timm dependency."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep


def _drop_path_schedule(depth: int, maximum: float) -> list[float]:
    if depth <= 0:
        return []
    return torch.linspace(0.0, float(maximum), depth).tolist()


class AnomalyForecastModel(nn.Module):
    """Shared direct anomaly-forecast contract for recent baselines."""

    baseline_kind = "oceanforecastbench_architecture_adapter"

    def __init__(self, config: dict):
        super().__init__()
        self.sequence_length = int(config.get("sequence_length", 1))
        self.prediction_length = int(config.get("prediction_length", 1))
        self.input_dim = int(config.get("actual_input_channels", 0))
        self.output_dim = int(config.get("actual_output_channels", 0))
        if min(
            self.sequence_length,
            self.prediction_length,
            self.input_dim,
            self.output_dim,
        ) <= 0:
            raise ValueError("recent baseline 的 sequence/prediction/channel 维度必须为正")


    def _flatten_history(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"模型输入必须为 (B,T,C,H,W)，实际 {tuple(x.shape)}")
        batch, steps, channels, height, width = x.shape
        if steps != self.sequence_length or channels != self.input_dim:
            raise ValueError(
                "模型输入协议不一致："
                f"T/C={steps}/{channels}，期望 {self.sequence_length}/{self.input_dim}"
            )
        return x.reshape(batch, steps * channels, height, width)

    def _finish_forecast(self, flat_forecast: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = flat_forecast.shape
        expected = self.prediction_length * self.output_dim
        if channels != expected:
            raise ValueError(f"预测 head 输出 {channels} 通道，期望 {expected}")
        return flat_forecast.reshape(
            batch, self.prediction_length, self.output_dim, height, width
        )


class AFNO2D(nn.Module):
    """Adaptive Fourier neural operator used by FourCastNet."""

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        sparsity_threshold: float,
        hard_thresholding_fraction: float,
        hidden_size_factor: int = 1,
    ):
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError("AFNO hidden size 必须能被 num_blocks 整除")
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.hidden_size_factor = int(hidden_size_factor)
        self.sparsity_threshold = float(sparsity_threshold)
        self.hard_thresholding_fraction = float(hard_thresholding_fraction)
        scale = 0.02
        expanded = self.block_size * self.hidden_size_factor
        self.w1 = nn.Parameter(
            scale * torch.randn(2, num_blocks, self.block_size, expanded)
        )
        self.b1 = nn.Parameter(scale * torch.randn(2, num_blocks, expanded))
        self.w2 = nn.Parameter(
            scale * torch.randn(2, num_blocks, expanded, self.block_size)
        )
        self.b2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        original_dtype = x.dtype
        batch, height, width, channels = x.shape
        spectrum = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")
        spectrum = spectrum.reshape(
            batch,
            height,
            width // 2 + 1,
            self.num_blocks,
            self.block_size,
        )
        real, imag = spectrum.real, spectrum.imag
        hidden_real = F.gelu(
            torch.einsum("...bi,bio->...bo", real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", imag, self.w1[1])
            + self.b1[0]
        )
        hidden_imag = F.gelu(
            torch.einsum("...bi,bio->...bo", imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", real, self.w1[1])
            + self.b1[1]
        )
        output_real = (
            torch.einsum("...bi,bio->...bo", hidden_real, self.w2[0])
            - torch.einsum("...bi,bio->...bo", hidden_imag, self.w2[1])
            + self.b2[0]
        )
        output_imag = (
            torch.einsum("...bi,bio->...bo", hidden_imag, self.w2[0])
            + torch.einsum("...bi,bio->...bo", hidden_real, self.w2[1])
            + self.b2[1]
        )

        fraction = self.hard_thresholding_fraction
        if fraction < 1.0:
            keep_h = max(1, int(math.ceil(height * fraction / 2.0)))
            keep_w = max(1, int(math.ceil((width // 2 + 1) * fraction)))
            mask = torch.zeros(
                (height, width // 2 + 1), dtype=output_real.dtype, device=x.device
            )
            mask[:keep_h, :keep_w] = 1
            mask[-keep_h:, :keep_w] = 1
            output_real = output_real * mask[None, :, :, None, None]
            output_imag = output_imag * mask[None, :, :, None, None]

        output_real = F.softshrink(output_real, lambd=self.sparsity_threshold)
        output_imag = F.softshrink(output_imag, lambd=self.sparsity_threshold)
        filtered = torch.complex(output_real, output_imag).reshape(
            batch, height, width // 2 + 1, channels
        )
        filtered = torch.fft.irfft2(
            filtered, s=(height, width), dim=(1, 2), norm="ortho"
        )
        return filtered.to(dtype=original_dtype) + residual


class AFNOBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_blocks: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        sparsity_threshold: float,
        hard_thresholding_fraction: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = AFNO2D(
            dim,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        x = x + self.drop_path(self.filter(normalized) - normalized)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class OceanForecastBenchFourCastNet(AnomalyForecastModel):
    """FourCastNet/AFNO architecture adapter trained on the local ORAS5 task."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.patch_size = int(config.get("baseline_patch_size", 2))
        embed_dim = int(config.get("baseline_embed_dim", 192))
        depth = int(config.get("baseline_depth", 8))
        num_blocks = int(config.get("afno_num_blocks", 8))
        if embed_dim % 4 != 0 or embed_dim % num_blocks != 0:
            raise ValueError("FourCastNet adapter embed_dim 必须能被4和 afno_num_blocks整除")
        dropout = float(config.get("baseline_drop_rate", 0.0))
        rates = _drop_path_schedule(
            depth, float(config.get("baseline_drop_path_rate", 0.0))
        )
        self.patch_embed = nn.Conv2d(
            self.sequence_length * self.input_dim,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.blocks = nn.ModuleList(
            [
                AFNOBlock(
                    embed_dim,
                    num_blocks=num_blocks,
                    mlp_ratio=float(config.get("baseline_mlp_ratio", 4.0)),
                    dropout=dropout,
                    drop_path=rates[index],
                    sparsity_threshold=float(config.get("afno_sparsity_threshold", 0.01)),
                    hard_thresholding_fraction=float(
                        config.get("afno_hard_thresholding_fraction", 1.0)
                    ),
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(
            embed_dim,
            self.prediction_length
            * self.output_dim
            * self.patch_size
            * self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat, height, width = _pad_to_patch(
            self._flatten_history(x), self.patch_size
        )
        tokens = self.patch_embed(flat).permute(0, 2, 3, 1)
        tokens = tokens + _sincos_position(
            tokens.shape[1],
            tokens.shape[2],
            tokens.shape[3],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        for block in self.blocks:
            tokens = block(tokens)
        values = self.head(self.norm(tokens))
        field = _unpatchify(
            values,
            self.patch_size,
            self.prediction_length * self.output_dim,
            height,
            width,
        )
        return self._finish_forecast(field)


class TokenTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = x + self.drop_path(attended)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class OceanForecastBenchClimaX(AnomalyForecastModel):
    """ClimaX variable-tokenization architecture adapter for ORAS5 histories."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.patch_size = int(config.get("baseline_patch_size", 2))
        embed_dim = int(config.get("baseline_embed_dim", 192))
        depth = int(config.get("baseline_depth", 6))
        heads = int(config.get("baseline_num_heads", 8))
        if embed_dim % 4 != 0 or embed_dim % heads != 0:
            raise ValueError("ClimaX adapter embed_dim 必须能被4和 num_heads整除")
        self.variable_slices = _ordered_channel_slices(config, self.input_dim)
        self.token_embeds = nn.ModuleList(
            [
                nn.Conv2d(
                    self.sequence_length * (stop - start),
                    embed_dim,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                )
                for _, start, stop in self.variable_slices
            ]
        )
        self.variable_embed = nn.Parameter(
            torch.zeros(1, 1, len(self.variable_slices), embed_dim)
        )
        self.variable_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.variable_aggregation = nn.MultiheadAttention(
            embed_dim,
            heads,
            dropout=float(config.get("baseline_attention_dropout", 0.0)),
            batch_first=True,
        )
        nn.init.trunc_normal_(self.variable_embed, std=0.02)
        nn.init.trunc_normal_(self.variable_query, std=0.02)

        dropout = float(config.get("baseline_drop_rate", 0.1))
        rates = _drop_path_schedule(
            depth, float(config.get("baseline_drop_path_rate", 0.2))
        )
        self.position_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TokenTransformerBlock(
                    embed_dim,
                    heads,
                    mlp_ratio=float(config.get("baseline_mlp_ratio", 4.0)),
                    dropout=dropout,
                    attention_dropout=float(
                        config.get("baseline_attention_dropout", 0.0)
                    ),
                    drop_path=rates[index],
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(
            embed_dim,
            self.prediction_length
            * self.output_dim
            * self.patch_size
            * self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"模型输入必须为 (B,T,C,H,W)，实际 {tuple(x.shape)}")
        batch, steps, channels, height, width = x.shape
        if steps != self.sequence_length or channels != self.input_dim:
            raise ValueError("ClimaX adapter 输入 T/C 与配置不一致")

        variable_tokens = []
        patch_h = patch_w = None
        for embed, (_, start, stop) in zip(self.token_embeds, self.variable_slices):
            variable = x[:, :, start:stop].reshape(
                batch, steps * (stop - start), height, width
            )
            variable, _, _ = _pad_to_patch(variable, self.patch_size)
            encoded = embed(variable)
            patch_h, patch_w = encoded.shape[-2:]
            variable_tokens.append(encoded.flatten(2).transpose(1, 2))

        stacked = torch.stack(variable_tokens, dim=2) + self.variable_embed
        patch_count = stacked.shape[1]
        stacked = stacked.reshape(
            batch * patch_count, len(self.variable_slices), stacked.shape[-1]
        )
        query = self.variable_query.expand(batch * patch_count, -1, -1)
        tokens, _ = self.variable_aggregation(
            query, stacked, stacked, need_weights=False
        )
        tokens = tokens.squeeze(1).reshape(batch, patch_count, -1)
        position = _sincos_position(
            int(patch_h),
            int(patch_w),
            tokens.shape[-1],
            device=tokens.device,
            dtype=tokens.dtype,
        ).reshape(1, patch_count, -1)
        tokens = self.position_dropout(tokens + position)
        for block in self.blocks:
            tokens = block(tokens)
        values = self.head(self.norm(tokens)).reshape(
            batch, int(patch_h), int(patch_w), -1
        )
        field = _unpatchify(
            values,
            self.patch_size,
            self.prediction_length * self.output_dim,
            height,
            width,
        )
        return self._finish_forecast(field)


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    return (
        x.reshape(
            batch,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(-1, window_size * window_size, channels)
    )


def _window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    windows_per_sample = (height // window_size) * (width // window_size)
    batch = windows.shape[0] // windows_per_sample
    channels = windows.shape[-1]
    return (
        windows.reshape(
            batch,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(batch, height, width, channels)
    )


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        heads: int,
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Swin attention dim 必须能被 heads 整除")
        self.dim = dim
        self.window_size = window_size
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(dropout)

        relative_count = (2 * window_size - 1) ** 2
        self.relative_position_bias = nn.Parameter(
            torch.zeros(relative_count, heads)
        )
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        ).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer(
            "relative_position_index", relative.sum(-1), persistent=False
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows, tokens, 3, self.heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        relative = self.relative_position_bias[
            self.relative_position_index.reshape(-1)
        ]
        relative = relative.reshape(tokens, tokens, self.heads).permute(2, 0, 1)
        attention = attention + relative.unsqueeze(0)
        if attention_mask is not None:
            window_count = attention_mask.shape[0]
            attention = attention.reshape(
                batch_windows // window_count,
                window_count,
                self.heads,
                tokens,
                tokens,
            )
            attention = attention + attention_mask[None, :, None].to(attention.dtype)
            attention = attention.reshape(
                batch_windows, self.heads, tokens, tokens
            )
        attention = self.attention_dropout(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(
            batch_windows, tokens, channels
        )
        return self.projection_dropout(self.proj(output))


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(
            dim,
            window_size=window_size,
            heads=heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def _attention_mask(
        self, padded_height: int, padded_width: int, device: torch.device
    ) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        mask = torch.zeros((1, padded_height, padded_width, 1), device=device)
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        label = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                mask[:, height_slice, width_slice] = label
                label += 1
        windows = _window_partition(mask, self.window_size).squeeze(-1)
        difference = windows.unsqueeze(1) - windows.unsqueeze(2)
        return difference.masked_fill(difference != 0, -100.0).masked_fill(
            difference == 0, 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        shortcut = x
        normalized = self.norm1(x)
        pad_h = (-height) % self.window_size
        pad_w = (-width) % self.window_size
        normalized = F.pad(normalized, (0, 0, 0, pad_w, 0, pad_h))
        padded_height, padded_width = normalized.shape[1:3]
        mask = self._attention_mask(padded_height, padded_width, x.device)
        if self.shift_size:
            normalized = torch.roll(
                normalized,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        windows = _window_partition(normalized, self.window_size)
        windows = self.attention(windows, mask)
        normalized = _window_reverse(
            windows, self.window_size, padded_height, padded_width
        )
        if self.shift_size:
            normalized = torch.roll(
                normalized,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        normalized = normalized[:, :height, :width]
        x = shortcut + self.drop_path(normalized)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[1:3]
        if height % 2 or width % 2:
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        merged = torch.cat(
            (
                x[:, 0::2, 0::2],
                x[:, 1::2, 0::2],
                x[:, 0::2, 1::2],
                x[:, 1::2, 1::2],
            ),
            dim=-1,
        )
        return self.reduction(self.norm(merged))


class PatchExpansion(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.expand = nn.Linear(input_dim, 4 * output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self, x: torch.Tensor, target_height: int, target_width: int
    ) -> torch.Tensor:
        batch, height, width, _ = x.shape
        x = self.expand(x).reshape(
            batch, height, width, 2, 2, self.output_dim
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(
            batch, height * 2, width * 2, self.output_dim
        )
        return self.norm(x[:, :target_height, :target_width])


def _make_swin_stage(
    *,
    dim: int,
    depth: int,
    heads: int,
    window_size: int,
    mlp_ratio: float,
    dropout: float,
    attention_dropout: float,
    drop_paths: Iterable[float],
) -> nn.ModuleList:
    rates = list(drop_paths)
    return nn.ModuleList(
        [
            SwinBlock(
                dim,
                heads=heads,
                window_size=window_size,
                shift_size=0 if index % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=rates[index],
            )
            for index in range(depth)
        ]
    )


class OceanForecastBenchSwin(AnomalyForecastModel):
    """Hierarchical shifted-window Swin adapter for the ORAS5 forecast task."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.patch_size = int(config.get("baseline_patch_size", 2))
        embed_dim = int(config.get("baseline_embed_dim", 96))
        depths = tuple(int(value) for value in config.get("swin_depths", [2, 4, 2]))
        heads = tuple(int(value) for value in config.get("swin_num_heads", [4, 8, 4]))
        if len(depths) != 3 or len(heads) != 3:
            raise ValueError("Swin adapter 要求3个 encoder/bottleneck/decoder stage")
        if any(depth <= 0 for depth in depths):
            raise ValueError("Swin stage depth 必须为正")
        if embed_dim % heads[0] or (2 * embed_dim) % heads[1] or embed_dim % heads[2]:
            raise ValueError("Swin stage 通道数必须能被对应 attention heads 整除")

        dropout = float(config.get("baseline_drop_rate", 0.1))
        attention_dropout = float(config.get("baseline_attention_dropout", 0.0))
        total_depth = sum(depths)
        rates = _drop_path_schedule(
            total_depth, float(config.get("baseline_drop_path_rate", 0.2))
        )
        offsets = (0, depths[0], depths[0] + depths[1])
        stage_kwargs = {
            "window_size": int(config.get("swin_window_size", 4)),
            "mlp_ratio": float(config.get("baseline_mlp_ratio", 4.0)),
            "dropout": dropout,
            "attention_dropout": attention_dropout,
        }
        self.patch_embed = nn.Conv2d(
            self.sequence_length * self.input_dim,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.stage1 = _make_swin_stage(
            dim=embed_dim,
            depth=depths[0],
            heads=heads[0],
            drop_paths=rates[offsets[0]: offsets[0] + depths[0]],
            **stage_kwargs,
        )
        self.merge = PatchMerging(embed_dim)
        self.stage2 = _make_swin_stage(
            dim=2 * embed_dim,
            depth=depths[1],
            heads=heads[1],
            drop_paths=rates[offsets[1]: offsets[1] + depths[1]],
            **stage_kwargs,
        )
        self.expand = PatchExpansion(2 * embed_dim, embed_dim)
        self.fuse = nn.Linear(2 * embed_dim, embed_dim)
        self.stage3 = _make_swin_stage(
            dim=embed_dim,
            depth=depths[2],
            heads=heads[2],
            drop_paths=rates[offsets[2]: offsets[2] + depths[2]],
            **stage_kwargs,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(
            embed_dim,
            self.prediction_length
            * self.output_dim
            * self.patch_size
            * self.patch_size,
        )

    @staticmethod
    def _run_stage(stage: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for block in stage:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat, height, width = _pad_to_patch(
            self._flatten_history(x), self.patch_size
        )
        tokens = self.patch_embed(flat).permute(0, 2, 3, 1)
        tokens = self._run_stage(self.stage1, tokens)
        skip = tokens
        tokens = self._run_stage(self.stage2, self.merge(tokens))
        tokens = self.expand(tokens, skip.shape[1], skip.shape[2])
        tokens = self.fuse(torch.cat((skip, tokens), dim=-1))
        tokens = self._run_stage(self.stage3, tokens)
        values = self.head(self.norm(tokens))
        field = _unpatchify(
            values,
            self.patch_size,
            self.prediction_length * self.output_dim,
            height,
            width,
        )
        return self._finish_forecast(field)


RECENT_BASELINE_REGISTRY = {
    "ofb_fourcastnet": OceanForecastBenchFourCastNet,
    "ofb-fourcastnet": OceanForecastBenchFourCastNet,
    "ofb_climax": OceanForecastBenchClimaX,
    "ofb-climax": OceanForecastBenchClimaX,
    "ofb_swin": OceanForecastBenchSwin,
    "ofb-swin": OceanForecastBenchSwin,
}


def is_recent_baseline(model_type: str) -> bool:
    return model_type.lower() in RECENT_BASELINE_REGISTRY


def create_recent_baseline(config: dict) -> nn.Module:
    model_type = str(config.get("model_type", "")).lower()
    try:
        model_class = RECENT_BASELINE_REGISTRY[model_type]
    except KeyError as exc:
        available = ", ".join(sorted(RECENT_BASELINE_REGISTRY))
        raise ValueError(
            f"Unknown recent baseline {model_type!r}. Available: {available}"
        ) from exc
    return model_class(config)
