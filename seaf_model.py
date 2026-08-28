"""SEAF: spectral-ensemble forecasting of joint ocean anomalies."""

from typing import List, Optional

import torch
import torch.nn as nn


def _adaptive_group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class ConvGNAct2d(nn.Module):
    """Small convolutional block used by the encoder and residual heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_adaptive_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralLowModeMixer(nn.Module):
    """Learn a diagonal filter over retained low spatial Fourier modes."""

    def __init__(
        self,
        channels: int,
        modes_y: int,
        modes_x: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)
        scale = 0.02
        self.weight_pos = nn.Parameter(
            scale * torch.randn(channels, self.modes_y, self.modes_x, 2)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(channels, self.modes_y, self.modes_x, 2)
        )
        self.mix = ConvGNAct2d(channels, channels, kernel_size=1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        modes_y = max(1, min(self.modes_y, height // 2 if height > 1 else 1))
        modes_x = max(1, min(self.modes_x, width // 2 + 1))

        spectrum = torch.fft.rfft2(x.float(), norm="ortho")
        filtered = torch.zeros_like(spectrum)
        positive = torch.view_as_complex(
            self.weight_pos[:, :modes_y, :modes_x].contiguous()
        ).to(spectrum.dtype)
        negative = torch.view_as_complex(
            self.weight_neg[:, :modes_y, :modes_x].contiguous()
        ).to(spectrum.dtype)
        filtered[:, :, :modes_y, :modes_x] = (
            spectrum[:, :, :modes_y, :modes_x] * positive.unsqueeze(0)
        )
        filtered[:, :, -modes_y:, :modes_x] = (
            spectrum[:, :, -modes_y:, :modes_x] * negative.unsqueeze(0)
        )
        spatial = torch.fft.irfft2(
            filtered, s=(height, width), norm="ortho"
        ).to(dtype=x.dtype)
        return x + self.mix(spatial)


class LowModeSpectralEncoder(nn.Module):
    """Encode flattened history channels with low-mode spatial mixing."""

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        modes_y: int,
        modes_x: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = [
            ConvGNAct2d(input_channels, hidden_dim, dropout=dropout)
        ]
        for _ in range(max(1, layers)):
            blocks.extend(
                [
                    SpectralLowModeMixer(
                        hidden_dim,
                        modes_y=modes_y,
                        modes_x=modes_x,
                        dropout=dropout,
                    ),
                    ConvGNAct2d(
                        hidden_dim, hidden_dim, kernel_size=1, dropout=dropout
                    ),
                ]
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NonSpectralControlEncoder(nn.Module):
    """Matched spatial control used only by the no-spectral ablation."""

    def __init__(
        self, input_channels: int, hidden_dim: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = [
            ConvGNAct2d(input_channels, hidden_dim, dropout=dropout)
        ]
        for _ in range(max(1, layers)):
            blocks.append(ConvGNAct2d(hidden_dim, hidden_dim, dropout=dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SEAFNet(nn.Module):
    """Formal SEAF model: spectral encoder and spatial ensemble gate."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.sequence_length = int(config.get("sequence_length", 12))
        self.prediction_length = int(config.get("prediction_length", 5))
        self.target_variables = list(config.get("target_variables", []))
        self.input_dim = int(
            config.get(
                "actual_input_channels",
                len(config.get("input_variables", []))
                * int(config.get("assumed_depth_levels", 2)),
            )
        )
        self.output_dim = int(
            config.get(
                "actual_output_channels",
                len(self.target_variables)
                * int(config.get("assumed_depth_levels", 2)),
            )
        )

        hidden_dim = int(config.get("seaf_hidden_dim", 64))
        dropout = float(config.get("dropout", 0.05))
        spectral_modes = list(config.get("seaf_spectral_modes", [8, 8]))
        spectral_layers = int(config.get("seaf_spectral_layers", 2))
        flattened_channels = self.sequence_length * self.input_dim

        self.disable_spectral = bool(config.get("ablation_disable_spectral", False))
        self.disable_ensemble = bool(config.get("ablation_disable_ensemble", False))
        if self.disable_spectral:
            self.spectral_encoder = NonSpectralControlEncoder(
                flattened_channels,
                hidden_dim,
                layers=spectral_layers,
                dropout=dropout,
            )
        else:
            self.spectral_encoder = LowModeSpectralEncoder(
                flattened_channels,
                hidden_dim,
                modes_y=int(spectral_modes[0]),
                modes_x=int(spectral_modes[1]),
                layers=spectral_layers,
                dropout=dropout,
            )

        member_count = (
            1
            if self.disable_ensemble
            else int(config.get("seaf_ensemble_members", 4))
        )
        output_channels = self.prediction_length * self.output_dim
        self.member_heads = nn.ModuleList(
            [
                nn.Sequential(
                    ConvGNAct2d(hidden_dim, hidden_dim, dropout=dropout),
                    nn.Conv2d(hidden_dim, output_channels, kernel_size=1),
                )
                for _ in range(member_count)
            ]
        )
        gate_hidden = max(8, hidden_dim // 2)
        self.ensemble_gate = None if self.disable_ensemble else nn.Sequential(
            nn.Conv2d(hidden_dim, gate_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, member_count, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del targets
        batch, sequence, channels, height, width = x.shape
        if sequence != self.sequence_length or channels != self.input_dim:
            raise ValueError(
                "SEAF input shape does not match the configured history: "
                f"got (T={sequence}, C={channels}), expected "
                f"(T={self.sequence_length}, C={self.input_dim})"
            )
        features = self.spectral_encoder(
            x.reshape(batch, sequence * channels, height, width)
        )
        members = torch.stack(
            [
                head(features).reshape(
                    batch,
                    self.prediction_length,
                    self.output_dim,
                    height,
                    width,
                )
                for head in self.member_heads
            ],
            dim=1,
        )
        if self.ensemble_gate is None:
            forecast = members[:, 0]
        else:
            weights = self.ensemble_gate(features).unsqueeze(2).unsqueeze(3)
            forecast = (members * weights).sum(dim=1)
        return forecast
