#!/usr/bin/env python3
"""Paper-level reimplementations for baselines without public official code.

These classes are not official model releases. They are explicitly marked
paper reimplementations so experiments can distinguish them from official
weights such as GLONET/WenHai.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _dims(config: dict) -> tuple[int, int, int, int]:
    seq_len = int(config.get("sequence_length", 12))
    pred_len = int(config.get("prediction_length", 5))
    in_ch = int(config.get("actual_input_channels", len(config.get("input_variables", []))))
    out_ch = int(config.get("actual_output_channels", len(config.get("target_variables", []))))
    return seq_len, pred_len, in_ch, out_ch


def _group_norm(channels: int) -> nn.Module:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            _group_norm(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 8, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        heads = max(1, min(heads, channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class PaperBase(nn.Module):
    official_source = "paper_reimplementation"

    def __init__(self, config: dict, hidden: int = 64):
        super().__init__()
        self.seq_len, self.pred_len, self.input_dim, self.output_dim = _dims(config)
        self.in_channels = self.seq_len * self.input_dim
        self.out_channels = self.pred_len * self.output_dim
        self.hidden = hidden

    def flatten_time(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        return x.reshape(b, t * c, h, w)

    def restore(self, y: torch.Tensor) -> torch.Tensor:
        b, _, h, w = y.shape
        return y.reshape(b, self.pred_len, self.output_dim, h, w)


class TianHaiPaperReimplementation(PaperBase):
    """Hierarchical air-sea interaction proxy with temporal gates and attention."""

    def __init__(self, config: dict):
        super().__init__(config, hidden=int(config.get("paper_hidden_dim", 64)))
        d = float(config.get("dropout", 0.05))
        self.temporal_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, self.hidden, 1),
            nn.GELU(),
            nn.Conv2d(self.hidden, self.in_channels, 1),
            nn.Sigmoid(),
        )
        self.encoder = nn.Sequential(
            ConvGNAct(self.in_channels, self.hidden, dropout=d),
            ConvGNAct(self.hidden, self.hidden, dropout=d),
            SpatialTransformerBlock(self.hidden, heads=8, layers=2, dropout=d),
        )
        self.head = nn.Conv2d(self.hidden, self.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.flatten_time(x)
        z = z * self.temporal_gate(z)
        return self.restore(self.head(self.encoder(z)))


class FuXiOceanPaperReimplementation(PaperBase):
    """Multi-timescale autoregressive branch reimplementation."""

    def __init__(self, config: dict):
        super().__init__(config, hidden=int(config.get("paper_hidden_dim", 64)))
        d = float(config.get("dropout", 0.05))
        self.short = ConvGNAct(self.input_dim * min(3, self.seq_len), self.hidden, dropout=d)
        self.medium = ConvGNAct(self.input_dim * min(6, self.seq_len), self.hidden, dropout=d)
        self.long = ConvGNAct(self.in_channels, self.hidden, dropout=d)
        self.mix = nn.Sequential(nn.Conv2d(self.hidden * 3, self.hidden, 1), nn.GELU())
        self.transform = SpatialTransformerBlock(self.hidden, heads=8, layers=2, dropout=d)
        self.head = nn.Conv2d(self.hidden, self.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.short(self.flatten_time(x[:, -min(3, self.seq_len):]))
        b = self.medium(self.flatten_time(x[:, -min(6, self.seq_len):]))
        c = self.long(self.flatten_time(x))
        return self.restore(self.head(self.transform(self.mix(torch.cat([a, b, c], dim=1)))))


class FuXiONSPaperReimplementation(PaperBase):
    """Operational ensemble-style reimplementation with learned members."""

    def __init__(self, config: dict):
        super().__init__(config, hidden=int(config.get("paper_hidden_dim", 48)))
        d = float(config.get("dropout", 0.05))
        members = int(config.get("paper_ensemble_members", 4))
        self.members = nn.ModuleList([
            nn.Sequential(
                ConvGNAct(self.in_channels, self.hidden, kernel=3 if i % 2 == 0 else 5, dropout=d),
                ConvGNAct(self.hidden, self.hidden, dropout=d),
                nn.Conv2d(self.hidden, self.out_channels, 1),
            )
            for i in range(members)
        ])
        self.weights = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.in_channels, members, 1), nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.flatten_time(x)
        w = self.weights(z)
        outputs = torch.stack([member(z) for member in self.members], dim=1)
        return self.restore((outputs * w.unsqueeze(2)).sum(dim=1))


class AxiomOceanPaperReimplementation(PaperBase):
    """3D time-channel encoder plus 2D global refinement reimplementation."""

    def __init__(self, config: dict):
        super().__init__(config, hidden=int(config.get("paper_hidden_dim", 48)))
        d = float(config.get("dropout", 0.05))
        self.temporal3d = nn.Sequential(
            nn.Conv3d(self.input_dim, self.hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(self.hidden),
            nn.GELU(),
            nn.Dropout3d(d) if d > 0 else nn.Identity(),
            nn.Conv3d(self.hidden, self.hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(self.hidden),
            nn.GELU(),
        )
        self.refiner = nn.Sequential(
            ConvGNAct(self.hidden, self.hidden, dropout=d),
            SpatialTransformerBlock(self.hidden, heads=6, layers=2, dropout=d),
            nn.Conv2d(self.hidden, self.out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.permute(0, 2, 1, 3, 4)
        z = self.temporal3d(z).mean(dim=2)
        return self.restore(self.refiner(z))


PAPER_REIMPLEMENTATION_REGISTRY = {
    "tianhai_paper": TianHaiPaperReimplementation,
    "tianhai-reimpl": TianHaiPaperReimplementation,
    "fuxi_ocean_paper": FuXiOceanPaperReimplementation,
    "fuxi-ocean-reimpl": FuXiOceanPaperReimplementation,
    "fuxi_ons_paper": FuXiONSPaperReimplementation,
    "fuxi-ons-reimpl": FuXiONSPaperReimplementation,
    "axiomocean_paper": AxiomOceanPaperReimplementation,
    "axiom-ocean-reimpl": AxiomOceanPaperReimplementation,
}


def is_paper_reimplementation(model_type: str) -> bool:
    return model_type.lower() in PAPER_REIMPLEMENTATION_REGISTRY


def create_paper_reimplementation_model(config: dict) -> nn.Module:
    model_type = str(config.get("model_type", "")).lower()
    if model_type not in PAPER_REIMPLEMENTATION_REGISTRY:
        available = ", ".join(sorted(PAPER_REIMPLEMENTATION_REGISTRY))
        raise ValueError(f"Unknown paper reimplementation model '{model_type}'. Available: {available}")
    return PAPER_REIMPLEMENTATION_REGISTRY[model_type](config)
