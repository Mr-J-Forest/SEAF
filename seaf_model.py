"""SEAF: lightweight forecasting of joint ocean anomaly evolution."""

from typing import Dict, List, Optional, Sequence

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


class TemporalDepthMixer(nn.Module):
    """Mix depth and time without treating either axis as generic channels.

    The grouped convolutions operate independently for every variable/month
    along depth and for every variable/depth along time.  Only the final 1x1
    projection combines the structured profile features.
    """

    def __init__(
        self,
        sequence_length: int,
        variable_count: int,
        depth_levels: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.variable_count = int(variable_count)
        self.depth_levels = int(depth_levels)

        depth_channels = self.variable_count * self.sequence_length
        temporal_channels = self.variable_count * self.depth_levels
        self.depth_mixer = nn.Sequential(
            nn.Conv1d(
                depth_channels,
                depth_channels,
                kernel_size=3,
                padding=1,
                groups=depth_channels,
                bias=False,
            ),
            nn.GroupNorm(_adaptive_group_count(depth_channels), depth_channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.temporal_mixer = nn.Sequential(
            nn.Conv1d(
                temporal_channels,
                temporal_channels,
                kernel_size=3,
                padding=1,
                groups=temporal_channels,
                bias=False,
            ),
            nn.GroupNorm(
                _adaptive_group_count(temporal_channels), temporal_channels
            ),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.pointwise_projection = ConvGNAct2d(
            self.sequence_length * self.variable_count * self.depth_levels,
            hidden_dim,
            kernel_size=1,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels, height, width = x.shape
        expected_channels = self.variable_count * self.depth_levels
        if sequence != self.sequence_length or channels != expected_channels:
            raise ValueError(
                "TemporalDepthMixer expected "
                f"[B,{self.sequence_length},{expected_channels},H,W], got "
                f"{tuple(x.shape)}"
            )

        profiles = x.reshape(
            batch,
            sequence,
            self.variable_count,
            self.depth_levels,
            height,
            width,
        )
        depth_view = profiles.permute(0, 4, 5, 2, 1, 3).reshape(
            batch * height * width,
            self.variable_count * self.sequence_length,
            self.depth_levels,
        )
        depth_view = depth_view + self.depth_mixer(depth_view)
        profiles = depth_view.reshape(
            batch,
            height,
            width,
            self.variable_count,
            self.sequence_length,
            self.depth_levels,
        ).permute(0, 4, 3, 5, 1, 2)

        temporal_view = profiles.permute(0, 4, 5, 2, 3, 1).reshape(
            batch * height * width,
            self.variable_count * self.depth_levels,
            self.sequence_length,
        )
        temporal_view = temporal_view + self.temporal_mixer(temporal_view)
        profiles = temporal_view.reshape(
            batch,
            height,
            width,
            self.variable_count,
            self.depth_levels,
            self.sequence_length,
        ).permute(0, 5, 3, 4, 1, 2)
        return self.pointwise_projection(
            profiles.reshape(
                batch,
                self.sequence_length * self.variable_count * self.depth_levels,
                height,
                width,
            )
        )


class LocalResidualBlock(nn.Module):
    """Local 3x3 residual update used beside the low-mode global path."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_adaptive_group_count(channels), channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_adaptive_group_count(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalGlobalSpatialBlock(nn.Module):
    """Fuse local residual updates with conservatively scaled spectral updates."""

    def __init__(
        self,
        channels: int,
        modes_y: int,
        modes_x: int,
        use_local_path: bool,
        use_spectral_path: bool,
        spectral_scale_init: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.local_path = (
            LocalResidualBlock(channels, dropout=dropout)
            if use_local_path
            else None
        )
        self.global_path = (
            SpectralLowModeMixer(
                channels,
                modes_y=modes_y,
                modes_x=modes_x,
                dropout=dropout,
            )
            if use_spectral_path
            else None
        )
        self.spectral_scale = (
            nn.Parameter(torch.tensor(float(spectral_scale_init)))
            if self.local_path is not None and self.global_path is not None
            else None
        )
        self.post_mix = ConvGNAct2d(
            channels, channels, kernel_size=1, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_path is None and self.global_path is not None:
            fused = self.global_path(x)
        else:
            fused = x
            if self.local_path is not None:
                fused = fused + self.local_path(x)
            if self.global_path is not None:
                global_delta = self.global_path(x) - x
                fused = fused + self.spectral_scale * global_delta
        return self.post_mix(fused)


def _parse_channel_slices(
    raw_slices: object, input_dim: int
) -> Dict[str, tuple[int, int]]:
    if not isinstance(raw_slices, dict):
        return {}
    parsed: Dict[str, tuple[int, int]] = {}
    for name, bounds in raw_slices.items():
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
        ):
            raise ValueError(f"Invalid input channel slice for {name!r}: {bounds!r}")
        start, stop = int(bounds[0]), int(bounds[1])
        if start < 0 or stop <= start or stop > input_dim:
            raise ValueError(f"Invalid input channel slice for {name!r}: {bounds!r}")
        parsed[str(name)] = (start, stop)
    return parsed


def _slice_indices(
    channel_slices: Dict[str, tuple[int, int]], names: Sequence[str]
) -> List[int]:
    indices: List[int] = []
    for name in names:
        start, stop = channel_slices[name]
        indices.extend(range(start, stop))
    return indices


class StructuredSEAFEncoder(nn.Module):
    """Schema-aware ocean/context encoder for the modular SEAF candidate."""

    def __init__(
        self,
        config: dict,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        modes_y: int,
        modes_x: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.sequence_length = int(config.get("sequence_length", 12))
        self.use_temporal_depth_mixer = bool(
            config.get("use_temporal_depth_mixer", False)
        )
        self.use_forcing_encoder = bool(
            config.get("use_forcing_encoder", False)
        )

        channel_slices = _parse_channel_slices(
            config.get("input_channel_slices", {}), input_dim
        )
        profile_variables = list(
            config.get("seaf_profile_variables", config.get("target_variables", []))
        )
        missing_profiles = [
            name for name in profile_variables if name not in channel_slices
        ]
        if not profile_variables or missing_profiles:
            raise ValueError(
                "Structured SEAF requires input_channel_slices for all profile "
                f"variables; missing={missing_profiles or profile_variables}"
            )
        profile_widths = [
            channel_slices[name][1] - channel_slices[name][0]
            for name in profile_variables
        ]
        if len(set(profile_widths)) != 1:
            raise ValueError(
                "SEAF profile variables must share one depth grid; "
                f"widths={dict(zip(profile_variables, profile_widths))}"
            )

        forcing_variables = list(config.get("external_dynamic_variables", []))
        missing_forcing = [
            name for name in forcing_variables if name not in channel_slices
        ]
        if self.use_forcing_encoder and (not forcing_variables or missing_forcing):
            raise ValueError(
                "use_forcing_encoder requires channel slices for every external "
                f"variable; missing={missing_forcing or forcing_variables}"
            )

        profile_indices = _slice_indices(channel_slices, profile_variables)
        forcing_indices = (
            _slice_indices(channel_slices, forcing_variables)
            if self.use_forcing_encoder
            else []
        )
        overlap = set(profile_indices) & set(forcing_indices)
        if overlap:
            raise ValueError(
                "Profile and forcing channel groups overlap: "
                f"indices={sorted(overlap)}"
            )
        reserved = set(profile_indices) | set(forcing_indices)
        context_indices = [index for index in range(input_dim) if index not in reserved]

        self.register_buffer(
            "profile_indices",
            torch.tensor(profile_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "context_indices",
            torch.tensor(context_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "forcing_indices",
            torch.tensor(forcing_indices, dtype=torch.long),
            persistent=False,
        )

        depth_levels = profile_widths[0]
        if self.use_temporal_depth_mixer:
            self.profile_encoder = TemporalDepthMixer(
                sequence_length=self.sequence_length,
                variable_count=len(profile_variables),
                depth_levels=depth_levels,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.profile_encoder = ConvGNAct2d(
                self.sequence_length * len(profile_indices),
                hidden_dim,
                dropout=dropout,
            )

        self.context_encoder = (
            ConvGNAct2d(
                self.sequence_length * len(context_indices),
                hidden_dim,
                dropout=dropout,
            )
            if context_indices
            else None
        )
        self.forcing_encoder = (
            nn.Sequential(
                ConvGNAct2d(
                    self.sequence_length * len(forcing_indices),
                    hidden_dim,
                    dropout=dropout,
                ),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            )
            if self.use_forcing_encoder
            else None
        )
        self.forcing_scale = (
            nn.Parameter(
                torch.tensor(float(config.get("seaf_forcing_scale_init", 0.1)))
            )
            if self.forcing_encoder is not None
            else None
        )

        self.spatial_blocks = nn.ModuleList(
            [
                LocalGlobalSpatialBlock(
                    hidden_dim,
                    modes_y=modes_y,
                    modes_x=modes_x,
                    use_local_path=bool(config.get("use_local_path", False)),
                    use_spectral_path=bool(config.get("use_spectral_path", True))
                    and not bool(config.get("ablation_disable_spectral", False)),
                    spectral_scale_init=float(
                        config.get("seaf_spectral_scale_init", 0.1)
                    ),
                    dropout=dropout,
                )
                for _ in range(max(1, layers))
            ]
        )

    @staticmethod
    def _select_and_flatten(
        x: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        selected = x.index_select(2, indices)
        batch, sequence, channels, height, width = selected.shape
        return selected.reshape(batch, sequence * channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        profile_history = x.index_select(2, self.profile_indices)
        if self.use_temporal_depth_mixer:
            features = self.profile_encoder(profile_history)
        else:
            features = self.profile_encoder(
                self._select_and_flatten(x, self.profile_indices)
            )

        if self.context_encoder is not None:
            features = features + self.context_encoder(
                self._select_and_flatten(x, self.context_indices)
            )
        if self.forcing_encoder is not None:
            forcing_features = self.forcing_encoder(
                self._select_and_flatten(x, self.forcing_indices)
            )
            features = features + self.forcing_scale * forcing_features

        for block in self.spatial_blocks:
            features = block(features)
        return features

    @staticmethod
    def _parameter_count(module: Optional[nn.Module]) -> int:
        return 0 if module is None else sum(p.numel() for p in module.parameters())

    def parameter_breakdown(self) -> Dict[str, int]:
        breakdown = {
            "profile_encoder": self._parameter_count(self.profile_encoder),
            "context_encoder": self._parameter_count(self.context_encoder),
            "forcing_encoder": self._parameter_count(self.forcing_encoder),
            "spatial_paths": self._parameter_count(self.spatial_blocks),
        }
        if self.forcing_scale is not None:
            breakdown["forcing_fusion_scale"] = self.forcing_scale.numel()
        return breakdown


class SEAFNet(nn.Module):
    """SEAF model; AP/DAP remain external evaluation references only."""

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
        self.uniform_ensemble = bool(config.get("ablation_uniform_ensemble", False))
        self.use_temporal_depth_mixer = bool(
            config.get("use_temporal_depth_mixer", False)
        )
        self.use_local_path = bool(config.get("use_local_path", False))
        self.use_spectral_path = bool(config.get("use_spectral_path", True))
        self.use_forcing_encoder = bool(
            config.get("use_forcing_encoder", False)
        )
        self.structured_encoder_enabled = any(
            (
                self.use_temporal_depth_mixer,
                self.use_local_path,
                self.use_forcing_encoder,
                not self.use_spectral_path,
            )
        )
        if self.structured_encoder_enabled:
            self.spectral_encoder = StructuredSEAFEncoder(
                config,
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                modes_y=int(spectral_modes[0]),
                modes_x=int(spectral_modes[1]),
                layers=spectral_layers,
            )
        elif self.disable_spectral:
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
        configured_router = str(config.get("router_type", "spatial")).lower()
        if self.disable_ensemble:
            self.router_type = "single"
        elif self.uniform_ensemble or configured_router == "uniform":
            self.router_type = "uniform"
        else:
            self.router_type = configured_router
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
        self.ensemble_gate = (
            nn.Sequential(
                nn.Conv2d(hidden_dim, gate_hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(gate_hidden, member_count, kernel_size=1),
                nn.Softmax(dim=1),
            )
            if self.router_type == "spatial"
            else None
        )
        self.lead_router_logits = (
            nn.Parameter(torch.zeros(self.prediction_length, member_count))
            if self.router_type == "lead"
            else None
        )

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del targets
        features = self.encode_features(x)
        return self.forecast_from_features(features)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a history once so compatible heads can reuse the representation."""
        batch, sequence, channels, height, width = x.shape
        if sequence != self.sequence_length or channels != self.input_dim:
            raise ValueError(
                "SEAF input shape does not match the configured history: "
                f"got (T={sequence}, C={channels}), expected "
                f"(T={self.sequence_length}, C={self.input_dim})"
            )
        if self.structured_encoder_enabled:
            features = self.spectral_encoder(x)
        else:
            features = self.spectral_encoder(
                x.reshape(batch, sequence * channels, height, width)
            )
        return features

    def forecast_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Run the frozen SEAF direct prediction path from encoded features."""
        if features.ndim != 4 or features.shape[1] != self.member_heads[0][0].net[0].out_channels:
            raise ValueError(
                "SEAF feature shape does not match the configured hidden width: "
                f"got {tuple(features.shape)}"
            )
        batch, _, height, width = features.shape
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
        if self.router_type == "single":
            forecast = members[:, 0]
        elif self.router_type == "uniform":
            forecast = members.mean(dim=1)
        elif self.router_type == "spatial":
            weights = self.ensemble_gate(features).unsqueeze(2).unsqueeze(3)
            forecast = (members * weights).sum(dim=1)
        elif self.router_type == "lead":
            weights = torch.softmax(self.lead_router_logits, dim=-1)
            weights = weights.transpose(0, 1).reshape(
                1,
                len(self.member_heads),
                self.prediction_length,
                1,
                1,
                1,
            )
            forecast = (members * weights).sum(dim=1)
        else:  # guarded by config validation; keep a local fail-fast boundary.
            raise RuntimeError(f"Unsupported SEAF router_type: {self.router_type!r}")
        return forecast

    @staticmethod
    def _parameter_count(module: Optional[nn.Module]) -> int:
        return 0 if module is None else sum(p.numel() for p in module.parameters())

    def parameter_breakdown(self) -> Dict[str, int]:
        if isinstance(self.spectral_encoder, StructuredSEAFEncoder):
            breakdown = self.spectral_encoder.parameter_breakdown()
        else:
            breakdown = {
                "encoder": self._parameter_count(self.spectral_encoder),
            }
        breakdown.update(
            {
                "member_heads": self._parameter_count(self.member_heads),
                "spatial_gate": self._parameter_count(self.ensemble_gate),
                "lead_router": (
                    0
                    if self.lead_router_logits is None
                    else self.lead_router_logits.numel()
                ),
            }
        )
        breakdown["total"] = sum(p.numel() for p in self.parameters())
        return breakdown

    def model_diagnostics(self) -> Dict[str, object]:
        diagnostics: Dict[str, object] = {
            "model_display_name": "SEAF",
            "router_type": self.router_type,
            "use_temporal_depth_mixer": self.use_temporal_depth_mixer,
            "use_local_path": self.use_local_path,
            "use_spectral_path": self.use_spectral_path and not self.disable_spectral,
            "use_forcing_encoder": self.use_forcing_encoder,
            "parameter_breakdown": self.parameter_breakdown(),
        }
        if self.lead_router_logits is not None:
            diagnostics["lead_member_weights"] = (
                torch.softmax(self.lead_router_logits.detach().float(), dim=-1)
                .cpu()
                .tolist()
            )
        if isinstance(self.spectral_encoder, StructuredSEAFEncoder):
            spectral_scales = [
                float(block.spectral_scale.detach().cpu())
                for block in self.spectral_encoder.spatial_blocks
                if block.spectral_scale is not None
            ]
            if spectral_scales:
                diagnostics["spectral_fusion_scales"] = spectral_scales
            if self.spectral_encoder.forcing_scale is not None:
                diagnostics["forcing_fusion_scale"] = float(
                    self.spectral_encoder.forcing_scale.detach().cpu()
                )
        return diagnostics
