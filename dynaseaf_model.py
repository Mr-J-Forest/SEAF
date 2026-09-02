"""DynaSEAF: dynamics-guided transport--innovation anomaly forecasting.

The module deliberately keeps the existing :class:`SEAFNet` direct forecast
path intact.  Future dynamics are predicted from the history representation
and are used only as model-generated conditioning; ground-truth future
dynamics are never accepted by ``forward``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from seaf_model import SEAFNet, _adaptive_group_count


def _parse_channel_slices(raw_slices: object) -> Dict[str, slice]:
    """Parse a schema map without inventing channel offsets."""
    if raw_slices is None:
        return {}
    if not isinstance(raw_slices, Mapping):
        raise ValueError("channel slices must be a mapping of variable to [start, stop]")
    parsed: Dict[str, slice] = {}
    for name, bounds in raw_slices.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Invalid channel slice for {name!r}: {bounds!r}")
        start, stop = int(bounds[0]), int(bounds[1])
        if start < 0 or stop <= start:
            raise ValueError(f"Invalid channel slice for {name!r}: {bounds!r}")
        parsed[str(name)] = slice(start, stop)
    return parsed


def _slice_width(channel_slice: slice) -> int:
    if channel_slice.start is None or channel_slice.stop is None:
        raise ValueError(f"Open-ended channel slice is not supported: {channel_slice}")
    return int(channel_slice.stop) - int(channel_slice.start)


class LeadEmbedding(nn.Module):
    """Embed one-based forecast leads and project them with a small MLP."""

    def __init__(
        self,
        max_lead: int,
        embedding_dim: int = 16,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.max_lead = int(max_lead)
        if self.max_lead <= 0:
            raise ValueError("max_lead must be positive")
        embedding_dim = int(embedding_dim)
        hidden_dim = embedding_dim if hidden_dim is None else int(hidden_dim)
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("LeadEmbedding dimensions must be positive")
        self.embedding = nn.Embedding(self.max_lead, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(
        self,
        leads: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Return ``[B, K, E]`` embeddings for one-based leads ``1..K``."""
        if not isinstance(leads, torch.Tensor):
            leads = torch.as_tensor(leads, dtype=torch.long)
        leads = leads.to(device=self.embedding.weight.device, dtype=torch.long)
        if leads.ndim == 0:
            leads = leads.reshape(1)
        if leads.ndim == 1:
            leads = leads.unsqueeze(0)
        if leads.ndim != 2:
            raise ValueError(f"leads must have shape [K] or [B,K], got {tuple(leads.shape)}")
        if leads.numel() and (
            bool((leads < 1).any().item())
            or bool((leads > self.max_lead).any().item())
        ):
            raise ValueError(
                f"lead indices must be one-based and within 1..{self.max_lead}"
            )
        projected = self.mlp(self.embedding(leads - 1))
        if batch_size is not None:
            batch_size = int(batch_size)
            if projected.shape[0] == 1:
                projected = projected.expand(batch_size, -1, -1)
            elif projected.shape[0] != batch_size:
                raise ValueError(
                    f"lead batch dimension {projected.shape[0]} != requested {batch_size}"
                )
        return projected


class DynamicsConditioner(nn.Module):
    """Fuse representation, predicted dynamics, and lead embeddings per grid cell."""

    def __init__(
        self,
        feature_channels: int,
        dynamics_channels: int,
        lead_channels: int,
        hidden_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.dynamics_channels = int(dynamics_channels)
        self.lead_channels = int(lead_channels)
        input_channels = (
            self.feature_channels + self.dynamics_channels + self.lead_channels
        )
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_adaptive_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self,
        features: torch.Tensor,
        dynamics: Optional[torch.Tensor],
        lead_features: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"features must be [B,F,H,W], got {tuple(features.shape)}")
        batch, _, height, width = features.shape
        if lead_features.ndim == 2:
            lead_features = lead_features.unsqueeze(0).expand(batch, -1, -1)
        if lead_features.ndim != 3 or lead_features.shape[0] != batch:
            raise ValueError(
                "lead_features must be [B,K,E] with the same batch as features"
            )
        lead_count = int(lead_features.shape[1])
        if lead_features.shape[2] != self.lead_channels:
            raise ValueError(
                f"lead embedding width {lead_features.shape[2]} != {self.lead_channels}"
            )

        feature_map = features.unsqueeze(1).expand(
            -1, lead_count, -1, -1, -1
        )
        lead_map = lead_features.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, height, width
        )
        pieces = [feature_map, lead_map]
        if self.dynamics_channels:
            if dynamics is None or dynamics.ndim != 5:
                raise ValueError(
                    "dynamics must be [B,K,D,H,W] for a dynamics-conditioned head"
                )
            if tuple(dynamics.shape[:2]) != (batch, lead_count):
                raise ValueError("dynamics batch/lead dimensions do not match conditioning")
            if tuple(dynamics.shape[-2:]) != (height, width):
                raise ValueError("dynamics spatial dimensions do not match features")
            if dynamics.shape[2] != self.dynamics_channels:
                raise ValueError(
                    f"dynamics channels {dynamics.shape[2]} != {self.dynamics_channels}"
                )
            pieces.insert(1, dynamics)
        fused = torch.cat(pieces, dim=2).reshape(
            batch * lead_count, -1, height, width
        )
        return self.net(fused).reshape(
            batch, lead_count, -1, height, width
        )


class _ConditionedFieldHead(nn.Module):
    """Shared lightweight per-lead field head."""

    def __init__(
        self,
        feature_channels: int,
        dynamics_channels: int,
        lead_channels: int,
        hidden_channels: int,
        output_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conditioner = DynamicsConditioner(
            feature_channels,
            dynamics_channels,
            lead_channels,
            hidden_channels,
            dropout=dropout,
        )
        self.output = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        dynamics: Optional[torch.Tensor],
        lead_features: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = self.conditioner(features, dynamics, lead_features)
        batch, lead_count, hidden, height, width = conditioned.shape
        values = self.output(
            conditioned.reshape(batch * lead_count, hidden, height, width)
        )
        return values.reshape(batch, lead_count, -1, height, width)


class FutureDynamicsHead(nn.Module):
    """Predict future UVEL/VVEL/SSHA/MLD from history representation only."""

    def __init__(
        self,
        feature_channels: int,
        lead_channels: int,
        output_channels: int,
        hidden_channels: int,
        dropout: float = 0.0,
        variable_slices: Optional[Mapping[str, slice]] = None,
    ) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.variable_slices = dict(variable_slices or {})
        self.head = _ConditionedFieldHead(
            feature_channels,
            dynamics_channels=0,
            lead_channels=lead_channels,
            hidden_channels=hidden_channels,
            output_channels=self.output_channels,
            dropout=dropout,
        )

    def forward(
        self,
        features: torch.Tensor,
        lead_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(features, None, lead_features)


class DeformationHead(nn.Module):
    """Predict a bounded effective dx/dy field from learned dynamics."""

    def __init__(
        self,
        feature_channels: int,
        dynamics_channels: int,
        lead_channels: int,
        deformation_channels: int,
        hidden_channels: int,
        max_deformation_cells: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.deformation_channels = int(deformation_channels)
        self.max_deformation_cells = float(max_deformation_cells)
        self.head = _ConditionedFieldHead(
            feature_channels,
            dynamics_channels=dynamics_channels,
            lead_channels=lead_channels,
            hidden_channels=hidden_channels,
            output_channels=2 * self.deformation_channels,
            dropout=dropout,
        )

    def forward(
        self,
        features: torch.Tensor,
        dynamics: torch.Tensor,
        lead_features: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.head(features, dynamics, lead_features)
        batch, lead_count, _, height, width = raw.shape
        raw = raw.reshape(
            batch, lead_count, 2, self.deformation_channels, height, width
        ).permute(0, 1, 3, 4, 5, 2)
        return torch.tanh(raw) * self.max_deformation_cells


class InnovationHead(nn.Module):
    """Predict non-transport anomaly evolution, optionally zero-initialized."""

    def __init__(
        self,
        feature_channels: int,
        dynamics_channels: int,
        lead_channels: int,
        output_channels: int,
        hidden_channels: int,
        dropout: float = 0.0,
        zero_initialize: bool = True,
    ) -> None:
        super().__init__()
        self.head = _ConditionedFieldHead(
            feature_channels,
            dynamics_channels=dynamics_channels,
            lead_channels=lead_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            dropout=dropout,
        )
        if zero_initialize:
            nn.init.zeros_(self.head.output.weight)
            nn.init.zeros_(self.head.output.bias)

    def forward(
        self,
        features: torch.Tensor,
        dynamics: torch.Tensor,
        lead_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(features, dynamics, lead_features)


class TransportDirectGate(nn.Module):
    """Predict a spatially resolved direct/transport mixing gate."""

    def __init__(
        self,
        feature_channels: int,
        dynamics_channels: int,
        lead_channels: int,
        output_channels: int,
        hidden_channels: int,
        initial_bias: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = _ConditionedFieldHead(
            feature_channels,
            dynamics_channels=dynamics_channels,
            lead_channels=lead_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            dropout=dropout,
        )
        # Start from the configured scalar prior rather than a spatially
        # noisy random gate; the feature-dependent decision is learned after
        # the direct/transport decomposition has a stable initial mixture.
        nn.init.zeros_(self.head.output.weight)
        nn.init.constant_(self.head.output.bias, float(initial_bias))

    def forward(
        self,
        features: torch.Tensor,
        dynamics: torch.Tensor,
        lead_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.head(features, dynamics, lead_features))


class DifferentiableAnomalyWarp(nn.Module):
    """Warp anomaly maps with cell-unit displacements and mask-aware sampling.

    A positive ``dx`` moves a structure toward increasing column index and a
    positive ``dy`` toward increasing row index.  The sampler therefore reads
    the source at ``destination - displacement``.  When a validity mask is
    present, value and mask are sampled separately and divided, preventing
    land/NaN values from diluting valid ocean cells.
    """

    def __init__(self, align_corners: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.align_corners = bool(align_corners)
        self.eps = float(eps)

    @staticmethod
    def _normalize_displacement(
        displacement: torch.Tensor,
        batch: int,
        channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if displacement.ndim == 4:
            if displacement.shape[1] == 2:
                displacement = displacement.permute(0, 2, 3, 1).unsqueeze(1)
            elif displacement.shape[-1] == 2:
                displacement = displacement.unsqueeze(1)
            else:
                raise ValueError(
                    "4-D displacement must be [B,2,H,W] or [B,H,W,2]"
                )
        elif displacement.ndim == 5:
            if displacement.shape[-1] == 2:
                pass
            elif displacement.shape[2] == 2:
                displacement = displacement.permute(0, 1, 3, 4, 2)
            else:
                raise ValueError(
                    "5-D displacement must be [B,C,H,W,2] or [B,C,2,H,W]"
                )
        else:
            raise ValueError(
                f"unsupported displacement shape: {tuple(displacement.shape)}"
            )

        if displacement.shape[0] != batch or tuple(displacement.shape[2:4]) != (
            height,
            width,
        ):
            raise ValueError(
                "displacement batch/spatial dimensions do not match source: "
                f"{tuple(displacement.shape)} vs {(batch, channels, height, width)}"
            )
        if displacement.shape[1] == 1:
            displacement = displacement.expand(-1, channels, -1, -1, -1)
        elif displacement.shape[1] != channels:
            raise ValueError(
                f"displacement channels {displacement.shape[1]} != source channels {channels}"
            )
        return displacement

    @staticmethod
    def _normalize_valid_mask(
        valid_mask: torch.Tensor,
        batch: int,
        channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)
        if valid_mask.ndim != 4 or tuple(valid_mask.shape[-2:]) != (height, width):
            raise ValueError(
                f"valid_mask must be [B,H,W] or [B,C,H,W], got {tuple(valid_mask.shape)}"
            )
        if valid_mask.shape[0] != batch:
            raise ValueError("valid_mask batch dimension does not match source")
        if valid_mask.shape[1] == 1:
            valid_mask = valid_mask.expand(-1, channels, -1, -1)
        elif valid_mask.shape[1] != channels:
            raise ValueError(
                f"valid_mask channels {valid_mask.shape[1]} != source channels {channels}"
            )
        return valid_mask

    def forward(
        self,
        source: torch.Tensor,
        displacement: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if source.ndim != 4:
            raise ValueError(f"source must be [B,C,H,W], got {tuple(source.shape)}")
        batch, channels, height, width = source.shape
        displacement = self._normalize_displacement(
            displacement, batch, channels, height, width
        )

        # grid_sample has incomplete low-precision CPU support.  Computing the
        # geometric operation in fp32 is also safer under CUDA AMP and keeps
        # the output finite before it is cast back to the model dtype.
        compute_dtype = (
            torch.float32
            if source.dtype in (torch.float16, torch.bfloat16)
            else source.dtype
        )
        source_work = torch.nan_to_num(source.to(compute_dtype), nan=0.0, posinf=0.0, neginf=0.0)
        displacement_work = torch.nan_to_num(
            displacement.to(compute_dtype), nan=0.0, posinf=0.0, neginf=0.0
        )

        row = torch.arange(height, device=source.device, dtype=compute_dtype)
        col = torch.arange(width, device=source.device, dtype=compute_dtype)
        yy, xx = torch.meshgrid(row, col, indexing="ij")
        if self.align_corners:
            x_base = 2.0 * xx / max(width - 1, 1) - 1.0
            y_base = 2.0 * yy / max(height - 1, 1) - 1.0
            x_scale = 2.0 / max(width - 1, 1)
            y_scale = 2.0 / max(height - 1, 1)
        else:
            x_base = (2.0 * (xx + 0.5) / max(width, 1)) - 1.0
            y_base = (2.0 * (yy + 0.5) / max(height, 1)) - 1.0
            x_scale = 2.0 / max(width, 1)
            y_scale = 2.0 / max(height, 1)

        grid_x = x_base.view(1, 1, height, width) - (
            displacement_work[..., 0] * x_scale
        )
        grid_y = y_base.view(1, 1, height, width) - (
            displacement_work[..., 1] * y_scale
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
            batch * channels, height, width, 2
        )

        finite_source = torch.isfinite(source)
        if valid_mask is None and bool(finite_source.all().item()):
            warped = F.grid_sample(
                source_work.reshape(batch * channels, 1, height, width),
                grid.reshape(batch * channels, height, width, 2),
                mode="bilinear",
                padding_mode="border",
                align_corners=self.align_corners,
            ).reshape(batch, channels, height, width)
        else:
            if valid_mask is None:
                mask = finite_source
            else:
                mask = self._normalize_valid_mask(
                    valid_mask.to(device=source.device),
                    batch,
                    channels,
                    height,
                    width,
                ).bool() & finite_source
            mask_work = mask.to(compute_dtype)
            numerator = F.grid_sample(
                (source_work * mask_work).reshape(batch * channels, 1, height, width),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=self.align_corners,
            )
            denominator = F.grid_sample(
                mask_work.reshape(batch * channels, 1, height, width),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=self.align_corners,
            )
            warped = torch.where(
                denominator > self.eps,
                numerator / denominator.clamp_min(self.eps),
                torch.zeros_like(numerator),
            ).reshape(batch, channels, height, width)
        return warped.to(dtype=source.dtype)


class DynaSEAFNet(nn.Module):
    """SEAF direct forecast plus transport--innovation decomposition."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = dict(config)
        self.sequence_length = int(config.get("sequence_length", 12))
        self.prediction_length = int(config.get("prediction_length", 5))
        self.target_variables = list(config.get("target_variables", []))
        self.use_future_dynamics_aux = bool(
            config.get("dynaseaf_use_future_dynamics_aux", True)
        )
        self.use_transport = bool(config.get("dynaseaf_use_transport", True))
        self.use_innovation = bool(config.get("dynaseaf_use_innovation", True))
        self.use_adaptive_gate = bool(
            config.get("dynaseaf_use_adaptive_gate", True)
        )

        direct_config = dict(config)
        direct_config["model_type"] = "seaf"
        if config.get("dynaseaf_use_temporal_depth_mixer", False):
            direct_config["use_temporal_depth_mixer"] = True
        self.direct_model = SEAFNet(direct_config)
        self.input_dim = self.direct_model.input_dim
        self.output_dim = self.direct_model.output_dim
        hidden_dim = int(config.get("seaf_hidden_dim", 64))
        dropout = float(config.get("dropout", 0.05))
        condition_hidden = max(8, hidden_dim // 2)
        lead_width = max(8, min(32, hidden_dim // 4))

        self.input_channel_slices = _parse_channel_slices(
            config.get("input_channel_slices")
        )
        self.target_channel_slices = _parse_channel_slices(
            config.get("target_channel_slices")
        )
        if not self.target_channel_slices:
            raise ValueError(
                "DynaSEAF requires explicit target_channel_slices from the data schema"
            )
        self._validate_target_schema()

        configured_dynamics = config.get(
            "dynaseaf_future_dynamics_variables",
            config.get(
                "future_dynamics_target_variables",
                ["UVEL", "VVEL", "SSHA", "MLD"],
            ),
        )
        if not isinstance(configured_dynamics, (list, tuple)) or not configured_dynamics:
            raise ValueError("dynaseaf_future_dynamics_variables must be a non-empty list")
        self.future_dynamics_variables = [str(name) for name in configured_dynamics]
        raw_dynamics_slices = config.get(
            "dynaseaf_future_dynamics_channel_slices",
            config.get("future_dynamics_target_channel_slices"),
        )
        parsed_dynamics_slices = _parse_channel_slices(raw_dynamics_slices)
        self.future_dynamics_channel_slices: Dict[str, slice] = {}
        dynamic_offset = 0
        for name in self.future_dynamics_variables:
            source_slice = parsed_dynamics_slices.get(name)
            if source_slice is None:
                source_slice = self.input_channel_slices.get(name)
            if source_slice is None:
                raise ValueError(
                    "DynaSEAF requires a schema slice for future dynamics variable "
                    f"{name!r}; future labels are not inferred from offsets"
                )
            width = _slice_width(source_slice)
            self.future_dynamics_channel_slices[name] = slice(
                dynamic_offset, dynamic_offset + width
            )
            dynamic_offset += width
        self.future_dynamics_channels = dynamic_offset

        raw_source_indices = config.get("dynaseaf_transport_source_channel_indices")
        source_indices: List[int] = []
        if raw_source_indices is not None:
            if not isinstance(raw_source_indices, (list, tuple)):
                raise ValueError("dynaseaf_transport_source_channel_indices must be a list")
            source_indices = [int(value) for value in raw_source_indices]
            if len(source_indices) != self.output_dim:
                raise ValueError(
                    "dynaseaf_transport_source_channel_indices must cover every target "
                    f"channel ({len(source_indices)} != {self.output_dim})"
                )
        else:
            source_indices = [-1] * self.output_dim
            for variable in self.target_variables:
                target_slice = self.target_channel_slices.get(variable)
                source_slice = self.input_channel_slices.get(variable)
                if target_slice is None or source_slice is None:
                    raise ValueError(
                        "DynaSEAF transport requires target variables to be present in "
                        f"the input schema: {variable!r}"
                    )
                if _slice_width(target_slice) != _slice_width(source_slice):
                    raise ValueError(
                        "DynaSEAF requires matching input/target depth channels for "
                        f"{variable!r}; provide an explicit transport source map for "
                        "a deliberate remapping"
                    )
                for offset in range(_slice_width(target_slice)):
                    target_index = int(target_slice.start) + offset
                    source_indices[target_index] = int(source_slice.start) + offset
            if any(value < 0 for value in source_indices):
                raise ValueError(
                    "target_channel_slices do not cover a complete output schema; "
                    "DynaSEAF will not guess transport channels"
                )
        if any(value < 0 or value >= self.input_dim for value in source_indices):
            raise ValueError("DynaSEAF transport source map contains an invalid input channel")
        self.register_buffer(
            "transport_source_indices",
            torch.tensor(source_indices, dtype=torch.long),
            persistent=False,
        )

        target_widths = [
            _slice_width(self.target_channel_slices[name])
            for name in self.target_variables
        ]
        raw_depth_map = config.get("dynaseaf_target_to_deformation_indices")
        if raw_depth_map is not None:
            if not isinstance(raw_depth_map, (list, tuple)) or len(raw_depth_map) != self.output_dim:
                raise ValueError(
                    "dynaseaf_target_to_deformation_indices must cover every target channel"
                )
            target_to_deformation = [int(value) for value in raw_depth_map]
            deformation_channels = max(target_to_deformation) + 1
        elif target_widths and len(set(target_widths)) == 1:
            deformation_channels = target_widths[0]
            target_to_deformation = []
            for variable in self.target_variables:
                target_slice = self.target_channel_slices[variable]
                target_to_deformation.extend(
                    range(deformation_channels)
                )
        else:
            # Unequal depth structures remain schema-safe: each target channel
            # receives a field from the shared, parameter-tied deformation head.
            deformation_channels = self.output_dim
            target_to_deformation = list(range(self.output_dim))
        if len(target_to_deformation) != self.output_dim:
            raise ValueError(
                "DynaSEAF deformation map does not cover every output channel"
            )
        if any(value < 0 or value >= deformation_channels for value in target_to_deformation):
            raise ValueError("DynaSEAF deformation map contains an invalid channel")
        self.deformation_channels = int(deformation_channels)
        self.register_buffer(
            "target_to_deformation_indices",
            torch.tensor(target_to_deformation, dtype=torch.long),
            persistent=False,
        )

        gate_resolution = str(config.get("dynaseaf_gate_resolution", "target")).lower()
        if gate_resolution in {"target", "channel", "target_channel"}:
            self.gate_resolution = "target"
            self.gate_channels = self.output_dim
            gate_map = list(range(self.output_dim))
        elif gate_resolution in {"depth", "shared_depth"}:
            self.gate_resolution = "depth"
            self.gate_channels = self.deformation_channels
            gate_map = target_to_deformation
        else:
            raise ValueError(
                "dynaseaf_gate_resolution must be 'target' or 'depth'"
            )
        self.register_buffer(
            "target_to_gate_indices",
            torch.tensor(gate_map, dtype=torch.long),
            persistent=False,
        )

        self.lead_embedding = LeadEmbedding(
            self.prediction_length,
            embedding_dim=lead_width,
            hidden_dim=lead_width,
        )
        needs_dynamics = bool(
            self.use_future_dynamics_aux
            or self.use_transport
            or self.use_innovation
            or (self.use_adaptive_gate and self.use_transport)
        )
        self.future_dynamics_head = (
            FutureDynamicsHead(
                self.direct_model.member_heads[0][0].net[0].out_channels,
                self.lead_embedding.output_dim,
                self.future_dynamics_channels,
                condition_hidden,
                dropout=dropout,
                variable_slices=self.future_dynamics_channel_slices,
            )
            if needs_dynamics
            else None
        )
        self.deformation_head = (
            DeformationHead(
                self.direct_model.member_heads[0][0].net[0].out_channels,
                self.future_dynamics_channels,
                self.lead_embedding.output_dim,
                self.deformation_channels,
                condition_hidden,
                max_deformation_cells=float(
                    config.get("dynaseaf_max_deformation_cells", 1.0)
                ),
                dropout=dropout,
            )
            if self.use_transport
            else None
        )
        self.warp = DifferentiableAnomalyWarp()
        self.innovation_head = (
            InnovationHead(
                self.direct_model.member_heads[0][0].net[0].out_channels,
                self.future_dynamics_channels,
                self.lead_embedding.output_dim,
                self.output_dim,
                condition_hidden,
                dropout=dropout,
                zero_initialize=bool(
                    config.get("dynaseaf_zero_init_innovation", True)
                ),
            )
            if self.use_innovation
            else None
        )
        self.transport_direct_gate = (
            TransportDirectGate(
                self.direct_model.member_heads[0][0].net[0].out_channels,
                self.future_dynamics_channels,
                self.lead_embedding.output_dim,
                self.gate_channels,
                condition_hidden,
                initial_bias=float(config.get("dynaseaf_gate_initial_bias", -1.7346)),
                dropout=dropout,
            )
            if self.use_transport and self.use_adaptive_gate
            else None
        )

    def _validate_target_schema(self) -> None:
        covered = set()
        for variable in self.target_variables:
            channel_slice = self.target_channel_slices.get(variable)
            if channel_slice is None:
                raise ValueError(f"target schema lacks variable {variable!r}")
            for index in range(int(channel_slice.start), int(channel_slice.stop)):
                if index in covered:
                    raise ValueError(f"target schema overlaps at channel {index}")
                covered.add(index)
        if covered != set(range(self.output_dim)):
            raise ValueError(
                "target_channel_slices must cover output channels exactly; "
                f"covered={sorted(covered)}, expected=0..{self.output_dim - 1}"
            )

    def _resolve_input_valid_mask(
        self,
        valid_mask: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if valid_mask is None:
            return None
        mask = valid_mask.to(device=x.device)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or tuple(mask.shape[0:1] + mask.shape[-2:]) != (
            x.shape[0],
            x.shape[-2],
            x.shape[-1],
        ):
            raise ValueError(
                "DynaSEAF valid_mask must be [B,H,W] or [B,C,H,W] over input spatial cells"
            )
        if mask.shape[1] == 1:
            return mask.expand(-1, self.input_dim, -1, -1).bool()
        if mask.shape[1] != self.input_dim:
            raise ValueError(
                f"DynaSEAF input valid_mask channels {mask.shape[1]} != {self.input_dim}"
            )
        return mask.bool()

    @staticmethod
    def mix_forecasts(
        direct_forecast: torch.Tensor,
        transport_forecast: torch.Tensor,
        innovation: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the final decomposition equation for testable components."""
        return (1.0 - gate) * direct_forecast + gate * transport_forecast + innovation

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"DynaSEAF input must be [B,T,C,H,W], got {tuple(x.shape)}")
        batch, sequence, channels, height, width = x.shape
        if sequence != self.sequence_length or channels != self.input_dim:
            raise ValueError(
                "DynaSEAF input shape does not match configured history: "
                f"got (T={sequence}, C={channels}), expected "
                f"(T={self.sequence_length}, C={self.input_dim})"
            )
        finite_input = torch.isfinite(x)
        input_mask = self._resolve_input_valid_mask(valid_mask, x)
        x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if input_mask is not None:
            x_clean = x_clean * input_mask.to(dtype=x_clean.dtype).unsqueeze(1)

        features = self.direct_model.encode_features(x_clean)
        direct_forecast = self.direct_model.forecast_from_features(features)
        leads = torch.arange(
            1, self.prediction_length + 1, device=features.device, dtype=torch.long
        )
        lead_features = self.lead_embedding(leads, batch_size=batch)
        if self.future_dynamics_head is None:
            predicted_dynamics = features.new_zeros(
                batch, self.prediction_length, self.future_dynamics_channels, height, width
            )
        else:
            predicted_dynamics = self.future_dynamics_head(features, lead_features)

        latest_source = x_clean[:, -1].index_select(1, self.transport_source_indices)
        latest_valid = finite_input[:, -1].index_select(1, self.transport_source_indices)
        if input_mask is not None:
            latest_valid = latest_valid & input_mask.index_select(
                1, self.transport_source_indices
            )

        if self.use_transport and self.deformation_head is not None:
            deformation = self.deformation_head(
                features, predicted_dynamics, lead_features
            )
            target_deformation = deformation.index_select(
                2, self.target_to_deformation_indices
            )
            source_for_warp = latest_source.unsqueeze(1).expand(
                -1, self.prediction_length, -1, -1, -1
            )
            source_valid_for_warp = latest_valid.unsqueeze(1).expand(
                -1, self.prediction_length, -1, -1, -1
            )
            transport_forecast = self.warp(
                source_for_warp.reshape(
                    batch * self.prediction_length,
                    self.output_dim,
                    height,
                    width,
                ),
                target_deformation.reshape(
                    batch * self.prediction_length,
                    self.output_dim,
                    height,
                    width,
                    2,
                ),
                source_valid_for_warp.reshape(
                    batch * self.prediction_length,
                    self.output_dim,
                    height,
                    width,
                ),
            ).reshape(
                batch, self.prediction_length, self.output_dim, height, width
            )
        else:
            deformation = features.new_zeros(
                batch,
                self.prediction_length,
                self.deformation_channels,
                height,
                width,
                2,
            )
            transport_forecast = latest_source.unsqueeze(1).expand(
                -1, self.prediction_length, -1, -1, -1
            )

        if not self.use_transport:
            gate = torch.zeros_like(direct_forecast)
        elif self.transport_direct_gate is None:
            # The no-gate ablation is deliberately a fixed equal mixture;
            # only the adaptive-gate variant learns this spatial decision.
            gate = torch.full_like(direct_forecast, 0.5)
        else:
            gate_values = self.transport_direct_gate(
                features, predicted_dynamics, lead_features
            )
            gate = gate_values.index_select(2, self.target_to_gate_indices)

        if self.innovation_head is None:
            innovation = torch.zeros_like(direct_forecast)
        else:
            innovation = self.innovation_head(
                features, predicted_dynamics, lead_features
            )
        forecast = self.mix_forecasts(
            direct_forecast, transport_forecast, innovation, gate
        )
        if not return_diagnostics:
            return forecast
        return {
            "forecast": forecast,
            "direct_forecast": direct_forecast,
            "transport_forecast": transport_forecast,
            "innovation": innovation,
            "gate": gate,
            "deformation": deformation,
            "predicted_dynamics": predicted_dynamics,
        }

    @staticmethod
    def _parameter_count(module: Optional[nn.Module]) -> int:
        return 0 if module is None else sum(p.numel() for p in module.parameters())

    def parameter_breakdown(self) -> Dict[str, int]:
        breakdown = {
            "direct_forecaster": self._parameter_count(self.direct_model),
            "lead_embedding": self._parameter_count(self.lead_embedding),
            "future_dynamics_head": self._parameter_count(self.future_dynamics_head),
            "deformation_head": self._parameter_count(self.deformation_head),
            "innovation_head": self._parameter_count(self.innovation_head),
            "transport_direct_gate": self._parameter_count(
                self.transport_direct_gate
            ),
            "warp": 0,
        }
        breakdown["total"] = sum(p.numel() for p in self.parameters())
        return breakdown

    def model_diagnostics(self) -> Dict[str, object]:
        return {
            "model_display_name": "DynaSEAF",
            "direct_model": "SEAFNet",
            "use_future_dynamics_aux": self.use_future_dynamics_aux,
            "use_transport": self.use_transport,
            "use_innovation": self.use_innovation,
            "use_adaptive_gate": bool(
                self.use_transport and self.transport_direct_gate is not None
            ),
            "gate_resolution": self.gate_resolution,
            "future_dynamics_variables": list(self.future_dynamics_variables),
            "future_dynamics_channel_slices": {
                name: [int(value.start), int(value.stop)]
                for name, value in self.future_dynamics_channel_slices.items()
            },
            "transport_source_indices": self.transport_source_indices.detach().cpu().tolist(),
            "deformation_channels": self.deformation_channels,
            "parameter_breakdown": self.parameter_breakdown(),
        }


__all__ = [
    "LeadEmbedding",
    "FutureDynamicsHead",
    "DynamicsConditioner",
    "DeformationHead",
    "DifferentiableAnomalyWarp",
    "InnovationHead",
    "TransportDirectGate",
    "DynaSEAFNet",
]
