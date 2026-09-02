"""Small deterministic configurations shared by DynaSEAF unit tests."""

from __future__ import annotations

from typing import Any

from config import DEFAULT_CONFIG


def small_dynaseaf_config(**overrides: Any) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update({
        "model_type": "dynaseaf",
        "model_display_name": "DynaSEAF",
        "sequence_length": 3,
        "prediction_length": 2,
        "actual_input_channels": 10,
        "actual_output_channels": 4,
        "input_variables": ["TEMP", "SALT", "UVEL", "VVEL", "SSHA", "MLD"],
        "target_variables": ["TEMP", "SALT"],
        "external_dynamic_variables": ["UVEL", "VVEL", "SSHA", "MLD"],
        "input_channel_slices": {
            "TEMP": [0, 2],
            "SALT": [2, 4],
            "UVEL": [4, 6],
            "VVEL": [6, 8],
            "SSHA": [8, 9],
            "MLD": [9, 10],
        },
        "target_channel_slices": {"TEMP": [0, 2], "SALT": [2, 4]},
        "future_dynamics_target_channel_slices": {
            "UVEL": [0, 2],
            "VVEL": [2, 4],
            "SSHA": [4, 5],
            "MLD": [5, 6],
        },
        "dynaseaf_future_dynamics_channel_slices": {
            "UVEL": [0, 2],
            "VVEL": [2, 4],
            "SSHA": [4, 5],
            "MLD": [5, 6],
        },
        "dynaseaf_future_dynamics_variables": ["UVEL", "VVEL", "SSHA", "MLD"],
        "enable_climatology_anomaly": True,
        "enable_target_climatology_anomaly": True,
        "anomaly_variables": ["TEMP", "SALT", "UVEL", "VVEL", "SSHA", "MLD"],
        "target_anomaly_variables": ["TEMP", "SALT"],
        "seaf_hidden_dim": 16,
        "seaf_spectral_modes": [2, 2],
        "seaf_spectral_layers": 1,
        "seaf_ensemble_members": 2,
        "dropout": 0.0,
        "use_temporal_depth_mixer": False,
        "use_local_path": False,
        "use_spectral_path": True,
        "use_forcing_encoder": False,
        "dynaseaf_use_future_dynamics_aux": True,
        "dynaseaf_use_transport": True,
        "dynaseaf_use_innovation": True,
        "dynaseaf_use_adaptive_gate": True,
        "dynaseaf_zero_init_innovation": True,
        "dynaseaf_max_deformation_cells": 1.0,
        "dynaseaf_gate_initial_bias": -1.7346,
        "use_gradient_loss": False,
    })
    config.update(overrides)
    return config
