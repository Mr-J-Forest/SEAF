import torch

from apex_restored_model import APEXNet
from model_factory import create_ocean_model
from seaf_model import SEAFNet


def _config(model_type="apex_restored"):
    return {
        "model_type": model_type,
        "sequence_length": 3,
        "prediction_length": 2,
        "target_variables": ["TEMP", "SALT"],
        "actual_input_channels": 4,
        "actual_output_channels": 4,
        "input_channel_slices": {"TEMP": [0, 2], "SALT": [2, 4]},
        "apex_hidden_dim": 8,
        "apex_spectral_modes": [2, 2],
        "apex_spectral_layers": 1,
        "apex_ensemble_members": 4,
        "enable_ap_residual": True,
        "dropout": 0.0,
    }


def test_complete_apex_components_and_identity_initialization():
    model = APEXNet(_config()).eval()
    assert isinstance(model.spectral_encoder, torch.nn.Module)
    assert len(model.member_heads) == 4
    assert model.ensemble_gate is not None
    assert model.enable_ap_residual is True
    assert model.persistence_slices == [(0, 2), (2, 4)]
    assert torch.equal(model.ap_scale, torch.tensor(1.0))

    inputs = torch.randn(2, 3, 4, 6, 8)
    with torch.no_grad():
        outputs = model(inputs)
    expected = inputs[:, -1].unsqueeze(1).expand(-1, 2, -1, -1, -1)
    assert outputs.shape == (2, 2, 4, 6, 8)
    assert torch.equal(outputs, expected)


def test_factory_adds_apex_without_changing_seaf():
    restored = create_ocean_model(_config())
    assert isinstance(restored, APEXNet)
    seaf_config = _config("seaf")
    seaf_config.update(
        {
            "seaf_hidden_dim": 8,
            "seaf_spectral_modes": [2, 2],
            "seaf_spectral_layers": 1,
            "seaf_ensemble_members": 4,
        }
    )
    current = create_ocean_model(seaf_config)
    assert isinstance(current, SEAFNet)
    assert not hasattr(current, "ap_scale")
