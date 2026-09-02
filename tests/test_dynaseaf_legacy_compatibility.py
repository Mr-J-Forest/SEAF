import torch

from config import DEFAULT_CONFIG
from model_factory import create_ocean_model
from seaf_model import SEAFNet
from tests.dynaseaf_test_utils import small_dynaseaf_config


def test_legacy_seaf_checkpoint_roundtrip_remains_strictly_loadable():
    config = DEFAULT_CONFIG.copy()
    config.update({
        "model_type": "seaf",
        "sequence_length": 3,
        "prediction_length": 2,
        "actual_input_channels": 8,
        "actual_output_channels": 3,
        "input_channel_slices": {"TEMP": [0, 2], "SALT": [2, 4]},
        "target_channel_slices": {"TEMP": [0, 1], "SALT": [1, 3]},
        "seaf_hidden_dim": 16,
        "seaf_spectral_modes": [2, 2],
        "seaf_spectral_layers": 1,
        "seaf_ensemble_members": 2,
        "dropout": 0.0,
    })
    torch.manual_seed(11)
    original = SEAFNet(config).eval()
    restored = SEAFNet(config).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    inputs = torch.randn(2, 3, 8, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(original(inputs), restored(inputs))


def test_dynaseaf_uses_seaf_as_direct_submodule_without_ap_or_dap_inputs():
    model = create_ocean_model(small_dynaseaf_config())
    assert isinstance(model.direct_model, SEAFNet)
    assert not hasattr(model, "ap_scale")
    assert not hasattr(model, "damped_persistence")


def test_dynaseaf_checkpoint_state_dict_roundtrip_is_strict():
    config = small_dynaseaf_config()
    torch.manual_seed(23)
    original = create_ocean_model(config).eval()
    restored = create_ocean_model(config).eval()
    restored.load_state_dict(original.state_dict(), strict=True)

    inputs = torch.randn(2, 3, 10, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(original(inputs), restored(inputs))
