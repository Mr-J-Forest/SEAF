import torch

from dynaseaf_model import DynaSEAFNet
from model_factory import create_ocean_model
from tests.dynaseaf_test_utils import small_dynaseaf_config


def test_final_decomposition_respects_zero_and_one_gate():
    direct = torch.full((1, 2, 3, 2, 2), 2.0)
    transport = torch.full_like(direct, 5.0)
    innovation = torch.full_like(direct, 0.25)

    zero_gate = DynaSEAFNet.mix_forecasts(
        direct, transport, innovation, torch.zeros_like(direct)
    )
    one_gate = DynaSEAFNet.mix_forecasts(
        direct, transport, innovation, torch.ones_like(direct)
    )
    torch.testing.assert_close(zero_gate, direct + innovation)
    torch.testing.assert_close(one_gate, transport + innovation)


def test_zero_initialized_innovation_is_zero_at_initialization():
    model = create_ocean_model(small_dynaseaf_config())
    inputs = torch.randn(1, 3, 10, 4, 4)
    diagnostics = model(inputs, return_diagnostics=True)
    torch.testing.assert_close(
        diagnostics["innovation"], torch.zeros_like(diagnostics["innovation"])
    )


def test_gate_prior_is_spatially_constant_and_near_configured_probability():
    config = small_dynaseaf_config()
    model = create_ocean_model(config)
    inputs = torch.randn(1, 3, 10, 4, 4)
    gate = model(inputs, return_diagnostics=True)["gate"]
    expected = torch.sigmoid(torch.tensor(config["dynaseaf_gate_initial_bias"]))
    torch.testing.assert_close(gate, torch.full_like(gate, expected), atol=1e-6, rtol=1e-6)
