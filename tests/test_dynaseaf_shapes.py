import torch

from model_factory import create_ocean_model
from tests.dynaseaf_test_utils import small_dynaseaf_config


def test_dynaseaf_forward_diagnostics_shapes_and_backward():
    torch.manual_seed(7)
    model = create_ocean_model(small_dynaseaf_config()).train()
    inputs = torch.randn(2, 3, 10, 5, 6, requires_grad=True)

    diagnostics = model(inputs, return_diagnostics=True)
    assert diagnostics["forecast"].shape == (2, 2, 4, 5, 6)
    assert diagnostics["direct_forecast"].shape == (2, 2, 4, 5, 6)
    assert diagnostics["transport_forecast"].shape == (2, 2, 4, 5, 6)
    assert diagnostics["innovation"].shape == (2, 2, 4, 5, 6)
    assert diagnostics["gate"].shape == (2, 2, 4, 5, 6)
    assert diagnostics["deformation"].shape == (2, 2, 2, 5, 6, 2)
    assert diagnostics["predicted_dynamics"].shape == (2, 2, 6, 5, 6)

    loss = diagnostics["forecast"].square().mean()
    loss = loss + diagnostics["predicted_dynamics"].square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(diagnostics["forecast"]).all()
