import torch

from model_factory import create_ocean_model
from tests.dynaseaf_test_utils import small_dynaseaf_config


def test_dynaseaf_cpu_bfloat16_autocast_smoke_is_finite():
    model = create_ocean_model(small_dynaseaf_config()).train()
    inputs = torch.randn(1, 3, 10, 4, 4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        diagnostics = model(inputs, return_diagnostics=True)
        loss = diagnostics["forecast"].square().mean()
        loss = loss + diagnostics["predicted_dynamics"].square().mean()
    loss.backward()
    assert torch.isfinite(diagnostics["forecast"]).all()
    assert torch.isfinite(diagnostics["predicted_dynamics"]).all()
