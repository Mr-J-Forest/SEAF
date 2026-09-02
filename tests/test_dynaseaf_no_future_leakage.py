import inspect

import torch

from model_factory import create_ocean_model
from tests.dynaseaf_test_utils import small_dynaseaf_config
from train import OceanModelTrainer


def test_forward_has_no_future_label_argument_and_is_label_independent():
    config = small_dynaseaf_config()
    model = create_ocean_model(config).eval()
    inputs = torch.randn(2, 3, 10, 4, 4)
    future_a = torch.randn(2, 2, 6, 4, 4)
    future_b = torch.flip(future_a, dims=(0, 1, 2, 3, 4))

    assert "future" not in inspect.signature(model.forward).parameters
    with torch.no_grad():
        prediction_a = model(inputs)
        prediction_b = model(inputs)
    torch.testing.assert_close(prediction_a, prediction_b)
    assert future_a.shape == future_b.shape  # labels are only an external loss input


def test_dynamics_labels_change_only_the_auxiliary_loss():
    trainer = OceanModelTrainer.__new__(OceanModelTrainer)
    trainer.config = config = small_dynaseaf_config()
    trainer.dynaseaf_future_dynamics_variables = list(
        config["dynaseaf_future_dynamics_variables"]
    )
    trainer.future_dynamics_target_channel_slices = dict(
        config["future_dynamics_target_channel_slices"]
    )
    predicted = torch.zeros(1, 2, 6, 2, 2)
    labels_a = torch.zeros_like(predicted)
    labels_b = torch.ones_like(predicted)
    loss_a, _ = trainer.compute_dynamics_loss(predicted, labels_a)
    loss_b, _ = trainer.compute_dynamics_loss(predicted, labels_b)
    torch.testing.assert_close(loss_a, torch.zeros(()))
    torch.testing.assert_close(loss_b, torch.ones(()))
