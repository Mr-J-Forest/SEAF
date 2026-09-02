from unittest import mock

import torch
import torch.optim as optim

from model_factory import create_ocean_model
from tests.dynaseaf_test_utils import small_dynaseaf_config
from train import OceanModelTrainer


def _minimal_dynaseaf_trainer():
    config = small_dynaseaf_config(
        use_gradient_loss=False,
        dynaseaf_lambda_dynamics=0.1,
        grad_clip_norm=1.0,
        log_interval=1,
    )
    model = create_ocean_model(config).train()
    trainer = OceanModelTrainer.__new__(OceanModelTrainer)
    trainer.config = config
    trainer.model = model
    trainer.forward_model = model
    trainer.is_dynaseaf = True
    trainer.dynaseaf_aux_enabled = True
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False
    trainer.amp_dtype = torch.bfloat16
    trainer.grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.compile_active = False
    trainer.compile_fallback_used = False
    trainer._compiled_forward_verified = False
    trainer.criterion = torch.nn.MSELoss()
    trainer.target_variables = ["TEMP", "SALT"]
    trainer.target_channel_slices = {"TEMP": slice(0, 2), "SALT": slice(2, 4)}
    trainer.target_loss_weights = {"TEMP": 0.5, "SALT": 0.5}
    trainer.dynaseaf_future_dynamics_variables = ["UVEL", "VVEL", "SSHA", "MLD"]
    trainer.future_dynamics_target_channel_slices = {
        "UVEL": [0, 2], "VVEL": [2, 4], "SSHA": [4, 5], "MLD": [5, 6]
    }
    trainer.optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer.epoch = 0
    trainer.nonfinite_grad_skip_limit = 30
    trainer.grad_overflow_skips_total = 0
    trainer._consecutive_nonfinite_grads = 0
    trainer.writer = mock.Mock()

    inputs = torch.randn(2, 3, 10, 4, 4)
    targets = torch.randn(2, 2, 4, 4, 4)
    future = torch.randn(2, 2, 6, 4, 4)
    valid = torch.ones_like(future)
    trainer.train_loader = [(inputs, targets, future, valid, torch.tensor([0, 1]))]
    return trainer


def test_train_epoch_consumes_auxiliary_labels_without_changing_input_shape():
    trainer = _minimal_dynaseaf_trainer()
    loss = trainer.train_epoch()
    assert torch.isfinite(torch.tensor(loss))
    assert len(trainer.dynaseaf_dynamics_losses) == 1
    assert trainer.dynaseaf_dynamics_losses[0] >= 0.0
