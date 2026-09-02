import torch

from dynaseaf_model import DifferentiableAnomalyWarp


def test_zero_displacement_is_identity():
    source = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    displacement = torch.zeros(1, 1, 4, 5, 2)
    warped = DifferentiableAnomalyWarp()(source, displacement)
    torch.testing.assert_close(warped, source)


def test_zero_displacement_is_identity_for_multiple_channels_without_mask():
    source = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    displacement = torch.zeros(2, 1, 4, 5, 2)
    warped = DifferentiableAnomalyWarp()(source, displacement)
    torch.testing.assert_close(warped, source)


def test_positive_x_displacement_moves_impulse_right():
    source = torch.zeros(1, 1, 3, 5)
    source[0, 0, 1, 1] = 1.0
    displacement = torch.zeros(1, 1, 3, 5, 2)
    displacement[..., 0] = 1.0
    warped = DifferentiableAnomalyWarp()(source, displacement)

    assert warped[0, 0, 1, 2] > 0.99
    assert warped[0, 0, 1, 1] < 0.01


def test_positive_y_displacement_moves_impulse_down():
    source = torch.zeros(1, 1, 5, 3)
    source[0, 0, 1, 1] = 1.0
    displacement = torch.zeros(1, 1, 5, 3, 2)
    displacement[..., 1] = 1.0
    warped = DifferentiableAnomalyWarp()(source, displacement)

    assert warped[0, 0, 2, 1] > 0.99
    assert warped[0, 0, 1, 1] < 0.01
