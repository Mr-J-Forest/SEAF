import torch

from dynaseaf_model import DifferentiableAnomalyWarp


def test_masked_nan_neighbor_does_not_contaminate_warp():
    source = torch.tensor([[[[1.0, float("nan"), 3.0]]]])
    mask = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    displacement = torch.zeros(1, 1, 1, 3, 2)
    displacement[..., 0] = -0.5
    warped = DifferentiableAnomalyWarp()(source, displacement, mask)

    assert torch.isfinite(warped).all()
    # At the center, masked renormalization keeps the valid right neighbor
    # instead of treating the NaN as a zero-valued ocean cell.
    torch.testing.assert_close(warped[0, 0, 0, 1], torch.tensor(3.0))


def test_invalid_source_cell_is_zero_when_no_valid_neighbor_exists():
    source = torch.tensor([[[[float("nan")]]]])
    mask = torch.zeros_like(source)
    displacement = torch.zeros(1, 1, 1, 1, 2)
    warped = DifferentiableAnomalyWarp()(source, displacement, mask)
    torch.testing.assert_close(warped, torch.zeros_like(warped))
