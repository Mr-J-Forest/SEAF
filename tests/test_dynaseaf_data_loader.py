import numpy as np

from data_loader import OceanDataset


def _dataset_for_auxiliary_target(return_future):
    dataset = OceanDataset.__new__(OceanDataset)
    dataset._preassembled_mmap_enabled = False
    dataset.sequences = [(0, 0)]
    dataset.sequence_length = 2
    dataset.prediction_length = 2
    dataset.input_variables = ["TEMP", "SALT", "UVEL", "VVEL", "SSHA", "MLD"]
    dataset.actual_input_variables = list(dataset.input_variables)
    dataset.target_variables = ["TEMP", "SALT"]
    dataset.future_dynamics_target_variables = ["UVEL", "VVEL", "SSHA", "MLD"]
    dataset.return_future_dynamics_targets = return_future
    dataset.return_sample_index = True
    dataset.include_climatology_features = False
    dataset.include_tendency_features = False
    dataset.config = {
        "enable_positional_encoding": False,
        "enable_time_encoding": False,
    }
    dataset.input_channel_slices = {}
    dataset.target_channel_slices = {}
    dataset.future_dynamics_target_channel_slices = {}

    def field(channels, offset):
        return (
            np.arange(5 * channels * 2 * 2, dtype=np.float32).reshape(5, channels, 2, 2)
            + offset
        )

    dataset.all_regions_data = [{
        "normalized_data": {
            "TEMP": field(2, 0),
            "SALT": field(2, 100),
            "UVEL": field(2, 200),
            "VVEL": field(2, 300),
            "SSHA": field(1, 400)[:, 0],
            "MLD": field(1, 500)[:, 0],
        }
    }]
    return dataset


def test_future_dynamics_are_appended_after_input_and_target_fields():
    dataset = _dataset_for_auxiliary_target(True)
    inputs, targets, future, mask, sample_index = dataset[0]

    assert inputs.shape == (2, 10, 2, 2)
    assert targets.shape == (2, 4, 2, 2)
    assert future.shape == (2, 6, 2, 2)
    assert mask.shape == future.shape
    assert sample_index == 0
    np.testing.assert_allclose(future[:, :2], dataset.all_regions_data[0]["normalized_data"]["UVEL"][2:4])
    np.testing.assert_allclose(inputs[:, 4:6], dataset.all_regions_data[0]["normalized_data"]["UVEL"][0:2])


def test_default_style_dataset_tuple_remains_backward_compatible():
    dataset = _dataset_for_auxiliary_target(False)
    result = dataset[0]
    assert len(result) == 3
    assert result[0].shape == (2, 10, 2, 2)
