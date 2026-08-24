import unittest
import random
import json
import inspect
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_CONFIG, load_config, validate_config
from convlstm_model import create_ocean_model
from data_loader import (
    OceanDataset,
    TimeGroupedBatchSampler,
    validate_expected_canonical_window_count,
)
from predict import (
    SmartOceanPredictor,
    balanced_group_sample_positions,
    finalize_weighted_blend,
    interleaved_batches,
)
from scripts.run_experiment_queue import (
    expand_matrix,
    is_complete,
    linux_process_tree_rss_bytes,
    main as queue_main,
    resolve_stage_experiments,
)
from scripts import select_learning_rates
from train import (
    OceanModelTrainer,
    capture_rng_state,
    peak_host_memory_bytes,
    restore_rng_state,
    set_seed,
    training_config_fingerprint,
)


class PipelineIntegrityTests(unittest.TestCase):
    def test_peak_host_memory_is_nonnegative_when_supported(self):
        peak_bytes = peak_host_memory_bytes()
        self.assertTrue(peak_bytes is None or peak_bytes >= 0)

    def test_default_training_run_does_not_evaluate_test_split(self):
        self.assertEqual(DEFAULT_CONFIG['post_training_evaluation'], 'validation')
        self.assertEqual(
            inspect.signature(OceanModelTrainer.evaluate).parameters['split'].default,
            'validation',
        )

    def test_unknown_model_type_is_rejected_instead_of_becoming_convlstm(self):
        config = self._small_tsc_config()
        config['model_type'] = 'misspelled_model'
        with self.assertRaisesRegex(ValueError, '未知 model_type'):
            validate_config(config)
        with self.assertRaisesRegex(ValueError, '未知 model_type'):
            create_ocean_model(config)

    def test_paper_reimplementation_models_share_the_forecast_contract(self):
        for model_type in (
            'tianhai_paper',
            'fuxi_ocean_paper',
            'fuxi_ons_paper',
            'axiomocean_paper',
        ):
            with self.subTest(model_type=model_type):
                config = self._small_tsc_config()
                config.update({
                    'model_type': model_type,
                    'paper_hidden_dim': 8,
                    'paper_ensemble_members': 2,
                })
                model = create_ocean_model(config).eval()
                inputs = torch.zeros((2, 3, 8, 4, 4))
                with torch.no_grad():
                    outputs = model(inputs)
                self.assertEqual(outputs.shape, (2, 2, 3, 4, 4))

    def test_frozen_canonical_window_count_fails_on_data_protocol_drift(self):
        class DatasetStub:
            def __init__(self, count):
                self.all_regions_data = [None] * count

        validate_expected_canonical_window_count(
            {'train': DatasetStub(151), 'validation': DatasetStub(151)}, 151
        )
        validate_expected_canonical_window_count(
            {'train': DatasetStub(1), 'validation': DatasetStub(151)},
            {'train': 1, 'validation': 151},
        )
        with self.assertRaisesRegex(ValueError, '空间窗口协议已变化'):
            validate_expected_canonical_window_count(
                {'train': DatasetStub(151), 'validation': DatasetStub(150)}, 151
            )

    @staticmethod
    def _small_tsc_config():
        config = DEFAULT_CONFIG.copy()
        config.update({
            'sequence_length': 3,
            'prediction_length': 2,
            'actual_input_channels': 8,
            'actual_output_channels': 3,
            'input_channel_slices': {'TEMP': [0, 2], 'SALT': [2, 4]},
            'target_channel_slices': {'TEMP': [0, 1], 'SALT': [1, 3]},
            'tsc_variables': ['TEMP', 'SALT'],
            'tsc_hidden_dim': 8,
            'tsc_output_dim': 4,
            'tsc_attention_heads': 2,
            'tsc_ffn_dim': 16,
            'tsc_fusion_hidden_dim': 16,
            'tsc_fusion_spectral_modes': [2, 2],
            'tsc_fusion_ensemble_members': 2,
            'tsc_fusion_transformer_heads': 4,
            'tsc_fusion_transformer_ffn_dim': 32,
            'global_token_bank_heads': 4,
            'global_token_bank_ffn_dim': 32,
            'dropout': 0.0,
        })
        return config

    def test_channel_schema_is_initialized_without_worker_side_effects(self):
        dataset = OceanDataset.__new__(OceanDataset)
        dataset.all_regions_data = [{
            'normalized_data': {
                'TEMP': np.zeros((5, 2, 3, 4)),
                'SALT': np.zeros((5, 3, 3, 4)),
                'SSHA': np.zeros((5, 3, 4)),
                'CLIMATOLOGY_TEMP': np.zeros((5, 2, 3, 4)),
                'SPATIAL_ENCODING': np.zeros((1, 4, 3, 4)),
                'TIME_ENCODING': np.zeros((5, 3, 1, 1)),
            }
        }]
        dataset.input_variables = ['TEMP', 'SALT', 'SSHA']
        dataset.target_variables = ['TEMP', 'SALT']
        dataset.include_climatology_features = True
        dataset.climatology_feature_variables = ['TEMP']
        dataset.config = {'enable_positional_encoding': True, 'enable_time_encoding': True}
        dataset.input_channel_slices = {}
        dataset.target_channel_slices = {}

        dataset._initialize_channel_schema()

        self.assertEqual(dataset.input_channel_slices['TEMP'], slice(0, 2))
        self.assertEqual(dataset.input_channel_slices['SALT'], slice(2, 5))
        self.assertEqual(dataset.input_channel_slices['TIME_ENCODING'], slice(12, 15))
        self.assertEqual(dataset.target_channel_slices['TEMP'], slice(0, 2))
        self.assertEqual(dataset.target_channel_slices['SALT'], slice(2, 5))

    def test_compact_auxiliary_features_broadcast_only_on_allowed_axes(self):
        spatial = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
        spatial_window = OceanDataset._slice_and_broadcast_input(
            spatial, 7, 5, 3, 4,
            variable='SPATIAL_ENCODING',
            allow_static_time=True,
        )
        self.assertEqual(spatial_window.shape, (5, 2, 3, 4))
        np.testing.assert_array_equal(spatial_window[0], spatial_window[-1])

        time = np.arange(20, dtype=np.float32).reshape(10, 2, 1, 1)
        time_window = OceanDataset._slice_and_broadcast_input(
            time, 3, 5, 3, 4,
            variable='TIME_ENCODING',
            allow_spatial_broadcast=True,
        )
        self.assertEqual(time_window.shape, (5, 2, 3, 4))
        np.testing.assert_array_equal(time_window[:, :, 0, 0], time[3:8, :, 0, 0])
        np.testing.assert_array_equal(time_window[:, :, 0, 0], time_window[:, :, -1, -1])

        with self.assertRaisesRegex(ValueError, '空间形状不兼容'):
            OceanDataset._slice_and_broadcast_input(
                np.zeros((5, 2, 2, 4), dtype=np.float32),
                0, 5, 3, 4,
                variable='BROKEN',
            )

    def test_spatial_encoding_is_periodic_and_stored_once(self):
        dataset = OceanDataset.__new__(OceanDataset)
        dataset.config = {'positional_encoding_frequencies': 2}
        dataset.lons = np.array([0.0, 90.0, 360.0], dtype=np.float32)
        dataset.lats = np.array([-30.0, 30.0], dtype=np.float32)

        encoded = dataset._build_spatial_encoding(
            {'lons': dataset.lons, 'lats': dataset.lats},
            time_steps=121,
            lat_size=2,
            lon_size=3,
        )

        self.assertEqual(encoded.shape, (1, 8, 2, 3))
        np.testing.assert_allclose(encoded[..., 0], encoded[..., -1], atol=1e-6)

    def test_external_scalers_change_cache_fingerprint(self):
        first = StandardScaler().fit(np.array([[0.0], [1.0]]))
        second = StandardScaler().fit(np.array([[10.0], [20.0]]))
        fp1 = OceanDataset._scalers_fingerprint({'TEMP': first})
        fp2 = OceanDataset._scalers_fingerprint({'TEMP': second})
        self.assertNotEqual(fp1, fp2)
        self.assertEqual(fp1, OceanDataset._scalers_fingerprint({'TEMP': first}))

    def test_cache_identity_excludes_rebuilt_encodings_and_output_subset(self):
        with tempfile.NamedTemporaryFile() as handle:
            dataset = OceanDataset.__new__(OceanDataset)
            dataset.data_path = handle.name
            dataset.config = DEFAULT_CONFIG.copy()
            dataset.stride_lon = 8.0
            dataset.stride_lat = 8.0
            dataset.train_ratio = 0.6
            dataset.val_ratio = 0.2
            dataset.provided_scalers = None
            dataset.sliding_enabled = True
            base_key = dataset._compute_cache_key()

            dataset.config = {**dataset.config, 'target_variables': ['TEMP']}
            self.assertEqual(dataset._compute_cache_key(), base_key)
            dataset.config = {**dataset.config, 'enable_positional_encoding': False}
            self.assertEqual(dataset._compute_cache_key(), base_key)

            dataset.sliding_enabled = False
            self.assertNotEqual(dataset._compute_cache_key(), base_key)

    def test_equivalent_validation_and_test_preprocessing_share_one_payload(self):
        with tempfile.NamedTemporaryFile() as handle, tempfile.TemporaryDirectory() as cache_dir:
            scaler = StandardScaler().fit(np.array([[0.0], [1.0]]))

            def dataset(mode):
                source = OceanDataset.__new__(OceanDataset)
                source.mode = mode
                source.data_path = handle.name
                source.config = {
                    **DEFAULT_CONFIG,
                    'cache_preprocessed_dir': cache_dir,
                }
                source.stride_lon = 8.0
                source.stride_lat = 8.0
                source.train_ratio = 0.6
                source.val_ratio = 0.2
                source.provided_scalers = {'TEMP': scaler}
                source.sliding_enabled = True
                return source

            self.assertEqual(dataset('val')._cache_dir(), dataset('test')._cache_dir())

    def test_temporal_split_views_share_arrays_but_own_sequence_indices(self):
        dataset = OceanDataset.__new__(OceanDataset)
        dataset.config = {
            'sliding_enabled': True,
            'train_stride_lon': 8.0,
            'train_stride_lat': 8.0,
            'val_stride_lon': 8.0,
            'val_stride_lat': 8.0,
            'test_stride_lon': 8.0,
            'test_stride_lat': 8.0,
            'split_context_policy': 'carry_history',
        }
        dataset.mode = 'train'
        dataset.sliding_enabled = True
        dataset.stride_lon = 8.0
        dataset.stride_lat = 8.0
        dataset.override_stride_lon = None
        dataset.override_stride_lat = None
        dataset.train_ratio = 0.6
        dataset.val_ratio = 0.2
        dataset.times = np.arange(121)
        dataset.sequence_length = 12
        dataset.prediction_length = 5
        dataset.all_regions_data = [{'region_type': 'sliding'}]
        dataset.scalers = {'TEMP': object()}
        dataset.provided_scalers = None
        dataset.return_sample_index = True
        dataset._split_data()
        dataset._create_sequences()

        validation = dataset.temporal_split_view('val')
        test = validation.temporal_split_view('test')

        self.assertIs(validation.all_regions_data, dataset.all_regions_data)
        self.assertIs(test.all_regions_data, dataset.all_regions_data)
        self.assertIs(validation.scalers, dataset.scalers)
        self.assertEqual([start for start, _ in dataset.sequences], list(range(56)))
        self.assertEqual([start for start, _ in validation.sequences], list(range(60, 80)))
        self.assertEqual([start for start, _ in test.sequences], list(range(84, 105)))
        self.assertFalse(validation.return_sample_index)
        self.assertTrue(dataset.return_sample_index)

    def test_temporal_split_view_rejects_different_spatial_protocol(self):
        dataset = OceanDataset.__new__(OceanDataset)
        dataset.config = {
            'sliding_enabled': True,
            'train_stride_lon': 8.0,
            'train_stride_lat': 8.0,
            'val_stride_lon': 4.0,
            'val_stride_lat': 4.0,
        }
        dataset.mode = 'train'
        dataset.sliding_enabled = True
        dataset.stride_lon = 8.0
        dataset.stride_lat = 8.0

        self.assertFalse(dataset.can_share_preprocessed_with_mode('val'))
        with self.assertRaisesRegex(ValueError, '不能共享数组'):
            dataset.temporal_split_view('val')

    def test_incomplete_cache_without_success_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = OceanDataset.__new__(OceanDataset)
            dataset.config = {'cache_preprocessed': True}
            dataset._cache_dir = lambda: temporary
            Path(temporary, 'metadata.json').write_text('{}', encoding='utf-8')
            Path(temporary, 'scalers.pkl').write_bytes(b'incomplete')

            self.assertFalse(dataset._try_load_from_cache())

    def test_missing_value_fallback_never_uses_validation_or_test_values(self):
        dataset = OceanDataset.__new__(OceanDataset)
        dataset.times = np.arange(4)
        dataset.train_ratio = 0.5
        data = np.asarray([
            [[np.nan]],
            [[np.inf]],
            [[10.0]],
            [[20.0]],
        ])

        filled = dataset._fill_missing_values(data.copy(), variable='UWND')

        np.testing.assert_array_equal(filled[:2], 0.0)
        np.testing.assert_array_equal(filled[2:, 0, 0], [10.0, 20.0])

    def test_inverse_transform_requires_exact_sample_provenance_length(self):
        dataset = OceanDataset.__new__(OceanDataset)
        dataset.sequences = [(0, 0), (1, 0)]

        with self.assertRaisesRegex(ValueError, '数量'):
            dataset._validated_sample_indices([0], sample_count=2)
        with self.assertRaisesRegex(IndexError, '超出数据集范围'):
            dataset._validated_sample_indices([0, 2], sample_count=2)

    def test_target_transforms_preserve_float32_storage(self):
        dataset = OceanDataset.__new__(OceanDataset)
        dataset.sequences = [(0, 0), (1, 0)]
        dataset.target_variables = ['TEMP']
        dataset.target_channel_slices = {'TEMP': slice(0, 1)}
        dataset.config = {}
        dataset.scalers = {
            'TEMP': StandardScaler().fit(np.asarray([[10.0], [14.0]]))
        }
        dataset.enable_climatology_anomaly = False
        dataset.anomaly_variables = []
        model_values = np.zeros((2, 1, 1, 1, 1), dtype=np.float32)

        physical = dataset.inverse_transform_targets(
            model_values,
            sample_indices=[0, 1],
        )
        recovered = dataset.transform_targets_to_model_space(
            physical,
            sample_indices=[0, 1],
        )

        self.assertEqual(physical.dtype, np.float32)
        self.assertEqual(recovered.dtype, np.float32)
        np.testing.assert_allclose(physical, 12.0)
        np.testing.assert_allclose(recovered, model_values, atol=1e-6)

    def test_uncovered_blend_cells_are_nan(self):
        pred_sum = np.ones((2, 2, 3), dtype=np.float64) * 4
        target_sum = np.ones((2, 2, 3), dtype=np.float64) * 2
        weights = np.array([[2.0, 0.0, 1.0], [0.0, 4.0, 0.0]])
        pred, target, mask = finalize_weighted_blend(pred_sum, target_sum, weights)

        self.assertTrue(np.isnan(pred[:, ~mask]).all())
        self.assertTrue(np.isnan(target[:, ~mask]).all())
        self.assertTrue(np.isfinite(pred[:, mask]).all())
        self.assertAlmostEqual(pred[0, 0, 0], 2.0)

    def test_sampled_geographic_error_uses_real_window_coordinates_and_schema(self):
        dataset = type('DatasetStub', (), {})()
        dataset.lons = np.asarray([0.0, 1.0, 2.0, 3.0])
        dataset.lats = np.asarray([10.0, 11.0])
        dataset.sequences = [(0, 0), (0, 1)]
        dataset.all_regions_data = [
            {'coords': {'lons': np.asarray([0.0, 1.0]), 'lats': dataset.lats}},
            {'coords': {'lons': np.asarray([1.0, 2.0]), 'lats': dataset.lats}},
        ]
        channel_slices = {'A': slice(0, 1), 'B': slice(1, 3)}
        targets = [torch.zeros((1, 3, 2, 2)), torch.zeros((1, 3, 2, 2))]
        predictions = [
            torch.cat((torch.ones((1, 1, 2, 2)), 2 * torch.ones((1, 2, 2, 2))), dim=1),
            torch.cat((3 * torch.ones((1, 1, 2, 2)), 4 * torch.ones((1, 2, 2, 2))), dim=1),
        ]

        with tempfile.TemporaryDirectory() as temporary:
            predictor = SmartOceanPredictor.__new__(SmartOceanPredictor)
            predictor.output_dir = temporary
            predictor._plot_geographic_error(
                predictions,
                targets,
                dataset,
                [0, 1],
                channel_slices,
            )
            with np.load(Path(temporary, 'sampled_geographic_metrics.npz')) as report:
                self.assertTrue(np.isnan(report['A_rmse'][0, 3]))
                np.testing.assert_allclose(report['A_rmse'][:, 0], 1.0)
                np.testing.assert_allclose(report['A_rmse'][:, 1], np.sqrt(5.0))
                np.testing.assert_allclose(report['A_rmse'][:, 2], 3.0)
                np.testing.assert_allclose(report['B_rmse'][:, 1], np.sqrt(10.0))

    def test_window_starts_include_terminal_domain_anchor(self):
        starts = OceanDataset._window_starts(0.5, 359.5, 32.0, 8.0)

        self.assertEqual(starts[0], 0.5)
        self.assertEqual(starts[-1], 327.5)
        self.assertEqual(starts[-2], 320.5)
        self.assertEqual(len(starts), len(set(starts)))

    def test_inference_batches_interleave_spatially_ordered_windows(self):
        self.assertEqual(
            interleaved_batches(range(10), 4),
            [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]],
        )
        dataset = type('DatasetStub', (), {})()
        dataset.sequences = [(0, region_idx) for region_idx in range(10)]
        batches = list(TimeGroupedBatchSampler(dataset, batch_size=4, shuffle=False))
        self.assertEqual(batches, [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]])

    def test_prediction_samples_span_time_groups_and_space(self):
        positions = balanced_group_sample_positions([151] * 10, 10)

        self.assertEqual([len(group) for group in positions], [1] * 10)
        self.assertEqual(positions[0], [0])
        self.assertEqual(positions[-1], [150])

    def test_weighted_loss_uses_explicit_unequal_channel_slices(self):
        trainer = OceanModelTrainer.__new__(OceanModelTrainer)
        trainer.config = {'use_gradient_loss': False}
        trainer.criterion = torch.nn.MSELoss()
        trainer.target_variables = ['A', 'B']
        trainer.target_channel_slices = {'A': slice(0, 1), 'B': slice(1, 3)}
        trainer.target_loss_weights = {'A': 0.25, 'B': 0.75}
        target = torch.zeros((1, 1, 3, 2, 2))
        output = torch.cat((torch.ones_like(target[:, :, :1]), 2 * torch.ones_like(target[:, :, 1:])), dim=2)

        loss, by_variable, grad_loss = trainer.compute_weighted_loss(output, target)

        self.assertAlmostEqual(loss.item(), 0.25 * 1.0 + 0.75 * 4.0)
        self.assertAlmostEqual(by_variable['A'], 1.0)
        self.assertAlmostEqual(by_variable['B'], 4.0)
        self.assertEqual(grad_loss, 0.0)

    def test_data_protocol_is_built_from_all_three_loaders(self):
        def dataset(mode, sequences):
            return type('DatasetStub', (), {
                'sequences': sequences,
                'sequence_length': 2,
                'prediction_length': 1,
                'all_regions_data': [{'region_type': mode}],
            })()

        class LoaderStub:
            def __init__(self, source, batches):
                self.dataset = source
                self._batches = batches

            def __len__(self):
                return self._batches

        trainer = OceanModelTrainer.__new__(OceanModelTrainer)
        trainer.config = {
            'split_context_policy': 'carry_history',
            'data_path': str(Path(__file__).resolve()),
        }
        trainer.train_loader = LoaderStub(dataset('train', [(0, 0), (1, 0)]), 2)
        trainer.val_loader = LoaderStub(dataset('validation', [(2, 0)]), 1)
        trainer.test_loader = LoaderStub(dataset('test', [(3, 0)]), 1)

        protocol = trainer._build_data_protocol()

        self.assertEqual(protocol['train']['samples'], 2)
        self.assertEqual(protocol['validation']['target_start_min'], 4)
        self.assertEqual(protocol['test']['target_end_max'], 5)
        self.assertEqual(protocol['loader_batches'], {
            'train': 2, 'validation': 1, 'test': 1,
        })

    def test_vector_gradient_loss_penalizes_reversed_edges(self):
        trainer = OceanModelTrainer.__new__(OceanModelTrainer)
        ramp = torch.arange(6, dtype=torch.float32).view(1, 1, 1, 1, 6).expand(1, 1, 1, 5, 6)

        trainer.config = {'gradient_loss_mode': 'vector'}
        vector_loss = trainer.compute_gradient_loss(-ramp, ramp)
        trainer.config = {'gradient_loss_mode': 'magnitude'}
        magnitude_loss = trainer.compute_gradient_loss(-ramp, ramp)

        self.assertGreater(vector_loss.item(), 0.0)
        self.assertAlmostEqual(magnitude_loss.item(), 0.0, places=7)

    def test_small_tsc_fusion_forward_and_backward(self):
        config = self._small_tsc_config()
        model = create_ocean_model(config)
        inputs = torch.randn(3, 3, 8, 5, 6, requires_grad=True)
        outputs = model(inputs)
        outputs.mean().backward()

        self.assertEqual(outputs.shape, (3, 2, 3, 5, 6))
        self.assertIsNotNone(inputs.grad)

    def test_external_global_bank_makes_inference_partition_invariant(self):
        config = self._small_tsc_config()
        model = create_ocean_model(config).eval()
        model.global_token_bank.gate.data.fill_(0.75)
        inputs = torch.randn(6, 3, 8, 5, 6)

        with torch.no_grad():
            full_output = model(inputs)
            bank = model.build_global_token_bank(inputs)
            partitioned = torch.cat([
                model(inputs[:2], global_bank_tokens=bank),
                model(inputs[2:5], global_bank_tokens=bank),
                model(inputs[5:], global_bank_tokens=bank),
            ])

        torch.testing.assert_close(partitioned, full_output, rtol=1e-5, atol=1e-6)

    def test_ablation_removes_disabled_parameters(self):
        full = create_ocean_model(self._small_tsc_config())

        no_tsc_config = self._small_tsc_config()
        no_tsc_config['ablation_disable_tsc'] = True
        no_tsc = create_ocean_model(no_tsc_config)
        self.assertIsNone(no_tsc.thermohaline_memory)
        self.assertLess(
            sum(p.numel() for p in no_tsc.parameters()),
            sum(p.numel() for p in full.parameters()),
        )

        no_ensemble_config = self._small_tsc_config()
        no_ensemble_config['ablation_disable_ensemble'] = True
        no_ensemble = create_ocean_model(no_ensemble_config)
        self.assertEqual(len(no_ensemble.member_heads), 1)
        self.assertIsNone(no_ensemble.ensemble_gate)

        no_persistence_config = self._small_tsc_config()
        no_persistence_config['enable_persistence_residual'] = False
        no_persistence = create_ocean_model(no_persistence_config)
        self.assertEqual(no_persistence.persistence_slices, [])
        self.assertIsNone(no_persistence.persistence_scale)

    def test_climatology_is_independent_from_anomaly_training_switch(self):
        dataset = OceanDataset.__new__(OceanDataset)
        dataset.times = np.array([200901, 200902, 201001, 201002])
        dataset.time_period_indices = np.array([0, 1, 0, 1])
        dataset.climatology_period = 2
        dataset.train_ratio = 1.0
        dataset.target_variables = ['TEMP']
        dataset.enable_climatology_anomaly = False
        dataset.anomaly_variables = {'TEMP'}
        dataset.include_climatology_features = True
        dataset.climatology_feature_variables = ['TEMP']
        region = {
            'data': {
                'TEMP': np.array([1.0, 10.0, 3.0, 14.0], dtype=np.float32)[:, None, None],
            }
        }

        dataset._compute_region_climatology(region)

        np.testing.assert_allclose(region['climatology']['TEMP'][:, 0, 0], [2.0, 12.0])
        self.assertEqual(region['anomaly_data'], {})

    def test_global_token_bank_requires_time_grouping(self):
        config = DEFAULT_CONFIG.copy()
        config['group_batches_by_time'] = False
        with self.assertRaisesRegex(ValueError, 'group_batches_by_time'):
            validate_config(config)

    def test_time_group_global_bank_requires_canonical_split_grid(self):
        config = DEFAULT_CONFIG.copy()
        config['test_stride_lon'] = config['train_stride_lon'] * 2
        with self.assertRaisesRegex(ValueError, 'canonical'):
            validate_config(config)

    def test_checkpoint_rng_snapshot_restores_all_cpu_generators(self):
        set_seed(31415)
        state = capture_rng_state()
        expected = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )

        set_seed(7)
        self.assertTrue(restore_rng_state(state))
        actual = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_resume_fingerprint_ignores_only_runtime_controls(self):
        base = {'epochs': 80, 'batch_size': 151, 'learning_rate': 8e-4, 'training_note': 'a'}
        extended = {
            **base,
            'epochs': 120,
            'training_note': 'b',
            'resume_dir': '/tmp/run',
            'post_training_evaluation': 'test',
        }
        base['post_training_evaluation'] = 'validation'
        changed_batch = {**base, 'batch_size': 64}

        self.assertEqual(
            training_config_fingerprint(base),
            training_config_fingerprint(extended),
        )
        self.assertNotEqual(
            training_config_fingerprint(base),
            training_config_fingerprint(changed_batch),
        )

    def test_config_extends_resolves_relative_paths_and_rejects_cycles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'base.json').write_text(
                json.dumps({'epochs': 80, 'batch_size': 16}), encoding='utf-8'
            )
            (root / 'child.json').write_text(
                json.dumps({'extends': 'base.json', 'batch_size': 32}), encoding='utf-8'
            )
            self.assertEqual(
                load_config(root / 'child.json'),
                {'epochs': 80, 'batch_size': 32},
            )

            (root / 'base.json').write_text(
                json.dumps({'extends': 'child.json'}), encoding='utf-8'
            )
            with self.assertRaisesRegex(ValueError, '循环'):
                load_config(root / 'child.json')

    def test_queue_accepts_auditable_no_evaluation_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / '_SUCCESS').write_text('done\n', encoding='utf-8')
            (run_dir / 'config.json').write_text('{}', encoding='utf-8')
            (run_dir / 'run_summary.json').write_text(json.dumps({
                'status': 'completed',
                'evaluation_scope': 'none',
                'evaluation_file': None,
                'training_source_hash': 'abc',
            }), encoding='utf-8')

            self.assertTrue(is_complete(run_dir, expected_source_hash='abc'))

    def test_queue_sums_linux_process_tree_rss(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc_root = Path(temporary)
            for pid, rss_kib, children in ((10, 100, '11 12'), (11, 25, ''), (12, 75, '')):
                process_dir = proc_root / str(pid)
                task_dir = process_dir / 'task' / str(pid)
                task_dir.mkdir(parents=True)
                (process_dir / 'status').write_text(
                    f'Name:\ttest\nVmRSS:\t{rss_kib} kB\n',
                    encoding='utf-8',
                )
                (task_dir / 'children').write_text(children, encoding='utf-8')

            self.assertEqual(
                linux_process_tree_rss_bytes(10, proc_root=proc_root),
                200 * 1024,
            )
            self.assertIsNone(linux_process_tree_rss_bytes(999, proc_root=proc_root))

    def test_queue_executes_and_finalizes_a_successful_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            matrix_path = project_root / 'matrix.json'
            config_path = project_root / 'config.json'
            matrix_path.write_text(json.dumps({
                'unit': [{
                    'name': 'fake',
                    'config': str(config_path),
                    'seeds': [42],
                    'overrides': {'post_training_evaluation': 'none'},
                }],
            }), encoding='utf-8')
            config_path.write_text('{}', encoding='utf-8')
            (project_root / 'source_state.json').write_text(json.dumps({
                'training_source_hash': 'abc',
            }), encoding='utf-8')

            class SuccessfulProcess:
                pid = 999999

                def __init__(self, command, **_kwargs):
                    run_dir = Path(command[command.index('--result_dir') + 1])
                    run_dir.mkdir(parents=True)
                    (run_dir / 'config.json').write_text('{}', encoding='utf-8')
                    (run_dir / 'run_summary.json').write_text(json.dumps({
                        'status': 'completed',
                        'evaluation_scope': 'none',
                        'evaluation_file': None,
                        'training_source_hash': 'abc',
                    }), encoding='utf-8')
                    (run_dir / '_SUCCESS').write_text('done\n', encoding='utf-8')

                def poll(self):
                    return 0

            with mock.patch(
                'scripts.run_experiment_queue.subprocess.Popen',
                SuccessfulProcess,
            ):
                returncode = queue_main(
                    [
                        '--matrix', str(matrix_path),
                        '--stage', 'unit',
                        '--campaign', 'abc_unit',
                    ],
                    project_root=project_root,
                )

            self.assertEqual(returncode, 0)
            state = json.loads((
                project_root / 'outputs' / 'results' / 'campaigns' / 'abc_unit'
                / 'experiment_queue_state.json'
            ).read_text(encoding='utf-8'))
            self.assertEqual(state['jobs'][0]['status'], 'completed')
            self.assertEqual(state['jobs'][0]['returncode'], 0)

    def test_learning_rate_selector_rejects_coarse_grid_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / 'global_lr_calibrate'
            losses_by_rate = {
                3e-4: [1.0, 1.0, 1.0],
                8e-4: [1.2, 1.2, 1.2],
                1.5e-3: [1.3, 1.3, 1.3],
                3e-3: [1.4, 1.4, 1.4],
            }
            for label, rate in (('3e4', 3e-4), ('8e4', 8e-4), ('15e4', 1.5e-3), ('3e3', 3e-3)):
                run_dir = stage / f'lr_full_{label}' / 'seed_42'
                run_dir.mkdir(parents=True)
                (run_dir / '_SUCCESS').write_text('done\n', encoding='utf-8')
                (run_dir / 'config.json').write_text(
                    json.dumps({'learning_rate': rate}), encoding='utf-8'
                )
                (run_dir / 'run_summary.json').write_text(json.dumps({
                    'evaluation_scope': 'none',
                    'validation_selection_losses': losses_by_rate[rate],
                    'training_source_hash': 'abc',
                }), encoding='utf-8')

            output = root / 'selection.json'
            with mock.patch.object(sys, 'argv', [
                'select_learning_rates.py', '--results-root', str(root), '--output', str(output),
            ]):
                self.assertEqual(select_learning_rates.main(), 1)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'incomplete')
            self.assertTrue(result['selections']['full']['requires_supplemental_calibration'])
            self.assertIn('boundary', result['errors'][0])

    def test_stage_overrides_are_frozen_into_expanded_jobs(self):
        matrix = {
            '_stage_overrides': {'screen': {'epochs': 30, 'post_training_evaluation': 'validation'}},
            'screen': [{
                'name': 'full',
                'config': 'full.json',
                'seeds': [42],
                'overrides': {'epochs': 5},
            }],
        }

        jobs = expand_matrix(matrix, ['screen'], only=None)

        self.assertEqual(jobs[0]['overrides'], {
            'epochs': 5,
            'post_training_evaluation': 'validation',
        })

    def test_stage_inheritance_replaces_seeds_without_copying_experiments(self):
        matrix = {
            'screen': [{
                'name': 'full', 'config': 'full.json', 'seeds': [42],
            }],
            'confirm_validation': {
                'from_stage': 'screen', 'seeds': [42, 123, 3407],
            },
        }
        resolved = resolve_stage_experiments(matrix, 'confirm_validation')
        self.assertEqual(resolved[0]['seeds'], [42, 123, 3407])
        self.assertEqual(matrix['screen'][0]['seeds'], [42])

    def test_evaluation_provenance_matches_prediction_order(self):
        dataset = OceanDataset.__new__(OceanDataset)
        dataset.sequences = [(3, 0), (3, 1), (4, 0)]
        dataset.sequence_length = 2
        dataset.prediction_length = 2
        dataset.climatology_period = 12
        dataset.times = np.asarray([f'2000-{month:02d}' for month in range(1, 13)])
        dataset.time_period_indices = np.arange(12)
        dataset.all_regions_data = [
            {
                'region_type': 'sliding',
                'lon_range': [130.0, 132.0],
                'lat_range': [5.0, 7.0],
                'coords': {'lons': [130.0, 132.0], 'lats': [5.0, 7.0]},
            },
            {
                'region_type': 'sliding',
                'lon_range': [140.0, 142.0],
                'lat_range': [15.0, 17.0],
                'coords': {'lons': [140.0, 142.0], 'lats': [15.0, 17.0]},
            },
        ]

        provenance = dataset.build_sample_provenance([2, 0])

        self.assertEqual(provenance['origin_ids'], [4, 3])
        self.assertEqual(provenance['region_ids'], [0, 0])
        self.assertEqual(provenance['target_time_indices'], [[6, 7], [5, 6]])
        self.assertEqual(provenance['target_period_ids'], [[6, 7], [5, 6]])
        self.assertEqual(provenance['samples'][0]['sample_index'], 2)

    def test_target_anchored_split_carries_history_without_target_leakage(self):
        dataset = OceanDataset.__new__(OceanDataset)
        dataset.mode = 'test'
        dataset.time_indices = list(range(96, 121))
        dataset.sequence_length = 12
        dataset.prediction_length = 5
        dataset.config = {'split_context_policy': 'carry_history'}
        dataset.all_regions_data = [{'region_type': 'sliding'}]

        dataset._create_sequences()

        starts = [start for start, _ in dataset.sequences]
        self.assertEqual(starts, list(range(84, 105)))
        for start in starts:
            self.assertGreaterEqual(start + dataset.sequence_length, 96)
            self.assertLessEqual(
                start + dataset.sequence_length + dataset.prediction_length - 1,
                120,
            )


if __name__ == '__main__':
    unittest.main()
