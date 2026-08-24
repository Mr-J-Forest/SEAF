import unittest

import numpy as np

from metrics_utils import (
    _basic_metrics,
    _field_metric_values,
    compute_metric_report,
    compute_period_group_report,
    compute_sample_group_report,
    resolve_variable_slices,
)


class MetricReportTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.target = rng.normal(size=(2, 2, 3, 3, 4))
        self.pred = self.target.copy()
        self.pred[:, :, 0:1] += 1.0
        self.pred[:, :, 1:3] += 2.0
        self.baseline = self.target.copy()
        self.baseline[:, :, 0:1] += 2.0
        self.baseline[:, :, 1:3] += 4.0
        self.climatology = rng.normal(size=self.target.shape)
        self.slices = {
            'TEMP': slice(0, 1),
            'SALT': slice(1, 3),
        }

    def test_multivariable_physical_metrics_do_not_mix_units(self):
        report = compute_metric_report(
            self.pred,
            self.target,
            ['TEMP', 'SALT'],
            channel_slices=self.slices,
            metric_space='physical',
        )

        self.assertIsNone(report['overall'])
        self.assertEqual(report['by_lead'], {})
        self.assertAlmostEqual(report['by_variable']['TEMP']['rmse'], 1.0)
        self.assertAlmostEqual(report['by_variable']['SALT']['rmse'], 2.0)
        self.assertIn('TEMP', report['by_variable_and_lead'])
        self.assertIn('lead_1', report['by_variable_and_lead']['SALT'])

    def test_multivariable_metrics_reject_inferred_or_invalid_channel_partitions(self):
        with self.assertRaisesRegex(ValueError, '缺少显式 channel slice'):
            resolve_variable_slices(['TEMP', 'SALT'], {}, 3)
        with self.assertRaisesRegex(ValueError, '无重叠且完整覆盖'):
            resolve_variable_slices(
                ['TEMP', 'SALT'],
                {'TEMP': [0, 2], 'SALT': [1, 3]},
                3,
            )
        with self.assertRaisesRegex(ValueError, '无重叠且完整覆盖'):
            resolve_variable_slices(
                ['TEMP', 'SALT'],
                {'TEMP': [0, 1], 'SALT': [2, 3]},
                3,
            )
        self.assertEqual(
            resolve_variable_slices(['TEMP'], {}, 3),
            {'TEMP': slice(0, 3)},
        )

    def test_normalized_metrics_allow_unit_compatible_overall(self):
        report = compute_metric_report(
            self.pred,
            self.target,
            ['TEMP', 'SALT'],
            channel_slices=self.slices,
            metric_space='normalized',
        )

        self.assertIsNotNone(report['overall'])
        self.assertTrue(report['unit_aggregation_allowed'])
        self.assertIn('lead_1', report['by_lead'])
        self.assertIn('overall', report['macro_field'])

    def test_macro_omits_redundant_field_mse_and_mae(self):
        report = compute_metric_report(
            self.pred,
            self.target,
            ['TEMP', 'SALT'],
            channel_slices=self.slices,
            metric_space='physical',
        )

        for variable_metrics in report['macro_field']['by_variable'].values():
            self.assertNotIn('mse', variable_metrics)
            self.assertNotIn('mae', variable_metrics)
            self.assertIn('rmse', variable_metrics)
            self.assertIn('r2', variable_metrics)

    def test_climatology_residual_reports_structure_only(self):
        report = compute_metric_report(
            self.pred,
            self.target,
            ['TEMP', 'SALT'],
            channel_slices=self.slices,
            baselines={'climatology': self.climatology},
            metric_space='physical',
        )

        temp_metrics = report['climatology_residual']['by_variable']['TEMP']
        self.assertEqual(set(temp_metrics), {'correlation', 'r2'})
        self.assertNotIn('rmse', temp_metrics)
        self.assertIn('does not change MAE/MSE/RMSE', report['climatology_residual']['note'])

    def test_baseline_skill_is_per_variable_and_per_lead(self):
        report = compute_metric_report(
            self.pred,
            self.target,
            ['TEMP', 'SALT'],
            channel_slices=self.slices,
            baselines={'persistence': self.baseline},
            metric_space='physical',
        )
        comparison = report['comparison']['persistence']

        self.assertIsNone(comparison['overall'])
        self.assertAlmostEqual(comparison['by_variable']['TEMP']['mse_skill'], 0.75)
        self.assertAlmostEqual(comparison['by_variable']['SALT']['mse_skill'], 0.75)
        self.assertAlmostEqual(
            comparison['by_variable_and_lead']['TEMP']['lead_1']['mse_skill'],
            0.75,
        )
        self.assertAlmostEqual(comparison['macro']['mse_skill']['mean'], 0.75)

    def test_spatial_mean_removed_is_explicitly_structure_only(self):
        report = compute_metric_report(
            self.pred,
            self.target,
            ['TEMP', 'SALT'],
            channel_slices=self.slices,
            metric_space='physical',
        )
        spatial = report['spatial_mean_removed']['by_variable']

        self.assertAlmostEqual(spatial['TEMP']['r2'], 1.0)
        self.assertAlmostEqual(spatial['SALT']['r2'], 1.0)
        self.assertEqual(set(spatial['TEMP']), {'correlation', 'r2'})

    def test_vectorized_field_metrics_match_scalar_formulas(self):
        rng = np.random.default_rng(19)
        target = rng.normal(size=(3, 2, 2, 4, 5))
        pred = 0.7 * target + rng.normal(scale=0.3, size=target.shape)
        actual = _field_metric_values(pred, target, chunk_size=3)

        expected = {'rmse': [], 'correlation': [], 'r2': []}
        for sample_idx in range(pred.shape[0]):
            for lead_idx in range(pred.shape[1]):
                for channel_idx in range(pred.shape[2]):
                    p = pred[sample_idx, lead_idx, channel_idx].reshape(-1)
                    t = target[sample_idx, lead_idx, channel_idx].reshape(-1)
                    diff = p - t
                    expected['rmse'].append(np.sqrt(np.mean(diff ** 2)))
                    expected['correlation'].append(np.corrcoef(p, t)[0, 1])
                    expected['r2'].append(1.0 - np.sum(diff ** 2) / np.sum((t - np.mean(t)) ** 2))

        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)

    def test_chunked_basic_metrics_match_numpy_definitions(self):
        rng = np.random.default_rng(23)
        target = rng.normal(loc=15.0, scale=3.0, size=(7, 3, 4, 5, 6))
        pred = 0.85 * target + rng.normal(scale=0.4, size=target.shape)
        actual = _basic_metrics(pred, target)
        diff = pred - target
        expected = {
            'mse': np.mean(diff ** 2),
            'mae': np.mean(np.abs(diff)),
            'rmse': np.sqrt(np.mean(diff ** 2)),
            'correlation': np.corrcoef(pred.reshape(-1), target.reshape(-1))[0, 1],
            'r2': 1.0 - np.sum(diff ** 2) / np.sum((target - np.mean(target)) ** 2),
        }
        for key, value in expected.items():
            self.assertAlmostEqual(actual[key], value, places=12)

    def test_metrics_ignore_only_nonfinite_pairs(self):
        target = np.array([[[[[1.0, 2.0, np.nan, 4.0]]]]])
        pred = np.array([[[[[1.5, np.inf, 3.0, 3.5]]]]])

        metrics = _basic_metrics(pred, target)

        # Only positions 0 and 3 are finite in both arrays.
        self.assertAlmostEqual(metrics['mae'], 0.5)
        self.assertAlmostEqual(metrics['rmse'], 0.5)

    def test_depth_metrics_preserve_physical_levels_and_skill(self):
        report = compute_metric_report(
            self.pred,
            self.target,
            ['TEMP', 'SALT'],
            channel_slices=self.slices,
            baselines={'persistence': self.baseline},
            metric_space='physical',
            depth_values=[5.0, 50.0],
        )

        # TEMP has one channel while two levels were supplied, so it is not
        # mislabeled. SALT has an exact two-channel/depth correspondence.
        self.assertNotIn('TEMP', report['by_variable_and_depth'])
        salt_depths = report['by_variable_and_depth']['SALT']
        self.assertEqual(salt_depths['depth_1']['depth'], 5.0)
        self.assertEqual(salt_depths['depth_2']['depth'], 50.0)
        self.assertAlmostEqual(salt_depths['depth_1']['rmse'], 2.0)
        skill = report['comparison']['persistence']['by_variable_and_depth']['SALT']
        self.assertAlmostEqual(skill['depth_2']['mse_skill'], 0.75)

    def test_sample_groups_retain_paired_origin_metrics(self):
        groups = compute_sample_group_report(
            self.pred,
            self.target,
            [10, 20],
            ['TEMP', 'SALT'],
            channel_slices=self.slices,
            baselines={'persistence': self.baseline},
            metric_space='physical',
        )

        self.assertEqual(groups['group_count'], 2)
        self.assertEqual(groups['groups']['10']['sample_count'], 1)
        temp = groups['groups']['20']['metrics']['by_variable']['TEMP']
        self.assertAlmostEqual(temp['rmse'], 1.0)
        self.assertNotIn('macro_field', groups['groups']['20']['metrics'])

    def test_period_groups_use_sample_lead_labels(self):
        period_ids = np.asarray([[0, 1], [1, 0]])
        groups = compute_period_group_report(
            self.pred,
            self.target,
            period_ids,
            ['TEMP', 'SALT'],
            channel_slices=self.slices,
            metric_space='physical',
        )

        self.assertEqual(groups['group_count'], 2)
        self.assertEqual(groups['groups']['0']['sample_count'], 2)
        self.assertAlmostEqual(
            groups['groups']['1']['metrics']['by_variable']['SALT']['rmse'],
            2.0,
        )


if __name__ == '__main__':
    unittest.main()
