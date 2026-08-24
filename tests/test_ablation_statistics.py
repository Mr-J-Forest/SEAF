import unittest

import numpy as np

from scripts.compare_ablation_contrasts import (
    benjamini_hochberg,
    circular_block_indices,
    paired_scores,
    summarize_bootstrap,
)


class AblationStatisticsTests(unittest.TestCase):
    def test_benjamini_hochberg_is_monotone_in_sorted_p_values(self):
        p_values = [0.01, 0.04, 0.03, 0.20]
        q_values = benjamini_hochberg(p_values)
        ordered = sorted(zip(p_values, q_values))
        self.assertTrue(all(left[1] <= right[1] for left, right in zip(ordered, ordered[1:])))
        self.assertTrue(all(p <= q <= 1.0 for p, q in zip(p_values, q_values)))

    def test_circular_blocks_preserve_requested_length(self):
        indices = circular_block_indices(7, 5, np.random.default_rng(3))
        self.assertEqual(indices.shape, (7,))
        self.assertTrue(np.all((0 <= indices) & (indices < 7)))
        for left, right in zip(indices[:4], indices[1:5]):
            self.assertEqual(int(right), (int(left) + 1) % 7)

    def test_constant_paired_improvement_has_exact_bootstrap_effect(self):
        candidate = {
            str(origin): {'TEMP': 1.0, 'SALT': 2.0}
            for origin in range(20)
        }
        reference = {
            str(origin): {'TEMP': 2.0, 'SALT': 4.0}
            for origin in range(20)
        }
        origins, scores = paired_scores(
            candidate, reference, ['TEMP', 'SALT']
        )
        self.assertEqual(len(origins), 20)
        report = summarize_bootstrap(
            [scores],
            ['TEMP', 'SALT'],
            replicates=200,
            block_length=5,
            seed=7,
            meaningful_reduction_fraction=0.01,
        )
        macro = report['macro_equal_variable_weight']
        self.assertAlmostEqual(
            macro['geometric_mse_reduction_fraction'], 0.5, places=12
        )
        self.assertEqual(report['screening_decision'], 'advance')
        self.assertEqual(report['confirmation_status'], 'supported')

    def test_origin_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identical forecast origins'):
            paired_scores(
                {'1': {'TEMP': 1.0}},
                {'2': {'TEMP': 1.0}},
                ['TEMP'],
            )


if __name__ == '__main__':
    unittest.main()
