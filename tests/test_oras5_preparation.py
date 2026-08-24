import unittest

import numpy as np

from scripts.prepare_oras5 import (
    VARIABLE_SPECS,
    apply_surface_mask,
    archive_url,
    depth_brackets,
    interpolate_masked_depths,
    parse_depths,
    select_specs,
)


class Oras5PreparationTests(unittest.TestCase):
    def test_depth_brackets_clip_surface_and_interpolate_interior(self):
        source = np.array([0.5, 10.0, 20.0], dtype=np.float64)
        target = np.array([0.0, 5.0, 20.0, 30.0], dtype=np.float64)

        lower, upper, weight = depth_brackets(source, target)

        np.testing.assert_array_equal(lower, [0, 0, 2, 2])
        np.testing.assert_array_equal(upper, [0, 1, 2, 2])
        np.testing.assert_allclose(weight, [0.0, 4.5 / 9.5, 0.0, 0.0])

    def test_vertical_interpolation_never_crosses_the_corrected_bottom_mask(self):
        values = np.array([
            [[0.0, 10.0]],
            [[10.0, 20.0]],
        ], dtype=np.float32)
        mask = np.array([
            [[True, True]],
            [[True, False]],
        ])

        result = interpolate_masked_depths(
            values,
            mask,
            np.array([0.0, 10.0]),
            np.array([0.0, 5.0, 10.0]),
        )

        np.testing.assert_allclose(result[:, 0, 0], [0.0, 5.0, 10.0])
        self.assertEqual(float(result[0, 0, 1]), 10.0)
        self.assertTrue(np.isnan(result[1:, 0, 1]).all())

    def test_surface_mask_turns_non_ocean_cells_into_nan(self):
        result = apply_surface_mask(
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[True, False]]),
        )
        self.assertEqual(float(result[0, 0]), 1.0)
        self.assertTrue(np.isnan(result[0, 1]))

    def test_manifest_uses_icdc_annual_control_member_archives(self):
        temp = select_specs(['TEMP'])[0]
        self.assertEqual(temp.source_name, 'votemper')
        self.assertEqual(temp.mask_file, 'tmask_r1x1.nc')
        self.assertTrue(
            archive_url(temp, 1979).endswith(
                '/votemper/opa0/votemper_ORAS5_1m_1979_r1x1.tar.gz'
            )
        )
        self.assertEqual(len(VARIABLE_SPECS), 10)

    def test_depth_parser_rejects_unsorted_or_duplicate_values(self):
        self.assertEqual(parse_depths('0,5,10'), (0.0, 5.0, 10.0))
        with self.assertRaises(Exception):
            parse_depths('0,10,5')
        with self.assertRaises(Exception):
            parse_depths('0,5,5')


if __name__ == '__main__':
    unittest.main()
