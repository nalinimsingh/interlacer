import unittest
import numpy as np
from interlacer import utils


class TestUtils(unittest.TestCase):
    """Tests for FFT and complex data utilities.

       Usage:
         $python -m unittest discover tests

    """

    def test_split_reim(self):
        compl_array = np.full((1, 3, 3), 1 + 1j)
        real_array = np.full((1, 3, 3), 1.0)
        imag_array = np.full((1, 3, 3), 1.0)
        split_array = np.stack([real_array, imag_array], 3)

        self.assertIsNone(
            np.testing.assert_array_equal(
                utils.split_reim(compl_array),
                split_array))

    def test_join_reim(self):
        compl_array = np.full((1, 3, 3), 1 + 1j)
        real_array = np.full((1, 3, 3), 1.0)
        imag_array = np.full((1, 3, 3), 1.0)
        split_array = np.stack([real_array, imag_array], 3)

        self.assertIsNone(
            np.testing.assert_array_equal(
                utils.join_reim(split_array),
                compl_array))

    def test_convert_to_frequency_domain(self):
        real_array = np.full((1, 3, 3), 1.0)
        imag_array = np.full((1, 3, 3), 1.0)
        split_array = np.stack([real_array, imag_array], 3)

        ffted_array = utils.convert_to_frequency_domain(split_array)
        check_array = utils.convert_to_image_domain(ffted_array)

        self.assertIsNone(
            np.testing.assert_array_equal(
                split_array, check_array))

    def test_convert_to_image_domain(self):
        real_array = np.full((1, 3, 3), 1.0)
        imag_array = np.full((1, 3, 3), 1.0)
        split_array = np.stack([real_array, imag_array], 3)

        iffted_array = utils.convert_to_image_domain(split_array)
        check_array = utils.convert_to_frequency_domain(iffted_array)

        self.assertIsNone(
            np.testing.assert_array_equal(
                split_array, check_array))


if __name__ == '__main__':
    unittest.main()
