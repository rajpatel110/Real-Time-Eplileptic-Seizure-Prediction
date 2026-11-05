import unittest
import numpy as np
from eeg_project.transforms import SignedLogFeatures


class SignedLogTests(unittest.TestCase):
    def test_training_scales_and_signed_values(self):
        transform = SignedLogFeatures().fit([[1., -2., 0.], [3., -6., 0.]])
        np.testing.assert_allclose(transform.scale_, [2., 4., 1.])
        values = np.array([[2., -4., 0.], [-6., 12., 5.]])
        expected = np.sign(values) * np.log1p(np.abs(values) / [2., 4., 1.])
        np.testing.assert_allclose(transform.transform(values), expected)
        original = transform.scale_.copy()
        transform.transform([[1e100, -1e100, 0.]])
        np.testing.assert_array_equal(transform.scale_, original)
        with self.assertRaises(ValueError):
            transform.transform([[1., 2.]])


if __name__ == '__main__':
    unittest.main()
