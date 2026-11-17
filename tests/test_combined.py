import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from features_combined import combine_features, combined_features, normalize_l2


class CombinedTests(unittest.TestCase):
    def test_group_scale_does_not_change_result(self):
        hsv, hog = np.array([2., 1.]), np.array([1., 3., 2.])
        np.testing.assert_allclose(combine_features(hsv, hog),
                                   combine_features(10 * hsv, 0.5 * hog), atol=1e-7)

    def test_weights_and_dimensions(self):
        result = combine_features(np.ones((3, 2)), np.ones((3, 4)), 0.25)
        self.assertEqual(result.shape, (3, 6))
        np.testing.assert_allclose(np.linalg.norm(result[:, :2], axis=1), 0.25)
        np.testing.assert_allclose(np.linalg.norm(result[:, 2:], axis=1), 0.75)

    def test_blank_group_stays_finite(self):
        self.assertTrue(np.isfinite(normalize_l2(np.zeros((2, 3)))).all())
        with self.assertRaises(ValueError):
            combine_features(np.ones(2), np.ones(3), 0)

    def test_image_feature_dimension(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, 32:] = 255
        self.assertEqual(combined_features(image, (16, 4, 4), 8).shape, (2020,))


if __name__ == '__main__':
    unittest.main()
