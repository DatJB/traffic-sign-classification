import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from features_hog import hog_features, create_hog


class HOGTests(unittest.TestCase):
    def test_dimensions_and_normalization(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, 30:34] = 255
        for cell, expected in [(8, 1764), (16, 324)]:
            vector = hog_features(image, cell)
            self.assertEqual(len(vector), expected)
            self.assertTrue(np.isfinite(vector).all())
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_blank_image_has_finite_zero_descriptor(self):
        vector = hog_features(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(np.count_nonzero(vector), 0)

    def test_edge_direction_changes_descriptor(self):
        vertical = np.zeros((64, 64, 3), dtype=np.uint8)
        vertical[:, 32:] = 255
        horizontal = vertical.transpose(1, 0, 2).copy()
        self.assertFalse(np.allclose(hog_features(vertical), hog_features(horizontal)))

    def test_invalid_geometry_fails_clearly(self):
        with self.assertRaises(ValueError):
            create_hog(7)


if __name__ == '__main__':
    unittest.main()
