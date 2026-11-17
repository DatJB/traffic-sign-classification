import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from features_hsv import hsv_histogram, predict_hsv


class HSVTests(unittest.TestCase):
    def test_white_max_values_are_in_histogram(self):
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        hist = hsv_histogram(white)
        self.assertEqual(hist.shape, (128,))
        self.assertAlmostEqual(float(hist.sum()), 1.0)
        self.assertEqual(hist.reshape(8, 4, 4)[0, 0, 3], 1.0)

    def test_same_colors_different_positions_same_histogram(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:32] = [0, 0, 255]
        image[32:] = [255, 0, 0]
        np.testing.assert_array_equal(hsv_histogram(image), hsv_histogram(image[::-1].copy()))

    def test_bundle_predicts_using_saved_bins(self):
        red = np.full((64, 64, 3), [0, 0, 255], dtype=np.uint8)
        blue = np.full((64, 64, 3), [255, 0, 0], dtype=np.uint8)
        bins = (16, 4, 4)
        model = KNeighborsClassifier(n_neighbors=1).fit(
            [hsv_histogram(red, bins), hsv_histogram(blue, bins)], [0, 1])
        bundle = {'bins': bins, 'size': (64, 64), 'normalization': 'l1_per_image', 'classifier': model}
        self.assertEqual(predict_hsv(bundle, red), 0)
        self.assertEqual(predict_hsv(bundle, blue), 1)


if __name__ == '__main__':
    unittest.main()
