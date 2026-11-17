import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from model_features import extract_shared_features, image_features


class SharedFeatureTests(unittest.TestCase):
    def test_shared_path_matches_each_models_independent_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = []
            for i in range(2):
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                image[:, 20 + i * 10:] = [0, 0, 255]
                images.append(image)
                success, encoded = cv2.imencode('.ppm', image)
                self.assertTrue(success)
                (root / f'{i}.ppm').write_bytes(encoded.tobytes())
            bundles = {
                'hsv': {'method': 'HSV + kNN', 'bins': [16, 4, 4], 'size': [64, 64]},
                'hog': {'method': 'HOG + kNN', 'cell_size': 8, 'size': [64, 64]},
                'combined': {'method': 'HSV + HOG + kNN', 'bins': [16, 4, 4], 'cell_size': 8,
                             'hsv_weight': 0.25, 'size': [64, 64]},
            }
            values, times, stages = extract_shared_features(pd.DataFrame({'path': ['1.ppm', '0.ppm']}), root, bundles)
            for method in bundles:
                for row, image in enumerate(reversed(images)):
                    np.testing.assert_allclose(values[method][row], image_features(bundles[method], image), atol=1e-7)
                self.assertGreaterEqual(times[method], 0)


if __name__ == '__main__':
    unittest.main()
