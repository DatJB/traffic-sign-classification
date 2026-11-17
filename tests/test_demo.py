import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from demo_service import decode_image, load_model, predict, read_test_results, validate_bundle, label_name
from model_features import image_features

ROOT = Path(__file__).resolve().parents[1]


class ImageInputTests(unittest.TestCase):
    def test_gray_and_alpha_match_existing_opencv_decoder(self):
        for image in [np.full((32, 40), 90, np.uint8), np.full((24, 30, 4), [20, 50, 200, 0], np.uint8)]:
            _, encoded = cv2.imencode('.png', image)
            result = decode_image(encoded.tobytes(), 'image.png')
            expected = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(result.bgr, expected)
            self.assertEqual(result.bgr.shape[2], 3)
            self.assertTrue(result.notes)

    def test_corrupt_empty_and_disguised_images_are_rejected(self):
        for payload, name in [(b'', 'a.png'), (b'bad image', 'a.jpg'), (b'x', 'a.gif')]:
            with self.assertRaises(ValueError):
                decode_image(payload, name)
        _, bmp = cv2.imencode('.bmp', np.zeros((20, 20, 3), np.uint8))
        with self.assertRaises(ValueError):
            decode_image(bmp.tobytes(), 'disguised.png')

    def test_missing_resource_and_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                load_model('hsv', root)
            self.assertEqual(read_test_results(root), [])
        with self.assertRaises(ValueError):
            validate_bundle('hsv', {'method': 'HSV + kNN', 'size': [64, 64]})

    def test_missing_vietnamese_name_uses_original_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            pd.DataFrame([{'class_id': 1, 'name_vi': None, 'name_en': 'Original name'}]).to_csv(
                root / 'configs/class_names.csv', index=False)
            self.assertEqual(label_name(1, {}, root), 'Original name')
            self.assertEqual(label_name(8, {}, root), 'Lớp 8')


class RealModelTests(unittest.TestCase):
    def test_three_saved_models_use_exact_existing_feature_pipeline(self):
        sample = next((ROOT / 'samples').glob('*.ppm'))
        image = decode_image(sample.read_bytes(), sample.name)
        for method in ['hsv', 'hog', 'combined']:
            with self.subTest(method=method), threadpool_limits(limits=4):
                bundle = load_model(method, ROOT)
                result = predict(bundle, image.bgr, ROOT)
                vector = image_features(bundle, image.bgr)
                expected = int(bundle['classifier'].predict(vector.reshape(1, -1))[0])
                self.assertEqual(result['class_id'], expected)
                self.assertEqual(result['n_features'], bundle['classifier'].n_features_in_)
                self.assertGreaterEqual(result['milliseconds'], 0)
                self.assertNotIn('true_class', result)


if __name__ == '__main__':
    unittest.main()
