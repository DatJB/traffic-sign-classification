"""Cấu hình đã khóa không được bị thay thế bởi mô hình khác."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from evaluate_test import freeze_experiment, sha256_file


class FreezeTests(unittest.TestCase):
    def test_changed_model_is_rejected_and_lock_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ['configs', 'splits', 'models']:
                (root / folder).mkdir()
            for name in ['train', 'validation', 'test']:
                (root / f'splits/{name}.csv').write_text('synthetic manifest', encoding='utf-8')
            hashes = {name: sha256_file(root / f'splits/{name}.csv') for name in ['train', 'validation']}
            classifier = KNeighborsClassifier(n_neighbors=1, metric='euclidean').fit(np.eye(2), [0, 1])
            for method in ['hsv', 'hog', 'combined']:
                folder = root / f'results/{method}'
                folder.mkdir(parents=True)
                best = {'k': 1, 'distance': 'euclidean', 'n_features': 2, 'hsv_weight': 0.25}
                summary = {'split': 'validation', 'search_budget': 12, 'split_sha256': hashes,
                           'best': best, 'train_images': 2}
                (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
                joblib.dump({'classifier': classifier, 'hsv_weight': 0.25}, root / f'models/{method}_knn.joblib')
            freeze_experiment(root)
            locked_path = root / 'configs/final_experiment.json'
            original = locked_path.read_bytes()
            # Một mô hình mới dù cùng tham số vẫn phải bị phát hiện bằng hash.
            joblib.dump({'classifier': classifier, 'hsv_weight': 0.25, 'changed': True},
                        root / 'models/combined_knn.joblib')
            with self.assertRaises(AssertionError):
                freeze_experiment(root)
            self.assertEqual(locked_path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
