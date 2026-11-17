"""Tests for leakage prevention and color/resize contracts (synthetic data)."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import numpy as np
import pandas as pd
from prepare_data import split_by_track
from prepare_data import prepare
from preprocessing import preprocess, read_bgr


class DataPipelineTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([
            {'path': f'{label}/{track}_{frame}.ppm', 'class_id': label,
             'track_id': f'{label:05d}:{track:05d}', 'source': 'official_train'}
            for label in range(3) for track in range(5) for frame in range(3)
        ] + [{'path': f'test/{label}.ppm', 'class_id': label, 'track_id': '',
              'source': 'official_test'} for label in range(3)])

    def test_group_isolation_coverage_and_determinism(self):
        data = self.fixture()
        a, b = split_by_track(data), split_by_track(data)
        for key in a:
            pd.testing.assert_frame_equal(a[key], b[key])
        self.assertFalse(set(a['train'].track_id) & set(a['validation'].track_id))
        self.assertEqual(sum(map(len, a.values())), len(data))
        self.assertEqual(set(a['test'].path), set(data[data.source == 'official_test'].path))
        for key in ('train', 'validation'):
            self.assertEqual(set(a[key].class_id), {0, 1, 2})

    def test_missing_track_and_invalid_fraction_fail(self):
        data = self.fixture()
        data.loc[0, 'track_id'] = ''
        with self.assertRaises(ValueError):
            split_by_track(data)
        with self.assertRaises(ValueError):
            split_by_track(self.fixture(), 1)

    def test_bgr_red_color_and_shape(self):
        red = np.zeros((20, 30, 3), dtype=np.uint8)
        red[:, :, 2] = 255
        hsv = preprocess(red, color='hsv')
        self.assertEqual(hsv.shape, (64, 64, 3))
        np.testing.assert_array_equal(hsv[0, 0], [0, 255, 255])
        np.testing.assert_array_equal(preprocess(red, color='rgb')[0, 0], [255, 0, 0])
        self.assertEqual(preprocess(red, color='gray').shape, (64, 64))
        with self.assertRaises(ValueError):
            preprocess(red, size=(0, 64))

    def test_corrupt_image_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.ppm'
            path.write_bytes(b'invalid image')
            with self.assertRaises(ValueError):
                read_bgr(path)
            path.write_bytes(b'')
            with self.assertRaises(ValueError):
                read_bgr(path)

    def test_fixed_manifest_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'splits/train.csv'
            manifest.parent.mkdir()
            manifest.write_text('do not change', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                prepare('missing_data', folder, 0.2, 42)
            self.assertEqual(manifest.read_text(encoding='utf-8'), 'do not change')


if __name__ == '__main__':
    unittest.main()
