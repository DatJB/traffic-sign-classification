import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from demo_service import demo_catalog, demo_image_bytes, decode_image

ROOT = Path(__file__).resolve().parents[1]


class CatalogTests(unittest.TestCase):
    def test_fixed_catalog_is_in_its_split_and_outside_train(self):
        train = set(pd.read_csv(ROOT / 'splits/train.csv').path)
        for split in ['validation', 'test']:
            catalog = demo_catalog(split, ROOT)
            pd.testing.assert_frame_equal(catalog, demo_catalog(split, ROOT))
            self.assertEqual(len(catalog), 86)
            self.assertEqual(catalog.class_id.nunique(), 43)
            self.assertTrue((catalog.groupby('class_id').size() == 2).all())
            self.assertFalse(set(catalog.path) & train)
            self.assertTrue(set(catalog.path) <= set(pd.read_csv(ROOT / f'splits/{split}.csv').path))
            path = catalog.iloc[0].path
            image = decode_image(demo_image_bytes(path, split, ROOT), Path(path).name)
            self.assertEqual(image.bgr.shape[2], 3)

    def test_missing_resources_and_path_escape(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                demo_catalog('test', root)
            with self.assertRaises(FileNotFoundError):
                demo_image_bytes('missing.ppm', 'test', root)
            with self.assertRaises(ValueError):
                demo_image_bytes('../../../outside.ppm', 'test', root)

    def test_zip_fallback_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            archive_dir = Path(root) / 'data/raw/archives'
            archive_dir.mkdir(parents=True)
            payload = b'P6\n1 1\n255\n\xff\x00\x00'
            relative = 'GTSRB/Final_Test/Images/00000.ppm'
            with ZipFile(archive_dir / 'GTSRB_Final_Test_Images.zip', 'w') as archive:
                archive.writestr(relative, payload)
            self.assertEqual(demo_image_bytes(relative, 'test', root), payload)


if __name__ == '__main__':
    unittest.main()
