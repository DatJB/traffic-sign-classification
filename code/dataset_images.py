"""Đọc nhanh nhiều PPM từ ZIP gốc; vẫn hỗ trợ thư mục đã giải nén."""
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np
from preprocessing import read_bgr


@contextmanager
def image_reader(root, split='train'):
    root = Path(root)
    name = 'GTSRB_Final_Test_Images.zip' if split == 'test' else 'GTSRB_Final_Training_Images.zip'
    path = root / 'archives' / name
    if path.exists():
        with ZipFile(path) as archive:
            def read_image(relative_path):
                payload = archive.read(relative_path)
                image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Không đọc được ảnh trong ZIP: {relative_path}')
                return image
            yield read_image
    else:
        yield lambda relative_path: read_bgr(root / relative_path)
