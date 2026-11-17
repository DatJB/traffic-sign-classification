"""Đặc trưng HOG dùng chung cho người 3 và người 4."""
from pathlib import Path
import cv2
import numpy as np
from preprocessing import read_bgr, preprocess


def create_hog(cell_size=8, size=(64, 64)):
    """Block gồm 2x2 cell, stride một cell, 9 hướng không dấu."""
    if cell_size not in (8, 16) or tuple(size) != (64, 64):
        raise ValueError('Cấu hình hỗ trợ: ảnh 64x64, cell 8 hoặc 16 pixel')
    return cv2.HOGDescriptor(
        _winSize=tuple(size),
        _blockSize=(2 * cell_size, 2 * cell_size),
        _blockStride=(cell_size, cell_size),
        _cellSize=(cell_size, cell_size),
        _nbins=9,
        _histogramNormType=cv2.HOGDescriptor_L2Hys,
        _L2HysThreshold=0.2,
        _gammaCorrection=False,
        _signedGradient=False,
    )


def hog_features(image, cell_size=8, size=(64, 64)):
    """Ảnh BGR -> gray -> HOG (L2-Hys từng block) -> L2 toàn vector."""
    gray = preprocess(image, size=size, color='gray')
    vector = create_hog(cell_size, size).compute(gray).flatten()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def extract_features(table, data_root, cell_size, size=(64, 64)):
    features = []
    for i, path in enumerate(table['path'], start=1):
        image = read_bgr(Path(data_root) / path)
        features.append(hog_features(image, cell_size, size))
        if i % 10000 == 0:
            print(f'  Đã trích HOG: {i}/{len(table)} ảnh', flush=True)
    return np.asarray(features, dtype=np.float32)


def predict_hog(bundle, image):
    """Tái sử dụng chính xác cấu hình đặc trưng đã lưu cùng kNN."""
    if bundle['normalization'] != 'block_l2hys_then_global_l2':
        raise ValueError('Không hỗ trợ cấu hình chuẩn hóa này')
    vector = hog_features(image, bundle['cell_size'], bundle['size'])
    return int(bundle['classifier'].predict(vector.reshape(1, -1))[0])
