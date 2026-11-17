"""Đặc trưng màu dùng chung cho người 2 và người 4."""
import cv2
import numpy as np
from preprocessing import read_bgr, preprocess


def hsv_histogram(image, bins=(8, 4, 4), size=(64, 64)):
    """Histogram HSV 3 chiều, trải phẳng và chuẩn hóa tổng bằng 1."""
    if len(bins) != 3 or any(int(b) != b or b < 1 for b in bins):
        raise ValueError('bins cần có 3 số nguyên dương: H, S, V')
    hsv = preprocess(image, size=size, color='hsv')
    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], None, list(map(int, bins)),
        [0, 180, 0, 256, 0, 256],
    )
    vector = histogram.flatten()
    return vector / vector.sum()  # Chuẩn hóa L1 riêng từng ảnh, không cần fit scaler.


def extract_features(table, data_root, bins, size=(64, 64)):
    """Mỗi dòng CSV -> một vector; giữ đúng thứ tự dòng để khớp nhãn."""
    from pathlib import Path
    features = []
    for i, path in enumerate(table['path'], start=1):
        image = read_bgr(Path(data_root) / path)
        features.append(hsv_histogram(image, bins, size))
        if i % 10000 == 0:
            print(f'  Đã trích HSV: {i}/{len(table)} ảnh', flush=True)
    return np.asarray(features, dtype=np.float32)


def predict_hsv(bundle, image):
    """Dùng đúng cấu hình đã lưu; đầu vào ảnh BGR uint8, đầu ra mã lớp."""
    if bundle['normalization'] != 'l1_per_image':
        raise ValueError('Không hỗ trợ cấu hình chuẩn hóa này')
    vector = hsv_histogram(image, bundle['bins'], bundle['size'])
    return int(bundle['classifier'].predict(vector.reshape(1, -1))[0])
