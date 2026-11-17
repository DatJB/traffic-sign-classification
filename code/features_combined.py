"""Ghép màu HSV và hình dạng HOG, không học thống kê từ validation/test."""
from pathlib import Path
import numpy as np
from dataset_images import image_reader
from features_hsv import hsv_histogram
from features_hog import hog_features


def normalize_l2(features):
    """Dùng được cho một vector hoặc ma trận: mỗi hàng có chuẩn L2 bằng 1."""
    features = np.asarray(features, dtype=np.float32)
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norm, 1e-12)


def combine_features(hsv, hog, hsv_weight=0.25):
    """x = [alpha * HSV_L2, (1-alpha) * HOG_L2]."""
    if not 0 < hsv_weight < 1:
        raise ValueError('Trọng số HSV phải nằm giữa 0 và 1')
    if np.shape(hsv)[:-1] != np.shape(hog)[:-1]:
        raise ValueError('Hai nhóm phải khớp số ảnh và thứ tự ảnh')
    color = hsv_weight * normalize_l2(hsv)
    shape = (1 - hsv_weight) * normalize_l2(hog)
    return np.concatenate([color, shape], axis=-1)


def extract_groups(table, data_root, bins, cell_size, size=(64, 64)):
    """Đọc mỗi ảnh một lần; cả hai nhóm đặc trưng lấy từ cùng ảnh BGR."""
    color, shape = [], []
    with image_reader(data_root, 'train') as read_image:
        for i, path in enumerate(table['path'], start=1):
            image = read_image(path)
            color.append(hsv_histogram(image, bins, size))
            shape.append(hog_features(image, cell_size, size))
            if i % 10000 == 0:
                print(f'  Đã trích HSV + HOG: {i}/{len(table)} ảnh', flush=True)
    return np.asarray(color, dtype=np.float32), np.asarray(shape, dtype=np.float32)


def combined_features(image, bins, cell_size, hsv_weight=0.25, size=(64, 64)):
    hsv = hsv_histogram(image, bins, size)
    hog = hog_features(image, cell_size, size)
    return combine_features(hsv, hog, hsv_weight)
