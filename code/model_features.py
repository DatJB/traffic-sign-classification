"""Dùng cùng hàm trích đặc trưng khi huấn luyện và đánh giá."""
from pathlib import Path
import numpy as np
from time import perf_counter
from features_hsv import hsv_histogram
from features_hog import hog_features
from features_combined import combined_features
from features_combined import combine_features
from preprocessing import read_bgr
from dataset_images import image_reader


def image_features(bundle, image):
    method = bundle['method']
    if method == 'HSV + kNN':
        return hsv_histogram(image, bundle['bins'], bundle['size'])
    if method == 'HOG + kNN':
        return hog_features(image, bundle['cell_size'], bundle['size'])
    if method == 'HSV + HOG + kNN':
        return combined_features(image, bundle['bins'], bundle['cell_size'],
                                 bundle['hsv_weight'], bundle['size'])
    raise ValueError(f'Phương pháp không hỗ trợ: {method}')


def extract_model_features(table, root, bundle):
    features = []
    for i, path in enumerate(table['path'], start=1):
        features.append(image_features(bundle, read_bgr(Path(root) / path)))
        if i % 5000 == 0:
            print(f'  Đã trích {i}/{len(table)} ảnh', flush=True)
    return np.asarray(features, dtype=np.float32)


def extract_shared_features(table, root, bundles):
    """Ba mô hình dùng cùng bins/cell: đọc ảnh một lần, tính từng công đoạn một lần."""
    hsv_model, hog_model, combined = [bundles[name] for name in ['hsv', 'hog', 'combined']]
    assert hsv_model['bins'] == combined['bins']
    assert hog_model['cell_size'] == combined['cell_size']
    assert hsv_model['size'] == hog_model['size'] == combined['size']
    values = {name: [] for name in bundles}
    times = {'read': 0., 'hsv': 0., 'hog': 0., 'combine': 0.}
    with image_reader(root, 'test') as read_image:
        for i, path in enumerate(table.path, start=1):
            start = perf_counter()
            image = read_image(path)
            times['read'] += perf_counter() - start
            start = perf_counter()
            hsv = hsv_histogram(image, hsv_model['bins'], hsv_model['size'])
            times['hsv'] += perf_counter() - start
            start = perf_counter()
            hog = hog_features(image, hog_model['cell_size'], hog_model['size'])
            times['hog'] += perf_counter() - start
            start = perf_counter()
            fused = combine_features(hsv, hog, combined['hsv_weight'])
            times['combine'] += perf_counter() - start
            for name, vector in [('hsv', hsv), ('hog', hog), ('combined', fused)]:
                values[name].append(vector)
            if i % 5000 == 0:
                print(f'  Đã trích đặc trưng chung: {i}/{len(table)} ảnh', flush=True)
    feature_seconds = {
        'hsv': times['read'] + times['hsv'],
        'hog': times['read'] + times['hog'],
        'combined': sum(times.values()),
    }
    # Bao gồm cả chi phí tạo ma trận đầu vào cho classifier.
    for name in values:
        start = perf_counter()
        values[name] = np.asarray(values[name], dtype=np.float32)
        feature_seconds[name] += perf_counter() - start
    return values, feature_seconds, times
