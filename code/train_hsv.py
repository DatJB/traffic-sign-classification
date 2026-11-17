"""HSV + kNN"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from features_hsv import extract_features
from evaluation import classification_metrics, per_class_report

# 2 cách chia bin x 3 giá trị k x 2 khoảng cách = 12 cấu hình.
# Nhóm cần giữ cùng ngân sách 12 cấu hình cho HOG và phương pháp kết hợp.
BIN_OPTIONS = [(8, 4, 4), (16, 4, 4)]
K_OPTIONS = [3, 5, 7]
DISTANCES = ['euclidean', 'manhattan']


def run_training(project='.'):
    project = Path(project).resolve()
    output = project / 'results/hsv'
    output.mkdir(parents=True, exist_ok=True)
    # Nếu lần chạy lại bị ngắt, không coi summary cũ là kết quả lần chạy mới.
    (output / 'summary.json').unlink(missing_ok=True)
    model_dir = project / 'models'
    model_dir.mkdir(exist_ok=True)
    train = pd.read_csv(project / 'splits/train.csv', keep_default_na=False)
    validation = pd.read_csv(project / 'splits/validation.csv', keep_default_na=False)
    assert not (set(train.track_id) & set(validation.track_id)), 'Trùng track!'
    assert set(train.class_id) == set(validation.class_id) == set(range(43))
    assert set(train.source) == set(validation.source) == {'official_train'}
    assert train.path.is_unique and validation.path.is_unique
    assert not (set(train.path) & set(validation.path))
    y_train, y_val = train.class_id.to_numpy(), validation.class_id.to_numpy()
    results = []
    best_rank = None
    best_model = None
    best_features = None
    best_prediction = None
    best_result = None

    # Giới hạn 4 thread để tránh tạo quá nhiều luồng khi tính khoảng cách.
    with threadpool_limits(limits=4):
        for bins in BIN_OPTIONS:
            print(f'\nTrích đặc trưng HSV, bins={bins}', flush=True)
            start = perf_counter()
            x_train = extract_features(train, project / 'data/raw', bins)
            train_feature_seconds = perf_counter() - start
            start = perf_counter()
            x_val = extract_features(validation, project / 'data/raw', bins)
            validation_feature_seconds = perf_counter() - start

            for k in K_OPTIONS:
                for distance in DISTANCES:
                    model = KNeighborsClassifier(n_neighbors=k, metric=distance,
                                                 weights='uniform', algorithm='brute', n_jobs=4)
                    start = perf_counter()
                    model.fit(x_train, y_train)  # Chỉ fit trên train.
                    fit_seconds = perf_counter() - start
                    start = perf_counter()
                    prediction = model.predict(x_val)
                    predict_seconds = perf_counter() - start
                    metrics = classification_metrics(y_val, prediction)
                    result = {
                        'h_bins': bins[0], 's_bins': bins[1], 'v_bins': bins[2],
                        'k': k, 'distance': distance, 'weights': 'uniform',
                        'n_features': x_train.shape[1], **metrics,
                        'train_feature_seconds': train_feature_seconds,
                        'validation_feature_seconds': validation_feature_seconds,
                        'fit_seconds': fit_seconds,
                        'predict_seconds': predict_seconds,
                        'classifier_ms_per_image': predict_seconds * 1000 / len(validation),
                        'end_to_end_ms_per_image': (validation_feature_seconds + predict_seconds) * 1000 / len(validation),
                    }
                    results.append(result)
                    pd.DataFrame(results).to_csv(output / 'validation_results.csv', index=False)
                    print(f'  k={k}, {distance}: Accuracy={metrics["accuracy"]:.4f}, '
                          f'Macro-F1={metrics["macro_f1"]:.4f}', flush=True)
                    # Ưu tiên Macro-F1; hòa thì Accuracy, số chiều ít hơn, k nhỏ hơn.
                    rank = (metrics['macro_f1'], metrics['accuracy'], -x_train.shape[1], -k)
                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        best_model, best_result = model, result.copy()
                        best_features, best_prediction = x_val, prediction.copy()

    bins = [best_result['h_bins'], best_result['s_bins'], best_result['v_bins']]
    names = pd.read_csv(project / 'configs/class_names.csv')
    bundle = {
        'method': 'HSV + kNN', 'bins': bins, 'size': [64, 64],
        'normalization': 'l1_per_image', 'classifier': best_model,
        'class_names': names.to_dict('records'),
        'training_scope': 'train only; hyperparameters chosen on validation',
    }
    model_path = model_dir / 'hsv_knn.joblib'
    joblib.dump(bundle, model_path)
    # Kiểm tra mô hình nạp lại cho cùng kết quả trên mẫu validation.
    loaded = joblib.load(model_path)
    with threadpool_limits(limits=4):
        np.testing.assert_array_equal(loaded['classifier'].predict(best_features[:100]), best_prediction[:100])
    prediction_table = validation[['path', 'class_id']].rename(columns={'class_id': 'true_class'})
    prediction_table['predicted_class'] = best_prediction
    prediction_table['correct'] = prediction_table.true_class == prediction_table.predicted_class
    prediction_table.to_csv(output / 'validation_predictions.csv', index=False)
    per_class_report(y_val, best_prediction).to_csv(output / 'validation_per_class.csv', index=False)
    summary = {
        'method': 'HSV + kNN', 'split': 'validation',
        'train_images': len(train), 'validation_images': len(validation),
        'search_budget': 12, 'selection_metric': 'macro_f1',
        'tie_break': 'accuracy, fewer features, smaller k, first distance in grid',
        'best': best_result,
        'split_sha256': {name: hashlib.sha256((project / f'splits/{name}.csv').read_bytes()).hexdigest()
                         for name in ['train', 'validation']},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'sklearn': sklearn.__version__, 'platform': platform.platform(), 'threads': 4},
        'timing_note': 'One full-validation batch per configuration; end-to-end includes disk read, resize, HSV, histogram L1 and kNN; excludes manual work. Feature extraction reused across k/metric. Not a repeated latency benchmark.',
        'test_evaluated': False,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nĐã lưu mô hình và kết quả validation. Chưa đọc hoặc đánh giá test.', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='.')
    run_training(parser.parse_args().project)
