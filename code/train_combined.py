"""HSV + HOG + kNN: chọn trọng số/k/khoảng cách trên validation."""
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

from features_combined import extract_groups, combine_features
from evaluation import classification_metrics, per_class_report

# Cố định bins/cell từ kết quả người 2-3; thử 12 cấu hình kết hợp mới.
HSV_WEIGHTS = [0.25, 0.5]
K_OPTIONS = [3, 5, 7]
DISTANCES = ['euclidean', 'manhattan']


def run_training(project='.'):
    project = Path(project).resolve()
    output = project / 'results/combined'
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

    # Kế thừa cấu hình đặc trưng, nhưng không dùng lại kNN của hai phương pháp.
    source_summaries = {
        name: json.loads((project / f'results/{name}/summary.json').read_text(encoding='utf-8'))
        for name in ['hsv', 'hog']
    }
    split_hashes = {name: hashlib.sha256((project / f'splits/{name}.csv').read_bytes()).hexdigest()
                    for name in ['train', 'validation']}
    for source in source_summaries.values():
        assert source['split_sha256'] == split_hashes, 'Kết quả nguồn dùng split khác!'
    hsv_config = source_summaries['hsv']['best']
    bins = [hsv_config['h_bins'], hsv_config['s_bins'], hsv_config['v_bins']]
    cell_size = source_summaries['hog']['best']['cell_size']
    print(f'Giữ bins={bins}, cell={cell_size}; thử 12 cấu hình kết hợp.', flush=True)
    with threadpool_limits(limits=4):
        start = perf_counter()
        hsv_train, hog_train = extract_groups(train, project / 'data/raw', bins, cell_size)
        train_base_seconds = perf_counter() - start
        start = perf_counter()
        hsv_val, hog_val = extract_groups(validation, project / 'data/raw', bins, cell_size)
        validation_base_seconds = perf_counter() - start

    with threadpool_limits(limits=4):
        for hsv_weight in HSV_WEIGHTS:
            print(f'\nTrọng số HSV={hsv_weight}, HOG={1-hsv_weight}', flush=True)
            start = perf_counter()
            x_train = combine_features(hsv_train, hog_train, hsv_weight)
            train_feature_seconds = train_base_seconds + perf_counter() - start
            start = perf_counter()
            x_val = combine_features(hsv_val, hog_val, hsv_weight)
            validation_feature_seconds = validation_base_seconds + perf_counter() - start

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
                        'hsv_weight': hsv_weight, 'hog_weight': 1-hsv_weight,
                        'h_bins': bins[0], 's_bins': bins[1], 'v_bins': bins[2], 'cell_size': cell_size,
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
                    print(f'  alpha={hsv_weight}, k={k}, {distance}: Accuracy={metrics["accuracy"]:.4f}, '
                          f'Macro-F1={metrics["macro_f1"]:.4f}', flush=True)
                    # Ưu tiên Macro-F1; hòa thì Accuracy, số chiều ít hơn, k nhỏ hơn.
                    rank = (metrics['macro_f1'], metrics['accuracy'], -x_train.shape[1], -k)
                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        best_model, best_result = model, result.copy()
                        best_features, best_prediction = x_val, prediction.copy()

    hsv_weight = best_result['hsv_weight']
    names = pd.read_csv(project / 'configs/class_names.csv')
    bundle = {
        'method': 'HSV + HOG + kNN', 'bins': bins, 'cell_size': cell_size,
        'hsv_weight': hsv_weight, 'size': [64, 64],
        'hog_parameters': {'block_cells': 2, 'stride': cell_size, 'orientations': 9,
                           'signed_gradient': False, 'gamma_correction': False, 'l2hys_threshold': 0.2},
        'normalization': 'l2_each_group_then_weighted_concat', 'classifier': best_model,
        'class_names': names.to_dict('records'),
        'training_scope': 'train only; hyperparameters chosen on validation',
    }
    model_path = model_dir / 'combined_knn.joblib'
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
        'method': 'HSV + HOG + kNN', 'split': 'validation',
        'train_images': len(train), 'validation_images': len(validation),
        'search_budget': 12, 'selection_metric': 'macro_f1',
        'tie_break': 'accuracy, fewer features, smaller k, first grid entry',
        'component_selection': 'bins and cell inherited from HSV/HOG validation selection; 12 additional fusion configurations',
        'component_best': {name: value['best'] for name, value in source_summaries.items()},
        'best': best_result,
        'split_sha256': {name: hashlib.sha256((project / f'splits/{name}.csv').read_bytes()).hexdigest()
                         for name in ['train', 'validation']},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'sklearn': sklearn.__version__, 'platform': platform.platform(), 'threads': 4},
        'timing_note': 'One full-validation batch per configuration; end-to-end includes disk read, resize, HSV/HOG, per-group L2, weighted concatenation and kNN; excludes manual work. Feature extraction reused across k/metric. Not a repeated latency benchmark.',
        'test_evaluated': False,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nĐã lưu mô hình và kết quả validation. Chưa đọc hoặc đánh giá test.', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='.')
    run_training(parser.parse_args().project)
