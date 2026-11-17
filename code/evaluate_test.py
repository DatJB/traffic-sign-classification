"""Khóa cấu hình trước, rồi đánh giá ba mô hình đã lưu trên test chính thức."""
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
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from evaluation import classification_metrics, per_class_report
from model_features import extract_shared_features

METHODS = ['hsv', 'hog', 'combined']


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze_experiment(project):
    """Chưa đọc nhãn test: chỉ ghi hash manifest, mô hình và tham số từ validation."""
    path = project / 'configs/final_experiment.json'
    config = {
        'selection_split': 'validation', 'evaluation_split': 'official_test',
        'training_scope': '31380 train only, no refit on validation',
        'split_sha256': {name: sha256_file(project / f'splits/{name}.csv')
                         for name in ['train', 'validation', 'test']},
        'models': {},
        'image_archive_sha256': {},
    }
    for name in ['GTSRB_Final_Training_Images.zip', 'GTSRB_Final_Test_Images.zip']:
        archive = project / 'data/raw/archives' / name
        if archive.exists():
            config['image_archive_sha256'][name] = sha256_file(archive)
    for method in METHODS:
        summary = json.loads((project / f'results/{method}/summary.json').read_text(encoding='utf-8'))
        assert summary['split'] == 'validation' and summary['search_budget'] == 12
        assert summary['split_sha256'] == {k: config['split_sha256'][k] for k in ['train', 'validation']}
        bundle = joblib.load(project / f'models/{method}_knn.joblib')
        classifier = bundle['classifier']
        assert classifier.n_neighbors == summary['best']['k']
        assert classifier.metric == summary['best']['distance']
        assert classifier._fit_X.shape == (summary['train_images'], summary['best']['n_features'])
        if method == 'combined':
            assert bundle['hsv_weight'] == summary['best']['hsv_weight']
        del bundle, classifier
        config['models'][method] = {
            'path': f'models/{method}_knn.joblib',
            'sha256': sha256_file(project / f'models/{method}_knn.joblib'),
            'selected_parameters': summary['best'],
        }
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == config, 'Cấu hình đã khóa thay đổi; không tự đánh giá lại test!'
    else:
        path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    return config


def evaluate(project='.'):
    project = Path(project).resolve()
    frozen = freeze_experiment(project)  # Khóa cả ba cấu hình TRƯỚC khi đọc nhãn test.
    experiment_hash = sha256_file(project / 'configs/final_experiment.json')
    output = project / 'results/test'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        print('Đã có kết quả test với đúng mô hình/split đã khóa; dùng lại, không chạy lại.', flush=True)
        saved = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        assert saved['experiment_sha256'] == experiment_hash, 'Kết quả test thuộc thí nghiệm khác!'
        return saved

    test = pd.read_csv(project / 'splits/test.csv', keep_default_na=False)
    assert len(test) == 12630 and test.path.is_unique
    assert set(test.class_id) == set(range(43)) and set(test.source) == {'official_test'}
    y_true = test.class_id.to_numpy()
    all_results = []
    with threadpool_limits(limits=4):
        bundles = {name: joblib.load(project / frozen['models'][name]['path']) for name in METHODS}
        features_by_method, feature_times, stage_times = extract_shared_features(
            test, project / 'data/raw', bundles)
        for method in METHODS:
            folder = output / method
            folder.mkdir(exist_ok=True)
            # Cho phép tiếp tục nếu quá trình trước bị gián đoạn; không dự đoán lại mô hình đã xong.
            if (folder / 'metrics.json').exists():
                saved = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
                assert saved['experiment_sha256'] == experiment_hash
                all_results.append(saved)
                continue
            print(f'\nĐánh giá TEST: {method}', flush=True)
            bundle = bundles[method]
            features = features_by_method[method]
            feature_seconds = feature_times[method]
            start = perf_counter()
            predicted = bundle['classifier'].predict(features)
            predict_seconds = perf_counter() - start
            metrics = classification_metrics(y_true, predicted)
            result = {
                'method': bundle['method'], 'split': 'test', 'n_images': len(test),
                'experiment_sha256': experiment_hash,
                **metrics, 'n_features': features.shape[1],
                'feature_seconds': feature_seconds, 'predict_seconds': predict_seconds,
                'classifier_ms_per_image': 1000 * predict_seconds / len(test),
                'end_to_end_ms_per_image': 1000 * (feature_seconds + predict_seconds) / len(test),
            }
            predictions = test[['path', 'class_id']].rename(columns={'class_id': 'true_class'})
            predictions['predicted_class'] = predicted
            predictions['correct'] = y_true == predicted
            predictions.to_csv(folder / 'predictions.csv', index=False)
            per_class_report(y_true, predicted).to_csv(folder / 'per_class.csv', index=False)
            matrix = confusion_matrix(y_true, predicted, labels=range(43))
            pd.DataFrame(matrix, index=range(43), columns=range(43)).rename_axis('true_class').to_csv(
                folder / 'confusion_matrix.csv')
            (folder / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            all_results.append(result)
            pd.DataFrame(all_results).to_csv(output / 'comparison.csv', index=False)
            print(f'  Accuracy={metrics["accuracy"]:.4f}, Macro-F1={metrics["macro_f1"]:.4f}', flush=True)
            del features_by_method[method], bundles[method], features, bundle

    pd.DataFrame(all_results).to_csv(output / 'comparison.csv', index=False)
    summary = {
        'split': 'test', 'frozen_config': 'configs/final_experiment.json', 'results': all_results,
        'experiment_sha256': sha256_file(project / 'configs/final_experiment.json'),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'sklearn': sklearn.__version__, 'platform': platform.platform(), 'threads': 4},
        'shared_feature_stage_seconds': stage_times,
        'timing_note': 'Shared image read and HSV/HOG extraction, timed per stage; each method is charged for its own stages including shared read. One prediction batch per model. No tuning or refit based on test.',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='.')
    evaluate(parser.parse_args().project)
