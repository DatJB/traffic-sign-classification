"""Vẽ kết quả validation từ CSV có sẵn; không dự đoán hoặc đọc test."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from evaluation import classification_metrics


def plot_results(project='.', include_combined=False):
    project = Path(project)
    comparison_dir = project / 'results/comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison = []
    reference_splits = None
    methods = ['hsv', 'hog', 'combined'] if include_combined else ['hsv', 'hog']
    suffix = '_all' if include_combined else ''
    for method in methods:
        folder = project / 'results' / method
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        assert summary['split'] == 'validation' and summary['test_evaluated'] is False
        if reference_splits is None:
            reference_splits = summary['split_sha256']
        assert summary['split_sha256'] == reference_splits, 'Các phương pháp dùng split khác nhau!'
        predictions = pd.read_csv(folder / 'validation_predictions.csv')
        metrics = classification_metrics(predictions.true_class, predictions.predicted_class)
        for name in metrics:
            assert np.isclose(metrics[name], summary['best'][name])

        matrix = confusion_matrix(predictions.true_class, predictions.predicted_class, labels=range(43))
        pd.DataFrame(matrix, index=range(43), columns=range(43)).rename_axis('true_class').to_csv(
            folder / 'validation_confusion_matrix.csv')
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        ax.set(xticks=range(43), yticks=range(43), xlabel='Predicted class ID',
               ylabel='True class ID', title=f'{summary["method"]}: validation (row-normalized)')
        ax.tick_params(axis='both', labelsize=7)
        plt.colorbar(im, ax=ax, label='Fraction within true class')
        fig.tight_layout()
        fig.savefig(folder / 'validation_confusion_matrix.png', dpi=150)
        plt.close(fig)
        best = summary['best']
        comparison.append({
            'method': summary['method'], 'split': 'validation',
            'search_budget': summary['search_budget'], **metrics,
            'n_features': best['n_features'], 'k': best['k'], 'distance': best['distance'],
            'classifier_ms_per_image': best['classifier_ms_per_image'],
            'end_to_end_ms_per_image': best['end_to_end_ms_per_image'],
        })

    table = pd.DataFrame(comparison)
    table.to_csv(comparison_dir / f'validation_comparison{suffix}.csv', index=False)
    ax = table.set_index('method')[['accuracy', 'macro_f1']].plot.bar(figsize=(8, 5), rot=0)
    ax.set(ylim=(0, 1), ylabel='Score', title='Validation comparison (12 configurations per method)')
    plt.tight_layout()
    plt.savefig(comparison_dir / f'validation_comparison{suffix}.png', dpi=160)
    plt.close()
    return table


if __name__ == '__main__':
    plt.switch_backend('Agg')  # Script chỉ lưu ảnh; khi import trong notebook giữ inline.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='.')
    parser.add_argument('--include-combined', action='store_true')
    args = parser.parse_args()
    print(plot_results(args.project, args.include_combined).to_string(index=False))
