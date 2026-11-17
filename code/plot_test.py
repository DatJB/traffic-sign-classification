"""Vẽ bảng, ma trận và ví dụ từ kết quả test đã có; không chọn lại tham số."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from preprocessing import read_bgr


def plot_results(project='.'):
    project = Path(project)
    root = project / 'results/test'
    table = pd.read_csv(root / 'comparison.csv')
    ax = table.set_index('method')[['accuracy', 'macro_f1']].plot.bar(figsize=(10, 5), rot=0)
    ax.set(ylim=(0, 1), ylabel='Score', title='Official test: configurations frozen before evaluation')
    plt.tight_layout()
    plt.savefig(root / 'comparison.png', dpi=160)
    plt.close()
    for method in ['hsv', 'hog', 'combined']:
        folder = root / method
        matrix = pd.read_csv(folder / 'confusion_matrix.csv', index_col=0).to_numpy()
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        ax.set(xticks=range(43), yticks=range(43), xlabel='Predicted class ID', ylabel='True class ID',
               title=f'{method.upper()}: official test (row-normalized)')
        ax.tick_params(axis='both', labelsize=7)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(folder / 'confusion_matrix.png', dpi=150)
        plt.close(fig)

        predictions = pd.read_csv(folder / 'predictions.csv')
        examples = pd.concat([
            predictions[predictions.correct].drop_duplicates('true_class').head(3),
            predictions[~predictions.correct].drop_duplicates('true_class').head(3),
        ])
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for ax in axes.flat:
            ax.axis('off')
        for ax, row in zip(axes.flat, examples.itertuples()):
            image = read_bgr(project / 'data/raw' / row.path)
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(f'True {row.true_class} / Pred {row.predicted_class}',
                         color='green' if row.correct else 'red')
        fig.tight_layout()
        fig.savefig(folder / 'examples.png', dpi=150)
        plt.close(fig)
    return table


if __name__ == '__main__':
    plt.switch_backend('Agg')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='.')
    print(plot_results(parser.parse_args().project).to_string(index=False))
