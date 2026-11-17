import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import discover, audit_images
from preprocessing import read_bgr, preprocess


def split_by_track(data, validation_fraction=0.2, seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    train = data[data.source == 'official_train'].copy()
    test = data[data.source == 'official_test'].copy()
    if train.track_id.isna().any() or (train.track_id == '').any():
        raise ValueError('Training tracks are missing')
    if train.groupby('track_id').class_id.nunique().max() != 1:
        raise ValueError('A group contains multiple classes')
    rng = np.random.default_rng(seed)
    validation_tracks = set()
    for label, rows in train.groupby('class_id', sort=True):
        tracks = sorted(rows.track_id.unique())
        if len(tracks) < 2:
            raise ValueError(f'Class {label}: need at least 2 tracks')
        count = max(1, min(len(tracks) - 1, int(np.floor(len(tracks) * validation_fraction + 0.5))))
        validation_tracks.update(rng.permutation(tracks)[:count])
    val_mask = train.track_id.isin(validation_tracks)
    parts = {'train': train[~val_mask].copy(), 'validation': train[val_mask].copy(), 'test': test}
    if set(parts['train'].track_id) & set(parts['validation'].track_id):
        raise AssertionError('Track leakage')
    path_sets = [set(x.path) for x in parts.values()]
    if sum(map(len, path_sets)) != len(set.union(*path_sets)) or sum(map(len, parts.values())) != len(data):
        raise AssertionError('Overlapping or lost images')
    return parts


def prepare(root, out, fraction, seed, overwrite=False):
    root, out = Path(root).resolve(), Path(out)
    splits = out / 'splits'
    if not overwrite and any((splits / f'{s}.csv').exists() for s in ('train', 'validation', 'test')):
        raise FileExistsError('Fixed splits already exist. Use --overwrite only for an intentional replacement.')
    metrics, figures = out / 'results/metrics', out / 'results/figures'
    for folder in (splits, metrics, figures, out / 'samples'):
        folder.mkdir(parents=True, exist_ok=True)
    data = discover(root)
    expected = {'official_train': 39209, 'official_test': 12630}
    if data.source.value_counts().to_dict() != expected:
        raise ValueError(f'Need final GTSRB counts {expected}, got {data.source.value_counts().to_dict()}')
    issues, dimensions = audit_images(data, root)
    issues.to_csv(metrics / 'data_issues.csv', index=False)
    dimensions.to_csv(metrics / 'image_dimensions.csv', index=False)
    if (issues.severity == 'error').any():
        raise ValueError('Image audit failed. See results/metrics/data_issues.csv; no splits generated.')
    parts = split_by_track(data, fraction, seed)
    counts = pd.DataFrame({name: part.class_id.value_counts().reindex(range(43), fill_value=0)
                           for name, part in parts.items()}).rename_axis('class_id')
    counts.to_csv(metrics / 'class_distribution.csv')
    hashes = {}
    for name, part in parts.items():
        dest = splits / f'{name}.csv'
        part.sort_values(['class_id', 'path']).to_csv(dest, index=False, lineterminator='\n')
        hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    summary = {'seed': seed, 'requested_validation_fraction': fraction,
               'strategy': 'per-class shuffle of complete tracks; round nearest track count',
               'actual_validation_fraction': len(parts['validation']) / (len(parts['train']) + len(parts['validation'])),
               'counts': {name: len(part) for name, part in parts.items()},
               'tracks': {name: int(part.track_id.nunique()) for name, part in parts.items() if name != 'test'},
               'classes': 43, 'decode_errors': 0, 'metadata_warnings': len(issues),
               'split_sha256': hashes,
               'inventory_sha256': hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
               'preprocessing': {'size': [64, 64], 'roi_crop': False, 'input': 'uint8 BGR',
                                 'resize': 'INTER_AREA if shrinking both axes, otherwise INTER_LINEAR'},
               'numpy_version': np.__version__, 'opencv_version': cv2.__version__}
    summary['environment'] = {'python': platform.python_version(), 'platform': platform.platform(),
                              'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}
    (splits / 'metadata.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    ax = counts.plot.bar(figsize=(18, 6), width=0.85)
    ax.set(xlabel='GTSRB class ID', ylabel='Number of images', title='Class distribution (fixed track split)')
    plt.tight_layout()
    plt.savefig(figures / 'class_distribution.png', dpi=160)
    plt.close()
    # Samples only from train, deterministic; no looking at test predictions.
    sample_rows = [parts['train'][parts['train'].class_id == c].sort_values('path').iloc[0] for c in range(10)]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    sample_manifest = []
    for ax, row in zip(axes.flat, sample_rows):
        source = root / row.path
        name = f'class_{row.class_id:02d}_{source.name}'
        shutil.copy2(source, out / 'samples' / name)
        ax.imshow(preprocess(read_bgr(source), color='rgb'))
        ax.set_title(f'Class {row.class_id}')
        ax.axis('off')
        sample_manifest.append({'sample': name, 'original_path': row.path, 'class_id': row.class_id, 'split': 'train'})
    pd.DataFrame(sample_manifest).to_csv(out / 'samples/manifest.csv', index=False)
    fig.suptitle('Training samples resized to 64 x 64 (classes 0-9)')
    fig.tight_layout()
    fig.savefig(figures / 'training_samples.png', dpi=160)
    plt.close(fig)
    # Visual label reference for all 43 classes, using train only.
    names_path = Path(__file__).resolve().parents[1] / 'configs/class_names.csv'
    names = pd.read_csv(names_path).set_index('class_id')
    if set(names.index) != set(range(43)) or not names.index.is_unique:
        raise ValueError('Class-name mapping must contain exactly IDs 0..42')
    import textwrap
    fig, axes = plt.subplots(9, 5, figsize=(16, 25))
    for label, ax in enumerate(axes.flat):
        ax.axis('off')
        if label >= 43:
            continue
        row = parts['train'][parts['train'].class_id == label].sort_values('path').iloc[0]
        ax.imshow(preprocess(read_bgr(root / row.path), color='rgb'))
        ax.set_title(f'{label}: ' + '\n'.join(textwrap.wrap(names.loc[label, 'name_en'], 27)), fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / 'class_reference.png', dpi=120)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/raw')
    parser.add_argument('--output', default='.')
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    prepare(args.root, args.output, args.validation_fraction, args.seed, args.overwrite)
