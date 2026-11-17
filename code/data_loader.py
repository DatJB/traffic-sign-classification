import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from preprocessing import read_bgr


def discover(root):
    root = Path(root).resolve()
    candidates = [p for p in root.rglob('*') if p.is_dir() and p.name in ('Training', 'Final_Training')]
    candidates = [p / 'Images' if (p / 'Images').is_dir() else p for p in candidates]
    if len(candidates) != 1:
        raise ValueError(f'Expected one Training/Final_Training directory, found {candidates}')
    train_dir = candidates[0]
    rows = []
    for folder in sorted(train_dir.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        csvs = sorted(folder.glob('GT-*.csv'))
        if len(csvs) > 1:
            raise ValueError(f'Ambiguous annotations: {folder}')
        if csvs:
            table = pd.read_csv(csvs[0], sep=';')
            for record in table.to_dict('records'):
                path = folder / record['Filename']
                label = int(record['ClassId'])
                if label != int(folder.name):
                    raise ValueError(f'Annotation/folder class mismatch: {path}')
                rows.append(_row(path, label, root, 'official_train', record))
            annotated = set(table['Filename'])
            if annotated != {p.name for p in folder.glob('*.ppm')}:
                raise ValueError(f'Annotation/image coverage mismatch: {folder}')
        else:
            # torchvision's fixed archive can encode labels in class directories.
            for path in sorted(folder.glob('*.ppm')):
                rows.append(_row(path, int(folder.name), root, 'official_train'))
    test_csvs = list(root.rglob('GT-final_test.csv'))
    test_dirs = [p for p in root.rglob('Images') if p.parent.name == 'Final_Test']
    if len(test_csvs) != 1 or len(test_dirs) != 1:
        raise ValueError('Need official Final_Test/Images and GT-final_test.csv (with ClassId)')
    table = pd.read_csv(test_csvs[0], sep=';')
    if 'ClassId' not in table:
        raise ValueError('Test annotations must include ClassId')
    if set(table['Filename']) != {p.name for p in test_dirs[0].glob('*.ppm')}:
        raise ValueError('Test annotation/image coverage mismatch')
    for record in table.to_dict('records'):
        rows.append(_row(test_dirs[0] / record['Filename'], int(record['ClassId']),
                         root, 'official_test', record))
    data = pd.DataFrame(rows)
    if data.empty or data['path'].duplicated().any():
        raise ValueError('Empty dataset or duplicate paths')
    if not data['class_id'].between(0, 42).all():
        raise ValueError('Class IDs must be 0..42')
    for source, subset in data.groupby('source'):
        if set(subset.class_id) != set(range(43)):
            raise ValueError(f'{source} does not contain all 43 classes')
    return data.sort_values(['source', 'class_id', 'path']).reset_index(drop=True)


def _row(path, label, root, source, annotation=None):
    if source == 'official_train':
        match = re.fullmatch(r'(\d{5})_(\d{5})', path.stem)
        if not match:
            raise ValueError(f'Track missing: {path}. Restore original filenames before splitting.')
        # Track numbers restart in each class: the group key must include class.
        track = f'{label:05d}:{match[1]}'
    else:
        track = ''  # Official flat test filenames do not expose tracks.
    row = {'path': path.relative_to(root).as_posix(), 'class_id': label,
           'track_id': track, 'source': source}
    if annotation:
        for key in ('Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2'):
            if key in annotation:
                row[key] = annotation[key]
    return row


def audit_images(data, root):
    """Decode every image, check metadata. Return all issues, do not drop records."""
    def check(row):
        issues = []
        dimension = None
        try:
            image = read_bgr(Path(root) / row['path'])
            h, w = image.shape[:2]
            dimension = {'path': row['path'], 'width': w, 'height': h}
            # Original GTSRB annotation Width/Height can have known inaccuracies;
            # these are reported, while decoded dimensions remain authoritative.
            for name, actual in [('Width', w), ('Height', h)]:
                if pd.notna(row.get(name)) and int(row[name]) != actual:
                    issues.append({'path': row['path'], 'severity': 'warning',
                                   'issue': f'{name}: annotation={row[name]}, decoded={actual}'})
        except (OSError, ValueError) as exc:
            issues.append({'path': row['path'], 'severity': 'error', 'issue': str(exc)})
        return issues, dimension
    issues, dimensions = [], []
    # I/O-bound decoding; map preserves manifest order for reproducible logs.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (errors, dimension) in enumerate(pool.map(check, data.to_dict('records')), start=1):
            issues.extend(errors)
            if dimension is not None:
                dimensions.append(dimension)
            if index % 10000 == 0:
                print(f'Audited {index}/{len(data)} images', flush=True)
    return pd.DataFrame(issues, columns=['path', 'severity', 'issue']), pd.DataFrame(dimensions)


def load_split(manifest, root):
    """Return a manifest with an absolute image_path column for teammates."""
    data = pd.read_csv(manifest, keep_default_na=False)
    base = Path(root).resolve()
    paths = [(base / p).resolve() for p in data['path']]
    if any(not p.is_relative_to(base) for p in paths):
        raise ValueError('Manifest path escapes data root')
    data['image_path'] = [str(p) for p in paths]
    return data
