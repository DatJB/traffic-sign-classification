"""Logic cho demo: chỉ nạp mô hình và dự đoán, không import/chạy huấn luyện."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import RLock
from time import perf_counter
import json
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from model_features import image_features
from preprocessing import preprocess
from features_hog import create_hog, hog_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    'hsv': {'title': 'HSV + kNN', 'normalization': 'l1_per_image'},
    'hog': {'title': 'HOG + kNN', 'normalization': 'block_l2hys_then_global_l2'},
    'combined': {'title': 'HSV + HOG + kNN', 'normalization': 'l2_each_group_then_weighted_concat'},
}
ALLOWED_EXTENSIONS = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.ppm': 'PPM'}
PREDICTION_LOCK = RLock()


@dataclass
class InputImage:
    bgr: np.ndarray
    notes: list
    filename: str
    original_mode: str


def decode_image(payload, filename):
    """Đọc bytes như read_bgr của dự án: imdecode(..., IMREAD_COLOR)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError('Chỉ hỗ trợ ảnh JPG, JPEG, PNG hoặc PPM.')
    if not payload:
        raise ValueError('Tệp ảnh rỗng. Vui lòng chọn ảnh khác.')
    if len(payload) > 20 * 1024 * 1024:
        raise ValueError('Ảnh vượt quá 20 MB. Hãy cắt và lưu ảnh nhỏ hơn.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as header:
                if header.format != ALLOWED_EXTENSIONS[suffix]:
                    raise ValueError('Nội dung ảnh không khớp định dạng của tên tệp.')
                if header.width * header.height > 16_000_000:
                    raise ValueError('Ảnh quá lớn (trên 16 triệu pixel). Hãy cắt vùng biển báo trước.')
                mode = header.mode
                has_alpha = 'A' in mode or 'transparency' in header.info
                header.verify()
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('OpenCV không giải mã được ảnh này.')
    except (UnidentifiedImageError, OSError, SyntaxError, cv2.error, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError('Ảnh bị lỗi hoặc không đọc được. Vui lòng chọn tệp ảnh hợp lệ.') from error
    notes = []
    if mode in ('1', 'L', 'LA', 'I', 'F', 'I;16'):
        notes.append('Ảnh xám được đọc thành 3 kênh BGR, giống bộ đọc ảnh của dự án.')
    if has_alpha:
        notes.append('Ảnh có alpha: OpenCV bỏ kênh trong suốt, giữ giá trị màu gốc. '
                     'Ảnh hiển thị là dữ liệu màu thực sự đưa vào mô hình; vùng trong suốt có thể ảnh hưởng dự đoán.')
    return InputImage(bgr, notes, Path(filename).name, mode)


def model_path(method, root=PROJECT_ROOT):
    if method not in METHODS:
        raise ValueError('Phương pháp không được hỗ trợ.')
    return Path(root) / 'models' / f'{method}_knn.joblib'


def validate_bundle(method, bundle):
    if not isinstance(bundle, dict) or bundle.get('method') != METHODS[method]['title']:
        raise ValueError('Tệp mô hình không đúng phương pháp đã chọn.')
    if bundle.get('normalization') != METHODS[method]['normalization']:
        raise ValueError('Thiếu hoặc sai cấu hình chuẩn hóa đã huấn luyện.')
    if tuple(bundle.get('size', [])) != (64, 64):
        raise ValueError('Thiếu cấu hình resize 64×64 đã huấn luyện.')
    if method in ('hsv', 'combined'):
        bins = bundle.get('bins', [])
        if len(bins) != 3 or any(not isinstance(b, (int, np.integer)) or b <= 0 for b in bins):
            raise ValueError('Thiếu cấu hình số bin HSV hợp lệ.')
        dimension = int(np.prod(bins))
    else:
        dimension = 0
    if method in ('hog', 'combined'):
        cell = bundle.get('cell_size')
        descriptor = create_hog(cell, bundle['size'])
        expected = {'block_cells': 2, 'stride': cell, 'orientations': 9,
                    'signed_gradient': False, 'gamma_correction': False, 'l2hys_threshold': 0.2}
        if bundle.get('hog_parameters') != expected:
            raise ValueError('Thiếu cấu hình HOG hoặc cấu hình không khớp hàm HOG hiện có.')
        dimension += descriptor.getDescriptorSize()
    if method == 'combined' and not 0 < bundle.get('hsv_weight', -1) < 1:
        raise ValueError('Thiếu trọng số kết hợp HSV/HOG hợp lệ.')
    classifier = bundle.get('classifier')
    if not isinstance(classifier, KNeighborsClassifier):
        raise ValueError('Tệp chưa chứa bộ phân loại kNN được hỗ trợ.')
    check_is_fitted(classifier)
    if not 1 <= classifier.n_neighbors <= classifier.n_samples_fit_:
        raise ValueError('Cấu hình k không hợp lệ với số mẫu đã huấn luyện.')
    if not np.array_equal(classifier.classes_, np.arange(43)):
        raise ValueError('Bộ phân loại không có đủ 43 mã lớp GTSRB 0–42.')
    if classifier.n_features_in_ != dimension:
        raise ValueError('Số chiều mô hình không khớp cấu hình đặc trưng đã lưu.')
    return bundle


def load_model(method, root=PROJECT_ROOT):
    path = model_path(method, root)
    if not path.is_file():
        raise FileNotFoundError(f'Thiếu {path.name}. Cần bổ sung mô hình đã huấn luyện vào models/.')
    return validate_bundle(method, joblib.load(path))


def clean_name(value):
    return str(value).strip() if pd.notna(value) else ''


def label_name(class_id, bundle, root=PROJECT_ROOT):
    """Ưu tiên bảng nhãn dự án, rồi bản nhúng trong mô hình; không tự dịch tên."""
    path = Path(root) / 'configs/class_names.csv'
    if path.exists():
        try:
            table = pd.read_csv(path)
            if 'class_id' in table.columns and table.class_id.is_unique:
                rows = table[table.class_id == class_id]
                if len(rows) == 1:
                    row = rows.iloc[0]
                    name = clean_name(row.get('name_vi')) or clean_name(row.get('name_en'))
                    if name:
                        return name
        except (OSError, ValueError):
            pass  # Dùng ánh xạ đã nhúng trong mô hình nếu CSV lỗi/thiếu tên.
    for row in bundle.get('class_names', []):
        if row.get('class_id') == class_id:
            return clean_name(row.get('name_vi')) or clean_name(row.get('name_en')) or f'Lớp {class_id}'
    return f'Lớp {class_id}'


def predict(bundle, bgr, root=PROJECT_ROOT):
    # Khóa vì threadpool_limits điều khiển tài nguyên dùng chung trong tiến trình.
    # Timer chỉ bao gồm tiền xử lý/đặc trưng/chuẩn hóa/predict; không gồm nạp mô hình.
    with PREDICTION_LOCK, threadpool_limits(limits=4):
        start = perf_counter()
        vector = image_features(bundle, bgr)
        predicted = int(bundle['classifier'].predict(vector.reshape(1, -1))[0])
        milliseconds = (perf_counter() - start) * 1000
    return {'method': bundle['method'], 'class_id': predicted,
            'name': label_name(predicted, bundle, root), 'milliseconds': milliseconds,
            'n_features': len(vector)}


def model_info(bundle):
    model = bundle['classifier']
    info = {'Phương pháp': bundle['method'], 'Resize': ' × '.join(map(str, bundle['size'])),
            'Số chiều': model.n_features_in_, 'k': model.n_neighbors,
            'Khoảng cách': model.metric, 'Bỏ phiếu': model.weights,
            'Chuẩn hóa': bundle['normalization']}
    if 'bins' in bundle:
        info['Bins HSV (H, S, V)'] = ' / '.join(map(str, bundle['bins']))
    if 'cell_size' in bundle:
        info['Cell HOG'] = bundle['cell_size']
        info['Block / stride'] = f"{2*bundle['cell_size']} / {bundle['cell_size']} px"
        info['Số hướng'] = bundle['hog_parameters']['orientations']
    if 'hsv_weight' in bundle:
        info['Trọng số HSV / HOG'] = f"{bundle['hsv_weight']} / {1-bundle['hsv_weight']}"
    return info


def processing_view(bundle, bgr):
    view = {'resized_rgb': preprocess(bgr, size=bundle['size'], color='rgb')}
    if 'bins' in bundle:
        view['hsv'] = preprocess(bgr, size=bundle['size'], color='hsv')
    if 'cell_size' in bundle:
        view['gray'] = preprocess(bgr, size=bundle['size'], color='gray')
        view['hog_vector'] = hog_features(bgr, bundle['cell_size'], bundle['size'])
        gx = cv2.Sobel(view['gray'], cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(view['gray'], cv2.CV_32F, 0, 1, ksize=1)
        magnitude = cv2.magnitude(gx, gy)
        view['gradient'] = magnitude / max(float(magnitude.max()), 1.0)
    return view


def read_test_results(root=PROJECT_ROOT):
    """Chỉ đọc số liệu test thực, không lấy nhầm CSV validation."""
    root = Path(root)
    path = root / 'results/test/summary.json'
    records = []
    if path.exists():
        summary = json.loads(path.read_text(encoding='utf-8'))
        if summary.get('split') != 'test':
            raise ValueError('Tệp kết quả không được đánh dấu là tập test.')
        records = summary.get('results', [])
    else:
        for method in METHODS:
            path = root / f'results/test/{method}/metrics.json'
            if path.exists():
                records.append(json.loads(path.read_text(encoding='utf-8')))
    valid = []
    for record in records:
        if (record.get('split') == 'test' and record.get('method') in [m['title'] for m in METHODS.values()]
                and all(np.isfinite(record.get(k, np.nan)) and 0 <= record[k] <= 1
                        for k in ['accuracy', 'macro_f1']) and record.get('n_images', 0) > 0):
            valid.append(record)
    return valid
