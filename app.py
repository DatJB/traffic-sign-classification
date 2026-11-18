"""Streamlit demo: streamlit run app.py (không huấn luyện mô hình)."""
from pathlib import Path
import hashlib
import html
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'code'))
from demo_service import (METHODS, ALLOWED_EXTENSIONS, decode_image, load_model,
                          model_path, model_info, predict, processing_view, read_test_results,
                          demo_catalog, demo_image_bytes, label_name)


@st.cache_resource(show_spinner=False, max_entries=3)
def cached_model(method, root, modification_time, file_size):
    return load_model(method, Path(root))


@st.cache_data(show_spinner=False)
def cached_catalog(split, root, modification_time, file_size):
    return demo_catalog(split, Path(root))


def error_details(error):
    with st.expander('Chi tiết kỹ thuật'):
        st.code(f'{type(error).__name__}: {error}')


def main():
    st.set_page_config(page_title='Nhận dạng biển báo giao thông', page_icon='🚦', layout='wide')
    st.markdown('''<style>
    .stApp {background:#F5F8FD;color:#18283E;}
    [data-testid="stSidebar"] {background:#fff;border-right:1px solid #DFE7F4;}
    .block-container {max-width:1280px;padding-top:2rem;padding-bottom:2.5rem;}
    h1,h2,h3 {color:#18283E;letter-spacing:-.025em;}
    .hero {border:1px solid #D8E6FC;background:linear-gradient(115deg,#fff,#EBF3FF);
      border-radius:20px;padding:26px 30px;margin-bottom:20px;}
    .hero h1 {font-size:2.15rem;margin:8px 0;line-height:1.2;}
    .eyebrow {font-size:.75rem;font-weight:700;color:#2563EB;letter-spacing:.12em;}
    .status {font-size:.85rem;margin:9px 0;color:#526782;}
    .result-name {font-size:1.65rem;font-weight:750;color:#174DAE;line-height:1.3;margin:16px 0;}
    .empty-result {padding:34px 8px;color:#61748E;line-height:1.7;}
    div[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:16px;}
    .stButton button {border-radius:10px;font-weight:650;}
    </style>''', unsafe_allow_html=True)
    st.markdown('''<div class="hero"><div class="eyebrow">GTSRB</div>
    <h1>Nhận dạng biển báo giao thông</h1></div>''', unsafe_allow_html=True)

    resources, errors, signatures = {}, {}, {}
    with st.spinner('Đang kiểm tra và nạp mô hình đã lưu…'):
        for method in METHODS:
            path = model_path(method, PROJECT_ROOT)
            try:
                stat = path.stat()
                signatures[method] = (stat.st_mtime_ns, stat.st_size)
                resources[method] = cached_model(method, str(PROJECT_ROOT), *signatures[method])
            except Exception as error:
                errors[method] = error

    with st.sidebar:
        st.markdown('### 🚦 Phương pháp')
        ready = list(resources)
        options = ready if ready else list(METHODS)
        default = options.index('hog') if 'hog' in options else 0
        method = st.selectbox('Chọn mô hình', options, index=default,
                              format_func=lambda key: METHODS[key]['title'], disabled=not ready,
                              key='method')
        st.markdown('#### Trạng thái tài nguyên')
        for key, spec in METHODS.items():
            icon, status = ('🟢', 'Sẵn sàng') if key in resources else ('⚪', 'Chưa sẵn sàng')
            st.markdown(f'<div class="status">{icon} <b>{spec["title"]}</b><br>{status}</div>', unsafe_allow_html=True)
        if method in resources:
            with st.expander('Thông tin mô hình', expanded=True):
                for name, value in model_info(resources[method]).items():
                    st.markdown(f'**{name}:** {value}')
        if errors:
            with st.expander('Cách bổ sung tài nguyên', expanded=not ready):
                for key, error in errors.items():
                    st.warning(f'{METHODS[key]["title"]}: chưa nạp được mô hình/cấu hình.')
                    st.caption(str(error))
                    st.code(f'models/{key}_knn.joblib')
                st.write('Bổ sung mô hình và cấu hình tương ứng.')

    source = st.radio('Nguồn ảnh', ['Tải ảnh lên', 'Ảnh mẫu', 'Ảnh validation / test'],
                      horizontal=True, key='image_source')
    payload, filename = None, ''
    dataset_identity = None
    if source == 'Tải ảnh lên':
        uploaded = st.file_uploader('Chọn ảnh JPG, JPEG, PNG hoặc PPM',
                                    type=['jpg', 'jpeg', 'png', 'ppm'], key='upload',
                                    help='Tối đa 20 MB.')
        if uploaded is not None:
            payload, filename = uploaded.getvalue(), uploaded.name
    elif source == 'Ảnh mẫu':
        folder = PROJECT_ROOT / 'samples'
        samples = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS) if folder.is_dir() else []
        if samples:
            filename = st.selectbox('Chọn ảnh mẫu', [p.name for p in samples], key='sample')
            try:
                payload = (folder / filename).read_bytes()
            except OSError as error:
                st.error('Không đọc được ảnh mẫu. Vui lòng chọn ảnh khác.')
                error_details(error)
        else:
            st.warning('Chưa có ảnh mẫu. Bổ sung ảnh vào samples/ hoặc chọn Tải ảnh lên.')
    else:
        split = st.selectbox('Tập dữ liệu', ['validation', 'test'],
                             format_func=lambda key: 'Validation' if key == 'validation'
                             else 'Test', key='dataset_split')
        try:
            stat = (PROJECT_ROOT / 'splits' / f'{split}.csv').stat()
            catalog = cached_catalog(split, str(PROJECT_ROOT), stat.st_mtime_ns, stat.st_size)
            options = catalog.path.tolist()
            positions = {path: index + 1 for index, path in enumerate(options)}
            relative_path = st.selectbox('Chọn ảnh chưa dùng huấn luyện', options,
                format_func=lambda path: f'{positions[path]:02d} · {"/".join(Path(path).parts[-2:])}',
                key=f'dataset_image_{split}')
            dataset_identity = (split, relative_path)
            filename = Path(relative_path).name
            payload = demo_image_bytes(relative_path, split, PROJECT_ROOT)
            if st.checkbox('Xem nhãn chuẩn từ bộ dữ liệu', key='reveal_dataset_label'):
                class_id = int(catalog.loc[catalog.path == relative_path, 'class_id'].iloc[0])
                name = label_name(class_id, resources.get(method, {}), PROJECT_ROOT)
                st.info(f'Nhãn chuẩn: {name} · Class ID {class_id}')
        except Exception as error:
            st.warning('Chưa đọc được ảnh validation/test. Cần danh sách trong splits/ và ảnh/ZIP GTSRB trong data/raw/.')
            error_details(error)

    # Khi đổi ảnh, mô hình hoặc tệp mô hình: xóa kết quả cũ trước khi render.
    fingerprint = (source, filename, hashlib.sha256(payload).hexdigest() if payload is not None else None,
                   method, tuple(signatures.items()), dataset_identity)
    if st.session_state.get('input_fingerprint') != fingerprint:
        st.session_state['input_fingerprint'] = fingerprint
        st.session_state.pop('prediction', None)
        st.session_state.pop('comparison', None)

    image = None
    if payload is not None:
        try:
            image = decode_image(payload, filename)
        except (ValueError, OSError) as error:
            st.error(str(error))
    if not ready:
        st.warning('Chưa có phương pháp sẵn sàng. Bổ sung mô hình/cấu hình theo hướng dẫn ở sidebar.')

    left, right = st.columns([1, 1], gap='large')
    with left, st.container(border=True):
        st.subheader('Ảnh đầu vào')
        if image is None:
            st.caption('Chọn một ảnh để bắt đầu.')
        else:
            height, width = image.bgr.shape[:2]
            st.image(cv2.cvtColor(image.bgr, cv2.COLOR_BGR2RGB), width=min(320, max(140, width * 2)))
            st.caption(f'{image.filename} · {width} × {height} pixel · Màu gốc: {image.original_mode}')
        classify = st.button('Phân loại', type='primary', width="stretch",
                             disabled=image is None or method not in resources, key='classify')
        if classify:
            st.session_state.pop('prediction', None)
            try:
                with st.spinner('Đang tiền xử lý và phân loại…'):
                    st.session_state['prediction'] = predict(resources[method], image.bgr, PROJECT_ROOT)
            except Exception as error:
                st.error('Không thể phân loại ảnh này. Kiểm tra mô hình hoặc chọn ảnh khác.')
                error_details(error)

    with right, st.container(border=True):
        st.subheader('Kết quả dự đoán')
        result = st.session_state.get('prediction')
        if result:
            st.markdown(f'<div class="result-name">{html.escape(result["name"])}</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            a.metric('Class ID', result['class_id'])
            b.metric('Thời gian xử lý', f'{result["milliseconds"]:.1f} ms')
            st.write(f'**Phương pháp:** {result["method"]}')
        else:
            st.markdown('<div class="empty-result">Chưa có kết quả.</div>', unsafe_allow_html=True)

    with st.expander('Xem quá trình xử lý'):
        if image is None or method not in resources:
            st.caption('Chọn ảnh và mô hình sẵn sàng để xem tiền xử lý.')
        else:
            bundle = resources[method]
            view = processing_view(bundle, image.bgr)
            st.image(view['resized_rgb'], caption='Ảnh resize', width=160)
            st.write(f'**Số chiều vector:** {bundle["classifier"].n_features_in_}')
            st.json(model_info(bundle), expanded=False)
            if 'hsv' in view:
                st.markdown('**Các kênh HSV**')
                for channel, title, maximum in zip(st.columns(3), ['H · 0–179', 'S · 0–255', 'V · 0–255'], [179, 255, 255]):
                    index = ['H', 'S', 'V'].index(title[0])
                    channel.image(view['hsv'][:, :, index].astype(np.float32) / maximum,
                                  caption=title, width=160)
            if 'gray' in view:
                a, b = st.columns(2)
                a.image(view['gray'], caption='Ảnh xám', width=160)
                b.image(view['gradient'], caption='Độ lớn gradient', width=160)
                st.markdown('**Vector HOG**')
                st.line_chart(pd.DataFrame({'HOG': view['hog_vector']}), height=170)

    with st.expander('So sánh các phương pháp', expanded=False):
        compare = st.button('So sánh các phương pháp', disabled=image is None or not resources, key='compare')
        if compare:
            rows, failures = [], []
            with st.spinner('Đang chạy các phương pháp trên cùng ảnh…'):
                for key, bundle in resources.items():
                    try:
                        rows.append(predict(bundle, image.bgr, PROJECT_ROOT))
                    except Exception as error:
                        failures.append((METHODS[key]['title'], str(error)))
            st.session_state['comparison'] = (rows, failures)
        if 'comparison' in st.session_state:
            rows, failures = st.session_state['comparison']
            if rows:
                table = pd.DataFrame(rows).rename(columns={'method': 'Phương pháp', 'name': 'Nhãn dự đoán',
                    'class_id': 'Class ID', 'milliseconds': 'Thời gian (ms)'})
                st.dataframe(table[['Phương pháp', 'Nhãn dự đoán', 'Class ID', 'Thời gian (ms)']].round(2),
                             hide_index=True, width="stretch")
                agree = pd.Series([r['class_id'] for r in rows]).value_counts().max()
                st.caption(f'{agree}/{len(rows)} phương pháp cùng nhãn.')
            for title, error in failures:
                st.warning(f'{title}: không dự đoán được.')
                with st.expander(f'Chi tiết lỗi {title}'):
                    st.code(error)

    with st.expander('Kết quả trên tập kiểm tra'):
        try:
            records = read_test_results(PROJECT_ROOT)
            if not records:
                st.info('Chưa có số liệu Accuracy/Macro-F1 của tập test.')
            else:
                table = pd.DataFrame([{'Phương pháp': r['method'], 'Accuracy': f'{100*r["accuracy"]:.2f}%',
                    'Macro-F1': f'{r["macro_f1"]:.4f}', 'Số ảnh test': r['n_images']} for r in records])
                st.dataframe(table, hide_index=True, width="stretch")
            tested = [key for key in METHODS
                      if any((PROJECT_ROOT / f'results/test/{key}/confusion_matrix.{ext}').exists()
                             for ext in ['png', 'csv'])]
            if tested:
                matrix_method = st.selectbox('Ma trận nhầm lẫn', tested,
                                             format_func=lambda key: METHODS[key]['title'], key='matrix_method')
                folder = PROJECT_ROOT / 'results/test' / matrix_method
                if (folder / 'confusion_matrix.png').exists():
                    st.image(str(folder / 'confusion_matrix.png'), width='stretch')
                elif (folder / 'confusion_matrix.csv').exists():
                    st.write('Ma trận số đếm đã lưu:')
                    st.dataframe(pd.read_csv(folder / 'confusion_matrix.csv', index_col=0), width="stretch")
            else:
                st.caption('Chưa có ma trận nhầm lẫn đã lưu.')
        except Exception as error:
            st.warning('Chưa đọc được kết quả test. Các chức năng phân loại vẫn dùng được.')
            error_details(error)


if __name__ == '__main__':
    main()
