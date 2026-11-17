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
                          model_path, model_info, predict, processing_view, read_test_results)


@st.cache_resource(show_spinner=False, max_entries=3)
def cached_model(method, root, modification_time, file_size):
    return load_model(method, Path(root))


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
    .hero p {color:#526782;font-size:1rem;margin:0;}
    .eyebrow {font-size:.75rem;font-weight:700;color:#2563EB;letter-spacing:.12em;}
    .status {font-size:.85rem;margin:9px 0;color:#526782;}
    .result-name {font-size:1.65rem;font-weight:750;color:#174DAE;line-height:1.3;margin:16px 0;}
    .empty-result {padding:34px 8px;color:#61748E;line-height:1.7;}
    div[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:16px;}
    .stButton button {border-radius:10px;font-weight:650;}
    </style>''', unsafe_allow_html=True)
    st.markdown('''<div class="hero"><div class="eyebrow">GTSRB · COMPUTER VISION DEMO</div>
    <h1>Nhận dạng biển báo giao thông</h1>
    <p>Phân loại 43 lớp biển báo Đức trên bộ dữ liệu GTSRB</p></div>''', unsafe_allow_html=True)

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
        st.caption('Mô hình đã huấn luyện · Không học lại khi dùng demo')
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
                st.write('Bổ sung file mô hình đã huấn luyện cùng cấu hình trong bundle. '
                         'Không đổi tên file của một phương pháp khác để thay thế.')
                st.caption('Hướng dẫn huấn luyện thủ công ở README. Giao diện không gọi huấn luyện.')

    st.info('Ảnh đầu vào cần được cắt chứa một biển báo. Giao diện chỉ phân loại ảnh; '
            'không tự phát hiện biển báo trong ảnh đường phố và không được kiểm chứng cho biển Việt Nam.')

    source = st.radio('Nguồn ảnh', ['Tải ảnh lên', 'Ảnh mẫu'], horizontal=True, key='image_source')
    payload, filename = None, ''
    if source == 'Tải ảnh lên':
        uploaded = st.file_uploader('Chọn ảnh JPG, JPEG, PNG hoặc PPM',
                                    type=['jpg', 'jpeg', 'png', 'ppm'], key='upload',
                                    help='Tối đa 20 MB. Mỗi ảnh chứa một biển báo đã cắt.')
        if uploaded is not None:
            payload, filename = uploaded.getvalue(), uploaded.name
    else:
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

    # Khi đổi ảnh, mô hình hoặc tệp mô hình: xóa kết quả cũ trước khi render.
    fingerprint = (source, filename, hashlib.sha256(payload).hexdigest() if payload is not None else None,
                   method, tuple(signatures.items()))
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
            st.caption('Không suy ra nhãn thật từ ảnh hoặc tên tệp. Ảnh nhỏ có thể mờ khi phóng to.')
            for note in image.notes:
                st.info(note)
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
            st.caption('Thời gian gồm tiền xử lý, trích/chuẩn hóa đặc trưng và dự đoán; không tính nạp mô hình.')
        else:
            st.markdown('<div class="empty-result">Chưa có kết quả cho ảnh và mô hình hiện tại.<br>'
                        'Nhấn <b>Phân loại</b> để bắt đầu.</div>', unsafe_allow_html=True)
        st.caption('Mô hình có thể vẫn trả về một lớp khi ảnh không phải biển báo thuộc GTSRB. '
                   'Nhãn dự đoán không phải cam kết chính xác.')

    with st.expander('Xem quá trình xử lý'):
        if image is None or method not in resources:
            st.caption('Chọn ảnh và mô hình sẵn sàng để xem tiền xử lý.')
        else:
            bundle = resources[method]
            view = processing_view(bundle, image.bgr)
            st.image(view['resized_rgb'], caption='Ảnh resize dùng trong pipeline', width=160)
            st.write(f'**Số chiều vector:** {bundle["classifier"].n_features_in_}')
            st.json(model_info(bundle), expanded=False)
            if 'hsv' in view:
                st.markdown('**Các kênh HSV** — chỉ đổi thang hiển thị, không đổi dữ liệu trích đặc trưng.')
                for channel, title, maximum in zip(st.columns(3), ['H · 0–179', 'S · 0–255', 'V · 0–255'], [179, 255, 255]):
                    index = ['H', 'S', 'V'].index(title[0])
                    channel.image(view['hsv'][:, :, index].astype(np.float32) / maximum,
                                  caption=title, width=160)
            if 'gray' in view:
                a, b = st.columns(2)
                a.image(view['gray'], caption='Ảnh xám dùng cho HOG', width=160)
                b.image(view['gradient'], caption='Độ lớn gradient — minh họa cạnh', width=160)
                st.caption('Gradient Sobel chỉ dùng minh họa, không thay HOG của OpenCV trong dự đoán.')
                st.markdown('**Vector HOG thực sự được trích**')
                st.line_chart(pd.DataFrame({'HOG': view['hog_vector']}), height=170)

    with st.expander('So sánh các phương pháp', expanded=False):
        st.caption('Cùng một ảnh qua các mô hình sẵn sàng. Thời gian không gồm nạp mô hình; '
                   'một lượt chạy có thể chịu ảnh hưởng của cache và tải máy.')
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
                st.caption(f'Có {agree}/{len(rows)} phương pháp cùng dự đoán một lớp. '
                           'Sự đồng thuận này không phải xác suất đúng.')
            for title, error in failures:
                st.warning(f'{title}: không dự đoán được.')
                with st.expander(f'Chi tiết lỗi {title}'):
                    st.code(error)

    with st.expander('Kết quả trên tập kiểm tra'):
        st.caption('Đây là đánh giá toàn bộ mô hình trên tập test GTSRB, '
                   'khác với dự đoán của ảnh đang mở. Giao diện chỉ đọc kết quả đã lưu.')
        try:
            records = read_test_results(PROJECT_ROOT)
            if not records:
                st.info('Chưa có số liệu Accuracy/Macro-F1 của tập test trong results/test/.')
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
                    st.caption(f'Nguồn: results/test/{matrix_method}/confusion_matrix.png · Chuẩn hóa theo hàng.')
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
