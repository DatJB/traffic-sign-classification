# Phân loại biển báo GTSRB bằng HSV/HOG và kNN

So sánh HSV + kNN, HOG + kNN và HSV + HOG + kNN trên 43 lớp biển báo Đức. Đầu vào là ảnh đã cắt chứa một biển báo; không bao gồm phát hiện biển trong ảnh đường phố hoặc nhận dạng biển Việt Nam.

## Giao diện demo Streamlit

Cài thư viện và chạy từ thư mục dự án:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Nếu `streamlit` chưa có trong PATH, dùng `python -m streamlit run app.py`. Mở địa chỉ localhost mà Streamlit in ra. Nhấn **Ctrl+C** ở terminal để dừng. Có thể gọi `streamlit run "C:\Users\Dat\Documents\Study\Projects\XLA_N7_CT\app.py"` từ thư mục khác: đường dẫn dữ liệu/mô hình trong app luôn tính từ vị trí `app.py`.

**Cách trình bày demo:**

1. Chọn HSV + kNN, HOG + kNN hoặc HSV + HOG + kNN ở sidebar. Chỉ mô hình đã nạp và kiểm tra cấu hình thành công mới được chọn.
2. Upload một JPG/JPEG/PNG/PPM, hoặc chọn ảnh từ `samples/`. Ảnh cần được cắt chứa một biển báo; tối đa 20 MB và 16 triệu pixel. Giao diện hiển thị kích thước thật, không suy ra nhãn thật từ tên ảnh.
3. Nhấn **Phân loại** để xem tên Việt, Class ID, phương pháp và thời gian tiền xử lý + đặc trưng + dự đoán (không tính nạp mô hình).
4. Mở **Xem quá trình xử lý** để xem resize 64×64, từng kênh HSV, ảnh xám, vector HOG thực và minh họa độ lớn gradient. Gradient chỉ để vẽ, không thay hàm HOG dùng khi huấn luyện.
5. Nhấn **So sánh các phương pháp** để chạy cùng ảnh qua những mô hình sẵn sàng. Đồng thuận không phải xác suất đúng.
6. Mở **Kết quả trên tập kiểm tra** để xem số liệu test và ma trận đã lưu, tách biệt với ảnh đang mở. Không có số liệu thì thông báo thiếu, không tự đánh giá lại hoặc tạo kết quả giả.

**Tài nguyên cần có:**

| Phương pháp | Mô hình | Cấu hình nằm trong bundle |
|---|---|---|
| HSV | `models/hsv_knn.joblib` | size, bins, normalization, classifier |
| HOG | `models/hog_knn.joblib` | size, cell_size, hog_parameters, normalization, classifier |
| Kết hợp | `models/combined_knn.joblib` | cấu hình HSV/HOG, hsv_weight, normalization, classifier |

Ánh xạ tên lấy từ `configs/class_names.csv`, dự phòng bằng `class_names` nhúng trong bundle. Thiếu tên Việt thì dùng tên Anh hoặc `Lớp <Class ID>`, không tự dịch/đoán. Các mô hình dùng khoảng 500 MB dữ liệu đặc trưng; nạp lần đầu cần thời gian và bộ nhớ, sau đó được cache theo đường dẫn/mtime/kích thước. File thiếu/hỏng/cấu hình không khớp sẽ được ghi rõ ở sidebar và khóa phương pháp tương ứng. Khôi phục bundle của đúng phương pháp cùng cấu hình; hướng dẫn huấn luyện thủ công nằm ở các mục Người 2–4. App không import/chạy script huấn luyện hay fit scaler.

Kết quả test đọc từ `results/test/summary.json` hoặc `results/test/<phương pháp>/metrics.json`; ma trận đọc PNG/CSV tại cùng thư mục. Đổi ảnh hoặc mô hình sẽ xóa kết quả cũ. RGB/BGR, resize, HOG/HSV, chuẩn hóa, thứ tự ghép và trọng số được tái sử dụng qua `code/model_features.py::image_features`. Ảnh xám và alpha được đọc bằng đúng `cv2.IMREAD_COLOR` như `read_bgr` của dự án: ảnh xám thành BGR, alpha bị bỏ chứ không ghép nền trắng; UI thông báo cách đọc và hiển thị đúng dữ liệu màu đưa vào mô hình.

**Giới hạn:** chỉ phân loại crop của biển thuộc GTSRB, không phát hiện vị trí biển trong ảnh đường phố; không cam kết cho biển Việt Nam. Ảnh ngoài phạm vi vẫn có thể nhận một trong 43 lớp. Không hiển thị phiếu kNN thành xác suất chính xác. Thời gian demo là một lượt trên ảnh hiện tại, chịu ảnh hưởng của cache/tải máy.

**Kiểm tra đã thực hiện:** 5 kiểm thử logic đọc ảnh/tài nguyên/ánh xạ/dự đoán mô hình thật; Streamlit AppTest cho khởi động, ảnh mẫu, so sánh ba phương pháp, xóa kết quả cũ, thiếu mô hình/cấu hình/ảnh mẫu/số liệu. Đã khởi động máy chủ và phân loại ảnh mẫu trong trình duyệt. Thao tác upload qua trình duyệt bị dừng giữa chừng nên chưa xác nhận đầy đủ qua trình duyệt; ảnh lỗi, ảnh xám, alpha và định dạng giả đã được kiểm tra ở lớp đọc ảnh. Không huấn luyện lại hoặc chạy lại tập test để làm giao diện.

Các file chính: `app.py` (giao diện), `code/demo_service.py` (nạp/kiểm tra mô hình, đọc ảnh và gọi pipeline), `.streamlit/config.toml` (theme sáng/xanh dương), `tests/test_demo.py` (kiểm thử). Cấu hình theme được Streamlit tự đọc khi chạy tại thư mục dự án.

Chạy lại kiểm tra demo (không huấn luyện):

```powershell
python -m unittest discover -s tests -p test_demo.py -v
python tests/check_demo_app.py
```

## Phần đã triển khai — Người 1

**Chạy bằng notebook:** mở `code/01_du_lieu_tien_xu_ly.ipynb` trong VS Code/Jupyter, chọn Python kernel và **Run All**. Notebook có code độc lập, giải thích bằng tiếng Việt, không cần import các module `.py`. Dữ liệu đã có nên giữ `DOWNLOAD = False`. Kết quả minh họa của notebook nằm trong `results/notebook/`; các split đã chốt được kiểm tra và dùng lại, không ghi đè. Khi dùng trên máy mới, cài các thư viện ở ô đầu và đổi `DOWNLOAD = True` để tải dữ liệu.

- Tải ba gói train, test và test labels; kiểm tra SHA-256 của Final Training và MD5 test đối chiếu với các bộ đọc dữ liệu tham chiếu; lưu SHA-256 và URL nguồn.
- Bộ đọc cấu trúc gốc, nhãn CSV phân cách `;`, ID lớp 0–42 và track gốc.
- Giải mã toàn bộ ảnh, kiểm tra ảnh thiếu/lỗi, nhãn không hợp lệ, ảnh/annotation không khớp; xuất nhật ký. Không âm thầm bỏ ảnh lỗi.
- Chia train/validation theo track trong từng lớp, lưu CSV cố định, seed và hash.
- Tiền xử lý chung: resize, BGR → HSV/gray/RGB; 10 ảnh mẫu train và biểu đồ thống kê.
- HSV/HOG và kNN đã có ở mục Người 2–3 bên dưới; phần kết hợp/đánh giá chung đã có ở mục Người 4; giao diện xem mục Streamlit ở đầu README.

Kết quả đã chạy: **31.380 train / 7.829 validation / 12.630 test**, đủ 43 lớp trong từng tập, không trùng track train–validation. Đã giải mã 51.839 ảnh, không có lỗi hoặc cảnh báo kích thước. Năm kiểm thử đã đạt.

## Phần đã triển khai — Người 2: HSV + kNN

Mở `code/02_hsv_knn.ipynb`, chọn Python kernel rồi **Run All**. Notebook giải thích histogram, L1, kNN và hiển thị kết quả đã lưu. Giữ `RUN_TRAINING = False` để xem ngay; đổi `True` để chạy lại toàn bộ grid.

Code gồm `features_hsv.py` (histogram HSV 3D chuẩn hóa L1), `train_hsv.py` (12 cấu hình trên validation) và `evaluation.py` (chỉ số dùng chung). Có thể huấn luyện từ thư mục dự án:

```powershell
python code/train_hsv.py
```

Dùng cùng 31.380 train / 7.829 validation của người 1, không đọc test. Grid: bins `(8,4,4)` / `(16,4,4)`, k=3/5/7, Euclidean/Manhattan, bỏ phiếu đồng đều. Ngân sách **12 cấu hình** cần được áp dụng chung cho HOG và kết hợp.

Cấu hình tốt nhất theo Macro-F1: **bins (16,4,4), k=3, Manhattan, 256 chiều**. Kết quả **validation**: Accuracy **17,45%**, Macro-F1 **0,1341**. Đây là baseline màu còn yếu; kết quả test và so sánh đầy đủ xem mục Người 4. L1 chuẩn hóa riêng từng ảnh nên không có scaler học thống kê validation.

- `models/hsv_knn.joblib`: kNN và cấu hình HSV, kích thước, tên lớp.
- `results/hsv/`: đủ 12 kết quả validation, dự đoán từng ảnh, chỉ số theo lớp, summary và hình minh họa.
- `reports/hsv_knn.md`: phần báo cáo màu và phân tích lỗi từ kết quả thực tế.

Người 4 tích hợp bằng `hsv_histogram(image, bins, size)` trong `features_hsv.py`. Khi chạy mô hình đã lưu:

```python
import joblib
from features_hsv import predict_hsv
from preprocessing import read_bgr

bundle = joblib.load('models/hsv_knn.joblib')
label = predict_hsv(bundle, read_bgr('samples/class_00_00000_00000.ppm'))
```

Ví dụ đường dẫn ảnh cần thay bằng ảnh thực tế trong `samples/`. Mô hình được fit trên train, cấu hình chọn bằng validation; chưa gộp validation vào train.

## Phần người 3 — HOG + kNN

Mở `code/03_hog_knn.ipynb`, chọn Python kernel và **Run All**. Notebook có giải thích gradient, cell/block, chuẩn hóa, minh họa HOG, bảng 12 cấu hình, ma trận nhầm lẫn và so sánh validation với HSV. Bảng dùng màu cố định để đọc được trên theme sáng/tối.

- `features_hog.py`: HOG trên ảnh xám 64×64; cell 8/16, block 2×2 cell, stride một cell, 9 hướng không dấu. Chuẩn hóa block L2-Hys rồi L2 toàn vector; không cần fit scaler.
- `train_hog.py`: 12 cấu hình (2 cell × 3 k × 2 khoảng cách), chọn Macro-F1 trên validation, chỉ fit train.
- `plot_validation.py`: tạo ma trận nhầm lẫn HSV/HOG và bảng/biểu đồ so sánh từ CSV đã lưu.
- `models/hog_knn.joblib`: mô hình cùng cấu hình HOG, chuẩn hóa và tên lớp.
- `results/hog/`, `results/comparison/`: kết quả và hình minh họa; `reports/hog_knn.md`: phần báo cáo HOG.

```powershell
python code/train_hog.py
python code/plot_validation.py
```

Trong notebook, giữ `RUN_TRAINING = False` để xem kết quả có sẵn; đổi `True` khi muốn chạy lại. Manhattan trên HOG 1.764 chiều có thể mất vài phút mỗi cấu hình. Không đổi split, không đọc test và không chạy lại HSV. Thời gian trong bảng là một batch validation chia số ảnh, chưa phải latency một ảnh đo lặp.

Đã chạy đủ 12 cấu hình HOG trên toàn train/validation. Cấu hình chọn: **cell 8, k=7, manhattan, 1764 chiều**; Accuracy validation **82.19%**, Macro-F1 **0.7761**. Kết quả test chung xem mục Người 4.

Người 4 dùng `hog_features(image, cell_size, size)` trong `features_hog.py`, với cell đã chọn và cùng preprocessing. `predict_hog(bundle, image)` nạp lại đúng cấu hình mô hình khi chạy demo.

## Phần người 4 — Kết hợp và đánh giá chung

Mở `code/04_hsv_hog_knn.ipynb` rồi **Run All** để xem kết quả đã lưu. Giữ `RUN_TRAINING=False`, `RUN_TEST=False`. Notebook có công thức chuẩn hóa riêng từng nhóm, bảng 12 cấu hình, so sánh validation/test, ma trận và ảnh đúng/sai. Chưa làm UI hoặc demo.

- `features_combined.py`, `train_combined.py`: ghép HSV/HOG và chọn trọng số/k/khoảng cách trên validation.
- `model_features.py`, `evaluate_test.py`: trích đúng đặc trưng, khóa mô hình/split trước khi đọc nhãn test và đánh giá cả ba phương pháp. Đã có kết quả thì dùng lại; hash khác sẽ dừng.
- `plot_validation.py --include-combined`, `plot_test.py`: vẽ từ CSV có sẵn.
- `models/combined_knn.joblib`, `configs/final_experiment.json`: mô hình kết hợp và cấu hình thí nghiệm đã khóa.
- `results/combined/`, `results/test/`, `reports/combined_knn.md`: CSV, hình và phần báo cáo kết hợp.

Ghép kế thừa bins/cell từ HSV/HOG và thử thêm 12 cấu hình: tổng quy trình là 36 cấu hình, không phải ba tối ưu hoàn toàn độc lập. Test giữ nguyên 12.630 ảnh; không fit lại trên validation hoặc điều chỉnh theo điểm test.

| Phương pháp | Accuracy test | Macro-F1 test |
|---|---:|---:|
| HSV + kNN | 16.99% | 0.1500 |
| HOG + kNN | 80.55% | 0.7767 |
| HSV + HOG + kNN | 79.29% | 0.7644 |

Chi tiết cách chạy và giới hạn đo thời gian nằm trong notebook/báo cáo người 4. Kết quả ở các mục Người 2–3 là validation; bảng trên là test sau khóa cấu hình.

## Cài đặt (PowerShell, tại thư mục dự án)

Khuyến nghị Python 3.11–3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Các lệnh bên dưới dùng `python`; nếu dùng môi trường ảo thì thay bằng `.\.venv\Scripts\python.exe` hoặc kích hoạt môi trường trước.

## Tải và chuẩn bị dữ liệu

```powershell
python code/download_data.py
python code/prepare_data.py --validation-fraction 0.2 --seed 42
python -m unittest discover -s tests -v
```

Nguồn benchmark: https://benchmark.ini.rub.de/gtsrb_dataset.html

Mirror công khai: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/

MD5 các gói test theo mã nguồn torchvision (không cần cài PyTorch): https://github.com/pytorch/vision/blob/main/torchvision/datasets/gtsrb.py

Ba gói sử dụng: `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip`, `GTSRB_Final_Test_GT.zip`. Nếu tải thủ công, đặt ZIP vào `data/raw/archives/`, rồi chạy lại script tải để xác thực và giải nén. Script không tải lại ZIP đã có checksum đúng. Cần mạng và dung lượng trống cho cả ZIP lẫn ảnh giải nén.

Cấu trúc thực tế được hỗ trợ:

```text
data/raw/
  archives/
  download_manifest.json
  GTSRB/Final_Training/Images/00000/00000_00000.ppm
  GTSRB/Final_Training/Images/00000/GT-00000.csv
  ... (43 thư mục lớp)
  GTSRB/Final_Test/Images/00000.ppm
  GT-final_test.csv
```

Bộ đọc cũng hỗ trợ cấu trúc `GTSRB/Training/`, nhưng lệnh chuẩn bị yêu cầu đủ 39.209 ảnh Final Training. Không dùng gói Training_fixed cũ 26.640 ảnh cho thí nghiệm chính. Nếu có nhiều bản train trong cùng root, bộ đọc dừng để tránh nhập nhằng. Với cấu trúc train không có CSV, nhãn lấy từ thư mục lớp; tên file gốc có track vẫn bắt buộc. Bản Kaggle có file đổi tên không được tự động chia ngẫu nhiên: cần khôi phục track từ dữ liệu gốc.

SHA-256 Final Training tham chiếu từ bộ đọc stable-datasets: https://galilai-group.github.io/stable-datasets/_modules/stable_datasets/images/gtsrb/

Gói train giai đoạn cũ đã được loại khỏi thư mục làm việc sau khi kiểm tra. Chỉ `data/raw/GTSRB/Final_Training/Images/` là nguồn train chính.

## Quy ước chia tập

- Test chính thức giữ nguyên, kể cả phân bố mất cân bằng.
- Chọn khoảng 20% số track của mỗi lớp làm validation, còn lại train. Mỗi lớp có ít nhất một track trong từng tập; số track chọn làm tròn gần nhất.
- Khóa nhóm là `class_id:track_number`, ví dụ `00001:00000`. Không dùng riêng tiền tố track vì số đó lặp lại giữa các lớp.
- Seed 42, thứ tự đầu vào được sắp xếp trước khi chọn. Tỉ lệ ảnh thực tế có thể khác 20% do chọn nguyên track.
- `splits/train.csv`, `validation.csv`, `test.csv` là nguồn dùng chung cho cả ba phương pháp; không tự chia lại trong từng module.
- Script từ chối ghi đè split sẵn có. Chỉ dùng `--overwrite` khi chủ động làm lại toàn bộ thí nghiệm; không đổi split để có điểm đẹp hơn.
- Không sử dụng test để chọn đặc trưng/tham số. Các thống kê dữ liệu test trong phần này chỉ kiểm tra tính đầy đủ.

## Đầu ra và bàn giao

| Đường dẫn | Nội dung |
|---|---|
| `splits/*.csv` | Đường dẫn ảnh tương đối với data root, class_id, track_id, nguồn chính thức và metadata annotation nếu có |
| `splits/metadata.json` | Seed, tỉ lệ, số ảnh/track, hash các split, cấu hình resize |
| `configs/class_names.csv` | 43 ID với tên Anh và diễn giải tiếng Việt; không gộp lớp gần giống |
| `results/metrics/class_distribution.csv` | Số ảnh từng lớp trong mỗi tập |
| `results/metrics/data_issues.csv` | Lỗi giải mã và cảnh báo metadata (có thể rỗng nếu không có lỗi) |
| `results/metrics/image_dimensions.csv` | Kích thước thực tế sau giải mã, trước resize |
| `results/figures/class_distribution.png` | Biểu đồ phân bố lớp |
| `results/figures/training_samples.png` | 10 ví dụ từ train |
| `results/figures/class_reference.png` | Ảnh tham chiếu 43 lớp kèm tên |
| `samples/` | 10 PPM train nguyên bản và manifest truy nguyên |
| `reports/data_preprocessing.md` | Phần báo cáo dữ liệu và tiền xử lý, kết quả kiểm tra thực tế |

## API dùng chung cho người 2–4

Trong các script đặt trong `code/`:

```python
from data_loader import load_split
from preprocessing import read_bgr, preprocess

train = load_split('splits/train.csv', 'data/raw')
row = train.iloc[0]
bgr = read_bgr(row.image_path)
hsv = preprocess(bgr, color='hsv')   # (64,64,3), uint8
gray = preprocess(bgr, color='gray') # (64,64), uint8
label = int(row.class_id)
```

- Đọc ảnh bằng OpenCV: thứ tự **BGR**. HSV uint8 của OpenCV: H ∈ [0,179], S/V ∈ [0,255]; miền histogram tương ứng `[0,180)`, `[0,256)`, `[0,256)`.
- Resize đầy đủ ảnh được cung cấp về 64×64, không cắt ROI thêm. Dùng `INTER_AREA` khi giảm cả hai chiều, ngược lại `INTER_LINEAR`. Chấp nhận thay đổi tỉ lệ hình học để có cùng kích thước đầu vào ban đầu; phải áp dụng nhất quán và có thể khảo sát trên validation sau.
- `color='rgb'` chỉ dành cho hiển thị Matplotlib. Không đưa ảnh RGB vào hàm vốn yêu cầu BGR.
- Không lọc nhiễu, CLAHE hoặc tăng cường dữ liệu ở cấu hình chung hiện tại; giữ baseline rõ ràng cho so sánh đặc trưng.
- Chuẩn hóa histogram từng ảnh không cần fit; scaler có học thống kê phải fit riêng trên train. Khi ghép HSV/HOG, chuẩn hóa riêng từng nhóm, chọn trọng số bằng validation.
- Module đặc trưng của người 2–3 cần nhận ảnh theo hợp đồng này, trả vector một chiều; người 4 dùng lại cùng module/cấu hình và lưu cùng mô hình.

## Tái lập và giới hạn

Giữ nguyên split CSV và `metadata.json`, lưu cấu hình thí nghiệm và phiên bản thư viện. Dữ liệu thô được bỏ qua trong Git để tránh đưa hàng trăm MB vào mã nguồn. Không coi bảng thống kê dữ liệu là kết quả phân loại. Hiện có kết quả validation cho HSV + kNN và HOG + kNN; đã có kết quả test chung ở mục Người 4 và giao diện Streamlit ở đầu README.
