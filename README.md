# Hướng dẫn chạy chương trình

## 1. Môi trường

- Windows 11, Python 3.13.2.
- Thư viện đã sử dụng: NumPy 2.3.1, OpenCV 4.14.0, scikit-learn 1.9.0, pandas 2.3.1, Matplotlib 3.10.7, joblib 1.5.2, Streamlit 1.49.1, Pillow 11.3.0.
- Các thư viện cần cài được khai báo trong `requirements.txt`. Khi dùng mô hình `.joblib` có sẵn, nên dùng cùng phiên bản scikit-learn đã lưu.

## 2. Cài đặt

Mở PowerShell tại thư mục gốc dự án, nơi có `app.py` và `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Để chạy demo, cần ba file mô hình trong `models/`: `hsv_knn.joblib`, `hog_knn.joblib`, `combined_knn.joblib`; file tên lớp `configs/class_names.csv`; ảnh mẫu trong `samples/`.

## 3. Lệnh chạy

### Chạy giao diện demo

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở địa chỉ **Local URL** được in trong terminal. Chọn phương pháp, chọn ảnh mẫu hoặc tải ảnh JPG/JPEG/PNG/PPM đã cắt chứa một biển báo, rồi nhấn **Phân loại**. Nhấn **Ctrl+C** để dừng chương trình. Demo dùng mô hình có sẵn, không cần huấn luyện lại.

### Tải dữ liệu

```powershell
.\.venv\Scripts\python.exe code/download_data.py
```

Chỉ cần tải toàn bộ dữ liệu khi muốn dùng bộ chọn ảnh validation/test hoặc chạy thí nghiệm. Ảnh mẫu và ảnh upload dùng được mà không cần toàn bộ GTSRB.

### Chuẩn bị dữ liệu khi chưa có split

```powershell
.\.venv\Scripts\python.exe code/prepare_data.py --validation-fraction 0.2 --seed 42
```

Nếu đã có `splits/train.csv`, `splits/validation.csv`, `splits/test.csv` thì bỏ qua lệnh này và dùng lại các file có sẵn. Script từ chối ghi đè split đã tồn tại.

### Tạo biểu đồ từ kết quả đã lưu

```powershell
.\.venv\Scripts\python.exe code/plot_validation.py --include-combined
.\.venv\Scripts\python.exe code/plot_test.py
```

Hai lệnh này không huấn luyện hoặc dự đoán lại.

## 4. Đầu ra mong đợi


| Lệnh                                   | Đầu ra                                                                                                                                                                                                            |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `streamlit run app.py`                  | Giao diện hiển thị ảnh đầu vào, tên lớp dự đoán, Class ID, phương pháp và thời gian xử lý. Có chức năng xem quá trình xử lý, so sánh ba phương pháp và xem kết quả test đã lưu. |
| `download_data.py`                      | Ba ZIP trong `data/raw/archives/`, ảnh và nhãn đã giải nén trong `data/raw/`, file `data/raw/download_manifest.json`.                                                                                        |
| `prepare_data.py`                       | Manifest và metadata trong `splits/`; thống kê trong `results/metrics/`; hình trong `results/figures/`; 10 ảnh mẫu và `manifest.csv` trong `samples/`. Số ảnh train/validation/test: 31.380/7.829/12.630.  |
| `plot_validation.py --include-combined` | Bảng và biểu đồ so sánh validation trong `results/comparison/`, ma trận của từng phương pháp trong thư mục kết quả tương ứng.                                                                    |
| `plot_test.py`                          | Biểu đồ `results/test/comparison.png`, ma trận nhầm lẫn và ảnh đúng/sai trong `results/test/hsv/`, `results/test/hog/`, `results/test/combined/`.                                                         |

## 5. Đường dẫn tải dữ liệu

Bộ dữ liệu: **GTSRB**, gồm 43 lớp biển báo.

- [Trang giới thiệu GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html).
- [Kho tải dữ liệu](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/).
- [GTSRB trên Hugging Face](https://huggingface.co/datasets/bazyl/GTSRB).
