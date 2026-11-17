# Phương pháp HOG + kNN — phần việc người 3

## Bài toán và dữ liệu

Phân loại 43 lớp biển báo Đức của GTSRB bằng đường nét và hình dạng, dùng cùng kNN như đối chứng HSV. Đầu vào là crop có một biển báo. Dùng nguyên 31.380 ảnh train và 7.829 ảnh validation do người 1 chia theo track; không có track chung giữa hai tập. Không đọc ảnh/nhãn test chính thức trong giai đoạn tìm tham số này.

## HOG và tiền xử lý

Ảnh BGR uint8 được resize toàn crop về 64×64 bằng hàm dùng chung, rồi chuyển ảnh xám. Không cắt thêm ROI, CLAHE hoặc tăng cường dữ liệu. HOG (Histogram of Oriented Gradients) được trích bằng OpenCV `HOGDescriptor`:

1. Tính gradient để mô tả thay đổi sáng và hướng cạnh. Độ lớn gradient cung cấp trọng số đóng góp vào histogram hướng.
2. Chia ảnh thành cell; mỗi cell có histogram 9 hướng không dấu trong khoảng 0–180°.
3. Ghép 2×2 cell thành một block, bước trượt bằng một cell. Các block chồng lấn giúp giữ bố cục cạnh cục bộ.
4. OpenCV chuẩn hóa block bằng L2-Hys (chuẩn L2, cắt ngưỡng 0,2 và chuẩn hóa lại).
5. Ghép các block thành vector, sau đó chuẩn hóa L2 toàn vector. Với ảnh không có gradient, trả vector 0 để tránh chia 0.

Hai bước chuẩn hóa đều thực hiện riêng từng ảnh, không học thống kê của validation/test. Cấu hình này không dùng StandardScaler; kNN chỉ fit trên train. Giữ gamma correction tắt và signed gradient tắt. Số hướng, block, ngưỡng L2-Hys và cách resize được cố định để tập trung khảo sát cell/k/khoảng cách.

| Cell | Block | Stride | Số block | Số chiều |
|---|---|---|---|---:|
| 8×8 | 16×16 | 8×8 | 7×7 | 7×7×4×9 = 1.764 |
| 16×16 | 32×32 | 16×16 | 3×3 | 3×3×4×9 = 324 |

Cell nhỏ giữ chi tiết hơn nhưng tăng số chiều, bộ nhớ và chi phí tìm láng giềng. Cell lớn là cấu hình thô hơn để đo đánh đổi; không mặc định một cấu hình tốt hơn trước khi xem validation. HOG bỏ thông tin màu, vẫn có thể bị ảnh hưởng bởi mờ, ảnh nhỏ, nghiêng, crop lệch và nền có nhiều cạnh.

## Thiết kế thí nghiệm

Grid gồm 2 cell size × 3 giá trị k (3, 5, 7) × 2 khoảng cách (Euclidean, Manhattan) = **12 cấu hình**, bằng ngân sách HSV. Giữ bỏ phiếu đồng đều (`uniform`), brute force, `n_jobs=4` và giới hạn thư viện tính toán 4 thread. Cùng số cấu hình không có nghĩa cùng chi phí tính toán, vì vector HOG và HSV khác số chiều.

Tiêu chí chọn là Macro-F1 cao nhất trên validation. Nếu hòa, lần lượt xét Accuracy cao hơn, số chiều ít hơn, k nhỏ hơn và thứ tự khoảng cách trong grid. Không chọn theo test hoặc thời gian. Mô hình cuối của bước này vẫn fit chỉ trên train, không gộp validation. Hash các danh sách và phiên bản môi trường được lưu trong summary.

Thời gian trích đặc trưng train/validation được đo riêng và dùng lại khi thay k/khoảng cách của cùng cấu hình HOG. Thời gian dự đoán là một batch toàn validation chia số ảnh; toàn luồng gồm đọc đĩa, resize, gray, HOG, chuẩn hóa và kNN, không gồm import, vẽ hình hoặc ghi file. Một lượt đo có thể chịu ảnh hưởng cache/tải máy; chưa coi là latency một ảnh đo lặp. So sánh thời gian cuối cùng cần chạy lại trên cùng máy, batch và điều kiện.

## Kết quả validation đã chạy

Đã hoàn tất đủ 12 cấu hình. Cấu hình chọn theo Macro-F1: **cell 8×8, block 16×16, stride 8, 9 hướng, k=7, manhattan**, 1,764 chiều.

| Chỉ số | Giá trị |
|---|---:|
| Ảnh đúng | 6,435 / 7,829 |
| Accuracy validation | 0.821944 (82.19%) |
| Macro-F1 validation | 0.776102 |
| Trích HOG train | 29.605 giây |
| Fit kNN | 0.034 giây |
| Trích HOG validation | 7.583 giây |
| Dự đoán validation | 141.851 giây |
| Thời gian phân loại trung bình | 18.119 ms/ảnh |
| Thời gian toàn luồng trung bình | 19.087 ms/ảnh |

Kết quả so sánh cấu hình được chọn của mỗi phương pháp trên **cùng validation**:

| Phương pháp | Accuracy | Macro-F1 | Số chiều | Phân loại (ms/ảnh) |
|---|---:|---:|---:|---:|
| HSV + kNN | 0.174479 | 0.134065 | 256 | 2.478 |
| HOG + kNN | 0.821944 | 0.776102 | 1764 | 18.119 |

Đây là so sánh validation sau chọn tham số trong grid đã công bố, không phải điểm test. Không suy rộng thứ hạng cho ảnh bất kỳ hoặc kết luận phương pháp kết hợp khi chưa thực nghiệm. Các ma trận nhầm lẫn PNG chuẩn hóa theo hàng; CSV lưu số đếm. Chỉ số từng lớp và đủ 12 cấu hình được công bố để tránh chỉ trình bày số trung bình.

### Một số cặp nhầm lẫn có hướng

| Lớp thật → dự đoán | Tên lớp | Số ảnh |
|---|---|---:|
| 1 → 5 | Giới hạn tốc độ 30 km/h → Giới hạn tốc độ 80 km/h | 59 |
| 2 → 5 | Giới hạn tốc độ 50 km/h → Giới hạn tốc độ 80 km/h | 59 |
| 9 → 10 | Cấm vượt → Cấm vượt đối với xe trên 3.5 tấn | 48 |
| 1 → 2 | Giới hạn tốc độ 30 km/h → Giới hạn tốc độ 50 km/h | 39 |
| 5 → 8 | Giới hạn tốc độ 80 km/h → Giới hạn tốc độ 120 km/h | 36 |
| 2 → 1 | Giới hạn tốc độ 50 km/h → Giới hạn tốc độ 30 km/h | 36 |
| 3 → 5 | Giới hạn tốc độ 60 km/h → Giới hạn tốc độ 80 km/h | 35 |
| 8 → 7 | Giới hạn tốc độ 120 km/h → Giới hạn tốc độ 100 km/h | 30 |

Các cặp lỗi là dữ liệu để xem lại hình dạng gần giống, ký hiệu mảnh, ảnh mờ/tối/nghiêng và nền. Bảng nhầm lẫn không tự chứng minh nguyên nhân; notebook có ví dụ đúng/sai để phân tích định tính. Minh họa ưu tiên lớp khác nhau, không chọn dựa trên việc ảnh đẹp. Cần tiếp tục phân tích khi đánh giá test sau khi chốt cả ba phương pháp.

## Sản phẩm bàn giao

- `code/features_hog.py`: cấu hình HOG, `hog_features` cho người 4 kết hợp với HSV và `predict_hog` cho demo.
- `code/train_hog.py`: grid validation và lưu mô hình, không đọc test.
- `code/plot_validation.py`: ma trận nhầm lẫn của HSV/HOG, bảng và biểu đồ so sánh validation từ CSV; kiểm tra hai phương pháp dùng cùng hash split.
- `code/03_hog_knn.ipynb`: giải thích từng bước, minh họa ảnh xám/gradient, kết quả và ảnh đúng/sai. Bảng dùng màu cố định, có cuộn ngang.
- `models/hog_knn.joblib`: kNN, kích thước, cell, cấu hình HOG, quy tắc chuẩn hóa và ánh xạ lớp.
- `results/hog/`: CSV đủ 12 cấu hình, dự đoán từng ảnh, chỉ số từng lớp, summary và hình minh họa.
- `results/comparison/`: CSV và biểu đồ so sánh HSV–HOG trên validation.

Chạy `python code/train_hog.py`, sau đó `python code/plot_validation.py` từ thư mục dự án. Hoặc mở notebook, giữ `RUN_TRAINING = False` để xem kết quả có sẵn; bật `True` nếu muốn chạy lại grid. Chạy lại thay mô hình/kết quả HOG, không thay split hoặc mô hình HSV. Người 4 cần dùng đúng `cell_size` được chọn và chuẩn hóa từng nhóm khi kết hợp, không tự thay tham số khi demo.

## Tài liệu và khai báo AI

- OpenCV HOGDescriptor: https://docs.opencv.org/4.13.0/d5/d33/structcv_1_1HOGDescriptor.html
- scikit-learn kNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- Theo thông tin người dùng, HOG được nhắc trong chương 8. Cần tìm hiểu thêm gradient, cell, block, chuẩn hóa và đối chiếu với slide học phần trước khi bảo vệ.

Codex hỗ trợ viết mã, chạy thực nghiệm/kiểm tra và soạn bản nháp. Thành viên phụ trách cần hiểu, kiểm chứng, chỉnh sửa và chịu trách nhiệm về nội dung nộp; cập nhật phân công và khai báo AI trong báo cáo chung.
