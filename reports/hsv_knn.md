# Phương pháp màu HSV + kNN — phần việc người 2

## Mục tiêu và dữ liệu

Đây là phương pháp đối chứng chỉ dựa trên phân bố màu để phân loại 43 lớp biển báo Đức trong GTSRB. Sử dụng nguyên danh sách chia tập của người 1: 31.380 ảnh train và 7.829 ảnh validation, không trùng track. Không đọc hoặc đánh giá test chính thức trong giai đoạn này. Hash của hai danh sách được lưu trong `results/hsv/summary.json` để truy nguyên.

## Biểu diễn đặc trưng

Đọc ảnh BGR uint8, resize toàn bộ crop về 64×64 bằng hàm tiền xử lý chung, sau đó chuyển HSV. Không cắt thêm ROI, không dùng CLAHE và không tăng cường dữ liệu.

Dùng `cv2.calcHist` tính histogram **3 chiều chung** trên H, S, V. Miền giá trị của OpenCV là H `[0,180)`, S/V `[0,256)`; các cận trên mở bảo đảm vẫn đếm pixel S/V=255. Với số bin `(bH,bS,bV)`, histogram có `bH × bS × bV` chiều. Histogram được trải phẳng và chia cho tổng số pixel:

`x[i] = count[i] / sum(count)`.

Đây là chuẩn hóa L1 riêng từng ảnh, không học thống kê từ các tập dữ liệu. Không dùng StandardScaler trong cấu hình này; không có scaler cần fit trên validation. kNN chỉ được fit bằng vector và nhãn train. Khi tích hợp HSV/HOG, người 4 cần dùng lại đúng hàm và chuẩn hóa/cân bằng từng nhóm đặc trưng trước khi ghép.

Histogram toàn cục mô tả tỉ lệ màu nhưng không lưu vị trí pixel. Hai biển cùng màu, khác số hoặc hướng mũi tên có thể có vector gần nhau. Viền nền, độ sáng và cân bằng trắng của ảnh cũng ảnh hưởng phân bố màu. Đây là những hạn chế cần kiểm tra khi đối chiếu HOG, không tự suy ra HOG tốt hơn khi chưa có kết quả.

## Thiết kế tìm tham số

| Thành phần | Giá trị |
|---|---|
| Bins HSV | `(8,4,4)` → 128 chiều; `(16,4,4)` → 256 chiều |
| k | 3, 5, 7 |
| Khoảng cách | Euclidean, Manhattan |
| Bỏ phiếu | Đồng đều (`uniform`), cố định |
| Tìm láng giềng | Brute force |
| Resize | 64×64, cố định |

Tổng **12 cấu hình**, chạy trên toàn bộ train/validation. Hai số bin H khảo sát mức chi tiết sắc màu trong khi giữ S/V cố định; các k nhỏ hạn chế việc gộp quá nhiều láng giềng của các lớp cùng màu. Đây là một grid có giới hạn, không khẳng định tối ưu toàn cục. Nhóm cần áp dụng cùng ngân sách 12 cấu hình cho HOG và phương pháp kết hợp trong so sánh chính.

Tiêu chí chọn chính là **Macro-F1 trên validation** vì GTSRB mất cân bằng lớp. Nếu hòa, xét Accuracy cao hơn, số chiều ít hơn, k nhỏ hơn và thứ tự khoảng cách trong grid. Không dùng thời gian hoặc test để chọn cấu hình. Sau khi chọn, mô hình vẫn chỉ chứa train; không gộp validation để huấn luyện lại ở bước này.

## Kết quả validation đã chạy

| Cấu hình được chọn | Giá trị |
|---|---:|
| Bins | `(16,4,4)` |
| Số chiều | 256 |
| k | 3 |
| Khoảng cách | Manhattan |
| Số ảnh dự đoán đúng | 1.366 / 7.829 |
| Accuracy | 0,1744795 (17,45%) |
| Macro-F1 | 0,1340654 |
| Thời gian trích đặc trưng train | 11,314 giây |
| Thời gian fit kNN | 0,00675 giây |
| Thời gian trích đặc trưng validation | 3,168 giây |
| Thời gian dự đoán validation | 19,401 giây |
| Thời gian phân loại trung bình | 2,478 ms/ảnh |
| Thời gian toàn luồng trung bình | 2,883 ms/ảnh |

Cấu hình k=5, Manhattan, cùng bins có Accuracy hơi cao hơn (17,51%) nhưng Macro-F1 thấp hơn (0,1330), nên không được chọn theo tiêu chí đã định. Toàn bộ 12 dòng kết quả nằm trong `validation_results.csv`, không chỉ báo cáo dòng tốt nhất.

Thời gian được đo một lượt theo batch toàn validation trên Windows, Python 3.13.2, NumPy 2.3.1, scikit-learn 1.9.0, giới hạn thư viện tính toán 4 thread và kNN `n_jobs=4`. kNN gần như chỉ lưu mẫu khi fit nên thời gian fit rất ngắn; chi phí lớn nằm ở tìm láng giềng khi dự đoán. Thời gian toàn luồng gồm đọc đĩa, resize, chuyển HSV, histogram/chuẩn hóa L1 và kNN; không bao gồm import, vẽ hình, ghi file hoặc thao tác người dùng. Trích đặc trưng dùng lại giữa các k/khoảng cách của cùng bộ bin. Đây chưa phải phép đo latency một ảnh lặp nhiều lần; cache đĩa và lượt chạy đầu ảnh hưởng thời gian. Chưa dùng các thời gian này để kết luận phương pháp nào nhanh hơn HOG hoặc kết hợp.

## Phân tích lỗi ban đầu

Kết quả còn thấp và chưa phù hợp để xem là bộ nhận dạng đáng tin cậy. Một số nhầm lẫn có hướng trên validation:

| Lớp thật → lớp dự đoán | Số ảnh |
|---|---:|
| 4 (70 km/h) → 2 (50 km/h) | 86 |
| 38 (đi về bên phải) → 35 (đi thẳng) | 83 |
| 8 (120 km/h) → 10 (cấm vượt xe trên 3,5 tấn) | 75 |
| 35 (đi thẳng) → 38 (đi về bên phải) | 72 |
| 5 (80 km/h) → 4 (70 km/h) | 67 |

Các cặp biển cùng họ màu nhưng khác chữ số hoặc hướng là minh họa phù hợp với hạn chế bỏ qua bố cục của histogram. Tuy nhiên, bảng nhầm lẫn này chưa chứng minh nguyên nhân duy nhất là màu; cần xem cả ảnh mờ, tối, nền và tỉ lệ biển trong crop. Notebook hiển thị ba ảnh đúng và ba ảnh sai đầu tiên theo CSV, không chỉ ảnh đẹp. Chưa có kết quả HOG hoặc kết hợp để kết luận so sánh.

## Tệp bàn giao và cách chạy

- `code/features_hsv.py`: trích đặc trưng và dự đoán theo cấu hình đã lưu.
- `code/train_hsv.py`: grid validation, lưu mô hình và CSV.
- `code/evaluation.py`: Accuracy, Macro-F1 với đủ 43 lớp và báo cáo Precision/Recall/F1 từng lớp dùng chung.
- `code/02_hsv_knn.ipynb`: giải thích, bảng kết quả và minh họa.
- `models/hsv_knn.joblib`: kNN, bins, size, quy tắc chuẩn hóa và ánh xạ lớp.
- `results/hsv/validation_results.csv`: đủ 12 cấu hình, số chiều và thời gian.
- `results/hsv/validation_predictions.csv`: nhãn thật/dự đoán từng ảnh validation.
- `results/hsv/validation_per_class.csv`: chỉ số 43 lớp.
- `results/hsv/summary.json`: cấu hình chọn, điểm số, hash split, điều kiện đo.

Chạy từ thư mục dự án: `python code/train_hsv.py`. Mở notebook rồi Run All để xem kết quả đã lưu; bật `RUN_TRAINING = True` nếu muốn chạy lại grid. Các kết quả HSV sẽ được thay thế khi chạy lại, split không đổi. Mô hình được lưu cùng cấu hình trích đặc trưng thay cho việc phải đoán tham số khi chạy demo.

## Tài liệu và khai báo AI

- OpenCV histogram: https://docs.opencv.org/4.x/d8/dbc/tutorial_histogram_calculation.html
- scikit-learn kNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- Kiến thức học phần đã thống nhất: RGB/HSV (chương 4), lược đồ màu, kNN và đánh giá phân loại (chương 8); đối chiếu với slide khi hoàn thiện trích dẫn học phần.

Codex hỗ trợ viết mã, chạy kiểm tra/thực nghiệm và soạn bản nháp. Thành viên phụ trách cần hiểu, kiểm chứng, chỉnh sửa và chịu trách nhiệm với nội dung nộp; bổ sung phân công và khai báo AI trong báo cáo chung.
