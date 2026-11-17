# Kết hợp HSV + HOG và tích hợp đánh giá — phần việc người 4

## Phạm vi và dữ liệu

Ghép đặc trưng màu và hình dạng, chọn trọng số/k/khoảng cách bằng validation, lưu mô hình và so sánh định lượng cả ba phương pháp. Chưa xây dựng giao diện hoặc chương trình demo.

Giữ nguyên split theo track của người 1: 31.380 train, 7.829 validation và 12.630 test chính thức GTSRB, mỗi tập đủ 43 lớp. Các mô hình chỉ được fit trên train. Không gộp validation để huấn luyện lại trước test, giúp cả ba phương pháp có cùng số mẫu huấn luyện. Nhãn test chỉ dùng để tính chỉ số sau khi cấu hình/mô hình đã khóa.

## Phương pháp kết hợp

Từ cùng ảnh BGR, dùng lại `hsv_histogram` và `hog_features` của người 2–3, cùng resize 64×64. Kế thừa bins HSV `(16,4,4)` và HOG cell 8 từ kết quả validation của các phần trước. HSV có 256 chiều; HOG có 1.764 chiều. Không cắt thêm ROI, không CLAHE và không tăng cường dữ liệu.

HSV ban đầu chuẩn hóa L1 theo tổng pixel; HOG có L2-Hys từng block và L2 toàn vector. Khi ghép, chuẩn hóa L2 lại riêng từng nhóm để không nối hai nhóm có thang đo khác nhau, rồi tính:

`x = [alpha × HSV_L2, (1-alpha) × HOG_L2]`.

Vector cuối có **2.020 chiều**. Nhóm vector 0 giữ nguyên để tránh chia 0. Chuẩn hóa này độc lập trên từng ảnh, không fit thống kê từ validation/test. Hai nhóm luôn được lấy theo cùng dòng ảnh; không ghép theo hai thứ tự khác nhau.

Alpha là hệ số màu, không phải xác suất hoặc tỉ lệ chính xác về đóng góp của màu trong quyết định. Euclidean bình phương dùng các hệ số bình phương, Manhattan dùng hệ số tuyến tính; chuẩn L2 bằng nhau không bảo đảm chuẩn L1 bằng nhau. Vì vậy phải chọn hệ số bằng validation, không mặc định alpha=0,5 là cân bằng tối ưu.

Đọc hàng loạt ưu tiên ZIP GTSRB gốc có sẵn để tránh chi phí mở nhiều tệp nhỏ; nếu không có ZIP sẽ dùng thư mục giải nén. SHA-256 ZIP đã được đối chiếu với bản tải và kiểm tra mẫu cho thấy pixel giải mã giống hệt ảnh giải nén. Nguồn đọc này không thay đổi tiền xử lý hoặc nhãn; hash ZIP được ghi khi khóa thí nghiệm.

## Chọn tham số và ngân sách

Thử alpha 0,25 và 0,5; k=3/5/7; khoảng cách Euclidean/Manhattan: **12 cấu hình kết hợp mới**. Hai alpha được đặt trước khi chạy, gồm hệ số đều và ưu tiên hình dạng. Bỏ phiếu đồng đều, brute force và 4 thread như hai đối chứng. Tiêu chí chính là Macro-F1; hòa thì Accuracy, số chiều, k nhỏ hơn và thứ tự grid. Không tối ưu lại bins/cell trong grid ghép.

**Công bố giới hạn so sánh:** HSV có 12 cấu hình, HOG có 12 cấu hình và ghép có 12 cấu hình mới. Phương pháp ghép kế thừa lựa chọn bins/cell từ 24 thử nghiệm trước, nên đây là tìm kiếm tuần tự có điều kiện, không phải ba quy trình tối ưu độc lập với cùng tổng chi phí. Tổng thí nghiệm là 36 cấu hình. Tập test được giữ kín cho đến khi chốt mọi cấu hình, nên không dùng kết quả test để chọn trọng số hoặc thay lựa chọn.

Lưu ý lượt chạy validation có gián đoạn kênh điều khiển; một số số đo thời gian kéo dài bất thường. Các thời gian validation chỉ để truy nguyên, không dùng xếp hạng hiệu năng hoặc chọn mô hình. Việc chọn cấu hình chỉ dựa trên Macro-F1/Accuracy.

Khôi phục sau gián đoạn: giữ nguyên 11 điểm số đã hoàn tất, đối chiếu hash split và thứ tự nhãn trước khi tái dùng đặc trưng train lưu trong hai mô hình cơ sở; chạy cấu hình thứ 12 còn thiếu. Mô hình được chọn được fit lại chỉ trên train và tính lại validation để xác nhận đúng điểm đã ghi. Thời gian trích train của cấu hình chạy tiếp không đo lại nên để trống/null, không tự điền số liệu.

## Khóa cấu hình và đánh giá chung

`evaluate_test.py` lưu `configs/final_experiment.json` trước khi đọc nhãn test. Tệp này chứa tham số chọn bằng validation, SHA-256 của từng mô hình và các split. Bộ đánh giá đối chiếu k/khoảng cách/số chiều/số mẫu huấn luyện của mô hình với summary, rồi chạy từng phương pháp trên cùng test.

Xuất Accuracy, Macro-F1 (đủ 43 lớp), Precision/Recall/F1 theo lớp, nhãn dự đoán từng ảnh và ma trận nhầm lẫn số đếm. PNG ma trận chuẩn hóa theo hàng. Có ảnh đúng và sai, ưu tiên các lớp khác nhau, không chỉ chọn ảnh đẹp. Phân tích lỗi sau test chỉ phục vụ thảo luận, không điều chỉnh mô hình trong cùng thí nghiệm.

Thời gian đo một batch đầy đủ, chia số ảnh: tách trích đặc trưng và phân loại; toàn luồng gồm đọc đĩa, resize, chuyển màu/xám, trích HSV/HOG, chuẩn hóa và ghép. Không gồm import, vẽ ảnh, ghi file hoặc thao tác người dùng. Các phương pháp chạy cùng máy và 4 thread; cache/tải máy vẫn ảnh hưởng. Đây không phải latency một ảnh đo lặp. Phần ghép tái sử dụng đặc trưng gốc khi chỉ đổi alpha/k/khoảng cách. Khi đánh giá test, ảnh được đọc một lần, HSV/HOG được tính một lần và chia sẻ cho ba mô hình. Thời gian trích đặc trưng của từng phương pháp được cộng từ các công đoạn tương ứng, bao gồm thời gian đọc ảnh chung; thời gian phân loại được đo riêng bằng predict của mỗi kNN. Việc chia sẻ này không thay vector đầu vào của mô hình, đã có kiểm thử đối chiếu với đường trích độc lập.

Đã có kết quả test và hash không đổi thì dùng lại, không chạy lại để lựa chọn kết quả đẹp hơn. Có thể tiếp tục nếu một lượt bị ngắt sau khi một phương pháp đã hoàn tất, nhưng phải khớp hash thí nghiệm.

## Kết quả thực nghiệm đã chạy

Đã chạy đủ 12 cấu hình kết hợp. Cấu hình chọn theo Macro-F1 validation: alpha HSV **0.25**, HOG **0.75**, k=**7**, khoảng cách **manhattan**; bins `(16,4,4)`, HOG cell 8, tổng 2.020 chiều.

### Validation — chỉ dùng chọn tham số

| Phương pháp | Accuracy | Macro-F1 | Số chiều |
|---|---:|---:|---:|
| HSV + kNN | 17.45% | 0.134065 | 256 |
| HOG + kNN | 82.19% | 0.776102 | 1764 |
| HSV + HOG + kNN | 80.27% | 0.762316 | 2020 |

### Test chính thức — sau khi khóa cả ba cấu hình

Cả ba phương pháp được đánh giá trên **12.630 ảnh test**, không chỉnh lại theo điểm số này.

| Phương pháp | Accuracy | Macro-F1 | Phân loại (ms/ảnh) | Toàn luồng (ms/ảnh) |
|---|---:|---:|---:|---:|
| HSV + kNN | 16.99% | 0.149986 | 4.692 | 5.282 |
| HOG + kNN | 80.55% | 0.776679 | 57.057 | 57.810 |
| HSV + HOG + kNN | 79.29% | 0.764445 | 79.361 | 80.469 |

### Ghép thay đổi những dự đoán nào so với HOG?

Trên cùng test: có **429 ảnh HOG sai nhưng ghép đúng** và **588 ảnh HOG đúng nhưng ghép sai**. Đây là so sánh theo từng ảnh từ CSV, không suy ra rằng màu luôn giúp hoặc luôn gây hại. Chênh lệch số đúng là -159 ảnh trong thí nghiệm này. Xem `results/combined/test_paired_with_hog.csv` để truy nguyên.

Ghép có thể bổ sung tín hiệu màu nhưng cũng đưa vào ảnh hưởng của nền/ánh sáng, hoặc làm thay đổi láng giềng của các biển cùng màu. Cần xem ảnh và cặp lớp nhầm lẫn trước khi giải thích; số liệu chưa tự chứng minh nguyên nhân. Kết luận chỉ giới hạn ở dữ liệu/cấu hình đã công bố, không khẳng định cho ảnh bất kỳ hoặc biển Việt Nam. Không thay trọng số sau khi thấy điểm test.

Mỗi phương pháp có đủ dự đoán từng ảnh, chỉ số 43 lớp và ma trận số đếm; các biểu đồ và ví dụ đúng/sai được lưu trong `results/test/`. Thời gian là trung bình một batch đầy đủ; điều kiện/cache khác nhau có thể ảnh hưởng và không được dùng làm số đo latency một ảnh lặp nhiều lần.

Đã đạt 18 kiểm thử; đối chiếu chỉ số từ CSV, nhãn train, mô hình nạp lại, tổng ma trận, hash mô hình/split và notebook đã thực thi. Chưa có UI hoặc demo.

## Bàn giao và cách chạy

- `features_combined.py`: chuẩn hóa từng nhóm, ghép có trọng số và trích từ cùng ảnh.
- `dataset_images.py`: đọc PPM từ ZIP gốc hoặc thư mục, không đổi pixel.
- `train_combined.py`: 12 cấu hình validation; lưu `models/combined_knn.joblib` với toàn bộ cấu hình và ánh xạ lớp.
- `model_features.py`: dùng đúng hàm đặc trưng tương ứng mô hình khi đánh giá.
- `evaluate_test.py`: khóa cấu hình và đánh giá ba mô hình đã lưu.
- `plot_validation.py --include-combined`: bảng/biểu đồ validation ba phương pháp.
- `plot_test.py`: bảng/biểu đồ test, ma trận và ảnh đúng/sai.
- `code/04_hsv_hog_knn.ipynb`: giải thích từng bước và tổng hợp kết quả. Mã hiển thị bằng văn bản, bảng có màu cố định và cuộn ngang.
- `results/combined/`: validation của ghép; `results/test/`: đánh giá chính thức; `results/comparison/`: bảng/biểu đồ validation chung.

Từ thư mục dự án, chạy theo thứ tự:

```powershell
python code/train_combined.py
python code/plot_validation.py --include-combined
python code/evaluate_test.py
python code/plot_test.py
```

Không chạy lại bước huấn luyện sau khi xem test để thay kết luận. Trong notebook giữ `RUN_TRAINING=False` và `RUN_TEST=False` để xem output đã có. Thí nghiệm mới cần thư mục và kết quả riêng để không trộn với test đã báo cáo.

## Kiến thức và khai báo AI

Vận dụng hệ màu HSV (chương 4), biểu diễn đặc trưng, kNN, chuẩn hóa và đánh giá phân loại (chương 8 theo thông tin đã thống nhất). HOG cần giải thích thêm gradient, cell, block và L2-Hys khi bảo vệ. Tham khảo các tài liệu OpenCV/scikit-learn đã ghi trong phần báo cáo HSV/HOG.

Codex hỗ trợ code, chạy kiểm tra/thực nghiệm và soạn bản nháp. Nhóm cần hiểu, kiểm chứng và chịu trách nhiệm với nội dung nộp; cập nhật tên người phụ trách và khai báo AI trong báo cáo chung.
