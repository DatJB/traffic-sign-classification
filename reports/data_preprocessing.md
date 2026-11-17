# Dữ liệu và tiền xử lý — phần việc người 1

## Bài toán và nguồn dữ liệu

Đồ án phân loại ảnh đã cắt chứa một biển báo vào một trong 43 lớp của GTSRB (German Traffic Sign Recognition Benchmark). Đây là các lớp biển báo Đức, không phải bộ nhận dạng biển báo Việt Nam. Chương trình không giải quyết phát hiện vị trí biển báo trong ảnh đường phố.

Nguồn benchmark: https://benchmark.ini.rub.de/gtsrb_dataset.html. Dữ liệu được tải từ mirror công khai ERDA được các bộ đọc GTSRB sử dụng. Ba tệp là `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip` và `GTSRB_Final_Test_GT.zip`. Gói test GT riêng chứa nhãn lớp; không suy diễn nhãn từ tên file test. URL và checksum thực tế được lưu trong `data/raw/download_manifest.json`.

Gói `Training_fixed` trong torchvision chứa tập train của giai đoạn trước, có 26.640 ảnh; vì vậy không dùng gói đó cho thí nghiệm chính. Lệnh chuẩn bị dữ liệu yêu cầu Final Training đủ 39.209 ảnh và test chính thức đủ 12.630 ảnh trước khi tiếp tục. Số lượng thực tế sau kiểm tra được ghi trong `splits/metadata.json`.

## Cấu trúc, kiểm tra và ánh xạ nhãn

Train được tổ chức thành thư mục lớp `00000` đến `00042`. CSV nhãn sử dụng dấu chấm phẩy. Bộ đọc đối chiếu nhãn trong CSV với thư mục lớp, xác nhận ID thuộc 0–42, kiểm tra ảnh thiếu/ảnh ngoài annotation, đường dẫn trùng và sự hiện diện của cả 43 lớp trong mỗi tập chính thức. Các ảnh đều được giải mã để phát hiện file hỏng; kích thước thực tế lưu trong `image_dimensions.csv`.

Không âm thầm bỏ ảnh lỗi vì việc đó có thể thay đổi tập đánh giá. Nếu giải mã thất bại, quá trình dừng trước khi xuất split. Metadata kích thước không khớp ảnh được ghi thành cảnh báo và dùng kích thước giải mã làm căn cứ cho resize; không coi lỗi metadata đó là nhãn sai. Nhật ký nằm ở `results/metrics/data_issues.csv`.

Ánh xạ ID và tên Anh/Việt nằm ở `configs/class_names.csv`. Tên Việt là diễn giải để trình bày, không khẳng định tương đương pháp lý với biển báo Việt Nam. Không gộp các biển cùng hình dạng nhưng khác ký hiệu; ví dụ các giới hạn tốc độ là những lớp riêng. `class_reference.png` dùng ảnh train minh họa đủ 43 ID để đối chiếu tên và ảnh.

## Chia tập tránh rò rỉ dữ liệu

Giữ nguyên test chính thức cho đánh giá cuối. Train chính thức được chia thành train và validation theo track của cùng một biển báo thực tế. Tên train gốc có dạng `TTTTT_FFFFF.ppm`, trong đó tiền tố là số track trong lớp. Do số track lặp lại giữa các lớp, khóa nhóm là cặp `(class_id, track_number)`.

Với seed 42, sắp xếp danh sách rồi chọn ngẫu nhiên khoảng 20% số track của từng lớp cho validation. Số track chọn được làm tròn gần nhất, giới hạn để mỗi lớp có ít nhất một track ở cả hai tập. Đây là cách giữ tương đối phân bố lớp, không làm cân bằng nhân tạo số ảnh các lớp. Tỉ lệ ảnh validation thực tế có thể lệch 20% do phải chọn nguyên track. Không được tự chia ngẫu nhiên theo ảnh nếu bản dữ liệu đã mất thông tin track; phải khôi phục từ bản gốc trước.

Danh sách `splits/train.csv`, `validation.csv`, `test.csv` được lưu cố định và dùng chung cho HSV + kNN, HOG + kNN và HSV + HOG + kNN. Bộ chia kiểm tra không trùng track giữa train/validation, không trùng đường dẫn giữa ba tập và không thất thoát ảnh. `metadata.json` lưu seed, số lượng và SHA-256 của từng manifest. Script từ chối ghi đè nếu không có cờ chủ động `--overwrite`.

Test không dùng để chọn số bin, cấu hình HOG, k, khoảng cách hoặc trọng số kết hợp. Nếu dùng tăng cường dữ liệu thì chỉ áp dụng train sau khi đã chia.

## Số lượng chia tập thực tế

Với seed 42 và tỉ lệ validation yêu cầu 0.2, danh sách được xác nhận từ dữ liệu Final Training:

| Tập | Số ảnh | Số track | Số lớp |
|---|---:|---:|---:|
| Train | 31.380 | 1.046 | 43 |
| Validation | 7.829 | 261 | 43 |
| Test chính thức | 12.630 | Không suy ra từ tên test phẳng | 43 |

Tổng train + validation là 39.209 ảnh; tổng toàn bộ dữ liệu là 51.839 ảnh. Không có track trùng giữa train và validation. Đây là số lượng dữ liệu, không phải kết quả phân loại. Tỉ lệ validation thực tế là 19,967% của train chính thức. Đã giải mã toàn bộ 51.839 ảnh: không có ảnh hỏng và không có cảnh báo kích thước metadata. Năm kiểm thử về chia tập, bảo vệ manifest và tiền xử lý đều đạt.

## Tiền xử lý và lý do chọn

1. Đọc ảnh màu BGR uint8 bằng OpenCV. Hàm đọc dùng `imdecode` với đường dẫn Unicode để làm việc trên Windows và hỗ trợ PPM gốc.
2. Resize toàn bộ ảnh biển báo được cung cấp về 64×64 pixel. Kích thước cố định giúp tạo vector HOG có cùng số chiều và giới hạn chi phí tính toán. Đây là cấu hình ban đầu, chưa phải tối ưu. Khi thu nhỏ cả hai chiều dùng `INTER_AREA`; trường hợp khác dùng `INTER_LINEAR`.
3. Với nhánh màu, chuyển BGR sang HSV. OpenCV uint8 quy ước H từ 0 đến 179, S và V từ 0 đến 255. Khi tính histogram phải dùng miền H `[0,180)` và S/V `[0,256)` để không bỏ giá trị tối đa.
4. Với nhánh HOG, chuyển ảnh BGR đã resize sang grayscale. Nhánh kết hợp lấy HSV và gray từ cùng ảnh, cùng kích thước.
5. Không cắt thêm theo ROI annotation, không dùng CLAHE/lọc nhiễu mặc định. Điều này giữ một cấu hình tiền xử lý chung cho ba cách biểu diễn; ảnh đầu vào vốn là crop biển báo và có thể còn viền nền. ROI được bảo lưu trong metadata để truy nguyên, không dùng như thông tin bổ sung chỉ có trong benchmark.

Resize vuông có thể làm biến dạng tỉ lệ; nền còn lại, ảnh tối/mờ/nghiêng và độ phân giải gốc thấp có thể gây khó cho cả màu và HOG. Đây là giới hạn cần phân tích ở phần kết quả, chưa có căn cứ kết luận ảnh hưởng định lượng.

Hàm tiền xử lý không học thống kê từ dữ liệu. Scaler ở bước đặc trưng phải chỉ fit trên train, sau đó dùng lại cho validation/test/demo. Nhánh kết hợp cần chuẩn hóa từng nhóm trước khi ghép và chọn trọng số bằng validation. Các bước này do người 2–4 triển khai cùng bộ phân loại.

## Bàn giao và kiến thức học phần

Đầu ra người 1 gồm bộ đọc, module tiền xử lý, split cố định, bảng phân bố lớp, biểu đồ, 10 ảnh mẫu train và hướng dẫn chạy. Chương 4 được vận dụng cho hệ màu RGB/HSV; chương 8 cho biểu diễn đặc trưng và thiết kế đánh giá. Khi hoàn thiện báo cáo cần đối chiếu số trang/nội dung với slide học phần.

Phần việc này mới xác nhận dữ liệu và luồng tiền xử lý. Chưa có mô hình kNN, dự đoán hoặc Accuracy/Macro-F1; mọi kết luận so sánh đặc trưng phải chờ thực nghiệm.

## Khai báo hỗ trợ AI

Codex hỗ trợ viết mã tải/đọc dữ liệu, chia tập, tiền xử lý, kiểm tra và soạn bản nháp tài liệu. Thành viên phụ trách cần đọc hiểu, kiểm chứng, chỉnh sửa và chịu trách nhiệm với nội dung nộp; bổ sung tên người và mức đóng góp thực tế trong báo cáo chung.
