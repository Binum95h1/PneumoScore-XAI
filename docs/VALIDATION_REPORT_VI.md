# Ensemble ResNet18 gốc + ECA seed 2026 đã kiểm tra trọng số

Cấu hình: ghép xác suất 50/50, ngưỡng 0,55 đã khóa từ phân tích validation trước. Dùng hai checkpoint cùng seed 2026, không train lại. Phiên kiểm tra dùng GPU Tesla T4, FP32, GPU số 0.

## Kiểm tra đã đạt

- Hai checkpoint đúng kiến trúc/seed, cùng manifest và backbone khởi tạo.
- Cả 1.028 ảnh validation khớp byte và pixel với manifest cũ.
- Xác suất từng model khớp CSV nguồn (atol 1e-6, rtol 1e-5), nhãn không thay đổi.
- model.pkl chứa cả hai bộ trọng số dạng NumPy và metadata. Đã nạp lại model.pkl và suy luận toàn bộ validation, khớp ensemble CSV và confusion matrix đã đo.
- CSV kiểm tra định dạng có đúng Image_id, Prediction. Đây là validation_format_check.csv, không phải bài nộp Test.

## Chỉ số validation của một cặp seed 2026

| Model | Ngưỡng | P (%) | Accuracy (%) | Sensitivity (%) | Specificity (%) | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | 0.35 | 97.8146 | 98.1518 | 99.6089 | 93.8697 | 3 | 16 |
| resnet18_eca | 0.50 | 97.9109 | 98.0545 | 98.9570 | 95.4023 | 8 | 12 |
| ensemble_50_50 | 0.55 | 97.9892 | 98.1518 | 99.0874 | 95.4023 | 7 | 12 |

## Tốc độ và dung lượng

| Model | Batch | Forward median (ms/batch) | P95 (ms/batch) | Images/s |
|---|---:|---:|---:|---:|
| resnet18 | 1 | 3.352 | 3.876 | 298.3 |
| ensemble_50_50 | 1 | 4.560 | 6.343 | 219.3 |
| resnet18 | 32 | 25.964 | 26.203 | 1232.5 |
| ensemble_50_50 | 32 | 52.217 | 52.448 | 612.8 |

resnet18: median end-to-end trên 1.028 ảnh, hai lượt đo thứ tự luân phiên = 6.408 giây (6.234 ms/ảnh theo thông lượng batch32).

ensemble_50_50: median end-to-end trên 1.028 ảnh, hai lượt đo thứ tự luân phiên = 6.484 giây (6.307 ms/ảnh theo thông lượng batch32).

model.pkl: 90,200,542 byte = 86.02 MiB. Tổng 22,354,055 tham số. Forward-only không gồm đọc ảnh/tiền xử lý. End-to-end có đọc ảnh/tiền xử lý, warm cache, batch32, workers2; ms/ảnh ở đây là thời gian trung bình theo batch, không phải độ trễ ảnh đơn. Batch1 mới phản ánh forward ảnh đơn.

## Cách dùng và lưu file

Giải nén vào PneumoScore-XAI/Outputs/Ensemble_Seed2026_Verified/. Giữ toàn bộ thư mục và model/resume cũ. Có thể dọn bản trùng trong Downloads sau khi kiểm tra bản đã lưu.

- model.pkl: trọng số hai model và cấu hình ghép/ngưỡng, cần giữ.
- submission.ipynb: mở trên Kaggle, Add Input model.pkl và Test BTC, điền TEST_DIR/mã đội/CSV mẫu, Run All để tự sinh CSV và ZIP bài nộp. Không sửa CSV bằng tay.
- verify_export.ipynb: tái lập phiên kiểm tra trên Kaggle T4; cần Output Stability, Output Ensemble Analysis và dataset BTC đã dùng.
- ensemble_inference.py: mã trong notebook để tham khảo và giải thích, không cần chạy riêng trên Windows.
- validation_*.csv, validation_metrics.csv, verification.json: bằng chứng suy luận lại và kiểm tra định dạng, không nộp làm kết quả Test.
- benchmark*.csv/json, environment.json, source_checkpoints.json và báo cáo này: lưu hồ sơ nghiên cứu.

Gói hiện tại là bộ model đã kiểm tra, chưa phải ZIP bài nộp vì chưa có ảnh Test BTC trong phiên này. Khi có Test, notebook sinh một .pkl, một .ipynb và team01.csv theo Điều 6. Không dùng nhãn Test.

P của cặp seed 2026 không phải trung bình ba seed. Các chỉ số vẫn là validation đã dùng chọn checkpoint/ensemble/ngưỡng; không chứng minh hiệu quả trên bệnh nhân mới. Giữ baseline làm phương án chính đã chuẩn hóa và ensemble làm ứng viên đối chứng.
