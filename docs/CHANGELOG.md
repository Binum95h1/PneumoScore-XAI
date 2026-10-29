# v1.1 — vá pipeline cũ,không thay mô hình

- Giữ class ResNet18Baseline,ECA,ResNet18ECA,SquarePad và inference_transform nguyên AST.
- Giữ trọng số cũ,seed2026,50/50,threshold0.55.
- Overlap trở thành cảnh báo/audit;mọi ID yêu cầu được dự đoán;gắn cờ cần BTC đối chiếu.
- Giữ Prediction dạng Normal/Pneumonia theo Điều6.
- Khớp ID chính xác trước,fallback duy nhất theo case/stem và ghi mapping;không đoán khi thiếu/nhập nhằng.
- WebP8-bit được quét;gom lỗi ảnh hỏng/16-bit/định dạng không hỗ trợ;không bịa nhãn hay tự đổi chính sách intensity.
- Portable pickle bằng builtins dtype/shape/bytes và reader chặn global. Tương thích đọc đúng file legacy đã xác minh bằng SHA256.
- Ghi thêm môi trường,CSV/ZIP hashes,audit và trạng thái run. Chuyển output cũ sang previous_runs trước khi chạy lại.
- Kiểm tra nhánh nộp còn hiệu lực trong Python -O.
- Bổ sung suite end-to-end,replay notebook trong ZIP ở process mới,kiểm tra tensor và toàn bộ1028 xác suất Validation.
- Review ECA,Grad-CAM,McNemar,bootstrap,split-half,shortcut và tính độc lập bệnh nhân.