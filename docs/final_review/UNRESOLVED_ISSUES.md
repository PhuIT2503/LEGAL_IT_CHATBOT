# Vấn đề chưa giải quyết

1. Chưa tái chạy đánh giá 301 câu. Tệp kết quả ngày 31/07/2026 có Naive 301/301, Article Expansion 301/301 nhưng Critic 23/301; vì vậy chưa có so sánh ba chế độ đầy đủ trong cùng một lần chạy và không được suy rộng từ ba case demo.
2. Cổng liên quan trong mã nguồn có hành vi fail-open khi LLM lỗi. Ba case được chọn không gặp lỗi gate, nhưng báo cáo nghiên cứu phải luôn phân biệt `accept` thật với `fail_open`.
3. Neo4j phát cảnh báo vì mã nguồn đọc `r.ghi_chu` trong khi schema hiện tại không có property này. Cảnh báo chưa làm truy vấn thất bại nhưng nên sửa truy vấn hoặc bổ sung schema nhất quán.
4. Chainlit không lưu bền vững thread demo đúng do SQLite không bind trực tiếp được `tags` kiểu list; log ghi `sqlite3.ProgrammingError`. Cần serialize tags theo định dạng data layer hỗ trợ trước khi dùng lịch sử làm bằng chứng dài hạn.
5. Chainlit chưa có blob storage client; element không được lưu bền vững. Ảnh giao diện vẫn là ảnh chụp trực tiếp của câu trả lời đã hoàn tất.
6. `data/.qdrant_gte_base` mang tên GTE nhưng cấu hình vector quan sát là 1.024 chiều, không khớp kích thước 768 của cấu hình GTE đã mô tả trong tài liệu bổ sung; bộ này không được sử dụng trong demo hay suy diễn chỉ số.
