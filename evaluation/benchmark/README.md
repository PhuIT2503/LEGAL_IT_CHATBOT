# Benchmark dataset

`benchmark.json` là bộ ground truth có version trong Git cho Retrieval Pipeline. Bộ mặc định gồm 30 tình huống: 10 category × 3 mức `Easy`, `Medium`, `Hard`.

## Schema bắt buộc

```json
{
  "id": "sql_injection_easy_001",
  "category": "SQL Injection",
  "difficulty": "Easy",
  "question": "...",
  "expected_domains": ["cybersecurity"],
  "expected_behaviors": ["unauthorized_access"],
  "expected_documents": ["Luật An ninh mạng 2025"],
  "expected_articles": ["13"],
  "expected_clauses": ["3"],
  "expected_points": ["h"]
}
```

Các mảng tọa độ pháp lý được ghép theo index. Nếu chỉ có một document nhưng nhiều article, document đó được tái sử dụng cho tất cả article. Chuỗi rỗng ở `expected_clauses` hoặc `expected_points` có nghĩa ground truth dừng ở cấp Điều, không phải thiếu field.

## Quy tắc biên soạn

- `id` duy nhất, ổn định; không tái sử dụng ID cho một tình huống khác.
- Category dùng tên đọc được; runner chấp nhận cả tên và slug khi lọc.
- Domain và behavior dùng canonical key của project, không dùng mô tả tự do.
- Chỉ ghi văn bản tồn tại trong corpus benchmark đang dùng.
- Ground truth phải được một người có chuyên môn pháp lý review trước khi dùng làm quality gate.
- Khi pháp luật hoặc corpus thay đổi, sửa ground truth trong một commit riêng và ghi lý do để kết quả lịch sử vẫn giải thích được.

Một testcase mới nên được chạy riêng bằng `--case-id` trước, sau đó chạy toàn bộ category để phát hiện tác động chéo.

