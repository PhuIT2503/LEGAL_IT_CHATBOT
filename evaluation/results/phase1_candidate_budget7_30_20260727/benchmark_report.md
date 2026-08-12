# Legal Retrieval Benchmark Report

- Run ID: `20260727_130056`
- Generated: `2026-07-27T13:00:56+07:00`
- Benchmark SHA-256: `126438a83f9902d950b140f74fd1f527b899ec9dc160ce1163f47c0c5d3e114c`
- Cases: **30/30** completed
- Runtime errors: **0**

## Overall Metrics

| Metric | Value |
|---|---:|
| Domain Recall | 57.83% |
| Domain Precision | 80.44% |
| Behavior Recall | 30.85% |
| Behavior Precision | 59.17% |
| Recall@5 | 38.33% |
| Recall@10 | 41.67% |
| MRR | 31.14% |
| Citation Accuracy | 18.89% |
| Wrong Domain Rate | 19.56% |
| Wrong Behavior Rate | 40.83% |
| Recursive Precision | 90.00% |
| Recursive Noise Rate | 10.00% |
| Applicability Accuracy | 55.33% |
| Average Retrieval Latency | 3976.8 ms |
| Average Total Latency | 96325.8 ms |

## Metrics by Category

| Category | Cases | Domain R | Behavior R | R@5 | R@10 | Citation | Wrong domain |
|---|---:|---:|---:|---:|---:|---:|---:|
| AI Copyright | 3 | 66.67% | 41.67% | 33.33% | 66.67% | 33.33% | 0.00% |
| AI Deepfake | 3 | 73.33% | 46.76% | 16.67% | 16.67% | 22.22% | 0.00% |
| Advertising | 3 | 80.56% | 33.33% | 66.67% | 66.67% | 22.22% | 0.00% |
| Consumer Protection | 3 | 27.78% | 0.00% | 33.33% | 33.33% | 22.22% | 33.33% |
| Cyber Attack | 3 | 44.44% | 34.52% | 33.33% | 33.33% | 0.00% | 64.44% |
| Electronic Transactions | 3 | 50.00% | 0.00% | 83.33% | 83.33% | 22.22% | 53.33% |
| Malware | 3 | 61.11% | 22.22% | 33.33% | 33.33% | 16.67% | 16.67% |
| Network Security | 3 | 66.67% | 16.67% | 33.33% | 33.33% | 16.67% | 0.00% |
| Personal Data | 3 | 66.67% | 50.00% | 33.33% | 33.33% | 33.33% | 11.11% |
| SQL Injection | 3 | 41.11% | 63.33% | 16.67% | 16.67% | 0.00% | 16.67% |

## Metrics by Difficulty

| Difficulty | Cases | Domain R | Behavior R | R@5 | R@10 | Citation | Applicability |
|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | 10 | 75.00% | 43.33% | 65.00% | 75.00% | 33.33% | 51.71% |
| Hard | 10 | 53.17% | 20.47% | 10.00% | 10.00% | 0.00% | 59.00% |
| Medium | 10 | 45.33% | 28.75% | 40.00% | 40.00% | 23.33% | 55.29% |

## Top Recurring Errors

- **Generation Grounding Error**: 29
- **Wrong Behavior**: 29
- **Wrong Citation**: 28
- **Applicability Error**: 24
- **Wrong Domain**: 24
- **Missing Relevant Law**: 19
- **Recursive Noise**: 3
- **Hallucinated Citation**: 0

## Top 20 Failed Cases

### 1. `cyber_attack_medium_001` — Cyber Attack / Medium

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Recursive Noise, Applicability Error, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Kẻ tấn công xâm nhập tài khoản quản trị website và chiếm quyền điều khiển hệ thống để lấy dữ liệu thì căn cứ nào điều chỉnh?

### 2. `advertising_hard_001` — Advertising / Hard

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Mạng xã hội phân tích hồ sơ người dùng để nhắm mục tiêu quảng cáo, đồng thời để người bán đăng nội dung quảng cáo gây nhầm lẫn. Nền tảng phải tuân thủ những nhóm nghĩa vụ nào?

### 3. `ai_copyright_hard_001` — AI Copyright / Hard

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Công ty sao chép hàng nghìn bản ghi có bản quyền để huấn luyện AI thương mại rồi phân phối sản phẩm mô phỏng nội dung gốc. Hãy phân biệt ngoại lệ và hành vi xâm phạm.

### 4. `consumer_medium_001` — Consumer Protection / Medium

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Sản phẩm có khuyết tật gây thiệt hại cho người mua thì tổ chức kinh doanh phải bồi thường khi nào và được miễn trách nhiệm trong trường hợp nào?

### 5. `cyber_attack_hard_001` — Cyber Attack / Hard

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Nhóm tấn công khai thác lỗ hổng, chiếm quyền máy chủ của hệ thống thông tin quan trọng rồi sao chép dữ liệu. Phân tích hành vi và chế tài có thể áp dụng.

### 6. `network_security_medium_001` — Network Security / Medium

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Hệ thống thông tin quan trọng về an ninh quốc gia phải áp dụng biện pháp bảo vệ và giám sát nào?

### 7. `personal_data_easy_001` — Personal Data / Easy

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Doanh nghiệp dùng số điện thoại khách hàng để gửi quảng cáo khi chưa có sự đồng ý thì có được phép không?

### 8. `sql_injection_hard_001` — SQL Injection / Hard

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Nhân viên vượt phạm vi quyền được cấp, dùng blind SQL Injection trích xuất từng phần cơ sở dữ liệu khách hàng và bán cho bên thứ ba. Hãy xác định các nhóm hành vi pháp lý.

### 9. `sql_injection_medium_001` — SQL Injection / Medium

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 50.00%
- Citation Accuracy: 0.00%
- Question: Lập trình viên thử SQL Injection trên website của khách hàng khi chưa được ủy quyền, đọc bảng tài khoản nhưng chưa phát tán dữ liệu. Trách nhiệm nào cần xem xét?

### 10. `electronic_transactions_medium_001` — Electronic Transactions / Medium

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 50.00%
- Citation Accuracy: 66.67%
- Question: Hợp đồng được hệ thống tự động tạo và gửi nhưng có lỗi nhập liệu thì bên nhập sai có quyền rút lại phần dữ liệu đó không?

### 11. `consumer_hard_001` — Consumer Protection / Hard

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Sàn thương mại điện tử đưa vào điều khoản mẫu cho phép đơn phương thay đổi giá và loại trừ toàn bộ trách nhiệm với người mua. Điều khoản nào cần bị kiểm tra?

### 12. `deepfake_hard_001` — AI Deepfake / Hard

- Errors: Wrong Behavior, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Công ty thu thập ảnh khách hàng để huấn luyện AI, tạo video deepfake quảng cáo rồi công khai video mà không xin đồng ý. Công ty có vi phạm và phải xử lý dữ liệu thế nào?

### 13. `deepfake_medium_001` — AI Deepfake / Medium

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Một nhãn hàng giả giọng nói và khuôn mặt của ca sĩ bằng AI, đăng video lên mạng để bán hàng dù ca sĩ không cho phép. Hành vi nào cần được xem xét?

### 14. `malware_hard_001` — Malware / Hard

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Applicability Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Tin tặc cài mã độc vào hệ thống thông tin quan trọng, duy trì quyền truy cập và gửi dữ liệu ra máy chủ bên ngoài. Cần truy xuất căn cứ về tấn công và bảo vệ hệ thống nào?

### 15. `malware_medium_001` — Malware / Medium

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Tổ chức nghiên cứu mã độc và lỗ hổng mạng để xây dựng giải pháp phòng chống có thuộc hoạt động nghiên cứu an ninh mạng không?

### 16. `network_security_hard_001` — Network Security / Hard

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Doanh nghiệp nghiên cứu giải pháp phát hiện lỗ hổng, mã độc và phương thức tấn công để bảo vệ hệ thống. Hoạt động nghiên cứu này được quy định ra sao?

### 17. `personal_data_hard_001` — Personal Data / Hard

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Missing Relevant Law, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Nền tảng vừa quyết định mục đích xử lý dữ liệu vừa thuê nhà cung cấp lưu trữ. Hãy xác định vai trò của các bên và nghĩa vụ khi chia sẻ dữ liệu.

### 18. `sql_injection_easy_001` — SQL Injection / Easy

- Errors: Wrong Domain, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Dùng SQL Injection khai thác lỗ hổng website và tải xuống cơ sở dữ liệu có phải là truy cập trái phép không?

### 19. `advertising_easy_001` — Advertising / Easy

- Errors: Wrong Behavior, Wrong Citation, Recursive Noise, Applicability Error, Generation Grounding Error
- Recall@10: 100.00%
- Citation Accuracy: 0.00%
- Question: Có được dùng dữ liệu cá nhân của khách hàng để quảng cáo trực tiếp khi họ chưa đồng ý không?

### 20. `cyber_attack_easy_001` — Cyber Attack / Easy

- Errors: Wrong Domain, Wrong Behavior, Wrong Citation, Applicability Error, Generation Grounding Error
- Recall@10: 100.00%
- Citation Accuracy: 0.00%
- Question: Một người dùng mật khẩu lấy cắp để đăng nhập trái phép vào tài khoản của người khác bị xử lý thế nào?

## Artifacts

- `benchmark_summary.json`: aggregate metrics and run metadata.
- `benchmark_details.json`: per-case trace, metrics and errors.
- `plots/`: visual metrics generated from the same summary.
