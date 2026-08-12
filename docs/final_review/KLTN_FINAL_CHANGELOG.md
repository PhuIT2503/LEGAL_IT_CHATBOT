# Nhật ký hoàn thiện khóa luận

## Tệp đầu vào và đầu ra

- Tệp nguồn: `KLTN_Final.docx`, SHA-256 `fbcdd2e3952f50ddaaf0a2d23aefe19701e62c37da1c404b226e6c136a1e8f7f`.
- Bản sao lưu: `KLTN_Final_Backup.docx`, cùng SHA-256 với tệp nguồn; tệp nguồn không bị ghi đè.
- Bản hoàn thiện: `KLTN_Final_v2.docx`, SHA-256 `5134a75b7d88ef1d2bfbb7aba0695a351786aedb8c952c6d8a61baf470c870aa`.
- Bản PDF xuất từ Microsoft Word: `KLTN_Final_v2.pdf`, SHA-256 `5599b2a56e820b16bd33487b2a35a7dfcc8cd2f04ff9a488d5c8b1b465518e3`.
- PDF có 49 trang Letter; đã render và kiểm tra trực quan đủ 49/49 trang.

## Thay đổi nội dung

- Bổ sung mô tả xuyên suốt từ corpus, chuẩn hóa Điều-Khoản-Điểm, biểu diễn dense/BM25, RRF, truy xuất top-k, khôi phục parent Article, duyệt đồ thị, cổng liên quan đến sinh câu trả lời.
- Làm rõ yêu cầu người dùng, tiêu chí chấp nhận, cách chạy CLI/Chainlit và các kịch bản đầu vào.
- Đồng bộ mô tả `THAM_CHIEU` với mã nguồn: kiểm tra cả chiều đi ra và chiều đi vào; duyệt tiếp tối đa hai hop; tổng số Điều lấy thêm không quá tám với cấu hình hiện tại.
- Cập nhật trạng thái chạy ngày 04/08/2026: Qdrant Base 1.211 parent/10.355 child; Neo4j 15.666 node, 14.137 quan hệ, 1.140 node Điều và 1.612 `THAM_CHIEU`; Chainlit và CLI chạy thành công với Qwen cục bộ.
- Điền Bảng 13-15 từ ba trace thực tế: `cat4_02`, `cat1_01`, `cat2_07`. Các bảng giữ rõ trường không được pipeline ghi và kết quả đối chiếu thủ công, không suy diễn.
- Ghi rõ case `cat2_07` chỉ bao phủ tường minh 2/4 required facts trong câu trả lời cuối; không tuyên bố kết quả đầy đủ.
- Giữ nguyên trạng thái đánh giá tổng: Naive 301/301, Article Expansion 301/301, Critic 23/301; ba case study không thay thế đánh giá đủ 301 câu.
- Làm sạch thuật ngữ nội bộ khỏi phần chính; chỉ giữ hai tên tệp nguồn tổng hợp tại Phụ lục A để truy vết.

## Hình, bảng, công thức và định dạng

- Chèn tám ảnh thực nghiệm vào đúng vị trí nội dung:
  - `demo_qdrant_collection.png`: Mục 4.3, Hình 7.
  - `demo_neo4j_tham_chieu.png`: Mục 4.4, Hình 8.
  - `demo_chainlit_overview.png`: Mục 4.5, Hình 10.
  - `demo_compare_all_cli.png`: Mục 4.5, Hình 11.
  - `demo_critic_trace_json.png`: Mục 4.5, Hình 12.
  - `demo_critic_no_gap.png`: sau Bảng 13, Hình 13.
  - `demo_critic_same_article.png`: sau Bảng 14, Hình 14.
  - `demo_critic_cross_article.png`: sau Bảng 15, Hình 15.
- Đánh số liên tục theo thứ tự xuất hiện: tổng cộng 17 hình và 22 bảng; danh mục hình/bảng đã cập nhật đúng trang.
- Mục lục được Microsoft Word cập nhật, trỏ đến trang nội dung 1-41.
- Giữ 47 đối tượng Office Math, trong đó có đủ mười công thức đánh số (1)-(10); không phát hiện công thức hoặc ký hiệu bị vỡ khi render.
- Chuyển phần ký tên cam đoan thành bảng hai cột không viền, có khoảng trống ký tên.
- Chuẩn hóa học hàm người hướng dẫn thành `ThS. Huỳnh Thanh Việt`.
- Khối lệnh CLI căn trái, dùng phông đơn cách 9 pt và không chứa đường dẫn tuyệt đối của máy.
- Loại trang trắng trước danh mục hình; không phát hiện bảng tràn lề, caption tách khỏi hình hoặc trang trắng trong PDF cuối.

## Kiểm tra cuối

- Không còn placeholder `[CHỜ ẢNH DEMO: ...]`.
- Không có mục `ABSTRACT`/`Keywords` tiếng Anh; chỉ có phần `TÓM TẮT` và `Từ khóa` tiếng Việt.
- Các từ khóa nội bộ `workspace`, `source-reported`, `raw`, `audit`, `live repo`, `phiên kiểm toán`, `phiên hoàn thiện` và cụm mặc định fine-tuned cũ không còn trong phần chính.
- Tên `STAIS2026_Legal_RAG_FinalFix.docx` và `Metrics-Fix.docx` chỉ còn trong Phụ lục A theo yêu cầu truy vết nguồn.
- PDF cuối đã được mở lại, kiểm tra metadata, số trang, trích xuất văn bản và toàn bộ ảnh render.
