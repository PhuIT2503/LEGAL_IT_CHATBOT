# Kỹ thuật Xây dựng Knowledge Graph và Chunking trong LEGAL IT CHATBOT

Tài liệu này trình bày chuyên sâu về hai kỹ thuật xử lý dữ liệu nền tảng của hệ thống: **Cắt nhỏ văn bản (Chunking) theo cấu trúc pháp luật** và **Trích xuất Đồ thị Tri thức (Knowledge Graph Extraction)**.

---

## 1. Kỹ thuật Chunking theo cấu trúc Điều - Khoản - Điểm

Hệ thống không sử dụng các phương pháp cắt chunk theo số lượng token (như RecursiveCharacterTextSplitter) vì sẽ làm đứt gãy mạch ý nghĩa pháp lý. Thay vào đó, dự án phát triển module `VBPLChunker` (`src/data_processing/chunking.py`) sử dụng Regular Expressions (Regex) để bóc tách theo đúng logic cấu trúc của văn bản pháp luật Việt Nam.

### Phương pháp phân tách:
- **Dùng Regex bắt cấu trúc:**
  - `dieu_pattern`: Bắt các thẻ Điều (VD: "Điều 1.", "Điều 114a.").
  - `khoan_pattern`: Bắt các Khoản (VD: "1.", "2.").
  - `diem_pattern`: Bắt các Điểm (VD: "a)", "b)", "đ)").
- **Tạo Child Chunks (Chunk con):** Mỗi thành phần nhỏ nhất (Lời dẫn Điều, Nội dung một Khoản, Nội dung một Điểm) được tách thành một chunk độc lập. 
  - *Ví dụ Chunk ID:* `95_2015_QH13_D114_K1_Pa` (Điểm a, Khoản 1, Điều 114, Luật 95/2015/QH13).
- **Gộp Parent Chunks (Chunk cha):** Tất cả các Child Chunks thuộc cùng một Điều sẽ được gắn chung một thuộc tính `dieu_id`. Hệ thống sẽ gộp tất cả nội dung này lại thành một Parent Chunk đại diện cho toàn bộ Điều.
- **Tác dụng:** Khi người dùng hỏi và vector search (Qdrant) truy xuất trúng một Điểm nhỏ (Child Chunk), hệ thống dễ dàng tra ngược ID để lấy toàn bộ văn cảnh của cả Điều (Parent Chunk), tránh tình trạng hiểu sai do thiếu vế câu hoặc điều kiện đi kèm ở lời dẫn.

---

## 2. Kỹ thuật xây dựng Knowledge Graph (KG)

Việc trích xuất Thực thể (Entities) và Quan hệ (Relationships) cho KG được thực hiện tự động bằng LLM kết hợp với các luật xử lý linh hoạt (Rule-based).

### 2.1. Cấu hình Model trích xuất
- **Mô hình sử dụng:** `Qwen/Qwen2.5-7B-Instruct`.
- **Phần cứng & Tối ưu hóa:** Mô hình được load qua HuggingFace với kỹ thuật lượng tử hóa 4-bit (`BitsAndBytesConfig`, dạng `nf4`). Cấu hình này giúp model 7 tỷ tham số chạy mượt mà trên các GPU giới hạn VRAM (như Google Colab T4 16GB) mà không bị tràn RAM.

### 2.2. Chiến lược Trích xuất 2 bước (2-Pass Extraction)
Văn bản pháp luật dài và rất nhiều chi tiết, nếu bắt LLM trích xuất cả Entities và Relations trong một lần sinh text, chuỗi JSON trả về rất dễ vượt quá giới hạn token (max_new_tokens), dẫn tới bị cắt đứt giữa chừng (truncated JSON). Dự án giải quyết bằng chiến lược 2-pass:
- **Pass 1 (Trích xuất Entities):** LLM chỉ tập trung quét và trả về danh sách các thực thể: `Dieu`, `Khoan`, `Diem`, `HanhVi`, `CheTai`, `ChuThe`, `NghiaVu`. (Max 2048 tokens).
- **Pass 2 (Trích xuất Relations):** Hệ thống cung cấp danh sách ID của các Entities vừa trích xuất từ Pass 1 và yêu cầu LLM tìm các mối liên hệ giữa chúng như: `CO_KHOAN`, `CO_DIEM`, `THUC_HIEN`, `BI_XU_PHAT`, `THAM_CHIEU`.
- **Sửa lỗi JSON tự động:** Các output JSON thô được bóc tách thẻ CoT (Chain-of-Thought `<thinking>...</thinking>`). Nếu JSON vẫn bị cắt dở do hết token, hàm `_repair_truncated_json` sẽ tự động tìm điểm đóng ngoặc hợp lệ gần nhất và vá lại chuỗi để lưu giữ tối đa dữ liệu.

### 2.3. Kết hợp Rule-based để xử lý Tham chiếu chéo (Cross-references)
Tham chiếu chéo (khi một Điều khoản nhắc đến văn bản hoặc Điều luật khác) là tính năng tối quan trọng.
- **Bước LLM trích xuất:** Prompt `build_cross_ref_prompt` được thiết kế đặc biệt để bắt LLM tìm các mẫu câu như *"theo quy định tại Điều 5 Luật này"* hoặc *"theo quy định của Bộ luật Hình sự"*, trả về một mảng chứa Node nguồn và thông tin tham chiếu đích.
- **Bước Rule-based xử lý (`LegalGraphBuilder`):** 
  - Các Entity ID được chuẩn hóa (Normalization) thành định dạng `snake_case` không dấu và ghép thêm tiền tố `van_ban_id` để tránh trùng lặp giữa các văn bản khác nhau (Deduplication).
  - Khi duyệt qua các cạnh `THAM_CHIEU` (Cross-document references), hệ thống tự động kiểm tra đồ thị. Nếu Điều được tham chiếu **chưa có sẵn** trong đồ thị cục bộ, Graph Builder dùng logic rule-based để tự động tạo ra một node "giả" (Placeholder Node, ví dụ: `is_placeholder: True`).
  - **Tác dụng:** Nhờ các placeholder nodes và edges `THAM_CHIEU` này, Critic Agent trong lúc phân tích câu hỏi có thể nhận biết "ngữ cảnh đang bị thiếu một Điều luật liên quan", từ đó trigger quá trình lấy thêm văn bản từ CSDL.
