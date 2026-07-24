# LEGAL IT CHATBOT - Mô tả Dự án Khóa luận Tốt nghiệp

Tài liệu này trình bày chi tiết về dự án Khóa luận Tốt nghiệp **LEGAL IT CHATBOT** - Hệ thống RAG hỗ trợ tra cứu và phân tích văn bản pháp luật, tập trung vào lĩnh vực Sở hữu trí tuệ và Công nghệ thông tin.

---

## 1. Ý tưởng và Vấn đề giải quyết
**Vấn đề:** 
Các hệ thống RAG (Retrieval-Augmented Generation) truyền thống thường gặp hạn chế khi xử lý văn bản pháp luật:
- **Thiếu chế tài bổ sung:** Một hành vi vi phạm thường có hình phạt chính và hình phạt bổ sung (hoặc biện pháp khắc phục hậu quả) nằm ở các Khoản, Điểm khác nhau trong cùng một Điều. RAG thuần thường chỉ lấy được chunk chứa hình phạt chính mà bỏ sót chế tài bổ sung.
- **Cấu trúc đa Khoản:** Nhiều Điều luật quy định danh sách điều kiện, quyền lợi dài nhiều Khoản. RAG thuần có thể lấy thiếu ngữ cảnh nếu chunk bị cắt nhỏ.
- **Tham chiếu chéo (Cross-reference):** Điều luật này dẫn chiếu sang Điều luật khác. RAG thuần không thể tự động đi theo liên kết dẫn chiếu này.

**Ý tưởng giải quyết:** 
Xây dựng một hệ thống RAG nâng cao kết hợp **Knowledge Graph (Đồ thị tri thức)** và **Critic Agent**. Thay vì nhồi nhét toàn bộ các Điều luật (dễ gây nhiễu và "Lost in the Middle"), hệ thống sử dụng đồ thị tri thức (Neo4j) để phát hiện các thông tin CÒN THIẾU (chế tài bổ sung, tham chiếu chéo) từ câu trả lời nháp. Sau đó, thông qua một "Cổng lọc ngữ nghĩa" (Relevance Gate) bằng LLM để xác nhận sự liên quan trước khi cập nhật vào ngữ cảnh để sinh câu trả lời cuối cùng đầy đủ và chính xác nhất.

---

## 2. Pipeline của các kịch bản (Modes)
Dự án được thiết kế với 3 kịch bản chạy độc lập để đánh giá và so sánh (A/B/C testing):

### Kịch bản 1: Mode `naive` (RAG truyền thống - Baseline)
- **Luồng xử lý:** Câu hỏi $\rightarrow$ Hybrid Search (Qdrant) $\rightarrow$ Lấy Top-K chunks nhỏ $\rightarrow$ LLM sinh câu trả lời trực tiếp.
- **Đặc điểm:** Không sử dụng Knowledge Graph, không có Critic Agent. Thường trả lời nhanh nhưng dễ bỏ sót thông tin liên quan nằm ngoài top-K chunk.

### Kịch bản 2: Mode `article_expand` (RAG mở rộng toàn Điều)
- **Luồng xử lý:** Câu hỏi $\rightarrow$ Hybrid Search quét tập chunk rộng hơn $\rightarrow$ Chọn các Điều (Parent chunks) thỏa mãn ngưỡng điểm $\rightarrow$ Trích xuất **toàn văn** của các Điều đó $\rightarrow$ LLM sinh câu trả lời trực tiếp.
- **Đặc điểm:** Không dùng Knowledge Graph và Critic Agent. Giải quyết được vấn đề thiếu cấu trúc trong cùng một Điều, nhưng ngữ cảnh bơm vào LLM rất lớn, dễ bị nhiễu thông tin (noise) và vẫn không giải quyết được vấn đề tham chiếu chéo sang Điều khác.

### Kịch bản 3: Mode `critic` (Critic Agent + Knowledge Graph - Đề xuất của khóa luận)
- **Bước 1 (Sinh nháp):** Lấy Top-K chunks nhỏ và yêu cầu LLM sinh ra **câu trả lời nháp (draft response)**.
- **Bước 2 (Critic Check qua Knowledge Graph):** Critic Agent sử dụng Neo4j để đối chiếu các Điều đã lấy trong top-K xem có bị thiếu sót gì không dựa trên 3 tiêu chí cấu trúc:
  1. *Same-Điều (chế tài):* Phát hiện hành vi có hình phạt bổ sung ở Khoản khác chưa được lấy.
  2. *Same-Điều (cấu trúc):* Phát hiện số chunk lấy được ít hơn tổng số chunk thực tế của Điều.
  3. *Cross-Điều:* Phát hiện có quan hệ tham chiếu (`THAM_CHIEU`) sang một Điều khác.
- **Bước 3 (Cổng lọc ngữ nghĩa - Relevance Gate):** Các nội dung ứng viên (candidate) được Critic lấy thêm sẽ phải qua một lệnh gọi LLM nhỏ (chỉ trả lời yes/no) để thẩm định xem nội dung đó có **thực sự cần thiết** để trả lời câu hỏi hay không. Điều này giúp giữ ngữ cảnh SẠCH, tránh lấy nhầm thông tin lạc đề.
- **Bước 4 (Sinh lại):** Nếu phát hiện thiếu và vượt qua được cổng lọc, phần còn thiếu được ghép vào ngữ cảnh và LLM **sinh lại (regenerate)** câu trả lời. Nếu không thiếu gì, trả về luôn câu trả lời nháp để tiết kiệm chi phí gọi LLM.

---

## 3. Các tham số chính trong hệ thống
Các siêu tham số (hyperparameters) được thiết lập nhằm tối ưu hóa quá trình retrieval và kiểm duyệt:
- `top_k = 5`: Số lượng chunk con (child chunks) tối đa lấy về từ bước Hybrid Search.
- `critic_score_ratio = 0.7`: Ngưỡng lọc tỷ lệ điểm trong mode `critic`. Critic Agent chỉ chạy kiểm tra (completeness-check) trên các Điều có điểm số $\ge 70\%$ so với Điều có điểm cao nhất. (Tránh các Điều nhiễu lọt vào top-k kéo theo tham chiếu lạc đề).
- `critic_max_dieu = 3`: Giới hạn tối đa số lượng Điều luật được đưa vào quá trình kiểm tra của Critic Agent.
- `article_expand_score_ratio = 0.6`: Ngưỡng lọc tỷ lệ điểm dành riêng cho mode `article_expand`, cho phép mở rộng các Điều có điểm số $\ge 60\%$ so với Điều điểm cao nhất (xét trên một tập candidate rộng).
- `prefetch_limit = 20`: Giới hạn candidate pool quét qua khi dùng cơ chế FusionQuery (RRF) để kết hợp dense và sparse vector trong Qdrant.

---

## 4. Các Models được sử dụng
- **Model Sinh văn bản (LLMs):**
  - Model chính: **Qwen2.5-7B-Instruct** (chạy local qua Ollama) phục vụ sinh câu trả lời, Agent Router và Relevance Gate.
  - Ngoài ra hệ thống hỗ trợ linh hoạt kết nối API tới **GPT-4o-mini (OpenAI)**, **Gemini 1.5 Flash (Google)** hoặc **Llama-3.3-70b**.
- **Model Nhúng (Embedding Model):** 
  - Sử dụng model tiếng Việt đã được **Fine-tune** riêng trên tập văn bản pháp luật: `AITeamVN/Vietnamese_Embedding_v2` (`ai_vietnamese_embedding_v2_finetuned_final`). 
  - Kích thước vector 1024 chiều (dense) kết hợp cùng BM25 (sparse vector) để chạy Hybrid Search.
- **Model Đánh giá (Judge LLM):** Sử dụng các model (như Qwen2.5-7B hoặc LLM trả phí) với `temperature=0.0` để đóng vai trò làm giám khảo khách quan chấm điểm Legal Completeness.

---

## 5. Metrics Đánh giá (Evaluation Metrics)
Hệ thống sử dụng 2 lớp chỉ số đánh giá nhằm phản ánh cả chất lượng RAG chung lẫn tính đặc thù pháp lý:

### 5.1. RAGAS Framework
Bộ chỉ số chuẩn cho hệ thống RAG:
- **Faithfulness:** Độ trung thực của câu trả lời so với ngữ cảnh (tránh ảo giác - hallucination).
- **Answer Relevancy:** Câu trả lời có đi thẳng vào trọng tâm câu hỏi hay không.
- **Context Precision:** Độ chính xác của ngữ cảnh (tài liệu đúng có được xếp hạng cao không).
- **Answer Correctness:** Mức độ đúng đắn của câu trả lời so với reference (ground truth).

### 5.2. Legal Completeness Rate (Tỷ lệ đầy đủ pháp lý)
Đây là chỉ số TRUNG TÂM của khóa luận, sử dụng phương pháp **LLM-as-a-judge**. 
- Mỗi câu hỏi testset đi kèm một danh sách `required_facts` (các ý bắt buộc phải có).
- Judge LLM sẽ chấm điểm Yes/No xem câu trả lời cuối cùng có thể hiện ĐÚNG và ĐỦ từng fact hay không. Tỷ lệ % số fact đạt được chính là Legal Completeness Rate.

### 5.3. Chỉ số Chi phí (Efficiency / Token Usage)
Đo lường sự đánh đổi giữa chất lượng và tài nguyên:
- **Token TB/câu (Total, Prompt, Completion):** Tổng token tiêu thụ cho toàn bộ pipeline (bao gồm sinh nháp, router, cổng lọc).
- **Số lệnh gọi LLM trung bình:** So sánh số lần gọi LLM (mode `critic` gọi nhiều lần hơn `naive`).
- **Hiệu quả ngữ cảnh (Token của lệnh sinh trả lời cuối cùng):** Token tại ĐÚNG lệnh gọi chốt hạ câu trả lời, giúp đánh giá xem mode nào đang nhét quá nhiều ngữ cảnh dư thừa vào prompt (thường `article_expand` sẽ tiêu tốn nhiều nhất ở bước này).
