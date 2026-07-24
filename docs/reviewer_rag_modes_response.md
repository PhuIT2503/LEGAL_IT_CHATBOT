# Phản hồi Reviewer: Chi tiết cấu hình ba kịch bản RAG

Tài liệu này bổ sung các thông số chi tiết (parameter-level) cho ba cơ chế RAG được đề cập trong bài báo nhằm giải đáp thắc mắc của Reviewer về tính tái lập (reproducibility) và cơ chế hoạt động cụ thể.

## 1. Bảng tóm tắt các tham số chính (Parameter Table)

Dưới đây là bảng tổng hợp các tham số cấu hình của ba kịch bản RAG:

| Parameter | Naive RAG | Article Expansion | KG Critic |
| :--- | :--- | :--- | :--- |
| **Candidate Retrieval (Chunks)** | Top-5 child chunks | Top-20 child chunks (top-k * 4) | Top-5 child chunks |
| **Max Candidate Articles** | N/A | Lấy tối đa **5** Điều (bằng top-k) | Lấy tối đa **4** Điều (`critic_max_dieu`) |
| **Article Score Aggregation** | Suy trực tiếp từ top-k | `MAX` score của các chunks thuộc cùng Điều | Suy trực tiếp từ top-k |
| **Score Threshold Policy** | N/A | $\ge 40\%$ điểm của Điều top 1 (`article_expand_score_ratio=0.4`) | $\ge 60\%$ điểm của Điều top 1 (`critic_score_ratio=0.6`) |
| **Semantic Gate Model** | N/A | N/A | Qwen2.5 7B (hoặc GPT-4o-mini tùy config) |
| **Semantic Gate Threshold** | N/A | N/A | Binary (`yes`/`no`) |
| **Evidence Retrieval Size** | N/A | Lấy Toàn văn (Parent Chunk) của Điều | Lấy chính xác các Child Chunks có mối liên hệ qua KG |

---

## 2. Chi tiết cơ chế Article Expansion
*Trả lời các câu hỏi về luồng mở rộng Điều luật:*

- **Wider candidate set là top-k bao nhiêu?**
  Hệ thống sử dụng một tập rộng hơn là 20 chunks (được tính bằng `min(top_k * 4, prefetch_limit)`) để đảm bảo phủ đủ các Điều luật khác nhau.
- **Article score được aggregate thế nào?**
  Điểm của một Điều (Article) được tính bằng **giá trị lớn nhất (`max`)** trong các điểm số của các child chunks thuộc về Điều đó xuất hiện trong tập candidate.
- **Threshold bao nhiêu và chọn tối đa bao nhiêu Articles?**
  Hệ thống chỉ mở rộng các Điều có điểm số đạt ít nhất **40%** so với Điều có điểm cao nhất (`article_expand_score_ratio = 0.4`), và lấy tối đa **5 Điều** (bằng đúng tham số `top_k`).
- **Full Articles được sắp xếp theo thứ tự nào?**
  Các Điều sau khi được mở rộng toàn văn sẽ được sắp xếp theo **thứ tự giảm dần của Article score** (tương đương với thứ tự Qdrant search score gốc).

---

## 3. Chi tiết cơ chế KG Critic
*Trả lời các câu hỏi về luồng Critic Agent kết hợp Knowledge Graph:*

- **Chọn candidate Articles bằng policy gì và lấy bao nhiêu Điều?**
  Critic chỉ kiểm tra các Điều (trong top-5 chunks ban đầu) có điểm số đạt ít nhất **60%** so với Điều cao điểm nhất (`critic_score_ratio = 0.6`). Số lượng Điều tối đa được kiểm tra là **4** (`critic_max_dieu`). Việc dùng threshold tỉ lệ này giúp loại bỏ các Điều nhiễu (điểm thấp hẳn) để không kéo theo các tham chiếu sai lệch làm loãng ngữ cảnh.
- **Same-Article evidence gap được xác định bằng rule nào?**
  Sử dụng hai rules về cấu trúc trên Knowledge Graph (Neo4j):
  1. `compound_penalty_behaviors`: Phát hiện hành vi có cả hình phạt chính và hình phạt bổ sung/khắc phục hậu quả (thường nằm ở các khoản khác nhau) nhưng chưa được retrieve đủ.
  2. `structurally_incomplete_dieu`: Phát hiện Điều có nhiều Khoản/Điểm (tổng số phần) nhưng số chunk retrieve được thực tế ít hơn tổng số phần đó.
- **Binary semantic relevance gate dùng model nào và Threshold là gì?**
  Cổng lọc ngữ nghĩa sử dụng **chung LLM chính của pipeline** (ví dụ: Qwen2.5 7B) thông qua một lệnh gọi độc lập. Kết quả phân loại là **nhị phân (Binary Yes/No)**, không dùng xác suất (threshold).
- **Prompt của Semantic Gate là gì?**
  Prompt yêu cầu mô hình phân loại với chỉ lệnh chặt chẽ: *"Bạn là trợ lý kiểm tra độ liên quan, CHỈ làm đúng 1 việc: xác định 1 đoạn văn bản pháp luật có THỰC SỰ CẦN THIẾT để trả lời ĐÚNG câu hỏi hay không... Chỉ trả lời đúng 1 từ: 'yes' hoặc 'no'."*
- **Candidate evidence retrieval lấy bao nhiêu chunks?**
  Thay vì nhồi toàn văn cả Điều (Parent Chunk) như Article Expansion gây nhiễu, Critic Agent tận dụng Knowledge Graph để chỉ định vị và fetch **chính xác các Child Chunks có mối liên hệ** (ví dụ: chỉ lấy đúng Khoản/Điểm chứa hình phạt bổ sung đang thiếu, hoặc đúng Khoản được tham chiếu chéo) trước khi đưa qua Semantic Gate. Điều này giúp giữ ngữ cảnh cuối cùng cực kỳ gọn và sạch.
- **Regeneration prompt là gì?**
  Khi sinh lại (regenerate), hệ thống **tái sử dụng nguyên văn prompt sinh câu trả lời gốc**. Tuy nhiên, thay vì yêu cầu LLM hợp nhất câu trả lời nháp và phần bổ sung (rất dễ gây rơi rớt thông tin), hệ thống sẽ **sinh lại từ đầu** bằng cách ghép phần context bổ sung của Critic vào ngữ cảnh gốc. Điều này đảm bảo cơ chế sinh câu trả lời của 3 kịch bản là hoàn toàn đồng nhất, chỉ khác nhau ở dữ liệu ngữ cảnh đầu vào.
