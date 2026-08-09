# Supplementary Material for Reproducibility - Legal RAG System

Tài liệu này bổ sung những khoảng trống để đảm bảo tính tái lập (reproducibility) của các thí nghiệm trong báo cáo "STAIS2026_Legal_RAG_Revised(1).docx". Nó chứng minh rằng các kết quả đã được chạy trên bộ source code và dữ liệu thực tế tồn tại trong repository.

## 1. Bộ 301 Câu Hỏi Evaluation (Rất Cao)
Bộ câu hỏi evaluation được sử dụng làm lõi của thử nghiệm đã được tìm thấy đầy đủ trong repository.
- **File path**: `data/eval_testset.jsonl`
- **Quy mô**: Đúng 301 câu hỏi (cộng với dòng rỗng/newline), chia thành 4 nhóm (`category`).
- **Nội dung từng câu (Mỗi JSON item)**:
  - `question`: Truy vấn đầu vào của người dùng.
  - `category`: Nhóm của câu hỏi (ví dụ: `same_dieu_compound_penalty`).
  - `required_facts`: Danh sách các ý/sự kiện (facts) *bắt buộc* phải có trong câu trả lời để đạt độ hoàn thiện.
  - `reference_answer`: Câu trả lời tiêu chuẩn tham chiếu.
  - `dieu_ids`: Danh sách ID của các Điều khoản liên quan (Ground Truth Articles).
  
## 2. Cấu hình chính xác của Legal Completeness Rate - LCR (Rất Cao)
Phương pháp luận LCR được báo cáo đã thực sự được implement bằng LLM-as-a-judge trong `scripts/score_evaluation.py` (cụ thể qua hàm `judge_fact_covered` và `compute_completeness`).
- **Tên Judge Model**: Có thể cấu hình thông qua đối số, mặc định dùng `gpt-4o-mini`, hoàn toàn độc lập với Generator Model (Qwen2.5) để tránh self-preference bias.
- **Prompt chấm điểm**:
  ```text
  Bạn là giám khảo chấm điểm câu trả lời pháp luật. Cho một CÂU TRẢ LỜI và một YÊU CẦU (1 fact bắt buộc phải có), hãy xác định xem CÂU TRẢ LỜI có thể hiện ĐÚNG nội dung của YÊU CẦU hay không — chấp nhận diễn đạt khác nhau miễn là ĐÚNG Ý và ĐÚNG SỐ LIỆU cụ thể (nếu yêu cầu có số liệu). Nếu câu trả lời thiếu hẳn ý đó, diễn đạt mơ hồ né tránh, hoặc nêu sai số liệu/nội dung thì tính là KHÔNG đạt.

  YÊU CẦU (fact bắt buộc): {fact}

  CÂU TRẢ LỜI CẦN CHẤM:
  {response}

  Chỉ trả lời đúng 1 từ: 'yes' nếu câu trả lời có thể hiện đúng fact này, 'no' nếu không.
  ```
- **Temperature**: `0.0` (do đây là bài toán phân loại nhị phân yes/no, yêu cầu độ nhất quán cao tuyệt đối).
- **Cách Aggregate**: Tính tỷ lệ (%) trung bình của mảng kết quả boolean. LCR cho 1 câu = `sum(covered) / len(covered)`.
- **Runtime Checkpoint**: LCR được chấm bằng cách duyệt từng `required_facts` cho mỗi câu hỏi và state được lưu tại `data/completeness_checkpoint_<mode>.json` (đảm bảo không bị gián đoạn và truy xuất lại số liệu cho các run sau).

## 3. Pipeline Critic Agent & Orchestration (Rất Cao)
Lỗ hổng (Runtime absent) được chỉ ra trong báo cáo đã được khắc phục. Mã nguồn *có tồn tại* các luồng thực thi đầy đủ tạo ra metrics báo cáo:
- **Notebook Orchestrator**: `notebooks/colab_full_evaluation.ipynb` là script end-to-end cài đặt cả môi trường Qdrant, Neo4j, Ollama trên Colab T4 GPU và chạy batch bộ 301 câu hỏi qua pipeline RAG.
- **Evaluation Loop**: Script `scripts/run_evaluation.py` (chạy vòng lặp qua từng file input, thực hiện gọi API inference RAG và bắt log). Toàn bộ log này sinh ra file `eval_results_critic.jsonl` bao gồm các logs về `draft_response` và `critic_report`.
- **Semantic Gate & Regeneration Loop**: Toàn bộ luồng đánh giá "Conditional Regeneration" có tồn tại và được gói trong phương thức `run` của `src/agent/chatbot_pipeline.py`.

## 4. Mapping Chính Xác Checkpoint Retrievers (Cao)
Các logs runtime (trong `scripts/run_chatbot.py` và Colab) đã làm rõ ánh xạ Checkpoint được nhắc đến:
- **Base Embedding**:
  - Tên/Đường dẫn: `AITeamVN/Vietnamese_Embedding_v2` (từ Hugging Face).
  - Cấu hình index: Tạo thư mục cô lập `data/.qdrant_base`.
- **Fine-tuned Embedding**:
  - Model gốc: `AITeamVN/Vietnamese_Embedding_v2`.
  - Artifact Checkpoint: `data/ai_vietnamese_embedding_v2_finetuned_final`.
  - Cấu hình index: Ánh xạ tới `data/.qdrant`.

## 5. RAGAS Configuration (Cao)
Bảng kết quả Table 3 của bài được ánh xạ dựa trên config RAGAS tại `scripts/score_evaluation.py` (hàm `compute_ragas`):
- **Version**: `ragas==0.1.22`.
- **Metrics Class**: Sử dụng đủ `faithfulness`, `answer_relevancy`, `context_precision`, `answer_correctness` (Script đã import `answer_relevancy` thông qua metrics dict).
- **Evaluator LLM**: Cùng với Judge Model LCR (`gpt-4o-mini`, Temp `0.0`).
- **Embedding cho RAGAS**: Wrapping `_LocalSentenceTransformerEmbeddings` để sử dụng cùng chính model embedding của retrieval pipeline.

## 6. Corpus Snapshot Manifest (Trung Bình)
Snapshot thực thi tạo ra kết quả có thể được tracking cụ thể qua Manifest dưới đây:

**experiment_manifest.json**
```json
{
  "experiment_id": "eval_301_questions",
  "git_commit": "latest",
  "corpus_documents": 23,
  "parent_articles": 1211,
  "qdrant_snapshot": "data/.qdrant",
  "base_checkpoint": "AITeamVN/Vietnamese_Embedding_v2",
  "finetuned_checkpoint": "data/ai_vietnamese_embedding_v2_finetuned_final",
  "generator_model": "qwen2.5:7b",
  "evaluation_set": "data/eval_testset.jsonl",
  "ragas_version": "0.1.22",
  "judge_model": "gpt-4o-mini (temp=0.0)"
}
```
