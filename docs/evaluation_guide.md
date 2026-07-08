# Hướng dẫn đánh giá (chạy trên Colab)

## 1. Bộ test đã có sẵn

`data/eval_testset.jsonl` — **50 câu hỏi thật**, lấy từ nội dung thật đã ingest (không tự bịa), chia 4 nhóm:

| Nhóm | Số case | Mục đích |
|---|---|---|
| `same_dieu_compound_penalty` | 13 | 1 Điều có cả hình phạt chính LẪN bổ sung (thường khác Khoản) |
| `cross_reference` | 13 | 1 Điều tham chiếu sang Điều KHÁC hẳn |
| `structural_multi_part` | 12 | 1 Điều có rất nhiều Khoản/Điểm liên quan (không phải chế tài, không phải tham chiếu) |
| `control_no_gap` | 12 | Điều đơn giản, tự đủ — nhóm đối chứng, kiểm tra Critic Agent không lãng phí khi không cần |

Mỗi dòng JSONL có: `id`, `category`, `dieu_ids` (Điều nguồn), `van_ban`, `question`, `required_facts` (checklist fact bắt buộc phải có trong câu trả lời đầy đủ), `reference_answer` (đáp án mẫu, dùng làm `ground_truth`/`reference` cho RAGAS).

File nguồn đầy đủ theo từng nhóm: `data/test_cat1_compound_penalty.json`, `test_cat2_cross_ref.json`, `test_cat3_structural.json`, `test_cat4_control.json`. Dữ liệu thô (nội dung Điều thật trước khi viết câu hỏi) nằm ở `data/test_gather_raw.json` — giữ lại để đối chiếu, không cần dùng khi chạy đánh giá.

## 2. Yêu cầu môi trường trên Colab

Pipeline cần 3 thứ chạy được (giống hệt môi trường Docker cục bộ):

1. **Qdrant (embedded)** — chỉ cần copy/upload thư mục `data/.qdrant` (đã ingest sẵn embedding) lên Colab, không cần server riêng (`qdrant-client` đọc thẳng từ thư mục local).
2. **Neo4j** — Colab không có Docker mặc định. Cách khả thi:
   - Cài Neo4j trực tiếp trong Colab bằng `apt-get`/`wget` bản Neo4j Community (nhiều hướng dẫn dạng "Neo4j on Google Colab" có sẵn), rồi restore dữ liệu bằng cách chạy lại `scripts/run_neo4j_ingest.py` + `scripts/build_cross_references.py` (cần thư mục `data/extracted_json`) trên Neo4j vừa cài trong Colab.
   - Hoặc dùng Neo4j AuraDB (free tier, cloud) và trỏ `NEO4J_URI` tới đó.
3. **LLM** — có 2 lựa chọn:
   - Cài Ollama trong Colab (`pip install ollama` hoặc script cài nhị phân) + `ollama pull qwen2.5:7b`, chạy nền `ollama serve` — tận dụng GPU Colab để suy luận nhanh hơn hẳn CPU cục bộ.
   - Hoặc đổi sang 1 LLM API khác (OpenAI/Anthropic) bằng cách sửa `build_llm()` trong `scripts/run_chatbot.py` — cấu trúc `ChatbotPipeline` không phụ thuộc cụ thể vào Ollama, chỉ cần 1 đối tượng LangChain `BaseChatModel` bất kỳ.

Set biến môi trường trước khi chạy (nếu khác giá trị mặc định):
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=legal_kg_2024
export OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_MODEL=qwen2.5:7b
```

## 3. Chạy đánh giá

```bash
pip install -r requirements.txt
pip install ragas datasets   # cho bước tính điểm RAGAS

# Bước 1: chạy pipeline thật qua cả 3 kịch bản, thu thập câu trả lời + ngữ cảnh
python scripts/run_evaluation.py --all-modes
#   --resume     : bỏ qua case đã chạy xong (nếu bị đứt giữa chừng)
#   --limit N    : test nhanh N câu đầu trước khi chạy full 50 câu
#   --top-k K    : đổi top_k retrieval (mặc định 5)

# Output: data/eval_results_naive.jsonl, eval_results_article_expand.jsonl, eval_results_critic.jsonl
```

Lưu ý: 50 câu × 3 kịch bản = 150 lượt gọi pipeline (kịch bản `critic` có thể tốn 2 lượt LLM/câu nếu phát hiện thiếu). Trên CPU máy cá nhân một câu mất 1-5 phút; trên GPU Colab (nếu đổi LLM chạy có GPU, hoặc Ollama tận dụng GPU) sẽ nhanh hơn đáng kể.

```bash
# Bước 2: chấm điểm (RAGAS + Legal Completeness Rate tùy biến)
python scripts/score_evaluation.py --modes naive article_expand critic
```

Output:
- `data/eval_scores_<mode>.csv` — chi tiết từng câu (completeness_rate).
- `data/eval_summary.csv` — bảng tổng hợp theo mode × category, gồm cả `completeness_rate` và các chỉ số RAGAS (`ragas_faithfulness`, `ragas_answer_relevancy`, `ragas_context_precision`, `ragas_answer_correctness`) nếu đã cài `ragas`.

## 4. Cách đọc kết quả

So sánh `completeness_rate` theo **category**, không chỉ theo tổng trung bình:

- Ở nhóm `control_no_gap`: cả 3 kịch bản nên có completeness_rate gần bằng nhau (~cao) — nếu `critic` thấp hơn `naive` ở nhóm này, nghĩa là Critic Agent đang "sửa hỏng" câu trả lời vốn đã đúng (regression cần điều tra).
- Ở nhóm `same_dieu_compound_penalty` và `cross_reference`: kỳ vọng `critic` > `naive` rõ rệt — đây là bằng chứng định lượng cho đóng góp chính của khóa luận.
- Ở nhóm `structural_multi_part`: kỳ vọng `critic` ≈ `article_expand` (cả 2 đều mở rộng toàn Điều nên đủ ngữ cảnh) nhưng `critic` nên gọn hơn ở các nhóm khác nhờ chỉ mở rộng khi thật sự phát hiện thiếu.
- Ở nhóm `cross_reference` và `same_dieu_compound_penalty` (khi thông tin còn thiếu nằm ở ĐIỀU KHÁC): kỳ vọng `critic` > `article_expand` rõ rệt — `article_expand` chỉ mở rộng trong phạm vi Điều đã retrieve, không tự tìm được Điều liên quan khác, đây là bằng chứng định lượng cho giá trị của việc dùng Knowledge Graph.
- `ragas_faithfulness` thấp ở `naive` là bằng chứng định lượng cho hiện tượng hallucination đã quan sát thấy thủ công (trích dẫn "khoản 3a.71" bịa, "Khoản 4 Điều 36" bịa).

## 5. Giới hạn cần nêu khi viết khóa luận

- `required_facts` và `reference_answer` do LLM (agent) soạn từ nội dung thật, không phải chuyên gia luật thẩm định — nên nêu rõ đây là "silver standard" tự động, không phải "gold standard" do luật sư kiểm duyệt, nếu có thời gian nên có 1 vòng review thủ công trước khi công bố số liệu chính thức.
- Completeness Rate dùng LLM-as-judge (cùng 1 model judge cho mọi kịch bản) — có rủi ro thiên vị nếu judge và generator là cùng 1 model; lý tưởng nên dùng model khác (mạnh hơn) làm judge nếu có điều kiện.
- Nhóm `same_dieu_compound_penalty` toàn bộ lấy từ Nghị định 15/2020/NĐ-CP (chỉ văn bản này có đủ pattern chế tài kép rõ ràng trong dữ liệu hiện có) — kết quả nhóm này phản ánh đặc điểm của loại văn bản xử phạt hành chính, chưa chắc tổng quát hóa sang loại văn bản khác.
