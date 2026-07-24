# Phương pháp kiểm định độ tin cậy của bộ test set (`data/eval_testset.jsonl`)

## 1. Mục đích

Bộ 301 câu hỏi dùng làm ground-truth xuyên suốt khóa luận (`data/eval_testset.jsonl`) được sinh bằng LLM (Claude Opus 4.8). Vì **Legal Completeness Rate** — chỉ số trung tâm để so sánh 3 kịch bản (naive/article_expand/critic) — được tính hoàn toàn dựa trên `required_facts` và `reference_answer` của bộ test này, độ tin cậy của chính bộ dữ liệu quyết định độ tin cậy của mọi kết quả đánh giá phía sau. Nếu fact/đáp án mẫu sai lệch so với văn bản luật gốc, hoặc câu hỏi bị gán nhầm nhóm, các số liệu Completeness Rate tính ra sẽ không phản ánh đúng thực tế.

Công cụ kiểm định (`scripts/validate_testset.py`, chạy qua `colab_validate_testset.ipynb`) đánh giá CHÍNH bộ test set này — không chạy chatbot, không đo naive/article_expand/critic.

> **Tóm tắt (dùng cho thân bài khóa luận):** Bộ test set đã được kiểm định độ tin cậy qua 9 tiêu chí (existence, schema, duplicate, groundedness, tự nhất quán, chất lượng câu hỏi, độ chính xác nhãn phân loại — chi tiết Phụ lục X), đạt ngưỡng tin cậy cao, đảm bảo tính hợp lệ của ground-truth dùng trong đánh giá.

## 2. Tổng quan phương pháp

Kiểm định gồm 2 lớp độc lập:

- **Lớp A — không cần LLM** (deterministic, chắc chắn tuyệt đối): 3 kiểm tra cấu trúc/thống kê đơn giản, không có sai số.
- **Lớp B — LLM-as-judge**: 6 chỉ số cần khả năng đọc hiểu ngôn ngữ tự nhiên và đối chiếu với văn bản luật gốc, được chấm bởi 1 LLM giám khảo độc lập (`--judge-model`, mặc định `gpt-4o-mini`; kết quả chính thức báo cáo dùng `o3-mini` vì chỉ số B6 — đòi hỏi suy luận nhiều bước qua các khoản tham chiếu chéo trong văn bản dài — cần model có khả năng suy luận mạnh để chấm chính xác).

Cả 2 lớp đối chiếu trực tiếp với **văn bản luật gốc thật** (`data/keep/*.docx`), được parse lại bằng đúng `VBPLChunker` (class dùng để ingest dữ liệu vào Qdrant/KG) — đảm bảo nội dung đối chiếu khớp 100% với những gì hệ thống RAG thực tế nhìn thấy khi retrieval.

## 3. Lớp A — Kiểm tra không cần LLM

### A1. `existence` — Điều luật có tồn tại thật không
Với mỗi `dieu_id` trong `dieu_ids` của từng câu, kiểm tra xem ID đó có khớp với 1 Điều thực sự tồn tại trong corpus đã parse hay không.

- **Ý nghĩa**: bắt các trường hợp Opus 4.8 bịa hoặc ghi sai số Điều/văn bản — đây là lỗi nghiêm trọng nhất có thể xảy ra, vì nếu Điều không tồn tại, toàn bộ câu hỏi đó vô giá trị.
- **Lưu ý kỹ thuật**: `dieu_id` trong `eval_testset.jsonl` dùng quy ước viết thường (theo Neo4j), khác với `dieu_id` do `VBPLChunker` sinh ra (giữ nguyên hoa/thường từ tên file gốc) — phải so khớp không phân biệt hoa/thường.

### A2. `schema` — số lượng `dieu_ids` có khớp category không
Kiểm tra: nhóm `cross_reference` phải có **≥ 2** `dieu_ids` (vì bản chất đòi hỏi kết hợp ≥ 2 Điều); 3 nhóm còn lại (`same_dieu_compound_penalty`, `structural_multi_part`, `control_no_gap`) phải có **đúng 1** `dieu_id` (vì bản chất chỉ xoay quanh 1 Điều duy nhất).

- **Ý nghĩa**: đây là điều kiện cần (không đủ) — vi phạm điều kiện này chắc chắn là lỗi gán nhãn.

### A3. `duplicate` — câu hỏi có bị trùng/gần trùng nhau không
So sánh từng cặp câu hỏi bằng độ tương đồng chuỗi (`difflib.SequenceMatcher`), gắn cờ các cặp có độ tương đồng ≥ 0.9.

- **Ý nghĩa**: đảm bảo 301 câu là 301 tình huống kiểm tra khác biệt thực sự, không phải các biến thể gần như giống hệt nhau làm phồng cỡ mẫu ảo.

## 4. Lớp B — LLM-as-judge

Mỗi câu được đối chiếu với **toàn văn Điều luật gốc** (nối tất cả `dieu_ids` liên quan, lấy nguyên văn từ corpus — không phải suy đoán từ kiến thức nền của model).

### B1. `facts_grounded` — từng fact có đúng với văn bản gốc không
Với mỗi `required_fact`, judge xác định fact đó có được văn bản gốc nêu đúng hay không (chấp nhận diễn đạt lại, nhưng phải đúng ý và đúng số liệu cụ thể).

- **Ý nghĩa**: đây là chỉ số quan trọng nhất — `required_facts` chính là "đề bài" dùng để tính Completeness Rate cho mọi mode; nếu fact sai, điểm Completeness Rate đo được sẽ không có ý nghĩa gì cả dù chatbot trả lời đúng hay sai.

### B2. `answer_grounded` — đáp án mẫu có trung thực với văn bản gốc không
Judge xác định `reference_answer` có bịa thêm nội dung không có trong văn bản gốc, hoặc mâu thuẫn với văn bản gốc hay không.

- **Ý nghĩa**: tương đương chỉ số Faithfulness của RAGAS, nhưng áp dụng lên chính đáp án MẪU (gold answer) thay vì câu trả lời do chatbot sinh ra.

### B3. `facts_covered_by_answer` — đáp án mẫu có nêu đủ mọi fact không
Kiểm tra tính tự-nhất-quán nội bộ: mỗi `required_fact` có thực sự được nhắc tới trong `reference_answer` của chính câu đó hay không.

- **Ý nghĩa**: bắt lỗi 2 trường dữ liệu (`required_facts` và `reference_answer`) bị lệch nhau ngay từ khi sinh dữ liệu — nếu 1 fact "bắt buộc" mà chính đáp án mẫu cũng không nhắc tới, đó là lỗi cấu trúc dữ liệu.

### B4. `natural_clear` — câu hỏi có tự nhiên, rõ ràng không
Judge đánh giá câu hỏi có được diễn đạt tự nhiên, không mơ hồ đa nghĩa, giống cách một người dùng thật sẽ hỏi hay không (không lộ vẻ "câu hỏi đề thi" máy tạo).

- **Ý nghĩa**: đảm bảo tính hiệu lực bên ngoài (external validity) của benchmark — kết quả đo trên các câu hỏi này phản ánh đúng khả năng phục vụ người dùng thật, không chỉ là khả năng "làm bài trắc nghiệm".

### B5. `requires_citation` — có bắt buộc phải biết đúng Điều này mới trả lời được không
Judge đánh giá: để trả lời đúng và đủ, có bắt buộc phải biết đúng nội dung Điều luật cụ thể được trích dẫn hay không (hay chỉ cần kiến thức pháp luật chung chung là đủ, không cần retrieval).

- **Ý nghĩa**: xác nhận câu hỏi thực sự kiểm tra được năng lực RAG/retrieval — nếu câu hỏi trả lời được mà không cần trích dẫn cụ thể, nó không đo được điều cần đo.

### B6. `category_correct` — nhãn category có đúng với nội dung không
Judge được cho xem định nghĩa đầy đủ của nhãn category (bên dưới), rồi đánh giá: để trả lời đúng và đủ câu hỏi này, đáp án có bắt buộc phải khớp đúng cấu trúc nội dung mô tả trong nhãn hay không — **dựa trên chính nội dung văn bản Điều luật, không dựa vào cách hành văn của câu hỏi** (câu hỏi luôn được phép đặt dạng mở, tự nhiên, không tự liệt kê trước nội dung).

- **Ý nghĩa**: 4 nhóm category là trục so sánh chính xuyên suốt khóa luận (mỗi nhóm kiểm tra đúng 1 dạng "gap" cụ thể) — nếu gán nhãn sai, toàn bộ phân tích theo nhóm sẽ bị méo.

**Định nghĩa 4 nhãn category** (dùng làm rubric cho judge):

| Nhãn | Định nghĩa |
|---|---|
| `same_dieu_compound_penalty` | Điều luật được trích dẫn quy định NHIỀU hình thức xử phạt/hậu quả pháp lý khác nhau (hình phạt kép, vd phạt tiền + tịch thu + biện pháp khắc phục) cho CÙNG một hành vi vi phạm. |
| `cross_reference` | Để trả lời đầy đủ, bắt buộc phải kết hợp thông tin từ HAI Điều khác nhau có quan hệ dẫn chiếu qua lại. |
| `structural_multi_part` | Cần thông tin từ NHIỀU khoản/điểm khác nhau trong CÙNG MỘT Điều để trả lời đầy đủ — không thể trả lời đủ chỉ bằng 1 khoản/điểm đơn lẻ. |
| `control_no_gap` | Toàn bộ nội dung cần thiết nằm trong ĐÚNG 1 Điều luật được trích dẫn (có thể cần một hoặc vài khoản trong chính Điều đó) — không cần bất kỳ Điều nào khác, không có khoảng trống nào cần tham chiếu/mở rộng ra ngoài Điều này. `no_gap` ở đây nghĩa là không có gap CROSS-ĐIỀU, không phải yêu cầu chỉ đúng 1 khoản. |

## 5. Cơ chế chấm điểm và checkpoint

- Mỗi câu được chấm qua 3 lệnh gọi LLM riêng biệt: (1) Groundedness (B1+B2+B3 gộp chung 1 lệnh gọi vì cùng cần đối chiếu văn bản gốc), (2) Question Quality (B4+B5), (3) Category Correctness (B6) — tách riêng để mỗi lệnh gọi tập trung đúng 1 loại phán đoán, tránh loãng khi nhồi quá nhiều tiêu chí vào 1 prompt.
- Kết quả từng câu lưu tại `data/testset_validation_checkpoint<suffix>.json` ngay sau khi chấm xong — chạy lại đúng lệnh cũ sẽ chỉ chấm tiếp câu chưa có, không mất chi phí/thời gian chấm lại.
- Cờ `--recheck-category` (hoặc `RECHECK_CATEGORY=True` trong notebook) cho phép chấm lại RIÊNG chỉ số B6 cho câu đã có checkpoint (vd sau khi tinh chỉnh rubric/định nghĩa nhãn), giữ nguyên B1-B5 đã chấm, tiết kiệm 2/3 chi phí so với chấm lại từ đầu.

## 6. Output

| File | Nội dung |
|---|---|
| `data/testset_validation_summary<suffix>.csv` | % đạt của từng chỉ số (A + B), theo từng nhóm category và tổng |
| `data/testset_validation_details<suffix>.csv` | Kết quả chi tiết từng câu — dùng để xem tay các câu bị gắn cờ |
| `data/testset_validation_issues<suffix>.json` | Danh sách đầy đủ các vi phạm A1/A2/A3 + câu không đạt từng chỉ số B (kèm lý do judge đưa ra) |

## 7. Kết quả kiểm định

Kết quả cuối — chấm toàn bộ 301 câu bằng 1 model giám khảo duy nhất (`o3-mini`, phù hợp cho các phán đoán cần suy luận nhiều bước như B6):

**Lớp A** (toàn bộ 301 câu):

| Kiểm tra | Kết quả |
|---|---|
| A1. existence | 0 lỗi — 100% Điều luật trích dẫn tồn tại thật trong corpus |
| A2. schema | 0 lỗi — 100% số lượng `dieu_ids` khớp đúng category |
| A3. duplicate | 2 cặp câu hỏi gần trùng nhau (similarity ≥ 0.9): `cat4_new1_11`↔`cat4_new3_03` (0.94), `cat4_new2_14`↔`cat4_new3_04` (0.95) — đã rà soát thủ công |

**Lớp B** (toàn bộ 301/301 câu):

| Chỉ số | same_dieu_compound_penalty (n=51) | cross_reference (n=16) | structural_multi_part (n=130) | control_no_gap (n=104) | TỔNG (n=301) |
|---|---|---|---|---|---|
| B1. facts_grounded | 99.3% | 100.0% | 99.2% | 99.0% | 99.2% |
| B2. answer_grounded | 100.0% | 100.0% | 99.2% | 99.0% | 99.3% |
| B3. facts_covered_by_answer | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| B4. natural_clear | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| B5. requires_citation | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| B6. category_correct | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Tất cả 6 chỉ số Lớp B đạt 99.0% trở lên, B3-B6 đạt tuyệt đối 100% — xác nhận bộ 301 câu test set đủ tin cậy để dùng làm ground-truth cho toàn bộ thí nghiệm so sánh 3 kịch bản trong khóa luận.

### Các câu còn lại cần xem tay (đã rà soát, không sửa số liệu để né tránh)

- **facts_grounded không đạt** (3 câu): `cat3_04`, `cat1_new3_04`, `cat4_new2_03`.
- **answer_grounded không đạt** (2 câu, trùng lặp với danh sách trên): `cat3_04`, `cat4_new2_03`.
