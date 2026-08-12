# Audit con số 21/23 bản ghi Critic

Ngày audit: 2026-08-08  
Repository: `/Users/nguyengiahuy/Project/LEGAL_IT_CHATBOT`  
Khóa luận nguồn được kiểm tra: `/Users/nguyengiahuy/Desktop/KLTN_dcs.docx`  
Git commit hiện tại: `9bf145e6744599e1e15c21e2011a02f7ecff87f5`

## 1. Kết luận chính

1. Không tìm thấy tệp dữ liệu có đúng 21 record trong repository hiện tại. Vì vậy, nguồn của con số 21 chưa đủ bằng chứng để xác minh.
2. Tệp duy nhất có đúng 23 dòng/23 JSON record là:
   `/Users/nguyengiahuy/Project/LEGAL_IT_CHATBOT/data/eval_results_critic_p0_base_qwen25_20260731.jsonl`.
3. Tệp 23 record này **không phải chat history**. Nội dung của nó khớp schema output mà `scripts/run_evaluation.py` ghi cho mode `critic`: mỗi record có ID câu hỏi, câu hỏi, câu trả lời, đáp án tham chiếu, ngữ cảnh truy xuất, ID Điều, token, bản nháp và báo cáo Critic.
4. Tệp 23 record chỉ chứng minh rằng **raw artifact Critic đang lưu trong repository là một tập con 23 câu**. Nó không đủ để kết luận toàn bộ thí nghiệm Critic chỉ chạy 23 câu hoặc dừng vĩnh viễn ở câu thứ 23.
5. Không tìm thấy raw output Critic đủ 301 câu, checkpoint LCR/RAGAS đủ 301 câu hoặc quyết định judge theo từng câu trong repository hiện tại.
6. Vì vậy, các số LCR, RAGAS, Article precision/recall và tài nguyên của phép đánh giá 301 câu vẫn phải giữ trạng thái **REPORTED**, nhưng lý do phải diễn đạt đúng: repository chưa lưu đầy đủ raw evaluation/scoring artifacts cần thiết để tái tính độc lập; không được dùng câu “Critic chỉ chạy 23/301 câu”.

## 2. Các nguồn đã kiểm tra

### 2.1. Mã chạy và chấm đánh giá

| Tệp | SHA-256 | Kết quả kiểm tra |
|---|---|---|
| `/Users/nguyengiahuy/Project/LEGAL_IT_CHATBOT/scripts/run_evaluation.py` | `d2f5e9b3e2f1db45a5de8a4430edfda3c7cbe46f70033fb245d92fb91b0074d8` | Ghi một JSON record cho mỗi case chạy thành công; mode Critic có thêm `draft_response` và `critic_report`. Exception của từng case được bỏ qua, nên số dòng output có thể nhỏ hơn testset. |
| `/Users/nguyengiahuy/Project/LEGAL_IT_CHATBOT/scripts/score_evaluation.py` | `6c244f542c89688920992512b9650fcfeb9e96076b054e5e8dd7cad6bb12ca32` | Tính LCR bằng LLM-as-judge, có checkpoint theo ID; tính RAGAS nếu dependency/API sẵn sàng; tổng hợp token và số lượt gọi từ raw result. |

### 2.2. Testset và raw evaluation output

| Tệp | Số record | Schema chính | Phân loại |
|---|---:|---|---|
| `/Users/nguyengiahuy/Project/LEGAL_IT_CHATBOT/data/eval_testset.jsonl` | 301 | `id`, `category`, `question`, `reference_answer`, `required_facts`, `dieu_ids`, `van_ban` | Testset 301 câu, VERIFIED. SHA-256: `58e999c2a33f762e352ef6f477ea1ef49784b3cfa7c9b291ace24e354eba1b96`. |
| `/Users/nguyengiahuy/Project/LEGAL_IT_CHATBOT/data/eval_results_naive_p0_base_qwen25_20260731.jsonl` | 301 | `id`, `mode`, `user_input`, `response`, `reference`, `retrieved_contexts`, `required_facts`, `dieu_ids`, `retrieved_dieu_ids_ranked`, `graph_fetched_dieu_ids`, `token_usage`, `final_answer_token_usage` | Raw evaluation Naive đủ 301 record. |
| `/Users/nguyengiahuy/Project/LEGAL_IT_CHATBOT/data/eval_results_article_expand_p0_base_qwen25_20260731.jsonl` | 301 | Như Naive | Raw evaluation Article Expansion đủ 301 record. |
| `/Users/nguyengiahuy/Project/LEGAL_IT_CHATBOT/data/eval_results_critic_p0_base_qwen25_20260731.jsonl` | 23 | Schema Naive/Expansion cộng `draft_response`, `critic_report`; mọi record có `mode="critic"` | Raw evaluation Critic dạng tập con, không phải chat history. SHA-256: `4f9db630aa86aacefaa95e93207d114ab0ad7fcfc61aa8d38d8e893894d67855`. |
| `data/eval_results_*_p0_smoke_20260731.jsonl` | 1 record/mode | Schema evaluation tương ứng | Smoke test, không phải run 301 câu. |
| `data/completeness_checkpoint_*_p0_smoke_20260731.json` | 1 record/mode | `completeness_rate`, `facts_covered` theo ID | Chỉ checkpoint smoke test. |
| `data/eval_scores_*_p0_smoke_20260731.csv` | 1 record/mode | `id`, `category`, `completeness_rate`, `response_preview` | Chỉ điểm smoke test. |

Không tìm thấy `ragas_checkpoint_<mode>...json`, checkpoint LCR đầy đủ, `eval_scores_<mode>.csv` đầy đủ hoặc `eval_summary.csv` đầy đủ cho run 301 câu.

### 2.3. Chat/UI history

Tệp chat history thực sự là:
`/Users/nguyengiahuy/Project/LEGAL_IT_CHATBOT/data/chainlit_history.db`.

- Loại: SQLite database của Chainlit.
- SHA-256 tại thời điểm audit: `e2283d5809c76e04b250755c05762e51f1668da6f97ea13ae9b46abf50eeff1b`.
- Bảng và số hàng tại thời điểm audit:
  - `threads`: 25;
  - `steps`: 334;
  - `users`: 1;
  - `elements`: 0;
  - `feedbacks`: 0.
- Trong `threads`, có 22 thread mang metadata mode/profile `critic`, nhưng đây là phiên UI; không phải 22 evaluation case.
- Không có chuỗi `cat1_02` hoặc `cat1_03` trong `steps`.
- Schema (`threadId`, `input`, `output`, `createdAt`, metadata giao diện...) khác hoàn toàn schema evaluation JSONL.

Kết luận: chat history và raw evaluation là hai nguồn độc lập. Không có bằng chứng cho thấy tệp 23 dòng là dữ liệu Chainlit.

## 3. Kiểm tra `cat1_02` và `cat1_03`

Hai ID xuất hiện trong đúng năm tệp dữ liệu hiện tại:

1. `data/eval_testset.jsonl`;
2. `data/eval_testset.json`;
3. `data/eval_results_naive_p0_base_qwen25_20260731.jsonl`;
4. `data/eval_results_article_expand_p0_base_qwen25_20260731.jsonl`;
5. `data/eval_results_critic_p0_base_qwen25_20260731.jsonl`.

Trong raw Critic:

- `cat1_02` có `mode="critic"`, câu trả lời, sáu ngữ cảnh, token, bản nháp và `critic_report` chứa cả `reinforced_top_dieu_multi_van_ban` và `rejected_by_relevance_gate`.
- `cat1_03` có `mode="critic"`, câu trả lời, sáu ngữ cảnh, token, bản nháp và `critic_report` chứa `rejected_by_relevance_gate`.

Vì vậy, việc dùng hai record này để minh họa nhánh safeguard/gate-reject là có nguồn raw evaluation trực tiếp. Không sửa đoạn minh họa này trong khóa luận.

## 4. Tệp 23 record thực sự là gì?

### A. Nguồn

`/Users/nguyengiahuy/Project/LEGAL_IT_CHATBOT/data/eval_results_critic_p0_base_qwen25_20260731.jsonl`.

### B. Schema

`category`, `critic_report`, `dieu_ids`, `draft_response`, `final_answer_token_usage`, `graph_fetched_dieu_ids`, `id`, `mode`, `reference`, `required_facts`, `response`, `retrieved_contexts`, `retrieved_dieu_ids_ranked`, `token_usage`, `user_input`.

### C. Chức năng

Đây là **partial Critic evaluation output**. Nó không phải chat history, UI history hoặc file log thuần túy. Nó cũng không phải file điểm LCR/RAGAS, vì chưa chứa quyết định judge hay điểm RAGAS theo từng câu.

23 ID tương ứng đúng 23 case đầu trong thứ tự hiện tại của testset: `cat1_01` đến `cat1_13`, sau đó `cat2_01` đến `cat2_10`. Điều này phù hợp với một lần chạy batch bị ngắt hoặc chỉ lưu một phần. Tuy nhiên, repository không có log đủ để xác định nguyên nhân dừng; do đó: **Chưa đủ bằng chứng để xác minh vì sao artifact chỉ có 23 record.**

### D. Có output Critic đủ 301 câu không?

Không tìm thấy trong repository hiện tại. Cũng không tìm thấy output 301 câu trong notebook `colab_full_evaluation.ipynb`; notebook không lưu cell output. Các tài liệu bài báo/Metrics-Fix chứa bảng tổng hợp 301 câu nhưng không chứa raw Critic 301 record hoặc raw judge decisions.

## 5. Đối chiếu raw artifact với số liệu tổng hợp

Raw output hiện có cho kết quả mô tả sau:

| Mode | n raw | Avg total tokens | Avg final-call tokens | Avg online calls | Avg context characters (tổng độ dài chuỗi context) |
|---|---:|---:|---:|---:|---:|
| Naive | 301 | 1.356,96 | 1.356,96 | 1,00 | 1.603,38 |
| Article Expansion | 301 | 3.166,40 | 3.166,40 | 1,00 | 8.353,66 |
| Critic artifact hiện có | 23 | 8.091,26 | 2.701,78 | 7,35 | 5.372,30 |

Các giá trị này không trùng toàn bộ với bảng tổng hợp đang báo cáo (ví dụ Critic 5,7 calls/câu và 2.187 final-call tokens). Do đó, không được dùng 23 record để tái tạo hoặc xác nhận bảng tổng hợp 301 câu; nhiều khả năng đây là artifact của cấu hình/run khác hoặc một phần của run. Chưa đủ bằng chứng để xác minh quan hệ chính xác giữa hai run.

## 6. Phân loại trạng thái bằng chứng

Quy ước:

- **VERIFIED**: tái tính/đối chiếu trực tiếp được từ raw artifact hiện tại.
- **REPORTED**: có trong bài báo hoặc Metrics-Fix nhưng raw artifact hiện tại chưa đủ để tái tính độc lập.
- **UNVERIFIED**: không tìm thấy nguồn đủ đáng tin để giữ như một kết quả.

| Claim/metric | Trạng thái | Bằng chứng và lý do |
|---|---|---|
| Testset có 301 câu | VERIFIED | Đếm trực tiếp `data/eval_testset.jsonl`; SHA-256 và bốn nhóm 51/16/130/104 được đối chiếu. |
| Raw Naive có 301 record | VERIFIED | Đếm và kiểm tra ID/schema trực tiếp. |
| Raw Article Expansion có 301 record | VERIFIED | Đếm và kiểm tra ID/schema trực tiếp. |
| Raw Critic artifact hiện lưu có 23 record | VERIFIED | Đếm và kiểm tra schema/ID trực tiếp. Đây không đồng nghĩa “toàn bộ Critic chỉ chạy 23 câu”. |
| “Critic chỉ chạy/dừng ở 21 hoặc 23 trên 301 câu” | UNVERIFIED | Repository không có run log hoặc bằng chứng đủ để kết luận phạm vi thực thi toàn bộ. |
| LCR 43,77% / 72,72% / 81,33% | REPORTED | Có trong `Metrics-Fix.docx` và bản bài báo cuối; không có judge decisions/checkpoint đầy đủ để tái tính. |
| LCR theo bốn nhóm | REPORTED | Có trong `Metrics-Fix.docx`/bài báo; thiếu raw Critic 301 và raw judge. |
| RAGAS | REPORTED | Có bảng tổng hợp; không có checkpoint/per-question RAGAS đủ để tái tính. |
| Article precision / Article recall | REPORTED | Có bảng tổng hợp; công thức trên raw artifact hiện tại không tái tạo được đúng các giá trị báo cáo và Critic chỉ có raw subset. |
| Token, context characters, 5,7 online calls/câu | REPORTED | Có trong nguồn tổng hợp; raw artifacts hiện tại cho các trung bình khác và không đủ Critic 301. |
| Giảm 16,5% final-call token | REPORTED (phép tính đã kiểm tra) | `2.618 -> 2.187` cho 16,4629%, làm tròn 16,5%; hai số đầu vào vẫn là REPORTED. |
| Giảm 42,8% context characters | REPORTED (phép tính đã kiểm tra) | `8.296 -> 4.748` cho 42,7676%, làm tròn 42,8%; hai số đầu vào vẫn là REPORTED. |
| +17,48 điểm phần trăm nhóm chế tài kép | REPORTED (phép tính đã kiểm tra) | `75,49 - 58,01 = 17,48`; hai LCR đầu vào vẫn là REPORTED. |
| +21,35 điểm phần trăm nhóm tham chiếu chéo | REPORTED (phép tính đã kiểm tra) | `70,83 - 49,48 = 21,35`; hai LCR đầu vào vẫn là REPORTED. |
| Con số 21 record | UNVERIFIED | Không tìm thấy tệp 21 record trong repository hiện tại. |

Không có metric nào trong danh sách được chuyển tự động từ REPORTED sang VERIFIED chỉ vì đã sửa cách diễn giải con số 23.

## 7. Các câu trong khóa luận cần sửa

Nguồn: `/Users/nguyengiahuy/Desktop/KLTN_dcs.docx`.

1. Tóm tắt, paragraph 46: bỏ quan hệ nhân quả duy nhất “vì Critic chỉ có 23/301”; thay bằng lý do đúng là thiếu raw Critic 301 và raw scoring artifacts.
2. Mục 3.9, paragraph 269: thay “chỉ 23 bản ghi cho Critic” bằng mô tả artifact đang lưu là tập con; bỏ hàm ý toàn bộ thí nghiệm chỉ chạy 23 câu.
3. Mục 4.6, paragraph 340: bỏ “Critic dừng ở 23 dòng”; ghi rõ repository chỉ lưu raw Critic dạng tập con và không có log để xác định toàn bộ run.
4. Mục 4.9, paragraph 349: làm rõ “chưa lưu trong repository raw Critic đủ 301 câu” thay vì suy ra phạm vi thực thi.
5. Mục 5.3, paragraph 363: thay “thiếu đầu ra Critic đầy đủ” bằng “thiếu raw artifact/judge decisions đủ để phân tích theo từng câu”.
6. Mục 5.9, paragraph 388: bỏ câu độc lập “Đầu ra Critic hiện chỉ có 23 bản ghi”; giữ đúng các giới hạn repeated runs/confidence interval/paired test và raw scoring.
7. Mục 6.1, paragraph 400: bỏ suy luận “23/301 nên LCR/RAGAS là REPORTED”; thay bằng lý do thiếu raw Critic 301 và raw judge/checkpoint.
8. Mục 6.2, paragraph 403: thay “chạy đủ 301 câu” bằng “phát hành đầy đủ raw artifact 301 câu”; repository không chứng minh rằng run 301 chưa từng được thực hiện.

Đoạn ở paragraph 314 về `cat1_02` và `cat1_03` **không cần sửa**, vì hai ID thực sự tồn tại trong raw Critic evaluation và có đúng critic trace được dùng để minh họa.

## 8. Vì sao có thể bị hiểu nhầm?

Ba nguồn cùng tồn tại nhưng có chức năng khác nhau:

1. `chainlit_history.db`: lịch sử UI/chat;
2. `eval_results_critic...jsonl`: raw output của evaluation runner;
3. `Metrics-Fix.docx` và bài báo: bảng tổng hợp kết quả 301 câu.

Việc chỉ nhìn số lượng record hoặc tên “history/log/output” mà không kiểm tra schema có thể dẫn tới trộn ba lớp dữ liệu. Audit này phân loại theo nội dung, ID và trường dữ liệu thay vì tên file.
