# Final Answer Output Redesign — Completion Report

Ngày kiểm chứng: 2026-07-27

## Kết luận

Thay đổi đã đạt toàn bộ guardrail của task trên benchmark 30 testcase:

- Citation Accuracy tăng từ 23.56% lên 30.00%.
- Recall@5 giữ nguyên 38.33%; Recall@10 giữ nguyên 41.67%.
- Domain Recall giữ nguyên 57.83%; Wrong Domain Rate giữ nguyên 19.56%.
- Hallucinated Citation giữ nguyên 0.
- Số LLM call trong cấu hình benchmark giữ nguyên 2/case.
- 0/30 output có kết luận mâu thuẫn.
- 30/30 output có đúng sáu section mới và không còn section của template cũ.

Benchmark đầu vào không thay đổi, cùng SHA-256:
`126438a83f9902d950b140f74fd1f527b899ec9dc160ce1163f47c0c5d3e114c`.

## Root cause

Output cũ có ba nguồn kết luận độc lập:

1. Preamble suy ra “có dấu hiệu” từ Behavior Card/Applicability.
2. Nội dung raw generation hoặc extractive fallback tự phân tích lại.
3. Phần trả lời cuối của fallback mặc định dùng kết luận bảo thủ “chưa đủ căn cứ”.

Vì ba nhánh không dùng chung một trạng thái, cùng một câu trả lời có thể vừa xác
nhận hành vi, vừa nói không xác định được dấu hiệu, rồi phủ định bằng kết luận
“chưa đủ căn cứ”. Fallback cũ cũng dùng lexical overlap để tạo câu phân tích dài,
lặp căn cứ và không đưa ra bước xử lý thực tế.

Một lỗi output-format thứ hai ảnh hưởng Citation Accuracy: dòng citation giữ hậu
tố như `(sửa đổi, bổ sung ...)` ngay trước dấu phẩy. Đây là citation hợp lệ về
nội dung nhưng parser benchmark không nhận được. Ngoài ra, khi nhiều source cùng
được giữ, thứ tự citation của raw draft không ổn định và có thể chọn source phụ
thay vì source trực tiếp nhất.

## Thiết kế đã triển khai

### Một nguồn kết luận duy nhất

`answer_assessment` được dựng một lần từ dữ liệu đã có:

- `status`
- `matched_facts`
- `missing_facts`
- `applicable_sources`
- `sanction_available`

Các section hiển thị đều đọc cùng object này. Không section nào tự suy ra một
status khác. Assessment chỉ dùng Behavior Card, facts trong question,
Applicability decisions và final grounded sources.

### Renderer và fallback thống nhất

Raw generation hợp lệ, salvaged draft và safe fallback đều đi qua cùng renderer,
với đúng sáu section:

1. `Kết luận sơ bộ`
2. `Vì sao`
3. `Quy định pháp luật liên quan`
4. `Còn cần làm rõ`
5. `Nên làm gì tiếp theo`
6. `Căn cứ pháp lý`

Khi raw generation bị validator loại, renderer không quay lại template cũ.
Không có source thì status là `NO_MATCH` và không tạo citation. Thiếu source chế
tài không làm mất kết luận về dấu hiệu hành vi.

### Citation budget trong output layer

Trong các source đã được Applicability giữ, renderer ưu tiên:

`KEEP → HIGH → cross_encoder_score → behavior_score → retrieval_score`

Chỉ một source đại diện được chọn mặc định. Source thứ hai chỉ được thêm khi
source đầu tiên dẫn chiếu rõ ràng sang một Điều của luật nền và target đó đã có
trong final context, đồng thời vẫn được Applicability giữ.

Không có retrieval/rerank mới, không dùng ground truth ở runtime và không gọi
LLM. Dòng danh mục căn cứ dùng tên văn bản gốc chuẩn hóa; heading vẫn giữ đầy đủ
provenance về văn bản sửa đổi.

## File đã sửa

Chỉ các file thuộc generation state, final renderer và test output:

- `src/agents/agent_generation/answer_assessment.py`
- `src/agents/agent_generation/node_generate_draft.py`
- `src/agents/agent_generation/node_generate_final.py`
- `src/agents/agent_generation/state.py`
- `src/agents/common/grounded_validation.py`
- `src/workflow/pipeline.py`
- `src/workflow/state.py`
- `tests/test_answer_assessment.py`
- `tests/test_grounded_validation.py`

Không sửa Domain Selection, Query Expansion, Dense/BM25/Hybrid Retrieval, Cross
Encoder, Behavior Extraction, Behavior Gate, Recursive Retrieval, Legal
Relevance, Applicability, benchmark, public API hoặc citation validator.

Các thay đổi đang tồn tại ở `.env.example`,
`src/agents/common/legal_relevance_filter.py` và
`tests/test_legal_relevance_filter.py` có trước task này và không thuộc phần triển
khai output.

## Before — testcase giả giọng

Đây là output đầy đủ của formatter/fallback cũ được replay với chính query và
Điều 7 Khoản 2 Điểm g. Nó tái hiện đúng ba kết luận xung đột đã được báo cáo.

```markdown
**Đánh giá sơ bộ:** Có dấu hiệu thuộc phạm vi điều chỉnh.

## Dấu hiệu đã xác định
- Chưa xác định được dấu hiệu phù hợp từ các dữ kiện hiện có.

> **Lưu ý:** Câu trả lời có thể chưa đầy đủ vì chưa truy xuất được toàn bộ căn cứ pháp luật.

## Tóm tắt tình huống
Một nhân viên dùng AI giả giọng giám đốc gọi cho kế toán yêu cầu chuyển tiền gấp. Nếu sự việc xảy ra thì người đó có thể vi phạm những quy định pháp luật nào?

## Các vấn đề pháp lý
- Xác định bản chất pháp lý của các hành vi được mô tả.
- Xác định điều kiện áp dụng và hậu quả pháp lý trong phạm vi căn cứ đã truy xuất.

## Phân tích
### Luật An ninh mạng 2025 — Điều 7
> “g) Sử dụng trí tuệ nhân tạo hoặc công nghệ mới để giả mạo video, hình ảnh, giọng nói của người khác trái quy định của pháp luật; tạo lập, đăng tải, phát tán thông tin quy định tại khoản 1 Điều này”
>
> (Trích từ: **Luật An ninh mạng 2025**, Điều 7, Khoản 2, Điểm g)

- **Nội dung điều luật:** Trọng tâm của quy định là: g) Sử dụng trí tuệ nhân tạo hoặc công nghệ mới để giả mạo video, hình ảnh, giọng nói của người khác trái quy định của pháp luật; tạo lập, đăng tải, phát tán thông tin quy định tại khoản 1 Điều này.
- **Hành vi phù hợp:** Một nhân viên dùng AI giả giọng giám đốc gọi cho kế toán yêu cầu chuyển tiền gấp
- **Phân tích:** Tình huống mô tả “Một nhân viên dùng AI giả giọng giám đốc gọi cho kế toán yêu cầu chuyển tiền gấp”. Trọng tâm của quy định là: g) Sử dụng trí tuệ nhân tạo hoặc công nghệ mới để giả mạo video, hình ảnh, giọng nói của người khác trái quy định của pháp luật; tạo lập, đăng tải, phát tán thông tin quy định tại khoản 1 Điều này. Khi đặt hai nội dung cạnh nhau, các yếu tố cần đối chiếu cụ thể là “giọng”; việc áp dụng chỉ giới hạn ở các yếu tố được chính đoạn luật này mô tả.
- **Điều kiện còn thiếu:** Đoạn trích không nêu thêm một điều kiện độc lập; kết luận hiện chỉ giới hạn ở việc đối chiếu hành vi “Một nhân viên dùng AI giả giọng giám đốc gọi cho kế toán yêu cầu chuyển tiền gấp” với nội dung đã trích.
- **Căn cứ pháp lý:** (Căn cứ: **Luật An ninh mạng 2025**, Điều 7, Khoản 2, Điểm g)
- **Đánh giá:** 🟡 Chưa đủ điều kiện kết luận: Đoạn trích không nêu thêm một điều kiện độc lập; kết luận hiện chỉ giới hạn ở việc đối chiếu hành vi “Một nhân viên dùng AI giả giọng giám đốc gọi cho kế toán yêu cầu chuyển tiền gấp” với nội dung đã trích. (Căn cứ: **Luật An ninh mạng 2025**, Điều 7, Khoản 2, Điểm g)

## Chế tài
- **Đã xác định được:** Có căn cứ liên quan đến hành vi cần đánh giá.
- **Chưa xác định được:** Chưa có đủ nội dung để xác định toàn bộ hậu quả và cách áp dụng.
- **Muốn kết luận cần:** Bổ sung căn cứ trực tiếp về hậu quả và các điều kiện áp dụng còn thiếu.

## Trả lời câu hỏi của người dùng
### 1. Một nhân viên dùng AI giả giọng giám đốc gọi cho kế toán yêu cầu chuyển tiền gấp. Nếu sự việc xảy ra thì người đó có thể vi phạm những quy định pháp luật nào?
- **Trả lời trực tiếp:** Chưa đủ căn cứ để kết luận dứt khoát câu hỏi này.
- **Vì sao:** Các nguồn hiện có xác định được một phần quy tắc liên quan: Trọng tâm của quy định là: g) Sử dụng trí tuệ nhân tạo hoặc công nghệ mới để giả mạo video, hình ảnh, giọng nói của người khác trái quy định của pháp luật; tạo lập, đăng tải, phát tán thông tin quy định tại khoản 1 Điều này. Việc kết luận chỉ được giới hạn trong các nội dung đó. (Căn cứ: **Luật An ninh mạng 2025**, Điều 7, Khoản 2, Điểm g)
- **Căn cứ:** (Căn cứ: **Luật An ninh mạng 2025**, Điều 7, Khoản 2, Điểm g)
- **Còn thiếu:** Cần bổ sung dữ kiện đáp ứng các điều kiện được nêu trong từng nguồn và căn cứ trực tiếp cho phần câu hỏi chưa được bao phủ.

## Căn cứ pháp lý

- **Luật An ninh mạng 2025**, Điều 7, Khoản 2, Điểm g.
```

## After — acceptance demo thật

Pipeline thật đã retrieve ba final sources. Raw generation và grounding repair
đều không qua validator; output dưới đây vì vậy là bằng chứng trực tiếp rằng
visible safe fallback dùng template mới.

```markdown
## Kết luận sơ bộ

Theo thông tin được cung cấp, hành vi này có dấu hiệu thuộc phạm vi điều chỉnh của Luật An ninh mạng 2025. Các nguồn hiện được truy xuất đủ để nhận diện dấu hiệu hành vi, nhưng chưa đủ để xác định chính xác trách nhiệm dân sự, hành chính, hình sự hoặc mức xử lý.

## Vì sao

- Có sử dụng trí tuệ nhân tạo.
- Có giả mạo giọng nói của người khác.
- Việc giả mạo được dùng để mạo danh người có thẩm quyền và yêu cầu chuyển tiền.

## Quy định pháp luật liên quan

### Luật An ninh mạng 2025 — Điều 7, Khoản 2, Điểm g

> “g) Sử dụng trí tuệ nhân tạo hoặc công nghệ mới để giả mạo video, hình ảnh, giọng nói của người khác trái quy định của pháp luật; tạo lập, đăng tải, phát tán thông tin quy định tại khoản 1 Điều này”

Các tình tiết đã xác định gồm: Có sử dụng trí tuệ nhân tạo; Có giả mạo giọng nói của người khác; Việc giả mạo được dùng để mạo danh người có thẩm quyền và yêu cầu chuyển tiền. Những tình tiết này tương ứng trực tiếp với nội dung được trích dẫn, nên hành vi có dấu hiệu thuộc phạm vi điều chỉnh của quy định này.

## Còn cần làm rõ

- Tiền hoặc tài sản đã được chuyển hay chưa.
- Người thực hiện có mục đích chiếm đoạt tài sản hay không.
- Đã phát sinh thiệt hại thực tế hay chưa.
- Hành vi mới ở giai đoạn chuẩn bị hay đã được thực hiện.
- Việc truy xuất căn cứ hiện chưa đầy đủ; có thể còn quy định liên quan chưa được đưa vào ngữ cảnh.
- Loại trách nhiệm cụ thể và mức xử lý chưa có căn cứ trực tiếp trong các nguồn hiện tại.

## Nên làm gì tiếp theo

- Dừng ngay việc tạo hoặc sử dụng nội dung giả mạo.
- Không chuyển tiền trước khi xác minh yêu cầu qua kênh liên lạc chính thức.
- Lưu giữ bản ghi cuộc gọi, số điện thoại, tin nhắn và chứng từ giao dịch.
- Nếu đã phát sinh giao dịch, thông báo ngay cho doanh nghiệp và ngân hàng.
- Nếu có dấu hiệu chiếm đoạt, thiệt hại hoặc hành vi tiếp diễn, liên hệ cơ quan có thẩm quyền.

## Căn cứ pháp lý

- Luật An ninh mạng 2025, Điều 7, Khoản 2, Điểm g.
```

Citation `Điều 7, Khoản 2, Điểm g` xuất hiện đúng hai lần: một lần ở heading
phân tích và một lần trong danh mục căn cứ.

## Test output

```text
45 focused tests: OK
95 production tests: OK
11 evaluation tests: OK
```

Test bao phủ:

- MATCH.
- PARTIAL_MATCH.
- Không có source.
- Không có chế tài.
- Snapshot giả giọng.
- Kết luận mâu thuẫn.
- Visible safe fallback.
- Lọc comment/placeholder khỏi `missing_facts`.
- Citation budget theo Applicability + CE score.
- Recovery từ dẫn chiếu luật.
- Chuẩn hóa hậu tố văn bản sửa đổi trong dòng citation.

## Benchmark before/after

| Metric | Before | After | Kết quả |
|---|---:|---:|---|
| Citation Accuracy | 23.56% | 30.00% | +6.44 điểm % |
| Recall@5 | 38.33% | 38.33% | giữ nguyên |
| Recall@10 | 41.67% | 41.67% | giữ nguyên |
| Domain Recall | 57.83% | 57.83% | giữ nguyên |
| Wrong Domain Rate | 19.56% | 19.56% | giữ nguyên |
| Recursive Noise | 10.00% | 10.00% | giữ nguyên |
| Hallucinated Citation | 0 | 0 | giữ nguyên |
| LLM calls/case | 2 | 2 | giữ nguyên |
| Total latency | 97.98 s | 107.52 s | +9.54 s |

Artifacts:

- Before:
  `evaluation/results/benchmark_30_legal_reasoning_preamble_only_20260727`
- After:
  `evaluation/results/benchmark_30_final_output_ce_priority_20260727`

Chênh lệch latency end-to-end đến từ biến động hai LLM call hiện hữu:

- Applicability: 49.08 s → 52.62 s.
- Phần còn lại sau retrieval/applicability, chủ yếu draft LLM:
  45.21 s → 51.09 s.
- Retrieval: 3.69 s → 3.82 s.

Renderer deterministic chạy trung bình khoảng `0.21 ms/output` trong
microbenchmark 10.000 lần, nên không phải nguồn của mức tăng 9.54 giây.

## Contradiction và fallback

- Số output có cả kết luận dương tính và phủ định toàn bộ hành vi: `0/30`.
- Số output sai/mất một trong sáu heading: `0/30`.
- Số output quay lại heading template cũ: `0/30`.
- Hallucinated Citation: `0`.
- Acceptance demo đã ép đi qua nhánh raw/repair bị validator loại và vẫn xuất
  đúng template mới.

## Không thay đổi kiến trúc/call/API

- Không thêm agent hoặc model.
- Không thêm LLM call.
- Không thay public API hoặc chatbot output transport.
- Không thay benchmark/evaluation metrics.
- Không thay retrieval, recursive retrieval hoặc Applicability.
- Không thay citation validator.

## Rollback

Rollback chỉ cần bỏ `answer_assessment` khỏi generation/workflow state, bỏ lời
gọi `build_answer_assessment` ở hai generation node và trả
`render_grounded_answer` về legacy branch. Không có index migration, API
migration hoặc thay đổi dữ liệu cần hoàn tác.
