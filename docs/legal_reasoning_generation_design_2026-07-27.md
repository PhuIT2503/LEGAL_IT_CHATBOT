# Legal Reasoning Layer & User-oriented Answer Generation

Ngày thiết kế: 2026-07-27

## 1. Phạm vi và guardrail

Thiết kế chỉ thay đổi lớp Generation và grounding contract nội bộ. Pipeline vẫn
giữ nguyên:

`Question → Retrieval → Recursive Retrieval → Applicability → Generation`

Không thay đổi Domain Selection, Behavior Extraction, Hybrid Retrieval,
Cross Encoder, Recursive Retrieval, Behavior Gate, Applicability, benchmark
runner, public API hoặc kiểu dữ liệu trả về chatbot. Không thêm graph node, model
hoặc LLM call. Generation vẫn chỉ được dùng các `SOURCE_ID` do grounding layer
cấp; citation hiển thị vẫn do hệ thống dựng lại từ nguồn retrieval.

## 2. Audit hiện trạng

### 2.1 Luồng Generation thực tế

Cả `generate_draft_node` và `generate_final_node` đang thực hiện cùng chuỗi:

1. Gọi `prepare_generation_context`.
2. Nhận context đã qua Legal Relevance, Candidate Budget và Applicability.
3. Chuyển context thành các `SOURCE_ID`.
4. Dựng prompt bằng `build_answer_prompt`.
5. Gọi LLM đúng một lần.
6. Ở cuối workflow, kiểm tra grounding, repair nếu được cấu hình, salvage hoặc
   fallback, sau đó render citation.

Điểm chèn phù hợp duy nhất là giữa bước 2 và bước 3/4. Tại đây đã có đủ:

- câu hỏi gốc;
- Behavior Card;
- các decision Applicability có `level`, `decision`, `behavior_matches` và
  `missing_conditions`;
- final context records và trạng thái completeness.

Không cần gọi lại retrieval hoặc Applicability.

### 2.2 Nguyên nhân câu trả lời khó dùng

Prompt hiện tại bắt model tổ chức câu trả lời chủ yếu theo từng SOURCE và lặp
sáu trường cho mỗi block: nội dung điều luật, hành vi phù hợp, phân tích, điều
kiện còn thiếu, căn cứ và đánh giá. Strict validator cũng bắt buộc cấu trúc này.
Hệ quả là:

- kết luận dành cho người dùng xuất hiện muộn;
- một tình tiết bị lặp qua tóm tắt, vấn đề pháp lý, từng SOURCE và synthesis;
- cùng một điều kiện thiếu có thể xuất hiện ở nhiều block;
- model phải tự suy ra mức kết luận từ context thô dù Applicability đã có tín
  hiệu cấu trúc tương ứng;
- khi strict validation thất bại, fallback tiếp tục dùng cùng cấu trúc dài.

Baseline gần nhất
`evaluation/results/benchmark_30_applicability_recovery_20260726` cho thấy:

| Chỉ số | Baseline |
|---|---:|
| Citation Accuracy | 21.67% |
| Hallucinated Citation | 0 |
| Recall@10 | 41.67% |
| Expected provisions trong final context | 12/44 |
| Expected provisions được cite đúng | 9/44 |
| Average total tokens | 6,585.5 |
| Average completion tokens | 1,711.0 |
| Average raw generation length | 2,833 ký tự |
| Average rendered answer length | 4,581 ký tự |

Retrieval là trần recall, nhưng trong phạm vi phase này có thể cải thiện cách
trình bày mà không làm thay đổi candidate pool.

### 2.3 Grounding và API phải giữ nguyên

`ChatbotWorkflow.run` chỉ công khai các field hiện hữu. Decision trace hiện là
state nội bộ và không được trả về public response. `render_grounded_answer`
đang là nguồn duy nhất dựng tên văn bản/Điều/Khoản/Điểm, vì vậy lớp mới không
được tự sinh citation hoặc tọa độ pháp lý.

## 3. Proposed Architecture

### 3.1 Reasoning Summary xác định, không dùng LLM

Thêm helper thuần xác định trong Generation:

```text
build_reasoning_summary(
  query,
  behavior_profile,
  applicability_decisions,
  final_context_records,
  retrieval_is_complete
) -> ReasoningSummary
```

Schema nội bộ:

```json
{
  "question_type": "legal_assessment",
  "preliminary_assessment": "LIKELY_IN_SCOPE|NO_SUPPORTED_MATCH|NEEDS_MORE_FACTS",
  "matched_elements": ["..."],
  "missing_elements": ["..."],
  "confidence": "HIGH|MEDIUM|LOW",
  "answer_strategy": {
    "code": "DIRECT_SCOPE_MATCH|NO_SUPPORTED_MATCH|INSUFFICIENT_FACTS",
    "opening": "Có dấu hiệu thuộc phạm vi điều chỉnh.|Chưa thấy dấu hiệu phù hợp.|Chưa đủ dữ kiện để xác định."
  }
}
```

Nguồn dữ liệu và quy tắc:

- Chỉ đọc decision có `decision_stage=applicability`.
- `matched_elements` chỉ lấy behavior key thuộc chính Behavior Card và được
  Applicability chấm `MATCH` hoặc `PARTIAL_MATCH` trên candidate được giữ.
- Behavior key được hiển thị bằng `description` đã tồn tại trong taxonomy;
  không trích keyword mới từ câu hỏi và không dùng văn bản luật để tạo fact.
- `missing_elements` chỉ lấy `missing_conditions` của candidate được giữ, loại
  giá trị biểu thị “không còn điều kiện thiếu”, rồi deduplicate.
- Có ít nhất một `KEEP` hoặc `WEAK_KEEP`: mở đầu
  “Có dấu hiệu thuộc phạm vi điều chỉnh”.
- Có decision Applicability nhưng tất cả bị `REMOVE`: mở đầu
  “Chưa thấy dấu hiệu phù hợp”.
- Không có decision hợp lệ hoặc không đủ dữ liệu để adjudicate: mở đầu
  “Chưa đủ dữ kiện để xác định”.
- `HIGH` chỉ khi có `KEEP`, retrieval complete và không còn điều kiện thiếu;
  còn candidate được giữ nhưng có gap/điều kiện thiếu là `MEDIUM`; không có
  candidate được giữ là `LOW`.

Summary không chứa tên văn bản, số Điều/Khoản/Điểm hoặc citation.

### 3.2 Pipeline mới trong cùng Generation node

```text
prepare_generation_context
        │
        ├── context + applicability decision trace (không đổi)
        │
        ▼
build_reasoning_summary (deterministic, 0 LLM call)
        │
        ├── internal structured log
        └── user-oriented preamble cho renderer
        ▼
build_grounded_sources → build_answer_prompt không đổi → existing LLM call
        ▼
existing validation / repair / salvage / fallback
        ▼
render_grounded_answer + deterministic preamble
```

Summary được đặt trong graph state để workflow cuối có thể render preamble,
nhưng không được thêm vào dict public do `ChatbotWorkflow.run` trả về.

### 3.3 Prompt isolation

Oracle triển khai ban đầu đã thử truyền một block `REASONING SUMMARY —
INTERNAL` vào prompt và yêu cầu model gom SOURCE theo vấn đề. Full benchmark
cho kết quả không đạt guardrail:

- Citation Accuracy: `21.67% → 18.33%`;
- average total tokens: `6,585.5 → 6,930.0`;
- average rendered length: `4,581 → 5,062` ký tự;
- Generation Grounding Error: `29 → 30`;
- Hallucinated Citation vẫn bằng `0` nhờ validator.

Vì vậy phần prompt này đã được rollback. Final architecture dựng summary trước
LLM call để instrument, nhưng cô lập nó khỏi prompt. Summary chỉ điều khiển
preamble xác định sau grounding validation. Template Generation và grounding
validator giữ nguyên, gồm các kiểm tra SOURCE_ID, citation cùng đoạn, tọa độ
pháp lý, số tiền chế tài và hậu quả pháp lý không có trong nguồn. Node sử dụng
`retrieval_is_complete` sau Applicability thay vì state cũ trước Applicability,
để completeness instruction phản ánh đúng retrieval gap đã có.

### 3.4 User-oriented preamble

Preamble được renderer dựng xác định, không giao cho model tự viết:

```markdown
**Đánh giá sơ bộ:** Có dấu hiệu thuộc phạm vi điều chỉnh.

## Dấu hiệu đã xác định
- [description của behavior đã được Applicability xác nhận]

## Cần thêm thông tin
- [missing_conditions nguyên nghĩa từ Applicability]
```

Mục “Cần thêm thông tin” chỉ xuất hiện khi Applicability thực sự trả điều kiện
thiếu. Nếu không có behavior match được xác nhận, mục “Dấu hiệu đã xác định”
nêu bảo thủ rằng chưa có dấu hiệu cấu trúc nào được xác nhận; nó không tạo fact
mới.

Preamble không có citation và không đưa ra hậu quả pháp lý. Phần phân tích pháp
lý bên dưới vẫn phải cite như hiện tại.

### 3.5 Instrumentation

Mỗi Generation invocation ghi một structured log, internal only:

- `reasoning_summary`
- `matched_elements`
- `missing_elements`
- `preliminary_assessment`
- `answer_strategy`

Log không được đưa vào chatbot output hoặc public response. Không thay benchmark
runner.

## 4. Expected Impact

Đây là phase Generation-only nên oracle cho các chỉ số retrieval là bất biến:

| Chỉ số | Kỳ vọng |
|---|---|
| Recall@10 | 41.67%, không đổi |
| Retrieval candidate/final context | Không đổi |
| Citation Accuracy | ≥ 21.67% |
| Hallucinated Citation | 0 |
| Wrong Domain | Không đổi |
| LLM call count | Không tăng |
| Applicability latency | Không đổi |

Expected UX/efficiency:

- 100% câu legal có opening strategy xác định từ Applicability state;
- các dấu hiệu hiển thị chỉ thuộc Behavior Card đã được xác nhận;
- các dữ kiện cần bổ sung chỉ đến từ Applicability;
- không kỳ vọng giảm completion token bằng prompt vì oracle đã bác bỏ hướng đó;
  preamble làm rendered answer dài thêm một lượng nhỏ, có kiểm soát.

Tổng latency có thể giảm nhẹ nếu output ngắn hơn, nhưng không kỳ vọng cải thiện
phần Retrieval hoặc Applicability. Helper xác định chỉ có độ phức tạp tuyến tính
theo số decision, chi phí dưới mức đo đáng kể so với LLM.

## 5. Risk Analysis

### 5.1 Applicability sai kéo opening sai

Opening là “đánh giá sơ bộ”, không phải kết luận vi phạm. `WEAK_KEEP` chỉ dẫn
đến câu “thuộc phạm vi điều chỉnh”, không tự nâng thành “vi phạm”. Rollback:
tắt truyền/render Reasoning Summary, trở lại prompt cũ.

### 5.2 Behavior description bị hiểu như kết luận pháp lý

Preamble chỉ mô tả dấu hiệu tình huống đã được structured matching xác nhận,
không nêu luật hoặc hậu quả. Phần pháp lý vẫn bắt buộc citation.

### 5.3 Missing conditions lặp hoặc mang câu phủ định

Normalize và deduplicate; loại các câu bắt đầu bằng “không còn điều kiện
thiếu”. Không tự rút gọn bằng LLM để tránh đổi nghĩa.

### 5.4 Prompt dài hơn

Rủi ro này đã xảy ra trong oracle và được loại bỏ bằng cách không truyền
Reasoning Summary vào prompt.

### 5.5 Validator/fallback không đồng nhất

Preamble được render sau validation nên áp dụng cho cả model answer, salvage và
fallback. Strict grounding contract không cần nới lỏng và không phụ thuộc model
có tuân thủ format preamble hay không.

## 6. Implementation Plan

### Phase GR-1 — Internal Reasoning Summary

- Thêm helper và unit test cho ba trạng thái opening.
- Tích hợp vào draft/final node ngay sau Applicability.
- Thêm field state nội bộ và structured logs.
- Không thay prompt/output ở bước này.

Benchmark độc lập: unit tests; xác nhận context, decision trace, call count và
public result keys không đổi.

### Phase GR-2 — User-oriented preamble

- Renderer thêm preamble xác định.
- Giữ Generation prompt nguyên trạng; không truyền summary vào model.
- Giữ toàn bộ grounding checks và citation rendering hiện hữu.

Benchmark độc lập: grounding regression tests, prompt tests, test fallback và
public API.

### Phase GR-3 — Full benchmark and decision gate

- Chạy nguyên benchmark 30 case, không sửa benchmark runner.
- So sánh Citation Accuracy, Hallucinated Citation, Recall@10, final context,
  token usage, average raw/rendered output length và latency.
- Chỉ giữ thay đổi nếu Citation Accuracy không giảm, Hallucinated Citation vẫn
  bằng 0, Recall@10/final context không đổi và LLM call count không tăng.

## 7. Rollback plan

Các thay đổi được tách theo lớp:

1. Phần truyền `reasoning_summary` vào prompt đã được rollback sau benchmark
   không đạt guardrail.
2. Bỏ tham số summary khỏi renderer để rollback preamble.
3. Bỏ helper/state field để rollback toàn bộ phase.

Không cần migration dữ liệu, rebuild index, thay benchmark hoặc đổi API. Nếu
full benchmark vi phạm bất kỳ guardrail citation/retrieval nào, rollback toàn
bộ GR-2 trước; nếu vẫn vi phạm, rollback GR-1.

## 8. Kết quả triển khai và benchmark gate

Full benchmark final:
`evaluation/results/benchmark_30_legal_reasoning_preamble_only_20260727`.

| Guardrail | Baseline | Final | Kết quả |
|---|---:|---:|---|
| Citation Accuracy | 21.67% | 23.56% | Pass |
| Recall@10 | 41.67% | 41.67% | Pass |
| Wrong Domain Rate | 19.56% | 19.56% | Pass |
| Hallucinated Citation | 0 | 0 | Pass |
| Average LLM calls | 2.0 | 2.0 | Pass |

Final variant giữ cả hai canary citation 100% (`ai_copyright_easy_001` và
`network_security_easy_001`) mà prompt experiment đã làm giảm lần lượt xuống
66.67% và 50%. Vì Applicability và Generation dùng local LLM, context count,
token và latency có run-to-run variance; các gain ngoài guardrail không được
quy trực tiếp cho helper xác định.
