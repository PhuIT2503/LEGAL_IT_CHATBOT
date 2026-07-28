# Legal Reasoning Layer — Implementation & Benchmark Report

Ngày: 2026-07-27

## Verdict

Final architecture đạt toàn bộ guardrail:

- Citation Accuracy không giảm: `21.67% → 23.56%`;
- Recall@10 giữ nguyên `41.67%`;
- Wrong Domain Rate giữ nguyên `19.56%`;
- Hallucinated Citation giữ nguyên `0`;
- số LLM call trung bình giữ nguyên `2.0`;
- public API, Retrieval, Recursive Retrieval, Applicability, benchmark runner
  và grounding validator không đổi.

Một thiết kế trung gian truyền Reasoning Summary vào prompt đã bị benchmark bác
bỏ và rollback. Final variant chỉ giữ deterministic summary, internal logging
và deterministic user-facing preamble sau grounding validation.

## Runs

| Run | Artifact |
|---|---|
| Baseline | `evaluation/results/benchmark_30_applicability_recovery_20260726/` |
| Failed prompt experiment | `evaluation/results/benchmark_30_legal_reasoning_20260727/` |
| Final preamble-only | `evaluation/results/benchmark_30_legal_reasoning_preamble_only_20260727/` |

Cả ba dùng benchmark SHA-256
`126438a83f9902d950b140f74fd1f527b899ec9dc160ce1163f47c0c5d3e114c`,
30 case, mode `critic`, `top_k=10`, Qwen2.5:7b và
`GROUNDING_REPAIR_ATTEMPTS=0`.

## Architecture implemented

Reasoning Summary được dựng sau `prepare_generation_context()` và trước call
Generation hiện hữu. Helper chỉ đọc:

- query gốc;
- Behavior Card;
- decision trace của Applicability;
- final context records;
- retrieval completeness.

Không có LLM call mới. Summary được lưu trong graph state nhưng không được thêm
vào dict public do `ChatbotWorkflow.run()` trả về.

Renderer dựng:

```markdown
**Đánh giá sơ bộ:** [một trong ba opening]

## Dấu hiệu đã xác định
- [taxonomy description đã được Applicability MATCH/PARTIAL_MATCH]

## Cần thêm thông tin
- [missing_conditions từ kept Applicability decision]
```

Preamble không chứa tên luật, Điều/Khoản/Điểm hoặc citation. Phần pháp lý bên
dưới vẫn đi qua validator và citation renderer cũ.

Generation prompt template không đổi trong final variant. Node chỉ dùng giá trị
`retrieval_is_complete` đã được `prepare_generation_context()` cập nhật, thay
cho state completeness trước Applicability, nên instruction “đầy đủ/chưa đầy
đủ” phản ánh đúng retrieval gap hiện hành.

## Benchmark comparison

| Metric | Baseline | Failed prompt | Final |
|---|---:|---:|---:|
| Citation Accuracy | 21.67% | 18.33% | **23.56%** |
| Recall@5 | 38.33% | 38.33% | 38.33% |
| Recall@10 | 41.67% | 41.67% | 41.67% |
| Wrong Domain Rate | 19.56% | 19.56% | 19.56% |
| Recursive Noise | 10.00% | 10.00% | 10.00% |
| Hallucinated Citation | 0 | 0 | 0 |
| Average Retrieval latency | 4.22s | 4.10s | 3.69s |
| Average total latency | 106.42s | 108.07s | 97.98s |
| Average prompt tokens | 4,874.5 | 5,165.8 | 4,825.2 |
| Average completion tokens | 1,711.0 | 1,764.3 | 1,609.1 |
| Average total tokens | 6,585.5 | 6,930.0 | 6,434.4 |
| Average LLM calls | 2.0 | 2.0 | 2.0 |
| Average raw generation length | 2,833 | 3,040 | 2,651 |
| Average rendered answer length | 4,581 | 5,062 | 4,736 |

Rendered answer final dài hơn baseline khoảng 155 ký tự do preamble mới. Raw
generation, token và latency giảm trong final run, nhưng không được claim là
causal gain: local Applicability giữ 107 context records thay vì 117 ở baseline,
cho thấy run-to-run variance downstream. Retrieval metrics hoàn toàn bất biến.

Theo phép đối chiếu provision nhất quán giữa hai artifacts, số expected
provision trong final context không đổi; số expected provision được cite đúng
cũng giữ nguyên. Macro Citation Accuracy tăng do precision của tập citation
rendered tốt hơn trong run final.

## Case-level citation evidence

So với baseline:

- `personal_data_medium_001`: `66.67% → 100%`;
- `malware_easy_001`: `50% → 100%`;
- `ai_copyright_medium_001`: `66.67% → 40%`;
- các case còn lại không đổi.

Hai canary baseline 100%:

- `ai_copyright_easy_001`: prompt experiment `66.67%`, final `100%`;
- `network_security_easy_001`: prompt experiment `50%`, final `100%`.

Hai canary xác nhận rollback prompt là cần thiết.

## Failed experiment and rollback

Reasoning Summary trong prompt làm model 7B lệch grounding contract thường xuyên
hơn, tăng output/token và làm Citation Accuracy giảm 3.34 điểm phần trăm.
Thay đổi này bị rollback hoàn toàn; artifact vẫn được giữ để audit.

Rollback không tác động deterministic summary/preamble vì:

- summary không phải SOURCE pháp luật;
- summary không quyết định citation;
- preamble được ghép sau grounding validation;
- preamble không làm validator chấp nhận một draft vốn không hợp lệ.

## Risks

1. Applicability có thể `WEAK_KEEP` candidate nhưng `missing_conditions` diễn
   đạt một điều kiện loại trừ hoặc không khớp. Phase này không được sửa
   Applicability, nên preamble truyền đạt bảo thủ chứ không viết lại.
2. Khi không còn SOURCE, safe-fallback contract tiếp tục trả đúng câu
   `INSUFFICIENT_GROUNDS`; opening nội bộ vẫn được log nhưng không phá exact
   fallback.
3. Model local không deterministic ở Applicability/Generation. Không nên claim
   latency/token gain chỉ từ một run.

## Tests

- Production unit/regression: `85/85` pass.
- Evaluation tests: `11/11` pass.
- Full benchmark: `30/30` complete, `0` runtime errors.

## Rollback

Rollback preamble: bỏ optional `reasoning_summary` khỏi
`render_grounded_answer()` và call site. Rollback toàn phase: bỏ helper
`legal_reasoning.py` cùng hai field state nội bộ. Không cần migration, rebuild
index hoặc sửa API.
