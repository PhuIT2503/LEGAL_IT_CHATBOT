# Long-scenario fact preservation and model-dependent final context

Ngày audit/implementation: 2026-07-27  
Regression: AI giả giọng giám đốc, yêu cầu chuyển 300 triệu; đã chuyển
300 triệu; phong tỏa 180 triệu; 120 triệu chuyển tiếp; A khai chỉ thử nghiệm
và chưa sử dụng tiền.

## Kết luận

Task đã hoàn tất trong đúng phạm vi:

- Không sửa renderer, output template, citation placement hoặc safe fallback.
- Không sửa benchmark, evaluation metrics, public API, response schema,
  Recursive Retrieval hoặc số LLM call.
- Production vẫn dùng 2 LLM call: `legal_applicability` và `draft`/Generation.
- 13 stated facts bắt buộc được giữ đầy đủ.
- Không còn hỏi lại trạng thái chuyển tiền, số tiền hoặc giai đoạn
  chuẩn bị/thực hiện trong long scenario.
- Ý định chiếm đoạt và tổn thất cuối cùng chỉ là inference/unknown, không phải
  stated fact.
- Hai model nhận cùng canonical Generation payload:
  `298cfe0571a2500a6860e1db772d6bb0aa3c6bb3f75a08a12b42d06ea4935758`.
- Benchmark 30 case giữ Citation Accuracy 30%, Recall@10 41,67%, Wrong Domain
  19,56% và Hallucinated Citation 0.

## 1. Root cause cụ thể

### 1.1 Không có canonical fact state

Pipeline trước thay đổi chỉ truyền raw query. Applicability và Generation tự
đọc lại một đoạn tình huống dài, không có cấu trúc chung để phân biệt:

- dữ kiện người dùng đã nêu;
- suy luận pháp lý;
- yếu tố chưa biết.

Hậu quả là số tiền, trạng thái giao dịch, lời khai và giai đoạn thực hiện không
được bảo toàn như invariant giữa các stage.

### 1.2 Missing facts được tạo bằng lexical trigger

`_scenario_missing_facts` cũ kiểm tra chuỗi như `chuyển tiền`, `thanh toán`,
`chiếm đoạt`, rồi thêm cố định:

- tiền/tài sản đã chuyển hay chưa;
- mục đích chiếm đoạt;
- thiệt hại thực tế;
- chuẩn bị hay đã thực hiện.

Hàm không đối chiếu với dữ kiện đã nêu. Ngoài contradiction, kết quả còn phụ
thuộc wording: `yêu cầu chuyển tiền` kích hoạt, còn `yêu cầu chuyển
300.000.000 đồng` không kích hoạt dù cùng nghĩa. Đây là nguyên nhân trực tiếp
của output lịch sử hỏi lại facts.

### 1.3 Behavior taxonomy bỏ sót alias “giả giọng”

Trước sửa, taxonomy chỉ có `giả mạo giọng nói`/`giọng nói giả`, không có
`giả giọng`. Behavior Card của exact regression query vì thế rỗng:

```json
{"actions":[],"objects":[],"purposes":[],"conditions":[]}
```

Luật An ninh mạng 2025 Điều 7 Khoản 2 Điểm g có mặt trong Retrieval, nhưng
Applicability của cả hai model bị validator hạ `LOW/INVALID`; candidate chỉ
sống nhờ `seed protection` dưới dạng `WEAK_KEEP`.

### 1.4 Applicability chỉ dựa vào output model và similarity

Không có deterministic required-element coverage. Vì vậy một nguồn có các từ
chung như công nghệ thông tin/giả mạo có thể được model chấm HIGH dù nguồn còn
yêu cầu giao kết hợp đồng, hợp đồng theo mẫu hoặc điều kiện giao dịch chung.

### 1.5 Model được dùng ở đâu

Model được chọn trong app thay cùng một `LLMClient` cho cả hai call:

1. Applicability: model đánh giá candidate.
2. Generation: model viết draft từ final context.

Retrieval, Domain Selection, Behavior extraction và CE là deterministic với
cùng index/config. Do đó final context từng phụ thuộc model ngay từ
Applicability, trước Generation.

## 2. Stage đầu tiên hai model khác nhau

### Replay ngay trước thay đổi

Qwen và GPT có cùng:

- normalized query;
- selected domains;
- Retrieval candidate;
- CE score;
- final article sau seed protection.

Khác biệt đầu tiên xuất hiện trong raw Applicability:

- `scope`;
- `explanation`;
- `missing_conditions`.

Cả hai sau validation vẫn thành `LOW/INVALID + WEAK_KEEP`, nên immediate replay
cho cùng final context. Snapshot lịch sử Qwen dùng Nghị định 15 Điều 33 Khoản
12 còn GPT rỗng không tái hiện trên checkout ngay trước sửa; nguyên nhân kiến
trúc vẫn được xác nhận: Applicability là model-dependent upstream stage và có
quyền quyết định final context.

Sau sửa, raw prose Applicability vẫn có thể khác, nhưng Generation payload chỉ
giữ stable projection: coordinates, decision, level, behavior matches và
deterministic element coverage. Hai full pipeline có cùng payload hash.

Artifacts:

- [Before Qwen](../evaluation/results/long_scenario_before_qwen_20260727.json)
- [Before GPT-4o-mini](../evaluation/results/long_scenario_before_gpt_20260727.json)
- [After Qwen](../evaluation/results/long_scenario_after_qwen_20260727.json)
- [After GPT-4o-mini](../evaluation/results/long_scenario_after_gpt_20260727.json)

## 3. Before/after fact state

### Before

Không có structured fact state. Answer assessment chỉ giữ ba fact tổng quát
khi source Điều 7(g) sống sót:

- dùng AI;
- giả mạo giọng nói;
- mạo danh người có thẩm quyền để yêu cầu chuyển tiền.

Các amount, trạng thái chuyển, phong tỏa, chuyển tiếp và lời khai không phải
invariant của pipeline.

### After — stated facts

```json
{
  "used_ai": true,
  "voice_impersonation": true,
  "impersonated_person_role": "giám đốc",
  "called_accountant": true,
  "requested_transfer": true,
  "requested_amount_vnd": 300000000,
  "transfer_executed": true,
  "transferred_amount_vnd": 300000000,
  "frozen_amount_vnd": 180000000,
  "onward_transferred_amount_vnd": 120000000,
  "act_was_executed": true,
  "actor_claimed_experiment": true,
  "actor_claimed_money_not_used": true
}
```

### After — supported inferences

```json
{
  "possible_appropriation_intent": true,
  "possible_financial_loss": true,
  "impersonation_used_to_induce_transfer": true
}
```

### After — unknown legal elements

- `final_unrecoverable_loss`;
- `full_appropriation_intent_evidence`;
- `liability_type`;
- `complete_sanction_basis`.

Không có `appropriation_intent` hoặc `final_unrecoverable_loss` trong
`stated_facts`.

## 4. Before/after missing facts

### Before

Static path có thể sinh ba contradiction bị cấm:

- “Tiền hoặc tài sản đã được chuyển hay chưa.”
- “Hành vi mới ở giai đoạn chuẩn bị hay đã được thực hiện.”
- “Số tiền chuyển là bao nhiêu.”

### After

`missing_fact_keys` chỉ còn:

```json
[
  "final_unrecoverable_loss",
  "full_appropriation_intent_evidence",
  "liability_type",
  "complete_sanction_basis"
]
```

Semantic alias guard map các cách diễn đạt về trạng thái chuyển, số tiền,
chuẩn bị/thực hiện, AI và giả giọng về stated facts. Cả full-pipeline output
và frozen-context output của hai model đều không chứa ba contradiction.

Guard được kích hoạt cho scenario có bằng chứng giao dịch đã thực hiện, vì vậy
snapshot UX của query ngắn/hypothetical không bị viết lại.

## 5. Nghị định 15 Điều 33 Khoản 12

Immediate replay hiện tại không retrieve candidate này, nên không inject nó để
làm đẹp kết quả. Guard được kiểm tra độc lập bằng chính đoạn corpus và một
Applicability response cố tình chấm `HIGH`.

| Element rút từ provision text | Scenario |
|---|---|
| software/IT application | matched bởi AI/technology |
| falsified information/image | partial factual overlap |
| contract conclusion | missing |
| standard-form contract/general terms | missing |

Kết quả:

- Before logic: LLM `HIGH` có thể trở thành `KEEP`.
- After: `element_applicability=NOT_APPLICABLE`, decision `REMOVE`.

Rule không hard-code document/article ID. Nó chỉ hard-remove khi retrieved text
chứa đầy đủ contract cluster nhưng scenario không có contract fact.

Test: `test_d_contract_provision_is_not_applicable_without_contract_fact`.

## 6. Trace Luật An ninh mạng 2025 Điều 7 Khoản 2 Điểm g

### Before

- Cybersecurity domain: opened.
- Hybrid rank: 2; RRF: 0,5.
- CE raw score: 0,0199707.
- Behavior score: 0 do Behavior Card rỗng.
- Applicability: `LOW/INVALID`, `WEAK_KEEP` nhờ seed.
- Final context: có.

### After

Hai model có cùng trace:

| Stage | Kết quả |
|---|---:|
| Cybersecurity domain | opened |
| Dense rank | 1 |
| BM25 rank trong top 20 | không có |
| Hybrid rank / RRF | 2 / 0,5 |
| CE raw score / raw CE rank | 0,1894318 / 9 |
| Post behavior rank | 3 |
| Behavior score / gate | 0,87 / PASS |
| Applicability | HIGH + KEEP |
| Element coverage | MATCH |
| Final context | có |

CE model/scoring code không bị sửa. CE score thay đổi vì behavior alias làm
query enrichment hiện hữu nhận đúng behavior card.

## 7. Frozen-context evaluation

Artifact:
[long_scenario_frozen_context_20260727.json](../evaluation/results/long_scenario_frozen_context_20260727.json)

Hai model nhận cùng hash:
`298cfe0571a2500a6860e1db772d6bb0aa3c6bb3f75a08a12b42d06ea4935758`.

| Tiêu chí | Qwen2.5 7B | GPT-4o-mini |
|---|---:|---:|
| Latency Generation | 54,25 s | 9,94 s |
| Prompt + completion tokens | 4.389 | 3.988 |
| Giữ 300/180/120 triệu | 3/3 | 3/3 |
| Missing-fact contradiction | 0 | 0 |
| Required headings | 5/5 | 5/5 |
| Có valid source marker | có | có |
| Raw draft qua strict validation | không | không |

Qwen raw draft còn hallucinate Bộ luật Hình sự/Điều 142 và mức phạt không có
trong source. GPT không tạo căn cứ mới nhưng dùng mức chắc chắn quá cao khi
Retrieval incomplete và phần chế tài thiếu citation. Kết luận này chỉ dựa trên
frozen-context test, không gộp lỗi Retrieval vào năng lực Generation.

Production validator chặn cả hai raw draft và dùng grounded fallback cũ.

## 8. Full-pipeline output của hai model

| Tiêu chí | Qwen2.5 7B | GPT-4o-mini |
|---|---:|---:|
| Final context | Điều 7(2)(g) | Điều 7(2)(g) |
| Generation payload hash | `298cfe…5758` | `298cfe…5758` |
| Total latency | 82,70 s | 46,76 s |
| Total tokens | 6.259 | 5.946 |
| LLM calls | 2 | 2 |
| Final answer | grounded fallback | grounded fallback |
| Ba amount facts | đầy đủ | đầy đủ |
| Answered facts trong missing | 0 | 0 |

Final answer sau validator của hai model giống nhau và cite đúng Luật An ninh
mạng 2025 Điều 7 Khoản 2 Điểm g. Không có source về loại trách nhiệm/chế tài,
nên output không tự suy ra BLHS hay mức xử lý.

## 9. Regression tests

```text
Focused output tests: 45/45 pass
Production tests: 101/101 pass, 2 subtests pass
  - gồm 95 test cũ + 6 test long-scenario mới
Evaluation tests: 11/11 pass
Template compliance: 30/30
Legacy template regression: 0/30
Conclusion contradiction: 0/30
Hallucinated Citation: 0
```

Sáu test mới bao phủ:

- stated fact preservation;
- không đưa answered fact vào missing;
- inference separation;
- contract provision NOT_APPLICABLE;
- same frozen payload hash;
- behavior alias `giả giọng`.

## 10. Benchmark trước/sau

Benchmark file/hash và runner không đổi:
`126438a83f9902d950b140f74fd1f527b899ec9dc160ce1163f47c0c5d3e114c`.

Baseline:
[benchmark_30_final_output_ce_priority_20260727](../evaluation/results/benchmark_30_final_output_ce_priority_20260727/benchmark_report.md)

After:
[benchmark_30_long_fact_preservation_20260727](../evaluation/results/benchmark_30_long_fact_preservation_20260727/benchmark_report.md)

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Citation Accuracy | 30,00% | 30,00% | 0 |
| Recall@5 | 38,33% | 38,33% | 0 |
| Recall@10 | 41,67% | 41,67% | 0 |
| Domain Recall | 57,83% | 57,83% | 0 |
| Wrong Domain Rate | 19,56% | 19,56% | 0 |
| Behavior Recall | 30,85% | 31,68% | +0,83 điểm % |
| Recursive Noise | 10,00% | 10,00% | 0 |
| Hallucinated Citation | 0 | 0 | 0 |
| LLM calls/case | 2 | 2 | 0 |
| Total latency | 107,52 s | 109,13 s | +1,60 s |
| Retrieval latency | 3,82 s | 4,02 s | +0,21 s |
| Generation grounding errors | 30 | 27 | -3 |
| Applicability Accuracy | 53,85% | 50,63% | -3,21 điểm % |

Guardrails Citation/Recall/Domain/Hallucination đều đạt. Applicability Accuracy
giảm dù Citation không đổi; đây là risk thật, nên thay đổi không được quảng bá
như một cải tiến Applicability tổng quát. Mục tiêu của nó là ngăn một
contract-specific provision trở thành principal source cho long scenario
không có contract fact.

## 11. Files sửa trong task

Production/internal pipeline:

- `src/agents/common/legal_scenario_facts.py`
- `src/agents/common/legal_element_coverage.py`
- `src/agents/common/legal_applicability.py`
- `src/agents/common/legal_relevance_filter.py`
- `src/retrieval/legal_behaviors.py`
- `src/agents/agent_retrieval/node_hybrid_search.py`
- `src/agents/agent_retrieval/state.py`
- `src/agents/agent_generation/answer_assessment.py`
- `src/agents/agent_generation/prompts.py`
- `src/agents/agent_generation/node_generate_draft.py`
- `src/agents/agent_generation/node_generate_final.py`
- `src/agents/agent_generation/state.py`
- `src/workflow/state.py`

Diagnostic/tests/docs:

- `evaluation/trace_long_scenario.py`
- `evaluation/frozen_context_long_scenario.py`
- `tests/test_long_scenario_fact_preservation.py`
- `docs/long_scenario_fact_preservation_2026-07-27.md`

Không sửa trong task:

- `src/agents/common/grounded_validation.py` renderer/fallback;
- `src/workflow/pipeline.py` public result/render path;
- benchmark runner/ground truth/metrics;
- Cross Encoder implementation;
- Recursive Retrieval;
- public API và output transport.

## 12. Xác nhận output redesign cũ

Output redesign cũ được giữ nguyên:

- cùng 6 heading;
- cùng renderer;
- cùng citation placement;
- cùng safe fallback;
- template compliance 30/30;
- conclusion contradiction 0/30;
- Hallucinated Citation 0;
- public API và chatbot output schema không đổi.

## Risk và rollback

### Risks

- Extractor deterministic chỉ bảo toàn taxonomy/pattern đã định nghĩa; scenario
  khác cách diễn đạt cần thêm alias có test.
- Required-element coverage hiện chủ ý hẹp. Mở rộng rule mà không benchmark có
  thể làm giảm Recall/Citation.
- Fact block làm prompt dài hơn; benchmark đo +1,60 giây/case, nằm trong biến
  động hai LLM call nhưng vẫn phải được coi là cost.
- Applicability Accuracy giảm 3,21 điểm %, dù các guardrail bắt buộc giữ nguyên.

### Rollback

Rollback độc lập, không cần migration:

1. Bỏ `scenario_fact_state` khỏi retrieval/generation/workflow state.
2. Bỏ Fact Preservation block và canonical payload khỏi generation nodes.
3. Bỏ deterministic element override khỏi Applicability.
4. Bỏ alias `giả giọng` nếu cần cô lập thay đổi Behavior.

Index, Qdrant data, benchmark, public API, renderer và chatbot output không cần
rollback.
