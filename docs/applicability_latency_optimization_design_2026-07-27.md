# Applicability Candidate Budget & Latency Optimization

Ngày audit: 2026-07-27

Baseline được dùng làm chuẩn:

- `evaluation/results/benchmark_30_applicability_recovery_20260726/`
- 30/30 testcase hoàn thành, không runtime error.
- `critic`, `top_k=10`, `qwen2.5:7b`.
- Benchmark SHA-256: `126438a83f9902d950b140f74fd1f527b899ec9dc160ce1163f47c0c5d3e114c`.

## Root Cause

### 1. Funnel và chi phí hiện tại

| Stage | Số lượng / latency |
|---|---:|
| Expected provisions | 44 |
| Expected provisions retrieved top-10 | 16 |
| Expected provisions tới Applicability | 16 |
| Expected provisions ở final context | 12 |
| Expected provisions được cite đúng | 9 |
| Retrieval records | 208 |
| Records sau Recursive | 212 |
| Unique article đưa vào Applicability | 170 |
| Article được Applicability giữ | 86 |
| Final context records | 117 |
| Retrieval + Recursive latency | 4.22 s/query |
| Applicability latency | 52.91 s/query |
| Generation + local post-processing latency | 49.29 s/query |
| Total latency | 106.42 s/query |

`Generation + local post-processing` là phần dư của total sau khi trừ Retrieval và
Applicability. Mỗi testcase có đúng hai LLM call: một Applicability và một
Generation; repair mặc định bằng 0, Critic phía sau no-op sau recursive retrieval.
Vì vậy phần dư 49.29 giây chủ yếu là Generation, nhưng benchmark hiện tại chưa
instrument riêng vài millisecond/second của validator và renderer.

Tỷ trọng latency:

| Stage | Tỷ trọng total |
|---|---:|
| Retrieval, gồm Recursive | 3.97% |
| Applicability | 49.72% |
| Generation + local post-processing | 46.32% |

Applicability là stage lớn nhất và là stage có quan hệ trực tiếp nhất với số
candidate. Pearson correlation giữa số article/call và latency Applicability là
`0.916`; hồi quy trên 30 call cho kết quả:

```text
applicability_latency_ms ≈ 8,497 + 7,837 × article_count
R² = 0.840
```

Đây là ước lượng theo run hiện có, không phải SLA của Ollama.

### 2. Vì sao Applicability phải đánh giá quá nhiều candidate

Applicability Recovery đã sửa đúng lỗi mất seed, nhưng làm cổng Legal Relevance
trở thành cổng gần như không còn khả năng giới hạn seed:

```text
Legal Relevance LOW + is_seed = true
→ seed_preserved = true
→ WEAK_KEEP
→ luôn đi tiếp tới Applicability
```

Kết quả:

- Trước recovery: 114 article được đánh giá, 14 article được giữ.
- Sau recovery: 170 article được đánh giá, 86 article được giữ.
- Candidate tăng `+49.12%`.
- Article được giữ tăng `+514.29%`.
- Context characters trung bình tăng từ 162 lên 1,460 ký tự/query.
- Raw generation output tăng từ 2,148 lên 2,833 ký tự/query.

Regression latency 30.45 giây được phân bổ:

| Thành phần | Before | After | Delta |
|---|---:|---:|---:|
| Retrieval + Recursive | 4.04 s | 4.22 s | +0.18 s |
| Applicability | 36.06 s | 52.91 s | +16.84 s |
| Generation + local | 35.86 s | 49.29 s | +13.42 s |
| Total | 75.97 s | 106.42 s | +30.45 s |

Root cause không phải Recursive. Root cause là seed protection không có candidate
budget phía sau, cộng với chính sách WEAK_KEEP làm context Generation phình lớn.

### 3. Phân bố quyết định Applicability

| Decision | Count | Tỷ lệ | Expected article group | Correctly cited article group |
|---|---:|---:|---:|---:|
| KEEP | 47 | 27.65% | 6 | 6 |
| WEAK_KEEP | 39 | 22.94% | 4 | 4 |
| REMOVE | 84 | 49.41% | 4 | 0 |
| **Total** | **170** | **100%** | **14** | **10** |

98 decision hợp lệ và 72 decision `INVALID`. Trong 39 WEAK_KEEP, 29 decision
`INVALID`; recovery giữ chúng dựa trên provenance seed thay vì tin output bịa hoặc
thiếu field.

84/170 candidate bị REMOVE sau khi đã tiêu tốn token và latency. 83/84 bị REMOVE
vì Applicability đánh giá LOW; một candidate bị loại vì sinh behavior ngoài
Behavior Card.

### 4. Candidate/article có xu hướng ổn định

Phân tích dưới đây chỉ dùng article xuất hiện ít nhất hai lần. Không được biến
thành global blacklist vì Applicability phụ thuộc query.

#### Luôn REMOVE trong benchmark

| Document — Article | REMOVE / occurrences | Expected | Cited |
|---|---:|---:|---:|
| NĐ 52/2013 — Điều 3 | 3/3 | 0 | 0 |
| VBHN 361+362 — Điều 32 | 2/2 | 0 | 0 |
| VBHN 361+362 — Điều 25a | 2/2 | 0 | 0 |
| Luật An ninh mạng 2025 — Điều 14 | 2/2 | 0 | 0 |
| Luật Giao dịch điện tử 2023 — Điều 37 | 2/2 | 0 | 0 |
| NĐ 52/2013 — Điều 76 | 2/2 | 0 | 0 |
| NĐ 52/2013 — Điều 36 | 2/2 | 0 | 0 |
| NĐ 53/2022 — Điều 30 | 2/2 | 0 | 0 |
| NĐ 53/2022 — Điều 3 | 2/2 | 0 | 0 |
| NĐ 53/2022 — Điều 4 | 2/2 | 0 | 0 |

Tổng cộng 21 lần đánh giá lặp có kết quả REMOVE, không thuộc ground truth và
không được cite.

#### Luôn WEAK_KEEP

Không có article lặp từ hai lần trở lên mà mọi lần đều WEAK_KEEP.

WEAK_KEEP là hiện tượng phụ thuộc query/Behavior Card/validator, không phải thuộc
tính ổn định của article. Các article có tỷ lệ WEAK_KEEP cao nhất nhưng không
đạt 100% gồm:

- NĐ 15/2020 — Điều 80: 3 WEAK_KEEP, 1 KEEP.
- NĐ 15/2020 — Điều 81: 2 WEAK_KEEP, 1 KEEP.
- Luật An ninh mạng 2025 — Điều 2: 3 WEAK_KEEP, 1 KEEP, 2 REMOVE.

#### Luôn KEEP

Chỉ có hai article lặp mà mọi lần đều KEEP:

| Document — Article | KEEP / occurrences | Expected | Correctly cited |
|---|---:|---:|---:|
| VBHN 361+362 — Điều 20 | 2/2 | 1 | 1 |
| NĐ 17/2023 — Điều 29 | 2/2 | 0 | 0 |

Luôn KEEP không đồng nghĩa luôn hữu dụng: NĐ 17/2023 Điều 29 không thuộc ground
truth trong hai lần xuất hiện và một lần tạo citation false positive.

### 5. Candidate gần như luôn vô dụng nhưng vẫn vào Applicability

Có.

- 10 article nêu trên tạo 21 lần đánh giá luôn REMOVE, không expected, không cite.
- Counterfactual replay dùng priority theo Legal Relevance score và budget 6 loại
  26/170 candidate.
- Cả 26 candidate bị loại đều không thuộc ground truth và không được cite trong
  run hiện tại.
- Trong 26 candidate này, Applicability đã đánh giá 18 REMOVE, 7 KEEP và
  1 WEAK_KEEP. Tám candidate “được giữ” nhưng vẫn không đóng góp citation, cho
  thấy LLM Applicability không phải là bộ xếp hạng đủ chính xác để tự giải quyết
  candidate overload.

Không dùng article ID làm blacklist. Dùng score/rank theo từng query.

### 6. Article lặp và duplicate

#### Trong cùng một query

- 208 retrieval records chứa 42 record vượt quá số unique article.
- Đây là các Khoản/Điểm khác nhau của cùng Điều, không phải duplicate legal unit.
- Exact legal-unit duplicate: 0.
- Exact chunk duplicate: 0.
- Sau Recursive có 212 records và vẫn chỉ 42 record vượt unique article.
- Bốn recursive article đều khác article seed; recursive overlap với seed: 0.
- Legal Relevance tạo 170 decision trên 170 unique article.
- Applicability tạo 170 decision trên 170 unique article.
- Duplicate decision trong Legal Relevance: 0.
- Duplicate decision trong Applicability: 0.

Article-level collapse/batching đã tồn tại trong `_group_sources`: các Khoản/Điểm
cùng Điều được đưa vào một `<CANDIDATE>` và nhận đúng một decision.

Vì vậy không có duplicate collapse an toàn nào còn lại ở Applicability. Xóa 42
record khác Khoản/Điểm sẽ có nguy cơ mất đúng expected provision và không được
benchmark ủng hộ.

#### Qua nhiều query

Nhiều article xuất hiện lại giữa các testcase, nhưng không thể cache decision
Applicability theo article vì cùng article có thể KEEP/WEAK_KEEP/REMOVE tùy hành
vi và điều kiện trong query. Ví dụ Luật An ninh mạng Điều 7 có 5 KEEP,
2 WEAK_KEEP và 1 REMOVE.

### 7. Early stopping

Không triển khai early stopping bên trong Applicability.

Hiện tại toàn bộ unique article được batch trong đúng một LLM call. Model phải trả
JSON hoàn chỉnh cho mọi ID. Khi nhận được decision đầu tiên thì chi phí call đã
phát sinh; tách thành nhiều batch để dừng sớm có thể tăng số LLM call, trái ràng
buộc và tăng overhead.

Thay thế hợp lý là pre-call adaptive budget: dừng bổ sung candidate khi đã lấy đủ
top-N theo priority. Đây là deterministic pruning trước LLM, không phải early stop
sau decision.

### 8. Candidate budget an toàn theo replay

Cap theo thứ tự article hiện tại không an toàn:

| Raw cap | Candidate còn lại | Expected article bị mất | Correct citation article bị mất |
|---:|---:|---:|---:|
| 4 | 108 | 2 | 2 |
| 5 | 129 | 2 | 2 |
| 6 | 144 | 2 | 2 |
| 7 | 155 | 0 | 0 |

Khi xếp theo raw Legal Relevance cross-encoder score đã có sẵn rồi lấy top-N:

| Priority cap | Candidate còn lại | Expected article bị mất | Correct citation article bị mất | Any cited article bị mất |
|---:|---:|---:|---:|---:|
| 4 | 108 | 3 | 2 | 6 |
| 5 | 129 | 2 | 2 | 3 |
| **6** | **144** | **0** | **0** | **0** |
| 7 | 155 | 0 | 0 | 0 |

Budget 6 + priority theo score là điểm nhỏ nhất mà trace replay chứng minh không
loại article ground-truth đã tới Applicability và không loại bất kỳ article nào
đã được cite.

Đây là bằng chứng counterfactual trên candidate/citation trace, chưa phải bằng
chứng end-to-end về output mới. LLM có thể đổi decision hoặc citation khi prompt
ngắn hơn. Vì vậy thay đổi chỉ được promote nếu full 30-case benchmark sau triển
khai đạt guardrail.

### 9. Validation update sau khi đóng băng thiết kế

Smoke A/B thật trên `ai_copyright_easy_001`, case có correct low-relevance seed:

| Config | Candidate | Citation Accuracy | Applicability latency | Total latency |
|---|---:|---:|---:|---:|
| Baseline authoritative run | 8 | 100% | 66.83 s | 112.96 s |
| Priority cap 6 | 6 | 66.67% | 50.61 s | 152.42 s |
| Priority cap 7 | 7 | 100% | 56.68 s | 105.26 s |

Cap 6 vẫn giữ đúng Điều 20 nhưng làm Qwen đổi một seed nhiễu sang WEAK_KEEP và
cite thêm Điều 26. Cap 7 chỉ prune Điều 19, giữ Citation Accuracy 100% và giảm
Applicability 10.16 giây trên case này. Vì guardrail ưu tiên Citation Accuracy,
default triển khai được nâng từ 6 lên 7 trước full benchmark. Bảng replay ở trên
được giữ nguyên để thể hiện cách ngưỡng ban đầu được chọn và vì sao live
validation phải override counterfactual replay.

### 10. Full 30-case validation và quyết định promotion

Full benchmark Candidate Budget 7:

- `evaluation/results/phase1_candidate_budget7_30_20260727/`
- 30/30 testcase hoàn thành, không runtime error.
- Cùng benchmark SHA-256, model, index, `critic` mode và `top_k=10` như baseline.

| Metric | Baseline | Budget 7 | Delta | Guardrail |
|---|---:|---:|---:|---|
| Citation Accuracy | 21.67% | 18.89% | -2.78 điểm % | **FAIL** |
| Recall@10 | 41.67% | 41.67% | 0 | PASS |
| Wrong Domain Rate | 19.56% | 19.56% | 0 | PASS |
| Hallucinated Citation | 0 | 0 | 0 | PASS |
| Applicability candidate | 170 | 155 | -8.82% | đạt target |
| Applicability latency | 52.91 s | 47.25 s | -10.70% | cải thiện |
| Total latency | 106.42 s | 96.33 s | -9.48% | cải thiện |

Budget tác động 9/30 query và prune 15 candidate. Hai thay đổi Citation Accuracy
khác cũng xuất hiện ở query không bị prune (`ai_copyright_medium_001` giảm
66.67 điểm %, `personal_data_medium_001` tăng 33.33 điểm %), xác nhận generation
có run-to-run variance. Tuy vậy, ở nhóm bị prune,
`network_security_easy_001` giảm từ 100% xuống 50%. Hai article bị budget loại
trong case này đều là REMOVE ở baseline và expected article vẫn còn, nhưng prompt
Applicability ngắn hơn làm các decision/context phía sau thay đổi và Generation
cite thêm hai căn cứ không thuộc ground truth.

Do ràng buộc là metric end-to-end, không phải chỉ bảo toàn expected article trong
trace, Phase 1 **không đạt điều kiện promotion**. Default production phải là
`LEGAL_APPLICABILITY_CANDIDATE_BUDGET=0` (unlimited). Selector và trace được giữ
ở trạng thái opt-in để A/B; không thay đổi đường chạy mặc định.

## Proposed Architecture

```text
Domain Selection
  → Behavior Card
  → Hybrid + Cross Encoder + Behavior Ranking
  → top-10 retrieval                     # không đổi
  → Recursive Retrieval                  # không đổi
  → Legal Relevance, article-level       # không đổi scoring
  → Candidate Priority
       priority = raw legal-relevance score
       tie-break = current stable article order
  → Optional Adaptive Candidate Budget
       threshold = score của article thứ 7
       keep <= 7 unique articles
       low-score seed vẫn đủ điều kiện cạnh tranh
  → one article-batched Applicability call
  → Applicability validation
  → optional Dynamic Weak Keep Quota
  → Generation                           # prompt không đổi
  → Grounding validator + renderer        # output không đổi
```

### Candidate Priority

Không tạo model hoặc LLM call mới. Dùng raw cross-encoder score mà Legal Relevance
đã tính cho query gốc.

Không dùng:

- global article blacklist;
- output Applicability từ query cũ;
- benchmark label;
- domain/behavior ground truth;
- fixed absolute score threshold.

### Adaptive Candidate Budget

Thiết kế thí nghiệm: 7 unique article; production default: unlimited vì full
benchmark không đạt Citation Accuracy guardrail.

- Nếu có tối đa 7 article, pipeline giữ nguyên.
- Nếu có hơn 7 article, chọn 7 article có Legal Relevance score cao nhất.
- Đây là per-query threshold, tự thích nghi với scale score của từng query.
- Legal Relevance LOW seed không bị loại trước khi cạnh tranh budget. Cơ chế
  recovery vẫn còn hiệu lực; hai low-relevance seed đã tạo correct citation trong
  benchmark đều nằm trong top-6 theo raw score và do đó cũng nằm trong top-7.
- Recall@10 được tính trên `retrieved_chunks` trước stage này nên về kiến trúc
  không thể bị thay đổi.

### Article-level batching

Giữ nguyên một call/query. Batching đã đúng và không có duplicate article trong
prompt. Chỉ giảm số `<CANDIDATE>`.

### Dynamic Weak Keep Quota

Đề xuất cho Phase 2, không gộp ngay với Candidate Budget.

Replay cho thấy quota một WEAK_KEEP/query, xếp theo Legal Relevance score:

- giữ đủ 4/4 expected WEAK_KEEP;
- giữ đủ 4/4 correctly cited WEAK_KEEP;
- loại 23/39 WEAK_KEEP không thuộc ground truth;
- sáu article bị loại từng tạo citation, nhưng đều là false-positive citation.

Quota này có khả năng tăng precision và giảm Generation context, nhưng ảnh hưởng
Generation là counterfactual mạnh hơn pre-call budget. Chỉ activate sau một
benchmark riêng.

### Những hướng không triển khai

| Hướng | Quyết định | Lý do |
|---|---|---|
| Duplicate Collapse mới | Không | 0 exact duplicate; article batching đã có |
| Early Stop Applicability | Không | Một batch call; không tiết kiệm được call đang chạy |
| Global article blacklist | Không | Decision phụ thuộc query; dễ overfit 30 case |
| Fixed absolute threshold | Không | Hai correct seed có relative relevance LOW |
| Strong behavior pruning mới | Không | Behavior coverage thấp; object-only/empty cards cần seed recovery |
| Cache applicability theo article | Không | Cùng article có decision khác nhau giữa query |
| Thêm LLM judge/reranker | Không | Trái ràng buộc không thêm LLM call |

## Expected Impact

### Phase 1: Candidate Budget 7

Ước lượng ban đầu từ replay và latency regression, kèm kết quả đo thật:

| Metric | Baseline | Expected | Observed |
|---|---:|---:|---:|
| Applicability candidate | 170 | 155 | 155 |
| Applicability candidate/query | 5.67 | 5.17 | 5.17 |
| Citation Accuracy | 21.67% | ≥21.67% | **18.89%** |
| Recall@10 | 41.67% | 41.67% | 41.67% |
| Wrong Domain Rate | 19.56% | 19.56% | 19.56% |
| Applicability latency | 52.91 s | khoảng 49.0 s | 47.25 s |
| Total latency | 106.42 s | khoảng 102–103 s | 96.33 s |

Latency đạt và vượt ước lượng, nhưng Citation Accuracy fail nên Expected Impact
của cấu hình được promotion là **không thay đổi so với baseline**: budget mặc
định unlimited.

### Phase 2: Weak Keep Quota 1

Chỉ ước lượng, phải benchmark riêng. Phase này chưa được triển khai vì điều kiện
Phase 1 pass không thỏa:

| Metric | Expected direction |
|---|---|
| Citation Accuracy | Không giảm; có khả năng tăng do bỏ false-positive sources |
| Recall@10 | Không đổi |
| Wrong Domain | Không đổi |
| Precision final context | Tăng |
| Applicability latency | Không đổi |
| Generation latency | Có thể giảm do context và số section giảm |

### Precision

Nếu decision của run hiện tại ổn định, Phase 1 loại 5 article được Applicability
giữ nhưng không expected/cited. Relevant-provision precision trong kept article
tăng xấp xỉ từ `12/86 = 13.95%` lên `12/81 = 14.81%`.

Đây là proxy theo benchmark ground truth vốn không exhaustive cho mọi căn cứ có
thể hợp pháp; không được diễn giải là legal precision tuyệt đối.

## Risk Analysis

### Candidate Budget

Rủi ro:

- Benchmark chỉ có 30 case; query ngoài distribution có thể cần hơn 7 article.
- LLM decision có thể đổi khi candidate ID/order/prompt length đổi.
- Một low relevance seed hữu ích có thể đứng ngoài top-7 ở query mới.

Giảm thiểu:

- Budget cấu hình bằng environment variable, rollback không đổi API.
- Chỉ cap unique article sau khi toàn bộ seed đã qua Legal Relevance recovery.
- Stable tie-break để kết quả deterministic.
- Full benchmark guardrail bắt buộc trước promotion.
- Ghi decision stage và reason cho candidate bị budget prune.

### Priority Score

Rủi ro:

- Raw Legal Relevance score không phải xác suất và scale thay đổi giữa query.
- Score có thể ưu tiên semantic similarity hơn legal applicability.

Giảm thiểu:

- Chỉ dùng ordinal ranking trong cùng query, không dùng absolute threshold.
- Applicability vẫn quyết định KEEP/WEAK_KEEP/REMOVE cho top budget.
- Không thay Phase 2 ranking, Recall@10 hoặc Recursive.

### Dynamic Weak Keep Quota

Rủi ro cao hơn:

- WEAK_KEEP là recovery path; quota quá thấp có thể tái tạo lỗi mất seed.
- Generation có thể chuyển sang cite một source khác sau khi source nhiễu bị bỏ.

Giảm thiểu:

- Tách Phase 2, mặc định off cho tới khi Phase 1 pass.
- Rank weak candidates theo cùng Legal Relevance score.
- Guardrail exact trên Citation Accuracy, Recall@10 và Wrong Domain.

### Seed Protection

Giữ mọi seed vô hạn gây latency; loại mọi LOW seed làm mất hai correct citation
đã recover. Thiết kế mới giữ seed protection như eligibility, không như unlimited
reservation. Đây là trade-off chính.

### Latency variance

Ollama latency phụ thuộc output length và runtime load. So sánh benchmark phải:

- cùng model/config/index;
- chạy cùng host;
- không chạy app cạnh tranh Qdrant/Ollama;
- báo cả mean và per-case;
- không coi chênh lệch nhỏ dưới run variance là thắng.

## Implementation Plan

### Phase 1 — Pre-Applicability Candidate Budget

Phạm vi:

- Thêm selector nội bộ sau Legal Relevance, trước Applicability.
- Priority theo raw Legal Relevance score.
- Thí nghiệm budget 7, cấu hình bằng
  `LEGAL_APPLICABILITY_CANDIDATE_BUDGET`; production default 0/unlimited.
- Thêm trace cho selected/pruned, priority score và reason.
- Không đổi public API, chatbot output, Generation prompt hoặc benchmark.

Benchmark độc lập:

1. Unit test selector, stable tie, fewer-than-budget, low seed eligibility.
2. Chạy toàn bộ test suite.
3. Chạy 30-case benchmark.
4. Pass khi:
   - Citation Accuracy ≥ 21.67%.
   - Recall@10 ≥ 41.67%.
   - Wrong Domain Rate ≤ 19.56%.
   - Hallucinated Citation = 0.
   - Applicability candidate ≤ 155 hoặc giải thích được variation.
   - Total latency giảm có ý nghĩa.

Rollback:

- Set budget về 0/unlimited hoặc revert selector; không migration dữ liệu.

Kết quả:

- Unit/integration suite pass 92 test trong Python 3.11 environment đầy đủ.
- Full 30-case benchmark fail Citation Accuracy guardrail.
- Selector không được activate mặc định; Phase 1 dừng ở opt-in instrumentation.

### Phase 2 — Dynamic Weak Keep Quota

Điều kiện bắt đầu:

- Phase 1 pass toàn bộ guardrail.

Phạm vi:

- Giữ toàn bộ KEEP.
- Với WEAK_KEEP, giữ top-1/query theo Legal Relevance score.
- Config riêng; có thể benchmark off/on mà không đổi Phase 1.
- Không thay validator hoặc prompt.

Benchmark độc lập:

1. Unit test bảo vệ top weak seed và deterministic ranking.
2. Chạy 30-case benchmark với Phase 1 cố định.
3. Chỉ activate mặc định nếu Citation Accuracy không giảm và total latency hoặc
   final-context precision cải thiện.

Rollback:

- Quota 0/unlimited; Candidate Budget Phase 1 vẫn hoạt động.

### Phase 3 — Promotion & Latency Instrumentation

Phạm vi:

- Instrument riêng Hybrid, Recursive, Legal Relevance, Applicability, Generation,
  grounding/renderer mà không thêm model call.
- Chạy benchmark A/B cuối cùng với từng feature flag.
- Chỉ promote tổ hợp nhỏ nhất pass guardrail.

Không triển khai trong Phase 3 nếu không có bằng chứng mới:

- early stop nhiều LLM batch;
- global blacklist;
- duplicate collapse Khoản/Điểm;
- behavior pruning mạnh hơn;
- thay Generation prompt.

## Decision

Không có thay đổi pruning nào đủ bằng chứng để activate trong production.

Đã triển khai dưới feature flag để phục vụ controlled A/B:

```text
Legal Relevance raw-score priority
+ optional top-7 unique-article budget
+ existing one-call article batching
+ existing seed eligibility protection
```

Default là unlimited. Dynamic Weak Keep Quota không được bắt đầu vì Phase 1 fail.
Early stopping, duplicate collapse mới, global blacklist và behavior pruning mới
không được triển khai từ benchmark hiện tại.
