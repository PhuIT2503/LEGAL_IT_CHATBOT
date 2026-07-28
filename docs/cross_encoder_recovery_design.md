# Cross Encoder Recovery — Audit and Design

Ngày audit: 2026-07-27

## Executive decision

Cross Encoder có measured leverage lớn, nhưng chưa có chiến lược nào đủ bằng
chứng để triển khai production trong task này.

- Official baseline: 16/44 expected provisions, macro Recall@10 41.67%.
- Perfect CE ranking + threshold oracle: 24/44, macro Recall@10 60.00%.
- Năm provisions có first failed stage là CE threshold.
- Sau legal adjudication, bốn trong năm là confirmed correct; một nhãn là
  likely incorrect.
- Hạ threshold về 0 phục hồi ba confirmed provisions và tăng macro Recall@10
  lên 50.00%, nhưng thêm 53 retrieval slots trên 30 case và chưa chứng minh
  Citation Accuracy không giảm.
- Weighting và document normalization replay chưa cho thấy gain ổn định.
- Citation Accuracy là downstream metric. Retrieval-only oracle không thể
  chứng minh guardrail `Citation Accuracy >= 21.67%`.

Vì vậy tài liệu chỉ đề xuất các phase thí nghiệm/shadow benchmark. Không đề xuất
production implementation hoặc thay đổi threshold lúc này.

## Scope and controls

Giữ nguyên:

- Generation;
- Applicability;
- Recursive Retrieval;
- public API và chatbot output;
- evaluation metrics;
- benchmark runner và benchmark labels;
- production Retrieval source/config/index.

Audit không gọi thêm LLM. Candidate pools được replay bằng model/index/config
hiện tại; mọi chiến lược được tính offline trên pool đóng băng.

Nguồn:

- `evaluation/benchmark/benchmark.json`
- `evaluation/results/benchmark_30_applicability_recovery_20260726/`
- `data/.qdrant_base`
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
  revision `1427fd652930e4ba29e8149678df786c240d8825`
- `CROSS_ENCODER_MAX_LENGTH=384`
- `CROSS_ENCODER_CANDIDATE_RATIO=0.20`

Retrieval forensic replay khớp exact top-10 chunks của 30/30 authoritative
testcases.

## Current CE path

```text
original + expanded RRF candidate union
→ build passage: domains + source + article title + chunk text
→ enrich original query bằng Behavior Card nếu có
→ Cross Encoder raw sigmoid score
→ pool-level min-max semantic normalization
→ weighted score:
   behavior-enabled = 0.64 semantic + 0.28 behavior + origin/rank bonus
   behavior-empty   = 0.90 semantic + origin/rank bonus
→ Behavior Gate
→ relative threshold = 0.20 × best score
→ article/clause dedup
→ balanced top-10
```

Một candidate có thể chết vì hai hiện tượng khác nhau:

1. CE làm candidate tụt hạng mạnh.
2. Candidate vẫn còn trong pool nhưng normalized/final score thấp hơn relative
   threshold.

Chỉ thay threshold không sửa ranking error. Chỉ thay weighting cũng không giúp
nếu mọi behavior score của target đều bằng 0.

## Root cause

### 1. Generic multilingual CE không được calibration cho Vietnamese legal text

Raw scores của năm losses nằm trong khoảng `0.002289–0.028664`, dù bốn
provisions có direct legal fit. Giá trị thấp không chỉ xuất hiện ở candidate
không liên quan; vì thế raw score không thể dùng như probability of legal
applicability.

### 2. Pool-level min-max normalization không ổn định giữa query

`semantic_score` phụ thuộc minimum/maximum của toàn candidate pool. Cùng một raw
score có thể nhận normalized score khác khi expansion thêm candidates hoặc
domain pool thay đổi. Relative threshold lại so score đã normalization với best
candidate của chính pool, tạo hai lớp query-relative calibration.

### 3. CE đảo thứ tự của strong RRF candidates

Trong 25 expected provisions đã vào candidate union:

- CE cải thiện rank: 6;
- giữ rank: 9;
- làm rank xấu đi: 10;
- average expected rank delta: `+1.04` rank, tức trung bình xấu hơn.

Ba confirmed CE losses bắt đầu ở union rank 1, 3 và 9 nhưng xuống CE rank 7, 10
và 10. Strong retrieval evidence đang bị CE phủ nhận.

### 4. Behavior score không cung cấp recovery signal

Cả năm CE losses có `behavior_score=0`. Tăng behavior weight hoặc cộng một
behavior bonus toàn cục không thể cứu chúng. Rule bonus dùng document/label biết
trước sẽ là ground-truth leakage, không phải ranking improvement.

### 5. Long article representation và truncation

Các expected ở cấp Điều như Điều 36 hoặc Điều 18 dùng article-level context.
Passage gồm domain, source, title rồi mới đến text và bị giới hạn 384 tokens.
Evidence trực tiếp có thể bị pha loãng hoặc nằm sau phần đầu. Đây là hypothesis
cần passage-ablation benchmark, chưa phải root cause đã chứng minh.

### 6. Ground truth noise làm CE trông xấu hơn thực tế

`cyber_attack_medium_001 — NĐ15 Điều 81` yêu cầu yếu tố chiếm đoạt tài sản mà
query không nêu. CE đưa provision này xuống rank 20 với raw score 0.002302 là
semantic behavior hợp lý, không nên được dùng làm recovery target.

## Evidence — five threshold losses

`Hybrid O/E` là rank/score RRF query gốc/mở rộng. `Union` là rank trong union
original-first. `CE rank` là rank sau weighted CE rerank, trước Behavior Gate và
threshold.

| Testcase / provision | Legal status | Hybrid O r/s | Hybrid E r/s | Union | CE rank | CE raw | Semantic | Behavior | Final | Threshold evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `cyber_attack_medium_001` — NĐ15 Đ.81 | Likely incorrect | 6 / 0.2500 | 12 / 0.1111 | 6 | 20 | 0.002302 | 0.009290 | 0 | 0.069279 | final `< 0.20 × best = 0.178400` |
| `malware_medium_001` — ANM Đ.36 | Confirmed | 1 / 0.7000 | 12 / 0.1250 | 1 | 7 | 0.020982 | 0.123040 | 0 | 0.195736 | semantic `< 0.20 × best = 0.200000` |
| `advertising_hard_001` — BVQLNTD Đ.10 | Confirmed | 12 / 0.1611 | 2 / 0.5000 | 12 | 20 | 0.027691 | 0.033683 | 0 | 0.082986 | final `< 0.20 × best = 0.128560` |
| `network_security_hard_001` — ANM Đ.36 | Confirmed | 3 / 0.4103 | — | 3 | 10 | 0.002289 | 0.021255 | 0 | 0.100558 | semantic `< 0.20 × best = 0.200000` |
| `electronic_transactions_medium_001` — GDĐT Đ.18 | Confirmed | 9 / 0.1429 | 16 / 0.1000 | 9 | 10 | 0.028664 | 0.028224 | 0 | 0.107624 | semantic `< 0.20 × best = 0.200000` |

Interpretation:

- Threshold-only có thể cứu Điều 36 ở `malware_medium`, Điều 36 ở
  `network_security_hard` và Điều 18 ở `electronic_transactions_medium`.
- Điều 10 ở `advertising_hard` đã ở CE rank 20; hạ threshold không đủ để đưa nó
  vào top-10.
- Điều 81 ở `cyber_attack_medium` vừa rank 20 vừa là likely-incorrect label;
  không được coi là target cho CE tuning.

## Oracle upper bound

### Official benchmark oracle

CE oracle cho expected provision đã qua Domain, Candidate Generation và Behavior
Gate:

1. gán expected candidate maximum CE/final score;
2. chạy lại threshold, dedup và balanced top-10;
3. không inject candidate không có trong union;
4. không thay Domain, Behavior, Applicability hoặc Generation.

| Replay | Macro R@10 | Hits | Delta |
|---|---:|---:|---:|
| Baseline | 41.67% | 16/44 | — |
| Restore five CE-threshold losses only | 53.33% | 21/44 | +5 |
| Perfect CE ranking + threshold, gồm ba top-k losses | 60.00% | 24/44 | +8 |

`60.00%` là upper bound label-assisted, không phải forecast cho một model thực.

### Adjudicated sensitivity

Không thay benchmark; đây chỉ là selection diagnostic.

| Confirmed-only scope | Macro recall trên 24 case có confirmed label | Hits |
|---|---:|---:|
| Baseline | 52.08% | 15/29 |
| Perfect CE ranking + threshold | 72.92% | 21/29 |

CE oracle vẫn có `+6` confirmed provisions sau khi loại likely-incorrect và
ambiguous labels. Leverage của CE là thật, không chỉ do benchmark noise.

### Citation Accuracy boundary

Không tồn tại retrieval-only oracle hợp lệ để chứng minh Citation Accuracy.
Thêm candidate có thể:

- bị Applicability loại và không thay answer;
- đi vào final context, giúp citation;
- hoặc gây competition/context drift và làm citation giảm.

Do đó mọi con số oracle trong tài liệu chỉ chứng minh Recall upper bound.
Guardrail Citation Accuracy chỉ được chứng minh bằng full unchanged downstream
benchmark trong shadow experiment tương lai.

## Strategy replay

### 1. Fixed threshold

Exact replay trên current candidate pools:

| CE ratio | Macro R@10 | Hits | Macro top-10 precision | Avg records | Recovered confirmed target |
|---:|---:|---:|---:|---:|---|
| 0.20 baseline | 41.67% | 16 | 14.81% | 6.93 | — |
| 0.15 | 41.67% | 16 | 13.65% | 7.50 | — |
| 0.10 | 45.00% | 17 | 14.28% | 7.97 | `malware_medium` Đ.36 |
| 0.05 | 45.00% | 17 | 13.37% | 8.37 | `malware_medium` Đ.36 |
| 0.00 | 50.00% | 19 | 14.04% | 8.70 | thêm `network_security_hard` Đ.36 và `electronic_transactions_medium` Đ.18 |

Kết luận:

- `0.10` có measured recall gain nhỏ nhưng chưa chứng minh Citation Accuracy.
- `0.05` thêm noise mà không thêm expected hit so với `0.10`.
- `0.00` tăng ba confirmed hits nhưng thêm 53 retrieval slots theo exact
  aggregate trên 30 case; sensitivity replay cho thấy 14 case nhận thêm
  candidates. Đây không phải “free recall”.
- Fixed threshold không cứu `advertising_hard` Đ.10.

Không đề xuất đổi production threshold.

### 2. Weighting

Exploratory replay recompute score trên frozen raw CE/behavior/RRF values:

| Weights `(CE, behavior, RRF)` | Macro R@10 | Hits | Changed cases | Baseline chunks removed / added |
|---|---:|---:|---:|---:|
| `(0.70, 0.20, 0.10)` | 41.67% | 16 | 18 | 8 / 7 |
| `(0.60, 0.20, 0.20)` | 41.67% | 16 | 23 | 10 / 16 |
| `(0.50, 0.20, 0.30)` | 41.67% | 16 | 23 | 15 / 24 |
| `(0.80, 0.00, 0.20)` | 41.67% | 16 | 23 | 28 / 14 |

Không configuration nào tăng expected hits. RRF weight tăng churn nhưng không
đưa bốn confirmed targets vào top-10. Không có bằng chứng để thay weighting.

### 3. Dynamic threshold

Tested rule:

```text
threshold = min(0.20 × best_score, score_at_target_pool_rank)
```

Rule giữ mọi baseline-passing record và chỉ mở threshold nếu pool sau gate có
ít hơn target. Exploratory results:

| Minimum pool target | Macro R@10 | Hits | Avg records | Baseline chunks lost | Added chunks |
|---:|---:|---:|---:|---:|---:|
| 5 | 41.67% | 16 | 6.90 | 0 | 3 |
| 7 | 41.67% | 16 | 7.27 | 0 | 14 |
| 10 | 48.33% | 18 | 8.23 | 0 | 43 |
| 12 | 50.00% | 19 | 8.53 | 0 | 52 |

Đây là sensitivity replay, không phải promotion evidence: target 10/12 tăng
context exposure đáng kể và Citation Accuracy chưa được đo. Dynamic threshold
chỉ đáng đưa vào shadow benchmark nếu có downstream guardrail.

### 4. Behavior bonus

Năm CE losses đều có behavior score 0. Các weighting grids giảm behavior weight
vẫn không tăng Recall. Muốn bonus riêng cho document/article biết trước sẽ dùng
benchmark label làm feature và gây leakage.

Decision: reject behavior bonus trong CE recovery phase hiện tại.

### 5. Document-aware normalization

Hai exploratory forms:

- min-max raw CE trong từng document;
- raw score chia maximum raw score của document.

| Normalization | Macro R@10 | Hits | Macro precision | Changed cases | Removed / added chunks |
|---|---:|---:|---:|---:|---:|
| Within-document min-max | 41.67% | 16 | 13.93% | 22 | 29 / 51 |
| Within-document max ratio | 43.33% | 17 | 13.44% | 23 | 26 / 68 |

Max-ratio cứu một hit nhưng gây churn/noise lớn. Nó ưu ái document có toàn bộ
raw scores thấp và làm score giữa documents mất ý nghĩa. Không triển khai.

## Proposed recovery architecture

Không thay production architecture trong task này. Architecture dành cho
offline evaluation:

```text
Frozen benchmark + adjudication labels
            │
            ├── frozen candidate pools, exact current passages
            │
            ├── passage ablations
            │     source/title/text order
            │     child-only vs parent-aware bounded representation
            │
            ├── CE model candidates pinned by revision
            │
            └── calibration/ranking policies
                  raw score
                  pairwise rank
                  RRF-preserving fallback
                  conservative dynamic threshold
                         │
                         ▼
             offline retrieval metrics
             + confirmed-only sensitivity
                         │
             only shortlisted variant
                         ▼
          full unchanged downstream shadow benchmark
          Citation / Wrong Domain / latency guardrails
```

Principles:

1. Optimize rank quality before threshold.
2. Never train/tune against likely-incorrect or ambiguous labels.
3. Preserve current model/threshold as control in every experiment.
4. No extra CE or LLM call in the production path.
5. No promotion based only on Recall@10.

## Design proposal

### Phase CE-1 — Frozen CE evaluation harness

Scope:

- serialize read-only candidate pool inputs/outputs outside production state;
- include query, behavior-enriched query, exact passage, truncation length, raw
  score, normalized score, RRF ranks and final ranks;
- attach adjudication status only for analysis, never as model input;
- verify current replay exact against authoritative 30/30 top-10.

Expected gain:

- Production benchmark: 0; audit infrastructure only.
- Removes measurement ambiguity around model vs threshold vs truncation.

Expected latency:

- Production: 0.
- Offline: one CE pass per frozen candidate pool/model.

Risk:

- Snapshot drift or label leakage into scoring.

Guardrail:

- Candidate snapshot hash and scorer revision must be recorded.
- Adjudication field is excluded from scorer inputs.

### Phase CE-2 — Model and passage bakeoff

Scope:

- compare current CE with a small number of pinned Vietnamese/multilingual
  rerankers on identical frozen pools;
- test passage representations independently from model choice;
- report MRR, expected rank delta, R@10, confirmed-only sensitivity, inference
  latency and score calibration;
- no broad hyperparameter search on the 30-case benchmark; use nested holdout or
  leave-one-category-out selection.

Promotion target:

- benchmark R@10 at least 48.33% on frozen retrieval;
- recover at least three of four confirmed CE-threshold losses;
- no currently retrieved expected provision lost;
- macro precision not below baseline 14.81% by more than 0.5 percentage point;
- no gain credit for likely-incorrect/ambiguous labels.

Expected gain:

- Realistic offline target: `+2–4` confirmed provisions;
- measured upper bound: `+6` confirmed provisions across threshold/top-k.

Expected latency:

- Offline depends on candidate model.
- A production candidate is ineligible if measured CE p95 exceeds current CE
  p95 by more than 10%; no extra model call is allowed.

Risk:

- Overfit 30 cases;
- larger model increases 4.22-second Retrieval latency;
- passage changes may favor long articles and raise wrong-domain noise.

Rollback:

- current model name, revision, passage format and ratio 0.20 remain the frozen
  control; reject candidate without touching production.

### Phase CE-3 — Conservative policy shadow benchmark

Scope:

- take at most one CE-2 winner;
- compare fixed ratio 0.10, conservative dynamic fill and RRF-preserving policy;
- run the existing benchmark runner and unchanged Applicability/Generation;
- no production rollout during this phase.

Required full-benchmark guardrails:

| Metric | Promotion condition |
|---|---:|
| Recall@10 | `> 41.67%` and reproduce offline gain |
| Citation Accuracy | `>= 21.67%` |
| Wrong Domain Rate | `<= 19.56%` |
| Hallucinated Citation | `= 0` |
| Recall@10 current-hit regression | 0 currently retrieved expected provisions lost |
| Retrieval latency | p95 increase `<= 10%` |
| Total latency | no statistically/materially adverse increase |

Expected gain:

- Planning range: macro R@10 45–50%;
- Citation Accuracy: no claimed gain; must be measured and non-decreasing.

Expected latency:

- Threshold/weight policy computation: negligible and no extra CE call.
- More retained context may increase Applicability/Generation work, so total
  latency must be measured rather than inferred from Retrieval latency.

Risk:

- Context competition can reduce Citation Accuracy even when Recall improves.
- Dynamic fill can expose 43–52 extra records across 30 cases.
- Small benchmark makes one-case changes look large.

Rollback:

- shadow-only experiment has no user traffic;
- reject the entire variant if any guardrail fails;
- restore exact current scorer revision, weights, ratio and passage format.

## Expected benchmark impact

| Strategy | Measured/upper-bound R@10 | Citation Accuracy evidence | Decision |
|---|---:|---|---|
| Current baseline | 41.67% | 21.67% | Control |
| Fixed ratio 0.10 | 45.00% | Not measured | Shadow candidate only |
| Fixed ratio 0 | 50.00% | Not measured; more noise | Do not promote |
| Dynamic fill 10 | 48.33% exploratory | Not measured | CE-3 candidate only |
| Weighting grids | 41.67% | No recall gain | Reject |
| Document max normalization | 43.33% | Precision/churn adverse | Reject |
| Perfect CE oracle | 60.00% | Oracle cannot establish citation | Upper bound only |

Không dự báo Citation Accuracy tăng tuyến tính theo Recall. Thêm đúng provision
vào top-10 chỉ tạo cơ hội cho Applicability/Generation; nó không bảo đảm
provision được giữ và trích dẫn.

## Risk analysis

### Ground truth risk

22.73% annotations là likely incorrect và 11.36% ambiguous theo engineering
adjudication. Tuning trực tiếp trên tất cả 44 labels có nguy cơ tạo semantic
regression có chủ đích.

### Calibration risk

Pool-relative min-max và threshold khiến thay domain/expansion có thể đổi score
dù query-document pair không đổi.

### Precision and wrong-domain risk

Threshold/dynamic fill thêm low-score candidates. Domain filter không bảo đảm
article-level relevance, nên Wrong Domain và within-domain noise đều có thể tăng.

### Latency risk

Một reranker lớn hơn làm tăng Retrieval latency; một pool rộng hơn có thể làm
tăng downstream work ngay cả khi CE call count không đổi.

### Benchmark overfit

30 cases không đủ để chọn nhiều model, passage formats và thresholds trên cùng
test set. CE-2 phải giới hạn hypotheses và dùng category holdout.

## Rollback plan

Task hiện tại không có production mutation.

Cho phase tương lai:

1. Pin current scorer model/revision, passage format, weights và threshold trong
   baseline manifest.
2. Mỗi variant chỉ thay một dimension.
3. Lưu exact candidate/output hashes.
4. Nếu một guardrail fail, bỏ variant; không “bù” bằng tune metric khác.
5. Production change chỉ được xem xét sau CE-3. Rollback phải khôi phục manifest
   baseline, không cần đổi API/index/benchmark.

## Final decision

Oracle chứng minh CE còn upper bound đáng kể, nhưng chưa chứng minh bất kỳ thay
đổi CE nào tăng Recall@10 mà không giảm Citation Accuracy.

Theo success criterion, chưa được đề xuất production implementation. Phase tiếp
theo hợp lệ duy nhất là CE-1/CE-2 offline và CE-3 shadow benchmark. Chỉ khi
CE-3 đạt đồng thời Citation Accuracy, Wrong Domain, hallucination và latency
guardrails mới lập implementation proposal riêng.
