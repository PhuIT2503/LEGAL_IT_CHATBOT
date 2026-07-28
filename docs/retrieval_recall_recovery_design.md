# Retrieval Recall Recovery — Forensic Audit & Design

Ngày audit: 2026-07-27

## Executive decision

Phase này chỉ audit Retrieval. Applicability, Recursive Retrieval và Generation
được giữ làm control variable.

Không có thay đổi pipeline nào được triển khai từ tài liệu này.

Nguồn chuẩn:

- Benchmark:
  `evaluation/benchmark/benchmark.json`
- Benchmark SHA-256:
  `126438a83f9902d950b140f74fd1f527b899ec9dc160ce1163f47c0c5d3e114c`
- Authoritative run:
  `evaluation/results/benchmark_30_applicability_recovery_20260726/`
- Mode/model/index:
  `critic`, `top_k=10`, `AITeamVN/Vietnamese_Embedding_v2`,
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`,
  `data/.qdrant_base`.

Retrieval-only forensic replay tái tạo đúng từng chunk top-10 của 30/30
testcase trong authoritative run. Số mismatch là 0.

Kết luận:

1. 44/44 expected provision tồn tại trong Qdrant corpus.
2. 28 provision bị mất trước top-10.
3. 19/28 loss xảy ra trước hoặc tại candidate generation:
   9 ở Domain Selection và 10 ở Dense/BM25 prefetch.
4. 9/28 loss xảy ra sau khi expected provision đã vào candidate union:
   1 ở Behavior Gate, 5 ở Cross-Encoder threshold và 3 ở top-k.
5. Hybrid RRF Merge và duplicate collapse không làm mất expected provision nào.
6. Oracle Cross Encoder hiện có upper bound lớn nhất trong một stage:
   16 → 24 expected provision, macro Recall@10 41.67% → 60.00%.
7. Oracle Domain chạy xuyên downstream đạt 16 → 22 provision,
   macro Recall@10 55.00%.
8. Tăng prefetch 20 → 40 chỉ cứu một provision. Hạ CE threshold về 0 chỉ cứu
   ba provision và làm retrieval precision proxy giảm.
9. Có ground-truth mismatch nghiêm trọng ở sáu testcase cyber/SQL/malware.
   Không được tối ưu retriever theo sáu nhãn này trước khi có legal adjudication.

## 1. Audit methodology

### 1.1 Code path đã đọc

Audit bao phủ toàn bộ source/config/test/evaluation trong repository, đặc biệt:

- Domain registry và selector:
  `src/retrieval/legal_domains.py`
- Dense/BM25/RRF:
  `src/retrieval/qdrant_hybrid_search.py`,
  `src/retrieval/bm25_sparse.py`
- Query Expansion:
  `src/agents/common/query_expansion.py`
- Behavior extraction/gate:
  `src/retrieval/legal_behaviors.py`,
  `src/agents/common/retrieval_ranking.py`
- Cross Encoder, threshold, dedup và top-k:
  `src/agents/common/cross_encoder_reranker.py`,
  `src/agents/common/retrieval_ranking.py`
- Orchestration:
  `src/agents/agent_retrieval/node_hybrid_search.py`,
  `src/agents/agent_retrieval/pipeline.py`,
  `src/workflow/pipeline.py`
- Ingest/chunk metadata:
  `src/data_ingestion/qdrant_local_ingest.py`,
  `src/data_processing/chunking.py`
- Evaluation contract:
  `evaluation/run_benchmark.py`,
  `evaluation/metrics.py`,
  `evaluation/reporting.py`
- Retrieval/domain/behavior/query-expansion tests trong `tests/` và
  deterministic metric tests trong `evaluation/tests/`.

### 1.2 Vì sao phải replay ngoài benchmark runner

`benchmark_details.json` chỉ lưu final top-10 và downstream trace. Nó không lưu:

- raw Dense prefetch;
- raw BM25 prefetch;
- RRF rank trước Cross Encoder;
- candidate pool trước/sau Behavior Gate;
- candidate pool trước/sau CE threshold;
- rank trước/sau dedup và balanced top-k.

Audit vì vậy chạy một replay retrieval-only, không gọi LLM, không sửa source,
không đổi index và không ghi vào pipeline state. Replay dùng đúng:

```text
domain filter
→ original Dense top-20 + BM25 top-20
→ Qdrant RRF top-20
→ expanded Dense/BM25/RRF top-20 nếu có
→ original-first candidate union
→ existing Cross Encoder + Behavior score
→ existing Behavior Gate
→ existing CE relevance threshold 0.20
→ existing document/article/clause dedup
→ existing balanced top-10
```

Diagnostic rank được mở read-only đến 200 để phân biệt “rank 21–200” với
“không xuất hiện trong top-200”. Pipeline thật vẫn dùng prefetch 20.

### 1.3 Quy ước attribution

Dense và BM25 là hai nhánh song song, không phải hai stage tuần tự. Vì vậy:

- `dense_bm25_candidate_generation` nghĩa là expected provision không có trong
  top-20 của cả hai nhánh, ở cả query gốc lẫn query mở rộng;
- không gán sai rằng Dense “giết” một candidate mà BM25 chưa từng nhận;
- Hybrid Merge chỉ chịu trách nhiệm nếu candidate có trong ít nhất một raw
  branch top-20 nhưng biến mất ở RRF top-20.

Với expected ở cấp Điều, rank/score trong bảng là candidate matching tốt nhất
của Điều đó tại stage tương ứng. Với expected có Khoản/Điểm, match là exact theo
evaluation contract.

### 1.4 Domain routing theo từng testcase

`Expected provision corpus domains` là hợp của domain metadata của mọi expected
provision trong testcase. Bảng này giải thích trực tiếp chín provision mất cơ
hội search ngay tại Domain Selection.

| Case | Selected domains | Expected provision corpus domains |
|---|---|---|
| `deepfake_easy_001` | `civil_personality`, `advertising`, `artificial_intelligence`, `cybersecurity`, `personal_data` | `cybersecurity` |
| `deepfake_medium_001` | `personal_data` | `cybersecurity` |
| `deepfake_hard_001` | `advertising`, `artificial_intelligence`, `cybersecurity`, `personal_data`, `civil_personality` | `personal_data`, `cybersecurity` |
| `personal_data_easy_001` | `advertising`, `telecommunications`, `personal_data` | `personal_data` |
| `personal_data_medium_001` | `personal_data` | `personal_data` |
| `personal_data_hard_001` | `data_governance` | `personal_data` |
| `cyber_attack_easy_001` | `cybersecurity`, `personal_data`, `digital_technology`, `electronic_transactions`, `telecommunications` | `administrative_penalty`, `advertising`, `cybersecurity`, `electronic_transactions`, `telecommunications` |
| `cyber_attack_medium_001` | `cybersecurity`, `personal_data`, `digital_technology`, `electronic_transactions`, `telecommunications` | `administrative_penalty`, `advertising`, `cybersecurity`, `electronic_transactions`, `telecommunications` |
| `cyber_attack_hard_001` | `cybersecurity`, `administrative_penalty`, `digital_technology` | `cybersecurity`, `administrative_penalty`, `advertising`, `electronic_transactions`, `telecommunications` |
| `sql_injection_easy_001` | `cybersecurity`, `data_governance` | `cybersecurity` |
| `sql_injection_medium_001` | `cybersecurity`, `digital_content` | `cybersecurity`, `administrative_penalty`, `advertising`, `electronic_transactions`, `telecommunications` |
| `sql_injection_hard_001` | `cybersecurity`, `data_governance` | `cybersecurity`, `personal_data` |
| `malware_easy_001` | `digital_technology`, `cybersecurity` | `cybersecurity` |
| `malware_medium_001` | `cybersecurity` | `cybersecurity` |
| `malware_hard_001` | `cybersecurity`, `digital_technology` | `cybersecurity` |
| `ai_copyright_easy_001` | `intellectual_property` | `intellectual_property` |
| `ai_copyright_medium_001` | `intellectual_property`, `artificial_intelligence` | `intellectual_property` |
| `ai_copyright_hard_001` | `intellectual_property` | `intellectual_property` |
| `advertising_easy_001` | `personal_data`, `advertising` | `personal_data` |
| `advertising_medium_001` | `advertising`, `consumer_protection` | `advertising`, `consumer_protection` |
| `advertising_hard_001` | `advertising`, `digital_content`, `consumer_protection` | `personal_data`, `advertising`, `consumer_protection` |
| `consumer_easy_001` | `consumer_protection` | `advertising`, `consumer_protection` |
| `consumer_medium_001` | `cybersecurity`, `personal_data`, `digital_technology`, `electronic_transactions`, `telecommunications` | `advertising`, `consumer_protection` |
| `consumer_hard_001` | `e_commerce` | `advertising`, `consumer_protection` |
| `network_security_easy_001` | `cybersecurity`, `digital_technology` | `cybersecurity` |
| `network_security_medium_001` | `digital_technology` | `cybersecurity` |
| `network_security_hard_001` | `cybersecurity` | `cybersecurity` |
| `electronic_transactions_easy_001` | `cybersecurity`, `personal_data`, `digital_technology`, `electronic_transactions`, `telecommunications` | `electronic_transactions` |
| `electronic_transactions_medium_001` | `cybersecurity`, `personal_data`, `digital_technology`, `electronic_transactions`, `telecommunications` | `electronic_transactions` |
| `electronic_transactions_hard_001` | `electronic_transactions` | `electronic_transactions` |

## 2. Current funnel

### 2.1 Retrieval provision funnel

```text
44 expected provisions; 44/44 có trong corpus
↓ Domain Selection
35 còn cơ hội search; 9 mất
↓ Dense/BM25 candidate generation + Query Expansion
25 vào candidate union; 10 mất
↓ Behavior Gate
24 còn lại; 1 mất
↓ Cross-Encoder threshold
19 còn lại; 5 mất
↓ Dedup
19 còn lại; 0 mất
↓ Balanced top-10
16 retrieved; 3 mất
```

### 2.2 End-to-end control funnel

Downstream giữ nguyên theo authoritative benchmark:

```text
44 expected
→ 16 retrieved top-10
→ 16 tới Applicability
→ 12 final context
→ 9 cited đúng
```

Baseline guardrails:

| Metric | Baseline |
|---|---:|
| Macro Recall@10 | 41.67% |
| Micro expected hits | 16/44 = 36.36% |
| Macro retrieval top-10 precision proxy | 14.81% |
| Citation Accuracy | 21.67% |
| Wrong Domain Rate | 19.56% |
| Hallucinated Citation | 0 |
| Retrieval latency | 4.22 s/query |
| Total latency | 106.42 s/query |

## 3. Loss attribution và stage contribution

### 3.1 First failed stage

| First failed stage | Lost provision | % của 28 loss | % của 44 expected | Isolated macro ceiling nếu restore trực tiếp |
|---|---:|---:|---:|---:|
| Dense + BM25 candidate generation | 10 | 35.71% | 22.73% | 60.00% |
| Domain Selection | 9 | 32.14% | 20.45% | 61.67% |
| Cross-Encoder threshold | 5 | 17.86% | 11.36% | 53.33% |
| Balanced top-k | 3 | 10.71% | 6.82% | 48.33% |
| Behavior Gate | 1 | 3.57% | 2.27% | 43.33% |
| Hybrid RRF Merge | 0 | 0% | 0% | 41.67% |
| Duplicate collapse | 0 | 0% | 0% | 41.67% |

`Isolated macro ceiling` trực tiếp đặt provision đầu tiên bị stage đó loại vào
top-10 và giữ các label khác như baseline. Đây là attribution ceiling, không
phải kết quả pipeline có thể đạt chỉ bằng một thay đổi tham số.

### 3.2 Exact loss table

| Case / expected provision | First failed stage | Exact reason |
|---|---|---|
| `deepfake_easy_001` — Luật An ninh mạng 2025 Đ.13 K.3 P.h | `dense_bm25_candidate_generation` | Dense original rank 27, expanded rank 68; BM25 không có trong top-200. |
| `deepfake_medium_001` — Luật An ninh mạng 2025 Đ.7 K.2 P.g | `domain_selection` | Corpus domain `cybersecurity`; selector chỉ mở `personal_data`. |
| `deepfake_medium_001` — Luật An ninh mạng 2025 Đ.13 K.3 P.h | `domain_selection` | Corpus domain `cybersecurity`; selector chỉ mở `personal_data`. |
| `deepfake_hard_001` — Luật Bảo vệ dữ liệu cá nhân 2025 Đ.28 K.1 | `behavior_gate` | Behavior score 0.04 < gate minimum 0.18. |
| `deepfake_hard_001` — Luật An ninh mạng 2025 Đ.13 K.3 P.h | `dense_bm25_candidate_generation` | Chỉ xuất hiện ở Dense expanded rank 200. |
| `personal_data_easy_001` — Luật Bảo vệ dữ liệu cá nhân 2025 Đ.28 K.1 | `top_k_balancing` | Sống qua threshold nhưng đứng dedup rank 16. |
| `personal_data_hard_001` — Luật Bảo vệ dữ liệu cá nhân 2025 Đ.2 | `domain_selection` | Corpus domain `personal_data`; selector chỉ mở `data_governance`. |
| `cyber_attack_medium_001` — Nghị Định 15/2020 Đ.81 | `cross_encoder_threshold` | Final score 0.069279 < threshold 0.178400. |
| `cyber_attack_medium_001` — Luật An ninh mạng 2025 Đ.13 K.3 P.h | `dense_bm25_candidate_generation` | Không có trong Dense/BM25 top-200. |
| `cyber_attack_hard_001` — Luật An ninh mạng 2025 Đ.13 K.3 P.h | `dense_bm25_candidate_generation` | Dense expanded rank 199; BM25 không có trong top-200. |
| `cyber_attack_hard_001` — Nghị Định 15/2020 Đ.81 | `dense_bm25_candidate_generation` | Dense ranks 33/28; BM25 ranks 168/83, đều ngoài prefetch 20. |
| `sql_injection_easy_001` — Luật An ninh mạng 2025 Đ.13 K.3 P.h | `dense_bm25_candidate_generation` | Không có trong Dense/BM25 top-200. |
| `sql_injection_medium_001` — Luật An ninh mạng 2025 Đ.13 K.3 P.h | `dense_bm25_candidate_generation` | Không có trong Dense/BM25 top-200. |
| `sql_injection_hard_001` — Luật An ninh mạng 2025 Đ.13 K.3 P.h | `dense_bm25_candidate_generation` | Chỉ có ở Dense expanded rank 187. |
| `sql_injection_hard_001` — Luật Bảo vệ dữ liệu cá nhân 2025 Đ.20 | `domain_selection` | Corpus domain `personal_data`; selector mở `cybersecurity`, `data_governance`. |
| `malware_medium_001` — Luật An ninh mạng 2025 Đ.36 | `cross_encoder_threshold` | Semantic score 0.123040 < threshold 0.200000. |
| `malware_hard_001` — Luật An ninh mạng 2025 Đ.13 K.3 P.h | `dense_bm25_candidate_generation` | Không có trong Dense/BM25 top-200. |
| `malware_hard_001` — Nghị Định 53/2022 Đ.11 | `top_k_balancing` | Sống qua threshold nhưng đứng dedup rank 11. |
| `ai_copyright_hard_001` — VBHN 361+362 Đ.35 | `top_k_balancing` | Sống qua threshold nhưng đứng dedup rank 16. |
| `ai_copyright_hard_001` — Nghị Định 17/2023 Đ.28 | `dense_bm25_candidate_generation` | Chỉ có ở Dense expanded rank 153. |
| `advertising_hard_001` — Luật Bảo vệ dữ liệu cá nhân 2025 Đ.28 K.1 | `domain_selection` | Corpus domain `personal_data`; selector không mở domain này. |
| `advertising_hard_001` — Luật Bảo vệ quyền lợi người tiêu dùng 2023 Đ.10 | `cross_encoder_threshold` | Final score 0.082986 < threshold 0.128560. |
| `consumer_medium_001` — Luật Bảo vệ quyền lợi người tiêu dùng 2023 Đ.34 | `domain_selection` | Selector fallback không chứa `consumer_protection` hoặc `advertising`. |
| `consumer_medium_001` — Luật Bảo vệ quyền lợi người tiêu dùng 2023 Đ.35 | `domain_selection` | Selector fallback không chứa `consumer_protection` hoặc `advertising`. |
| `consumer_hard_001` — Luật Bảo vệ quyền lợi người tiêu dùng 2023 Đ.25 | `domain_selection` | Selector chỉ mở `e_commerce`; document payload là `consumer_protection`, `advertising`. |
| `network_security_medium_001` — Nghị Định 53/2022 Đ.11 | `domain_selection` | Corpus domain `cybersecurity`; selector chỉ mở `digital_technology`. |
| `network_security_hard_001` — Luật An ninh mạng 2025 Đ.36 | `cross_encoder_threshold` | Semantic score 0.021255 < threshold 0.200000. |
| `electronic_transactions_medium_001` — Luật Giao dịch điện tử 2023 Đ.18 | `cross_encoder_threshold` | Semantic score 0.028224 < threshold 0.200000. |

## 4. Serious benchmark integrity finding

Luật An ninh mạng 2025 Điều 13 khoản 3 điểm h xuất hiện làm expected provision
trong chín testcase. Nội dung trong corpus là:

> Mạo danh, giả mạo thông tin, hình ảnh, giọng nói của cá nhân, gây ảnh hưởng
> đến uy tín, danh dự, nhân phẩm của cá nhân.

Ba testcase deepfake có quan hệ ngữ nghĩa trực tiếp với provision này. Sáu
testcase sau không mô tả mạo danh/giả mạo danh tính, hình ảnh hoặc giọng nói:

- `cyber_attack_medium_001`
- `cyber_attack_hard_001`
- `sql_injection_easy_001`
- `sql_injection_medium_001`
- `sql_injection_hard_001`
- `malware_hard_001`

Sáu label này chiếm 21.43% của 28 retrieval loss. Trong cùng corpus, nội dung về
xâm nhập/khai thác lỗ hổng nằm ở các provision khác, ví dụ Điều 7 khoản 5 và
Điều 18 khoản 1 điểm d.

Đây là dấu hiệu ground-truth semantic mismatch nghiêm trọng, không phải corpus
missing. Theo phạm vi phase này:

- không sửa benchmark;
- không tự chọn provision thay thế;
- không dùng sáu label này để tune embedding, BM25, expansion hoặc Cross Encoder;
- yêu cầu legal reviewer adjudicate trong một phase/commit benchmark riêng.

Mọi oracle bên dưới vẫn báo cả số liệu raw 44 label để không thay đổi benchmark.

## 5. Oracle replay

### 5.1 Định nghĩa

Mỗi oracle chỉ hoàn hảo hóa một stage và replay các stage downstream nguyên
trạng:

- **Oracle Domain:** bổ sung domain payload của expected document vào domain
  filter, sau đó chạy lại Dense/BM25/RRF/CE/gates/top-10.
- **Oracle Dense:** đặt một representative matching tốt nhất của mỗi expected
  provision được domain cho phép vào Dense prefetch, chạy RRF và downstream.
- **Oracle BM25:** tương tự trên BM25 prefetch.
- **Oracle Query Expansion:** bảo đảm expected candidate được domain cho phép
  có trong expanded branch, rồi chạy downstream.
- **Oracle Merge:** restore expected có trong raw prefetch nhưng bị RRF bỏ.
- **Oracle Behavior:** chỉ restore expected bị Behavior Gate loại.
- **Oracle Cross Encoder:** với expected đã sống qua Behavior Gate, đặt score
  tối đa rồi chạy threshold, dedup và balanced top-10.
- **Oracle Top-k:** ưu tiên expected đã sống tới dedup trước khi lấy top-10.

Đây là label-assisted upper bound, không phải thiết kế được phép dùng ground
truth ở production.

### 5.2 Downstream-constrained upper bounds

| Oracle | Macro Recall@10 | Micro hits | Gain so với 16 | Cases improved |
|---|---:|---:|---:|---:|
| Baseline | 41.67% | 16/44 | — | — |
| Domain | 55.00% | 22/44 | +6 | 4 |
| Dense | 46.67% | 19/44 | +3 | 3 |
| BM25 | 46.67% | 19/44 | +3 | 3 |
| Query Expansion | 45.00% | 18/44 | +2 | 2 |
| Hybrid Merge | 41.67% | 16/44 | 0 | 0 |
| Behavior Gate | 43.33% | 17/44 | +1 | 1 |
| Cross Encoder ranking + threshold | **60.00%** | **24/44** | **+8** | **8** |
| Top-k | 48.33% | 19/44 | +3 | 3 |

Oracle Domain chỉ cứu 6/9 domain loss vì ba provision mới được mở domain vẫn
bị candidate generation/ranking downstream loại.

Oracle Cross Encoder cứu cả năm threshold loss và ba top-k loss. Nó không cứu
candidate chưa vào union hoặc candidate bị Behavior Gate loại.

Combined Oracle Domain + Cross Encoder đạt:

```text
Macro Recall@10 = 78.33%
Micro hits = 32/44
```

Nếu thêm Oracle Behavior, kết quả là:

```text
Macro Recall@10 = 80.00%
Micro hits = 33/44
```

Combined oracle chứng minh mục tiêu 24–28 provision nằm dưới measured upper
bound. Nó không chứng minh một thay đổi production cụ thể sẽ đạt mức đó.

## 6. Component analysis

### 6.1 Dense, BM25, Hybrid và union

Metric dưới đây đo candidate trước downstream gates, với domain filter thật.

| Component | Macro provision recall | Micro hits | Macro precision proxy | Avg rank của hit | Macro MRR | Avg candidates |
|---|---:|---:|---:|---:|---:|---:|
| Dense original top-20 | 55.00% | 21 | 9.11% | 2.48 | 0.4395 | 20.00 |
| BM25 original top-20 | 51.67% | 21 | 8.63% | 5.52 | 0.2487 | 20.00 |
| RRF Hybrid original top-20 | 60.00% | 24 | 8.91% | 4.13 | 0.4082 | 20.00 |
| Original + expanded candidate union | 61.67% | 25 | 7.04% | 5.16 | 0.4093 | 27.53 |

Dense và BM25 cùng có 21 provision hit nhưng không phải cùng một tập. RRF tăng
lên 24, chứng minh hybrid merge đang đóng góp recall. Không có expected
provision nào bị RRF top-20 loại sau khi đã có trong raw prefetch.

### 6.2 Candidate-generation misses

Trong 10 provision không vào prefetch top-20:

| Diagnostic top-200 status | Count |
|---|---:|
| Dense rank 21–200, BM25 ngoài top-200 | 5 |
| Dense và BM25 đều rank 21–200 | 1 |
| Cả Dense và BM25 ngoài top-200 | 4 |

Sáu provision có thể được nhìn thấy bằng pool rất rộng; bốn provision không
được Dense/BM25 tìm trong top-200. Tuy nhiên sáu trong tám loss lặp của Điều 13
khoản 3 điểm h là ground-truth mismatch đã nêu, nên không được dùng làm lý do
tăng prefetch.

### 6.3 Query Expansion

| Measurement | Result |
|---|---:|
| Cases có expansion | 20/30 |
| Candidate thêm vào union | 226 |
| Candidate thêm trung bình | 7.53/query |
| Expected provision thêm vào union | 1 |
| Expected provision được cứu vào top-10 | 1 |
| Expected provision bị expansion drift khỏi top-10 | 0 |
| Macro Recall@10 original-only | 40.00% |
| Macro Recall@10 có expansion | 41.67% |
| Union precision trước expansion | 8.91% |
| Union precision sau expansion | 7.04% |

Provision duy nhất được cứu là Nghị Định 15/2020 Điều 81 trong
`sql_injection_medium_001`.

Expansion hiện không gây expected loss, nhưng hiệu suất thấp: 226 candidate mới
để cứu một provision. Không nên mở rộng rule hàng loạt mà không đo expansion
drift và candidate precision.

### 6.4 Cross Encoder

Trong 25 expected provision đã vào candidate union:

| Rank movement sau Cross Encoder | Count |
|---|---:|
| Tăng hạng | 6 |
| Giảm hạng | 10 |
| Không đổi | 9 |
| Average rank delta | +1.04 rank, tức trung bình xấu đi |
| Bị threshold loại hoàn toàn | 5 |

Ba expected provision sống qua threshold nhưng bị top-10 loại có CE/dedup rank:

- `personal_data_easy_001`, Điều 28 khoản 1: rank 18;
- `malware_hard_001`, Nghị Định 53 Điều 11: rank 11;
- `ai_copyright_hard_001`, VBHN Điều 35: rank 20.

Parameter replay:

| Prefetch | CE ratio | Macro R@10 | Micro hits | Precision proxy | Avg top-10 count |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.20 baseline | 41.67% | 16 | 14.81% | 6.93 |
| 20 | 0.10 | 45.00% | 17 | 14.28% | 7.97 |
| 20 | 0.00 | 50.00% | 19 | 14.04% | 8.70 |
| 40 | 0.20 | 43.33% | 17 | 14.65% | 7.90 |
| 40 | 0.10 | 46.67% | 18 | 13.72% | 8.67 |
| 40 | 0.00 | 48.33% | 19 | 14.10% | 9.10 |

Kết luận:

- fixed threshold relaxation chỉ cứu tối đa ba provision;
- tăng prefetch 20 → 40 ở threshold hiện tại chỉ cứu một provision;
- kết hợp pool rộng và threshold 0 không vượt pool 20/threshold 0;
- Oracle CE đạt 24 vì sửa **ranking**, không chỉ nới threshold.

Do đó không đề xuất “threshold = 0” hoặc “prefetch = 40” như thay đổi production.

### 6.5 Behavior Gate

Behavior benchmark có:

```text
114 expected behavior instances
34 unique expected behavior keys
22 keys có trong taxonomy
12 keys không có trong taxonomy
```

12 missing taxonomy keys:

```text
automated_electronic_contract
confirm_receipt
correct_input_error
deploy_malware
detect_or_remove_malware
receive_data_message
research_malware
secure_information_system
send_data_message
supply_defective_product
use_electronic_notice
use_unfair_contract_term
```

Trong 114 behavior instances:

- 42 được extractor lấy đúng;
- 72 bị bỏ sót;
- 15/72 do key chưa tồn tại trong taxonomy;
- 57/72 là supported key nhưng signal/extractor không nhận ra;
- extractor tạo 6 extra behavior instances.

Behavior Gate:

- kích hoạt ở 7/30 query;
- loại 191 candidate records;
- 189 removed records không match ground truth của case;
- hai expected provisions có ít nhất một matching record bị gate loại;
- một expected provision bị loại hoàn toàn khỏi retrieval:
  `deepfake_hard_001`, Luật Bảo vệ dữ liệu cá nhân Điều 28 khoản 1;
- 578 records không match ground truth vẫn còn sau gate trên toàn benchmark.

`False keep`/`false reject` ở đây là benchmark proxy. Ground truth không
exhaustive mọi căn cứ pháp lý có thể hữu ích, nên 578 không được diễn giải là
578 legal false positives.

### 6.6 Dedup và balanced top-k

- Dedup loại 28 records trong 14 query.
- Không expected provision nào bị mất hoàn toàn do dedup.
- Balanced top-k làm mất ba expected provision.
- Hybrid Merge làm mất 0 expected provision.

Không có bằng chứng để redesign RRF hoặc bỏ dedup trong phase tiếp theo.

## 7. Error clustering

| Cluster | Provision | % loss | Diễn giải |
|---|---:|---:|---|
| A — Candidate generation miss | 10 | 35.71% | Không vào raw Dense/BM25 top-20 ở cả original/expanded |
| B — Domain closed | 9 | 32.14% | Expected document bị Qdrant filter chặn |
| C — CE threshold false reject | 5 | 17.86% | Candidate đã vào union nhưng score dưới relative threshold |
| D — CE/top-k rank crowding | 3 | 10.71% | Candidate sống qua gates nhưng rank 11/16/20 |
| E — Behavior false reject | 1 | 3.57% | Candidate trực tiếp bị behavior score 0.04 loại |

Cluster A cần tách tiếp:

- sáu loss là suspected benchmark mismatch của Điều 13 khoản 3 điểm h;
- bốn loss còn lại cần candidate-generation investigation thật;
- không được quy toàn bộ 10 loss cho embedding.

Cluster B có bảy case:

- `deepfake_medium_001`: 2 provisions;
- `personal_data_hard_001`: 1;
- `sql_injection_hard_001`: 1;
- `advertising_hard_001`: 1;
- `consumer_medium_001`: 2;
- `consumer_hard_001`: 1;
- `network_security_medium_001`: 1.

## 8. Ranking forensics — all 44 expected provisions

Notation:

- `r/s`: rank/score;
- `Dense O`, `BM25 O`, `Hybrid O`: original query;
- `Hybrid E`: expanded query;
- `CE r/raw`: rank sau rerank/raw sigmoid CE score;
- `—`: stage không nhìn thấy provision hoặc stage không chạy cho provision đó.

| # | Case / expected provision | First failed | Dense O r/s | BM25 O r/s | Hybrid O r/s | Hybrid E r/s | CE r/raw | Behavior | Final score | Post-threshold | Top-10 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `deepfake_easy_001` — ANM Đ.7 K.2 P.g | retrieved | 1/0.4311 | 10/23.7036 | 1/0.5909 | 7/0.2500 | 2/0.2353 | 0.6300 | 0.4692 | 1 | 1 |
| 2 | `deepfake_easy_001` — ANM Đ.13 K.3 P.h | candidate generation | 27/0.2956 | — | — | — | — | — | — | — | — |
| 3 | `deepfake_medium_001` — ANM Đ.7 K.2 P.g | domain | — | — | — | — | — | — | — | — | — |
| 4 | `deepfake_medium_001` — ANM Đ.13 K.3 P.h | domain | — | — | — | — | — | — | — | — | — |
| 5 | `deepfake_hard_001` — BVDLCN Đ.28 K.1 | behavior | 112/0.2353 | 3/37.0097 | 5/0.2500 | 2/0.6111 | 1/0.0482 | 0.0400 | 0.7098 | — | — |
| 6 | `deepfake_hard_001` — ANM Đ.13 K.3 P.h | candidate generation | — | — | — | — | — | — | — | — | — |
| 7 | `personal_data_easy_001` — BVDLCN Đ.28 K.1 | top-k | 4/0.4801 | 29/20.5060 | 7/0.2000 | 1/0.6429 | 18/0.0421 | 0.2500 | 0.1621 | 18 | — |
| 8 | `personal_data_medium_001` — BVDLCN Đ.20 | retrieved | 7/0.3568 | 9/37.6723 | 9/0.1726 | — | 1/0.2522 | 0.0500 | 0.7151 | 1 | 1 |
| 9 | `personal_data_hard_001` — BVDLCN Đ.2 | domain | — | — | — | — | — | — | — | — | — |
| 10 | `cyber_attack_easy_001` — NĐ15 Đ.80 | retrieved | 1/0.5849 | 4/25.0889 | 1/0.6111 | 2/0.5000 | 1/0.0358 | 0.1000 | 0.7480 | 1 | 2 |
| 11 | `cyber_attack_medium_001` — NĐ15 Đ.81 | CE threshold | 3/0.3944 | 56/16.6447 | 6/0.2500 | 12/0.1111 | 20/0.0023 | 0 | 0.0693 | — | — |
| 12 | `cyber_attack_medium_001` — ANM Đ.13 K.3 P.h | candidate generation | — | — | — | — | — | — | — | — | — |
| 13 | `cyber_attack_hard_001` — ANM Đ.13 K.3 P.h | candidate generation | — | — | — | — | — | — | — | — | — |
| 14 | `cyber_attack_hard_001` — NĐ15 Đ.81 | candidate generation | 33/0.5107 | 168/18.4008 | — | — | — | — | — | — | — |
| 15 | `sql_injection_easy_001` — ANM Đ.13 K.3 P.h | candidate generation | — | — | — | — | — | — | — | — | — |
| 16 | `sql_injection_medium_001` — ANM Đ.13 K.3 P.h | candidate generation | — | — | — | — | — | — | — | — | — |
| 17 | `sql_injection_medium_001` — NĐ15 Đ.81 | retrieved | 26/0.2589 | — | — | 12/0.1111 | 8/0.0025 | 0.3130 | 0.1772 | 5 | 5 |
| 18 | `sql_injection_hard_001` — ANM Đ.13 K.3 P.h | candidate generation | — | — | — | — | — | — | — | — | — |
| 19 | `sql_injection_hard_001` — BVDLCN Đ.20 | domain | — | — | — | — | — | — | — | — | — |
| 20 | `malware_easy_001` — NĐ53 Đ.10 | retrieved | 1/0.4562 | 1/48.7670 | 1/1.0000 | — | 1/0.3351 | 0.6500 | 0.9020 | 1 | 2 |
| 21 | `malware_medium_001` — ANM Đ.36 | CE threshold | 1/0.4752 | 4/42.0607 | 1/0.7000 | 12/0.1250 | 7/0.0210 | 0 | 0.1957 | — | — |
| 22 | `malware_hard_001` — ANM Đ.13 K.3 P.h | candidate generation | — | — | — | — | — | — | — | — | — |
| 23 | `malware_hard_001` — NĐ53 Đ.11 | top-k | 36/0.3045 | 6/42.6067 | 11/0.1429 | — | 11/0.0028 | 0.6500 | 0.4306 | 11 | — |
| 24 | `ai_copyright_easy_001` — VBHN Đ.20 | retrieved | 6/0.4182 | 45/31.2305 | 12/0.1429 | — | 7/0.0255 | 0.6500 | 0.3859 | 7 | 7 |
| 25 | `ai_copyright_medium_001` — VBHN Đ.25 | retrieved | 3/0.4910 | 4/43.2396 | 3/0.4000 | 2/0.5000 | 1/0.9706 | 0.6500 | 0.8835 | 1 | 2 |
| 26 | `ai_copyright_hard_001` — VBHN Đ.35 | top-k | 3/0.4117 | 16/36.0964 | 5/0.3088 | — | 20/0.0023 | 0.6500 | 0.2553 | 20 | — |
| 27 | `ai_copyright_hard_001` — NĐ17 Đ.28 | candidate generation | — | — | — | — | — | — | — | — | — |
| 28 | `advertising_easy_001` — BVDLCN Đ.28 K.1 | retrieved | 1/0.5642 | 3/34.0102 | 2/0.7500 | 2/0.6250 | 2/0.8582 | 0.9000 | 0.8919 | 2 | 2 |
| 29 | `advertising_medium_001` — BVQLNTD Đ.10 | retrieved | 1/0.4409 | 4/30.3923 | 1/0.7000 | 6/0.2381 | 1/0.8778 | 0 | 0.7033 | 1 | 1 |
| 30 | `advertising_hard_001` — BVDLCN Đ.28 K.1 | domain | — | — | — | — | — | — | — | — | — |
| 31 | `advertising_hard_001` — BVQLNTD Đ.10 | CE threshold | 8/0.3472 | 16/35.8796 | 12/0.1611 | 2/0.5000 | 20/0.0277 | 0 | 0.0830 | — | — |
| 32 | `consumer_easy_001` — BVQLNTD Đ.10 | retrieved | 1/0.4684 | 5/34.2988 | 1/0.6429 | — | 1/0.6848 | 0 | 1.0000 | 1 | 1 |
| 33 | `consumer_medium_001` — BVQLNTD Đ.34 | domain | — | — | — | — | — | — | — | — | — |
| 34 | `consumer_medium_001` — BVQLNTD Đ.35 | domain | — | — | — | — | — | — | — | — | — |
| 35 | `consumer_hard_001` — BVQLNTD Đ.25 | domain | — | — | — | — | — | — | — | — | — |
| 36 | `network_security_easy_001` — NĐ53 Đ.10 | retrieved | 1/0.5032 | 1/53.8171 | 1/1.0000 | — | 1/0.9970 | 0 | 1.0000 | 1 | 2 |
| 37 | `network_security_medium_001` — NĐ53 Đ.11 | domain | — | — | — | — | — | — | — | — | — |
| 38 | `network_security_hard_001` — ANM Đ.36 | CE threshold | 2/0.3381 | 11/31.6911 | 3/0.4103 | — | 10/0.0023 | 0 | 0.1006 | — | — |
| 39 | `electronic_transactions_easy_001` — GDĐT Đ.38 | retrieved | 1/0.6438 | 1/50.4931 | 1/1.0000 | 1/1.0000 | 1/0.9999 | 0 | 1.0000 | 1 | 1 |
| 40 | `electronic_transactions_medium_001` — GDĐT Đ.18 | CE threshold | 29/0.3453 | 6/34.6941 | 9/0.1429 | 16/0.1000 | 10/0.0287 | 0 | 0.1076 | — | — |
| 41 | `electronic_transactions_medium_001` — GDĐT Đ.14 | retrieved | 1/0.6048 | 1/60.9299 | 1/1.0000 | 1/1.0000 | 1/0.9908 | 0 | 1.0000 | 1 | 1 |
| 42 | `electronic_transactions_hard_001` — GDĐT Đ.15 | retrieved | 1/0.4409 | 8/45.0418 | 2/0.6111 | 7/0.2381 | 5/0.0720 | 0 | 0.3526 | 5 | 5 |
| 43 | `electronic_transactions_hard_001` — GDĐT Đ.16 | retrieved | 3/0.3904 | 1/50.3795 | 1/0.6111 | — | 3/0.0833 | 0 | 0.4040 | 3 | 3 |
| 44 | `electronic_transactions_hard_001` — GDĐT Đ.17 | retrieved | 2/0.4036 | 2/50.1153 | 3/0.5000 | 14/0.1167 | 2/0.1143 | 0 | 0.5022 | 2 | 2 |

Abbreviations:

- ANM: Luật An ninh mạng 2025
- BVDLCN: Luật Bảo vệ dữ liệu cá nhân 2025
- BVQLNTD: Luật Bảo vệ quyền lợi người tiêu dùng 2023
- GDĐT: Luật Giao dịch điện tử 2023
- NĐ15/NĐ17/NĐ53: các nghị định tương ứng
- VBHN: `2023_361 + 362_11-VBHN-VPQH`

### 8.1 Rank transition của 25 provision đã vào candidate union

Bảng 44 provision phía trên cung cấp raw rank/score và first failed stage.
Bảng dưới tập trung vào sự dịch chuyển thứ hạng qua từng phép biến đổi
downstream. `—` nghĩa là provision đã bị loại hoặc không có expanded branch.

| Case / provision | Hybrid O | Hybrid E | Union | CE | Post behavior | Post threshold | Post dedup | Top-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `deepfake_easy_001` — ANM Đ.7 K.2 P.g | 1 | 7 | 1 | 2 | 1 | 1 | 1 | 1 |
| `deepfake_hard_001` — BVDLCN Đ.28 K.1 | 5 | 2 | 5 | 1 | — | — | — | — |
| `personal_data_easy_001` — BVDLCN Đ.28 K.1 | 7 | 1 | 7 | 18 | 18 | 18 | 16 | — |
| `personal_data_medium_001` — BVDLCN Đ.20 | 9 | — | 9 | 1 | 1 | 1 | 1 | 1 |
| `cyber_attack_easy_001` — NĐ15 Đ.80 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 2 |
| `cyber_attack_medium_001` — NĐ15 Đ.81 | 6 | 12 | 6 | 20 | 20 | — | — | — |
| `sql_injection_medium_001` — NĐ15 Đ.81 | — | 12 | 30 | 8 | 5 | 5 | 5 | 5 |
| `malware_easy_001` — NĐ53 Đ.10 | 1 | — | 1 | 1 | 1 | 1 | 1 | 2 |
| `malware_medium_001` — ANM Đ.36 | 1 | 12 | 1 | 7 | 7 | — | — | — |
| `malware_hard_001` — NĐ53 Đ.11 | 11 | — | 11 | 11 | 11 | 11 | 11 | — |
| `ai_copyright_easy_001` — VBHN Đ.20 | 12 | — | 12 | 7 | 7 | 7 | 7 | 7 |
| `ai_copyright_medium_001` — VBHN Đ.25 | 3 | 2 | 3 | 1 | 1 | 1 | 1 | 2 |
| `ai_copyright_hard_001` — VBHN Đ.35 | 5 | — | 5 | 20 | 20 | 20 | 16 | — |
| `advertising_easy_001` — BVDLCN Đ.28 K.1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| `advertising_medium_001` — BVQLNTD Đ.10 | 1 | 6 | 1 | 1 | 1 | 1 | 1 | 1 |
| `advertising_hard_001` — BVQLNTD Đ.10 | 12 | 2 | 12 | 20 | 20 | — | — | — |
| `consumer_easy_001` — BVQLNTD Đ.10 | 1 | — | 1 | 1 | 1 | 1 | 1 | 1 |
| `network_security_easy_001` — NĐ53 Đ.10 | 1 | — | 1 | 1 | 1 | 1 | 1 | 2 |
| `network_security_hard_001` — ANM Đ.36 | 3 | — | 3 | 10 | 10 | — | — | — |
| `electronic_transactions_easy_001` — GDĐT Đ.38 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `electronic_transactions_medium_001` — GDĐT Đ.18 | 9 | 16 | 9 | 10 | 10 | — | — | — |
| `electronic_transactions_medium_001` — GDĐT Đ.14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `electronic_transactions_hard_001` — GDĐT Đ.15 | 2 | 7 | 2 | 5 | 5 | 5 | 5 | 5 |
| `electronic_transactions_hard_001` — GDĐT Đ.16 | 1 | — | 1 | 3 | 3 | 3 | 3 | 3 |
| `electronic_transactions_hard_001` — GDĐT Đ.17 | 3 | 14 | 3 | 2 | 2 | 2 | 2 | 2 |

## 9. Top five improvement opportunities

### 1. Cross-Encoder ranking quality, không phải chỉ threshold

Evidence:

- candidate union đã có 25 expected;
- final top-10 chỉ có 16;
- CE làm 10/25 expected giảm hạng;
- CE oracle đạt 24/44 và macro Recall@10 60%;
- threshold 0 chỉ đạt 19/44.

Opportunity:

- benchmark các CE model/input representation trên frozen candidate pools;
- đo rank delta, Recall@10, precision và latency;
- chỉ sau khi chọn được ranking tốt hơn mới tune threshold.

Không triển khai ngay:

- fixed threshold 0;
- label-aware boost;
- bỏ Cross Encoder.

### 2. Domain coverage recovery

Evidence:

- 9 provision bị chặn trước search;
- exact Domain oracle chạy xuyên downstream cứu 6, đạt 22/44;
- lỗi tập trung ở bảy case và các signal/fallback cụ thể.

Opportunity:

- bổ sung signal theo query wording thực tế;
- sửa fallback để không loại `consumer_protection` trong query không match;
- kiểm tra multi-domain coverage trước khi tăng `max_domains`.

Guardrail bắt buộc: Wrong Domain Rate không vượt 19.56%.

### 3. Dense candidate generation cho legitimate misses

Evidence:

- Dense original có MRR 0.4395, cao hơn BM25 0.2487;
- sáu candidate-generation misses xuất hiện ở Dense rank 21–200;
- prefetch 40 chỉ cứu một provision;
- bốn candidate misses ngoài Dense top-200.

Opportunity:

- sau legal adjudication, lập tập legitimate misses;
- so sánh embedding hiện tại với candidate model khác trên cùng corpus snapshot;
- đánh giá Recall@20/40, MRR và query latency độc lập;
- không tune theo sáu suspected wrong labels.

### 4. Targeted BM25 / Query Expansion recovery

Evidence:

- BM25 tìm 21 provision nhưng average hit rank 5.52;
- Hybrid tăng từ 21 lên 24, nên BM25 vẫn có contribution;
- expansion cứu một provision, không gây expected drift, nhưng thêm 226 candidates;
- một số legitimate misses dùng wording khác xa provision.

Opportunity:

- đánh giá accent folding/token normalization và field-aware text offline;
- thêm expansion rule theo failure cluster cụ thể, một rule/experiment;
- báo expansion-only expected gain và noise cho từng rule.

Không triển khai expansion rộng hoặc append toàn bộ thuật ngữ domain.

### 5. Behavior taxonomy/extractor recovery

Evidence:

- 12 missing taxonomy keys;
- 57 supported behavior instances bị extractor bỏ sót;
- một expected provision bị Behavior Gate false reject;
- Behavior oracle chỉ tăng trực tiếp một provision nhưng Behavior Card còn đi
  vào CE query và final score, nên ảnh hưởng ranking gián tiếp có thể lớn hơn.

Opportunity:

- tách taxonomy coverage và signal extraction thành hai benchmark riêng;
- giữ conservative activation;
- không hạ behavior minimum toàn cục trước khi đo false keep.

## 10. Expected gain

Không cộng cơ học các oracle độc lập. Measured bounds:

| Scope | Macro R@10 | Micro hits |
|---|---:|---:|
| Baseline | 41.67% | 16 |
| Domain oracle | 55.00% | 22 |
| CE oracle | 60.00% | 24 |
| Domain + CE oracle | 78.33% | 32 |
| Domain + CE + Behavior oracle | 80.00% | 33 |

Target 24–28 provision là khả thi dưới measured upper bound. Planning target
bảo thủ:

- Domain phase: tìm cách capture một phần của +6 constrained gain;
- CE phase: target 20–24 total hits, không kỳ vọng lấy đủ oracle +8 ngay;
- candidate-generation phase: +1–3 legitimate hits, vì pool widening/query
  expansion replay chỉ chứng minh mức này;
- Behavior phase: ít nhất không còn expected false reject và Behavior Recall
  tăng độc lập.

Không dự báo Citation Accuracy tăng tuyến tính theo Retrieval Recall.
Applicability và Generation vẫn là hai cổng downstream.

## 11. Risk analysis

### Benchmark integrity

Rủi ro lớn nhất là tối ưu retriever để đưa một provision không điều chỉnh hành
vi lên top-10. Sáu suspected labels phải được legal reviewer adjudicate trước
khi dùng làm acceptance target.

### Domain

Thêm signal/domain có thể tăng candidate recall nhưng tăng Wrong Domain và pool
noise. Oracle Domain dùng ground truth nên là upper bound, không phản ánh
precision của một selector production.

### Cross Encoder

- đổi model/input có thể tăng latency;
- relative score scale có thể thay đổi;
- threshold relaxation tăng context count và có thể làm Citation Accuracy giảm;
- benchmark 30 case dễ overfit.

### Dense/BM25/Expansion

- pool rộng làm CE chậm hơn;
- expansion hiện thêm 226 candidate để cứu một provision;
- BM25 normalization thay đổi toàn corpus score distribution và có thể cần
  re-ingest sparse vectors;
- embedding model mới cần rebuild dense vectors.

### Behavior

Taxonomy rộng hơn có thể làm gate kích hoạt sai. False keep/false reject theo
benchmark chỉ là proxy vì legal ground truth không exhaustive.

## 12. Rollback contract

Mỗi implementation phase tương lai phải:

- nằm trong commit/feature flag độc lập;
- không kèm thay đổi benchmark;
- giữ corpus snapshot/index riêng khi model/vector thay đổi;
- có cấu hình quay lại baseline;
- không migration destructive;
- rollback ngay nếu bất kỳ guardrail nào fail.

Rollback unit:

| Phase tương lai | Rollback |
|---|---|
| Domain signals/fallback | Revert domain-only commit/config |
| CE model/input/threshold | Restore pinned model/revision/ratio 0.20 |
| Dense model | Switch về existing embedding index/model |
| BM25 | Restore existing sparse index and tokenizer |
| Query Expansion | Disable đúng rule mới |
| Behavior taxonomy/extractor | Revert taxonomy hoặc signal commit riêng |

## 13. Benchmark guardrails

Mọi phase phải chạy targeted subset rồi full 30-case benchmark, cùng hash,
model/index/mode trừ đúng biến đang thí nghiệm.

Hard guardrails:

```text
Citation Accuracy >= 21.67%
Macro Recall@10 >= 41.67%
Micro expected hits >= 16/44
Wrong Domain Rate <= 19.56%
Hallucinated Citation = 0
Applicability config = control/unlimited
Applicability, Recursive, Generation prompt/code unchanged
Public API và output format unchanged
```

Stage-specific metrics bắt buộc:

- Dense/BM25/Hybrid Recall@20, precision, MRR, average expected rank;
- candidate-union expected hits và pool size;
- CE expected rank movement;
- Behavior false reject/keep proxy;
- final top-10 precision;
- retrieval latency và total latency.

Không promote nếu chỉ có oracle/replay tăng. Phải có full end-to-end benchmark
thật đạt guardrail.

## 14. Implementation roadmap — proposal only

### Phase 0 — Ground-truth adjudication

Phạm vi:

- legal reviewer kiểm tra sáu Điều 13 khoản 3 điểm h labels;
- ghi quyết định và nguồn pháp lý;
- nếu benchmark cần sửa, thực hiện trong commit benchmark-only riêng.

Expected gain:

- không làm metric “đẹp” hơn một cách nhân tạo;
- làm denominator và retrieval target hợp lệ.

Risk:

- metric lịch sử không còn so sánh trực tiếp nếu benchmark hash đổi.

Rollback:

- giữ artifact/hash cũ; không rewrite historical results.

### Phase 1 — Domain Recall Recovery

Phạm vi:

- chỉ sửa Domain signals/fallback;
- target bảy case có domain loss;
- không đổi Dense/BM25/CE/Behavior.

Measured upper bound:

- 22/44, macro Recall@10 55%.

Acceptance:

- domain-lost provisions giảm;
- Wrong Domain không tăng;
- Citation Accuracy không giảm.

### Phase 2 — Cross-Encoder Ranking Recovery

Phạm vi:

- frozen candidate pools trước CE;
- benchmark model/input representation trước;
- threshold tuning là experiment sau, không gộp.

Measured upper bound:

- 24/44, macro Recall@10 60%.

Acceptance:

- expected provisions giảm hạng không quá baseline 10;
- threshold false reject giảm;
- top-10 precision và Citation Accuracy không giảm;
- latency được báo riêng.

### Phase 3 — Candidate Generation Recovery

Phạm vi:

Ba experiment độc lập:

1. Dense embedding/model;
2. BM25 normalization/weighting;
3. targeted Query Expansion.

Không gộp model, sparse index và expansion trong một run.

Evidence target:

- legitimate candidate-generation misses sau Phase 0;
- prefetch 40 không được coi là default solution vì chỉ cứu một provision.

Acceptance:

- candidate union expected hits tăng;
- pool noise/latency có giới hạn;
- downstream Recall@10 và Citation Accuracy đạt guardrail.

### Phase 4 — Behavior Recall Recovery

Phạm vi:

1. thêm 12 missing taxonomy keys;
2. benchmark;
3. sau đó mới sửa signal cho 57 supported misses;
4. benchmark riêng lần nữa.

Acceptance:

- Behavior Recall tăng;
- expected Behavior Gate false reject bằng 0;
- Wrong Behavior và retrieval precision không xấu đi;
- không thay Applicability.

### Phase 5 — Combined promotion

Chỉ bắt đầu khi từng phase độc lập đã pass.

- chạy A/B theo từng feature flag;
- promote tổ hợp nhỏ nhất đạt 24–28 micro hits;
- không giả định gain cộng tuyến tính;
- full end-to-end benchmark quyết định cuối cùng.

## 15. Final decision

Không triển khai Retrieval change trong phase audit này.

Các kết luận được benchmark chứng minh:

- RRF Hybrid đang có contribution và không làm mất expected;
- Query Expansion có gain nhỏ, không có measured expected drift;
- duplicate collapse không phải bottleneck;
- tăng prefetch đơn thuần không đủ;
- hạ CE threshold đơn thuần không đủ;
- Domain và CE ranking có upper bound lớn nhất;
- candidate-generation loss đang bị pha trộn với benchmark-label error.

Bước tiếp theo bắt buộc là Phase 0 legal adjudication. Chỉ sau đó mới được chọn
Phase 1 implementation hypothesis.
