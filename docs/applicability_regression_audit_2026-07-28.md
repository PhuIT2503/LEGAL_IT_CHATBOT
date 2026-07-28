# Applicability regression audit — 2026-07-28

## Kết luận

Không tìm thấy `True regression` trên bất kỳ provision nào đã được legal
adjudication là **Confirmed correct**. Vì vậy audit này **không sửa production
code**, không chạy benchmark promotion mới và không rollback long-scenario fix.

Tín hiệu `Applicability Accuracy 53,85% → 50,63%` là có thật theo scoring
contract hiện hành, nhưng không đồng nghĩa hệ thống làm mất căn cứ đúng:

- 15 confirmed provisions đã vào Applicability ở cả hai run;
- số confirmed provision được giữ tăng từ `11/15` lên `15/15`;
- bốn false negative cũ được phục hồi: Luật Giao dịch điện tử 2023 Điều 38,
  Điều 15, Điều 16 và Điều 17;
- không có confirmed provision nào chuyển từ giữ sang loại;
- Citation Accuracy toàn benchmark giữ nguyên `30,00%`;
- Recall@10 giữ nguyên `41,67%`;
- Hallucinated Citation giữ nguyên `0`.

Metric giảm vì evaluator coi mọi article không nằm trong expected list là
negative tuyệt đối. Candidate-level confusion matrix chuyển từ
`TP=12, FN=4, TN=80, FP=74` sang `TP=16, FN=0, TN=72, FP=83`. Bốn false
negative được sửa, nhưng chín candidate không được liệt kê được giữ thêm; một
candidate negative mới được đưa vào pool và bị loại đúng. Tổng số decision đúng
theo micro average vì thế giảm `92/170 → 88/171`, còn macro average theo
testcase giảm `53,85% → 50,63%`.

## 1. Phạm vi và bằng chứng

Hai artifact dùng cùng benchmark SHA-256
`126438a83f9902d950b140f74fd1f527b899ec9dc160ce1163f47c0c5d3e114c`,
cùng commit `198958af...`, cùng `critic/top_k=10`, embedding model và
`qwen2.5:7b`.

| Run | Artifact | SHA-256 details |
|---|---|---|
| Before | `benchmark_30_final_output_ce_priority_20260727` | `5a1e66d7...d752df` |
| After | `benchmark_30_long_fact_preservation_20260727` | `2975c57e...54899f` |

Paired diff dùng đúng scoring semantics trong `evaluation.metrics`:

- identity là normalized `document + article`;
- `KEEP` và `WEAK_KEEP` đều là kept;
- expected clause/point được hạ về article-level;
- mỗi article chỉ tính một lần, decision đầu tiên được giữ;
- row được xuất khi decision, confidence, final-context membership, citation
  membership hoặc evaluator correctness thay đổi.

Artifact chi tiết:

- `evaluation/results/applicability_paired_diff_20260728.json`;
- `evaluation/results/applicability_confirmed_only_20260728.json`.

## 2. Root cause

### 2.1 Metric đang trộn positive-label recall với unadjudicated negatives

`applicability_accuracy()` là binary accuracy trên toàn candidate pool. Một
candidate chỉ được xem là positive nếu document/article có trong expected list;
mọi article khác tự động là negative, dù benchmark không hề adjudicate article
đó.

Đây là nguyên nhân trực tiếp của nghịch lý:

| Candidate micro | Before | After | Delta |
|---|---:|---:|---:|
| Candidate | 170 | 171 | +1 |
| True positive | 12 | 16 | +4 |
| False negative | 4 | 0 | -4 |
| True negative | 80 | 72 | -8 |
| False positive | 74 | 83 | +9 |
| Accuracy | 54,12% | 51,46% | -2,66 điểm |
| Exact-label precision | 13,95% | 16,16% | +2,21 điểm |
| Exact-label recall | 75,00% | 100,00% | +25 điểm |

Macro metric công bố là trung bình 30 testcase nên có giá trị
`53,85% → 50,63%`, hơi khác micro accuracy ở trên.

Mười changed rows có source text cho thấy nhãn negative tuyệt đối là đáng nghi,
ví dụ:

- Luật Bảo vệ quyền lợi người tiêu dùng 2023 Điều 18 yêu cầu sự đồng ý khi dùng
  thông tin người tiêu dùng;
- Nghị định 15/2020 Điều 80 trực tiếp nói tới truy cập trái phép và thu thập
  thông tin;
- Nghị định 17/2023 Điều 66 trực tiếp mô tả xâm phạm quyền sao chép và dẫn chiếu
  ngoại lệ Điều 25;
- Luật Giao dịch điện tử 2023 Điều 37 trực tiếp dẫn chiếu Điều 15–18 về nhận,
  gửi, thời điểm và địa điểm.

Audit chỉ gắn nhóm `C. Benchmark label questionable`; không thay expected list
và không tự biến các row đó thành ground truth mới.

### 2.2 Applicability prompt bị perturb trên toàn bộ 30 case

Long-scenario change thêm `FACT_STATE` vào Applicability prompt. Fact state luôn
có `normalized_question` và `question_sections`, kể cả query ngắn không có
stated fact chuyên biệt. Vì vậy model input của cả 30 case đều thay đổi, không
chỉ long scenario.

Benchmark runner dùng Qwen với `temperature=0.2`. Một paired replay duy nhất
không đủ tách:

1. ảnh hưởng semantic thật của fact card;
2. prompt-position sensitivity;
3. sampling variance.

Trace không lưu nguyên `model_input` và `raw_model_output`; nó chỉ lưu parsed
decision. Vì không có confirmed regression, audit không thêm observability code
hoặc LLM replay.

### 2.3 Required-element guard không gây metric regression

After run có 171 decisions:

- 170 `PARTIAL_MATCH`;
- 1 `MATCH`;
- 0 `NOT_APPLICABLE`.

Không có deterministic `NOT_APPLICABLE` override nào được kích hoạt trong
benchmark 30 case. Candidate `MATCH` duy nhất là candidate mới trong
`deepfake_medium_001` và model vẫn quyết định `REMOVE`. Do đó contract guard,
required-element coverage và parser `PARTIAL_MATCH` không thể là nguyên nhân
làm mất confirmed provision.

### 2.4 Behavior alias chỉ giải thích một candidate pool delta

Before có 170 decisions, after có 171. Candidate mới là Nghị định 52/2013
Điều 35 ở `deepfake_medium_001`; nó xuất hiện sau khi alias “giả giọng” làm
Behavior Recall tăng `30,85% → 31,68%`. Candidate này bị `REMOVE/LOW`, không vào
context hoặc citation. Đây là upstream pool difference, không phải paired
Applicability regression.

## 3. Metric theo ground-truth quality

Official accuracy không thể tách sạch theo quality vì 126+ candidate negative
không có legal adjudication. Metric đúng để trả lời “confirmed label có bị mất
không?” là **positive-label retention among provisions that reached
Applicability**.

| Ground-truth quality | Reached Applicability | Before kept | After kept | Before | After |
|---|---:|---:|---:|---:|---:|
| All expected labels | 16 | 12 | 16 | 75,00% | 100,00% |
| Confirmed correct | 15 | 11 | 15 | 73,33% | 100,00% |
| Likely incorrect | 1 | 1 | 1 | 100,00% | 100,00% |
| Ambiguous | 0 | 0 | 0 | N/A | N/A |

Diễn giải quan trọng:

- `Likely incorrect = 100%` không phải kết quả tốt về pháp lý; nó cho biết
  provision có nhãn yếu vẫn bị giữ ở cả hai run.
- Không có ambiguous provision nào đi tới Applicability, nên không thể suy ra
  accuracy cho nhóm này.
- Trong tổng 29 confirmed annotations, 14 không bao giờ tới Applicability. Đây
  vẫn là Retrieval gap, không phải Applicability regression.

Để đối chiếu với macro metric cũ, cohort testcase chồng lấn cho kết quả:

| Cohort chứa quality | Case | Before macro | After macro |
|---|---:|---:|---:|
| Confirmed | 24 | 56,31% | 52,41% |
| Likely incorrect | 6 | 47,95% | 45,63% |
| Ambiguous | 4 | 54,17% | 57,54% |

Cohort macro vẫn chấm unlisted candidates là negative và một testcase có thể
nằm trong nhiều cohort. Vì vậy không được dùng bảng này để kết luận confirmed
provision bị mất.

## 4. Paired decision diff

Có 48 changed rows trên union 171 candidates:

| Class | Số row | Ý nghĩa audit |
|---|---:|---|
| A. Correct improvement | 15 | 4 confirmed recoveries + 11 thay đổi đúng theo frozen evaluator |
| B. True regression | 0 | Không có row đủ bằng chứng |
| C. Benchmark label questionable | 10 | Source có direct/strong supporting fit nhưng expected list không liệt kê |
| D. Evaluator inconsistency | 5 | 1 pool mismatch; 4 citation changes dù keep/context không đổi |
| E. No downstream impact | 7 | Confidence/reason đổi, keep/context/citation không đổi |
| F. Insufficient evidence | 11 | Unadjudicated negative + một stochastic replay không đủ xác nhận |

`Ctx` và `Cit` dùng `0/1` cho article membership.

| # | Testcase | Candidate provision | Quality | Before → After | Ctx | Cit | Class |
|---:|---|---|---|---|---:|---:|:---:|
| 1 | `deepfake_medium_001` | NĐ 52/2013 Đ.35 | Unadj. negative | ∅ → REMOVE/LOW | 0→0 | 0→0 | D |
| 2 | `personal_data_easy_001` | Luật BVQLNTD 2023 Đ.18 | Unadj. negative | REMOVE/LOW → KEEP/HIGH | 0→1 | 0→0 | C |
| 3 | `personal_data_easy_001` | Luật CNTT 2006 Đ.70 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | F |
| 4 | `personal_data_easy_001` | NĐ 15/2020 Đ.94 | Unadj. negative | WEAK_KEEP/MEDIUM → KEEP/HIGH | 1→1 | 0→1 | C |
| 5 | `personal_data_easy_001` | NĐ 52/2013 Đ.4 | Unadj. negative | KEEP/HIGH → WEAK_KEEP/LOW | 1→1 | 1→0 | D |
| 6 | `personal_data_medium_001` | Luật BVDLCN 2025 Đ.20 | Confirmed | KEEP/HIGH → WEAK_KEEP/LOW | 1→1 | 1→1 | E |
| 7 | `personal_data_medium_001` | Luật BVDLCN 2025 Đ.28 | Unadj. negative | WEAK_KEEP/MEDIUM → WEAK_KEEP/LOW | 1→1 | 0→0 | E |
| 8 | `personal_data_hard_001` | Luật Dữ liệu 2024 Đ.14 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→1 | F |
| 9 | `cyber_attack_easy_001` | NĐ 15/2020 Đ.81 | Unadj. negative | KEEP/HIGH → WEAK_KEEP/MEDIUM | 1→1 | 0→0 | E |
| 10 | `cyber_attack_medium_001` | Luật ANM 2025 Đ.14 | Unadj. negative | WEAK_KEEP/MEDIUM → REMOVE/LOW | 1→0 | 0→0 | A |
| 11 | `sql_injection_easy_001` | Luật ANM 2025 Đ.2 | Unadj. negative | KEEP/HIGH → WEAK_KEEP/LOW | 1→1 | 1→1 | E |
| 12 | `sql_injection_hard_001` | Luật ANM 2025 Đ.7 | Unadj. negative | WEAK_KEEP/LOW → WEAK_KEEP/LOW | 1→1 | 1→0 | D |
| 13 | `sql_injection_hard_001` | NĐ 15/2020 Đ.80 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→1 | C |
| 14 | `malware_hard_001` | NĐ 53/2022 Đ.10 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/MEDIUM | 0→1 | 0→0 | C |
| 15 | `malware_hard_001` | NĐ 53/2022 Đ.4 | Unadj. negative | WEAK_KEEP/MEDIUM → REMOVE/LOW | 1→0 | 0→0 | A |
| 16 | `ai_copyright_easy_001` | VBHN Luật SHTT Đ.19 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | F |
| 17 | `ai_copyright_easy_001` | VBHN Luật SHTT Đ.4 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | F |
| 18 | `ai_copyright_easy_001` | NĐ 17/2023 Đ.14 | Unadj. negative | KEEP/HIGH → REMOVE/LOW | 1→0 | 1→0 | A |
| 19 | `ai_copyright_easy_001` | NĐ 17/2023 Đ.66 | Unadj. negative | KEEP/HIGH → WEAK_KEEP/LOW | 1→1 | 0→1 | D |
| 20 | `ai_copyright_medium_001` | VBHN Luật SHTT Đ.25 | Confirmed | KEEP/HIGH → WEAK_KEEP/LOW | 1→1 | 1→0 | D |
| 21 | `ai_copyright_medium_001` | VBHN Luật SHTT Đ.32 | Unadj. negative | KEEP/HIGH → REMOVE/LOW | 1→0 | 0→0 | A |
| 22 | `ai_copyright_medium_001` | NĐ 17/2023 Đ.29 | Unadj. negative | KEEP/HIGH → WEAK_KEEP/LOW | 1→1 | 0→0 | E |
| 23 | `ai_copyright_medium_001` | NĐ 17/2023 Đ.37 | Unadj. negative | KEEP/HIGH → REMOVE/LOW | 1→0 | 0→0 | A |
| 24 | `ai_copyright_medium_001` | NĐ 17/2023 Đ.66 | Unadj. negative | REMOVE/LOW → KEEP/HIGH | 0→1 | 0→1 | C |
| 25 | `ai_copyright_hard_001` | VBHN Luật SHTT Đ.4 | Unadj. negative | WEAK_KEEP/LOW → KEEP/HIGH | 1→1 | 0→0 | E |
| 26 | `ai_copyright_hard_001` | NĐ 17/2023 Đ.29 | Unadj. negative | KEEP/HIGH → REMOVE/LOW | 1→0 | 0→0 | A |
| 27 | `advertising_hard_001` | Luật BVQLNTD 2023 Đ.39 | Unadj. negative | WEAK_KEEP/LOW → REMOVE/LOW | 1→0 | 0→0 | A |
| 28 | `advertising_hard_001` | NĐ 147/2024 Đ.35 | Unadj. negative | WEAK_KEEP/LOW → REMOVE/LOW | 1→0 | 1→0 | A |
| 29 | `advertising_hard_001` | NĐ 147/2024 Đ.4 | Unadj. negative | WEAK_KEEP/LOW → REMOVE/LOW | 1→0 | 0→0 | A |
| 30 | `consumer_easy_001` | Luật BVQLNTD 2023 Đ.9 | Unadj. negative | WEAK_KEEP/LOW → REMOVE/LOW | 1→0 | 0→0 | A |
| 31 | `consumer_medium_001` | NĐ 52/2013 Đ.74 | Unadj. negative | KEEP/HIGH → REMOVE/LOW | 1→0 | 0→0 | A |
| 32 | `consumer_medium_001` | NĐ 53/2022 Đ.22 | Unadj. negative | REMOVE/LOW → KEEP/HIGH | 0→1 | 0→0 | F |
| 33 | `network_security_medium_001` | Luật CNCNS 2025 Đ.34 | Unadj. negative | WEAK_KEEP/MEDIUM → WEAK_KEEP/LOW | 1→1 | 0→0 | E |
| 34 | `network_security_hard_001` | Luật ANM 2025 Đ.15 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | C |
| 35 | `network_security_hard_001` | NĐ 53/2022 Đ.10 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | C |
| 36 | `network_security_hard_001` | NĐ 53/2022 Đ.11 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→1 | C |
| 37 | `network_security_hard_001` | NĐ 53/2022 Đ.16 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | C |
| 38 | `electronic_transactions_easy_001` | Luật GDĐT 2023 Đ.34 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | F |
| 39 | `electronic_transactions_easy_001` | Luật GDĐT 2023 Đ.35 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | F |
| 40 | `electronic_transactions_easy_001` | Luật GDĐT 2023 Đ.36 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | F |
| 41 | `electronic_transactions_easy_001` | Luật GDĐT 2023 Đ.37 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | F |
| 42 | `electronic_transactions_easy_001` | Luật GDĐT 2023 Đ.38 | Confirmed | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→1 | A |
| 43 | `electronic_transactions_easy_001` | NĐ 52/2013 Đ.12 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | F |
| 44 | `electronic_transactions_hard_001` | Luật GDĐT 2023 Đ.15 | Confirmed | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | A |
| 45 | `electronic_transactions_hard_001` | Luật GDĐT 2023 Đ.16 | Confirmed | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | A |
| 46 | `electronic_transactions_hard_001` | Luật GDĐT 2023 Đ.17 | Confirmed | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | A |
| 47 | `electronic_transactions_hard_001` | Luật GDĐT 2023 Đ.37 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→1 | C |
| 48 | `electronic_transactions_hard_001` | NĐ 52/2013 Đ.10 | Unadj. negative | REMOVE/LOW → WEAK_KEEP/LOW | 0→1 | 0→0 | F |

Full document names, expected clause/point, before/after reasons, provision text,
element coverage và classification evidence nằm trong paired JSON.

## 5. Confirmed-only findings

Bốn confirmed recoveries:

| Testcase | Provision | Before | After | Context | Citation |
|---|---|---|---|---|---|
| `electronic_transactions_easy_001` | Luật GDĐT 2023 Đ.38 | REMOVE/LOW | WEAK_KEEP/LOW | 0→1 | 0→1 |
| `electronic_transactions_hard_001` | Luật GDĐT 2023 Đ.15 | REMOVE/LOW | WEAK_KEEP/LOW | 0→1 | 0→0 |
| `electronic_transactions_hard_001` | Luật GDĐT 2023 Đ.16 | REMOVE/LOW | WEAK_KEEP/LOW | 0→1 | 0→0 |
| `electronic_transactions_hard_001` | Luật GDĐT 2023 Đ.17 | REMOVE/LOW | WEAK_KEEP/LOW | 0→1 | 0→0 |

Hai confirmed rows có confidence giảm nhưng không bị loại:

- Luật BVDLCN 2025 Điều 20: `KEEP/HIGH → WEAK_KEEP/LOW`, context và citation
  đều giữ nguyên;
- VBHN Luật SHTT Điều 25: `KEEP/HIGH → WEAK_KEEP/LOW`, context giữ nguyên;
  citation đổi là Generation selection variance, không phải mất Applicability
  context.

Kết luận confirmed-only: `4 recovered`, `11 stable kept`,
`14 stable not reached`, `0 true regression`.

## 6. Applicability payload audit

Vì không có true confirmed regression, không có row nào đủ điều kiện bắt buộc
replay payload. Source/artifact audit cho mười failure mode:

| Failure mode | Evidence | Kết luận |
|---|---|---|
| Fact đúng bị mất | Question và retrieval guardrails ổn định; không có confirmed loss | Không thấy |
| Inference bị xem như stated | `supported_inferences` rỗng trên changed benchmark cases | Không thấy |
| Unknown bị xem như phủ định | Decision đổi cả ở case có và không có unknown; confirmed retention tăng | Không chứng minh |
| Required element quá rộng | 170/171 after decisions là `PARTIAL_MATCH` | Không thấy |
| Required element quá chặt | Không có `NOT_APPLICABLE` | Không thấy |
| PARTIAL parsed thành NOT | Không có parsed `NOT_APPLICABLE` | Loại trừ |
| Fact alias không nhận | “giả giọng” alias hoạt động và chỉ thêm một candidate bị loại đúng | Không phải regression |
| Long guard áp dụng query ngắn | FACT_STATE được đưa vào mọi prompt | Có prompt-scope perturbation, chưa có semantic regression |
| Contract guard tổng quát hóa | Không có contract `NOT_APPLICABLE` override trong 30 case | Loại trừ |
| Deterministic override ghi đè model | Override count = 0 | Loại trừ |

Observability limitation: benchmark trace hiện tại không persist nguyên
`model_input` và `raw_model_output`, nên hai field này không thể phục dựng
byte-for-byte từ artifact. Parsed output, reason, source text và element fields
đã được giữ đầy đủ trong paired JSON. Audit không sửa runner để bổ sung trace vì
task cấm sửa benchmark runner.

## 7. Downstream impact

Trên 30 case:

- final-context set thay đổi ở 14 case;
- citation set thay đổi ở 9 case;
- rendered answer byte-for-byte thay đổi ở 23 case;
- grounding validation chuyển `invalid → valid` ở 3 case và không có
  `valid → invalid`;
- Citation Accuracy tổng thể vẫn `30,00%`;
- không có hallucinated citation.

Không thể quy 23 answer text changes cho Applicability: Generation cùng model ở
`temperature=0.2`, và bốn row nhóm D có citation đổi dù keep/drop cùng
final-context membership không đổi. Benchmark artifacts cũng không persist
`answer_assessment`, nên audit không bịa một before/after assessment.

Không có chuỗi:

```text
confirmed provision kept
→ removed by Applicability
→ removed from final context
→ correct citation lost
```

Ngược lại, Điều 38 có chuỗi recovery đầy đủ tới citation. Điều 15–17 được phục
hồi vào context nhưng Generation chưa cite.

## 8. Quyết định patch và rollback

Không patch production vì điều kiện bắt buộc “true regression trên
confirmed-correct label” không xảy ra. Sửa prompt để ép 20 unadjudicated
negative candidates về `REMOVE` sẽ tối ưu metric mù quáng và có nguy cơ xóa các
căn cứ thay thế thật sự liên quan.

Do không có production patch:

- không có code rollback;
- không cần chạy lại benchmark 30 case;
- rollback artifact audit, nếu cần, chỉ là xóa ba file deliverable của task;
- long-scenario fix tiếp tục ở trạng thái review hiện tại.

Promotion guardrails được đánh giá trên artifact after hiện có:

| Guardrail | Yêu cầu | Kết quả |
|---|---:|---:|
| Confirmed Applicability | không giảm | 73,33% → 100,00% |
| Overall Applicability | phục hồi hoặc adjudication rõ | giảm do unadjudicated negatives; đã tách bằng chứng |
| Citation Accuracy | ≥ 30,00% | 30,00% |
| Recall@10 | ≥ 41,67% | 41,67% |
| Wrong Domain | ≤ 19,56% | 19,56% |
| Hallucinated Citation | 0 | 0 |
| LLM calls | không tăng | 2 → 2 |
| Benchmark labels | không sửa | xác nhận |

## 9. Verification

Các suite được chạy bằng project virtual environment sau khi tạo deliverables:

```text
.venv/bin/python -m pytest -q tests
101 passed, 2 subtests passed

.venv/bin/python -m pytest -q evaluation/tests
11 passed
```

Hai JSON đều parse hợp lệ. Audit assertion xác nhận:

- 48 changed rows;
- 29 confirmed annotations;
- 4 confirmed recoveries;
- 0 confirmed true regression;
- không có duplicate article decision trong từng testcase/run;
- benchmark hash trước và sau giống nhau.

Template compliance `30/30`, conclusion contradiction `0/30` và Hallucinated
Citation `0` là kết quả của after benchmark/final-output validation đã chạy
trước task này. Vì không có production patch, audit không chạy lại LLM
benchmark chỉ để tái lấy cùng promotion evidence.

## 10. Files và xác nhận phạm vi

Task này chỉ tạo:

- `docs/applicability_regression_audit_2026-07-28.md`;
- `evaluation/results/applicability_paired_diff_20260728.json`;
- `evaluation/results/applicability_confirmed_only_20260728.json`.

Không sửa:

- production Applicability, Retrieval, Generation hoặc workflow code;
- benchmark data, benchmark runner hoặc evaluation metrics;
- output redesign, renderer, Generation template hoặc safe fallback;
- public API;
- long-scenario behavior.

Không đề xuất phase mới trong audit này.
