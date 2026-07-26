# Applicability Recovery & Retrieval Preservation — Regression Analysis

## Verdict

The change passes every explicit success guardrail for candidate recovery and citation safety, but the overall result is mixed rather than an unqualified success:

- Relevant candidate preservation improved substantially.
- Citation Accuracy increased substantially.
- Retrieval metrics and domain/recursive guardrails did not regress.
- No hallucinated citation escaped final validation.
- Applicability Accuracy and total latency regressed because far more seed candidates were evaluated and retained, including non-ground-truth candidates.

No Domain Selection, Behavior Extraction, Hybrid Retrieval, Recursive Retrieval, Generation prompt, public API, or rendered output format was changed.

## Runs compared

| Run | Directory |
|---|---|
| Before | `evaluation/results/benchmark_30_real_20260726/` |
| After | `evaluation/results/benchmark_30_applicability_recovery_20260726/` |

Both runs use the same 30-case benchmark, `critic` mode, `top_k=10`, embedding model `AITeamVN/Vietnamese_Embedding_v2`, and `qwen2.5:7b`.

## End-to-end funnel

| Stage | Before | After | Delta |
|---|---:|---:|---:|
| Expected provisions | 44 | 44 | 0 |
| Retrieved in top-10 | 16 | 16 | 0 |
| Reached Applicability | 13 | 16 | +3 |
| Kept in final context | 2 | 12 | +10 |
| Correctly cited provisions | 2 | 9 | +7 |
| Cases with empty final context | 21 | 6 | -15 |

Relevant-candidate recall among candidates evaluated by Applicability increased from `2/13 = 15.38%` to `12/16 = 75.00%`.

At article level, Applicability evaluated 114 candidates before and 170 after. It kept 14 articles before and 86 after. Relevant precision among kept articles was effectively flat (`14.29%` to `13.95%`), which explains the lower aggregate Applicability Accuracy despite the much higher relevant recall.

## Metric comparison

| Metric | Before | After | Delta | Guardrail |
|---|---:|---:|---:|---|
| Domain Recall | 57.83% | 57.83% | 0.00 pp | Unchanged |
| Behavior Recall | 30.85% | 30.85% | 0.00 pp | Unchanged |
| Recall@5 | 38.33% | 38.33% | 0.00 pp | Unchanged |
| Recall@10 | 41.67% | 41.67% | 0.00 pp | Pass: did not decrease |
| Citation Accuracy | 5.56% | 21.67% | **+16.11 pp** | Pass: increased |
| Wrong Domain Rate | 19.56% | 19.56% | 0.00 pp | Pass: did not increase |
| Applicability Accuracy | 74.33% | 53.83% | **-20.51 pp** | Regression |
| Recursive Noise Rate | 10.00% | 10.00% | 0.00 pp | Pass: did not increase |
| Hallucinated Citation cases | 0 | 0 | 0 | Pass |
| Average Retrieval Latency | 4.04 s | 4.22 s | +0.18 s | Small run variance |
| Average Total Latency | 75.97 s | 106.42 s | **+30.45 s** | Regression |

The Applicability Accuracy result must be interpreted together with the benchmark's known true-negative dominance. The new decision policy recovers false negatives, but also retains many additional candidates that are not present in the benchmark ground truth. The target recall improvement is real; precision did not improve.

## Why the benchmark identified the correct bugs

### Seed loss at Legal Relevance

Before the change, three correct top-10 provisions never reached Applicability. After seed preservation, all 16 top-10 expected provisions reached Applicability. This directly validates the benchmark diagnosis that Legal Relevance was removing correct Phase 2 seeds too early.

### Empty primary action

Before the change, object-only and purpose-only Behavior Cards produced errors such as `HIGH nhưng không match primary action`. After making the primary-action check conditional on a primary action actually existing, correct provisions survived in Malware, Advertising, and Network Security cases.

### Validator-only rejection

The decision layer now separates the LLM applicability level/validation status from the final `KEEP`, `WEAK_KEEP`, or `REMOVE` action. Invented behavior remains `INVALID` and is never trusted. A retrieval seed may be weakly preserved from provenance when the validator output is structurally invalid, but Generation receives only retrieved source text—not the invented behavior.

## Testcase outcomes

### Improved end-to-end citation

| Testcase | Citation Accuracy before | After |
|---|---:|---:|
| `malware_easy_001` | 0.00% | 50.00% |
| `ai_copyright_easy_001` | 0.00% | 100.00% |
| `ai_copyright_medium_001` | 0.00% | 66.67% |
| `advertising_medium_001` | 0.00% | 66.67% |
| `consumer_easy_001` | 0.00% | 66.67% |
| `network_security_easy_001` | 0.00% | 100.00% |
| `electronic_transactions_medium_001` | 0.00% | 66.67% |

### Candidate preservation improved, citation unchanged

- `cyber_attack_easy_001`
- `sql_injection_medium_001`
- `advertising_easy_001`

Each gained one expected provision in final context, but the rendered answer still failed to cite it correctly.

### Regressed

- `personal_data_medium_001`: Citation Accuracy decreased from 100.00% to 66.67%. The expected provision remained in final context; the regression occurred in final citation selection/rendering under the larger context.

### Unchanged

- `deepfake_easy_001`
- `deepfake_medium_001`
- `deepfake_hard_001`
- `personal_data_easy_001`
- `personal_data_hard_001`
- `cyber_attack_medium_001`
- `cyber_attack_hard_001`
- `sql_injection_easy_001`
- `sql_injection_hard_001`
- `malware_medium_001`
- `malware_hard_001`
- `ai_copyright_hard_001`
- `advertising_hard_001`
- `consumer_medium_001`
- `consumer_hard_001`
- `network_security_medium_001`
- `network_security_hard_001`
- `electronic_transactions_easy_001`
- `electronic_transactions_hard_001`

Most unchanged failures lack the expected provision in top-10 or are blocked by domain/behavior coverage, which this task explicitly did not modify.

## Instrumentation added

Every relevance/applicability decision now records or logs:

- `seed_preserved`
- `seed_survived`
- `seed_removed`
- `reason_removed`
- `behavior_preserved`
- `relevance_removed`
- `applicability_removed`
- `decision_stage`
- final `decision` (`KEEP`, `WEAK_KEEP`, or `REMOVE`)

The benchmark stores these records under `actual.retrieval_decisions`, allowing future runs to identify the exact removal stage without parsing text logs.

## Success-criteria result

| Criterion | Result |
|---|---|
| At least 8 correct candidates survive Applicability | Pass: 12 |
| Citation Accuracy increases | Pass: 5.56% → 21.67% |
| Recall@10 does not decrease | Pass: unchanged at 41.67% |
| Wrong Domain Rate does not increase | Pass: unchanged at 19.56% |
| Recursive Noise does not increase | Pass: unchanged at 10.00% |
| Hallucination does not increase | Pass: remains 0 |

The scoped recovery objective therefore passes. However, because Applicability Accuracy and latency regressed, this version should be described as a recall-oriented recovery with a measurable precision/cost trade-off, not as a universally better Applicability system.
