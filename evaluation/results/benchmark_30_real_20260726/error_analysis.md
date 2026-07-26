# Error Analysis — Real Benchmark 30 Cases

## Run configuration

- Run ID: `20260726_100742`
- Benchmark: 30 cases, 10 categories, 3 difficulty levels
- Completed: 30/30
- Runtime errors: 0
- Mode: `critic`
- Retrieval top-k: 10
- Embedding: `AITeamVN/Vietnamese_Embedding_v2`
- Generation model: `qwen2.5:7b`
- Corpus: local Qdrant snapshot at `data/.qdrant_base`
- Source revision at analysis time: `9bf145e6744599e1e15c21e2011a02f7ecff87f5` (`main`)

> The run was executed inside the application Docker image without mounting `.git`, so the automatic `git` fields in `benchmark_summary.json` are empty. The source revision above was recorded from the host workspace after the run. The worktree was dirty and contains the user's earlier project changes.

## Headline metrics

| Metric | Result |
|---|---:|
| Domain Recall | 57.83% |
| Domain Precision | 80.44% |
| Behavior Recall | 30.85% |
| Behavior Precision | 59.17% |
| Recall@5 | 38.33% |
| Recall@10 | 41.67% |
| MRR | 0.3114 |
| Citation Accuracy | 5.56% |
| Wrong Domain Rate | 19.56% |
| Wrong Behavior Rate | 40.83% |
| Recursive Precision | 90.00% |
| Recursive Noise Rate | 10.00% |
| Applicability Accuracy | 74.33% |
| Average Retrieval Latency | 4.04 s |
| Average Total Latency | 75.97 s |

The aggregate Applicability Accuracy is misleading because it is dominated by true negatives. Relevant-candidate preservation is a better diagnostic for this run.

## End-to-end evidence funnel

```text
44 expected legal provisions
        ↓
16 found in retrieval top-10 (36.36% micro recall)
        ↓
13 reached Applicability evaluation
        ↓
2 were retained in final context
        ↓
2 cases produced matching citations
```

- 15/44 expected provisions appeared in top-5.
- 16/44 appeared in top-10.
- Increasing top-k from 5 to 10 recovered only one additional expected provision. The main problem is candidate ordering/selection, not merely an insufficient top-k.
- 21/30 cases had an empty final legal context before Generation.
- Only 2/16 relevant top-10 candidates survived to final context: 12.5% preservation.

## Root-cause classification

The categories below overlap and therefore must not be summed as if they were mutually exclusive case counts.

| Root cause | Evidence | Impact |
|---|---|---:|
| Corpus missing | Every expected document/article/clause/point was found in the 18 source DOCX files | 0 expected provisions |
| Domain selection blocked retrieval | Expected document domain was absent from `selected_domains` | 9 provisions across 7 cases |
| Within-domain retrieval/ranking | Correct document domain was open, but expected provision was absent from top-10 | 19 provisions across 14 cases |
| Unsupported behavior taxonomy | Expected behavior key does not exist in the current taxonomy | 15 labels across 11 cases |
| Supported behavior not extracted | Expected key exists but was not emitted | 57 missed labels across 22 cases |
| Legal Relevance pre-filter | Correct retrieved provision did not reach Applicability | 3 provisions |
| Applicability rejection | Correct provision reached Applicability but was not retained | 11 of 13 evaluated provisions |
| Recursive noise | Recursive candidates included irrelevant provisions | 3 cases |
| Raw Generation grounding | Raw answer failed grounding validation | 29 cases |
| Final hallucinated citation | Unsupported citations escaped final validation | 0 cases |

### 1. Corpus completeness is not the current blocker

The benchmark expected 44 document/article/clause/point combinations. A direct audit of all 18 DOCX corpus files found every expected provision. Therefore, for this 30-case benchmark, ingesting more legislation is not supported as the first corrective action.

This finding only establishes presence in the source corpus. It does not prove that every provision was indexed with ideal chunk text or metadata.

### 2. Domain selection blocked nine expected provisions

| Case | Expected provision blocked by domain selection |
|---|---|
| `deepfake_medium` | Luật An ninh mạng, Điều 7; Điều 13 |
| `personal_data_hard` | Personal Data document, Điều 2 |
| `sql_injection_hard` | Personal Data document, Điều 20 |
| `advertising_hard` | Personal Data document, Điều 28 |
| `consumer_medium` | Consumer Protection document, Điều 34; Điều 35 |
| `consumer_hard` | Consumer Protection document, Điều 25 |
| `network_security_medium` | Nghị định 53, Điều 11 |

Five cases triggered the generic domain fallback:

- `cyber_attack_easy`
- `cyber_attack_medium`
- `consumer_medium`
- `electronic_transactions_easy`
- `electronic_transactions_medium`

The fallback domains were `cybersecurity`, `personal_data`, `digital_technology`, `electronic_transactions`, and `telecommunications`. This is especially damaging for `consumer_medium`, whose actual consumer-protection domain was not opened.

### 3. Nineteen provisions disappeared inside an already-correct domain

The domain filter was not the blocker for these candidates; the expected document's domain was selected, yet the provision did not reach top-10.

| Case | Expected provision absent from top-10 |
|---|---|
| `deepfake_easy` | Luật An ninh mạng, Điều 13 |
| `deepfake_hard` | Personal Data, Điều 28; Luật An ninh mạng, Điều 13 |
| `personal_data_easy` | Personal Data, Điều 28 |
| `cyber_attack_medium` | Nghị định 15, Điều 81; Luật An ninh mạng, Điều 13 |
| `cyber_attack_hard` | Luật An ninh mạng, Điều 13; Nghị định 15, Điều 81 |
| `sql_injection_easy` | Luật An ninh mạng, Điều 13 |
| `sql_injection_medium` | Luật An ninh mạng, Điều 13 |
| `sql_injection_hard` | Luật An ninh mạng, Điều 13 |
| `malware_medium` | Luật An ninh mạng, Điều 36 |
| `malware_hard` | Luật An ninh mạng, Điều 13; Nghị định 53, Điều 11 |
| `ai_copyright_hard` | Intellectual Property, Điều 35; Nghị định 17, Điều 28 |
| `advertising_hard` | Consumer Protection, Điều 10 |
| `network_security_hard` | Luật An ninh mạng, Điều 36 |
| `electronic_transactions_medium` | Electronic Transactions, Điều 18 |

This is the strongest evidence for a within-domain candidate-generation/ranking issue. The repeated failure to surface Luật An ninh mạng Điều 13 across multiple cyber and SQL-injection questions is the clearest recurring pattern.

### 4. Behavior metric combines taxonomy coverage and extraction quality

Twelve expected behavior keys are unsupported by the current taxonomy:

- `automated_electronic_contract`
- `confirm_receipt`
- `correct_input_error`
- `deploy_malware`
- `detect_or_remove_malware`
- `receive_data_message`
- `research_malware`
- `secure_information_system`
- `send_data_message`
- `supply_defective_product`
- `use_electronic_notice`
- `use_unfair_contract_term`

These account for 15 expected labels across 11 cases, concentrated in Malware, Consumer Protection, Network Security, and Electronic Transactions.

Even after excluding unsupported keys, extraction still missed 57 expected labels across 22 cases and emitted six false-positive labels across six cases. The most frequently missed supported keys were:

| Behavior | Misses |
|---|---:|
| `commercial_gain` | 6 |
| `personal_data` | 5 |
| `without_authorization` | 5 |
| `extract_or_download_data` | 5 |
| `without_consent` | 4 |
| `process_personal_data_without_consent` | 4 |
| `unauthorized_access` | 4 |
| `data_exfiltration` | 3 |
| `use_copyrighted_work` | 3 |

Therefore, the 30.85% Behavior Recall must not be interpreted as a pure extraction-model score. It measures both taxonomy coverage and extraction behavior.

### 5. Correct candidates are mostly lost after retrieval

Three correct top-10 provisions were removed before Applicability:

- `sql_injection_medium`: Nghị định 15, Điều 81
- `ai_copyright_easy`: Intellectual Property, Điều 20
- `ai_copyright_medium`: Intellectual Property, Điều 25

Of the 13 expected provisions that did reach Applicability, only two were retained:

- `deepfake_easy`: Luật An ninh mạng, Điều 7
- `personal_data_medium`: Personal Data, Điều 20

A recurring validator failure occurs when the Behavior Card contains only objects or conditions and no primary action. A HIGH applicability result is then invalidated because it cannot match a primary action that does not exist. Cases exposing this pattern include:

- `cyber_attack_easy`: only `without_authorization`; correct Nghị định 15 Điều 80 was dropped.
- `malware_easy`: only `website_or_information_system`; correct Nghị định 53 Điều 10 was dropped.
- `advertising_medium`: only `advertising`; correct Consumer Protection Điều 10 was dropped.
- `ai_copyright_medium`: only `copyrighted_work_or_recording`; correct Intellectual Property Điều 25 was dropped after relevance filtering.

Electronic Transactions cases produced an empty Behavior Card. Applicability subsequently emitted an unknown behavior key, which the integrity validator correctly rejected. The rejection is safe, but it leaves the final context empty; the upstream taxonomy/extraction gap is the initiating cause.

### 6. Recursive Retrieval is not the dominant failure

Recursive noise was observed in only three cases:

- `cyber_attack_medium`
- `advertising_easy`
- `network_security_easy`

Recursive Precision was 90%, substantially better than the retrieval and final-context preservation metrics. There is no benchmark evidence to prioritize Recursive Retrieval changes before domain, within-domain ranking, behavior coverage, and Applicability preservation.

### 7. Generation is unsafe raw, but final citation validation is fail-closed

Raw grounding validation failed in 29/30 cases. Raw drafts attempted unsupported citations, but the final citation parser/validator prevented unsupported citations from escaping to the rendered output. After correcting an evaluator parsing false positive, the final Hallucinated Citation count is zero.

Citation Accuracy remains only 5.56%, chiefly because 21 cases entered Generation with no final legal context and only two relevant provisions survived the full pipeline. This means citation quality is currently a downstream symptom as well as a Generation issue. Prompt changes should not be the first response to this benchmark.

## Difficulty trend

| Difficulty | Domain Recall | Behavior Recall | Recall@5 | Recall@10 | Citation Accuracy |
|---|---:|---:|---:|---:|---:|
| Easy | 75.00% | 43.33% | 65.00% | 75.00% | 6.67% |
| Medium | 45.33% | 28.75% | 40.00% | 40.00% | 10.00% |
| Hard | 53.17% | 20.47% | 10.00% | 10.00% | 0.00% |

The sharp Recall@10 decline from 75% on Easy to 10% on Hard shows that isolated smoke tests substantially overestimate retrieval quality.

## Per-case retrieval outcome

| Case | R@5 | R@10 | Main observed blocker |
|---|---:|---:|---|
| `deepfake_easy` | 0.50 | 0.50 | One expected article missing; one survived end-to-end |
| `deepfake_medium` | 0.00 | 0.00 | Domain selection |
| `deepfake_hard` | 0.00 | 0.00 | Within-domain ranking |
| `personal_data_easy` | 0.00 | 0.00 | Within-domain ranking |
| `personal_data_medium` | 1.00 | 1.00 | Passing reference case |
| `personal_data_hard` | 0.00 | 0.00 | Domain selection |
| `cyber_attack_easy` | 1.00 | 1.00 | Applicability primary-action validation |
| `cyber_attack_medium` | 0.00 | 0.00 | Within-domain ranking; recursive noise |
| `cyber_attack_hard` | 0.00 | 0.00 | Within-domain ranking |
| `sql_injection_easy` | 0.00 | 0.00 | Within-domain ranking |
| `sql_injection_medium` | 0.50 | 0.50 | Legal Relevance pre-filter |
| `sql_injection_hard` | 0.00 | 0.00 | Domain selection and within-domain ranking |
| `malware_easy` | 1.00 | 1.00 | Applicability primary-action validation |
| `malware_medium` | 0.00 | 0.00 | Behavior taxonomy; within-domain ranking |
| `malware_hard` | 0.00 | 0.00 | Behavior taxonomy; within-domain ranking |
| `ai_copyright_easy` | 0.00 | 1.00 | Legal Relevance pre-filter |
| `ai_copyright_medium` | 1.00 | 1.00 | Legal Relevance/Applicability preservation |
| `ai_copyright_hard` | 0.00 | 0.00 | Within-domain ranking |
| `advertising_easy` | 1.00 | 1.00 | Applicability; recursive noise |
| `advertising_medium` | 1.00 | 1.00 | Applicability primary-action validation |
| `advertising_hard` | 0.00 | 0.00 | Domain selection and within-domain ranking |
| `consumer_easy` | 1.00 | 1.00 | Applicability/final-context preservation |
| `consumer_medium` | 0.00 | 0.00 | Generic domain fallback; taxonomy gap |
| `consumer_hard` | 0.00 | 0.00 | Domain selection; taxonomy gap |
| `network_security_easy` | 1.00 | 1.00 | Applicability primary-action validation; recursive noise |
| `network_security_medium` | 0.00 | 0.00 | Domain selection; taxonomy gap |
| `network_security_hard` | 0.00 | 0.00 | Within-domain ranking; taxonomy gap |
| `electronic_transactions_easy` | 1.00 | 1.00 | Empty Behavior Card; Applicability rejects invented key |
| `electronic_transactions_medium` | 0.50 | 0.50 | One article missing; empty Behavior Card |
| `electronic_transactions_hard` | 1.00 | 1.00 | Empty Behavior Card; Applicability rejects invented key |

## Data-driven next priorities

No pipeline code was changed during this run. Based on the evidence, the next experiments should be isolated and measured in this order:

1. **Preserve correct candidates through Legal Relevance and Applicability.** This is the largest immediate funnel loss: only 2/16 relevant retrieved candidates survive. In particular, test primary-action validation when the Behavior Card has no primary action.
2. **Separate behavior taxonomy coverage from extraction accuracy.** Add a benchmark metric for supported-only Behavior Recall before changing taxonomy or extraction rules. The current number conflates two causes.
3. **Improve within-domain retrieval/ranking.** Nineteen expected provisions are missing even though their domain is already open. Repeated Luật An ninh mạng Điều 13 misses should form the first regression subset.
4. **Improve domain signal coverage.** Nine provisions are impossible to retrieve under selected domain filters, and five cases enter a generic fallback. Consumer Protection is a particularly clear failure.
5. **Address Generation grounding only after relevant final context is reliably non-empty.** Raw Generation is problematic, but changing its prompt now would mask the larger upstream losses.
6. **Leave Recursive Retrieval unchanged for now.** Its measured noise is comparatively low.

Each proposed change should be evaluated against this frozen run and the same benchmark SHA-256. A change should be accepted only if it improves its target metric without degrading final-context preservation, hallucinated-citation count, or latency beyond an agreed threshold.

## Evaluation-instrumentation corrections made during the run

Two evaluator-only defects were discovered and fixed; no retrieval or chatbot pipeline code was modified:

1. Candidate coordinates now fall back to parsing `chunk_id` when article/clause/point metadata is absent. Without this, real retrieval hits were incorrectly scored as misses.
2. Citation parsing now requires a plausible legal document name. Without this, prose such as “Nội dung điều luật: Khoản 1, Điều 10” was falsely classified as a hallucinated citation.

The existing trace was deterministically rescored after the citation-parser correction. The final `Hallucinated Citation` count is therefore 0, not the earlier false-positive count.
