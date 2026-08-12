# Legal Retrieval Benchmark Report

- Run ID: `20260726_111221`
- Generated: `2026-07-26T11:12:21+00:00`
- Benchmark SHA-256: `126438a83f9902d950b140f74fd1f527b899ec9dc160ce1163f47c0c5d3e114c`
- Cases: **1/1** completed
- Runtime errors: **0**

## Overall Metrics

| Metric | Value |
|---|---:|
| Domain Recall | 100.00% |
| Domain Precision | 100.00% |
| Behavior Recall | 50.00% |
| Behavior Precision | 100.00% |
| Recall@5 | 100.00% |
| Recall@10 | 100.00% |
| MRR | 50.00% |
| Citation Accuracy | 66.67% |
| Wrong Domain Rate | 0.00% |
| Wrong Behavior Rate | 0.00% |
| Recursive Precision | 100.00% |
| Recursive Noise Rate | 0.00% |
| Applicability Accuracy | 20.00% |
| Average Retrieval Latency | 9104.5 ms |
| Average Total Latency | 116348.3 ms |

## Metrics by Category

| Category | Cases | Domain R | Behavior R | R@5 | R@10 | Citation | Wrong domain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Malware | 1 | 100.00% | 50.00% | 100.00% | 100.00% | 66.67% | 0.00% |

## Metrics by Difficulty

| Difficulty | Cases | Domain R | Behavior R | R@5 | R@10 | Citation | Applicability |
|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | 1 | 100.00% | 50.00% | 100.00% | 100.00% | 66.67% | 20.00% |

## Top Recurring Errors

- **Applicability Error**: 1
- **Generation Grounding Error**: 1
- **Wrong Behavior**: 1
- **Wrong Citation**: 1
- **Hallucinated Citation**: 0
- **Missing Relevant Law**: 0
- **Recursive Noise**: 0
- **Wrong Domain**: 0

## Top 20 Failed Cases

### 1. `malware_easy_001` — Malware / Easy

- Errors: Wrong Behavior, Wrong Citation, Applicability Error, Generation Grounding Error
- Recall@10: 100.00%
- Citation Accuracy: 66.67%
- Question: Hệ thống thông tin phải thực hiện biện pháp nào để phát hiện và loại bỏ mã độc trong phần cứng, phần mềm?

## Artifacts

- `benchmark_summary.json`: aggregate metrics and run metadata.
- `benchmark_details.json`: per-case trace, metrics and errors.
- `plots/`: visual metrics generated from the same summary.
