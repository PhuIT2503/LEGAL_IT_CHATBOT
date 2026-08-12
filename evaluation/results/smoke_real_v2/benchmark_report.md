# Legal Retrieval Benchmark Report

- Run ID: `20260726_092835`
- Generated: `2026-07-26T09:28:35+00:00`
- Benchmark SHA-256: `126438a83f9902d950b140f74fd1f527b899ec9dc160ce1163f47c0c5d3e114c`
- Cases: **1/1** completed
- Runtime errors: **0**

## Overall Metrics

| Metric | Value |
|---|---:|
| Domain Recall | 50.00% |
| Domain Precision | 100.00% |
| Behavior Recall | 100.00% |
| Behavior Precision | 100.00% |
| Recall@5 | 0.00% |
| Recall@10 | 0.00% |
| MRR | 0.00% |
| Citation Accuracy | 0.00% |
| Wrong Domain Rate | 0.00% |
| Wrong Behavior Rate | 0.00% |
| Recursive Precision | 100.00% |
| Recursive Noise Rate | 0.00% |
| Applicability Accuracy | 0.00% |
| Average Retrieval Latency | 8772.8 ms |
| Average Total Latency | 142879.2 ms |

## Metrics by Category

| Category | Cases | Domain R | Behavior R | R@5 | R@10 | Citation | Wrong domain |
|---|---:|---:|---:|---:|---:|---:|---:|
| SQL Injection | 1 | 50.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## Metrics by Difficulty

| Difficulty | Cases | Domain R | Behavior R | R@5 | R@10 | Citation | Applicability |
|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | 1 | 50.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## Top Recurring Errors

- **Applicability Error**: 1
- **Generation Grounding Error**: 1
- **Missing Relevant Law**: 1
- **Wrong Citation**: 1
- **Wrong Domain**: 1
- **Hallucinated Citation**: 0
- **Recursive Noise**: 0
- **Wrong Behavior**: 0

## Top 20 Failed Cases

### 1. `sql_injection_easy_001` — SQL Injection / Easy

- Errors: Wrong Domain, Wrong Citation, Missing Relevant Law, Applicability Error, Generation Grounding Error
- Recall@10: 0.00%
- Citation Accuracy: 0.00%
- Question: Dùng SQL Injection khai thác lỗ hổng website và tải xuống cơ sở dữ liệu có phải là truy cập trái phép không?

## Artifacts

- `benchmark_summary.json`: aggregate metrics and run metadata.
- `benchmark_details.json`: per-case trace, metrics and errors.
- `plots/`: visual metrics generated from the same summary.
