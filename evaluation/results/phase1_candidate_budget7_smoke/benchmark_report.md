# Legal Retrieval Benchmark Report

- Run ID: `20260727_121121`
- Generated: `2026-07-27T12:11:21+07:00`
- Benchmark SHA-256: `126438a83f9902d950b140f74fd1f527b899ec9dc160ce1163f47c0c5d3e114c`
- Cases: **1/1** completed
- Runtime errors: **0**

## Overall Metrics

| Metric | Value |
|---|---:|
| Domain Recall | 50.00% |
| Domain Precision | 100.00% |
| Behavior Recall | 50.00% |
| Behavior Precision | 100.00% |
| Recall@5 | 0.00% |
| Recall@10 | 100.00% |
| MRR | 14.29% |
| Citation Accuracy | 100.00% |
| Wrong Domain Rate | 0.00% |
| Wrong Behavior Rate | 0.00% |
| Recursive Precision | 100.00% |
| Recursive Noise Rate | 0.00% |
| Applicability Accuracy | 71.43% |
| Average Retrieval Latency | 8891.3 ms |
| Average Total Latency | 105260.3 ms |

## Metrics by Category

| Category | Cases | Domain R | Behavior R | R@5 | R@10 | Citation | Wrong domain |
|---|---:|---:|---:|---:|---:|---:|---:|
| AI Copyright | 1 | 50.00% | 50.00% | 0.00% | 100.00% | 100.00% | 0.00% |

## Metrics by Difficulty

| Difficulty | Cases | Domain R | Behavior R | R@5 | R@10 | Citation | Applicability |
|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | 1 | 50.00% | 50.00% | 0.00% | 100.00% | 100.00% | 71.43% |

## Top Recurring Errors

- **Applicability Error**: 1
- **Generation Grounding Error**: 1
- **Wrong Behavior**: 1
- **Wrong Domain**: 1
- **Hallucinated Citation**: 0
- **Missing Relevant Law**: 0
- **Recursive Noise**: 0
- **Wrong Citation**: 0

## Top 20 Failed Cases

### 1. `ai_copyright_easy_001` — AI Copyright / Easy

- Errors: Wrong Domain, Wrong Behavior, Applicability Error, Generation Grounding Error
- Recall@10: 100.00%
- Citation Accuracy: 100.00%
- Question: Dùng tác phẩm có bản quyền của người khác làm dữ liệu đầu vào cho công cụ AI có liên quan đến quyền tài sản nào của tác giả?

## Artifacts

- `benchmark_summary.json`: aggregate metrics and run metadata.
- `benchmark_details.json`: per-case trace, metrics and errors.
- `plots/`: visual metrics generated from the same summary.
