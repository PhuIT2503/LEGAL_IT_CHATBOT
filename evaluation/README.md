# Legal Chatbot Benchmark & Evaluation

Framework này đánh giá end-to-end pipeline hiện tại mà không sửa Retrieval, Behavior, Recursive Retrieval, Applicability, Generation, public API hay output chatbot. Mỗi testcase gọi pipeline đúng một lần. Mọi metric và phân loại lỗi đều là deterministic; không có LLM judge bổ sung.

## Kiến trúc

```text
benchmark.json
      │
      ▼
run_benchmark.py ── gọi ChatbotWorkflow.run() đúng 1 lần/case
      │
      ├── quan sát Retrieval output
      │     ├── selected domains
      │     ├── Behavior Card
      │     ├── ranked candidates
      │     └── recursive candidates + provenance
      │
      ├── quan sát Applicability decisions
      ├── quan sát context thực tế vào Generation
      ├── đọc citation và rendered answer
      ▼
metrics.py ── deterministic scoring + error taxonomy
      ▼
reporting.py
      ├── benchmark_summary.json
      ├── benchmark_details.json
      ├── benchmark_report.md
      └── plots/*.png
```

Runner bọc tạm các method đã tồn tại để ghi trace và khôi phục chúng ngay sau mỗi case. Cơ chế này không thay đổi state, tham số, số lần gọi model hay kết quả của production pipeline.

## Chuẩn bị môi trường

Các dịch vụ và biến môi trường giống khi chạy chatbot hiện tại:

```bash
docker compose up -d neo4j ollama
export QDRANT_PATH=data/.qdrant_base
export EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding_v2
export OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_MODEL=qwen2.5:7b
```

Qdrant embedded chỉ cho một process mở cùng index. Nếu giao diện `legal_app` đang chạy với `data/.qdrant_base`, hãy dừng app trong lúc benchmark hoặc trỏ `QDRANT_PATH` tới một bản index riêng dành cho evaluation.

Để sinh PNG cần `matplotlib`. Nếu package chưa có, runner vẫn sinh đầy đủ JSON/Markdown và ghi `plot_error` vào summary:

```bash
python -m pip install matplotlib
```

## Chạy benchmark

Toàn bộ benchmark:

```bash
python evaluation/run_benchmark.py
```

Theo category hoặc difficulty:

```bash
python evaluation/run_benchmark.py --category deepfake
python evaluation/run_benchmark.py --category sql_injection
python evaluation/run_benchmark.py --difficulty hard
```

Chạy một case khi điều tra regression:

```bash
python evaluation/run_benchmark.py --case-id sql_injection_easy_001
```

Chỉ kiểm tra dataset, không load model và không gọi chatbot:

```bash
python evaluation/run_benchmark.py --validate-only
```

Mặc định runner lấy `top_k=10` để có thể tính cả Recall@5 và Recall@10. Nếu cần tái hiện một cấu hình khác, ghi rõ trong command; cấu hình được lưu trong summary:

```bash
python evaluation/run_benchmark.py --top-k 10 --output-dir evaluation/results/experiment_p2
```

## Ý nghĩa metric

| Metric | Ý nghĩa |
|---|---|
| Domain Recall | Tỷ lệ domain ground truth xuất hiện trong domain được chọn. |
| Domain Precision | Tỷ lệ domain được chọn thuộc ground truth. |
| Behavior Recall | Tỷ lệ behavior ground truth xuất hiện trong Behavior Card. |
| Behavior Precision | Tỷ lệ behavior trong Behavior Card thuộc ground truth. |
| Recall@5 / Recall@10 | Tỷ lệ tọa độ pháp lý kỳ vọng có mặt trong 5/10 candidate đầu. |
| MRR | Nghịch đảo thứ hạng candidate đúng đầu tiên; đúng ở rank 1 nhận 1.0. |
| Citation Accuracy | F1 giữa các Điều/Khoản/Điểm trong rendered answer và ground truth; phạt cả viện dẫn sai lẫn bỏ sót. |
| Wrong Domain Rate | `1 - Domain Precision`; phần domain được chọn sai. |
| Wrong Behavior Rate | `1 - Behavior Precision`; phần behavior được trích sai. |
| Recursive Precision | Tỷ lệ recursive candidate khớp ground truth. Không expand nhận 1.0. |
| Recursive Noise Rate | `1 - Recursive Precision`; phần recursive context gây nhiễu. Không expand nhận 0.0. |
| Applicability Accuracy | Accuracy nhị phân keep/drop trên các Điều được Applicability đánh giá. |
| Average Retrieval Latency | Thời gian riêng của Retrieval Agent, gồm hybrid/rerank/recursive. |
| Average Total Latency | Thời gian end-to-end của một lần `pipeline.run()`. |

Các metric tổng là macro-average: mỗi câu hỏi có trọng số bằng nhau. `benchmark_details.json` giữ trace đầy đủ để có thể kiểm tra nguyên nhân, không chỉ nhìn điểm tổng.

## Error taxonomy

- `Wrong Domain`: domain precision hoặc recall chưa đạt 100%.
- `Wrong Behavior`: Behavior Card thừa hoặc thiếu behavior ground truth.
- `Wrong Citation`: rendered citation không khớp đầy đủ ground truth.
- `Missing Relevant Law`: Recall@10 chưa lấy đủ căn cứ kỳ vọng.
- `Recursive Noise`: recursive expansion thêm Điều không liên quan.
- `Applicability Error`: quyết định keep/drop khác nhãn liên quan suy ra từ ground truth.
- `Generation Grounding Error`: raw generation không qua grounding validator hiện tại.
- `Hallucinated Citation`: rendered answer viện dẫn tọa độ không tồn tại trong final grounded sources.

Một case có thể thuộc nhiều loại lỗi. Điều này giúp phân biệt lỗi upstream (domain/behavior/retrieval) với lỗi downstream (applicability/generation/citation).

## Thêm testcase

1. Mở `evaluation/benchmark/benchmark.json` và thêm object theo schema trong [benchmark/README.md](benchmark/README.md).
2. Chọn `id` mới, không đổi ID của case cũ.
3. Dùng canonical domain/behavior key và tên văn bản đúng trong corpus.
4. Ghi tọa độ theo các mảng song song; dùng chuỗi rỗng khi ground truth chỉ tới cấp Điều.
5. Nhờ người có chuyên môn pháp lý review expected provisions.
6. Chạy:

```bash
python evaluation/run_benchmark.py --validate-only
python evaluation/run_benchmark.py --case-id <new_case_id>
```

## Đọc report

- Bắt đầu ở `benchmark_report.md` để xem overall, category, difficulty và 20 case thất bại nặng nhất.
- Dùng `benchmark_summary.json` cho CI, dashboard ngoài hoặc so sánh hai commit.
- Mở `benchmark_details.json` khi cần xem candidate, score, recursive provenance, Applicability, final context và citation của từng case.
- Các plot dùng cùng dữ liệu summary, không tính metric lần thứ hai.

Ví dụ một summary rút gọn:

```json
{
  "selected_cases": 30,
  "completed_cases": 30,
  "overall_metrics": {
    "domain_recall": 0.91,
    "behavior_recall": 0.86,
    "recall_at_5": 0.78,
    "recall_at_10": 0.87,
    "citation_accuracy": 0.94,
    "wrong_domain_rate": 0.08
  },
  "error_counts": {
    "Wrong Domain": 5,
    "Missing Relevant Law": 7,
    "Hallucinated Citation": 0
  }
}
```

Các số trên chỉ minh họa format, không phải kết quả đo của pipeline hiện tại.

## Regression và tính tái lập

Mỗi summary lưu hash SHA-256 của benchmark, Git commit/dirty state, mode, top-k, embedding model, Qdrant path và Ollama model. Khi so sánh trước/sau, phải giữ các trường này giống nhau. Nên lưu output từng thí nghiệm vào một thư mục riêng và không sửa ground truth trong cùng commit với thay đổi Retrieval.

Chạy unit test của metric engine:

```bash
python -m unittest discover -s evaluation/tests -v
```
