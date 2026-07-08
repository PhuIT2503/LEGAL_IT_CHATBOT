"""
scripts/run_evaluation.py
==========================
Chạy bộ test đánh giá (data/eval_testset.jsonl, 50 câu — 4 nhóm: chế tài kép,
cross-Điều, structural đa Khoản, control) qua pipeline thật (naive/article_expand/
critic), thu thập kết quả ở định dạng SẴN SÀNG cho RAGAS.

Yêu cầu môi trường (giống hệt run_chatbot.py):
    - Neo4j đang chạy (NEO4J_URI, mặc định bolt://localhost:7687)
    - Ollama đang chạy với model qwen2.5:7b (OLLAMA_BASE_URL)
    - Qdrant embedded đã ingest xong tại data/.qdrant

Cách dùng:
    python scripts/run_evaluation.py --mode critic
    python scripts/run_evaluation.py --mode naive --resume
    python scripts/run_evaluation.py --all-modes          # chạy cả 3 mode, tuần tự
    python scripts/run_evaluation.py --mode critic --limit 5   # test nhanh 5 câu đầu

Output: data/eval_results_<mode>.jsonl — mỗi dòng 1 JSON với các field:
    id, category, question (user_input), answer (response), reference,
    retrieved_contexts (list[str] — dùng cho RAGAS context_precision/recall),
    required_facts, mode, critic_report (chỉ có ở mode critic)
Định dạng field đặt tên theo chuẩn RAGAS (user_input/response/reference/
retrieved_contexts) để np.array hóa trực tiếp thành ragas.EvaluationDataset.
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_chatbot import build_llm, build_pipeline  # noqa: E402

TESTSET_PATH = PROJECT_ROOT / "data" / "eval_testset.jsonl"


def load_testset(testset_path: Path = TESTSET_PATH):
    cases = []
    with open(testset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_done_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def run_mode(pipeline, cases, mode: str, resume: bool, limit: int | None, output_suffix: str = ""):
    output_path = PROJECT_ROOT / "data" / f"eval_results_{mode}{output_suffix}.jsonl"
    done_ids = load_done_ids(output_path) if resume else set()

    cases_to_run = [c for c in cases if c["id"] not in done_ids]
    if limit:
        cases_to_run = cases_to_run[:limit]

    print(f"\n=== Mode: {mode} — {len(cases_to_run)}/{len(cases)} câu cần chạy "
          f"({len(done_ids)} đã có sẵn nếu --resume) ===")

    mode_write = "a" if resume and output_path.exists() else "w"
    with open(output_path, mode_write, encoding="utf-8") as out_f:
        for i, case in enumerate(cases_to_run, 1):
            print(f"[{i}/{len(cases_to_run)}] {case['id']} ({case['category']}): {case['question'][:70]}...")
            try:
                result = pipeline.run(case["question"], mode=mode)
            except Exception as e:
                print(f"  LỖI: {e}")
                continue

            # Ngữ cảnh dùng để đánh giá (RAGAS context_precision/recall cần list[str]):
            # với critic, gộp cả context_texts gốc lẫn graph_context (đúng những gì
            # thực sự được đưa vào prompt sinh câu trả lời cuối).
            retrieved_contexts = [c["text"] for c in result.get("retrieved_chunks", []) if c.get("text")]
            if result.get("graph_context"):
                retrieved_contexts.append(result["graph_context"])

            row = {
                "id": case["id"],
                "category": case["category"],
                "mode": mode,
                "user_input": case["question"],
                "response": result.get("final_response", ""),
                "reference": case.get("reference_answer", ""),
                "retrieved_contexts": retrieved_contexts,
                "required_facts": case.get("required_facts", []),
                "dieu_ids": case.get("dieu_ids", []),
                # Thứ tự Điều theo retrieval gốc (top-k thuần, CHUNG cho cả 3 mode —
                # dùng để tính MRR/nDCG của bước retrieval, không phụ thuộc mode).
                "retrieved_dieu_ids_ranked": result.get("retrieved_dieu_ids", []),
                # Điều được article_expand/critic fetch thêm ngoài top-k (rỗng ở naive) —
                # dùng để tính context precision/recall trên ngữ cảnh CUỐI CÙNG đưa vào LLM.
                "graph_fetched_dieu_ids": result.get("graph_fetched_dieu_ids", []),
                # Tổng token (prompt+completion) qua MỌI lệnh gọi LLM trong 1 câu hỏi —
                # dùng để so sánh CHI PHÍ CẢ PIPELINE giữa 3 kịch bản (xem _invoke_llm).
                "token_usage": result.get("token_usage", {}),
                # Token của ĐÚNG lệnh gọi sinh ra câu trả lời cuối cùng (không cộng
                # router/gate/draft-bị-bỏ) — chỉ số đúng cho "hiệu quả ngữ cảnh" khi
                # so sánh 3 kịch bản (xem run() trong pipeline).
                "final_answer_token_usage": result.get("final_answer_token_usage", {}),
            }
            if mode == "critic":
                row["draft_response"] = result.get("draft_response", "")
                row["critic_report"] = result.get("critic_report", {})

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"Đã lưu: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Chạy bộ test đánh giá qua pipeline thật")
    parser.add_argument("--mode", choices=["naive", "article_expand", "critic"], default=None)
    parser.add_argument("--all-modes", action="store_true", help="Chạy tuần tự cả 3 mode")
    parser.add_argument("--resume", action="store_true", help="Bỏ qua case đã có trong file output (theo id)")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N case đầu tiên (test nhanh)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--testset", type=str, default=None, help="Đường dẫn file testset .jsonl khác (mặc định data/eval_testset.jsonl)")
    parser.add_argument("--output-suffix", type=str, default="", help="Hậu tố thêm vào tên file output (vd '_stratified10') để không ghi đè eval_results_<mode>.jsonl")
    args = parser.parse_args()

    if not args.mode and not args.all_modes:
        parser.error("Phải chỉ định --mode hoặc --all-modes")

    testset_path = Path(args.testset) if args.testset else TESTSET_PATH
    cases = load_testset(testset_path)
    print(f"Đã load {len(cases)} câu hỏi test từ {testset_path}")

    llm = build_llm()
    pipeline = build_pipeline(llm, top_k=args.top_k)

    modes = ["naive", "article_expand", "critic"] if args.all_modes else [args.mode]
    for mode in modes:
        run_mode(pipeline, cases, mode, args.resume, args.limit, args.output_suffix)


if __name__ == "__main__":
    main()
