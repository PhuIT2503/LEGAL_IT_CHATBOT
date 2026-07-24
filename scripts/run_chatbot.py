"""
scripts/run_chatbot.py
=======================
Chạy thử toàn bộ pipeline Chatbot (Hybrid Search + Critic Agent + LLM) end-to-end.

Yêu cầu trước khi chạy:
    1. docker compose up -d neo4j ollama
    2. docker compose --profile ingest up kg-ingest      (nạp Knowledge Graph)
    3. python src/data_ingestion/qdrant_local_ingest.py --data-dir data/keep --model data/ai_vietnamese_embedding_v2_finetuned_final
       (nạp Qdrant embedded — lưu file cục bộ tại data/.qdrant, không cần Qdrant server)
    4. docker exec legal_ollama ollama pull qwen2.5:7b

Cách dùng:
    python scripts/run_chatbot.py --query "Việc phá hoại thông tin trên môi trường mạng bị xử lý như thế nào?"
    python scripts/run_chatbot.py --query "..." --mode naive          # Kịch bản 1: RAG truyền thống (baseline)
    python scripts/run_chatbot.py --query "..." --mode article_expand # Kịch bản 2: mở rộng toàn Điều (không KG, không Critic Agent)
    python scripts/run_chatbot.py --query "..." --mode critic         # Kịch bản 3: Critic Agent (mặc định, đề xuất khóa luận)
    python scripts/run_chatbot.py --query "..." --compare-all         # chạy cả 3 kịch bản để so sánh
    python scripts/run_chatbot.py                                     # chế độ nhập tương tác

So sánh với embedding GỐC (chưa fine-tune) — không cần sửa code, chỉ set 2 biến môi
trường trước khi chạy (build_pipeline() tự đọc, mặc định vẫn dùng bản fine-tune nếu
không set):
    QDRANT_PATH=data/.qdrant_base EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding_v2 \\
        python scripts/run_chatbot.py --query "..."
    (data/.qdrant_base phải được ingest riêng trước bằng model gốc, xem
    src/data_ingestion/qdrant_local_ingest.py --model AITeamVN/Vietnamese_Embedding_v2
    --db-path data/.qdrant_base — KHÔNG dùng chung data/.qdrant với bản fine-tune vì
    2 model có không gian embedding khác nhau, lẫn vào sẽ sai kết quả retrieval.)
"""

import os
import sys
import logging
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI
from src.workflow.pipeline import ChatbotWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_chatbot")


def build_llm() -> ChatOpenAI:
    """LLM mặc định: Qwen2.5-7B chạy qua Ollama (OpenAI-compatible API), không cần API key."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    logger.info(f"LLM: {model} @ {base_url}")
    return ChatOpenAI(model=model, base_url=base_url, api_key="ollama", temperature=0.2)


def build_pipeline(llm, top_k: int = 5, skip_router: bool = False) -> ChatbotWorkflow:
    # QDRANT_PATH/EMBEDDING_MODEL cho phép trỏ sang 1 index/model KHÁC (vd so sánh
    # embedding gốc chưa fine-tune) mà không cần sửa code — mặc định vẫn là Qdrant
    # + embedding fine-tune chuẩn của repo nếu không set 2 biến môi trường này.
    return ChatbotWorkflow(
        llm=llm,
        # Qdrant chạy ở chế độ embedded/local-file (data/.qdrant trong repo) để dễ
        # đóng gói/chuyển dự án — chỉ set QDRANT_URL nếu thật sự có Qdrant server riêng.
        qdrant_url=os.getenv("QDRANT_URL") or None,
        qdrant_path=os.getenv("QDRANT_PATH", str(PROJECT_ROOT / "data" / ".qdrant")),
        embedding_model_name=os.getenv("EMBEDDING_MODEL", "data/ai_vietnamese_embedding_v2_finetuned_final"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pass=os.getenv("NEO4J_PASSWORD", "legal_kg_2024"),
        top_k=top_k,
        skip_router=skip_router,
    )


MODE_LABEL = {
    "naive": "Kịch bản 1: RAG truyền thống (baseline, không graph)",
    "article_expand": "Kịch bản 2: mở rộng toàn Điều (không dùng quan hệ KG, không có Critic Agent)",
    "critic": "Kịch bản 3: Critic Agent (đề xuất khóa luận)",
}


def ask(pipeline: ChatbotWorkflow, query: str, mode: str):
    result = pipeline.run(query, mode=mode)

    print(f"\n{'='*90}")
    print(f"Câu hỏi: {query}")
    print(f"Chế độ: {MODE_LABEL.get(mode, mode)}")
    print(f"Số chunk retrieved: {len(result['retrieved_chunks'])} | Số Điều: {len(result['retrieved_dieu_ids'])}")
    print(f"{'='*90}")

    if mode == "critic" and result.get("draft_response"):
        print(f"\n--- CÂU TRẢ LỜI NHÁP (trước khi Critic Agent kiểm tra) ---\n{result['draft_response']}")

    print(f"\n--- TRẢ LỜI CUỐI ---\n{result['final_response']}")

    if mode == "critic":
        report = result.get("critic_report") or {}
        if report:
            print(f"\n--- CRITIC REPORT (is_complete={report.get('is_complete')}) ---")
            for s in report.get("suggestions", []):
                print(f"  [{s['action']}] {s['reason']}")
    elif mode == "article_expand" and result.get("graph_context"):
        print(f"\n--- NGỮ CẢNH MỞ RỘNG (toàn văn Điều, {len(result['graph_context'])} ký tự) ---")
    print()
    return result


def main():
    parser = argparse.ArgumentParser(description="Chạy thử Chatbot pipeline (RAG + Critic Agent) end-to-end")
    parser.add_argument("--query", default=None, help="Câu hỏi; bỏ trống để vào chế độ nhập tương tác")
    parser.add_argument("--mode", choices=["naive", "article_expand", "critic"], default="critic",
                         help="Kịch bản chạy: naive (1) | article_expand (2) | critic (3, mặc định)")
    parser.add_argument("--compare-all", action="store_true", help="Chạy cả 3 kịch bản trên cùng 1 câu hỏi để so sánh")
    parser.add_argument("--top-k", type=int, default=5, help="Số chunk lấy về từ hybrid search (mặc định 5)")
    args = parser.parse_args()

    llm = build_llm()
    pipeline = build_pipeline(llm, top_k=args.top_k)

    if args.query:
        if args.compare_all:
            for mode in ("naive", "article_expand", "critic"):
                ask(pipeline, args.query, mode=mode)
        else:
            ask(pipeline, args.query, mode=args.mode)
        return

    print("Chế độ tương tác — nhập câu hỏi (Ctrl+C để thoát):")
    while True:
        try:
            q = input("\n> ").strip()
            if not q:
                continue
            ask(pipeline, q, mode=args.mode)
        except (KeyboardInterrupt, EOFError):
            print("\nThoát.")
            break


if __name__ == "__main__":
    main()
