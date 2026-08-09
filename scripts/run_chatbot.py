"""
scripts/run_chatbot.py
=======================
Chạy thử toàn bộ pipeline Chatbot (Hybrid Search + Critic Agent + LLM) end-to-end.

Yêu cầu trước khi chạy:
    1. docker compose --profile app up -d app
       (tự kéo theo: neo4j + kg-ingest nạp Knowledge Graph, qdrant +
        qdrant-ingest nạp 4 collection vector từ data/keep — cổng 6333)
    2. CHỈ khi muốn dùng Qwen2.5 7B local (không bắt buộc):
       docker compose --profile ollama up -d ollama
       docker exec legal_ollama ollama pull qwen2.5:7b

Cách dùng:
    python scripts/run_chatbot.py --query "Việc phá hoại thông tin trên môi trường mạng bị xử lý như thế nào?"
    python scripts/run_chatbot.py --query "..." --mode naive          # Kịch bản 1: RAG truyền thống (baseline)
    python scripts/run_chatbot.py --query "..." --mode article_expand # Kịch bản 2: mở rộng toàn Điều (không KG, không Critic Agent)
    python scripts/run_chatbot.py --query "..." --mode critic         # Kịch bản 3: Critic Agent (mặc định, đề xuất khóa luận)
    python scripts/run_chatbot.py --query "..." --compare-all         # chạy cả 3 kịch bản để so sánh
    python scripts/run_chatbot.py                                     # chế độ nhập tương tác

Đổi sang model embedding khác — không cần sửa code, chỉ set biến môi trường trước
khi chạy (build_pipeline() tự đọc). Mỗi model có 1 CẶP collection riêng trên cùng
container qdrant, KHÔNG được dùng lẫn vì 2 không gian embedding khác dimension:
    QDRANT_CHILD_COLLECTION=legal_child_chunks_gte \\
    QDRANT_PARENT_COLLECTION=legal_parent_chunks_gte \\
    EMBEDDING_MODEL=Alibaba-NLP/gte-multilingual-base \\
        python scripts/run_chatbot.py --query "..."

Chạy offline không có Qdrant server (vd Colab, chỉ còn thư mục embedded cũ):
    QDRANT_URL= QDRANT_PATH=data/.qdrant_base python scripts/run_chatbot.py --query "..."
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
    # Giữ khớp với app.py (LLM_TEMPERATURE) — đánh giá và chat tương tác phải
    # dùng CÙNG cấu hình sinh, nếu không số đo không phản ánh hành vi thật.
    return ChatOpenAI(model=model, base_url=base_url, api_key="ollama", temperature=0.2)


def build_pipeline(llm, top_k: int = 5, skip_router: bool = False) -> ChatbotWorkflow:
    # Các biến QDRANT_* / EMBEDDING_MODEL cho phép trỏ sang cặp collection/model
    # KHÁC (vd so sánh 2 model embedding) mà không cần sửa code.
    #
    # 2 chế độ, tự nhận biết:
    # - Mặc định (máy có Docker): đọc Qdrant server ở container qdrant, collection
    #   có hậu tố theo model embedding, BM25 index trong data/bm25/.
    # - Embedded/local-file (Colab, không có server): chỉ cần set QDRANT_PATH trỏ
    #   vào thư mục .qdrant* đã upload sẵn — tự chuyển về tên collection cũ
    #   (không hậu tố) và đọc BM25 index ngay trong thư mục đó, đúng như trước
    #   khi chuyển sang Docker. Set QDRANT_URL="" cũng cho kết quả tương tự.
    qdrant_path = os.getenv("QDRANT_PATH")
    default_url = "" if qdrant_path else "http://localhost:6333"
    qdrant_url = os.getenv("QDRANT_URL", default_url) or None
    suffix = "" if qdrant_url is None else "_base"
    # Model embedding mặc định phải khớp ĐÚNG index đang dùng: collection *_base
    # trên server được ingest bằng Vietnamese_Embedding_v2, còn thư mục
    # data/.qdrant cũ (chế độ embedded) là bản fine-tune.
    default_embedding = (
        "data/ai_vietnamese_embedding_v2_finetuned_final" if qdrant_url is None
        else "AITeamVN/Vietnamese_Embedding_v2"
    )

    return ChatbotWorkflow(
        llm=llm,
        qdrant_url=qdrant_url,
        qdrant_path=qdrant_path or str(PROJECT_ROOT / "data" / ".qdrant"),
        qdrant_child_col=os.getenv("QDRANT_CHILD_COLLECTION", f"legal_child_chunks{suffix}"),
        qdrant_parent_col=os.getenv("QDRANT_PARENT_COLLECTION", f"legal_parent_chunks{suffix}"),
        # Chế độ embedded: BM25 index nằm ngay trong thư mục Qdrant (bm25_dir=None
        # -> ChatbotWorkflow tự lùi về qdrant_path).
        bm25_dir=os.getenv("BM25_DIR") or (None if qdrant_url is None else str(PROJECT_ROOT / "data" / "bm25")),
        embedding_model_name=os.getenv("EMBEDDING_MODEL", default_embedding),
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
