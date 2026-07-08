import sys
import logging
from pathlib import Path

# Thêm thư mục gốc vào path để import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_chatbot import build_llm, build_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')


def test_pipeline():
    """Smoke test nhanh: chit-chat + legal RAG (có Critic Agent). Dùng scripts/run_chatbot.py để test tương tác/so sánh A/B."""
    llm = build_llm()
    pipeline = build_pipeline(llm)

    print("\n--- TEST 1: CHIT-CHAT ---")
    chit_chat_query = "Xin chào, bạn là ai và có thể giúp tôi việc gì?"
    print(f"User: {chit_chat_query}")
    resp1 = pipeline.run(chit_chat_query)
    print(f"Bot:\n{resp1['final_response']}\n")

    print("\n--- TEST 2: LEGAL RAG (Kịch bản 3 — Critic Agent kiểm tra đầy đủ chính/phụ) ---")
    legal_query = "Việc phá hoại thông tin trên môi trường mạng bị xử lý như thế nào?"
    print(f"User: {legal_query}")
    try:
        resp2 = pipeline.run(legal_query, mode="critic")
        print(f"Bot:\n{resp2['final_response']}\n")
        print(f"Critic report: {resp2['critic_report']}\n")
    except Exception as e:
        print(f"Lỗi chạy RAG (có thể do Qdrant/Neo4j/Ollama chưa lên): {e}")


if __name__ == "__main__":
    test_pipeline()
