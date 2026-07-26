import logging

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def finalize_draft_node(state: dict) -> dict:
    """Critic không phát hiện thiếu gì -> câu trả lời nháp đã đủ, dùng thẳng làm câu trả lời cuối.

    Glue thuần túy của workflow (2 dòng, không LLM, không KG) — không phải 1 agent riêng.
    """
    logger.debug("Critic Agent: dùng kết quả đã sinh từ context recursive retrieval.")
    draft = state.get("draft_response", "")
    return {"final_response": draft, "messages": [AIMessage(content=draft)]}
