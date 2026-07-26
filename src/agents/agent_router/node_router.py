import logging

from src.agents.agent_router.state import RouterState

logger = logging.getLogger(__name__)


def route_query_node(state: RouterState, *, llm_client) -> dict:
    logger.debug("Routing query...")
    query = state["query"]

    prompt = (
        "Bạn là trợ lý ảo. Hãy phân loại câu hỏi sau đây thuộc loại nào:\n"
        "1. 'chit_chat': Những câu chào hỏi, cảm ơn, hỏi thăm thông thường.\n"
        "2. 'legal': Câu hỏi về pháp luật, hình phạt, luật thương mại, an ninh mạng, CNTT, v.v.\n\n"
        f"Câu hỏi: {query}\n\n"
        "Chỉ trả về đúng 1 từ: 'chit_chat' hoặc 'legal'."
    )

    resp = llm_client.invoke(prompt, tag="router")
    content = resp.content.strip().lower()

    is_chit_chat = "chit_chat" in content
    logger.debug("Routed as: %s", "chit_chat" if is_chit_chat else "legal")

    return {"is_chit_chat": is_chit_chat}
