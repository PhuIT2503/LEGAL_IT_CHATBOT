import logging

from langchain_core.messages import AIMessage

from src.agents.agent_router.state import RouterState

logger = logging.getLogger(__name__)


def chit_chat_node(state: RouterState, *, llm_client) -> dict:
    logger.info("Handling chit-chat...")
    query = state["query"]

    prompt = (
        "Bạn là trợ lý ảo pháp luật nhiệt tình, thân thiện. "
        "Hãy trả lời tin nhắn sau của người dùng một cách ngắn gọn, lịch sự:\n\n"
        f"Người dùng: {query}"
    )
    progress = state.get("progress_callback")
    if progress:
        progress("write")
    resp = llm_client.invoke(prompt, tag="chit_chat")

    return {"final_response": resp.content, "messages": [AIMessage(content=resp.content)]}
