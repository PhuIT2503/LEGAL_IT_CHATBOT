import logging

from langchain_core.messages import AIMessage

from src.agents.agent_generation.state import GenerationState
from src.agents.agent_generation.prompts import format_context_block, build_answer_prompt
from src.agents.common.legal_relevance_filter import prepare_generation_context

logger = logging.getLogger(__name__)


def generate_final_node(state: GenerationState, *, llm_client) -> dict:
    """
    Sinh câu trả lời 1 LẦN DUY NHẤT từ context_texts (+ graph_context nếu có,
    dùng cho Kịch bản 2 — graph_context rỗng ở Kịch bản 1 nên hành vi y hệt
    RAG thuần; cũng dùng lại ở bước regenerate của Kịch bản 3).
    """
    logger.debug("Generation bắt đầu.")
    query = state["query"]
    progress = state.get("progress_callback")
    if progress:
        progress("analyze")
    context_texts, relevance_update = prepare_generation_context(
        state,
        llm_client=llm_client,
    )

    context_text = format_context_block(context_texts, query=query)
    prompt = build_answer_prompt(
        query,
        context_text,
        is_complete=state.get("retrieval_is_complete", True),
    )

    if progress:
        progress("write")

    resp = llm_client.invoke(prompt, tag="final_generate")
    return {
        "final_response": resp.content,
        "messages": [AIMessage(content=resp.content)],
        **relevance_update,
    }
