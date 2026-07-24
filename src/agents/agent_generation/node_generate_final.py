import logging

from langchain_core.messages import AIMessage

from src.agents.agent_generation.state import GenerationState
from src.agents.agent_generation.prompts import format_context_block, build_answer_prompt

logger = logging.getLogger(__name__)


def generate_final_node(state: GenerationState, *, llm_client) -> dict:
    """
    Sinh câu trả lời 1 LẦN DUY NHẤT từ context_texts (+ graph_context nếu có,
    dùng cho Kịch bản 2 — graph_context rỗng ở Kịch bản 1 nên hành vi y hệt
    RAG thuần; cũng dùng lại ở bước regenerate của Kịch bản 3).
    """
    logger.info("Generating single-pass response...")
    query = state["query"]
    context_texts = list(state.get("context_texts", []))
    graph_context = state.get("graph_context", "")

    if graph_context:
        context_texts.append(graph_context)

    context_text = format_context_block(context_texts)
    prompt = build_answer_prompt(query, context_text)

    resp = llm_client.invoke(prompt, tag="final_generate")
    return {"final_response": resp.content, "messages": [AIMessage(content=resp.content)]}
