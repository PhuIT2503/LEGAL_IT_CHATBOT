import logging

from src.agents.agent_generation.state import GenerationState
from src.agents.agent_generation.prompts import format_context_block, build_answer_prompt
from src.agents.common.legal_relevance_filter import prepare_generation_context

logger = logging.getLogger(__name__)


def generate_draft_node(state: GenerationState, *, llm_client) -> dict:
    """
    Kịch bản 3, bước generation: sinh câu trả lời từ context đã
    recursive-retrieve đầy đủ (hoặc đã đánh dấu incomplete) — y hệt
    đầu vào của Kịch bản 1 (dùng chung nguyên văn prompt với generate_final_node
    qua build_answer_prompt — bắt buộc giống hệt nhau để 3 kịch bản chỉ khác
    nhau ở NỘI DUNG ngữ cảnh, không khác cách sinh câu trả lời).
    """
    logger.debug("Generation bắt đầu cho nhánh Critic.")
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

    resp = llm_client.invoke(prompt, tag="draft")
    return {"draft_response": resp.content, **relevance_update}
