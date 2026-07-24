import logging

from src.agents.agent_generation.state import GenerationState
from src.agents.agent_generation.prompts import format_context_block, build_answer_prompt

logger = logging.getLogger(__name__)


def generate_draft_node(state: GenerationState, *, llm_client) -> dict:
    """
    Kịch bản 3, bước 1: sinh câu trả lời NHÁP chỉ từ top-k chunk thô — y hệt
    đầu vào của Kịch bản 1 (dùng chung nguyên văn prompt với generate_final_node
    qua build_answer_prompt — bắt buộc giống hệt nhau để 3 kịch bản chỉ khác
    nhau ở NỘI DUNG ngữ cảnh, không khác cách sinh câu trả lời).
    """
    logger.info("Critic Agent: sinh câu trả lời nháp từ top-k thuần...")
    query = state["query"]
    context_text = format_context_block(state.get("context_texts", []))
    prompt = build_answer_prompt(query, context_text)

    resp = llm_client.invoke(prompt, tag="draft")
    return {"draft_response": resp.content}
