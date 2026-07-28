import logging

from langchain_core.messages import AIMessage

from src.agents.agent_generation.state import GenerationState
from src.agents.agent_generation.prompts import (
    build_answer_prompt,
    build_generation_payload,
    format_context_block,
)
from src.agents.agent_generation.answer_assessment import build_answer_assessment
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

    effective_is_complete = relevance_update.get(
        "retrieval_is_complete",
        state.get("retrieval_is_complete", True),
    )
    answer_assessment = build_answer_assessment(
        query=query,
        behavior_profile=state.get("behavior_profile"),
        retrieval_decisions=relevance_update.get("retrieval_decisions", ()),
        context_texts=context_texts,
        final_context_records=relevance_update.get("context_records", ()),
        retrieval_is_complete=effective_is_complete,
        scenario_fact_state=relevance_update.get("scenario_fact_state"),
    )
    context_text = format_context_block(context_texts, query=query)
    applicability_results = [
        decision
        for decision in relevance_update.get("retrieval_decisions", ())
        if decision.get("decision_stage") == "applicability"
    ]
    generation_payload = build_generation_payload(
        query=query,
        context_text=context_text,
        is_complete=effective_is_complete,
        scenario_fact_state=relevance_update.get("scenario_fact_state", {}),
        behavior_profile=state.get("behavior_profile"),
        applicability_results=applicability_results,
        answer_assessment=answer_assessment,
    )
    prompt = build_answer_prompt(
        query,
        context_text,
        is_complete=effective_is_complete,
        generation_payload=generation_payload,
    )

    if progress:
        progress("write")

    resp = llm_client.invoke(prompt, tag="final_generate")
    return {
        "final_response": resp.content,
        "messages": [AIMessage(content=resp.content)],
        "answer_assessment": answer_assessment,
        "generation_payload": generation_payload,
        **relevance_update,
    }
