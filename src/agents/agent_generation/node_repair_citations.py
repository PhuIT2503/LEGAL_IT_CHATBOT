import logging

from langchain_core.messages import AIMessage

from src.agents.agent_generation.citation_coverage import (
    find_missing_required_citations,
    format_required_citations,
)
from src.agents.agent_generation.prompts import format_context_block

logger = logging.getLogger(__name__)


def repair_citations_node(state: dict, *, llm_client) -> dict:
    """Repair a final answer once when it omits a mandatory KG citation."""
    required = state.get("required_citations", [])
    answer = state.get("final_response", "")
    missing = find_missing_required_citations(answer, required)
    if not missing:
        return {"missing_required_citations": []}

    logger.info(
        "Critic Agent: câu trả lời cuối thiếu căn cứ bắt buộc %s — repair một lần.",
        ", ".join(item["label"] for item in missing),
    )

    context_texts = list(state.get("context_texts", []))
    graph_context = state.get("graph_context", "")
    if graph_context:
        context_texts.append(graph_context)
    context_text = format_context_block(context_texts)

    prompt = (
        "Bạn đang sửa một câu trả lời pháp lý đã có. Giữ nguyên mọi nội dung đúng, không rút gọn làm mất "
        "ý và không thêm suy đoán ngoài ngữ cảnh. Câu trả lời hiện thiếu một hoặc nhiều căn cứ pháp lý mà "
        "Critic đã xác nhận là bắt buộc. Hãy bổ sung rõ từng Điều còn thiếu, giải thích vai trò của Điều đó "
        "và quan hệ dẫn chiếu với Điều còn lại. Chỉ trả về câu trả lời hoàn chỉnh sau khi sửa.\n\n"
        f"{format_required_citations(missing)}\n\n"
        f"================ NGỮ CẢNH ================\n{context_text}\n\n"
        f"================ CÂU TRẢ LỜI HIỆN TẠI ================\n{answer}\n\n"
        f"Câu hỏi gốc: {state.get('query', '')}"
    )
    resp = llm_client.invoke(prompt, tag="citation_repair")
    repaired = resp.content
    remaining = find_missing_required_citations(repaired, required)
    if remaining:
        logger.warning(
            "Citation repair vẫn thiếu: %s",
            ", ".join(item["label"] for item in remaining),
        )
    return {
        "final_response": repaired,
        "missing_required_citations": remaining,
        "messages": [AIMessage(content=repaired)],
    }
