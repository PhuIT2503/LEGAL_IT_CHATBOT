import logging

from src.agents.agent_article_expand.state import ArticleExpandState

logger = logging.getLogger(__name__)


def expand_article_node(state: ArticleExpandState, *, dieu_content_store) -> dict:
    """
    Kịch bản 2 (RAG mở rộng toàn Điều, KHÔNG dùng quan hệ Knowledge Graph,
    KHÔNG có Critic Agent) — dùng để SO SÁNH, KHÔNG phải cơ chế đề xuất.

    Mở rộng article_expand_dieu_ids — đủ top_k Điều KHÁC NHAU, suy từ 1
    tập chunk RỘNG hơn top-k thuần rồi lọc qua article_expand_score_ratio
    (xem agent_retrieval) — KHÔNG dùng retrieved_dieu_ids (tập hẹp suy từ
    ĐÚNG top-k chunk, dành riêng cho naive/critic) và KHÔNG dùng
    critic_score_ratio/critic_max_dieu (đó là ngưỡng riêng của agent_critic).
    article_expand vẫn CÓ lọc theo tỷ lệ điểm — KHÔNG lấy mù hoàn toàn (đã
    thử và bỏ: lấy mù top_k Điều đầu tiên của tập rộng, không lọc gì, khiến
    baseline dính phải Điều lạc hẳn chủ đề từ Nghị định/Luật khác, sập điểm
    xuống dưới cả naive — 0.642 -> 0.15 trên cat1) — chỉ khác critic ở việc
    KHÔNG có bước "phát hiện thiếu qua KG" và dùng cap riêng (bằng top_k
    thay vì critic_max_dieu thấp hơn).

    Với MỖI Điều, chỉ lấy TOÀN VĂN CHÍNH Điều đó (qua Qdrant parent chunk)
    — KHÔNG đi theo bất kỳ quan hệ nào trong Knowledge Graph (không
    THAM_CHIEU sang Điều khác, không dò HanhVi/CheTai/ChuThe/...). Đây
    CHÍNH LÀ điểm khác biệt cốt lõi với Kịch bản 3: baseline này chỉ biết
    "trong phạm vi retrieval, Điều nào cần xem đầy đủ", còn Critic Agent
    còn biết dùng KG để tìm thông tin NẰM NGOÀI phạm vi đó (Điều khác
    được tham chiếu, chế tài phụ nằm ở Khoản/Điều khác).
    """
    logger.info("Mở rộng toàn văn Điều trong phạm vi (không dùng quan hệ KG, không lọc thêm)...")
    retrieved_dieu_ids = state.get("article_expand_dieu_ids", [])
    if not retrieved_dieu_ids:
        return {"graph_context": "", "graph_fetched_dieu_ids": []}

    graph_text = ""
    for dieu_id in retrieved_dieu_ids:
        content = dieu_content_store.fetch_parent_content(dieu_id)
        if content:
            graph_text += f"[Toàn văn Điều {dieu_id}]\n{content}\n\n"

    logger.info(f"Full-article expand: {len(graph_text)} ký tự, {len(retrieved_dieu_ids)} Điều.")
    return {"graph_context": graph_text, "graph_fetched_dieu_ids": retrieved_dieu_ids}
