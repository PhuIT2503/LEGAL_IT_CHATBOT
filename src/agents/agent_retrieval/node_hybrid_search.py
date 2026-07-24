import logging

from src.retrieval.qdrant_hybrid_search import hybrid_search
from src.knowledge_graph.graph_builder import to_dieu_node_id
from src.agents.common.focus_dieu import compute_focus_dieu_ids
from src.agents.agent_retrieval.state import RetrievalState

logger = logging.getLogger(__name__)


def hybrid_search_node(
    state: RetrievalState,
    *,
    qdrant_client,
    embedding_model,
    bm25,
    qdrant_child_col: str,
    qdrant_parent_col: str,
    top_k: int,
    prefetch_limit: int,
    article_expand_score_ratio: float,
) -> dict:
    """
    RAG THUẦN — chỉ lấy đúng top-k chunk, KHÔNG expand, KHÔNG dùng KG.
    Dùng chung cho cả 3 kịch bản để đảm bảo so sánh công bằng.

    Trả về 2 danh sách Điều RIÊNG BIỆT, phục vụ 2 mục đích khác nhau:
    - retrieved_dieu_ids/dieu_scores: suy TRỰC TIẾP từ ĐÚNG top-k chunk
      (như thiết kế gốc, KHÔNG đổi) — dùng cho naive (MRR/nDCG chung) và
      critic (compute_focus_dieu_ids) — 2 kịch bản này giữ nguyên hành vi
      cũ hoàn toàn.
    - article_expand_dieu_ids: suy từ 1 tập chunk RỘNG HƠN top-k
      (wide_chunks), lọc qua compute_focus_dieu_ids với
      article_expand_score_ratio rồi cap tối đa top_k — CHỈ dùng riêng cho
      expand_full_article. Lý do tách riêng: nếu suy Điều trực tiếp từ
      top-k CHUNK, nhiều chunk top-k có thể trùng vào CÙNG 1 Điều (vd top-5
      chunk nhưng chỉ thuộc 2 Điều khác nhau, do 1 Điều nhiều Khoản cùng
      khớp câu hỏi) — khiến article_expand chỉ có 2 Điều để "mở rộng toàn
      Điều" dù top_k=5, trái đúng bản chất baseline này (mở rộng lên K ĐIỀU
      khác nhau, không phải K chunk). BẮT BUỘC lọc theo tỷ lệ điểm (KHÔNG
      lấy mù top_k đầu tiên của tập rộng) — đã quan sát thực tế lấy mù
      khiến article_expand dính phải Điều LẠC HẲN chủ đề (từ Nghị định/Luật
      khác hoàn toàn) do độ liên quan giảm rất nhanh trong tập 20 chunk, sập
      điểm hoàn toàn xuống dưới cả naive.
    """
    logger.info("Retrieving documents via Hybrid Search (top-k thuần, không expand)...")
    query = state["query"]

    # Tập chunk RỘNG hơn top_k, CHỈ dùng để suy article_expand_dieu_ids —
    # không vượt quá prefetch_limit (giới hạn candidate pool thật sự có
    # sẵn để xếp hạng, hỏi rộng hơn cũng không có thêm ứng viên mới).
    wide_limit = min(top_k * 4, prefetch_limit)

    try:
        result = hybrid_search(
            query=query,
            child_collection=qdrant_child_col,
            parent_collection=qdrant_parent_col,
            limit=wide_limit,
            prefetch_limit=prefetch_limit,
            fusion="rrf",
            include_parent=False,
            client=qdrant_client,
            model=embedding_model,
            bm25=bm25,
        )
        wide_hits = result.get("children", [])
    except Exception as e:
        logger.warning(f"Qdrant search failed (Qdrant có thể chưa ingest dữ liệu): {e}")
        wide_hits = []

    wide_chunks = []
    for p in wide_hits:
        payload = getattr(p, "payload", {}) or {}
        wide_chunks.append({
            "chunk_id": payload.get("id"),
            "text": payload.get("content", ""),
            "dieu_id_raw": payload.get("dieu_id", ""),
            "van_ban_id_raw": payload.get("van_ban_id", ""),
            "score": getattr(p, "score", None),
        })

    # retrieved_chunks/context_texts: ĐÚNG top-k chunk thuần — GIỮ NGUYÊN
    # như thiết kế gốc (cắt từ đầu tập rộng, thứ tự score giảm dần không đổi).
    retrieved_chunks = wide_chunks[:top_k]
    context_texts = [c["text"] for c in retrieved_chunks if c["text"]]

    # retrieved_dieu_ids/dieu_scores: suy TỪ ĐÚNG top-k chunk (KHÔNG đổi
    # so với thiết kế gốc) — dùng cho naive (MRR/nDCG) và critic
    # (compute_focus_dieu_ids), GIỮ NGUYÊN hành vi cũ.
    dieu_best_score: dict = {}
    dieu_order: list = []
    for c in retrieved_chunks:
        d = to_dieu_node_id(c["van_ban_id_raw"], c["dieu_id_raw"])
        if not d:
            continue
        score = c["score"] or 0.0
        if d not in dieu_best_score:
            dieu_order.append(d)
            dieu_best_score[d] = score
        else:
            dieu_best_score[d] = max(dieu_best_score[d], score)
    retrieved_dieu_ids = dieu_order

    # article_expand_dieu_ids: thu thập TOÀN BỘ Điều khác nhau trong tập
    # rộng (không cắt sớm) cùng điểm tốt nhất, rồi lọc qua
    # compute_focus_dieu_ids với article_expand_score_ratio (chặt hơn
    # critic_score_ratio) + cap tối đa top_k. KHÔNG lấy mù toàn bộ top_k
    # đầu tiên của tập rộng — độ liên quan giảm nhanh trong tập 20 chunk,
    # lấy mù sẽ dính phải Điều lạc hẳn chủ đề (đã quan sát thực tế).
    wide_dieu_best_score: dict = {}
    wide_dieu_order: list = []
    for c in wide_chunks:
        d = to_dieu_node_id(c["van_ban_id_raw"], c["dieu_id_raw"])
        if not d:
            continue
        score = c["score"] or 0.0
        if d not in wide_dieu_best_score:
            wide_dieu_order.append(d)
            wide_dieu_best_score[d] = score
        else:
            wide_dieu_best_score[d] = max(wide_dieu_best_score[d], score)
    article_expand_dieu_ids = compute_focus_dieu_ids(
        wide_dieu_order, wide_dieu_best_score,
        score_ratio=article_expand_score_ratio, max_dieu=top_k,
    )

    logger.info(
        f"Retrieved {len(retrieved_chunks)} chunk top-k -> {len(retrieved_dieu_ids)} Điều (naive/critic); "
        f"article_expand: {len(article_expand_dieu_ids)} Điều khác nhau (quét rộng {len(wide_chunks)} chunk)."
    )
    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieved_dieu_ids": retrieved_dieu_ids,
        "dieu_scores": dieu_best_score,
        "context_texts": context_texts,
        "article_expand_dieu_ids": article_expand_dieu_ids,
    }
