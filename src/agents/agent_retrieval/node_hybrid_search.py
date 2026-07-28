import logging
import os

from src.retrieval.qdrant_hybrid_search import hybrid_search_in_domains as hybrid_search
from src.knowledge_graph.graph_builder import to_dieu_node_id
from src.agents.common.focus_dieu import compute_focus_dieu_ids
from src.agents.common.legal_response import sort_context_records
from src.agents.common.legal_scenario_facts import extract_legal_scenario_facts
from src.agents.common.retrieval_ranking import (
    deduplicate_context_records,
    filter_behavior_aware_relevant,
    filter_behaviorally_relevant,
    filter_semantically_relevant,
    rerank_context_records_with_behavior,
    select_balanced_top_k,
)
from src.agents.common.query_expansion import expand_legal_query
from src.agents.common.retrieval_contract import (
    annotate_retrieval_contract_records,
    assess_retrieval_contract,
    build_retrieval_rescue_query,
    retrieval_contract_roles,
)
from src.agents.common.retrieval_provenance import normalise_provenance_record
from src.agents.agent_retrieval.state import RetrievalState
from src.retrieval.legal_domains import (
    count_candidates_by_domain,
    document_legal_domains,
    select_legal_domains,
)
from src.retrieval.legal_behaviors import extract_legal_behavior
from src.retrieval.legal_event import extract_canonical_legal_event

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
    cross_encoder_reranker=None,
) -> dict:
    """
    Candidate generation dùng hybrid search cho query gốc và query expansion.
    Expansion chỉ tăng recall; behavior card và reranking chỉ dùng query gốc.

    ``retrieved_chunks`` là top-K dùng làm context. ``retrieved_dieu_ids`` là
    tập con semantic mạnh dùng làm seed recursive. Hai tập cố ý tách nhau để
    một candidate recall yếu không kéo theo toàn Điều và chuỗi tham chiếu.
    """
    logger.debug("[retrieve] Hybrid Search bắt đầu.")
    query = state["query"]
    legal_event = extract_canonical_legal_event(query)
    scenario_fact_state = extract_legal_scenario_facts(
        query, event=legal_event
    ).as_dict()
    behavior_profile = extract_legal_behavior(query, event=legal_event)
    logger.debug("[retrieve] behavior_profile=%s", behavior_profile.as_dict())
    domain_selection = select_legal_domains(query, event=legal_event)
    selected_domains = list(domain_selection.selected)
    logger.debug(
        "[retrieve] selected_domains=%s fallback=%s scores=%s",
        selected_domains,
        domain_selection.used_fallback,
        domain_selection.scores,
    )
    logger.debug(
        "[retrieve] filtered_domains=%s",
        list(domain_selection.filtered),
    )
    expanded_query, expansion_terms = expand_legal_query(query, event=legal_event)

    # Cross encoder chỉ rerank candidate pool nhỏ (mặc định 20/query), không
    # quét toàn corpus nên độ trễ được giữ trong giới hạn.
    wide_limit = min(top_k * 4, prefetch_limit)

    try:
        search_queries = [("original", query)]
        if expanded_query != query:
            search_queries.append(("expanded", expanded_query))
        candidate_by_id: dict[str, dict] = {}
        candidate_order: list[str] = []
        def merge_search_result(search_origin: str, result: dict) -> None:
            for rank, hit in enumerate(result.get("children", []), start=1):
                payload = getattr(hit, "payload", {}) or {}
                hit_id = str(payload.get("id") or getattr(hit, "id", ""))
                if hit_id not in candidate_by_id:
                    candidate_order.append(hit_id)
                    candidate_by_id[hit_id] = {
                        "chunk_id": payload.get("id"),
                        "parent_id": payload.get("parent_id"),
                        "text": payload.get("content", ""),
                        "dieu_id_raw": payload.get("dieu_id", ""),
                        "van_ban_id_raw": payload.get("van_ban_id", ""),
                        "source": (payload.get("metadata") or {}).get("source", ""),
                        "metadata": payload.get("metadata") or {},
                    }
                candidate = candidate_by_id[hit_id]
                candidate[f"{search_origin}_rank"] = rank
                candidate[f"{search_origin}_rrf_score"] = float(
                    getattr(hit, "score", 0.0) or 0.0
                )

        for search_origin, search_query in search_queries:
            result = hybrid_search(
                query=search_query,
                child_collection=qdrant_child_col,
                parent_collection=qdrant_parent_col,
                limit=wide_limit,
                prefetch_limit=prefetch_limit,
                fusion="rrf",
                include_parent=False,
                client=qdrant_client,
                model=embedding_model,
                bm25=bm25,
                legal_domains=selected_domains,
            )
            merge_search_result(search_origin, result)

        initial_records = [
            candidate_by_id[hit_id] for hit_id in candidate_order
        ]
        initial_contract_audit = assess_retrieval_contract(
            legal_event, initial_records
        )
        rescue_query = build_retrieval_rescue_query(
            legal_event, initial_contract_audit.missing_roles
        )
        if initial_contract_audit.requires_rescue and rescue_query:
            logger.info(
                "[retrieve] deterministic_contract_rescue missing=%s",
                list(initial_contract_audit.missing_roles),
            )
            rescue_result = hybrid_search(
                query=rescue_query,
                child_collection=qdrant_child_col,
                parent_collection=qdrant_parent_col,
                limit=wide_limit,
                prefetch_limit=prefetch_limit,
                fusion="rrf",
                include_parent=False,
                client=qdrant_client,
                model=embedding_model,
                bm25=bm25,
                legal_domains=list(legal_event.required_domains)
                or selected_domains,
            )
            merge_search_result("contract_rescue", rescue_result)

        post_search_records = [
            candidate_by_id[hit_id] for hit_id in candidate_order
        ]
        post_search_audit = assess_retrieval_contract(
            legal_event, post_search_records
        )
        if post_search_audit.requires_rescue:
            # The child-vector query can still miss a short clause inside a
            # long article. Scan only parent articles in required domains and
            # recover a parent whose own text satisfies the missing role.
            # This is structural Qdrant lookup: no embedding and no LLM call.
            missing_roles = set(post_search_audit.missing_roles)
            parent_offset = None
            parent_rank = 0
            while missing_roles:
                parent_points, parent_offset = qdrant_client.scroll(
                    collection_name=qdrant_parent_col,
                    limit=128,
                    offset=parent_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if not parent_points:
                    break
                for point in parent_points:
                    payload = point.payload or {}
                    source = str(
                        (payload.get("metadata") or {}).get("source", "")
                    )
                    if not set(
                        document_legal_domains(source)
                    ).intersection(legal_event.required_domains):
                        continue
                    roles = set(
                        retrieval_contract_roles(
                            legal_event,
                            f"{source}\n"
                            f"{payload.get('content', '')}",
                        )
                    )
                    recovered_roles = roles.intersection(missing_roles)
                    if not recovered_roles:
                        continue
                    parent_rank += 1
                    hit_id = str(payload.get("id") or point.id)
                    if hit_id not in candidate_by_id:
                        candidate_order.append(hit_id)
                        candidate_by_id[hit_id] = {
                            "chunk_id": payload.get("id"),
                            "parent_id": payload.get("id"),
                            "text": payload.get("content", ""),
                            "dieu_id_raw": payload.get("dieu_id", ""),
                            "van_ban_id_raw": payload.get("van_ban_id", ""),
                            "source": (
                                payload.get("metadata") or {}
                            ).get("source", ""),
                            "metadata": payload.get("metadata") or {},
                            "contract_parent_recovery": True,
                        }
                    candidate_by_id[hit_id][
                        "contract_parent_rank"
                    ] = parent_rank
                    missing_roles.difference_update(recovered_roles)
                if not parent_offset:
                    break
            if missing_roles != set(post_search_audit.missing_roles):
                logger.info(
                    "[retrieve] contract_parent_recovery remaining=%s",
                    sorted(missing_roles),
                )
    except Exception as e:
        logger.warning(f"Qdrant search failed (Qdrant có thể chưa ingest dữ liệu): {e}")
        candidate_by_id = {}
        candidate_order = []

    wide_chunks = [candidate_by_id[hit_id] for hit_id in candidate_order]
    wide_chunks = annotate_retrieval_contract_records(legal_event, wide_chunks)
    final_contract_audit = assess_retrieval_contract(legal_event, wide_chunks)
    logger.info(
        "[retrieve] retrieval_contract=%s",
        final_contract_audit.as_dict(),
    )
    candidate_count_by_domain = count_candidates_by_domain(
        wide_chunks, selected_domains
    )
    logger.debug(
        "[retrieve] candidate_count_by_domain=%s total_unique_candidates=%d",
        candidate_count_by_domain,
        len(wide_chunks),
    )

    # Expansion chỉ đóng góp candidate recall. Final ranking dùng query GỐC
    # cộng behavior card được trích trực tiếp từ query gốc.
    wide_chunks = rerank_context_records_with_behavior(
        query,
        wide_chunks,
        behavior_profile=behavior_profile,
        reranker=cross_encoder_reranker,
    )
    reranker_available = bool(
        wide_chunks and wide_chunks[0].get("reranker_available")
    )
    contract_protected = {
        str(chunk.get("chunk_id") or ""): chunk
        for chunk in wide_chunks
        if chunk.get("retrieval_contract_roles")
    }
    if reranker_available:
        # Behavior gate chạy trước cổng score: nếu Cross Encoder ưu tiên một
        # đoạn chỉ trùng video/quảng cáo, behavior vẫn có thể loại đoạn đó và
        # giữ lại Điều khớp hành vi trước khi áp ngưỡng tương đối.
        wide_chunks, behavior_filtered = filter_behaviorally_relevant(
            wide_chunks,
            behavior_profile=behavior_profile,
            minimum=float(os.getenv("BEHAVIOR_GATE_MIN_SCORE", "0.18")),
            activation=float(os.getenv("BEHAVIOR_GATE_ACTIVATION_SCORE", "0.35")),
        )
        relevance_filter = (
            filter_behavior_aware_relevant
            if not behavior_profile.is_empty
            else filter_semantically_relevant
        )
        wide_chunks = relevance_filter(
            wide_chunks,
            ratio=float(os.getenv("CROSS_ENCODER_CANDIDATE_RATIO", "0.20")),
            minimum=float(os.getenv("CROSS_ENCODER_CANDIDATE_MIN_SCORE", "0.0")),
        )
    else:
        # Không có cross encoder: chỉ tin candidate từ query gốc, không cho
        # expansion-only seed đi vào generation/recursive retrieval.
        wide_chunks = [
            chunk for chunk in wide_chunks if chunk.get("original_rank") is not None
        ]
        wide_chunks, behavior_filtered = filter_behaviorally_relevant(
            wide_chunks,
            behavior_profile=behavior_profile,
            minimum=float(os.getenv("BEHAVIOR_GATE_MIN_SCORE", "0.18")),
            activation=float(os.getenv("BEHAVIOR_GATE_ACTIVATION_SCORE", "0.35")),
        )
    if contract_protected:
        present = {str(chunk.get("chunk_id") or "") for chunk in wide_chunks}
        wide_chunks.extend(
            chunk
            for chunk_id, chunk in contract_protected.items()
            if chunk_id not in present
        )
        wide_chunks.sort(key=lambda chunk: -float(chunk.get("score") or 0.0))
    logger.debug(
        "[retrieve] behavior_filtered=%d behavior_scores=%s",
        behavior_filtered,
        [
            {
                "chunk_id": chunk.get("chunk_id"),
                "score": round(float(chunk.get("behavior_score") or 0.0), 4),
                "actions": chunk.get("matched_behavior_actions") or [],
            }
            for chunk in wide_chunks[:10]
        ],
    )

    # Dedup chạy sau semantic + behavior gate để giữ chunk tốt nhất của từng
    # document + article + clause.
    wide_chunks, duplicate_removed = deduplicate_context_records(wide_chunks)
    logger.debug("[retrieve] duplicate_removed=%d", duplicate_removed)
    relevance_field = "score" if not behavior_profile.is_empty else "semantic_score"
    best_relevance = max(
        (float(chunk.get(relevance_field) or 0.0) for chunk in wide_chunks),
        default=0.0,
    )
    retrieval_is_relevant = bool(wide_chunks)

    # Top-K context sau semantic rerank và diversity theo Điều.
    retrieved_chunks = select_balanced_top_k(query, wide_chunks, top_k)
    # Khi có nhiều văn bản cùng liên quan, ưu tiên căn cứ xác
    # định hành vi (Luật/Bộ luật/NĐ chuyên ngành) trước căn cứ
    # xử phạt. Score vẫn là tie-breaker trong cùng một nhóm văn bản.
    ordered_context_chunks = sort_context_records(retrieved_chunks)
    context_records = [
        normalise_provenance_record(
            chunk,
            is_seed=True,
            recursive_depth=0,
            expansion_reason="phase2_final_candidate",
        )
        for chunk in ordered_context_chunks
    ]
    context_texts = [c["text"] for c in context_records if c["text"]]

    # Recursive retrieval dùng tập seed CHẶT hơn top-K hiển thị. Khi có
    # behavior card, seed dựa trên điểm tổng hợp thay vì semantic thuần.
    if reranker_available:
        recursive_filter = (
            filter_behavior_aware_relevant
            if not behavior_profile.is_empty
            else filter_semantically_relevant
        )
        recursive_seed_chunks = recursive_filter(
            retrieved_chunks,
            ratio=float(os.getenv("RECURSIVE_SEED_SCORE_RATIO", "0.70")),
            minimum=float(os.getenv("RECURSIVE_SEED_MIN_SCORE", "0.0")),
        )
    else:
        recursive_seed_chunks = retrieved_chunks[:1]

    dieu_best_score: dict = {}
    dieu_order: list = []
    for c in recursive_seed_chunks:
        d = to_dieu_node_id(c["van_ban_id_raw"], c["dieu_id_raw"])
        if not d:
            continue
        score = float(c.get("score") or c.get("semantic_score") or 0.0)
        if d not in dieu_best_score:
            dieu_order.append(d)
            dieu_best_score[d] = score
        else:
            dieu_best_score[d] = max(dieu_best_score[d], score)
    retrieved_dieu_ids = dieu_order

    # Baseline article-expand vẫn lấy các Điều khác nhau trong candidate pool
    # đã qua semantic gate, giữ nguyên output/API đánh giá hiện có.
    wide_dieu_best_score: dict = {}
    wide_dieu_order: list = []
    for c in wide_chunks:
        d = to_dieu_node_id(c["van_ban_id_raw"], c["dieu_id_raw"])
        if not d:
            continue
        score = float(c.get("score") or c.get("semantic_score") or 0.0)
        if d not in wide_dieu_best_score:
            wide_dieu_order.append(d)
            wide_dieu_best_score[d] = score
        else:
            wide_dieu_best_score[d] = max(wide_dieu_best_score[d], score)
    article_expand_dieu_ids = compute_focus_dieu_ids(
        wide_dieu_order, wide_dieu_best_score,
        score_ratio=article_expand_score_ratio, max_dieu=top_k,
    )

    logger.debug(
        f"Retrieved {len(retrieved_chunks)} chunk top-k -> {len(retrieved_dieu_ids)} seed semantic mạnh; "
        f"article_expand: {len(article_expand_dieu_ids)} Điều khác nhau (quét rộng {len(wide_chunks)} chunk)."
    )
    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieved_dieu_ids": retrieved_dieu_ids,
        "dieu_scores": dieu_best_score,
        "context_texts": context_texts,
        "context_records": context_records,
        # Card đã được tạo ở Phase 2; các bước downstream chỉ được đọc lại,
        # tuyệt đối không rewrite hoặc trích xuất một hành vi khác.
        "behavior_profile": behavior_profile.as_dict(),
        "scenario_fact_state": scenario_fact_state,
        "article_expand_dieu_ids": article_expand_dieu_ids,
        "expanded_query": expanded_query,
        "query_expansion_terms": expansion_terms,
        "retrieval_is_relevant": retrieval_is_relevant,
        "retrieval_relevance": best_relevance,
        "retrieval_contract": final_contract_audit.as_dict(),
    }
