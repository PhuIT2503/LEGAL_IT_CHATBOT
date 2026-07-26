"""Recursive retrieval cho nhánh Critic, chạy hoàn toàn trước generation.

Thuật toán tái sử dụng các tín hiệu completeness sẵn có trong
CriticQueryEngine: thiếu Khoản/Điểm cùng Điều, chế tài kép và tham
chiếu liên Điều. Không thêm agent hay thay API của workflow.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.agents.common.focus_dieu import compute_focus_dieu_ids
from src.agents.common.legal_response import sort_context_records
from src.agents.common.retrieval_provenance import (
    behavior_profile_from_value,
    normalise_provenance_record,
    provenance_log_record,
    records_to_context_texts,
)
from src.knowledge_graph.graph_builder import to_dieu_node_id
from src.retrieval.legal_behaviors import extract_legal_behavior, score_behavior_relevance

logger = logging.getLogger(__name__)


def _part_counts(retrieved_chunks: list[dict[str, Any]]) -> dict[str, int]:
    parts: dict[str, set[str]] = {}
    for chunk in retrieved_chunks:
        dieu_id = to_dieu_node_id(chunk.get("van_ban_id_raw", ""), chunk.get("dieu_id_raw", ""))
        chunk_id = chunk.get("chunk_id")
        if dieu_id and chunk_id:
            parts.setdefault(dieu_id, set()).add(str(chunk_id))
    return {dieu_id: len(chunk_ids) for dieu_id, chunk_ids in parts.items()}


def _seed_graph_node_id(record: dict[str, Any]) -> str:
    """Đổi raw child ID sang đúng namespace node Neo4j, giữ K/P suffix."""

    raw_dieu = str(record.get("dieu_id_raw") or "")
    canonical_dieu = to_dieu_node_id(
        record.get("van_ban_id_raw", ""), raw_dieu
    )
    chunk_id = str(record.get("chunk_id") or "")
    if not canonical_dieu:
        return ""
    lowered_chunk = chunk_id.casefold()
    lowered_dieu = raw_dieu.casefold()
    position = lowered_chunk.find(lowered_dieu) if lowered_dieu else -1
    suffix = chunk_id[position + len(raw_dieu) :] if position >= 0 else ""
    return f"{canonical_dieu}{suffix}"


def recursive_retrieve(
    state: dict[str, Any],
    *,
    dieu_content_store,
    critic_query_engine,
    critic_score_ratio: float,
    critic_max_dieu: int,
    max_depth: int = 3,
    max_iterations: int = 5,
) -> dict[str, Any]:
    """Mở rộng ngữ cảnh tới khi đủ hoặc chạm giới hạn an toàn."""

    query = state.get("query", "")
    retrieved_chunks = list(state.get("retrieved_chunks", []))
    behavior_profile = behavior_profile_from_value(state.get("behavior_profile"))
    if behavior_profile.is_empty:
        # Backward-compatible cho caller cũ/test không truyền state Phase 2.
        behavior_profile = extract_legal_behavior(query)
    behavior_threshold = float(os.getenv("BEHAVIOR_GATE_MIN_SCORE", "0.18"))
    seed_records = [
        normalise_provenance_record(
            chunk,
            is_seed=True,
            recursive_depth=0,
            expansion_reason="phase2_final_candidate",
        )
        for chunk in retrieved_chunks
    ]
    for seed in seed_records:
        logger.debug("[retrieve] seed_candidate=%s", provenance_log_record(seed))
    logger.debug(
        "[retrieve] behavior_before_recursive=%s",
        [
            {
                "chunk_id": seed.get("chunk_id"),
                "behavior_score": seed.get("behavior_score"),
                "matched_actions": seed.get("matched_behavior_actions") or [],
            }
            for seed in seed_records
        ],
    )
    all_dieu_ids = list(state.get("retrieved_dieu_ids", []))
    dieu_scores = dict(state.get("dieu_scores", {}))
    focus_dieu_ids = compute_focus_dieu_ids(
        all_dieu_ids,
        dieu_scores,
        score_ratio=critic_score_ratio,
        max_dieu=critic_max_dieu,
    )

    if not focus_dieu_ids:
        return {
            "context_records": seed_records,
            "context_texts": records_to_context_texts(seed_records),
            "graph_context": "",
            "graph_fetched_dieu_ids": [],
            "critic_report": {"is_complete": False, "unresolved": ["Không xác định được Điều liên quan."]},
            "retrieval_is_complete": False,
            "recursive_retrieval_done": True,
        }

    total_parts = {dieu_id: dieu_content_store.child_chunk_count(dieu_id) for dieu_id in focus_dieu_ids}
    try:
        focus_node_ids = [
            _seed_graph_node_id(record)
            for record in seed_records
            if record.get("chunk_id")
            and to_dieu_node_id(
                record.get("van_ban_id_raw", ""), record.get("dieu_id_raw", "")
            ) in set(focus_dieu_ids)
        ]
        try:
            report = critic_query_engine.check_retrieval_completeness(
                focus_dieu_ids,
                all_dieu_ids,
                _part_counts(retrieved_chunks),
                total_parts,
                focus_node_ids=focus_node_ids,
            )
        except TypeError:
            # Giữ tương thích cho custom CriticQueryEngine cũ. Production
            # engine hỗ trợ focus_node_ids; fallback chỉ phục vụ caller ngoài.
            report = critic_query_engine.check_retrieval_completeness(
                focus_dieu_ids,
                all_dieu_ids,
                _part_counts(retrieved_chunks),
                total_parts,
            )
    except Exception as exc:
        # Neo4j là lớp bổ sung; nếu tạm thời không truy cập được
        # thì vẫn generate từ top-k, nhưng bắt buộc đánh dấu incomplete.
        logger.debug("[retrieve] Không kiểm tra được completeness.", exc_info=True)
        return {
            "context_records": seed_records,
            "context_texts": records_to_context_texts(seed_records),
            "graph_context": "",
            "graph_fetched_dieu_ids": [],
            "critic_report": {"is_complete": False, "unresolved": ["Không kiểm tra được toàn bộ tham chiếu."]},
            "retrieval_is_complete": False,
            "recursive_retrieval_done": True,
        }
    report["is_complete_initial"] = report.get("is_complete", False)

    fetched_records: dict[str, dict[str, Any]] = {}
    rejected_records: list[dict[str, Any]] = []
    unresolved: list[str] = []
    all_known = set(all_dieu_ids)
    max_total_fetch = max(critic_max_dieu * max_depth, critic_max_dieu)
    iterations = 0

    seed_by_node = {
        _seed_graph_node_id(record): record
        for record in seed_records
        if record.get("chunk_id")
    }
    seed_by_dieu: dict[str, dict[str, Any]] = {}
    for record in seed_records:
        canonical = to_dieu_node_id(
            record.get("van_ban_id_raw", ""), record.get("dieu_id_raw", "")
        )
        if canonical:
            seed_by_dieu.setdefault(canonical, record)

    def fetch(
        dieu_id: str,
        reason: str,
        *,
        depth: int,
        source_record: dict[str, Any] | None,
        missing_node_id: str = "",
    ) -> dict[str, Any] | None:
        if dieu_id in fetched_records:
            return fetched_records[dieu_id]
        if len(fetched_records) >= max_total_fetch:
            unresolved.append(f"Chạm giới hạn {max_total_fetch} Điều khi truy xuất {dieu_id}.")
            return None
        record = dieu_content_store.fetch_parent_record(dieu_id)
        if not record or not record.get("text"):
            unresolved.append(f"Không lấy được toàn văn {dieu_id}.")
            return None

        match = score_behavior_relevance(
            behavior_profile,
            f"{record.get('source', '')}\n{record.get('text', '')}",
        )
        chain = list((source_record or {}).get("provenance_chain") or ())
        if missing_node_id:
            chain.append(missing_node_id)
        parent_chunk_id = str(record.get("chunk_id") or "")
        if parent_chunk_id and (not chain or chain[-1] != parent_chunk_id):
            chain.append(parent_chunk_id)
        candidate = normalise_provenance_record(
            {
                **record,
                "behavior_score": match.score,
                "behavior_action_score": match.action_score,
                "behavior_object_score": match.object_score,
                "behavior_purpose_score": match.purpose_score,
                "behavior_condition_score": match.condition_score,
                "matched_behavior_actions": list(match.matched_actions),
                "matched_behavior_objects": list(match.matched_objects),
                "matched_behavior_purposes": list(match.matched_purposes),
                "matched_behavior_conditions": list(match.matched_conditions),
                # Graph expansion không có dense/RRF/CE score. Giữ explicit
                # None thay vì giả tạo một retrieval score.
                "retrieval_score": None,
                "cross_encoder_score": None,
            },
            is_seed=False,
            recursive_depth=depth,
            expansion_reason=reason,
            provenance_chain=chain,
        )
        logger.debug("[retrieve] recursive_candidate=%s", provenance_log_record(candidate))
        if behavior_profile.has_primary_action and match.score < behavior_threshold:
            rejected = {
                **provenance_log_record(candidate),
                "reason_rejected": (
                    f"behavior_score={match.score:.4f} < threshold={behavior_threshold:.4f}"
                ),
            }
            rejected_records.append(rejected)
            unresolved.append(
                f"Loại {dieu_id} vì không khớp Behavior Card của seed."
            )
            logger.debug("[retrieve] reason_rejected=%s", rejected)
            return None

        fetched_records[dieu_id] = candidate
        all_known.add(dieu_id)
        logger.debug(
            "[retrieve] provenance_chain=%s",
            {
                "chunk_id": candidate.get("chunk_id"),
                "chain": candidate.get("provenance_chain"),
                "expansion_reason": reason,
            },
        )
        return candidate

    # Chỉ mở rộng theo tham chiếu nối đúng node seed (Khoản/Điểm). Tín hiệu
    # "Điều có nhiều phần" không đủ để thay seed bằng toàn parent article.
    # Full article chỉ được thêm khi có bằng chứng mạnh như chế tài kép.
    initial_references = list(report.get("missing_references", []))
    compound_ids = [
        item.get("dieu_id")
        for item in report.get("compound_penalty_behaviors", [])
        if item.get("dieu_id")
    ]
    frontier: list[dict[str, Any]] = []
    if initial_references or compound_ids:
        iterations = 1
    for ref in initial_references:
        dieu_id = ref.get("missing_dieu_id")
        if not dieu_id:
            continue
        source_record = seed_by_node.get(str(ref.get("focus_node_id") or ""))
        source_record = source_record or next(iter(seed_records), None)
        candidate = fetch(
            dieu_id,
            ref.get("reason", "căn cứ được tham chiếu từ seed"),
            depth=1,
            source_record=source_record,
            missing_node_id=str(ref.get("missing_node_id") or ""),
        )
        if candidate is not None:
            frontier.append(
                {
                    "dieu_id": dieu_id,
                    "node_id": str(ref.get("missing_node_id") or ""),
                    "record": candidate,
                }
            )

    for dieu_id in compound_ids:
        source_record = seed_by_dieu.get(dieu_id)
        fetch(
            dieu_id,
            "mở rộng toàn Điều do graph xác nhận chế tài chính và bổ sung",
            depth=1,
            source_record=source_record,
        )

    # Các Điều vừa được dẫn chiếu có thể tiếp tục dẫn chiếu sang
    # Điều khác. Mỗi vòng chỉ mở rộng frontier mới, có visited và cap.
    depth = 1
    while frontier and depth < max_depth and iterations < max_iterations:
        iterations += 1
        try:
            references = critic_query_engine.find_missing_references(
                [item["dieu_id"] for item in frontier],
                list(all_known),
                focus_node_ids=[item["node_id"] for item in frontier if item.get("node_id")],
            )
        except Exception:
            logger.debug("[retrieve] Không quét được tham chiếu bắc cầu.", exc_info=True)
            unresolved.append("Không kiểm tra được tham chiếu bắc cầu.")
            break
        next_frontier: list[dict[str, Any]] = []
        for ref in references:
            dieu_id = ref.get("missing_dieu_id")
            if not dieu_id or dieu_id in all_known:
                continue
            source = next(
                (
                    item["record"]
                    for item in frontier
                    if item.get("node_id") == ref.get("focus_node_id")
                ),
                frontier[0]["record"] if frontier else None,
            )
            candidate = fetch(
                dieu_id,
                ref.get("reason", "tham chiếu bắc cầu"),
                depth=depth + 1,
                source_record=source,
                missing_node_id=str(ref.get("missing_node_id") or ""),
            )
            if candidate is not None:
                next_frontier.append(
                    {
                        "dieu_id": dieu_id,
                        "node_id": str(ref.get("missing_node_id") or ""),
                        "record": candidate,
                    }
                )
        frontier = next_frontier
        depth += 1

    # Nếu dừng vì cap trong khi frontier vẫn còn tham chiếu chưa xử lý,
    # completeness phải là False; tuyệt đối không coi cap là "đã đủ".
    if frontier:
        try:
            remaining = critic_query_engine.find_missing_references(
                [item["dieu_id"] for item in frontier],
                list(all_known),
                focus_node_ids=[item["node_id"] for item in frontier if item.get("node_id")],
            )
        except Exception:
            remaining = []
            unresolved.append("Không xác minh được các tham chiếu còn lại.")
        for ref in remaining:
            dieu_id = ref.get("missing_dieu_id")
            if dieu_id and dieu_id not in all_known:
                unresolved.append(f"Còn tham chiếu chưa truy xuất: {dieu_id}.")

    # Seed luôn tồn tại nguyên vẹn. Parent/recursive record chỉ được thêm,
    # tuyệt đối không thay thế chunk seed ban đầu.
    context_records = list(seed_records) + list(fetched_records.values())
    context_records = sort_context_records(context_records)
    context_texts = records_to_context_texts(context_records)

    structurally_skipped = [
        item.get("dieu_id")
        for item in report.get("structurally_incomplete_dieu", [])
        if item.get("dieu_id") not in compound_ids
    ]
    if structurally_skipped:
        unresolved.extend(
            f"Chưa mở rộng toàn {dieu_id}: chỉ có tín hiệu thiếu cấu trúc, không có căn cứ hành vi/tham chiếu từ seed."
            for dieu_id in structurally_skipped
        )
        logger.debug(
            "[retrieve] reason_rejected=%s",
            {
                "candidate_type": "article_parent",
                "dieu_ids": structurally_skipped,
                "reason": "structural incompleteness alone cannot replace a granular seed",
            },
        )

    report.update(
        {
            "is_complete": not unresolved,
            "iterations": iterations,
            "max_iterations": max_iterations,
            "depth": depth,
            "max_depth": max_depth,
            "fetched_dieu_ids": list(fetched_records),
            "recursive_rejected": rejected_records,
            "structural_expansion_skipped": structurally_skipped,
            "unresolved": unresolved,
        }
    )
    logger.debug(
        "[retrieve] recursive complete=%s iterations=%s depth=%s fetched=%s unresolved=%s query=%r",
        report["is_complete"],
        iterations,
        depth,
        len(fetched_records),
        len(unresolved),
        query,
    )
    logger.debug(
        "[retrieve] behavior_after_recursive=%s",
        {
            "accepted": [provenance_log_record(record) for record in fetched_records.values()],
            "rejected": rejected_records,
        },
    )

    return {
        "context_records": context_records,
        "context_texts": context_texts,
        # Đã trộn các parent vào context_texts trước generation. Không
        # dùng graph_context để tránh node regenerate gọi model lần hai.
        "graph_context": "",
        "graph_fetched_dieu_ids": list(fetched_records),
        "critic_report": report,
        "retrieval_is_complete": report["is_complete"],
        "recursive_retrieval_done": True,
    }
