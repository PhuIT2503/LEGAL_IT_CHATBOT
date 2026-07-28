"""Xếp hạng candidate retrieval bằng query gốc và cross encoder."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable
import unicodedata

from src.agents.common.cross_encoder_reranker import CrossEncoderReranker
from src.agents.common.legal_response import document_priority
from src.retrieval.legal_behaviors import (
    BehaviorMatch,
    BehaviorProfile,
    score_behavior_relevance,
)


_STOPWORDS = {
    "và", "là", "có", "của", "cho", "được", "bị", "thì", "một", "những",
    "các", "này", "đó", "trong", "trên", "với", "về", "theo", "tại", "hay",
    "không", "nào", "như", "để", "khi", "nếu", "do", "từ", "đã", "đang",
    "công", "ty", "người", "dùng", "việt", "nam", "quy", "định", "pháp", "luật",
}


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value or "")
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    ).casefold().replace("đ", "d")


def lexical_relevance(query: str, text: str) -> float:
    """Heuristic nhẹ chỉ dùng chọn SOURCE/hiển thị, không rerank retrieval."""

    stopwords = {_fold(value) for value in _STOPWORDS}
    query_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", _fold(query))
        if token not in stopwords and (len(token) >= 3 or token.isdigit())
    ]
    if not query_tokens:
        return 0.0
    text_tokens = set(re.findall(r"[a-z0-9]+", _fold(text)))
    unique_query = list(dict.fromkeys(query_tokens))
    weights = {token: 1.0 + math.log1p(len(token)) for token in unique_query}
    total = sum(weights.values()) or 1.0
    return 4.0 * sum(
        weight for token, weight in weights.items() if token in text_tokens
    ) / total


def _original_rank(record: dict[str, Any], fallback: int) -> int:
    value = record.get("original_rank")
    return int(value) if isinstance(value, (int, float)) and value > 0 else fallback


def _rerank_context_records(
    query: str,
    records: Iterable[dict[str, Any]],
    *,
    reranker: CrossEncoderReranker | Any | None = None,
    behavior_profile: BehaviorProfile | None = None,
) -> list[dict[str, Any]]:
    """Cross-encode bằng query gốc/behavior card; expansion không rerank.

    Candidate có mặt trong kết quả query gốc nhận một bonus nhỏ. Expansion-only
    vẫn có thể đi lên nếu cross encoder đánh giá thật sự phù hợp, nhưng không
    thể thắng chỉ nhờ chứa hàng loạt từ được expansion thêm vào.
    """

    items = [dict(record) for record in records]
    if not items:
        return []
    scorer = reranker or CrossEncoderReranker()
    behavior_enabled = bool(behavior_profile and not behavior_profile.is_empty)
    behavior_matches = [
        score_behavior_relevance(
            behavior_profile,
            f"{item.get('source', '')}\n{item.get('text', '')}",
        )
        if behavior_profile
        else BehaviorMatch(0.0, 0.0, 0.0, 0.0, 0.0)
        for item in items
    ]

    if behavior_enabled:
        passages = []
        for item in items:
            metadata = item.get("metadata") or {}
            domains = ", ".join(metadata.get("legal_domains") or [])
            passages.append(
                "\n".join(
                    part
                    for part in (
                        f"Domain pháp luật: {domains}" if domains else "",
                        str(item.get("source") or ""),
                        str(metadata.get("dieu_title") or ""),
                        str(item.get("text") or ""),
                    )
                    if part
                )
            )
        rerank_query = behavior_profile.enrich_query(query)
    else:
        passages = [
            f"{item.get('source', '')}\n{item.get('text', '')}".strip()
            for item in items
        ]
        rerank_query = query

    semantic_scores = scorer.score(rerank_query, passages)

    if semantic_scores is None or len(semantic_scores) != len(items):
        # Fail-safe: query gốc trước, đúng rank RRF của nó; candidate chỉ có từ
        # expansion đứng sau. Không dùng lexical overlap làm fallback.
        for index, (item, behavior_match) in enumerate(
            zip(items, behavior_matches), start=1
        ):
            original = item.get("original_rank") is not None
            item["reranker_available"] = False
            item["semantic_score"] = 1.0 / _original_rank(item, index) if original else 0.0
            _attach_behavior_match(item, behavior_match)
            if behavior_enabled:
                item["score"] = min(
                    1.0,
                    0.70 * float(item["semantic_score"])
                    + 0.30 * behavior_match.score,
                )
            else:
                item["score"] = item["semantic_score"]
        if behavior_enabled:
            return sorted(
                items,
                key=lambda item: (
                    -float(item.get("score") or 0.0),
                    item.get("original_rank") is None,
                    _original_rank(item, 10**9),
                ),
            )
        return sorted(
            items,
            key=lambda item: (
                item.get("original_rank") is None,
                _original_rank(item, 10**9),
                int(item.get("expanded_rank") or 10**9),
            ),
        )

    lowest = min(float(score) for score in semantic_scores)
    highest = max(float(score) for score in semantic_scores)
    spread = highest - lowest
    normalised_scores = [
        (float(score) - lowest) / spread if spread > 1e-12 else 1.0
        for score in semantic_scores
    ]
    for index, (item, raw_score, semantic_score, behavior_match) in enumerate(
        zip(items, semantic_scores, normalised_scores, behavior_matches), start=1
    ):
        original = item.get("original_rank") is not None
        original_bonus = (0.06 if behavior_enabled else 0.08) if original else 0.0
        reciprocal_bonus = 0.02 / _original_rank(item, index) if original else 0.0
        if behavior_enabled:
            final_score = min(
                1.0,
                float(semantic_score) * 0.64
                + behavior_match.score * 0.28
                + original_bonus
                + reciprocal_bonus,
            )
        else:
            final_score = min(
                1.0,
                float(semantic_score) * 0.90
                + original_bonus
                + reciprocal_bonus,
            )
        item["reranker_available"] = True
        item["cross_encoder_score"] = float(raw_score)
        item["semantic_score"] = float(semantic_score)
        _attach_behavior_match(item, behavior_match)
        item["score"] = final_score

    return sorted(
        items,
        key=lambda item: (
            -float(item.get("score") or 0.0),
            item.get("original_rank") is None,
            _original_rank(item, 10**9),
            int(item.get("expanded_rank") or 10**9),
        ),
    )


def _attach_behavior_match(item: dict[str, Any], match: BehaviorMatch) -> None:
    item["behavior_score"] = float(match.score)
    item["behavior_action_score"] = float(match.action_score)
    item["behavior_object_score"] = float(match.object_score)
    item["behavior_purpose_score"] = float(match.purpose_score)
    item["behavior_condition_score"] = float(match.condition_score)
    item["matched_behavior_actions"] = list(match.matched_actions)
    item["matched_behavior_objects"] = list(match.matched_objects)
    item["matched_behavior_purposes"] = list(match.matched_purposes)
    item["matched_behavior_conditions"] = list(match.matched_conditions)


def rerank_context_records(
    query: str,
    records: Iterable[dict[str, Any]],
    *,
    reranker: CrossEncoderReranker | Any | None = None,
) -> list[dict[str, Any]]:
    """Public API cũ: rerank bằng query gốc, không behavior enrichment."""

    return _rerank_context_records(query, records, reranker=reranker)


def rerank_context_records_with_behavior(
    query: str,
    records: Iterable[dict[str, Any]],
    *,
    behavior_profile: BehaviorProfile,
    reranker: CrossEncoderReranker | Any | None = None,
) -> list[dict[str, Any]]:
    """Internal Phase 2 entrypoint: Cross Encoder + behavior relevance."""

    return _rerank_context_records(
        query,
        records,
        reranker=reranker,
        behavior_profile=behavior_profile,
    )


def filter_semantically_relevant(
    records: Iterable[dict[str, Any]],
    *,
    ratio: float,
    minimum: float,
) -> list[dict[str, Any]]:
    """Cổng relevance theo cross-encoder score, dùng chung cho top-K/seed."""

    items = list(records)
    if not items:
        return []
    best = max(float(item.get("semantic_score") or 0.0) for item in items)
    if best <= 0:
        return []
    threshold = max(minimum, best * ratio)
    return [
        item
        for item in items
        if float(item.get("semantic_score") or 0.0) >= threshold
    ]


def filter_behavior_aware_relevant(
    records: Iterable[dict[str, Any]],
    *,
    ratio: float,
    minimum: float,
) -> list[dict[str, Any]]:
    """Cổng relevance theo điểm tổng hợp CE + behavior của Phase 2."""

    items = list(records)
    if not items:
        return []
    best = max(float(item.get("score") or 0.0) for item in items)
    if best <= 0:
        return []
    threshold = max(minimum, best * ratio)
    return [item for item in items if float(item.get("score") or 0.0) >= threshold]


def filter_behaviorally_relevant(
    records: Iterable[dict[str, Any]],
    *,
    behavior_profile: BehaviorProfile,
    minimum: float,
    activation: float,
) -> tuple[list[dict[str, Any]], int]:
    """Loại candidate không khớp action khi đã có ít nhất một match tin cậy.

    Nếu taxonomy chưa nhận ra hành vi trong bất kỳ candidate nào, cổng không
    kích hoạt để tránh làm rỗng context do rule coverage chưa đủ.
    """

    items = list(records)
    if not items or not behavior_profile.has_primary_action:
        return items, 0
    best = max(float(item.get("behavior_score") or 0.0) for item in items)
    if best < activation:
        return items, 0
    kept = [
        item
        for item in items
        if float(item.get("behavior_score") or 0.0) >= minimum
    ]
    if not kept:
        return items, 0
    return kept, len(items) - len(kept)


_CLAUSE_ID_RE = re.compile(r"_K(?P<clause>\d+)(?:_|$)", re.IGNORECASE)


def retrieval_unit_key(chunk: dict[str, Any]) -> tuple[str, str, str]:
    """Định danh đơn vị document + article + clause của một candidate.

    Corpus cũ chưa có ``clause_number`` trong metadata nên vẫn phải suy ra từ
    chunk ID. Điểm a/b trong cùng một Khoản có cùng key và chỉ giữ chunk có
    semantic rank cao nhất.
    """

    metadata = chunk.get("metadata") or {}
    document = str(
        chunk.get("van_ban_id_raw")
        or metadata.get("doc_id")
        or chunk.get("source")
        or ""
    )
    article = str(
        chunk.get("dieu_id_raw")
        or metadata.get("article_number")
        or chunk.get("chunk_id")
        or ""
    )
    clause = metadata.get("clause_number")
    if clause in (None, ""):
        match = _CLAUSE_ID_RE.search(str(chunk.get("chunk_id") or ""))
        clause = match.group("clause") if match else ""
    return document, article, str(clause)


def deduplicate_context_records(records: Iterable[dict]) -> tuple[list[dict], int]:
    """Giữ candidate đầu tiên (đang có rank tốt nhất) cho mỗi đơn vị luật."""

    unique: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    duplicate_removed = 0
    for record in records:
        key = retrieval_unit_key(record)
        if key in seen:
            duplicate_removed += 1
            continue
        seen.add(key)
        unique.append(record)
    return unique, duplicate_removed


def select_balanced_top_k(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Giữ semantic order và không trùng document + article + clause."""

    asks_sanction = bool(
        re.search(
            r"\b(?:xử phạt|mức phạt|phạt tiền|chế tài|bị phạt|bị xử lý|xử lý như thế nào)\b",
            query,
            re.IGNORECASE,
        )
    )
    law_candidate = next(
        (
            chunk
            for chunk in chunks
            if document_priority(
                str(chunk.get("source") or ""), str(chunk.get("text") or "")
            )
            <= 2
        ),
        None,
    )
    substantive_candidate = law_candidate or next(
        (
            chunk
            for chunk in chunks
            if document_priority(
                str(chunk.get("source") or ""), str(chunk.get("text") or "")
            )
            == 3
        ),
        None,
    )
    penalty_candidate = next(
        (
            chunk
            for chunk in chunks
            if document_priority(
                str(chunk.get("source") or ""), str(chunk.get("text") or "")
            )
            == 4
        ),
        None,
    ) if asks_sanction else None

    selected: list[dict] = []
    seen_units: set[tuple[str, str, str]] = set()

    def add(chunk: dict | None) -> None:
        if not chunk or len(selected) >= top_k:
            return
        unit_key = retrieval_unit_key(chunk)
        if unit_key in seen_units:
            return
        selected.append(chunk)
        seen_units.add(unit_key)

    # Event contracts are attached only to provisions whose own text directly
    # supplies a required legal role. Reserve at most one slot per role so a
    # low CE score cannot silently remove the prohibition or its consequence.
    covered_contract_roles: set[str] = set()
    for chunk in chunks:
        roles = set(chunk.get("retrieval_contract_roles") or [])
        new_roles = roles - covered_contract_roles
        if not new_roles:
            continue
        add(chunk)
        covered_contract_roles.update(new_roles)

    # Hệ thống phải có căn cứ xác định hành vi trước khi đưa chế tài. Chỉ
    # reserve một slot; phần còn lại vẫn theo semantic rank của cross encoder.
    add(substantive_candidate)
    if asks_sanction:
        add(penalty_candidate)

    for chunk in chunks:
        add(chunk)
        if len(selected) >= top_k:
            break

    for chunk in chunks:
        if len(selected) >= top_k:
            break
        # Fallback vẫn phải đi qua cùng cổng dedup; không append trực tiếp.
        add(chunk)
    return selected[:top_k]
