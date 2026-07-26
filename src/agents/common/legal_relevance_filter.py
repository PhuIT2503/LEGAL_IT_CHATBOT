"""Cổng relevance pháp lý cuối cùng trước generation.

Retrieval tối ưu recall nên top-K vẫn có thể chứa Điều chỉ gần về từ vựng.
Module này đánh giá lại theo từng Điều bằng *query gốc*, giữ toàn bộ Điều khi
ít nhất một Khoản/Điểm liên quan, và loại Điều mức ``Thấp`` khỏi prompt. Kết
quả đánh giá chỉ được ghi debug log, không trở thành nội dung trả lời.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import os
import re
from typing import Any, Iterable, Sequence

from src.agents.common.cross_encoder_reranker import (
    CrossEncoderReranker,
    get_cross_encoder_reranker,
)
from src.agents.common.grounded_validation import GroundedSource, build_grounded_sources
from src.agents.common.legal_applicability import (
    REMOVE as APPLICABILITY_REMOVE,
    check_legal_applicability,
)
from src.agents.common.retrieval_provenance import (
    article_key,
    behavior_profile_from_value,
    records_to_context_texts,
)
from src.agents.common.retrieval_ranking import lexical_relevance
from src.retrieval.legal_behaviors import BehaviorProfile, extract_legal_behavior

logger = logging.getLogger(__name__)

HIGH = "Cao"
MEDIUM = "Trung bình"
LOW = "Thấp"

KEEP = "KEEP"
WEAK_KEEP = "WEAK_KEEP"
REMOVE = "REMOVE"


@dataclass(frozen=True)
class LegalRelevanceDecision:
    """Kết quả nội bộ cho một Điều; không render ra câu trả lời."""

    document: str
    article: str
    scope: str
    score: float
    relative_score: float
    level: str
    method: str
    decision: str = KEEP
    is_seed: bool = False
    behavior_score: float = 0.0
    seed_preserved: bool = False
    behavior_preserved: bool = False
    relevance_removed: bool = False
    reason_removed: str = ""
    decision_stage: str = "legal_relevance"


@dataclass(frozen=True)
class LegalRelevanceResult:
    contexts: tuple[str, ...]
    decisions: tuple[LegalRelevanceDecision, ...]

    @property
    def kept_count(self) -> int:
        return sum(decision.decision != REMOVE for decision in self.decisions)


def _article_key(source: GroundedSource) -> tuple[str, str]:
    document = " ".join(source.document.casefold().split())
    article = " ".join(source.article.casefold().split())
    return document, article


def _short_scope(sources: Sequence[GroundedSource], max_chars: int = 260) -> str:
    """Tóm tắt phạm vi bằng phần mở đầu retrieved, không nhờ LLM tự đặt luật."""

    body = next((source.body for source in sources if source.body), "")
    scope = re.split(r"(?<=[.;:])\s+", " ".join(body.split()), maxsplit=1)[0]
    if len(scope) > max_chars:
        scope = scope[:max_chars].rstrip(" ,;:") + "…"
    return scope or "Không xác định được phạm vi từ nội dung đã truy xuất."


def _source_passage(source: GroundedSource) -> str:
    return (
        f"{source.document}. Điều {source.article}. "
        f"{source.body or source.text}"
    ).strip()


def _fallback_scores(query: str, sources: Sequence[GroundedSource]) -> list[float]:
    """Fallback có giới hạn khi model lỗi; không dùng để rerank retrieval."""

    return [
        min(1.0, lexical_relevance(query, _source_passage(source)) / 4.0)
        for source in sources
    ]


def _classify(score: float, relative_score: float, *, method: str) -> str:
    if method == "cross_encoder":
        high_min = float(os.getenv("LEGAL_RELEVANCE_HIGH_MIN_SCORE", "0.005"))
        medium_min = float(os.getenv("LEGAL_RELEVANCE_MEDIUM_MIN_SCORE", "0.0005"))
        high_ratio = float(os.getenv("LEGAL_RELEVANCE_HIGH_RATIO", "0.65"))
        medium_ratio = float(os.getenv("LEGAL_RELEVANCE_MEDIUM_RATIO", "0.15"))
    else:
        # Lexical chỉ là fail-safe. Ngưỡng tuyệt đối cao hơn để từ chung như
        # "dữ liệu" không đủ giữ một Điều lạc chủ đề.
        high_min = float(os.getenv("LEGAL_RELEVANCE_FALLBACK_HIGH_MIN", "0.55"))
        medium_min = float(os.getenv("LEGAL_RELEVANCE_FALLBACK_MEDIUM_MIN", "0.20"))
        high_ratio = 0.65
        medium_ratio = 0.35

    if score >= high_min and relative_score >= high_ratio:
        return HIGH
    if score >= medium_min and relative_score >= medium_ratio:
        return MEDIUM
    return LOW


def filter_legal_contexts(
    query: str,
    context_texts: Iterable[str],
    *,
    reranker: CrossEncoderReranker | Any | None = None,
    candidate_records: Iterable[dict[str, Any]] | None = None,
    behavior_profile: BehaviorProfile | None = None,
) -> LegalRelevanceResult:
    """Đánh giá và loại Điều relevance thấp trước khi tạo SOURCE/prompt.

    Cross encoder chấm từng Khoản/Điểm để không bỏ sót phần liên quan nằm sâu
    trong một Điều dài; điểm của Điều là điểm cao nhất trong các phần của nó.
    Query expansion tuyệt đối không được truyền vào hàm này.
    """

    contexts = [str(context) for context in context_texts if str(context).strip()]
    sources = build_grounded_sources(contexts, limit=256)
    if not sources:
        logger.debug(
            "[legal_relevance] Không có Điều luật parse được; loại toàn bộ context trước generation."
        )
        return LegalRelevanceResult(contexts=(), decisions=())

    groups: dict[tuple[str, str], list[GroundedSource]] = {}
    for source in sources:
        groups.setdefault(_article_key(source), []).append(source)

    records_by_article: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in candidate_records or ():
        records_by_article.setdefault(article_key(record), []).append(record)

    # Reuse the same thresholds as the Phase 2 behavior gate.  This keeps the
    # preservation decision aligned with Retrieval instead of introducing a
    # second, unrelated notion of "strong behavior".
    behavior_preserve_threshold = float(
        os.getenv("BEHAVIOR_GATE_ACTIVATION_SCORE", "0.35")
    )

    scorer = reranker or get_cross_encoder_reranker()
    passages = [_source_passage(source) for source in sources]
    scored = scorer.score(query, passages)
    if scored is None or len(scored) != len(sources):
        method = "lexical_fallback"
        source_scores = _fallback_scores(query, sources)
    else:
        method = "cross_encoder"
        source_scores = [float(score) for score in scored]

    score_by_source = {
        id(source): score for source, score in zip(sources, source_scores)
    }
    article_scores = {
        key: max(score_by_source[id(source)] for source in article_sources)
        for key, article_sources in groups.items()
    }
    best_score = max(article_scores.values(), default=0.0)

    kept_contexts: list[str] = []
    decisions: list[LegalRelevanceDecision] = []
    for key, article_sources in groups.items():
        score = article_scores[key]
        relative_score = score / best_score if best_score > 0 else 0.0
        level = _classify(score, relative_score, method=method)
        article_records = records_by_article.get(key, [])
        is_seed = any(bool(record.get("is_seed")) for record in article_records)
        behavior_score = max(
            (float(record.get("behavior_score") or 0.0) for record in article_records),
            default=0.0,
        )
        seed_preserved = level == LOW and is_seed
        behavior_preserved = (
            level == LOW
            and not is_seed
            and behavior_score >= behavior_preserve_threshold
        )
        if level != LOW:
            final_decision = KEEP
        elif seed_preserved or behavior_preserved:
            final_decision = WEAK_KEEP
        else:
            final_decision = REMOVE
        reason_removed = ""
        if final_decision == REMOVE:
            reason_removed = (
                "legal relevance thấp, không phải seed và "
                f"behavior_score={behavior_score:.4f} < "
                f"{behavior_preserve_threshold:.4f}"
            )
        first = article_sources[0]
        decision = LegalRelevanceDecision(
            document=first.document,
            article=first.article,
            scope=_short_scope(article_sources),
            score=score,
            relative_score=relative_score,
            level=level,
            method=method,
            decision=final_decision,
            is_seed=is_seed,
            behavior_score=behavior_score,
            seed_preserved=seed_preserved,
            behavior_preserved=behavior_preserved,
            relevance_removed=final_decision == REMOVE,
            reason_removed=reason_removed,
        )
        decisions.append(decision)
        logger.debug(
            "[legal_relevance] document=%r article=%r scope=%r behavior=%r "
            "score=%.6f relative=%.3f level=%s method=%s decision=%s "
            "is_seed=%s seed_preserved=%s behavior_preserved=%s "
            "relevance_removed=%s reason_removed=%r decision_stage=%s",
            decision.document,
            decision.article,
            decision.scope,
            query,
            decision.score,
            decision.relative_score,
            decision.level,
            decision.method,
            decision.decision,
            decision.is_seed,
            decision.seed_preserved,
            decision.behavior_preserved,
            decision.relevance_removed,
            decision.reason_removed,
            decision.decision_stage,
        )
        if decision.decision != REMOVE:
            # Chỉ dựng lại từ text retrieval; không thêm mô tả phạm vi hoặc
            # nhãn relevance vào context mà model nhìn thấy.
            kept_contexts.append("\n".join(source.text for source in article_sources))

    logger.debug(
        "[legal_relevance] Giữ %s/%s Điều trước generation.",
        len(kept_contexts),
        len(groups),
    )
    return LegalRelevanceResult(
        contexts=tuple(kept_contexts),
        decisions=tuple(decisions),
    )


def prepare_generation_context(
    state: dict[str, Any],
    *,
    reranker: CrossEncoderReranker | Any | None = None,
    llm_client=None,
) -> tuple[list[str], dict[str, Any]]:
    """Chạy semantic relevance rồi legal applicability trước generation."""

    context_records = [dict(record) for record in state.get("context_records", [])]
    combined = (
        records_to_context_texts(context_records)
        if context_records
        else list(state.get("context_texts", []))
    )
    graph_context = str(state.get("graph_context", "") or "")
    if graph_context:
        combined.append(graph_context)
    behavior_profile = behavior_profile_from_value(state.get("behavior_profile"))
    if behavior_profile.is_empty:
        behavior_profile = extract_legal_behavior(state.get("query", ""))
    result = filter_legal_contexts(
        state.get("query", ""),
        combined,
        reranker=reranker,
        candidate_records=context_records,
        behavior_profile=behavior_profile,
    )
    filtered = list(result.contexts)
    relevance_by_key = {
        (" ".join(decision.document.casefold().split()), " ".join(decision.article.casefold().split())): decision
        for decision in result.decisions
    }
    kept_relevance_keys = {
        key
        for key, decision in relevance_by_key.items()
        if decision.decision != REMOVE
    }
    filtered_records = []
    for record in context_records:
        key = article_key(record)
        decision = relevance_by_key.get(key)
        if key not in kept_relevance_keys or decision is None:
            continue
        annotated = dict(record)
        annotated.update(
            {
                "seed_preserved": decision.seed_preserved,
                "behavior_preserved": decision.behavior_preserved,
                "relevance_removed": False,
                "applicability_removed": False,
                "reason_removed": "",
                "decision_stage": decision.decision_stage,
                "relevance_decision": decision.decision,
            }
        )
        filtered_records.append(annotated)
    decision_trace = [asdict(decision) for decision in result.decisions]
    retrieval_gap = False
    if filtered and llm_client is not None:
        applicability = check_legal_applicability(
            state.get("query", ""),
            filtered,
            llm_client=llm_client,
            behavior_profile=behavior_profile,
            candidate_records=filtered_records,
        )
        filtered = list(applicability.contexts)
        kept_applicability_keys = {
            (" ".join(decision.document.casefold().split()), " ".join(decision.article.casefold().split()))
            for decision in applicability.decisions
            if decision.decision != APPLICABILITY_REMOVE
        }
        applicability_by_key = {
            (" ".join(decision.document.casefold().split()), " ".join(decision.article.casefold().split())): decision
            for decision in applicability.decisions
        }
        next_records: list[dict[str, Any]] = []
        for record in filtered_records:
            key = article_key(record)
            decision = applicability_by_key.get(key)
            if key not in kept_applicability_keys or decision is None:
                continue
            annotated = dict(record)
            annotated.update(
                {
                    "seed_survived": decision.seed_survived,
                    "seed_removed": False,
                    "behavior_preserved": bool(
                        annotated.get("behavior_preserved")
                        or decision.behavior_preserved
                    ),
                    "applicability_removed": False,
                    "reason_removed": "",
                    "decision_stage": decision.decision_stage,
                    "applicability_decision": decision.decision,
                }
            )
            next_records.append(annotated)
        filtered_records = next_records
        decision_trace.extend(asdict(decision) for decision in applicability.decisions)
        retrieval_gap = applicability.retrieval_gap
    update = {
        "context_texts": filtered,
        "context_records": filtered_records,
        # graph_context đã được gộp và lọc; để trống tránh validator đọc lại
        # bản chưa lọc ở cuối workflow.
        "graph_context": "",
        "retrieval_is_relevant": bool(filtered),
        # Internal instrumentation only.  Workflow/public response does not
        # expose this field, while the benchmark can identify the exact stage
        # at which a candidate was removed.
        "retrieval_decisions": decision_trace,
    }
    if llm_client is not None:
        update["retrieval_is_complete"] = bool(
            state.get("retrieval_is_complete", True)
        ) and not retrieval_gap
    return filtered, update
