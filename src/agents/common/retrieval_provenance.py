"""Chuẩn hoá provenance nội bộ từ Retrieval tới sát Generation.

Public API vẫn tiếp tục dùng ``context_texts``.  ``context_records`` là carrier
nội bộ song song để không làm mất chunk/score/recursive lineage khi các cổng
relevance và applicability xử lý context.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from src.retrieval.legal_behaviors import BehaviorProfile


_CHUNK_COORDINATES_RE = re.compile(
    r"_D(?P<article>\d+[a-zA-Z]*)"
    r"(?:_K(?P<clause>\d+))?"
    r"(?:_P(?P<point>[^_]+))?",
    re.IGNORECASE,
)
_TEXT_COORDINATES_RE = re.compile(
    r"\[\s*(?:Điểm\s+(?P<point>[^,\]]+)\s*,\s*)?"
    r"(?:Khoản\s+(?P<clause>[^,\]]+)\s*,\s*)?"
    r"Điều\s+(?P<article>[^,\]]+)\s*,\s*(?P<document>[^\]]+)\]",
    re.IGNORECASE,
)


def behavior_profile_from_value(value: Any) -> BehaviorProfile:
    """Khôi phục đúng Behavior Card Phase 2 từ state serializable."""

    if isinstance(value, BehaviorProfile):
        return value
    payload = value if isinstance(value, dict) else {}
    return BehaviorProfile(
        actions=tuple(payload.get("actions") or ()),
        objects=tuple(payload.get("objects") or ()),
        purposes=tuple(payload.get("purposes") or ()),
        conditions=tuple(payload.get("conditions") or ()),
    )


def _coordinates(record: dict[str, Any]) -> tuple[str, str, str, str]:
    metadata = record.get("metadata") or {}
    text = str(record.get("text") or "")
    chunk_id = str(record.get("chunk_id") or "")
    text_match = _TEXT_COORDINATES_RE.search(text)
    id_match = _CHUNK_COORDINATES_RE.search(chunk_id)

    document = str(
        record.get("document")
        or record.get("source")
        or metadata.get("source")
        or (text_match.group("document") if text_match else "")
    ).removesuffix(".docx")
    article = str(
        record.get("article")
        or metadata.get("article_number")
        or (text_match.group("article") if text_match else "")
        or (id_match.group("article") if id_match else "")
    )
    clause = str(
        record.get("clause")
        or metadata.get("clause_number")
        or (text_match.group("clause") if text_match and text_match.group("clause") else "")
        or (id_match.group("clause") if id_match and id_match.group("clause") else "")
    )
    point = str(
        record.get("point")
        or metadata.get("point_number")
        or (text_match.group("point") if text_match and text_match.group("point") else "")
        or (id_match.group("point") if id_match and id_match.group("point") else "")
    )
    return document, article, clause, point


def normalise_provenance_record(
    record: dict[str, Any],
    *,
    is_seed: bool,
    recursive_depth: int,
    expansion_reason: str,
    provenance_chain: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Bổ sung schema provenance mà không xoá field ranking hiện hữu."""

    item = dict(record)
    metadata = dict(item.get("metadata") or {})
    item["metadata"] = metadata
    document, article, clause, point = _coordinates(item)
    chunk_id = str(item.get("chunk_id") or item.get("id") or "")
    parent_id = str(
        item.get("parent_id")
        or metadata.get("parent_id")
        or (chunk_id if chunk_id.endswith("_PARENT") else "")
    )
    retrieval_score = item.get("retrieval_score")
    if retrieval_score is None and is_seed:
        retrieval_score = item.get("score")

    item.update(
        {
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "document": document,
            "article": article,
            "clause": clause,
            "point": point,
            "retrieval_score": retrieval_score,
            "behavior_score": float(item.get("behavior_score") or 0.0),
            "cross_encoder_score": item.get("cross_encoder_score"),
            "recursive_depth": int(recursive_depth),
            "is_seed": bool(is_seed),
            "expansion_reason": expansion_reason,
            "provenance_chain": list(provenance_chain or ([chunk_id] if chunk_id else [])),
        }
    )
    return item


def article_key(record: dict[str, Any]) -> tuple[str, str]:
    document, article, _, _ = _coordinates(record)
    return " ".join(document.casefold().split()), " ".join(article.casefold().split())


def records_to_context_texts(records: Iterable[dict[str, Any]]) -> list[str]:
    return [str(record.get("text") or "") for record in records if str(record.get("text") or "").strip()]


def provenance_log_record(record: dict[str, Any]) -> dict[str, Any]:
    """Payload log ngắn, ổn định và không ghi nguyên văn luật."""

    return {
        key: record.get(key)
        for key in (
            "document",
            "article",
            "clause",
            "point",
            "chunk_id",
            "parent_id",
            "retrieval_score",
            "behavior_score",
            "cross_encoder_score",
            "recursive_depth",
            "is_seed",
            "expansion_reason",
            "provenance_chain",
        )
    }
