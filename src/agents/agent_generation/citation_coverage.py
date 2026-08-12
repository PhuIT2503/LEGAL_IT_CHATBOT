"""Helpers for enforcing citation coverage in the final Critic answer.

The KG critic can guarantee that an article is present in the final context,
but an LLM may still omit that article from its answer.  These helpers keep
the coverage check deterministic: an article accepted as mandatory must be
named explicitly as ``Điều <number>`` in the answer.
"""

import re
from typing import Any, Dict, Iterable, List


_DIEU_ID_RE = re.compile(r"_D(?P<number>\d+[A-Za-z]*)$")


def citation_label(dieu_id: str) -> str:
    """Return a human-readable citation label from a canonical article id."""
    match = _DIEU_ID_RE.search(dieu_id or "")
    return f"Điều {match.group('number')}" if match else dieu_id


def make_required_citation(dieu_id: str, reason: str) -> Dict[str, str]:
    return {
        "dieu_id": dieu_id,
        "label": citation_label(dieu_id),
        "reason": reason,
    }


def deduplicate_required_citations(items: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Preserve order while merging duplicate article requirements."""
    result: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        dieu_id = str(item.get("dieu_id") or "")
        if not dieu_id or dieu_id in seen:
            continue
        seen.add(dieu_id)
        result.append(
            {
                "dieu_id": dieu_id,
                "label": str(item.get("label") or citation_label(dieu_id)),
                "reason": str(item.get("reason") or "Căn cứ pháp lý liên quan đã được Critic xác nhận."),
            }
        )
    return result


def find_missing_required_citations(
    answer: str, required_citations: Iterable[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Return mandatory articles that are not explicitly cited in ``answer``."""
    missing: List[Dict[str, str]] = []
    for item in deduplicate_required_citations(required_citations):
        label = item["label"]
        # Canonical labels are ``Điều <number>``.  Word boundaries prevent
        # ``Điều 1`` from being falsely matched by ``Điều 10``.
        match = _DIEU_ID_RE.search(item["dieu_id"])
        if match:
            number = re.escape(match.group("number"))
            pattern = rf"(?<!\w)Điều\s+{number}(?!\w)"
            present = re.search(pattern, answer or "", flags=re.IGNORECASE) is not None
        else:
            present = label.lower() in (answer or "").lower()
        if not present:
            missing.append(item)
    return missing


def format_required_citations(required_citations: Iterable[Dict[str, Any]]) -> str:
    """Build a concise, non-CoT coverage contract for the answer generator."""
    items = deduplicate_required_citations(required_citations)
    if not items:
        return ""
    lines = [
        "CÁC CĂN CỨ BẮT BUỘC ĐÃ ĐƯỢC CRITIC KIỂM TRA LÀ LIÊN QUAN:",
        *[f"- {item['label']}: {item['reason']}" for item in items],
        "Trong câu trả lời cuối, phải nêu rõ từng Điều trên và giải thích vai trò của Điều đó; "
        "đặc biệt phải thể hiện quan hệ dẫn chiếu giữa các Điều. Không chỉ liệt kê số Điều.",
    ]
    return "\n".join(lines)
